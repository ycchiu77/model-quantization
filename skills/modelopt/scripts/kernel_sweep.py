#!/usr/bin/env python3
"""Benchmark every vLLM GEMM kernel that can run a quantized checkpoint on this GPU.

Starts vLLM once with defaults to see which kernel it picks, looks up that kernel's
candidate table inside the vLLM image, keeps the candidates vLLM says this GPU
supports, then forces each one (VLLM_DISABLED_KERNELS = all others in the table)
and measures throughput (bench.py) and perplexity (ppl.py). Run on the host.

    python3 kernel_sweep.py --checkpoint-dir <host dir with exports> --name <export folder> \
        --gpu 1 --port 8899 --parquet <host eval parquet> -- --max-model-len 32768 --gpu-memory-utilization 0.5

Works on x86 workstations and on Jetson (detected via /etc/nv_tegra_release): there the
container gets `--runtime=nvidia --gpus all` instead of a device index. Kernels that can't run
on the platform at all (e.g. FlashInfer JIT kernels on Jetson, where the image has no nvcc) go
in --base-disabled; they stay disabled in every run and are never forced.

bench.py/ppl.py need requests, aiohttp and datasets. If the host's Python lacks them, pass
--client-container modelopt: they then run inside that container (which has them after Step 1),
reaching the server through the Docker bridge, and --parquet is a path inside that container.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

# Line-buffer stdout so a redirected sweep still shows progress per kernel.
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

HERE = Path(__file__).resolve().parent
IS_JETSON = os.path.exists("/etc/nv_tegra_release")
KERNEL_RE = re.compile(r"(?:Selected|Using) ([A-Za-z0-9]+Kernel)\b")

LIST_TABLES = r"""
import json, vllm.model_executor.kernels.linear as L
from vllm.platforms import PlatformEnum, current_platform
cc = current_platform.get_device_capability(); cc = cc[0] * 10 + cc[1]
out = {}
for name in dir(L):
    d = getattr(L, name)
    if not name.startswith('_POSSIBLE_') or not isinstance(d, dict):
        continue
    rows = []
    for k in d.get(PlatformEnum.CUDA, []):
        try:
            ok, why = k.is_supported(cc)
        except Exception as e:
            ok, why = False, type(e).__name__
        rows.append([k.__name__, bool(ok), why or ''])
    out[name] = rows
print('TABLES=' + json.dumps(out))
"""


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def list_tables(a):
    sh("docker rm -f kernel-sweep-list")
    r = subprocess.run(["docker", "run", "--name", "kernel-sweep-list", *shlex.split(a.gpu_flags), "--entrypoint", "python3",
                        a.image, "-c", LIST_TABLES], text=True, capture_output=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("TABLES=")), None)
    if line is None:
        sys.exit(f"Could not list vLLM kernel tables:\n{r.stderr[-2000:]}")
    return json.loads(line[len("TABLES="):])


def serve_and_measure(a, disabled, label):
    name = "kernel-sweep-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", label)  # one kept container per kernel
    sh(f"docker rm -f {name}")
    disabled = list(dict.fromkeys(a.base_disabled + disabled))
    env = f"-e VLLM_DISABLED_KERNELS={','.join(disabled)}" if disabled else ""
    env += "".join(f" -e {shlex.quote(e)}" for e in a.env)
    serve = (f"docker run -d --name {name} {a.gpu_flags} --ipc=host --shm-size 2g "
             f"-p {a.port}:8000 -e HF_HUB_OFFLINE=1 {env} "
             f"-v {a.checkpoint_dir}:/workspace/out:ro -v {a.hf_cache}:/root/.cache/huggingface "
             f"{a.image} /workspace/out/{a.name} {' '.join(a.vllm_args)}")
    sh(serve)
    url = f"http://localhost:{a.port}"
    deadline = time.time() + (1800 if IS_JETSON else 900)  # Jetson: slower load + compile
    while time.time() < deadline:
        if sh(f"curl -sf {url}/v1/models").returncode == 0:
            break
        if not sh(f"docker ps -q -f name=^{name}$").stdout.strip():
            break
        time.sleep(5)
    lg = sh(f"docker logs {name}")
    logs = lg.stdout + lg.stderr
    res = {"label": label, "kernel": None, "found": [], "single": None, "concurrent": None, "ppl": None, "error": None}
    res["found"] = list(dict.fromkeys(KERNEL_RE.findall(logs)))
    res["kernel"] = res["found"][0] if res["found"] else None
    if sh(f"curl -sf {url}/v1/models").returncode != 0:
        reason = [l for l in logs.splitlines() if label in l and "disabled by environment" not in l
                  and not KERNEL_RE.search(l)]
        errs = [l for l in logs.splitlines() if re.search(r"(Error|error):", l) and "Warning" not in l
                and "Engine core initialization failed" not in l]
        msg = (errs[-1] if errs else reason[-1] if reason else "server did not start")
        res["error"] = re.sub(r"^.*?\] ", "", msg).strip()[:200]
        sh(f"docker stop {name}")  # keep it: docker logs shows why it failed
        return res
    if a.client_container:
        c = a.client_container
        run = lambda script, args: sh(f"docker exec -e HF_DATASETS_CACHE=/tmp/kernel_sweep_dscache {c} "
                                      f"python3 /tmp/kernel_sweep/{script} {args}").stdout
        curl = f"http://{a.bridge_gw}:{a.port}"
    else:
        run = lambda script, args: sh(f"{sys.executable} {HERE / script} {args}",
                                      env={**os.environ, "HF_DATASETS_CACHE": a.datasets_cache}).stdout
        curl = url
    b = run("bench.py", curl)
    m1, m2 = re.search(r"single: ([\d.]+)", b), re.search(r"8-concurrent: ([\d.]+)", b)
    res["single"], res["concurrent"] = (float(m1.group(1)) if m1 else None), (float(m2.group(1)) if m2 else None)
    p = run("ppl.py", f"--url {curl} --parquet {a.parquet} --size {a.ppl_size}")
    m = re.search(r"Perplexity \(.*?\): ([\d.]+)", p)
    res["ppl"] = float(m.group(1)) if m else None
    sh(f"docker stop {name}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True, help="host dir mounted as /workspace/out")
    p.add_argument("--name", required=True, help="export folder name inside --checkpoint-dir")
    p.add_argument("--gpu", required=not IS_JETSON, help="device index (x86, required); ignored on Jetson")
    p.add_argument("--gpu-flags", help="docker GPU flags; default: --gpus device=<gpu>, or "
                   "--runtime=nvidia --gpus all on Jetson")
    p.add_argument("--base-disabled", default="", help="comma list of kernels disabled in every run "
                   "and never forced (e.g. the FlashInfer kernels on Jetson)")
    p.add_argument("--env", action="append", default=[], help="KEY=VAL passed to every vLLM container; repeatable")
    p.add_argument("--port", default="8899")
    p.add_argument("--parquet", required=True, help="eval parquet for ppl.py (host path, or a path in --client-container)")
    p.add_argument("--client-container", help="run bench.py/ppl.py in this running container (e.g. modelopt) "
                   "instead of the host's Python")
    p.add_argument("--ppl-size", type=int, default=64)
    p.add_argument("--image", required=True, help="the same <image> used in Steps 3-5")
    p.add_argument("--hf-cache", default=os.path.realpath(os.path.expanduser("~/.cache/huggingface")))
    p.add_argument("--datasets-cache", default="/tmp/kernel_sweep_dscache")
    p.add_argument("--out", help="write results JSON here")
    p.add_argument("vllm_args", nargs=argparse.REMAINDER, help="after --: extra vllm serve args")
    a = p.parse_args()
    a.vllm_args = [x for x in a.vllm_args if x != "--"]
    if a.gpu_flags is None:
        a.gpu_flags = "--runtime=nvidia --gpus all" if IS_JETSON else f"--gpus '\"device={a.gpu}\"'"
    a.base_disabled = [k for k in a.base_disabled.split(",") if k]
    if a.client_container:
        sh(f"docker exec {a.client_container} mkdir -p /tmp/kernel_sweep")
        for f in ("bench.py", "ppl.py"):
            sh(f"docker cp {HERE / f} {a.client_container}:/tmp/kernel_sweep/{f}")
        a.bridge_gw = sh("docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'").stdout.strip()

    print("== default run")
    default = serve_and_measure(a, [], "default")
    if default["error"]:
        sys.exit(f"Default server failed: {default['error']}")
    tables = list_tables(a)
    table_name, rows = None, None
    for k in default["found"]:
        hit = next(((n, r) for n, r in tables.items() if n != "_POSSIBLE_KERNELS"
                    and any(name == k for name, _, _ in r)), None)
        if hit:
            (table_name, rows), default["kernel"] = hit, k
            break
    if not rows:
        sys.exit(f"None of the kernels in the log ({default['found']}) are in a GEMM kernel table")
    names = [k for k, _, _ in rows]
    weight_only = {k for k, _, _ in tables.get("_POSSIBLE_WFP8A16_KERNELS", [])} if "FP8" in table_name else set()
    qcfg = Path(a.checkpoint_dir, a.name, "hf_quant_config.json")
    algo = json.loads(qcfg.read_text())["quantization"].get("quant_algo") if qcfg.exists() else None
    if "FP4" in table_name and algo in ("NVFP4", "MXFP4"):
        # W4A4 checkpoint: Marlin dequantizes to BF16, so it skips the FP4 activations the user chose
        # (for W4A16_NVFP4, weight-only is the point, so Marlin stays eligible)
        weight_only |= {k for k in names if "Marlin" in k}
    print(f"default kernel: {default['kernel']} (table {table_name})")

    results = [dict(default, label=default["kernel"] + " (default)")]
    if algo == "W4A16_NVFP4":
        # vLLM routes weight-only NVFP4 to Marlin regardless of VLLM_DISABLED_KERNELS (verified on 0.27.1),
        # so forcing other candidates just re-serves Marlin
        print("W4A16_NVFP4 checkpoint: vLLM always uses Marlin for it, nothing to sweep")
        rows = []
    for k, ok, why in rows:
        if k == default["kernel"]:
            continue
        if not ok:
            results.append({"label": k, "kernel": None, "error": f"not supported on this GPU: {why}"})
            continue
        if k in a.base_disabled:
            results.append({"label": k, "kernel": None, "error": "in --base-disabled, not tried"})
            continue
        print(f"== forcing {k}")
        r = serve_and_measure(a, [n for n in names if n != k], k)
        if r["kernel"] and r["kernel"] != k and not r["error"]:
            r["error"] = f"vLLM picked {r['kernel']} instead"
        results.append(r)

    base_ppl = default["ppl"]
    print(f"\n| Kernel | tok/s single | tok/s 8 concurrent | Perplexity | Note |\n|---|---|---|---|---|")
    for r in results:
        if r.get("error"):
            print(f"| {r['label']} | — | — | — | {r['error']} |")
            continue
        note = "weight-only (BF16 activations)" if r["kernel"] in weight_only else ""
        print(f"| {r['label']} | {r['single']} | {r['concurrent']} | {r['ppl']} | {note} |")
    ok = [r for r in results if not r.get("error") and r["concurrent"] and r["ppl"]
          and r["kernel"] not in weight_only and (base_ppl is None or r["ppl"] <= base_ppl * 1.01)]
    if ok:
        best = max(ok, key=lambda r: r["concurrent"])
        print(f"\nFastest at 8 concurrent with perplexity within 1% of default: {best['kernel']}")
        if best["kernel"] != default["kernel"]:
            supported = {k for k, ok, _ in rows if ok}
            ahead = [n for n in names[:names.index(best["kernel"])] if n in supported]
            print(f"Serve with: -e VLLM_DISABLED_KERNELS={','.join(dict.fromkeys(a.base_disabled + ahead))}")
        elif a.base_disabled:
            print(f"Keep vLLM's default (still serve with -e VLLM_DISABLED_KERNELS={','.join(a.base_disabled)}).")
        else:
            print("Keep vLLM's default.")
    if a.out:
        Path(a.out).write_text(json.dumps({"table": table_name, "results": results}, indent=2))


if __name__ == "__main__":
    main()
