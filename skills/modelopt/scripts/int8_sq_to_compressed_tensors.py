#!/usr/bin/env python3
"""Convert a ModelOpt INT8 SmoothQuant (W8A8_SQ_PER_CHANNEL / INT8_WO) export into a
compressed-tensors W8A8 checkpoint that vLLM's compressed-tensors loader can serve.

vLLM's ModelOpt loader rejects W8A8 quant_algos, and vLLM has no per-linear SmoothQuant
input scale. So the smoothing is folded into the weight (W' = dequant(W) * pre_quant_scale,
requantized symmetric per output channel), the static per-tensor input_scale is dropped and
activations are quantized dynamically per token. Real INT8 x INT8 compute, numerically close
to but not identical with the original export. On Jetson Thor also set
VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel (CUTLASS INT8 has no sm_110 build;
without it serving dies with a kernel-selection error instead of falling back to Triton).

    python3 int8_sq_to_compressed_tensors.py <modelopt int8 export dir> <new output dir>

The output dir must not exist. Run inside the modelopt container (needs CUDA for speed).
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ACCEPTED = ("W8A8_SQ_PER_CHANNEL", "W8A8", "INT8_WO")
# Kept in FP16 by vLLM; merged with the export's own exclude_modules.
ALWAYS_IGNORE = ["re:.*visual.*", "re:.*lm_head.*", "re:.*embed_tokens.*",
                 "re:.*conv1d.*", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*in_proj_ba$"]


def requantize(q, w_scale, pqs):
    w = q.cuda().float() * w_scale.cuda().float().reshape(-1, 1) * pqs.cuda().float().reshape(1, -1)
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    q_new = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    return q_new.cpu(), scale.to(torch.float16).cpu()


def glob_to_re(glob):
    return "re:^" + re.escape(glob).replace(r"\*", ".*") + "$"


def main(src, dst):
    src, dst = Path(src), Path(dst)
    q = json.loads((src / "hf_quant_config.json").read_text())["quantization"]
    algo = q.get("quant_algo")
    if algo not in ACCEPTED:
        sys.exit(f"expected quant_algo in {ACCEPTED}, got {algo}")
    weight_only = algo == "INT8_WO"  # no activation quantization at all
    dst.mkdir(parents=True, exist_ok=False)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    new_map, n_q = {}, 0
    for shard in sorted(set(index["weight_map"].values())):
        t = load_file(src / shard)
        out = {}
        for k, v in t.items():
            if k.endswith(".weight") and v.dtype == torch.int8:
                base = k[: -len(".weight")]
                pqs = t.get(base + ".pre_quant_scale")
                if pqs is None:
                    pqs = torch.ones(v.shape[1])  # hf_ptq folds it into the preceding norm
                out[k], out[base + ".weight_scale"] = requantize(v, t[base + ".weight_scale"], pqs)
                n_q += 1
            elif k.endswith((".weight_scale", ".input_scale", ".pre_quant_scale")):
                continue  # replaced above / dropped (dynamic activations)
            else:
                out[k] = v
        save_file(out, dst / shard, metadata={"format": "pt"})
        new_map.update({k: shard for k in out})
        print(f"{shard}: {len(out)} tensors", flush=True)

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": new_map}, indent=2))

    ignore = ALWAYS_IGNORE + [glob_to_re(g) for g in q.get("exclude_modules", [])]
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
                        "dynamic": False, "group_size": None, "actorder": None, "observer": "minmax"},
            "input_activations": None if weight_only else {
                "num_bits": 8, "type": "int", "symmetric": True, "strategy": "token",
                "dynamic": True, "group_size": None, "observer": None},
            "output_activations": None,
        }},
        "ignore": sorted(set(ignore)),
        "kv_cache_scheme": None,
        "source": f"converted from a ModelOpt {algo} export (smoothing folded into weights)",
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    skip = {"config.json", "model.safetensors.index.json", "hf_quant_config.json"}
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in skip:
            (dst / f.name).write_bytes(f.read_bytes())
    print(f"requantized {n_q} INT8 weights -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-2])
    main(*sys.argv[1:3])
