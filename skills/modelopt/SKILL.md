---
name: modelopt
description: Quantize an LLM or VLM checkpoint with NVIDIA's Model Optimizer (ModelOpt) and serve it with vLLM, end to end — laying out which precisions the actual GPU accelerates so the user can choose, running the official hf_ptq.py pipeline offline, and verifying speed and accuracy against BF16 before calling it done. Use this whenever the user asks to quantize, compress, shrink, or speed up a model (e.g. "quantize this model to FP8", "make this smaller for a 48GB card", "try INT4 on this checkpoint"), mentions ModelOpt/modelopt/mtq/hf_ptq, or wants a precision/throughput/accuracy comparison across BF16/FP8/INT4/NVFP4 variants of the same model. Works on x86-64 Ubuntu workstations/servers with discrete GPUs and on NVIDIA Jetson (AGX Thor, AGX Orin; aarch64, unified memory). Everything runs inside a Docker container from the vllm/vllm-openai image, never in the host's Python. Also trigger if the user reports a quantized model behaving strangely (garbage output, huge perplexity) or a vLLM serving error mentioning "modelopt", "quant_algo", or a quantization ValidationError. For turning the quantized checkpoint into a TensorRT engine afterwards, hand off to the totrt skill.
---

# ModelOpt quantization (serve with vLLM)

This skill quantizes a model with NVIDIA Model Optimizer's post-training
quantization (PTQ) and gets it serving through vLLM — not just exported. Every
pitfall below was hit and root-caused on real runs (a 27B hybrid-attention VLM
and a 3B LLM on RTX 6000 Ada, and the same 27B VLM on a Jetson AGX Thor), so
skip straight to them rather than rediscovering them.

**Two platforms, one workflow.** The steps are the same on an x86-64 Ubuntu
machine with discrete GPUs and on a Jetson. Step 0 decides which one you're
on, and the differences are confined to a few places, marked **Jetson:**
below. Wherever a command shows `<gpu-flags>` or `<image>`, fill them in from
Step 0 / Step 1:

| | x86-64 Ubuntu, discrete GPU | Jetson (Thor / Orin) |
|---|---|---|
| `<gpu-flags>` | `--gpus '"device=<N>"'` (or `--gpus all`) | `--runtime=nvidia --gpus all` (one integrated GPU, no index) |
| `<image>` | `vllm/vllm-openai:v0.32.0` | `v0.32.0` if Step 1's preflight passes, otherwise the verified `v0.27.1` or a Jetson build (Step 1) |
| GPU memory | separate VRAM per card (`nvidia-smi`) | one LPDDR pool shared with the CPU and OS (`free -g`); `nvidia-smi` shows `N/A` |
| FlashInfer | works | works if the image has `nvcc` (stock `v0.27.1` does); some Jetson builds don't, and then FlashInfer kernels must be disabled (Step 4) |
| Speed reference | driver/clocks fixed | power mode matters: check `nvpmodel -q` before benchmarking (Step 5) |

Version set used for the 2026-10-28 x86 run: `vllm/vllm-openai:v0.32.0`,
torch 2.13.0+cu130, transformers 5.19.0, `nvidia-modelopt==0.45.0`. The
reference sweeps at the end of this skill were measured on `v0.27.1`; keep
that label rather than retroactively attributing them to 0.32.

The output is a quantized Hugging Face–format checkpoint. It's the main
artifact: vLLM serves it directly, and the `totrt` skill can turn it into a
TensorRT engine later. Quantization and engine building are deliberately
separate: the checkpoint is mostly hardware-neutral, while an engine is tied
to one GPU type and TensorRT-LLM version.

Bundled: `scripts/bench.py` (throughput) and `scripts/ppl.py` (perplexity),
both against a running server (Step 5), and `scripts/kernel_sweep.py`, which
benchmarks every vLLM GEMM kernel that can run the checkpoint (Step 4b), and
`scripts/make_launch_script.py`, which writes a ready-to-run `launch_vllm.sh`
into each export (Step 6), `scripts/fix_vlm_keys.py`, which repairs a VLM
export whose vision-tower keys lost a nesting level (Step 3), and
`scripts/int8_sq_to_compressed_tensors.py` +
`scripts/int4_awq_to_compressed_tensors.py`, which convert INT8 SmoothQuant /
INT4 AWQ exports into the compressed-tensors format vLLM can serve (Step 3c).

**Everything runs in Docker, never on the host.** Quantization (`hf_ptq.py`,
`mtq.quantize()`) runs in the `modelopt` container and serving runs in a
`vllm-serve` container, both from the same vLLM image (`<image>`, Step 1). The
host only runs read-only checks (`nvidia-smi`, `df`, `ss`, and on Jetson
`free`, `tegrastats`, `nvpmodel -q`), `docker`, `curl`,
and the bundled measurement scripts, which are just HTTP clients. Don't
`pip install` ModelOpt, torch, vLLM or transformers into the host's Python or a
host venv, and don't run `hf_ptq.py` on the host. This still applies if the
host already has some of these packages, or if a host install looks quicker.
If the model needs package versions that conflict with the image, put them in
a uv venv *inside* the container (Step 1b). Don't change the image's own
Python, and never install on the host. The reasons:

- The verified combination of torch, transformers, ModelOpt and vLLM exists
  only in this image. A host install picks up whatever torch and CUDA happen to
  be there. Earlier attempts outside the image hit torch re-downloads, numpy
  ABI breaks and `torchvision::nms` errors.
- The same pinned image gives the same results on every machine.
- Host package installs can break other users' and projects' environments.

If Docker can't be used (not installed, `permission denied` on the Docker
socket, or no GPU inside containers), **stop and tell the user** what's
missing and how to fix it (Step 1's preflight). Don't quietly fall back to the
host. Only run on the host if the user explicitly asks for it after hearing
this.

**No `docker run --rm`.** Give every container a fixed name (`modelopt`,
`vllm-serve`, `modelopt-preflight`, `modelopt-bench`, `kernel-sweep-<kernel>`,
`kernel-sweep-list`, and `vllm-<export-name>` from Step 6's `launch_vllm.sh`) and leave it in place when it exits. When done with a
server, `docker stop` it rather than removing it. With `--rm`, a
failed run takes its logs and filesystem with it, so there's nothing left to
debug with. A stopped container holds no GPU memory; inspect it with
`docker logs <name>`. Before reusing a name, run
`docker rm <name> 2>/dev/null` (add `-f` only for one of these skill-owned
names that is still running). Never remove containers with other names; on a
shared machine they belong to someone else.

**Default to running the whole thing end to end** — quantize, serve, sweep the
GEMM kernels, measure — without pausing between stages to ask
"should I proceed?". Only stop for a genuine fork in the road, like a qformat
that turns out unsupported by vLLM (see Step 2).

**Show progress as a text bar.** After each major stage, print a compact
checkpoint plus a stage table, e.g.:

```
[■■■■□□□□□□]  4/10  Pulling vLLM 0.32.0 image (model downloaded ✓)
```

| # | Stage | Status |
|---|---|---|
| 1 | Hardware check | ✅ |
| 2 | Input confirmation | ✅ |
| ... | ... | ⏳ running / pending |

Claude Code has no UI progress bar, so this is the replacement: it keeps a
long run legible, makes a stall obvious, and shows which fallback is active.
Update it at stage boundaries (download, container/preflight, quantization
start and finish, export validation, each serve, each measurement, kernel
sweep, launch script), not after every command. Use `✅ / ⏳ / ❌ / pending`
and a 10-cell bar sized to the number of stages in this run — the skill's
standard stages are Step 0, input confirmation, Step 1, Step 1b, Step 2,
Step 3 (or 3b), export sanity, Step 4, Step 4b, Step 5, Step 6. For a model
outside the verified path (e.g. an embedding/pooling model or a brand-new
architecture), add its extra steps to the table rather than padding the count.

**Confirm the inputs before running anything.** Unlike "should I proceed",
these aren't things to guess. Run Step 0's `nvidia-smi` check first (it's
read-only and takes a second) so the precision question is grounded in the
actual hardware. Then ask once, in a single round, for whatever the request
doesn't already state:

- **Input model**: an HF repo ID (e.g. `Qwen/Qwen3.8-27B`) or a host path to a
  local checkpoint. A local path gets mounted read-only (Step 1).
- **Precision**: the user decides, never pick it silently. Offer the formats
  from Step 2's table that make sense for this GPU (on Jetson, by model:
  Thor and Orin differ completely, see Step 0). For each one, say what it
  buys on *this* hardware (native speedup or only a smaller footprint) and
  whether vLLM can serve it. Mark one as recommended, but let the user choose.
  They can pick several to compare; quantize each into its own
  `<model-name>-<qformat>/`. If the user already named a precision, don't ask
  again. If it's a poor fit (e.g. NVFP4 on Ada or Orin for speed, FP8 on
  Orin), say why once and let them confirm or switch. `int4_awq`/`int8_sq`
  are a good fit but need the extra Step 3c conversion for vLLM — mention
  that up front, not after the export fails to load.
- **KV cache precision** (part of the precision choice): `hf_ptq.py` quantizes
  the KV cache to FP8 by default. Recommend keeping it in BF16
  (`--kv_cache_qformat none`): on Qwen2.5-3B it accounted for most of the
  accuracy loss (Step 3). Offer FP8 KV cache only when KV-cache memory is the
  real constraint, and never if a TensorRT engine is wanted.
- **Calibration**: whether to run the data-driven calibration pass. Never
  decide this from bit width — check the preset itself:
  `need_calibration(QUANT_CFG_CHOICES[<qformat>])`. Calibration is **required**
  for presets that enable a *static* activation quantizer or a calibrated
  algorithm: `fp8` (W8A8), `nvfp4` (W4A4), `w4a8_*`, `int4_awq`, `int8`,
  `mxfp8`, and similar. It is **not** required — and `hf_ptq.py` skips it
  automatically, logging "Dynamic quantization. Calibration skipped." — for
  weight-only or dynamic-per-token presets: `w4a16_nvfp4`, `fp8_pb_wo`,
  `fp8_2d_blockwise_weight_only`, `fp8_pc_pt`, `int8_wo`,
  `int4_blockwise_weight_only`, `nvfp4_mlp_weight_only`,
  `mxfp4_mlp_weight_only`. Recommend skipping calibration whenever the preset
  permits it; where it does not, say so and let the user choose the
  calibrated format (recommended, 512 samples from a cached chat dataset;
  ~3 minutes on a 3B) rather than a no-calibration alternative. `--calib_size 0`
  is **not** a way to skip — on a preset that needs calibration `hf_ptq.py`
  raises `IndexError` in the dataloader. A genuinely no-calibration FP8 export
  only comes from a weight-only/dynamic FP8 preset, which gives up the FP8
  compute speedup.
- **TensorRT engine later?** If yes, the checkpoint must have a BF16 KV cache
  (an FP8 KV cache gives garbage output on TensorRT-LLM + Ada), and only `fp8`
  has been verified to build. Mention `/totrt` as the next step at the end.
- **Host output dir**: where the quantized checkpoints should end up on the
  host. Every export goes under it as `<model-name>-<qformat>/`. Always ask
  this, even when a `modelopt` container already exists. An existing
  container already has some host folder mounted at `/workspace/out` from an
  earlier run, and reusing it without asking writes the export wherever that
  earlier run pointed. Look up the current mount first (`docker inspect
  modelopt --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'`)
  and offer it as the default, giving the full host path the export will
  land in: "Export will go to `/data/quantized/<model-name>-<qformat>/`
  (already mounted in the existing container). OK, or somewhere else?"
- **Host HF cache dir**: where downloaded weights live. Default to
  `~/.cache/huggingface` if the user doesn't care, but check its disk first
  (see below). Skip this one if the input is a local path.

Think in host paths, not container paths. The container is disposable, so
anything not under a mount (Step 1) is gone when it exits. A checkpoint
written to `/tmp` or an unmounted path inside the container looks like a
success (no error, `hf_quant_config.json` and all) and then just isn't there
on the next run.

Check free space before downloading or exporting. A model needs about 2 bytes
per parameter in BF16 (27B ≈ 54GB), plus ~1 byte/param for an FP8 export and
~0.6 for a 4-bit one. **Jetson:** the built-in eMMC/root partition is small,
so weights, exports *and* Docker's image store (`docker info | grep 'Docker
Root Dir'`; images are 10–30GB each) belong on an NVMe or external SSD. If
Docker's root dir is on the root partition and it's nearly full, tell the
user before pulling. A bind mount to a path that doesn't exist yet gets
created silently by Docker as an empty root-owned directory, often on the root
partition, so a "missing cache" can turn into a full root disk mid-download:

```bash
df -h <host-output-dir> <host-hf-cache-dir> "$(docker info --format '{{.DockerRootDir}}')"
```

Size from the stored dtype, not the parameter count: read `torch_dtype` /
`dtype` in `config.json` (for a VLM it's often only under `text_config`, as
in Cosmos-Reason2). The summed size of the `*.safetensors` files is
authoritative: when the top-level and `text_config` dtypes disagree (PaliGemma's
`bfloat16` branch says `float32` in `text_config` but is 5.5GB of BF16), trust
the file sizes. Some repos store FP32 on `main` (4 bytes/param,
twice the BF16 size) and publish BF16 on a `bfloat16` branch, e.g.
`google/paligemma-3b-pt-224` (11.7GB FP32 on `main`, 5.5GB on `bfloat16`).
Download that branch (`hf download <repo> --revision bfloat16`) and use its
snapshot path. Also check that a model that "is cached" really has its
weights: a snapshot with only tokenizer/config files is not a download
(`ls <snapshot>/*.safetensors`).

Once these are pinned down, run everything else uninterrupted.

## Step 0: Check what the GPU actually accelerates

Precision choice should follow from hardware, not from what's newest. First
find out which platform this is:

```bash
uname -m                                         # x86_64 or aarch64
cat /etc/nv_tegra_release 2>/dev/null | head -1  # present only on Jetson: L4T release (R36 = JetPack 6, R38+ = JetPack 7)
nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total --format=csv
```

On x86 also see who else is on the GPUs:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

**Jetson:** `nvidia-smi` reports `memory.used`/`memory.total` as `[N/A]` and
usually lists no compute apps, because GPU memory *is* system memory. Look at
the shared pool and at other GPU users instead:

```bash
free -g                                              # "available" is what quantization + serving can use
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'   # other vLLM/LLM containers holding memory
tegrastats --interval 1000 | head -3                 # GR3D_FREQ = GPU load, RAM = pool in use
nvpmodel -q 2>/dev/null | head -2                    # power mode (MAXN for benchmarks)
```

On JetPack 5/6 (`nvidia-smi` missing or minimal) read the module from
`cat /proc/device-tree/model` instead: AGX Orin / Orin NX / Orin Nano are
compute capability 8.7.

| Compute capability | Architecture | Native low-precision support |
|---|---|---|
| 7.5 | Turing | INT8 |
| 8.0 / 8.6 / 8.9 | Ampere / Ada | INT8, FP8 (4th-gen tensor cores on 8.9/Ada and some 9.0) |
| **8.7** | **Jetson Orin** (Ampere) | INT8 only. **No FP8, no FP4 tensor cores**: FP8 and NVFP4 both run weight-only through Marlin |
| 9.0 | Hopper | FP8 (native, fast) |
| 10.0+ | Blackwell (datacenter) | FP8, **NVFP4** (native FP4 tensor cores) |
| **11.0** | **Jetson AGX Thor** (Blackwell) | FP8, **NVFP4** (native FP4 tensor cores) |
| 12.0 / 12.1 | Blackwell (RTX 50 / RTX PRO, DGX Spark) | FP8, NVFP4 |

The one thing worth internalizing: **NVFP4 compute requires Blackwell
(SM100+)**. On Ada or Hopper, there is no FP4 tensor core datapath — picking
an NVFP4 *compute* format there doesn't error, it silently runs on a software
fallback kernel (see the W4A16 note in Step 2) and buys you no speed, only a
smaller checkpoint. If the GPU is Ada/Hopper, **FP8 is almost always the one
to recommend** (the user still makes the call).

**Jetson specifics.** Decode on a Jetson is even more memory-bandwidth-bound
than on a discrete card (Thor's LPDDR5X: 273GB/s peak; a device memcpy measured
~120GB/s on this Thor), so bytes per weight dominate speed and 4-bit pays off more than on
x86:

- **Thor:** recommend `w4a16_nvfp4` by default. It measured lossless on
  Qwen3.8-27B (gsm8k 92.6 → 92.6, half the memory of BF16). Offer `nvfp4`
  (W4A4, real FP4 compute) when concurrency or KV headroom matters more than
  accuracy: it cost 8 gsm8k points on the same model, and +16% perplexity on
  Qwen2.5-1.5B. FP8 is the safe middle option: 1.5x BF16 throughput at +1.5%
  perplexity on Qwen2.5-1.5B, against ~1.95x and +8% for `w4a16_nvfp4`
  (reference results below). For Qwen3.5-family hybrid models (Qwen3.8)
  the `quantize-nvfp4-thor` skill has a recipe that was run end to end on
  this hardware.
- **Orin:** with no FP8/FP4 compute, every 4-bit format is a footprint and
  bandwidth win only. `w4a16_nvfp4` (Marlin) is the one to recommend. FP8
  also runs weight-only (FP8 Marlin) and so mainly halves memory. Orin is
  usually also on JetPack 6 (CUDA 12.x), which changes the image (Step 1).
- **Memory:** the model must fit in the shared pool next to the OS. BF16
  calibration holds the full BF16 model plus activations (27B ≈ 54GB+) in the
  same RAM that serving needs later, so quantization and serving can't run at
  the same time, and any other model server must be stopped first (ask before
  stopping someone else's container). Reach for 4-bit mainly when
VRAM is the actual constraint (e.g. fitting on a single card that BF16/FP8
doesn't fit on). Be precise about speed: at small batch sizes decoding is
limited by reading weights from memory, so 4-bit weights *are* much faster
than BF16 even through a software kernel (Qwen3-4B NVFP4 via Marlin: 2.2x
BF16). What they don't beat is FP8: on Qwen3.8-27B, `w4a16_nvfp4` tied FP8 for
single requests and was slower under concurrent load.

Pick a GPU nobody else is using if you can. Another process on the same GPU
cut every throughput number by ~20% in a real run; that matters for Step 5's
comparisons. (Jetson has one GPU, so this means: nothing else running.)

## Step 1: Environment — the vLLM image plus ModelOpt

The host needs Docker, the NVIDIA Container Toolkit, and a driver whose `CUDA
Version` (top-right of `nvidia-smi`) is at least the image's CUDA build:
`vllm/vllm-openai:v0.32.0` is torch 2.13.0 + CUDA 13.0, so driver 580+. On an
older driver, choose an older vLLM tag built for CUDA 12.x (the verified
fallback is `v0.27.1`).

**Jetson:** the driver comes with JetPack, not from a driver package, and
can't be upgraded on its own. JetPack 7 (L4T R38+, Thor; this one reports
driver 595 / CUDA 13.2) runs CUDA 13.0 images. JetPack 6 (L4T R36, Orin) is
CUDA 12.6 and can't run the CUDA 13 images (`v0.32.0` or `v0.27.1`). There,
use a Jetson build of vLLM for
JetPack 6 (e.g. from `jetson-containers` / `dustynv/vllm:*-r36.*`) and check
that its torch is ≥2.8 for ModelOpt 0.45 (older ones need ModelOpt 0.43,
see Step 1b). The toolkit and `nvidia` runtime come with JetPack
(`docker info | grep -i runtimes` should list `nvidia`).

**Preflight: prove Docker can reach the GPU before anything else.** On a new
machine this is the step that usually fails:

```bash
docker info --format '{{.ServerVersion}}'            # Docker installed and usable
docker rm modelopt-preflight 2>/dev/null
docker run --name modelopt-preflight <gpu-flags> --entrypoint nvidia-smi <image> -L  # GPU visible in a container (also pulls the image, ~10GB)
```

**Jetson:** also prove the image's torch has kernels for this GPU, since
seeing the GPU isn't the same as being able to run on it, and check for
`nvcc` (it decides whether FlashInfer can be used, Step 4):

```bash
docker rm modelopt-preflight 2>/dev/null
docker run --name modelopt-preflight --runtime=nvidia --gpus all --entrypoint python3 <image> -c "
import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('arch list:', torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16); print('matmul ok', (x @ x).float().abs().mean().item())
import shutil; print('nvcc:', shutil.which('nvcc'))"
```

Stock `vllm/vllm-openai:v0.27.1` passes this on Thor (verified 2026-10-27:
torch 2.13.0+cu130, arch list `sm_80 … sm_110 sm_120`, `nvcc` present; the
arm64 image is 33GB on disk). `v0.32.0` has not been verified on Thor, so use
`v0.27.1` there unless the 0.32.0 preflight also passes.

`no kernel image is available for execution on the device` (or an arch list
with nothing at or below the GPU's capability and no PTX) means this image
can't run here. Switch `<image>` to a Jetson build rather than patching:
on Thor, `vllm/vllm-openai:<tag>-aarch64-ubuntu2404` (v0.28.0 is what the
`quantize-nvfp4-thor` skill served with) or NVIDIA's
`ghcr.io/nvidia-ai-iot/vllm:*-jetson-thor` images. Check `docker images`
first: a suitable one is often already there, and they're 30GB to pull. Use
the same `<image>` for every container in this skill, and note it in the
final report.
| Failure | Cause | What to tell the user |
|---|---|---|
| `docker: command not found` | No Docker | Install Docker Engine and the NVIDIA Container Toolkit. |
| `permission denied ... docker.sock` | User isn't in the `docker` group | `sudo usermod -aG docker $USER`, then log out and back in (or run `newgrp docker`). |
| `could not select device driver "" with capabilities: [[gpu]]` | No NVIDIA Container Toolkit | Install `nvidia-container-toolkit`, run `sudo nvidia-ctk runtime configure --runtime=docker`, then restart Docker. On Jetson it ships with JetPack (`sudo apt install nvidia-container`), and `--runtime=nvidia` needs the same `nvidia-ctk` step. |
| `unknown or invalid runtime name: nvidia` | Jetson: runtime not registered with Docker | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. |
| `CUDA driver version is insufficient` / `forward compatibility` | Driver older than the image's CUDA | x86: use an older vLLM tag built for CUDA 12.x, or upgrade the driver. Jetson: use an image built for the installed JetPack (above). |
| `exec format error` | Image for the wrong CPU architecture (an amd64-only tag on Jetson) | Use a multi-arch or `-aarch64` tag. |
| `no kernel image is available for execution on the device` | Jetson: image's torch not built for this GPU | Use a Jetson build (above). |

These fixes need sudo, so ask the user to run them (with the `!` prefix) and
don't try to work around them. If the preflight fails, stop here. Don't
continue on the host (see the top of this skill).

The same image quantizes and serves. In `v0.32.0` its torch (2.13.0) already satisfies ModelOpt's
requirement (`nvidia-modelopt==0.45.0` needs `torch>=2.8`), so only ModelOpt
and a few small packages get installed on top.

**Reuse an existing container if there is one.** An idle container running
`sleep infinity` holds no GPU memory, and it keeps the installed packages:

```bash
docker ps -a --filter name=^modelopt$ --format '{{.Names}} {{.Image}} {{.Status}} {{.Mounts}}'
docker exec modelopt python3 -c "import modelopt; print(modelopt.__version__)"
```

If it exists, uses `<image>`, has ModelOpt 0.45.0, and its mounts match
what the user confirmed (the `/workspace/out` source is the output dir they
chose, the HF cache is theirs, and a local input model is mounted at
`/workspace/model`), `docker start modelopt` and skip to Step 2. If a mount
doesn't match, Docker can't change it on an existing container. Remove the
container and create it again with the command below. Anything installed only
inside it is lost (ModelOpt reinstalls in minutes; a Step 1b uv venv
rebuilds from scratch unless `/opt/venvs` was mounted), so tell the user that
before removing it. If it holds anything not on a mount (a Step 1b venv under
`/opt/venvs`, cloned repos under `/opt/src`), or the user can't be asked,
don't remove it: `docker stop modelopt && docker rename modelopt
modelopt-old-<date>` keeps it intact, and the user can delete it later. Mount
a host directory at `/opt/venvs` on the new container so this doesn't repeat,
and a host pip cache at `/root/.cache/pip`. If the old container had no
mounted pip cache, seed the new one before installing:
`docker cp modelopt-old-<date>:/root/.cache/pip - | docker cp - modelopt:/root/.cache/`
(on a ~55KB/s PyPI day this turned a 28+ minute install into 5 minutes), then
`docker exec modelopt chown -R $(id -u):$(id -g) /root/.cache/pip` so the host
user can manage it. If the old container's `/opt/Model-Optimizer` is at the same
tag (`docker exec modelopt-old-<date> git -C /opt/Model-Optimizer describe --tags`),
copy it too instead of cloning: `docker cp modelopt-old-<date>:/opt/Model-Optimizer - | docker cp - modelopt:/opt/`
(the clone took 7 of 9 install minutes on a slow-GitHub day). Never write the export to the old mount because it's
already there. A container that has been up for
days can lose the GPU: `nvidia-smi` inside it fails with `Failed to initialize
NVML: Unknown Error` and torch reports no CUDA. This happens when the host
reloads systemd/cgroups. `docker restart modelopt` fixes it and keeps
everything installed in the container. Otherwise create it. On a shared
x86 machine use `--gpus '"device=N"'` rather than `all`; on Jetson
`<gpu-flags>` is `--runtime=nvidia --gpus all`. The image's entrypoint
is `vllm serve`, so override it:

```bash
mkdir -p <host-output-dir> <host-hf-cache-dir>
docker run -d --name modelopt <gpu-flags> --ipc=host --shm-size 2g \
  --entrypoint sleep \
  -v <host-output-dir>:/workspace/out \
  -v <host-hf-cache-dir>:/root/.cache/huggingface \
  -v <host-state-dir>/venvs:/opt/venvs \
  -v <host-state-dir>/pip-cache:/root/.cache/pip \
  -e HF_TOKEN \
  -w /workspace \
  <image> infinity
```

`<host-state-dir>` is any host folder that outlives containers (e.g. next to
the output dir). Mounting the pip cache means a recreated container
reinstalls ModelOpt from local wheels in minutes instead of re-downloading
~350MB from PyPI.

**Jetson:** add `--ulimit memlock=-1 --ulimit stack=67108864` (pinned
host memory is the GPU's memory here). Jetson images built on
`jetson-containers` may use a different entrypoint or have no `vllm serve`
entrypoint at all; check `docker inspect <image> --format
'{{json .Config.Entrypoint}}'` and pass `--entrypoint ""` plus an explicit
`vllm serve …` in Step 4 when it isn't `vllm serve`.

If the input model is a local checkpoint, also add
`-v <host-model-dir>:/workspace/model:ro`. `HF_TOKEN` is passed through from
the host environment and is only needed for gated repos (a token file in the
mounted cache is picked up automatically). Use the resolved real path for the
cache if `~/.cache/huggingface` is a symlink (`readlink -f`).

Install ModelOpt and the calibration extras only. **Don't** reinstall torch,
torchvision or transformers: the image's versions match each other, and
replacing any of them starts a multi-GB download or an ABI mismatch. Dry-run
first:

```bash
docker exec modelopt bash -lc '
pip install --dry-run nvidia-modelopt==0.45.0 datasets hf_transfer 2>&1 | grep "Would install"
# must not list torch / torchvision / nvidia-cudnn / nvidia-cublas. A small
# nvidia-nccl-cu13 downgrade (2.30.7 -> 2.29.7) is expected and harmless for
# calibration (x86); it is one reason serving uses a fresh container. On a
# Jetson image whose torch is a custom build (version like 2.x.0a0+nv…), a
# dry-run listing torch means pip would replace it with a PyPI wheel that has
# no Jetson kernels: stop and pin a ModelOpt version that accepts that torch.
# Slow (see the timing note below): start the Step 5 BF16 baseline meanwhile.
pip install -q --retries 10 --timeout 30 nvidia-modelopt==0.45.0 datasets hf_transfer
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
git clone -q --depth 1 --branch 0.45.0 https://github.com/NVIDIA/Model-Optimizer.git /opt/Model-Optimizer
grep -v "^flash-attn" /opt/Model-Optimizer/examples/llm_ptq/requirements.txt > /tmp/reqs.txt
pip install -q --retries 10 --timeout 30 -r /tmp/reqs.txt
python3 -c "import torch, modelopt, transformers; print(torch.__version__, modelopt.__version__, transformers.__version__)"'
```

Expect 3–40 minutes depending on PyPI speed (17 and 38 minutes measured on
different days). Run it in the background and do the Step 5 BF16 baseline
serves and measurements in the meantime, in their own `vllm-serve` container
(PyPI ran at ~900KB/s on one day). **Stop that server (`docker stop vllm-serve`)
before Step 3**: at `--gpu-memory-utilization 0.8` it holds ~38GB of a 48GB
card, leaving too little to load the BF16 model and calibrate. Clone the Model-Optimizer repo at the tag matching
the installed `nvidia-modelopt` (mismatched tags can mean different CLI args or
missing exclusion logic). `flash-attn` is skipped because its pin forces a 10+
minute source build that does nothing for calibration; `hf_ptq.py` gets
`--attn_implementation sdpa` instead. Two warnings at import are expected and
harmless: `Failed to import modelopt vllm plugin` and `transformers X is not
tested with current version of modelopt`. `hf_ptq.py`'s `Unknown model type
<Arch>. Continue exporting...` (seen on LFM2) is harmless too.

**Stall rule:** if the first `pip install` (or the `git clone`) is still running after ~15 minutes,
check throughput with `docker exec modelopt cat /proc/net/dev` twice a minute
apart. Below ~100KB/s it will take over an hour; seed the pip cache from an
older container (pip cache, Model-Optimizer clone) as above instead of waiting.

Older setups this replaces, and why: `nvcr.io/nvidia/pytorch:25.01-py3` has a
torch too old for current ModelOpt (pip then downloads ~3GB of torch from PyPI),
a dead `pypi.ngc.nvidia.com` index baked into `/usr/pip.conf`, and an alpha
torchvision that crashes under `transformers>=5`. A venv reusing an older
image's torch via `--system-site-packages` breaks (new torch pulled anyway, or
numpy/torchvision ABI errors). That's why Step 1b's venvs are fully isolated. The TensorRT-LLM image pins ModelOpt to 0.37
(no `w4a16_nvfp4`, Hub-only calibration data), which is why quantization
doesn't happen there.

## Step 1b: Get the model's code, not only its weights

Some models can't be loaded from weights and `config.json` alone. Check
before quantizing, because the failure otherwise shows up halfway into a run.
It looks like `does not recognize this architecture`, `KeyError:
'<model_type>'`, or `No module named 'transformers_modules.…'`.

```bash
docker exec -i modelopt python3 - <<'EOF'   # -i: pass the heredoc through
import json, os, transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
d = "<snapshot path, or /workspace/model>"
c = json.load(open(f"{d}/config.json"))
print("model_type:", c.get("model_type"), "| architectures:", c.get("architectures"))
print("auto_map:", c.get("auto_map"))
print(".py files present:", sorted(f for f in os.listdir(d) if f.endswith(".py")))
print(f"model_type known to transformers {transformers.__version__}:", c.get("model_type") in CONFIG_MAPPING)
EOF
```

| What the check shows | What it means | What to do |
|---|---|---|
| `model_type` is known, no `auto_map` | Standard architecture | Nothing to do. |
| `auto_map` present, but some of the `.py` files it names are missing | Custom code on the Hub, which the weights-only copy lacks | Download the code files (below). |
| `auto_map` entries of the form `<org>/<repo>--modeling_x.Class` | The code lives in *another* Hub repo, fetched at load time | Download that repo's code too, while the Hub is reachable. Step 3 runs offline. |
| `model_type` unknown, no `auto_map` | Needs a newer transformers than the image has | Use an isolated venv (below) with the transformers version the model card names. |
| The model card says to install a package or clone a GitHub repo (typical for VLAs: openpi, Isaac-GR00T, LeRobot …) | The architecture isn't in transformers at all | Clone it at the release tag matching the checkpoint (e.g. `n1.7-release` for GR00T N1.7; an older clone lacks the new model class) to `/opt/src/<name>` in the container, install it into an isolated venv, and quantize with Step 3b. |

After cloning, check for **Git LFS pointers**. Demo data and assets in GitHub
repos are often stored in LFS, and a clone without `git-lfs` gets 130-byte
text stubs instead (`find <repo> -size -200c -name "*.mp4"`). Run `apt-get
install -y git-lfs && git lfs pull` in the clone, or copy the files from a
host clone that has them.

**Code that names other Hub repos.** Model code often loads a second model by
repo ID, e.g. GR00T's backbone `"model_name": "nvidia/Cosmos-Reason2-2B"` in
`config.json`. That breaks offline runs: transformers 4.57's tokenizer loader
calls the Hub's `model_info()` for any non-local ID even with
`HF_HUB_OFFLINE=1` (`OfflineModeIsEnabled: Cannot reach
https://huggingface.co/api/models/...`). Point it at the cached snapshot from
a working copy (below), and patch every config that carries the name. For
GR00T that's `config.json` *and* `processor_config.json` →
`processor_kwargs.model_name`, where it was absent and fell back to a code
default. Some code picks classes by matching the name string (GR00T:
`"nvidia/Cosmos-Reason2" in model_name`, otherwise `Unsupported model name`).
In that case use a local path that still contains the ID (this works for
N1.7's substring check). **GR00T N1.6 is different:** its backbone code
(`nvidia/Eagle-Block2A-2B-v2`) is vendored in the repo under
`gr00t/model/modules/nvidia/`, so nothing is fetched from the Hub, and
`processing_gr00t_n1d6.py` asserts `model_name ==
"nvidia/Eagle-Block2A-2B-v2"` exactly. Leave that name unchanged for N1.6. Its
demo data is `demo_data/gr1.PickNPlace` (embodiment `gr1`), stored in LFS.
For N1.7:

```bash
mkdir -p /workspace/models/nvidia && ln -sfn <cached snapshot of nvidia/Cosmos-Reason2-2B> /workspace/models/nvidia/Cosmos-Reason2-2B
# then set model_name to /workspace/models/nvidia/Cosmos-Reason2-2B in the working copy's configs
```

**Downloading missing code.** It's small, so fetch only the code and config
files:

```bash
docker exec modelopt hf download <org>/<repo> --include "*.py" "*.json" "*.model" "*.tiktoken" "tokenizer*" \
  --local-dir /workspace/code/<repo>
```

- **HF repo input:** download into the cache with `hf download <org>/<repo>
  --include "*.py"` instead, so the snapshot itself becomes complete.
- **Local checkpoint, or configs that need patching:** the source is
  read-only (a mount, or HF cache blobs shared with other runs), so assemble a
  working copy. Symlink everything except the files you change, and write
  those as real files.
  Symlink the weights and copy the code beside them, then pass
  `/workspace/model-full` to `hf_ptq.py`:
  ```bash
  docker exec modelopt bash -c 'mkdir -p /workspace/model-full && ln -sf /workspace/model/* /workspace/model-full/ && cp -n /workspace/code/<repo>/*.py /workspace/model-full/'
  ```
- **Unknown source:** if you can't tell which repo a local checkpoint came from
  (check `_name_or_path` in `config.json` and any README), ask the user rather
  than guessing.
- **Tell the user what will run:** `--trust_remote_code` executes that repo's
  Python, so say which repo (and commit) the code came from.

**Isolated environment with uv, when requirements conflict.** If the model's
code needs versions the image doesn't have (another transformers, a pinned
torch, an extra package that would downgrade something), don't install them
into the image's Python. Serving and every other model depend on it. Make a
uv venv inside the `modelopt` container instead.

If the repo ships a `uv.lock`, reproduce its environment exactly. This is
what the authors tested, including prebuilt wheels such as flash-attn:

```bash
docker exec modelopt bash -lc '
pip install -q uv
cd /opt/src/<repo>
UV_PROJECT_ENVIRONMENT=/opt/venvs/<model-name> uv sync --frozen --managed-python --python <repo requires-python>
'
```

Otherwise build one by hand from the model's requirements:

```bash
docker exec modelopt bash -lc '
pip install -q uv
uv venv /opt/venvs/<model-name> --managed-python --python 3.12   # fully isolated: no --system-site-packages
. /opt/venvs/<model-name>/bin/activate
uv pip install --dry-run torch==2.13.0 nvidia-modelopt==0.45.0 datasets <model requirements> \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
'
```

**Always pass `--managed-python`.** Without it, uv reuses the image's Ubuntu
`python3.X` if one matches, and that has no development headers. ModelOpt's
FP8 CUDA extension and Triton compile at runtime and then fail with
`Python.h: No such file or directory`. uv-managed Python includes the headers.
Rebuilding a venv reuses uv's package cache, so the re-sync took 13 seconds,
against 22 minutes for the first download of the GR00T lock (15GB venv).

- **torch:**
  - Keep the image's torch version and CUDA build when the model allows it.
    The driver already supports that build.
  - If the model pins a different torch, choose a build whose CUDA version is
    no newer than what the driver shows in `nvidia-smi`.
  - torch is a multi-GB download, so tell the user the size from the dry-run
    before installing.
  - **Jetson:** Thor (JetPack 7) is SBSA, so the regular aarch64 `cu130`
    wheels from `download.pytorch.org/whl/cu130` work. Orin (JetPack 6) needs
    Jetson wheels from `https://pypi.jetson-ai-lab.io/jp6/cu126` (Thor's
    equivalent: `/jp7/cu130`). A plain PyPI torch on Orin imports fine but
    reports `torch.cuda.is_available() == False`. Re-run the Step 1 matmul
    check inside the venv.
- **Pin exactly, never open-ended.** Use `transformers==4.57.3`, not
  `transformers<5`. With only an upper bound, uv backtracked to transformers
  4.12.2 (from 2021) in a dry-run. Read the resolved versions in the dry-run
  output before installing.
- **ModelOpt version must fit the model's torch and transformers.**
  - 0.45 and 0.44 need `torch>=2.8`.
  - 0.43.0 accepts torch 2.6+ and transformers 4.53–4.x. It was used for GR00T
    N1.7 (torch 2.7.1, transformers 4.57.3).
  - 0.40–0.43 all require `transformers>=4.53,<5` for the `hf` extra. A model
    pinned below 4.53 needs **0.39.0** (`transformers>=4.48,<4.58`). GR00T
    N1.6 (`n1.6.1-release`) pins `torch==2.7.1` and `transformers==4.51.3`, so
    it needs ModelOpt 0.39.0, not 0.43.
  - Check a release with
    `curl -s https://pypi.org/pypi/nvidia-modelopt/<ver>/json`
    (`requires_dist`), and dry-run so it doesn't pull a different torch.
- **ModelOpt and the Model-Optimizer clone:** install ModelOpt in the
  venv too, plus `examples/llm_ptq/requirements.txt` minus flash-attn as in
  Step 1. Reuse the `/opt/Model-Optimizer` clone. If the transformers version
  the model needs is one ModelOpt can't import with (an error, not just the
  "not tested" warning), tell the user. Don't patch ModelOpt.
- **Running quantization:** call the venv's interpreter explicitly:
  `docker exec modelopt /opt/venvs/<model-name>/bin/python hf_ptq.py …`
  (Step 3), or run the Step 3b script with it.
- **Recording the environment:** run `uv pip freeze` into
  `/workspace/out/<model-name>-<qformat>/quant_env.txt`, so the export records
  what produced it.
- **Persistence:** the venv lives in the container and is lost if the
  container is removed. To keep it, mount a host directory at `/opt/venvs`
  when creating the container (Step 1).
- **Serving:** still uses the stock vLLM image (Step 4). vLLM can serve
  custom-code models with `--trust-remote-code` only if the image's own
  transformers can run that code. If it can't, tell the user. The options are
  a newer vLLM tag, or evaluating with the model's own inference code in the
  venv. Don't install into the serving container.

This path was run end to end on GR00T N1.7 (see Step 3b's reference).

**Check that vLLM can serve the architecture before quantizing.** transformers
being able to run a model doesn't mean vLLM can. Look up `architectures[0]`
from `config.json` in the serving image's registry:

```bash
docker exec modelopt python3 -c "import sys; from vllm.model_executor.models.registry import _VLLM_MODELS as M; print(sys.argv[1], sys.argv[1] in M)" <architectures[0]>
```

`True` → continue. `False` → vLLM falls back to its generic Transformers
backend, which only handles models whose every token mixer is attention.
Conv, Mamba or linear-attention layers fail at serve time with `Layer 0
(conv) does not dispatch through the Transformers attention interface`
(LiquidAI LFM3 on vLLM 0.32, which wasn't on vLLM `main` either as of
2026-10-29). Settle it with a 2-minute BF16 serve: the Step 4 command
pointed at the original snapshot path. **If BF16 doesn't serve, the export
won't either. Stop here and tell the user, before spending time on
calibration.** Their options: wait for vLLM support, serve with another
engine the model card names, or quantize anyway and keep the checkpoint for
later (say plainly it can't be served or measured today). A VLA that isn't a
text generator (GR00T, OpenVLA, π0) isn't served by vLLM at all; it takes
Step 3b's own inference-code path, so this check doesn't apply to it.

## Step 2: Pick the qformat the user chose, and check it against vLLM

List the formats this ModelOpt version actually has rather than guessing from
memory — the preset names shift between releases:

```bash
docker exec modelopt python3 -c "from modelopt.recipe.presets import QUANT_CFG_CHOICES; print(sorted(QUANT_CFG_CHOICES))"
```

ModelOpt 0.45.0 has 40+ presets. These are the ones to present when asking
about precision. If the user's choice isn't in the installed list, tell them
and ask again rather than substituting a nearby format yourself:

| Goal | qformat | Calibration | vLLM | TensorRT engine (`/totrt`) | Notes |
|---|---|---|---|---|---|
| Best speed/accuracy on Ada/Hopper | `fp8` | **required** (static activations) | ✅ verified | ✅ verified | Native tensor cores. Usual recommendation. |
| Smallest footprint, speed secondary | `w4a16_nvfp4` | **not needed** (weight-only; skipped automatically) | ✅ verified (Marlin) | not tested | See warning below. |
| INT8 SmoothQuant (real INT8×INT8) | `int8_sq` | **required** (SQ) | ✅ after Step 3c conversion | not tested | Needs the Step 3c converter; **Jetson Thor:** extra env var there. |
| INT4 weight-only | `int4_awq` | **required** (AWQ) | ✅ after Step 3c conversion | not tested | Needs the Step 3c converter (raw export is rejected; see below). |
| Native speed *and* size on Blackwell | `nvfp4` | **required** (static activations) | ✅ verified on Ada (runs weight-only via Marlin) | not tested | Real FP4 compute only where Step 0 confirmed Blackwell. On Ada, vLLM logs `Using MarlinNvFp4LinearKernel` and ignores the FP4 activations, so it behaves like `w4a16_nvfp4`. |

**The INT4 trap:** `int4_awq` (classic weight-only AWQ) quantizes and exports
fine, but its exported `quant_algo: W4A16_AWQ` is **not** in the list vLLM's
ModelOpt loader accepts, and the INT8 SQ export hits the same wall. Serving
the raw export fails with:

```
ValueError: ModelOpt currently only supports: ['FP8', 'FP8_PER_CHANNEL_PER_TOKEN',
'FP8_PB_WO', 'NVFP4', 'W4A16_NVFP4', 'MXFP8', 'MIXED_PRECISION'] quantizations in vLLM.
```

The fix is Step 3c: convert the export to compressed-tensors format, which
vLLM's other loader serves (INT8 verified on Thor, INT4 numerically validated
against BF16). Run the conversion — don't skip quantization or drop to
`w4a16_nvfp4` just because the raw export fails. `w4a16_nvfp4` remains a fine
alternative when you don't specifically need INT: it's weight-only 4-bit
(activations stay BF16), and on non-Blackwell hardware vLLM runs it through a
Marlin dequant-on-the-fly kernel rather than true FP4 tensor core ops. You'll
see this logged at serve time, and it's expected, not a bug:

```
Your GPU does not have native support for FP4 computation but FP4 quantization
is being used. Weight-only FP4 compression will be used leveraging the Marlin
kernel. This may degrade performance for compute-heavy workloads.
```

**On Thor this warning is misleading:** vLLM prints it whenever it serves a
checkpoint through Marlin, and it always serves `w4a16_nvfp4` through Marlin,
even though Thor has FP4 tensor cores. For a W4A16 checkpoint weight-only is
what was asked for, so it's expected. For an `nvfp4` (W4A4) checkpoint on
Thor, though, this warning means a real FP4 kernel was disabled or failed:
check the `Using …Kernel for NVFP4 GEMM` line.

Set expectations accordingly: measured on Ada, `w4a16_nvfp4` came out roughly
tied with FP8 on single-request latency and *slower* under concurrent load —
the smaller checkpoint doesn't translate into a speed win without real FP4
hardware. Its only genuine advantage there is disk/VRAM footprint. If a user
wants INT4 specifically for speed on non-Blackwell hardware, tell them this
upfront rather than letting them discover it after the quantize+serve cycle.

## Step 3: Quantize with the official script, offline

First check the quantization GPU is free (`nvidia-smi --query-gpu=index,memory.used --format=csv`);
stop any baseline `vllm-serve` still running from Step 1.

Use `hf_ptq.py`, not a custom script calling `mtq.quantize()` +
`export_hf_checkpoint()` directly. This matters a lot for VLMs, MoE models,
and hybrid-attention (Mamba/GatedDeltaNet-style) models: `hf_ptq.py` has
built-in detection (`is_multimodal_model`,
`extract_and_prepare_language_model_from_vl`) that correctly identifies and
excludes vision towers, MTP heads, and router/gate layers from quantization.
A hand-rolled script using the bare `mtq.quantize()` API will still wrap those
modules' quantizers — but since calibration data for a text model never
exercises the vision tower, those quantizers end up uncalibrated while still
active at export, producing a checkpoint with broken vision-tower weights.
(Plain dense LLMs are fine either way.) If `hf_ptq.py` can't load the model or
misdetects it, use Step 3b instead.

**Run without the Hub.** On a shared machine, Hugging Face can rate-limit the
whole IP address (`429 Too Many Requests`, even with a token, even on
`whoami`); it lasted 45+ minutes in a real run. Two things make the run fully
offline:

1. **Pass the model's local snapshot path**, not its repo ID, once it's in the
   cache. A model that isn't cached yet has to be downloaded once while the
   Hub is reachable (`hf download <repo>` in the container).
   ```bash
   docker exec modelopt bash -c 'ls -d /root/.cache/huggingface/hub/models--<Org>--<Name>/snapshots/*/'
   ```
2. **When the preset requires calibration, calibrate from a local `.jsonl` file.** ModelOpt 0.45 accepts a `.jsonl`
   path as `--dataset` and auto-detects `messages` (applies the model's chat
   template), `prompt`, `text` or `input` columns. Build one from any cached
   chat dataset, e.g. `HuggingFaceH4/ultrachat_200k`. Calibration only needs
   representative activations, not domain-matched text:
   ```bash
   docker exec modelopt python3 -c "
   import glob, json, pandas as pd
   f = sorted(glob.glob('/root/.cache/huggingface/hub/datasets--HuggingFaceH4--ultrachat_200k/snapshots/*/data/train_sft-00000-*.parquet'))[0]
   df = pd.read_parquet(f, columns=['messages']).head(512)
   with open('/workspace/calib.jsonl', 'w') as fh:
       for m in df['messages']:
           fh.write(json.dumps({'messages': [dict(role=x['role'], content=x['content']) for x in m]}) + '\n')
   "
   ```
   **Base / `-pt` / `-base` checkpoints have no chat template**, and
   `messages` rows then fail with `Cannot use chat template functions because
   tokenizer.chat_template is not set`. Check first with
   `AutoTokenizer.from_pretrained('<snapshot>').chat_template is None`; if so,
   write a `text` column instead (same snippet, different row):
   `fh.write(json.dumps({'text': '\n\n'.join(x['content'] for x in m)}) + '\n')`
   into `/workspace/calib_text.jsonl`, and pass that as `--dataset`.
   If no chat dataset is cached and the Hub is reachable, `--dataset
   cnn_dailymail` (streams `abisee/cnn_dailymail`) also works. Don't rely on
   the default (`cnn_nemotron_v2_mix`), which pulls in a gated NVIDIA dataset.

**Calibration is decided by the preset, not by `--calib_size`.** Check it before
offering the choice:
`python3 -c "from modelopt.recipe.presets import QUANT_CFG_CHOICES; from modelopt.torch.quantization.config import need_calibration; print(need_calibration(QUANT_CFG_CHOICES['<qformat>']))"`.
For `fp8` (W8A8), `nvfp4`, `w4a8_*`, `int4_awq` and `int8` this prints `True`:
pass `--dataset` and `--calib_size 512`. For `w4a16_nvfp4`, `fp8_pb_wo`,
`fp8_2d_blockwise_weight_only`, `fp8_pc_pt`, `int8_wo`, `int4_blockwise_weight_only`,
`nvfp4_mlp_weight_only` it prints `False`; the script logs "Dynamic quantization.
Calibration skipped." and ignores the dataset, so recommend skipping it. Do not
use `--calib_size 0` to skip a preset that requires calibration — `hf_ptq.py`
raises `IndexError: list index out of range` in the dataloader (verified on
ModelOpt 0.45 with `fp8`).

Then:

```bash
docker exec -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 modelopt bash -lc \
  "cd /opt/Model-Optimizer/examples/llm_ptq && python3 hf_ptq.py \
    --pyt_ckpt_path <snapshot path, /workspace/model, or /workspace/model-full (Step 1b)> \
    --qformat <from Step 2> \
    --kv_cache_qformat none \
    --dataset /workspace/calib.jsonl \
    --calib_size 512 \
    --attn_implementation sdpa \
    --export_path /workspace/out/<model-name>-<qformat> \
    --trust_remote_code"
```

(Qwen2.5-3B: ~3.5 minutes. `--kv_cache_qformat none` works even though
`none` isn't printed in the KV preset list.)

**KV cache:** without `--kv_cache_qformat none`, `hf_ptq.py` also quantizes
the KV cache to FP8 with fixed scales of 1.0. Measured on Qwen2.5-3B-Instruct
(64 ultrachat samples, served by vLLM):

| | Perplexity |
|---|---|
| BF16 | 4.10 |
| FP8 weights, BF16 KV cache | 4.18 (+1.9%) |
| FP8 weights + FP8 KV cache | 4.46 (+8.7%) |

So the FP8 KV cache costs most of the accuracy. It also breaks generation if
the checkpoint later goes through `/totrt`.

After it finishes, confirm the export exists **on the host**
(`ls <host-output-dir>/<model-name>-<qformat>`), not just inside the
container. That check is what proves it'll survive the container.

**Then give the files back to the host user.** The container runs as root,
so the export is root-owned. Some files, including `model.safetensors`, are
created mode `0600`, which means the user can't even read their own model
outside Docker:

```bash
docker exec modelopt chown -R $(id -u):$(id -g) /workspace/out/<model-name>-<qformat>
docker exec modelopt chmod -R u+rw,go+r /workspace/out/<model-name>-<qformat>
```

Sanity-check the export before moving on:

```bash
python3 -m json.tool <host-output-dir>/<model-name>-<qformat>/hf_quant_config.json | head -20
```

Confirm `quant_algo` and `kv_cache_quant_algo` match what the user chose, and
that `exclude_modules` mentions vision/MTP/router patterns if the source model
has them (grep for `visual`, `mtp`, `router`). `hf_ptq.py` also prints a
sample generation before and after quantization; they should read similarly.
If the **before** sample is already incoherent, the preview proves nothing
either way: it continues a calibration sample truncated mid-turn, using the
model's own `generation_config` (sampling, repetition penalty). Don't debug
the model over it; judge quality by Step 5's perplexity.

**VLMs: compare weight-name prefixes with the original.** transformers 5 can
re-nest a vision tower on save, and vLLM then fails to load it. PaliGemma
(SigLIP) exported as `model.vision_tower.<x>` while vLLM 0.32 expects
`model.vision_tower.vision_model.<x>` (the original's layout), failing with
`There is no module or parameter named 'vision_tower.embeddings'`:

```bash
docker exec -i modelopt python3 - <<'EOF'
import json, glob, collections
from safetensors import safe_open
exp = "/workspace/out/<model-name>-<qformat>"
orig = "<original snapshot path>"
pre = lambda keys: collections.Counter(".".join(k.split(".")[:4]) for k in keys if "vision" in k or "visual" in k)
ek = [k for f in glob.glob(exp + "/*.safetensors") for k in safe_open(f, "pt").keys()]
idx = glob.glob(orig + "/*.index.json")
ok = json.load(open(idx[0]))["weight_map"].keys() if idx else [k for f in glob.glob(orig + "/*.safetensors") for k in safe_open(f, "pt").keys()]
print("export:", sorted(pre(ek))[:8]); print("orig:  ", sorted(pre(ok))[:8])
EOF
```

If a tower lost a nesting level, insert the missing level and keep the
export's own top-level prefix: `model.vision_tower.X` →
`model.vision_tower.vision_model.X` (not the original's `vision_tower.…`
without `model.`). The bundled script does exactly that, and it is safe to
re-run:

```bash
docker cp <skill-dir>/scripts/fix_vlm_keys.py modelopt:/workspace/fix_vlm_keys.py
docker exec modelopt python3 /workspace/fix_vlm_keys.py /workspace/out/<model-name>-<qformat> --dry-run
docker exec modelopt python3 /workspace/fix_vlm_keys.py /workspace/out/<model-name>-<qformat>
```

The script keeps each rewritten file's owner and mode, so the earlier
chown/chmod still holds. If you renamed keys some other way, re-run those two
lines afterwards: a file saved as root in the container is `root:root 0600`
and unreadable to the host user.

Defaults fit PaliGemma/SigLIP; for another tower pass `--prefix` and
`--insert` to match what the comparison showed. Don't patch vLLM or ModelOpt.
Verified on PaliGemma twice: 437 keys renamed, then the FP8 export served and
matched BF16 on image questions.

## Step 3b: Fallback — `mtq.quantize()` on the language model only

Use this only when `hf_ptq.py` can't do the job:

- **It can't load the model.** Typical for VLAs (OpenVLA, π0, GR00T, …) and
  other custom-code policies that aren't a standard `AutoModel`.
- **It loads, but its VLM detection misses.** `is_multimodal_model()` only
  recognizes `config.vision_config`, a `language_model` attribute, and a few
  special cases. If it returns False, or the language model lives somewhere
  other than `model.model.language_model` / `model.language_model`, the whole
  model gets quantized, vision encoder and action head included. Check before
  trusting it:
  `python3 -c "from modelopt.torch.export.model_utils import is_multimodal_model, get_language_model_from_vl; ..."`
  on the loaded model.

The rule that makes this safe: **quantize the language-model submodule, but
calibrate through the full model.** Calling `mtq.quantize()` on the whole
model is exactly the mistake Step 3 warns about. Wrapping only the submodule
leaves everything else untouched in BF16:

```python
import modelopt.torch.quantization as mtq
try:                                               # ModelOpt 0.45+
    from modelopt.recipe.presets import QUANT_CFG_CHOICES
    CFG = QUANT_CFG_CHOICES["<user's qformat>"]
except ImportError:                                # 0.39–0.44 (e.g. the GR00T venvs): same presets as constants
    CFG = mtq.FP8_DEFAULT_CFG                      # or mtq.NVFP4_DEFAULT_CFG, mtq.W4A8_AWQ_BETA_CFG, ...

for name, mod in policy.named_children():          # find the real LM path first
    print(name, type(mod).__name__, sum(p.numel() for p in mod.parameters()) / 1e9, "B")

lm = policy.language_model                         # whatever the printout showed

def forward_loop(_lm):
    with torch.no_grad():
        for batch in calib_batches:                # real inputs, see below
            policy(**batch)                        # full forward: vision -> projector -> LM -> head

mtq.quantize(lm, CFG, forward_loop)
mtq.print_quant_summary(policy)                    # enabled quantizers must sit only under the LM
```

`modelopt.recipe.presets` exists only in ModelOpt 0.45+; older venvs have
the same presets as `mtq.<FMT>_CFG` constants (`fp8` → `FP8_DEFAULT_CFG`), and
the Step 2 `need_calibration` one-liner needs the same substitution there.
Either way you get the preset `hf_ptq.py --qformat` would
use, including its default exclusions (`lm_head`, routers/gates). Things to
keep in mind:

- **Keep the action head (and any value/flow-matching head) in BF16.** It's
  small, so there's nothing to gain, and its outputs are robot actions, where
  quantization error matters directly.
- **Calibrate on real observations + instructions.** The LM's activations
  depend on the visual tokens, so text-only data gives the wrong activation
  ranges for a policy. A few hundred samples from the robot's own dataset are
  enough.
- **Decide how the result will be loaded *before* quantizing.**
  `modelopt.torch.export.export_hf_checkpoint(policy, export_dir=...)` only
  works for standard HF architectures, and vLLM won't serve a VLA. For custom
  models, save with `modelopt.torch.opt.save(policy,
  "/workspace/out/<name>-<qformat>/modelopt_state.pth")` and restore with
  `mto.restore()` in the model's own inference code. If nothing downstream can
  load it, tell the user before spending GPU time on calibration.
- **Verify with the model's own eval**, not perplexity: a few rollouts or
  offline action-error against held-out demos, compared with the BF16 policy.
  For a diffusion or flow-matching action head, fix the noise seed so BF16 and
  quantized see the same noise. Also measure BF16 against *itself* with
  different seeds. That's the yardstick for whether the quantized difference
  matters.
- **Scope the quantizers with name patterns, then prove it.** Instead of
  calling `mtq.quantize()` once per submodule, you can quantize the whole
  policy with a config that starts all-off (`"default": {"enable": False}`)
  and turns the preset's `*weight_quantizer` / `*input_quantizer` settings
  back on only under scope patterns such as `*language_model.layers.*`. Then
  list every enabled `TensorQuantizer` and fail if any is outside the scopes.
  Don't trust the summary by eye: ModelOpt inserts quantizers everywhere (1750
  for GR00T), and only the count of enabled ones (672) tells you the scope
  held.
- **Save small.** Save `{"modelopt_state": mto.modelopt_state(model),
  "quantizer_state_dict": {k: v for k, v in model.state_dict().items() if
  "_quantizer" in k}}`: 0.9MB for GR00T, versus another copy of the weights
  with `mto.save`. To restore: load the original model,
  `mto.restore_from_modelopt_state(model, state["modelopt_state"])`, then
  `model.load_state_dict(state["quantizer_state_dict"], strict=False)`. Check
  that no `_quantizer` keys are missing. Evaluate the restored model, not the
  one still in memory, so the saved file is what gets verified.
- **Fake quantization is slower, not faster.** In PyTorch, `mtq` simulates
  FP8 (GR00T: 137 → 632 ms per action chunk, plus a one-time ~140s build of
  `modelopt_cuda_ext_fp8`). The deliverable is the accuracy result and a
  calibrated state for a later ONNX/TensorRT export. Say so up front.
- **Keep the model-specific scripts with the export** (quantize, restore, eval).
  Nothing generic can load the result, so they are part of the artifact. Then
  run Step 3's chown/chmod on the export folder (the outputs are root-owned).

**Reference: GR00T N1.7-3B** (`nvidia/GR00T-N1.7-3B`, FP8, RTX 6000 Ada,
DROID `demo_data/droid_sample` with embodiment
`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`). The scope was the 16 Qwen3-VL text
layers plus the 32 DiT blocks of the action head, 2.2B of 3.1B parameters.
The vision tower, VL self-attention and state/action encoders and decoders
stayed BF16. Calibration took 39s on episodes 0–1 (action horizon 8, 4
denoising steps):

| Episode | MSE vs ground truth, BF16 → FP8 | FP8 vs BF16, same seed | BF16 vs BF16, other seeds |
|---|---|---|---|
| 0 | 0.0143 → 0.0139 | 6.7% | 21–23% |
| 1 | 0.0030 → 0.0030 | 2.5% | 14–20% |
| 2 (held out) | 0.0273 → 0.0267 | 1.5% | 12–13% |

(Relative differences are the norm of the prediction difference divided by
the spread of the BF16 predictions.) FP8 moved the actions 3–9x less than the
diffusion noise already does, and error against ground truth didn't get
worse.

## Step 3c: Convert an INT8/INT4 export so vLLM can load it

Only for `int8_sq` / `int8_weight_only` / `int4_awq` exports that will be
served by vLLM. FP8/NVFP4 exports load natively — skip this step for them.
The ModelOpt loader rejects the INT `quant_algo`s (Step 2's trap), but vLLM's
*compressed-tensors* loader takes the same math; the bundled converter rewrites
the export into a `compressed-tensors` checkpoint that serves from Step 4 on
with **no `--quantization` flag** (auto-detected from the rewritten
`config.json`), and Steps 4/4b/5/6 proceed unchanged against the converted dir:

```bash
docker cp <skill-dir>/scripts/int8_sq_to_compressed_tensors.py modelopt:/workspace/
docker exec modelopt python3 /workspace/int8_sq_to_compressed_tensors.py \
  /workspace/out/<model>-int8_sq /workspace/out/<model>-int8_sq-ct
# or, for INT4 AWQ:
docker cp <skill-dir>/scripts/int4_awq_to_compressed_tensors.py modelopt:/workspace/
docker exec modelopt python3 /workspace/int4_awq_to_compressed_tensors.py \
  /workspace/out/<model>-int4_awq /workspace/out/<model>-int4_awq-ct
```

The output dir must not exist; both scripts run in the `modelopt` container
(needs CUDA for speed) and take minutes, not hours — no calibration, pure
repacking. What they do, and why:

- **Smoothing folded into the weights.** Both formats store `W·diag(s)^-1`
  plus a per-linear `pre_quant_scale` that the activation must be multiplied
  by at runtime. vLLM has no per-linear pre-quant scale, and sibling
  projections (q/k/v, gate/up) share one norm with *different* `s`, so it
  can't be folded into the norm. The converters compute `W_eff =
  dequant(W) × pre_quant_scale` and requantize — a double quantization, so
  always run the Step 5 perplexity check against BF16 before shipping.
- **INT8:** requantized symmetric per output channel; the static calibrated
  `input_scale` is dropped and activations quantize dynamically per token
  (real INT8×INT8). `int8_weight_only` exports convert too — they just get
  `input_activations: null`.
- **INT4:** two repacks. ModelOpt packs signed nibbles along the *output* dim
  (`uint8 [out/2, in]`, row 2i = low nibble); compressed-tensors packs along
  the *input* dim (`int32 [out, in/8]`, column j → nibble j%8 of word j//8).
  The converter unpacks, requantizes to CT's symmetric convention
  (`scale = max(max/7, -min/8)` per group), and repacks. vLLM runs it through
  Marlin (dequant-on-the-fly, A16 — same class as `w4a16_nvfp4`, so expect
  footprint wins, not an FP4-class speed win, versus FP8).

**Jetson Thor:** the vLLM image's CUTLASS INT8 kernel has no sm_110 build and
kernel selection dies instead of falling back — add
`-e VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel` to the Step 4
`docker run` (Triton handles INT8 fine on Thor).

Validation status: INT8 conversion was validated end-to-end on Thor (27B VLM:
served, all linears on INT8 tensor-core kernels, gsm8k within 1 pt of BF16).
INT4 conversion is numerically validated per-tensor against BF16 (correct
nibble order/pqs direction proven by controls, ~0.99 cosine = normal AWQ g128
error, pack round-trip exact), but a full INT4-CT *serving* run has not been
done in this skill yet — treat Steps 4/5 for it as first contact: check the
log for the Marlin kernel line and watch the Step 5 ppl closely.

## Step 4: Serve with vLLM

Serve from a **fresh container** of the same image, so the packages installed
into `modelopt` (including the NCCL downgrade) never touch serving. The idle
`modelopt` container holds no GPU memory, so leave it running. Pick a free
host port (`ss -ltn`):

```bash
docker rm vllm-serve 2>/dev/null   # an old stopped one; a running one is left alone (and this run then fails: stop it first)
docker run -d --name vllm-serve <gpu-flags> --ipc=host --shm-size 2g -p <port>:8000 \
  -e HF_HUB_OFFLINE=1 \
  -v <host-output-dir>:/workspace/out:ro \
  -v <host-hf-cache-dir>:/root/.cache/huggingface \
  <image> \
  /workspace/out/<model-name>-<qformat> \
  --tensor-parallel-size <1 or more, based on VRAM> \
  --gpu-memory-utilization <fraction> \
  --max-model-len <len, see below> \
  --trust-remote-code
timeout 900 bash -c 'until curl -sf localhost:<port>/v1/models >/dev/null; do sleep 5; done'
docker logs vllm-serve 2>&1 | grep -iE "modelopt|Selected .*Kernel|kv cache|error"
```

The image's entrypoint is already `vllm serve`, so the model path comes
first. `HF_HUB_OFFLINE=1` keeps a rate-limited Hub from blocking startup.
`--gpu-memory-utilization` is a fraction of the *whole* card, so on a shared
GPU it must fit in what's actually free; 0.8 on an otherwise idle 48GB card
is what every reference run here used.

`--max-model-len`: the model's real context if it fits. If vLLM refuses with
`… KV cache is needed, which is larger than the available KV cache memory …
estimated maximum model length is N` (Qwen3-4B-2507's 262144 at 0.8), use
what the deployment needs (32768 in the reference runs), at most N. **Use the
same value for the BF16 baseline, the Step 4b sweep, Step 5 and
`launch_vllm.sh`, and report it** alongside the results.

**Embedding / pooling models (verified on EmbeddingGemma-2, v0.32.0).**
Use `--runner pooling`; 0.32 derives `embed` from the model config, so there
is no `--task` flag. Two Ada-specific workarounds are needed for encoder-only
models with wide heads:

- v0.32's Triton prefill kernel can request more shared memory than Ada has
  (101,376 B; it asked for 163,840 B at `BLOCK=128`). Upstream added a
  block-size fitting fix, but no released image contains it, so mount a
  patched `vllm/v1/attention/ops/triton_prefill_attention.py` that halves
  `BLOCK` until `element_size * BLOCK * (2 * head_dim + BLOCK)` fits.
- Marlin's FP8 kernel crashes on this model's custom linear classes
  (`QKVParallelLinear` has no `orig_dtype`). Set
  `VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel`; Humming then handles
  WFP8A16 and quality was unchanged (STS Pearson 0.8861 → 0.8855).

Verify with `/v1/embeddings` and an STS correlation, not perplexity.

**Jetson serving changes:**

- **Memory:** `--gpu-memory-utilization` is a fraction of the whole shared
  pool (122GB on a 128GB Thor), which the OS, page cache and the idle
  `modelopt` container's CPU-side memory also use. Size it from `free -g`'s
  "available" minus ~8–10GB of headroom, not from a discrete-GPU habit
  like 0.9: going over gets the server or the whole desktop OOM-killed rather
  than a clean CUDA OOM. `cudaMemGetInfo` under-reports free memory on Thor
  (99GiB vs 124GB available measured), so vLLM's own "free memory" check can
  be pessimistic; if startup says there isn't enough memory while `free -g`
  disagrees, drop page cache only with the user's OK (`sudo sh -c 'echo 3 >
  /proc/sys/vm/drop_caches'`).
- **FlashInfer needs `nvcc` in the image.** FlashInfer compiles kernels at
  runtime. Stock `v0.27.1` ships `nvcc`, and its FlashInfer FP8 and NVFP4
  GEMMs ran fine on Thor (they're even vLLM's defaults there), so nothing
  needs disabling. Jetson builds without `nvcc` (the Step 1 preflight prints
  `nvcc: None`; the `v0.28.0-aarch64` image the `quantize-nvfp4-thor` skill
  used was one) fail with `No supported CUDA architectures`, `sm75+` or
  `Invalid backend`. For those, serve with Triton attention and without
  FlashInfer's sampler and NVFP4 GEMMs, and carry the same list into Step 4b
  as `--base-disabled`:
  ```bash
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_DISABLED_KERNELS=FlashInferCutlassNvFp4LinearKernel,FlashInferCuteDslNvFp4LinearKernel,FlashInferB12xNvFp4LinearKernel,FlashInferTrtllmNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel \
  ... --attention-backend TRITON_ATTN
  ```
  These names are from vLLM 0.28; list the kernel tables in your image
  (`kernel_sweep.py`'s `LIST_TABLES` snippet) if a name isn't recognized.
- **Add** `--ulimit memlock=-1 --ulimit stack=67108864`, and mount host
  dirs for `/root/.cache/vllm`, `/root/.triton` and `/root/.cache/torch`:
  Triton and `torch.compile` caches take minutes to rebuild on Jetson's CPU,
  so a reused cache cuts restart time a lot.
- **Startup is slow** (loading a 27B checkpoint took ~66s, mostly disk read
  off a USB SSD at 0.47GB/s; plus compile and CUDA-graph capture; even a
  1.5B model took 80–200s to be ready on Thor). Raise the readiness timeout
  to 1800s, and keep checkpoints on NVMe if there is one.

**Don't pass `--quantization`.** vLLM auto-detects the method from the
checkpoint's `hf_quant_config.json` (`producer.name: modelopt`). An explicit
flag gets validated *against* that, and a mismatch — even one that sounds
right, like `awq_marlin` for an AWQ export — refuses to start with a pydantic
`ValidationError` (`Quantization method specified in the model config
(modelopt) does not match the quantization method specified in the
'quantization' argument`).

Confirm from the log, not by assuming:

- `quantization=modelopt` in the `Initializing a V1 LLM engine` config line
  (vLLM 0.32) — auto-detection worked. Older vLLM (0.27) printed `Detected
  ModelOpt fp8 checkpoint` / `Detected ModelOpt NVFP4 checkpoint` instead.
- `Selected CutlassFP8ScaledMMLinearKernel` (or another `...Kernel`) — the
  real accelerated kernel activated, not a silent BF16 fallback.
- `Using KV cache scaling factor 1.0` — only expected if the user chose an FP8
  KV cache.

**If the model has hybrid linear-attention layers** (Mamba/GatedDeltaNet —
check the config for `linear_attention` layer types), watch for this error
during CUDA graph capture:

```
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (172)...
Please lower max_num_seqs to at most 172 or increase gpu_memory_utilization.
```

These architectures need a separate per-sequence "Mamba cache block" pool,
and vLLM's default `--max-num-seqs 256` can exceed what fits. The error tells
you the exact ceiling — pass `--max-num-seqs <that number or lower>` and retry.
Don't guess upfront; it depends on model size and `--gpu-memory-utilization`.

**If the model is a reasoning/"thinking" model** (emits `<think>...</think>`
before the real answer), add `--reasoning-parser <name>` (e.g. `qwen3`) so the
API response separates the trace into a `reasoning` field.

## Step 4b: Benchmark the GEMM kernels and pick the fastest

vLLM has several matrix-multiply (GEMM) kernels per quantization format. It
tries them in a fixed order and takes the first that fits, which isn't always
the fastest on a given GPU: for FP8 on Ada, forcing a non-default kernel was
~9% faster. Always run the bundled sweep once per checkpoint, before the final
measurements in Step 5 (stop the Step 4 server first, since it uses the same
GPU and port):

```bash
python3 <skill-dir>/scripts/kernel_sweep.py --image <image> --hf-cache <host-hf-cache-dir> \
  --checkpoint-dir <host-output-dir> --name <model-name>-<qformat> \
  --gpu <N> --port <port> --parquet <host eval parquet> --datasets-cache <scratch dir> \
  --out <scratch dir>/kernel_sweep.json \
  -- --max-model-len <len> --gpu-memory-utilization <fraction> [--trust-remote-code ...]
```

The winning row of the sweep (same scripts, same settings) can be reported
as the quantized model's Step 5 number, so you don't need to measure it a
second time; you still measure BF16 in Step 5. Step 6's launch test then
confirms the `Selected …Kernel` line for the chosen kernel.

`--image` and (on x86) `--gpu` are required, in this script and in Step 6's
`make_launch_script.py`. Pass the same `<image>` as Steps 3–5 so the sweep
measures the vLLM you actually serve with, and the GPU you were given rather
than GPU 0, which on a shared machine may be someone else's.

What it does:

1. Serves the checkpoint with vLLM's defaults and reads which GEMM kernel vLLM
   picked from the log (`Selected …Kernel` / `Using …Kernel for …`).
2. Finds that kernel's candidate table inside the vLLM image
   (`vllm.model_executor.kernels.linear._POSSIBLE_*_KERNELS`) and asks each
   candidate whether it supports this GPU (`is_supported(compute_capability)`),
   so the list always matches the installed vLLM version.
3. Forces each supported candidate in turn (`VLLM_DISABLED_KERNELS` = all the
   others in the table), and records throughput with `bench.py` and
   perplexity with `ppl.py`. For a kernel that won't start, it records vLLM's
   stated reason instead.
4. Prints a table and the fastest kernel at 8 concurrent whose perplexity is
   within 1% of the default, plus the `VLLM_DISABLED_KERNELS` value to serve
   with. That value only needs to list the kernels *ahead* of the winner in
   vLLM's order.

**Jetson:** the script detects Jetson (`/etc/nv_tegra_release`) and uses
`--runtime=nvidia --gpus all` itself (`--gpu` is ignored; override with
`--gpu-flags`). Pass serving env with `--env`. If the image has no `nvcc`
(Step 4), also pass the FlashInfer list as `--base-disabled <list>`, so those
kernels stay off in every run and aren't tried. This is the command that ran
on Thor with stock `v0.27.1` (no `--base-disabled` needed there):

```bash
python3 <skill-dir>/scripts/kernel_sweep.py --image <image> \
  --checkpoint-dir <host-output-dir> --name <model-name>-<qformat> --port <port> \
  --hf-cache <host-hf-cache-dir> --client-container modelopt \
  --parquet /workspace/out/<eval parquet inside the container> --out <scratch dir>/kernel_sweep.json \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  -- --max-model-len <len> --gpu-memory-utilization <fraction> --attention-backend TRITON_ATTN [--trust-remote-code ...]
```

`--client-container modelopt` runs `bench.py`/`ppl.py` inside the `modelopt`
container (which has `requests`/`aiohttp`/`datasets` after Step 1), reaching
the server through the Docker bridge gateway, so the host's Python needs
nothing. The parquet path is then a path inside that container. Use it on
x86 too whenever the host lacks those packages.

If the image's entrypoint isn't `vllm serve` (Step 1), the sweep can't start
it as is; serve and measure by hand with Step 4/5 for each kernel instead.

It takes ~2–3 minutes per candidate (FP8 on Ada: ~6 candidates), several
times that on Jetson because of the slower load and compile. Use the
same `--max-model-len` / `--gpu-memory-utilization` as the real deployment,
and a GPU nobody else is using, or the ranking reflects contention rather
than the kernels. Kernels that also appear in vLLM's weight-only FP8 table
(e.g. Humming), and Marlin for FP4 checkpoints, are marked "weight-only
(BF16 activations)" and never recommended: they don't do the FP8/FP4 compute
the user
chose.

Then serve with the winner for Step 5 and for the user. **If the winner is
less than ~2% faster than the default at 8 concurrent, call it a tie and keep
the default** (no `VLLM_DISABLED_KERNELS`), on every platform: a sub-2% gap is
within run-to-run noise. Otherwise, e.g.:

```bash
docker run ... -e VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel ... <image> ...
```

and confirm the log's `Selected …Kernel` line names it. On Jetson the
`VLLM_DISABLED_KERNELS` it prints already includes the `--base-disabled` list.

**Reference sweeps** (vLLM 0.27.1, RTX 6000 Ada, same GPU and session within
each table):

FP8, Qwen2.5-3B (BF16 KV cache):

| Kernel | tok/s (1 / 8 concurrent) | Perplexity | Note |
|---|---|---|---|
| CUTLASS (default) | 161 / 1340 | 4.18 | |
| **PerTensorTorch** (`torch._scaled_mm`, cuBLASLt) | **177 / 1448** | 4.17 | winner: `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel` |
| Humming | 173 / 1439 | 4.14 | weight-only (BF16 activations), not recommended |
| FlashInfer | — | — | needs compute capability 10.0+ (FlashInfer *attention* still runs) |
| ChannelWiseTorch | — | — | needs per-channel scales (`fp8_pc_pt` export) |
| FP8 Marlin | — | — | weight-only fallback for GPUs without FP8; on v0.27.1 skipped unless `VLLM_TEST_FORCE_FP8_MARLIN=1`. On v0.32.0 it runs in the sweep but is flagged weight-only; don't pick it |

NVFP4, Qwen3-4B (FP8 KV cache):

| Kernel | tok/s (1 / 8 concurrent) | Perplexity | Note |
|---|---|---|---|
| **Marlin (default)** | **213 / 1601** | 5.69 | weight-only FP4, BF16 math; keep the default |
| Humming | 207 / 1565 | 5.69 | |
| Emulation | 70 / 530 | 5.86 | generic fallback, slow and less accurate |
| FlashInfer TensorRT-LLM, FlashInfer cuDNN | — | — | pass vLLM's check, then crash: `Invalid backend: 89`; `No compiled nvfp4 quantization kernel for SM 89` |
| CUTLASS, FlashInfer CUTLASS, FlashInfer CuteDsl | — | — | need compute capability 10.x |
| FBGEMM | — | — | `fbgemm_gpu required` (not in the image) |

Every kernel that does real FP4 math (FP4 activations too) needs Blackwell,
so on Ada (and Orin) NVFP4 is always weight-only. Note the TensorRT-LLM and
cuDNN rows: passing `is_supported()` doesn't guarantee a kernel runs, which is
why the sweep actually starts a server for each candidate.

NVFP4 (W4A4), Qwen2.5-1.5B-Instruct, **Jetson AGX Thor** (vLLM 0.27.1 stock
image, 120W mode, BF16 KV cache, Triton attention, wikitext-2 perplexity):

| Kernel | tok/s (1 / 8 concurrent) | Perplexity | Note |
|---|---|---|---|
| FlashInfer CUTLASS (default) | 142.7 / 1064 | 14.52 | real FP4 |
| **FlashInfer cuDNN** | **146.9 / 1071** | 14.51 | sweep's pick, but within ~1% of the default: noise level, keep the default |
| CUTLASS (vLLM) | 141.2 / 1048 | 14.51 | real FP4; the fallback when FlashInfer is unusable |
| Marlin | 138.9 / 1032 | 13.48 | weight-only (BF16 activations): better perplexity, not FP4 compute |
| Emulation | 37.8 / 294 | 14.51 | |
| FlashInfer TensorRT-LLM | — | — | `mm_fp4 does not support backend 'trtllm' with capability 110` |
| Humming | — | — | `KeyError: 110` (no sm_110 entry) |
| FlashInfer CuteDsl, FBGEMM | — | — | need sm_10x / `fbgemm_gpu` |

For `w4a16_nvfp4` there's nothing to sweep: vLLM uses Marlin for it whatever
`VLLM_DISABLED_KERNELS` says (every forced run on Thor still logged
`Using MarlinNvFp4LinearKernel`), and the script stops after the default run
when the checkpoint's `quant_algo` is `W4A16_NVFP4`.

On Thor every FP4 kernel lands within ~4%, because a small model's decode is
bandwidth-bound. When the winner is that close to the default, report it as
a tie and keep the default rather than adding a `VLLM_DISABLED_KERNELS` list.

## Step 5: Verify correctness — and avoid the one method that lies to you

**Never load a ModelOpt-exported quantized checkpoint through plain
`transformers.AutoModelForCausalLM.from_pretrained()` to check correctness.**
`transformers` doesn't recognize ModelOpt's `quant_method` and silently
*skips* quantization-aware loading — it logs something like `Hence, we will
skip the quantization` and treats the `weight_scale`/`input_scale` tensors as
unexpected keys, discarding them. The model still "loads" without error, but
produces garbage: measured perplexity in the *millions* on a checkpoint that
was actually fine, easily mistaken for a real quantization failure.

Evaluate through the vLLM server instead. Start with a generation check, and
read the answer: a broken setup can produce fluent text that ignores the
question, which a "does it produce words" check would miss:

```bash
curl -s http://localhost:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/workspace/out/<model-name>-<qformat>", "messages": [{"role": "user", "content": "<test prompt>"}], "max_tokens": 200, "temperature": 0}'
```

**VLMs: also send an image.** Text perplexity only exercises the language
model. Send one base64 `image_url` request to the BF16 and the quantized
server at `temperature 0`, with 2–3 prompts that have short, checkable
answers:

- chat VLMs: `Describe this image.` plus one-word questions such as `What
  color are the flowers? Answer in one word.` and `What is the main object?
  Answer in one word.`
- pretrained-only PaliGemma-style models: `caption en`, `answer en what color
  is the <object>?`, `answer en what is in the image?`

Pass if every short answer matches BF16 and the descriptions name the same
objects and attributes. A synonym-level difference in an open caption is not
a failure when the short answers agree (PaliGemma FP8 once captioned
"tree" where BF16 said "flower", with identical VQA answers). Run the check
against the kernel you will actually serve (Step 4b's winner), since a
different GEMM kernel can flip a near-tie token.

```bash
IMG=$(base64 -w0 <some.jpg>)
curl -s http://localhost:<port>/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "<served name>", "temperature": 0, "max_tokens": 80,
  "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}, {"type": "text", "text": "Describe this image."}]}]}'
```

Then measure with the bundled scripts. They're plain HTTP clients, so they
can run on the host:

```bash
python3 <skill-dir>/scripts/bench.py http://localhost:<port>
HF_DATASETS_CACHE=<scratch dir> python3 <skill-dir>/scripts/ppl.py --url http://localhost:<port> \
  --parquet <host path to a cached eval parquet, e.g. ultrachat_200k test_sft> --size 64
```

- `bench.py` prints single-request and 8-concurrent throughput (256 tokens,
  `ignore_eos`). Single-request timing alone understates what batching gets.
- `ppl.py` scores prompt tokens via `echo` + `logprobs` on `/v1/completions`,
  so it measures the real serving path, kernels included.
- Both need `requests`, `datasets` and `aiohttp`. If the host's Python doesn't
  have them, don't pip-install into the system Python. Run them in the
  `modelopt` container, which has all three after Step 1 (the bare vLLM image
  has no `pyarrow`/`datasets`, so `ppl.py` can't run there), and reach the
  server through the Docker bridge gateway:
  ```bash
  docker cp <skill-dir>/scripts/bench.py modelopt:/workspace/bench.py; docker cp <skill-dir>/scripts/ppl.py modelopt:/workspace/ppl.py
  GW=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
  docker exec modelopt python3 /workspace/bench.py http://$GW:<port>
  docker exec -e HF_DATASETS_CACHE=/tmp/dscache modelopt python3 /workspace/ppl.py --url http://$GW:<port> --parquet /workspace/out/<eval parquet> --size 64
  ```
  The eval parquet has to be under a mount of that container (e.g. copy it
  into `<host-output-dir>/data/`). Step 4b does the same with
  `--client-container modelopt`. A host venv also works, but it means
  downloading these packages a second time: on a slow PyPI day that took
  over 30 minutes on the Jetson (the first attempt failed). Point
  `HF_DATASETS_CACHE` at a scratch directory: the shared HF datasets cache can
  hold root-owned lock files from containers (`PermissionError` on a `.lock`).
  Pass the parquet path rather than a dataset name, so nothing touches the Hub.

**Always measure the BF16 original the same way, in the same session** (same
GPU, same memory settings, same samples, same image). **Jetson:** also the
same power mode. Clocks depend on `nvpmodel` (and on whether `jetson_clocks`
is running), and a lower mode or thermal throttling moves throughput far
more than the kernel choice does. Record `nvpmodel -q` with the results.
Changing the mode needs sudo and can reboot the device, so ask the user
rather than doing it. BF16 of a big model may not fit next to the OS in the
shared pool at a useful `--max-model-len`; if so, say so and compare at a
shorter length that fits both. Absolute throughput
moved ~20% between days depending on what else was running on the GPU, so
numbers from different sessions aren't comparable.

**Pass bar:** perplexity within about +2% of BF16 for FP8 and weight-only
4-bit (measured: Qwen3-4B −0.2%, Cosmos-Reason2-2B +1.1%, PaliGemma-3B +1.0%,
Qwen2.5-3B FP8 +1.9%), and, for a VLM, matching image answers. Beyond that,
investigate before calling it done (KV-cache precision is the usual cause,
Step 3). W4A4 `nvfp4` can legitimately cost more; report the number and let
the user judge.

## Reference results (Qwen2.5-3B-Instruct, 1x RTX 6000 Ada, uncontended GPU)

| Variant | Perplexity | vLLM tok/s (1 / 8 concurrent) |
|---|---|---|
| BF16 | 4.10 | 115 / 899 |
| FP8, BF16 KV cache, CUTLASS kernel | 4.18 | 161 / 1340 |
| FP8, BF16 KV cache, PerTensorTorch kernel | 4.18 | 177 / 1448 |
| FP8 + FP8 KV cache | 4.46 | (measured on a shared GPU only) |
| Same FP8 checkpoint as a TensorRT engine (`/totrt`) | — | 193 / 1395 |

Qwen3-4B-Instruct-2507, same GPU and session, max length 32K:

| Variant | Size | Weights in GPU | Perplexity | vLLM tok/s (1 / 8 concurrent) |
|---|---|---|---|---|
| BF16 | 7.6 GB | 8.6 GiB | 5.41 | 95 / 726 |
| NVFP4 (Marlin, weight-only on Ada) + FP8 KV cache | 2.7 GB | 3.1 GiB | 5.69 (+5.2%) | 208 / 1578 |

Qwen3.8-27B (hybrid-attention VLM) on Jetson AGX Thor, from the
`quantize-nvfp4-thor` run (ModelOpt 0.46, vLLM 0.28, gsm8k 5-shot strict):

| Variant | Size | gsm8k | Notes |
|---|---|---|---|
| BF16 | 52 GB | 92.6% | |
| `w4a16_nvfp4` (Marlin) | 26 GB | 92.6% | BF16 KV cache |
| `nvfp4` W4A4 (CUTLASS FP4), 512 calib samples | 26 GB | 84.2% | FP8 KV cache, ~2x KV headroom |

Qwen2.5-1.5B-Instruct, **Jetson AGX Thor** (stock `vllm/vllm-openai:v0.27.1`,
120W mode, BF16 KV cache, Triton attention, max length 4096, 512 wikitext-2
calibration samples, perplexity on 64 wikitext-2 test chunks; verified
end to end 2026-10-27):

| Variant | Size | Perplexity | vLLM tok/s (1 / 8 concurrent) | Kernel |
|---|---|---|---|---|
| BF16 | 3.1 GB | 12.49 | 71 / 537 | |
| `fp8` | 1.7 GB | 12.68 (+1.5%) | 111 / 824 | FlashInferFP8ScaledMM (default) |
| `w4a16_nvfp4` (weight-only FP4) | 1.2 GB | 13.49 (+8%) | 140 / 1029 | Marlin (the only one vLLM uses for it) |
| `nvfp4` (W4A4, real FP4) | 1.2 GB | 14.51 (+16%) | 142 / 1064 | FlashInferCutlassNvFp4 (default) |

On Thor `w4a16_nvfp4` is about as fast as W4A4 (decode is bandwidth-bound,
and both read the same 4-bit weights) at half the accuracy cost. So it's the
right default, and W4A4's real FP4 math only pays off when prefill or large
batches make the GEMMs compute-bound. `w4a16_nvfp4` needs no calibration data
(its scales come from the weights; the export has no `input_scale`), while
`fp8` and `nvfp4` calibrate their activation scales on `--dataset`.

Each `hf_ptq.py` run took ~3 minutes on Thor, including model load.

An FP8 export that isn't clearly faster than BF16 on the same GPU, or whose
perplexity is far off BF16, is worth investigating before reporting.

## Comparing precisions on the same model

When a user wants a BF16/FP8/INT4 comparison table (not just "quantize this
once"), the comparison is only fair per-GPU, not per-replica: if BF16 needs
tensor parallelism across N GPUs to fit while the quantized version fits on
one, a raw total-throughput comparison favors BF16 unfairly (it's using more
hardware). Normalize to throughput-per-GPU when that's the case, and say so
explicitly — it's the difference between "quantization made it faster" (often
not quite true) and "quantization let one GPU do what used to need two"
(usually the actual win).

## Step 6: Leave a launch script with each checkpoint

Always finish by writing `launch_vllm.sh` into every export folder, so the user
can serve it later without rebuilding the `docker run` command. Pass it the
settings that were *verified* in Steps 4–5 (image, memory, max length, any
`VLLM_DISABLED_KERNELS` from Step 4b, the Jetson env/attention flags if they
were used), not new guesses:

```bash
python3 <skill-dir>/scripts/make_launch_script.py \
  --export-dir <host-output-dir>/<model-name>-<qformat> \
  --image <image> --port <free port from Step 4> --max-model-len <len> --gpu-memory-utilization <fraction> \
  --gpu <N> [--disabled-kernels <Step 4b value>] [--env KEY=VAL ...] [--entrypoint-override] \
  -- [--attention-backend TRITON_ATTN --trust-remote-code --reasoning-parser qwen3 --max-num-seqs <n> ...]
```

- The platform is detected the same way `kernel_sweep.py` does it: Jetson
  gets `--runtime=nvidia --gpus all` plus the memlock/stack ulimits and an
  1800s readiness timeout; x86 gets `--gpus '"device=<N>"'`.
- The script mounts *its own folder* read-only as the model, so it keeps
  working if the checkpoint is moved or copied to another machine of the same
  kind. Compile caches go to `<export-dir>/../.vllm-cache` (`--cache-dir` to
  change), which makes restarts much faster.
- It's a plain bash script the user can read and edit. Usage:
  `./launch_vllm.sh` (start, wait until ready, print the detected format,
  the GEMM kernel and a sample `curl`), `stop`, `logs`, `status`. You can
  override settings per run without editing: `PORT=8001 GPU_MEM=0.4
  MAX_LEN=8192 ./launch_vllm.sh`. It refuses to start if its container is
  already running or the port is taken, and it never uses `--rm`.
- `--entrypoint-override` is for images whose entrypoint isn't `vllm serve`
  (Step 1). `--quantization` is rejected (Step 4).
- Test it once: run it, send one request, run `./launch_vllm.sh stop`. On Thor
  this worked for the `w4a16_nvfp4` reference checkpoint (ready in 101s).

## Finishing

Report the export path on the host (after Step 3c, the serving artifact is the
converted `-ct` dir — report both), the `launch_vllm.sh` path (one line on how
to start and stop it), the comparison table against BF16, which
vLLM kernel to use, the platform (and on Jetson the image and power mode), and
anything that couldn't be verified. If the user wants
a TensorRT engine, the next step is `/totrt <host-output-dir>/<model-name>-<qformat>`.
Leave the `modelopt` container running (idle) for next time and stop the
serving container (`docker stop vllm-serve`, or `./launch_vllm.sh stop`); don't
remove it, per the no-`--rm` rule above.
