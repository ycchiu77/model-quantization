# model-quantization

Claude Code skills for quantizing LLM / VLM checkpoints.

| skill | what it does |
|---|---|
| [`modelopt`](skills/modelopt/SKILL.md) | Quantize a checkpoint with NVIDIA Model Optimizer (`hf_ptq.py`) and serve it with vLLM: lists the precisions the GPU actually accelerates (BF16 / FP8 / INT8-SQ / INT4-AWQ / NVFP4), runs the PTQ offline in a `vllm/vllm-openai` container, and checks speed and accuracy against BF16. x86-64 discrete GPUs and Jetson (AGX Thor / Orin). |

## Install
Copy the skill into your Claude Code skills folder:
```bash
git clone git@github.com:ycchiusieve/model-quantization.git
cp -r model-quantization/skills/modelopt ~/.claude/skills/        # all projects
# or: cp -r model-quantization/skills/modelopt <project>/.claude/skills/   # one project
```
Then in Claude Code ask e.g. "quantize this model to FP8" or run `/modelopt`.

## Layout
```
skills/modelopt/
  SKILL.md                              instructions Claude follows
  scripts/
    make_launch_script.py               writes the vLLM serve script for a quantized checkpoint
    bench.py, ppl.py                    throughput and perplexity through a running vLLM server
    kernel_sweep.py                     benchmarks every vLLM GEMM kernel that can run the checkpoint on this GPU
    int4_awq_to_compressed_tensors.py   ModelOpt INT4-AWQ  -> compressed-tensors (for vLLM)
    int8_sq_to_compressed_tensors.py    ModelOpt INT8-SQ   -> compressed-tensors (for vLLM)
    fix_vlm_keys.py                     re-inserts the vision-tower nesting level a VLM export drops
```
