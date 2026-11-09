#!/usr/bin/env python3
"""Convert a ModelOpt INT4 AWQ (W4A16_AWQ) export into a compressed-tensors W4A16
("pack-quantized") checkpoint that vLLM's compressed-tensors loader can serve.

vLLM's ModelOpt loader rejects W4A16_AWQ, but its compressed-tensors WNa16 path (Marlin
dequant-on-the-fly, A16) takes the same math. Two format differences are handled here:

1. Packing axis. ModelOpt packs the signed 4-bit values along the OUTPUT dim
   (uint8 [out/2, in], row 2i = low nibble, row 2i+1 = high nibble; see
   modelopt/torch/export/quant_utils.py:pack_int4_in_uint8). compressed-tensors packs
   along the INPUT dim (int32 [out, in/8], column j of the weight -> nibble j%8 of word
   j//8, two's complement). So: unpack, requantize, repack.
2. AWQ pre_quant_scale. AWQ stores W*diag(s)^-1 and expects the activation multiplied by
   s at runtime; vLLM has no per-linear pre-quant scale (and siblings q/k/v, gate/up share
   one norm, so it cannot be folded into the norm). It is folded into the weight instead
   (W' = dequant(W) * pre_quant_scale) and requantized per group — a double quantization,
   so always run the Step 5 perplexity check against BF16 before shipping.

    python3 int4_awq_to_compressed_tensors.py <modelopt int4_awq export dir> <new output dir>

The output dir must not exist. Run inside the modelopt container (needs CUDA for speed).
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Kept unquantized by vLLM; merged with the export's own exclude_modules.
ALWAYS_IGNORE = ["re:.*visual.*", "re:.*lm_head.*", "re:.*embed_tokens.*",
                 "re:.*conv1d.*", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*in_proj_ba$"]


def unpack_modelopt(wu8):
    """uint8 [out/2, in] (out-dim pairs, low nibble = row 2i) -> signed int16 [out, in]."""
    lo = (wu8 & 0xF).to(torch.int16)
    hi = (wu8 >> 4).to(torch.int16)
    w = torch.empty((wu8.shape[0] * 2, wu8.shape[1]), dtype=torch.int16, device=wu8.device)
    w[0::2] = torch.where(lo > 7, lo - 16, lo)
    w[1::2] = torch.where(hi > 7, hi - 16, hi)
    return w


def requantize(q, w_scale, pqs, group):
    """ModelOpt ints -> effective FP weights (fold AWQ smoothing) -> CT symmetric ints."""
    w = unpack_modelopt(q).cuda().float()
    w *= w_scale.cuda().float().repeat_interleave(group, dim=1)
    w *= pqs.cuda().float().reshape(1, -1)  # de-smooth: stored W is W*diag(s)^-1
    out, k = w.shape
    wg = w.reshape(out, k // group, group)
    # compressed-tensors symmetric scale: max(max/7, -min/8) per (row, group)
    scale = torch.maximum(wg.amax(2) / 7.0, (-wg.amin(2)) / 8.0).clamp(min=1e-12)
    q_new = torch.round(wg / scale.unsqueeze(2)).clamp(-8, 7).to(torch.int32)
    return q_new.reshape(out, k), scale.to(torch.float16).cpu()


def pack_ct(q_int32):
    """signed int32 [out, in] in [-8,7] -> CT weight_packed int32 [out, in/8]."""
    k = q_int32.shape[1]
    assert k % 8 == 0, f"input dim {k} not divisible by 8; pad or keep group packing aligned"
    q = q_int32.cuda() & 0xF  # two's complement nibbles
    res = torch.zeros((q.shape[0], k // 8), dtype=torch.int32, device=q.device)
    for i in range(8):
        res |= q[:, i::8] << (4 * i)
    return res.cpu()


def glob_to_re(glob):
    return "re:^" + re.escape(glob).replace(r"\*", ".*") + "$"


def main(src, dst):
    src, dst = Path(src), Path(dst)
    q = json.loads((src / "hf_quant_config.json").read_text())["quantization"]
    if q.get("quant_algo") != "W4A16_AWQ":
        sys.exit(f"expected quant_algo W4A16_AWQ, got {q.get('quant_algo')}")
    if q.get("has_zero_point"):
        sys.exit("asymmetric AWQ (has_zero_point) not supported by this converter")
    group = int(q.get("group_size", 128))
    dst.mkdir(parents=True, exist_ok=False)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    new_map, n_q = {}, 0
    for shard in sorted(set(index["weight_map"].values())):
        t = load_file(src / shard)
        out = {}
        for k, v in t.items():
            if k.endswith(".weight") and v.dtype == torch.uint8:
                base = k[: -len(".weight")]
                w_scale = t[base + ".weight_scale"]
                if w_scale is None or w_scale.shape[0] != v.shape[0] * 2:
                    sys.exit(f"{base}: expected weight_scale [out, in/{group}] with out={v.shape[0] * 2}, "
                             f"weight uint8 [out/2, in] — is this really a W4A16_AWQ export?")
                pqs = t.get(base + ".pre_quant_scale")
                if pqs is None:
                    pqs = torch.ones(v.shape[1])  # already folded (e.g. into the preceding norm)
                q_new, scale = requantize(v, w_scale, pqs, group)
                out[k[: -len(".weight")] + ".weight_packed"] = pack_ct(q_new)
                out[base + ".weight_scale"] = scale
                out[base + ".weight_shape"] = torch.tensor(
                    [v.shape[0] * 2, v.shape[1]], dtype=torch.int32)
                n_q += 1
            elif k.endswith((".weight_scale", ".input_scale", ".pre_quant_scale")):
                continue  # replaced above / dropped
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
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
                        "dynamic": False, "group_size": group, "actorder": None, "observer": "minmax"},
            "input_activations": None,  # A16: activations stay BF16, Marlin dequantizes weights
            "output_activations": None,
        }},
        "ignore": sorted(set(ignore)),
        "kv_cache_scheme": None,
        "source": "converted from a ModelOpt W4A16_AWQ export (smoothing folded into weights, "
                  "output-dim packing repacked to input-dim)",
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    skip = {"config.json", "model.safetensors.index.json", "hf_quant_config.json"}
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in skip:
            (dst / f.name).write_bytes(f.read_bytes())
    print(f"repacked {n_q} INT4 weights -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-2])
    main(*sys.argv[1:3])
