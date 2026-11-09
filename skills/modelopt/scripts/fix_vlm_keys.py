#!/usr/bin/env python3
"""Re-insert a nesting level that a ModelOpt export dropped from a VLM's vision tower.

transformers 5 can save some vision towers one level flatter than vLLM expects.
PaliGemma (SigLIP) exports as   model.vision_tower.<x>
while vLLM 0.32 loads           model.vision_tower.vision_model.<x>
and fails with "There is no module or parameter named 'vision_tower.embeddings'".

Run inside the modelopt container (it has safetensors + torch):

  python3 fix_vlm_keys.py /workspace/out/<model-name>-<qformat> --dry-run
  python3 fix_vlm_keys.py /workspace/out/<model-name>-<qformat>

Defaults fit PaliGemma; pass --prefix/--insert for another tower. Keys that
already carry the inserted level are left alone, so running it twice is safe.
"""

import argparse
import glob
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("export_dir")
    p.add_argument("--prefix", default="model.vision_tower.", help="keys under this prefix get the level inserted")
    p.add_argument("--insert", default="vision_model.", help="level to insert right after --prefix")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    target = a.prefix + a.insert
    files = sorted(glob.glob(os.path.join(a.export_dir, "*.safetensors")))
    if not files:
        raise SystemExit(f"no *.safetensors in {a.export_dir}")

    total = 0
    renames = {}
    for f in files:
        with safe_open(f, "pt") as st:
            keys = list(st.keys())
        todo = [k for k in keys if k.startswith(a.prefix) and not k.startswith(target)]
        print(f"{os.path.basename(f)}: {len(todo)} keys to rename" + (f", e.g. {todo[0]} -> {target}{todo[0][len(a.prefix):]}" if todo else ""))
        if not todo or a.dry_run:
            total += len(todo)
            continue
        with safe_open(f, "pt") as st:
            meta = st.metadata()
            tensors = {k: st.get_tensor(k) for k in keys}
        out = {}
        for k, v in tensors.items():
            if k in todo:
                nk = target + k[len(a.prefix):]
                renames[k] = nk
                k = nk
            out[k] = v
        st_before = os.stat(f)
        save_file(out, f, metadata=meta)
        # save_file writes a new root-owned 0600 file; keep the export readable to the host user
        os.chown(f, st_before.st_uid, st_before.st_gid)
        os.chmod(f, st_before.st_mode & 0o7777)
        total += len(todo)

    # sharded exports: keep the index in step with the renamed keys
    idx = os.path.join(a.export_dir, "model.safetensors.index.json")
    if renames and os.path.exists(idx):
        j = json.load(open(idx))
        j["weight_map"] = {renames.get(k, k): v for k, v in j["weight_map"].items()}
        json.dump(j, open(idx, "w"), indent=2)

    print(("would rename" if a.dry_run else "renamed"), total, "keys")


if __name__ == "__main__":
    main()
