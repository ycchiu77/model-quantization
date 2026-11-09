#!/usr/bin/env python3
"""Perplexity through a running vLLM server, using prompt-token logprobs (echo=True).

Scores the real serving path (quantized kernels included). TensorRT-LLM's server
rejects `logprobs`, so point this at vLLM.

    python3 ppl.py --url http://localhost:8000 --parquet <cached ultrachat test_sft .parquet> --size 64
"""
import argparse
import math

import requests
from datasets import load_dataset


def nll_for_text(url, model, text):
    r = requests.post(f"{url}/v1/completions", timeout=120, json={
        "model": model, "prompt": text[:2000], "max_tokens": 1,
        "echo": True, "logprobs": 1, "temperature": 0,
    })
    r.raise_for_status()
    lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    # first entry has no context (None); last is the generated token, not part of the prompt
    lps = [lp for lp in lps[:-1] if lp is not None]
    return -sum(lps), len(lps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--parquet", required=True, help="local parquet with a 'messages' (chat) or 'text' column")
    p.add_argument("--size", type=int, default=64)
    args = p.parse_args()

    model = requests.get(f"{args.url}/v1/models", timeout=30).json()["data"][0]["id"]
    ds = load_dataset("parquet", data_files={"eval": args.parquet}, split="eval").select(range(args.size))
    if "messages" in ds.column_names:
        texts = [" ".join(m["content"] for m in row["messages"]) for row in ds]
    else:
        texts = ds["text"]

    total_nll, total_tok = 0.0, 0
    for t in texts:
        nll, n = nll_for_text(args.url, model, t)
        total_nll += nll
        total_tok += n
    print(f"Perplexity ({model}): {math.exp(total_nll / total_tok):.4f} over {total_tok} tokens, {len(texts)} samples")


if __name__ == "__main__":
    main()
