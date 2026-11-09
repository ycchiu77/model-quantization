import asyncio, sys, time
import aiohttp

URL = sys.argv[1]
PROMPT = "Write a detailed explanation of how transformers process text, covering tokenization, attention, and feed-forward layers."
MAX_TOKENS = 256


async def one(session, model):
    t = time.perf_counter()
    async with session.post(f"{URL}/v1/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS, "temperature": 0, "ignore_eos": True,
    }) as r:
        d = await r.json()
    return d["usage"]["completion_tokens"], time.perf_counter() - t


async def main():
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{URL}/v1/models") as r:
            model = (await r.json())["data"][0]["id"]
        await one(s, model)  # warmup
        toks, dt = await one(s, model)
        print(f"single: {toks / dt:.1f} tok/s ({dt:.2f}s for {toks} tok)")
        t = time.perf_counter()
        res = await asyncio.gather(*[one(s, model) for _ in range(8)])
        dt = time.perf_counter() - t
        print(f"8-concurrent: {sum(r[0] for r in res) / dt:.1f} tok/s aggregate")


asyncio.run(main())
