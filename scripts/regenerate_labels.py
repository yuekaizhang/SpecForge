#!/usr/bin/env python3
"""Regenerate training labels using the TARGET model's own greedy generation.

Optimized: multiprocess audio encoding (CPU) pipelined with async HTTP sending (GPU).
Supports multiple sglang servers for multi-GPU parallelism.

Usage:
    # Launch 8 servers first (one per GPU), then:
    python scripts/regenerate_labels.py \
        --server http://127.0.0.1:31020,...,http://127.0.0.1:31027 \
        --num-samples 0 --workers 8 --concurrency 128 \
        --out regen_labels/regen_labels.jsonl
"""
import argparse
import asyncio
import base64
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import soundfile as sf

# --- Worker function for multiprocess audio encoding (runs in separate process) ---
def encode_clip(audio_bytes):
    """Decode raw audio bytes → 16kHz mono WAV → base64 string. CPU-only."""
    arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        n = int(round(len(arr) / sr * 16000))
        arr = np.interp(
            np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr
        ).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, 16000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- Async HTTP sending ---
async def send_one(session, url, model, prompt, audio_b64, sem):
    async with sem:
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        }
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                return f"ERROR: {json.dumps(data)[:200]}"
        except Exception as e:
            return f"ERROR: {e}"


async def send_batch(server_urls, model, prompt, b64_list, concurrency):
    import aiohttp
    urls = [f"{s}/v1/chat/completions" for s in server_urls]
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 16)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = [
            send_one(session, urls[i % len(urls)], model, prompt, b64, sem)
            for i, b64 in enumerate(b64_list)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:31020",
                    help="Comma-separated sglang server URLs")
    ap.add_argument("--model", default="qwen2audio")
    ap.add_argument("--dataset", default="carlot/AIShell")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-samples", type=int, default=0, help="0=all")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel workers for audio encoding (CPU)")
    ap.add_argument("--concurrency", type=int, default=128,
                    help="Max concurrent HTTP requests to servers")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    servers = [s.strip() for s in args.server.split(",")]
    print(f"Servers: {len(servers)}, workers: {args.workers}, concurrency: {args.concurrency}")
    instruction = "请将这段音频转写为文本。"

    # Load dataset (non-streaming for random access + known length)
    from datasets import load_dataset
    from datasets.features import Audio
    print(f"Loading {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    total = len(ds) if args.num_samples <= 0 else min(args.num_samples, len(ds))
    print(f"Total clips to process: {total}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f_out = open(args.out, "w", buffering=1)  # line-buffered!

    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
    t0 = time.time()
    processed = 0
    match_count = 0
    cs = args.chunk_size

    for chunk_start in range(0, total, cs):
        chunk_end = min(chunk_start + cs, total)
        chunk_indices = list(range(chunk_start, chunk_end))

        # --- Phase 1: parallel audio encoding (CPU, multiprocess) ---
        futures = {}
        for idx in chunk_indices:
            audio_bytes = ds[idx]["audio"]["bytes"]
            futures[pool.submit(encode_clip, audio_bytes)] = idx

        b64_map = {}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                b64_map[idx] = fut.result()
            except Exception as e:
                b64_map[idx] = None
                print(f"  encode error idx={idx}: {e}")

        b64_ordered = [b64_map[i] for i in chunk_indices]
        gt_ordered = ["".join(ds[i]["transcription"].split()) for i in chunk_indices]

        # Skip clips with encoding errors
        valid = [(i, b, g) for i, b, g in zip(chunk_indices, b64_ordered, gt_ordered) if b is not None]
        if not valid:
            continue
        v_indices, v_b64, v_gt = zip(*valid)

        # --- Phase 2: async HTTP send to servers (GPU) ---
        results = asyncio.run(send_batch(servers, args.model, instruction, list(v_b64), args.concurrency))

        # --- Phase 3: write results (line-buffered, immediate flush) ---
        for idx, gt, gen in zip(v_indices, v_gt, results):
            if isinstance(gen, Exception):
                gen = f"ERROR: {gen}"
            if gt == gen:
                match_count += 1
            f_out.write(json.dumps({"idx": idx, "gt": gt, "target_gen": gen}, ensure_ascii=False) + "\n")

        processed += len(valid)
        elapsed = time.time() - t0
        rate = processed / elapsed
        eta = (total - processed) / rate if rate > 0 else 0
        print(f"  [{processed}/{total}] {rate:.1f} clips/s, "
              f"match={match_count}/{processed} ({match_count/max(processed,1):.0%}), "
              f"ETA {eta/60:.1f}min", flush=True)

    pool.shutdown()
    f_out.close()
    elapsed = time.time() - t0
    print(f"\nDone: {processed} clips in {elapsed/60:.1f}min ({processed/elapsed:.1f} clips/s)")
    print(f"Exact match: {match_count}/{processed} = {match_count/max(processed,1):.0%}")
    print(f"Saved to: {args.out}")

    # Show first 5
    with open(args.out) as f:
        for j, line in enumerate(f):
            if j >= 5: break
            d = json.loads(line)
            m = "✓" if d["gt"] == d["target_gen"] else "✗"
            print(f"  [{m}] GT:  {d['gt']}")
            print(f"       GEN: {d['target_gen']}")


if __name__ == "__main__":
    main()
