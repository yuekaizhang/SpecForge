#!/usr/bin/env python3
"""Regenerate training labels using the TARGET model's own greedy generation.

Uses a running sglang server — launch it first:
    python -m sglang.launch_server --model-path Qwen/Qwen2-Audio-7B-Instruct \
        --served-model-name qwen2audio --trust-remote-code --dp-size 8 --port 31020

Then run this script:
    python scripts/regenerate_labels.py --server http://127.0.0.1:31020 \
        --num-samples 0 --out regen_labels.jsonl

No wav files written — audio is sent as inline base64 in each request.
Output is a JSONL with {idx, gt, target_gen} for every clip.
"""
import argparse
import asyncio
import base64
import io
import json
import os
import time

import aiohttp
import numpy as np
import soundfile as sf
from datasets import load_dataset
from datasets.features import Audio


async def send_request(session, url, model, prompt, audio_b64, semaphore):
    """Send one chat/completions request with inline base64 audio."""
    async with semaphore:
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 128,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            else:
                return f"ERROR: {json.dumps(data)[:200]}"


async def regenerate_batch(server_urls, model, prompt, clips, concurrency):
    """Send all clips concurrently, round-robin across multiple servers."""
    urls = [f"{s}/v1/chat/completions" for s in server_urls]
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 8)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i, clip in enumerate(clips):
            url = urls[i % len(urls)]  # round-robin across servers
            tasks.append(send_request(session, url, model, prompt, clip["audio_b64"], sem))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for r in results:
        if isinstance(r, Exception):
            out.append(f"ERROR: {r}")
        else:
            out.append(r)
    return out


def audio_bytes_to_wav_b64(audio_bytes):
    """Decode raw audio bytes (any format) → 16kHz mono wav → base64 string."""
    arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        n = int(round(len(arr) / sr * 16000))
        arr = np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, 16000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:31020",
                    help="Running sglang server URL(s), comma-separated for multi-GPU")
    ap.add_argument("--model", default="qwen2audio")
    ap.add_argument("--dataset", default="carlot/AIShell")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="Max concurrent requests (server batches internally)")
    ap.add_argument("--chunk-size", type=int, default=500,
                    help="Process in chunks (for progress + memory)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    servers = [s.strip() for s in args.server.split(",")]
    print(f"Using {len(servers)} server(s): {servers}")
    instruction = "请将这段音频转写为文本。"

    # === Load dataset (streaming, raw bytes) ===
    print(f"Loading {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    # === Process in chunks ===
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f_out = open(args.out, "w")
    total_processed = 0
    total_match = 0
    t0 = time.time()
    chunk = []
    chunk_start_idx = 0

    for i, ex in enumerate(ds):
        if args.num_samples > 0 and i >= args.num_samples:
            break

        gt = "".join(ex["transcription"].split())  # space-stripped
        audio_b64 = audio_bytes_to_wav_b64(ex["audio"]["bytes"])
        chunk.append({"idx": i, "gt": gt, "audio_b64": audio_b64})

        if len(chunk) >= args.chunk_size:
            # Process chunk
            print(f"Generating chunk [{chunk_start_idx}..{chunk_start_idx+len(chunk)-1}] "
                  f"({len(chunk)} clips, concurrency={args.concurrency})...", flush=True)
            gens = asyncio.run(regenerate_batch(
                servers, args.model, instruction, chunk, args.concurrency
            ))
            for c, gen in zip(chunk, gens):
                match = (c["gt"] == gen)
                if match:
                    total_match += 1
                f_out.write(json.dumps({
                    "idx": c["idx"], "gt": c["gt"], "target_gen": gen
                }, ensure_ascii=False) + "\n")
            total_processed += len(chunk)
            elapsed = time.time() - t0
            rate = total_processed / elapsed
            print(f"  [{total_processed} done] {rate:.1f} clips/s, "
                  f"exact_match={total_match}/{total_processed} ({total_match/total_processed:.0%})",
                  flush=True)
            chunk = []
            chunk_start_idx = total_processed

    # Final chunk
    if chunk:
        print(f"Generating final chunk [{chunk_start_idx}..{chunk_start_idx+len(chunk)-1}]...", flush=True)
        gens = asyncio.run(regenerate_batch(
            args.server, args.model, instruction, chunk, args.concurrency
        ))
        for c, gen in zip(chunk, gens):
            match = (c["gt"] == gen)
            if match:
                total_match += 1
            f_out.write(json.dumps({
                "idx": c["idx"], "gt": c["gt"], "target_gen": gen
            }, ensure_ascii=False) + "\n")
        total_processed += len(chunk)

    f_out.close()
    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"Total: {total_processed} clips in {elapsed/60:.1f} min ({total_processed/elapsed:.1f} clips/s)")
    print(f"Exact match (GT == target gen): {total_match}/{total_processed} = {total_match/total_processed:.0%}")
    print(f"Saved to: {args.out}")
    print(f"\nFirst 5 examples:")
    with open(args.out) as f:
        for j, line in enumerate(f):
            if j >= 5:
                break
            d = json.loads(line)
            m = "✓" if d["gt"] == d["target_gen"] else "✗"
            print(f"  [{m}] GT:  {d['gt']}")
            print(f"       GEN: {d['target_gen']}")


if __name__ == "__main__":
    main()
