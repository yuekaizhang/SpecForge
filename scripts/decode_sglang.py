#!/usr/bin/env python3
"""ASR decode benchmark via sglang server, with/without EAGLE3 speculative decoding.

Decodes AISHELL test set, measures latency/RTF/CER, and compares
baseline (no draft) vs speculative (with draft) performance.

Outputs:
  <output_dir>/
    results.jsonl      — per-utterance: {idx, gt, hyp, cer, audio_dur_s, latency_s, rtf}
    errors.txt         — utterances where hyp != gt (with CER)
    rtf.txt            — RTF statistics (mean, p50, p90, p99)
    summary.txt        — overall CER, RTF, latency, accept_length

Usage:
    # Start server first (with or without --speculative-algorithm EAGLE3)
    # Then:
    python scripts/decode_sglang.py \
        --server http://127.0.0.1:31040 \
        --dataset carlot/AIShell --split test \
        --prompt "Detect the language and recognize the speech: <|zh|>" \
        --output-dir results/sft_eagle3_test \
        --concurrency 16 --num-samples 0
"""
import argparse
import asyncio
import base64
import io
import json
import os
import time
from typing import List, Tuple

import aiohttp
import numpy as np
import soundfile as sf


# ===================== CER (Character Error Rate) =====================
def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Levenshtein edit distance between two character lists."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[m]


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate for Chinese (character-level)."""
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float("inf")
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ===================== Audio encoding =====================
def encode_audio_b64(audio_bytes: bytes) -> Tuple[str, float]:
    """Decode raw audio bytes → 16kHz mono WAV base64 + duration in seconds."""
    arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        n = int(round(len(arr) / sr * 16000))
        arr = np.interp(
            np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr
        ).astype(np.float32)
        sr = 16000
    duration_s = len(arr) / sr
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, duration_s


# ===================== Async HTTP =====================
async def decode_one(session, url, model, prompt, audio_b64, sem):
    """Send one request, return (generated_text, latency_s)."""
    async with sem:
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        t0 = time.monotonic()
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                latency = time.monotonic() - t0
                if "choices" in data:
                    return data["choices"][0]["message"]["content"], latency
                return f"ERROR: {json.dumps(data)[:200]}", latency
        except Exception as e:
            return f"ERROR: {e}", time.monotonic() - t0


async def decode_batch(server_url, model, prompt, clips, concurrency):
    """Decode a batch of clips. Returns list of (text, latency_s)."""
    url = f"{server_url}/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = [
            decode_one(session, url, model, prompt, clip["b64"], sem)
            for clip in clips
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="ASR decode benchmark via sglang")
    ap.add_argument("--server", default="http://127.0.0.1:31040")
    ap.add_argument("--model", default="qwen2audio-sft")
    ap.add_argument("--dataset", default="carlot/AIShell")
    ap.add_argument("--subset", default=None, help="Dataset config/subset name")
    ap.add_argument("--split", default="test")
    ap.add_argument("--text-column", default="transcription", help="Column with GT text")
    ap.add_argument(
        "--prompt",
        default="Detect the language and recognize the speech: <|zh|>",
    )
    ap.add_argument("--num-samples", type=int, default=0, help="0=all")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for audio encoding",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load dataset ----
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    from datasets import load_dataset
    from datasets.features import Audio

    print(f"Loading {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, args.subset, split=args.split) if args.subset else load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    total = len(ds) if args.num_samples <= 0 else min(args.num_samples, len(ds))
    print(f"Total: {total} utterances, concurrency={args.concurrency}")

    # ---- Check server health + get spec info ----
    import requests

    assert requests.get(f"{args.server}/health", timeout=10).ok, "Server not healthy"
    server_info = requests.get(f"{args.server}/get_server_info", timeout=10).json()
    spec_algo = server_info.get("speculative_algorithm", "none")
    draft_path = server_info.get("speculative_draft_model_path", "none")
    print(f"Server: spec_algo={spec_algo}, draft={draft_path}")

    # ---- Process ----
    results_path = os.path.join(args.output_dir, "results.jsonl")
    f_results = open(results_path, "w", buffering=1)

    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    )
    all_results = []
    t_total_start = time.time()
    processed = 0
    cs = args.chunk_size

    for chunk_start in range(0, total, cs):
        chunk_end = min(chunk_start + cs, total)
        chunk_indices = list(range(chunk_start, chunk_end))

        # Phase 1: parallel audio encoding
        futures = {}
        for idx in chunk_indices:
            futures[pool.submit(encode_audio_b64, ds[idx]["audio"]["bytes"])] = idx
        clips = {}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                b64, dur = fut.result()
                clips[idx] = {"b64": b64, "dur": dur}
            except Exception as e:
                clips[idx] = None
                print(f"  encode error idx={idx}: {e}")

        valid = [
            (i, clips[i]) for i in chunk_indices if clips[i] is not None
        ]
        if not valid:
            continue
        v_indices, v_clips = zip(*valid)

        # Phase 2: async decode
        outputs = asyncio.run(
            decode_batch(
                args.server,
                args.model,
                args.prompt,
                list(v_clips),
                args.concurrency,
            )
        )

        # Phase 3: collect results
        for idx, clip, out in zip(v_indices, v_clips, outputs):
            gt = "".join(ds[int(idx)][args.text_column].split())
            if isinstance(out, Exception):
                hyp, latency = f"ERROR: {out}", 0.0
            else:
                hyp, latency = out

            cer = compute_cer(gt, hyp) if not hyp.startswith("ERROR") else -1.0
            rtf = latency / clip["dur"] if clip["dur"] > 0 else 0.0

            rec = {
                "idx": int(idx),
                "gt": gt,
                "hyp": hyp,
                "cer": round(cer, 4),
                "audio_dur_s": round(clip["dur"], 3),
                "latency_s": round(latency, 3),
                "rtf": round(rtf, 4),
            }
            all_results.append(rec)
            f_results.write(json.dumps(rec, ensure_ascii=False) + "\n")

        processed += len(valid)
        elapsed = time.time() - t_total_start
        print(
            f"  [{processed}/{total}] {processed/elapsed:.1f} utt/s, "
            f"elapsed {elapsed:.0f}s",
            flush=True,
        )

    pool.shutdown()
    f_results.close()
    total_elapsed = time.time() - t_total_start

    # ---- Get accept length from server (if speculative) ----
    accept_info = ""
    if spec_algo and spec_algo != "none" and spec_algo != "None":
        try:
            info2 = requests.get(
                f"{args.server}/get_server_info", timeout=10
            ).json()
            # Try various field names
            for k in [
                "avg_spec_accept_length",
                "spec_accept_length",
                "accept_length",
            ]:
                if k in info2:
                    accept_info = f"avg_accept_length={info2[k]}"
                    break
            if not accept_info:
                # Parse from decode log if available
                accept_info = "(check server decode log for accept len)"
        except Exception:
            pass

    # ---- Compute stats ----
    valid_results = [r for r in all_results if r["cer"] >= 0]
    cers = [r["cer"] for r in valid_results]
    rtfs = [r["rtf"] for r in valid_results]
    latencies = [r["latency_s"] for r in valid_results]
    audio_durs = [r["audio_dur_s"] for r in valid_results]

    mean_cer = np.mean(cers) if cers else 0
    mean_rtf = np.mean(rtfs) if rtfs else 0
    p50_rtf = np.percentile(rtfs, 50) if rtfs else 0
    p90_rtf = np.percentile(rtfs, 90) if rtfs else 0
    p99_rtf = np.percentile(rtfs, 99) if rtfs else 0
    mean_latency = np.mean(latencies) if latencies else 0
    total_audio = sum(audio_durs)
    overall_rtf = total_elapsed / total_audio if total_audio > 0 else 0

    # ---- errors.txt ----
    errors_path = os.path.join(args.output_dir, "errors.txt")
    with open(errors_path, "w") as f:
        n_err = 0
        for r in valid_results:
            if r["cer"] > 0:
                n_err += 1
                f.write(f"[CER={r['cer']:.2%}] idx={r['idx']}\n")
                f.write(f"  GT:  {r['gt']}\n")
                f.write(f"  HYP: {r['hyp']}\n\n")
        f.write(f"\nTotal errors: {n_err}/{len(valid_results)}\n")

    # ---- rtf.txt ----
    rtf_path = os.path.join(args.output_dir, "rtf.txt")
    with open(rtf_path, "w") as f:
        f.write(f"Per-utterance RTF (latency / audio_duration):\n")
        f.write(f"  mean:  {mean_rtf:.4f}\n")
        f.write(f"  p50:   {p50_rtf:.4f}\n")
        f.write(f"  p90:   {p90_rtf:.4f}\n")
        f.write(f"  p99:   {p99_rtf:.4f}\n")
        f.write(f"\nOverall RTF (wall_time / total_audio): {overall_rtf:.4f}\n")
        f.write(f"  wall_time:   {total_elapsed:.1f}s\n")
        f.write(f"  total_audio: {total_audio:.1f}s\n")
        f.write(f"\nPer-utterance latency:\n")
        f.write(f"  mean: {mean_latency:.3f}s\n")
        f.write(
            f"  p50:  {np.percentile(latencies, 50):.3f}s\n" if latencies else ""
        )
        f.write(
            f"  p90:  {np.percentile(latencies, 90):.3f}s\n" if latencies else ""
        )

    # ---- summary.txt ----
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"=== ASR Decode Benchmark ===\n")
        f.write(f"Server:       {args.server}\n")
        f.write(f"Model:        {args.model}\n")
        f.write(f"Spec algo:    {spec_algo}\n")
        f.write(f"Draft model:  {draft_path}\n")
        f.write(f"Dataset:      {args.dataset} split={args.split}\n")
        f.write(f"Prompt:       {args.prompt}\n")
        f.write(f"Utterances:   {len(valid_results)}\n")
        f.write(f"Concurrency:  {args.concurrency}\n\n")
        f.write(f"--- Accuracy ---\n")
        f.write(f"CER:          {mean_cer:.2%}\n")
        n_exact = sum(1 for r in valid_results if r["cer"] == 0)
        f.write(
            f"Exact match:  {n_exact}/{len(valid_results)} "
            f"({n_exact/max(len(valid_results),1):.0%})\n\n"
        )
        f.write(f"--- Latency / Throughput ---\n")
        f.write(f"Wall time:         {total_elapsed:.1f}s\n")
        f.write(f"Total audio:       {total_audio:.1f}s\n")
        f.write(f"Overall RTF:       {overall_rtf:.4f}\n")
        f.write(f"Mean utt latency:  {mean_latency:.3f}s\n")
        f.write(f"Mean utt RTF:      {mean_rtf:.4f}\n")
        f.write(f"Throughput:        {len(valid_results)/total_elapsed:.1f} utt/s\n\n")
        if accept_info:
            f.write(f"--- Speculative Decoding ---\n")
            f.write(f"{accept_info}\n")

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(open(summary_path).read())
    print(f"Output dir: {args.output_dir}")
    print(f"  results.jsonl  ({len(all_results)} utterances)")
    print(f"  errors.txt     ({sum(1 for r in valid_results if r['cer']>0)} errors)")
    print(f"  rtf.txt")
    print(f"  summary.txt")


if __name__ == "__main__":
    main()
