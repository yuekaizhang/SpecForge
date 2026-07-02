#!/usr/bin/env python3
"""Profile prefill vs decode time for Qwen2-Audio ASR, with/without EAGLE3 draft.

Starts sglang server on specified GPU(s), sends streaming requests to measure:
  - TTFT (Time To First Token) ≈ prefill time
  - Decode time = total - TTFT
  - Total latency, generated tokens, tokens/s

Usage:
    python scripts/perf_prefill_decode.py \
        --target-model yuekai/qwen2_audio_aishell_sft \
        --gpu 7 \
        --num-samples 50 \
        --draft-model SpecForge/outputs/qwen2-audio-sft-eagle3/epoch_7_step_118000 \
        --spec-algorithm EAGLE3
"""
import argparse
import base64
import io
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import requests
import soundfile as sf


def encode_audio_b64(audio_bytes):
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


def start_server(target_model, gpu, port, draft_model=None, spec_algorithm=None):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", target_model,
        "--port", str(port),
        "--tp", str(len(gpu.split(","))),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--mem-fraction-static", "0.85",
        "--attention-backend", "flashinfer",
    ]
    if draft_model and spec_algorithm:
        cmd += [
            "--speculative-algorithm", spec_algorithm,
            "--speculative-draft-model-path", draft_model,
        ]
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Wait for server ready
    for i in range(300):
        time.sleep(2)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.ok:
                print(f"Server ready after {(i+1)*2}s")
                return proc
        except Exception:
            pass
        # Check if process died
        if proc.poll() is not None:
            out = proc.stdout.read().decode()
            print(f"Server died with code {proc.returncode}")
            print(out[-3000:])
            return None
    print("Server failed to start within 600s")
    proc.kill()
    return None


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def profile_one_streaming(server_url, prompt, audio_b64):
    """Send one streaming request, measure TTFT and total time."""
    payload = {
        "model": "default",
        "temperature": 0,
        "max_tokens": 200,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    t0 = time.monotonic()
    ttft = None
    generated_text = ""
    num_tokens = 0

    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    )
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content and ttft is None:
                ttft = time.monotonic() - t0
            if content:
                generated_text += content
                num_tokens += 1
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    total = time.monotonic() - t0
    if ttft is None:
        ttft = total
    decode_time = total - ttft
    return {
        "ttft": ttft,
        "decode_time": decode_time,
        "total": total,
        "num_tokens": num_tokens,
        "text": generated_text,
    }


def get_accept_length(server_url):
    """Get average accept length from server metrics."""
    try:
        info = requests.get(f"{server_url}/get_server_info", timeout=5).json()
        metrics_raw = requests.get(f"{server_url}/metrics", timeout=5).text
        # Parse accept length from prometheus metrics
        for line in metrics_raw.split("\n"):
            if "spec_accept_length_sum" in line and not line.startswith("#"):
                accept_sum = float(line.split()[-1])
            if "spec_accept_length_count" in line and not line.startswith("#"):
                accept_count = float(line.split()[-1])
        if accept_count > 0:
            return accept_sum / accept_count
    except Exception:
        pass
    return None


def run_benchmark(server_url, clips, prompt, num_warmup=3):
    """Run profiling on clips, return results."""
    results = []
    total = len(clips)

    # Warmup
    print(f"Warming up with {min(num_warmup, total)} samples...")
    for i in range(min(num_warmup, total)):
        profile_one_streaming(server_url, prompt, clips[i]["b64"])

    # Profile
    print(f"Profiling {total} samples...")
    for i, clip in enumerate(clips):
        r = profile_one_streaming(server_url, prompt, clip["b64"])
        r["audio_dur"] = clip["dur"]
        r["idx"] = i
        results.append(r)
        if (i + 1) % 10 == 0:
            avg_ttft = np.mean([x["ttft"] for x in results])
            avg_decode = np.mean([x["decode_time"] for x in results])
            avg_total = np.mean([x["total"] for x in results])
            print(
                f"  [{i+1}/{total}] avg TTFT={avg_ttft:.3f}s  "
                f"decode={avg_decode:.3f}s  total={avg_total:.3f}s"
            )

    return results


def print_summary(label, results, accept_length=None):
    ttfts = [r["ttft"] for r in results]
    decodes = [r["decode_time"] for r in results]
    totals = [r["total"] for r in results]
    ntokens = [r["num_tokens"] for r in results]
    audio_durs = [r["audio_dur"] for r in results]
    tps = [r["num_tokens"] / r["decode_time"] if r["decode_time"] > 0 else 0 for r in results]
    rtfs = [r["total"] / r["audio_dur"] if r["audio_dur"] > 0 else 0 for r in results]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Samples:          {len(results)}")
    print(f"  TTFT (prefill):   mean={np.mean(ttfts):.4f}s  p50={np.median(ttfts):.4f}s  p90={np.percentile(ttfts,90):.4f}s")
    print(f"  Decode time:      mean={np.mean(decodes):.4f}s  p50={np.median(decodes):.4f}s  p90={np.percentile(decodes,90):.4f}s")
    print(f"  Total latency:    mean={np.mean(totals):.4f}s  p50={np.median(totals):.4f}s  p90={np.percentile(totals,90):.4f}s")
    print(f"  Tokens generated: mean={np.mean(ntokens):.1f}")
    print(f"  Decode tok/s:     mean={np.mean(tps):.1f}  p50={np.median(tps):.1f}")
    print(f"  RTF:              mean={np.mean(rtfs):.4f}  p50={np.median(rtfs):.4f}")
    if accept_length is not None:
        print(f"  Accept length:    {accept_length:.2f}")
    print(f"{'='*60}\n")

    return {
        "label": label,
        "n": len(results),
        "ttft_mean": float(np.mean(ttfts)),
        "ttft_p50": float(np.median(ttfts)),
        "ttft_p90": float(np.percentile(ttfts, 90)),
        "decode_mean": float(np.mean(decodes)),
        "decode_p50": float(np.median(decodes)),
        "decode_p90": float(np.percentile(decodes, 90)),
        "total_mean": float(np.mean(totals)),
        "total_p50": float(np.median(totals)),
        "total_p90": float(np.percentile(totals, 90)),
        "tokens_mean": float(np.mean(ntokens)),
        "tps_mean": float(np.mean(tps)),
        "rtf_mean": float(np.mean(rtfs)),
        "accept_length": accept_length,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default="yuekai/qwen2_audio_aishell_sft")
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--spec-algorithm", default="EAGLE3")
    ap.add_argument("--gpu", default="7")
    ap.add_argument("--port", type=int, default=30020)
    ap.add_argument("--dataset", default="carlot/AIShell")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--prompt", default="Detect the language and recognize the speech: <|zh|>")
    ap.add_argument("--output-dir", default="results/perf_profiling")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-draft", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and encode audio clips
    from datasets import load_dataset
    from datasets.features import Audio

    print(f"Loading {args.dataset} split={args.split}...")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    n = min(args.num_samples, len(ds)) if args.num_samples > 0 else len(ds)

    print(f"Encoding {n} audio clips...")
    clips = []
    for i in range(n):
        b64, dur = encode_audio_b64(ds[i]["audio"]["bytes"])
        clips.append({"b64": b64, "dur": dur})
    print(f"Encoded {len(clips)} clips")

    summaries = []

    # === Baseline (no draft) ===
    if not args.skip_baseline:
        print("\n" + "=" * 60)
        print("  BASELINE (no speculative decoding)")
        print("=" * 60)
        proc = start_server(args.target_model, args.gpu, args.port)
        if proc:
            try:
                results_base = run_benchmark(
                    f"http://127.0.0.1:{args.port}", clips, args.prompt
                )
                s = print_summary("BASELINE (no draft)", results_base)
                summaries.append(s)
                with open(os.path.join(args.output_dir, "baseline_results.json"), "w") as f:
                    json.dump(results_base, f, indent=2, ensure_ascii=False)
            finally:
                stop_server(proc)
                time.sleep(5)

    # === With draft ===
    if not args.skip_draft and args.draft_model:
        print("\n" + "=" * 60)
        print(f"  SPECULATIVE ({args.spec_algorithm})")
        print("=" * 60)
        proc = start_server(
            args.target_model, args.gpu, args.port,
            draft_model=args.draft_model,
            spec_algorithm=args.spec_algorithm,
        )
        if proc:
            try:
                results_spec = run_benchmark(
                    f"http://127.0.0.1:{args.port}", clips, args.prompt
                )
                accept_len = get_accept_length(f"http://127.0.0.1:{args.port}")
                s = print_summary(
                    f"SPECULATIVE ({args.spec_algorithm})", results_spec, accept_len
                )
                summaries.append(s)
                with open(os.path.join(args.output_dir, "spec_results.json"), "w") as f:
                    json.dump(results_spec, f, indent=2, ensure_ascii=False)
            finally:
                stop_server(proc)

    # === Comparison ===
    if len(summaries) == 2:
        base, spec = summaries
        print("\n" + "=" * 60)
        print("  COMPARISON")
        print("=" * 60)
        print(f"  Prefill (TTFT):  {base['ttft_mean']:.4f}s → {spec['ttft_mean']:.4f}s  ({spec['ttft_mean']/base['ttft_mean']:.2f}x)")
        print(f"  Decode:          {base['decode_mean']:.4f}s → {spec['decode_mean']:.4f}s  ({spec['decode_mean']/base['decode_mean']:.2f}x)")
        print(f"  Total:           {base['total_mean']:.4f}s → {spec['total_mean']:.4f}s  ({spec['total_mean']/base['total_mean']:.2f}x)")
        print(f"  Decode tok/s:    {base['tps_mean']:.1f} → {spec['tps_mean']:.1f}  ({spec['tps_mean']/base['tps_mean']:.2f}x)")
        print(f"  RTF:             {base['rtf_mean']:.4f} → {spec['rtf_mean']:.4f}")
        if spec["accept_length"]:
            print(f"  Accept length:   {spec['accept_length']:.2f}")
        print(f"{'='*60}")

    # Save summary
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
