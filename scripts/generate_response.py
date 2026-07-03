#!/usr/bin/env python3
"""Generate target-model responses for EAGLE3 label training (Qwen3-Omni + LibriSpeech).

Sends each audio clip to sglang servers via the OpenAI chat API with the message
template:

    [{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "data:audio/flac;base64,<b64>"}},
        {"type": "text", "text": "Transcribe the English audio into text."}]}]

(`audio_url` + base64 data URI is the ONLY audio content type the sglang chat API
accepts; it is the wire equivalent of the local {"type": "audio", "audio": path}
template. LibriSpeech FLAC bytes are sent as-is — the server's load_audio decodes
any soundfile/torchcodec-readable container and resamples to 16 kHz mono.)

Output JSONL (one line per clip, NO audio):
    {"idx": <int dataset row>, "id": "<librispeech id>", "gt": "<original text>",
     "target_gen": "<greedy generation or ERROR: ...>"}

This plugs into training via `train_eagle3.py --label-override <out.jsonl>`
(consumer reads idx + target_gen; idx = raw dataset row index, so dataset id,
subset and split MUST match between this script and training).

Usage:
    # Launch servers first: bash examples/launch_qwen3_omni_servers.sh
    python scripts/generate_response.py \
        --model qwen3-omni \
        --server-address 127.0.0.1:30000 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 \
        --concurrency 32 \
        --temperature 0.0 \
        --out outputs/librispeech_clean_train100_qwen3omni.jsonl
"""
import argparse
import asyncio
import base64
import json
import os
import re
import string
import time

import aiohttp


def build_payload(model, prompt, audio_b64, mime, temperature, max_tokens):
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "audio_url",
             "audio_url": {"url": f"data:{mime};base64,{audio_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }


async def send_one(session, url, payload, sem):
    async with sem:
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                return f"ERROR: {json.dumps(data)[:300]}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"


async def send_chunk(servers, items, args):
    """items: list of (idx, b64). Per-server semaphore of args.concurrency."""
    urls = [f"http://{s}/v1/chat/completions" for s in servers]
    sems = [asyncio.Semaphore(args.concurrency) for _ in servers]
    conn = aiohttp.TCPConnector(limit=0)  # per-server control via semaphores
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = []
        for k, (idx, b64) in enumerate(items):
            s = k % len(urls)
            payload = build_payload(args.model, args.prompt, b64, args.audio_mime,
                                    args.temperature, args.max_tokens)
            tasks.append(send_one(session, urls[s], payload, sems[s]))
        return await asyncio.gather(*tasks, return_exceptions=True)


_PUNCT = str.maketrans("", "", string.punctuation + "，。！？、；：""''")


def norm(s):
    """Loose normalization for the progress match stat ONLY (never for output)."""
    return re.sub(r"\s+", " ", s.upper().translate(_PUNCT)).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3-omni", help="served model name")
    ap.add_argument("--server-address", nargs="+",
                    default=["127.0.0.1:30000", "127.0.0.1:30001",
                             "127.0.0.1:30002", "127.0.0.1:30003"],
                    help="host:port list; requests are round-robined")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="max in-flight requests PER SERVER")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--dataset", default="openslr/librispeech_asr")
    ap.add_argument("--subset", default="clean",
                    help="dataset config name; MUST also be pinned at training")
    ap.add_argument("--split", default="train.100")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--id-column", default="id")
    ap.add_argument("--prompt", default="Transcribe the English audio into text.")
    ap.add_argument("--audio-mime", default="audio/flac",
                    help="mime for the data URI (informational; server sniffs bytes)")
    ap.add_argument("--num-samples", type=int, default=0, help="0=all")
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Servers: {args.server_address} (concurrency {args.concurrency}/server)")
    print(f"Decoding: temperature={args.temperature} max_tokens={args.max_tokens}")
    print(f"Prompt: {args.prompt!r}")

    from datasets import load_dataset
    from datasets.features import Audio
    print(f"Loading {args.dataset} [{args.subset}] split={args.split} ...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    total = len(ds) if args.num_samples <= 0 else min(args.num_samples, len(ds))
    print(f"Rows: {len(ds)}; processing {total}")

    # Resume: skip rows already generated successfully in --out.
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not str(d.get("target_gen", "")).startswith("ERROR"):
                    done.add(d["idx"])
        print(f"Resume: {len(done)} rows already done in {args.out}")

    todo = [i for i in range(total) if i not in done]
    if not todo:
        print("Nothing to do.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f_out = open(args.out, "a", buffering=1)  # append + line-buffered

    t0 = time.time()
    processed = 0
    match = 0
    for c0 in range(0, len(todo), args.chunk_size):
        chunk = todo[c0:c0 + args.chunk_size]
        items, gts, ids = [], [], []
        for i in chunk:
            ex = ds[i]
            raw = ex["audio"]["bytes"]
            if raw is None:
                continue
            items.append((i, base64.b64encode(raw).decode("ascii")))
            gts.append(ex[args.text_column])
            ids.append(str(ex.get(args.id_column, i)))

        results = asyncio.run(send_chunk(args.server_address, items, args))

        for (idx, _), gt, rid, gen in zip(items, gts, ids, results):
            if isinstance(gen, Exception):
                gen = f"ERROR: {gen}"
            if norm(gt) == norm(gen if isinstance(gen, str) else ""):
                match += 1
            f_out.write(json.dumps(
                {"idx": idx, "id": rid, "gt": gt, "target_gen": gen},
                ensure_ascii=False) + "\n")

        processed += len(items)
        el = time.time() - t0
        rate = processed / el
        eta = (len(todo) - processed) / rate if rate > 0 else 0
        print(f"  [{processed}/{len(todo)}] {rate:.1f} clips/s, "
              f"norm-match={match}/{processed} ({match / max(processed, 1):.0%}), "
              f"ETA {eta / 60:.1f} min", flush=True)

    f_out.close()
    el = time.time() - t0
    print(f"\nDone: {processed} clips in {el / 60:.1f} min ({processed / el:.1f} clips/s)")
    print(f"Saved to: {args.out}")

    with open(args.out) as f:
        shown = 0
        for line in f:
            d = json.loads(line)
            if d["idx"] not in set(todo[:5]):
                continue
            m = "≈" if norm(d["gt"]) == norm(str(d["target_gen"])) else "✗"
            print(f"  [{m}] GT : {d['gt'][:90]}")
            print(f"      GEN: {str(d['target_gen'])[:90]}")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
