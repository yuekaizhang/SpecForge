"""Prepare a small subset of carlot/AIShell for Qwen2-Audio EAGLE3 training.

Decodes audio with soundfile (avoids torchcodec) to 16 kHz mono float32 and
saves a HF dataset to disk with columns: audio={"array","sampling_rate"}, transcription.
"""
import argparse
import io

import numpy as np
import soundfile as sf
from datasets import Dataset, load_dataset
from datasets.features import Audio


def decode_16k_mono(audio_field):
    # audio_field is {"bytes": ..., "path": ...} because we cast decode=False
    data = audio_field["bytes"]
    arr, sr = sf.read(io.BytesIO(data), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        n = int(round(len(arr) / sr * 16000))
        arr = np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr).astype(
            np.float32
        )
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset("carlot/AIShell", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    for i, ex in enumerate(ds):
        if i >= args.num_samples:
            break
        arr = decode_16k_mono(ex["audio"])
        rows.append({"array": arr, "sampling_rate": 16000, "transcription": ex["transcription"]})

    out = Dataset.from_list(
        [{"audio": {"array": r["array"], "sampling_rate": 16000}, "transcription": r["transcription"]} for r in rows]
    )
    out.save_to_disk(args.out)
    print(f"saved {len(out)} examples to {args.out}")
    print("example transcription:", out[0]["transcription"])
    print("audio len (samples):", len(out[0]["audio"]["array"]))


if __name__ == "__main__":
    main()
