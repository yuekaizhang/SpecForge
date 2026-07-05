#!/usr/bin/env python3
"""Assemble the multi-domain EAGLE3 training dataset.

For each source: join the regenerated labels (target_gen) by raw dataset row
idx, keep only rows with a valid generation, and normalize to three columns:
  audio (Audio(decode=False) layout), transcription (= target_gen),
  source (dataset tag, for stats/debugging).
Then concatenate all sources, GLOBAL-shuffle (seed 42), and save_to_disk.
Training consumes the directory via train_eagle3.py's load_from_disk branch
(labels are baked into 'transcription'; no --label-override / idx alignment).

IMPLEMENTATION NOTE: the audio column is never passed through .map or
Features.encode_example — Audio.encode_example imports torchcodec, which is
broken in this env (no FFmpeg). select/add_column/remove_columns/cast_column
are pure arrow ops and never encode.

Usage:
    python scripts/assemble_multidomain.py --out outputs/multidomain/combined_train
"""
import argparse
import json
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets.features import Audio

SOURCES = [
    # name, dataset(or dir), config, split
    ("gigaspeech", "speechcolab/gigaspeech", "s", "train"),
    ("ami", "edinburghcstr/ami", "ihm", "train"),
    ("spgispeech", "kensho/spgispeech", "S", "train"),
    ("earnings22", "sanchit-gandhi/earnings22_split", None, "train"),
    ("librispeech", "openslr/librispeech_asr", "clean", "train.100"),
    ("voxpopuli", "outputs/multidomain/voxpopuli_en_train4k", None, "train"),
]


def load_labels(path):
    """idx -> target_gen, keeping the LAST non-ERROR row per idx (dedupes retries)."""
    good, bad = {}, set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gen = str(d.get("target_gen", ""))
            if gen.startswith("ERROR") or not gen.strip():
                bad.add(d["idx"])
            else:
                good[d["idx"]] = gen
                bad.discard(d["idx"])
    return good, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="outputs/multidomain")
    ap.add_argument("--out", default="outputs/multidomain/combined_train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parts = []
    for name, dataset, cfg, split in SOURCES:
        labels, bad = load_labels(os.path.join(args.labels_dir, f"{name}_labels.jsonl"))
        print(f"[{name}] labels={len(labels)} dropped_error_idx={len(bad)}", flush=True)

        ds = load_from_disk(dataset) if os.path.isdir(dataset) else load_dataset(
            dataset, cfg, split=split
        )
        ds = ds.cast_column("audio", Audio(decode=False))  # arrow storage cast only
        assert len(labels) <= len(ds), (name, len(labels), len(ds))

        keep = sorted(labels.keys())
        ds = ds.select(keep)
        ds = ds.remove_columns([c for c in ds.column_names if c != "audio"])
        ds = ds.add_column("transcription", [labels[i] for i in keep])
        ds = ds.add_column("source", [name] * len(keep))
        print(f"[{name}] rows={len(ds)} features={list(ds.features)}", flush=True)
        parts.append(ds)

    combined = concatenate_datasets(parts)
    print(f"combined rows={len(combined)}", flush=True)
    combined = combined.shuffle(seed=args.seed)  # GLOBAL shuffle; save_to_disk flattens
    combined.save_to_disk(args.out)
    print(f"saved -> {args.out}", flush=True)

    print("first 20 sources after shuffle:", combined["source"][:20], flush=True)
    from collections import Counter

    print("source counts:", Counter(combined["source"]), flush=True)
    print("ASSEMBLY_DONE", flush=True)


if __name__ == "__main__":
    main()
