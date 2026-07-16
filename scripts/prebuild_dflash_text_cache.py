"""Prebuild the DFlash text preprocessing cache with a SINGLE process.

train_dflash.py runs build_eagle3_dataset independently on every rank. With
num_proc=1 and 8 ranks, all 8 write their own ~11 GB map cache into the same
dir simultaneously -> ~88 GB peak -> "Disk quota exceeded". The map cache key
is GPU-count-independent, so building it once here (with multiprocessing for
speed) lets all 8 training ranks hit the warm cache via load_from_cache_file
and skip the write entirely.

The cache_key MUST match train_dflash.build_dataloader byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from specforge.data import build_eagle3_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--chat-template", default="qwen")
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--num-proc", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Byte-for-byte identical to train_dflash.build_dataloader. This is a text
    # run: no --instruction and no --keep-transcription-spaces, so neither
    # suffix is appended.
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    cache_dir = os.path.join(args.cache_dir, "processed_dataset")
    print(f"cache_key   = {cache_key}")
    print(f"cache_file  = {os.path.join(cache_dir, cache_key + '.pkl')}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    print(f"loaded {len(train_dataset)} rows; preprocessing with num_proc={args.num_proc}...")

    build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=False,
        is_audio=False,
        processor=None,
        instruction=None,
        strip_transcription_whitespace=True,
        cache_dir=cache_dir,
        cache_key=cache_key,
        num_proc=args.num_proc,
    )
    print("PREBUILD_DONE")


if __name__ == "__main__":
    main()
