#!/usr/bin/env python3
"""Pre-build the processed-dataset + vocab-mapping caches for the multi-domain
EAGLE3 run, so sbatch chunks start warm (skips ~3 h of rank-0 preprocessing).

CPU-only: no GPUs, no target-model weights — just tokenizer + processor + mel
extraction. The cache key REPLICATES train_eagle3.py's build_dataloaders string
byte-for-byte for the args used by examples/run_qwen3_omni_eagle3_multidomain.sh;
if any of those args change, this must change too (or the cache goes cold).
"""
import hashlib
import os

from datasets import load_from_disk
from transformers import AutoProcessor, AutoTokenizer

from specforge.data import build_eagle3_dataset, generate_vocab_mapping_file

# ==== MUST mirror examples/run_qwen3_omni_eagle3_multidomain.sh (full run) ====
TRAIN_DATA_PATH = "outputs/multidomain/combined_train"   # literal arg string
MAX_LENGTH = 1024
CHAT_TEMPLATE = "qwen"
TARGET_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
INSTRUCTION = "Transcribe the English audio into text."
KEEP_SPACES = True            # --keep-transcription-spaces
CACHE_DIR = "./cache"
DRAFT_VOCAB_SIZE = 32000      # configs/qwen3-omni-30b-eagle3.json
TARGET_VOCAB_SIZE = 152064

# ==== cache key: replicate train_eagle3.py build_dataloaders exactly ====
cache_params_string = (
    f"{TRAIN_DATA_PATH}-"
    f"{MAX_LENGTH}-"
    f"{CHAT_TEMPLATE}-"
    f"{TARGET_MODEL}"
)
# label_override: None -> no suffix
cache_params_string += f"-instruction:{INSTRUCTION}"
if KEEP_SPACES:
    cache_params_string += "-keep_spaces"
# text_column == 'transcription' -> no suffix; train_config None -> no suffix;
# train_split == 'train' (default) -> no suffix
cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
print(f"cache_key = {cache_key}")
print(f"  -> {CACHE_DIR}/processed_dataset/{cache_key}.pkl")
print(f"  -> {CACHE_DIR}/vocab_mapping/{cache_key}.pt")

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
processor = AutoProcessor.from_pretrained(TARGET_MODEL)
train_dataset = load_from_disk(TRAIN_DATA_PATH)
print(f"rows: {len(train_dataset)}")

train_eagle3_dataset = build_eagle3_dataset(
    dataset=train_dataset,
    tokenizer=tokenizer,
    chat_template=CHAT_TEMPLATE,
    max_length=MAX_LENGTH,
    cache_dir=os.path.join(CACHE_DIR, "processed_dataset"),
    cache_key=cache_key,
    is_vlm=False,
    is_audio=True,
    is_preformatted=False,
    processor=processor,
    num_proc=1,
    train_only_last_turn=False,
    instruction=INSTRUCTION,
    strip_transcription_whitespace=not KEEP_SPACES,
)
print(f"processed rows: {len(train_eagle3_dataset)}")

vocab_mapping_path = generate_vocab_mapping_file(
    dataset=train_eagle3_dataset,
    target_vocab_size=TARGET_VOCAB_SIZE,
    draft_vocab_size=DRAFT_VOCAB_SIZE,
    cache_dir=os.path.join(CACHE_DIR, "vocab_mapping"),
    cache_key=cache_key,
)
print(f"vocab mapping: {vocab_mapping_path}")
print("PREBUILD_DONE")
