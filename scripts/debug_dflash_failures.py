#!/usr/bin/env python3
"""Per-position / per-block-offset failure analysis of a DFlash draft vs target.

DFlash predicts a BLOCK of `block_size` tokens from each anchor position: offset 0
is the (given) anchor token, offsets 1..block_size-1 are the drafted tokens. A draft
token is "accepted" in sglang when it matches the target model's greedy token at that
position. This script teacher-forces the target on the GT transcription, runs the REAL
DFlash draft forward (reusing OnlineDFlashModel's noise-embed / position-id / sdpa-mask
helpers so it matches training exactly), and for every drafted position where
draft.top1 != target.greedy logs the full context. It also reports per-offset accuracy
(offset 1 vs 2 vs 3 ...), the dominant failure categories, and the accept-length implied
by the per-offset acceptance chain.

Usage:
    PY=.../.specforge_env/bin/python
    CUDA_VISIBLE_DEVICES=1 HF_HOME=... TORCHDYNAMO_DISABLE=1 $PY scripts/debug_dflash_failures.py \
        --target yuekai/qwen2_audio_aishell_sft \
        --draft outputs/qwen2-audio-sft-dflash-fixedhead/epoch_3_step_42000 \
        --instruction "Detect the language and recognize the speech: <|zh|>" \
        --num-samples 100 --output-dir results/dflash_debug_failures
"""
import argparse
import io
import json
import os

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from datasets.features import Audio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from specforge.core.dflash import OnlineDFlashModel, create_dflash_sdpa_mask
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead


def _to_16k_mono(audio):
    if isinstance(audio, dict) and audio.get("array") is not None:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", 16000))
    else:
        data = audio["bytes"] if audio.get("bytes") is not None else audio["path"]
        src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
        arr, sr = sf.read(src, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        n = int(round(len(arr) / sr * 16000))
        arr = np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr).astype(np.float32)
    return arr


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="yuekai/qwen2_audio_aishell_sft")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--instruction", default="Detect the language and recognize the speech: <|zh|>")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output-dir", default="results/dflash_debug_failures")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    # --- draft config ---
    with open(os.path.join(args.draft, "config.json")) as f:
        dcfg = json.load(f)
    block_size = dcfg["block_size"]
    target_layer_ids = dcfg["dflash_config"]["target_layer_ids"]
    mask_token_id = dcfg["dflash_config"]["mask_token_id"]
    print(f"DFlash: block_size={block_size}, layers={target_layer_ids}, mask={mask_token_id}")

    print("Loading target model...")
    target = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.target, dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.target)
    tokenizer = processor.tokenizer

    print("Loading draft model...")
    draft = DFlashDraftModel.from_pretrained(args.draft, dtype=torch.bfloat16).eval().to(device)
    draft.config._attn_implementation = "sdpa"

    # untied head + embeddings (matches sglang serving; the fix)
    comps = TargetEmbeddingsAndHead.from_pretrained(
        args.target,
        embed_key="language_model.model.embed_tokens.weight",
        lm_head_key="language_model.lm_head.weight",
        device="cuda", dtype=torch.bfloat16,
    )
    dflash = OnlineDFlashModel(
        draft_model=draft, target_lm_head=comps.lm_head, target_embed_tokens=comps.embed_tokens,
        mask_token_id=mask_token_id, block_size=block_size, attention_backend="sdpa",
    ).eval().to(device)

    print(f"Loading AISHELL split={args.split}...")
    ds = load_dataset("carlot/AIShell", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    f_jsonl = open(os.path.join(args.output_dir, "failures.jsonl"), "w", buffering=1)
    f_txt = open(os.path.join(args.output_dir, "failures.txt"), "w", buffering=1)

    total = 0
    matches = 0
    off_total = {k: 0 for k in range(1, block_size)}
    off_match = {k: 0 for k in range(1, block_size)}
    failure_types = {}

    assistant_header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    for sample_idx, ex in enumerate(ds):
        if sample_idx >= args.num_samples:
            break
        transcription = ex.get("transcription", "")
        if not transcription:
            continue
        gt = "".join(transcription.split())
        audio_arr = _to_16k_mono(ex["audio"])

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "x.wav"},
                {"type": "text", "text": args.instruction},
            ]},
            {"role": "assistant", "content": gt},
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        enc = processor(text=text, audio=[audio_arr], sampling_rate=16000, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]

        # --- target forward: aux hidden (5 layers, +1 offset) + greedy ---
        out = target(
            input_ids=input_ids, attention_mask=enc["attention_mask"],
            input_features=enc["input_features"], feature_attention_mask=enc["feature_attention_mask"],
            output_hidden_states=True, use_cache=False,
        )
        aux_hidden = torch.cat([out.hidden_states[lid + 1] for lid in target_layer_ids], dim=-1)
        target_greedy = out.logits[0].argmax(-1)  # target_greedy[p] = greedy token at position p+1

        # assistant start
        assistant_start = 0
        for i in range(seq_len - len(assistant_header_ids)):
            if input_ids[0, i:i + len(assistant_header_ids)].tolist() == assistant_header_ids:
                assistant_start = i + len(assistant_header_ids)
        if assistant_start == 0 or seq_len - assistant_start < 2:
            continue

        # --- deterministic full-coverage anchors: draft a block from each position ---
        lo = max(assistant_start - 1, 0)
        anchors = torch.arange(lo, seq_len - 1, device=device).unsqueeze(0)  # [1, N]
        keep = torch.ones_like(anchors, dtype=torch.bool)

        # reuse the REAL training helpers so geometry matches exactly
        noise_embedding = dflash._create_noise_embed(input_ids, anchors, keep)
        ctx_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        draft_pos = dflash._create_position_ids(anchors)
        full_pos = torch.cat([ctx_pos, draft_pos], dim=1)
        attn_mask = create_dflash_sdpa_mask(anchors, keep, seq_len, block_size, device)

        out_hidden = draft(
            position_ids=full_pos, noise_embedding=noise_embedding,
            target_hidden=aux_hidden, attention_mask=attn_mask,
        )
        logits = comps.lm_head(out_hidden)  # [1, N*bs, V]
        N = anchors.shape[1]
        logits = logits.view(1, N, block_size, -1)

        sample_fail = 0
        for n in range(N):
            a = anchors[0, n].item()
            for k in range(1, block_size):
                abs_pos = a + k
                if abs_pos >= seq_len:
                    continue
                # only score positions inside the assistant transcription
                if abs_pos < assistant_start:
                    continue
                ref = target_greedy[abs_pos - 1].item()  # target greedy token AT abs_pos
                probs = torch.softmax(logits[0, n, k].float(), -1)
                topv, topi = probs.topk(args.top_k)
                pred = topi[0].item()

                total += 1
                off_total[k] += 1
                if pred == ref:
                    matches += 1
                    off_match[k] += 1
                    continue
                sample_fail += 1

                ref_in_topk = ref in topi.tolist()
                if ref_in_topk:
                    cat = "in_topk_not_top1"
                elif probs.max().item() > 0.9:
                    cat = "confident_wrong"
                else:
                    cat = "uncertain"
                failure_types[cat] = failure_types.get(cat, 0) + 1

                decoded_so_far = tokenizer.decode(input_ids[0, assistant_start:a + 1].tolist())
                gt_tok = tokenizer.decode([input_ids[0, abs_pos].item()])
                ref_tok = tokenizer.decode([ref])
                topk = [{
                    "rank": j + 1, "token": tokenizer.decode([topi[j].item()]),
                    "token_id": topi[j].item(), "prob": round(topv[j].item(), 4),
                    "is_ref": topi[j].item() == ref,
                } for j in range(args.top_k)]

                rec = {
                    "sample_idx": sample_idx, "gt": gt, "block_offset": k,
                    "anchor_pos_in_trans": a - assistant_start + 1,
                    "decoded_so_far": decoded_so_far,
                    "target_greedy_token": ref_tok, "gt_token": gt_tok,
                    "target_greedy_eq_gt": ref == input_ids[0, abs_pos].item(),
                    "draft_top_k": topk, "ref_in_draft_topk": ref_in_topk,
                    "failure_category": cat,
                }
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

                topk_str = " | ".join(
                    f"{'→' if p['is_ref'] else ' '}{p['token']}({p['prob']:.2f})" for p in topk
                )
                f_txt.write("─" * 80 + "\n")
                f_txt.write(f"Sample {sample_idx} | offset +{k} | [{cat}]\n")
                f_txt.write(f"  GT:        {gt}\n")
                f_txt.write(f"  Accepted:  {decoded_so_far}█  (draft must emit next)\n")
                f_txt.write(f"  target-greedy: '{ref_tok}'  gt: '{gt_tok}'"
                            f"{'  [target≠gt]' if ref != input_ids[0, abs_pos].item() else ''}\n")
                f_txt.write(f"  draft top-{args.top_k}: {topk_str}\n\n")

        if (sample_idx + 1) % 20 == 0:
            print(f"  [{sample_idx+1}/{args.num_samples}] acc={matches/max(total,1):.1%} "
                  f"fails={total-matches}", flush=True)

    f_jsonl.close()
    f_txt.close()

    # --- accept length implied by per-offset acceptance chain ---
    p = {k: off_match[k] / max(off_total[k], 1) for k in range(1, block_size)}
    chain, prod = 0.0, 1.0
    for k in range(1, block_size):
        prod *= p[k]
        chain += prod
    implied_accept = 1.0 + chain  # +1 bonus token (target's own next token)

    overall = matches / max(total, 1)
    stats = os.path.join(args.output_dir, "stats.txt")
    with open(stats, "w") as f:
        f.write("=== DFlash Draft Failure Analysis ===\n")
        f.write(f"Target: {args.target}\nDraft:  {args.draft}\n")
        f.write(f"Samples: {args.num_samples} (split={args.split})  block_size={block_size}\n")
        f.write(f"Reference: target model greedy token (acceptance criterion)\n\n")
        f.write(f"Scored drafted positions: {total}\n")
        f.write(f"Draft top-1 == target greedy: {matches} ({overall:.1%})\n")
        f.write(f"Failures: {total-matches} ({(total-matches)/max(total,1):.1%})\n\n")
        f.write("Per-block-offset acceptance (P a drafted token at this offset is accepted):\n")
        for k in range(1, block_size):
            f.write(f"  offset +{k}: {off_match[k]}/{off_total[k]} = {p[k]:.1%}\n")
        f.write(f"\nImplied accept length = 1 + sum_k prod_{{j<=k}} p_j = {implied_accept:.2f}\n")
        f.write("  (sanity-check vs sglang-measured accept length)\n\n")
        f.write("Failure categories:\n")
        for c, n in sorted(failure_types.items(), key=lambda x: -x[1]):
            f.write(f"  {c:20s}: {n:5d} ({n/max(total-matches,1):.0%})\n")
        f.write("\n  in_topk_not_top1 = target token in draft top-k but not rank 1\n")
        f.write("  confident_wrong  = draft >90% confident on a WRONG token\n")
        f.write("  uncertain        = draft uncertain and wrong\n")

    print("\n" + "=" * 60)
    print(open(stats).read())
    print(f"Output: {args.output_dir}/  (failures.jsonl, failures.txt, stats.txt)")


if __name__ == "__main__":
    import sys
    main()
    # HF streaming datasets leave background HTTP threads that crash the
    # interpreter at finalization (PyGILState_Release). Files are already
    # flushed (line-buffered + closed), so hard-exit to avoid the core dump.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
