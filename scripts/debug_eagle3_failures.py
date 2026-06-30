#!/usr/bin/env python3
"""Detailed per-position failure analysis of EAGLE3 draft vs target.

For each transcription position where the draft's top-1 ≠ target's top-1,
logs the full context: decoded-so-far, draft top-5 predictions, actual target
token, GT, and probabilities. Writes to a structured JSONL + human-readable txt.

Usage:
    source ../.specforge_env/bin/activate
    export HF_HOME=.../.hf_cache TORCHDYNAMO_DISABLE=1 SPECFORGE_REFERENCE_LOSS=1
    python scripts/debug_eagle3_failures.py \
        --target yuekai/qwen2_audio_aishell_sft \
        --draft outputs/qwen2-audio-sft-eagle3/epoch_7_step_118000 \
        --vocab-mapping cache/vocab_mapping/078bbb5f371e98924d5e9c85f19427b5.pt \
        --instruction "Detect the language and recognize the speech: <|zh|>" \
        --num-samples 100 --output-dir results/debug_failures
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

from specforge.modeling.auto import AutoEagle3DraftModel


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
    ap.add_argument("--vocab-mapping", required=True)
    ap.add_argument("--instruction", default="Detect the language and recognize the speech: <|zh|>")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output-dir", default="results/debug_failures")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading target model...")
    target = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.target, torch_dtype=torch.bfloat16
    ).eval().cuda()
    processor = AutoProcessor.from_pretrained(args.target)
    tokenizer = processor.tokenizer

    print("Loading draft model...")
    draft = AutoEagle3DraftModel.from_pretrained(args.draft, torch_dtype=torch.bfloat16).cuda()
    draft.load_embedding(args.target, embedding_key="language_model.model.embed_tokens.weight")
    draft.load_vocab_mapping(args.vocab_mapping)
    draft.eval()
    d2t = draft.d2t

    print(f"Loading AISHELL split={args.split}...")
    ds = load_dataset("carlot/AIShell", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    failures_jsonl = os.path.join(args.output_dir, "failures.jsonl")
    failures_txt = os.path.join(args.output_dir, "failures.txt")
    stats_txt = os.path.join(args.output_dir, "stats.txt")

    f_jsonl = open(failures_jsonl, "w", buffering=1)
    f_txt = open(failures_txt, "w", buffering=1)

    # Aggregate stats
    total_positions = 0
    total_matches = 0
    total_failures = 0
    failure_types = {}  # category → count

    assistant_header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    for sample_idx, ex in enumerate(ds):
        if sample_idx >= args.num_samples:
            break

        transcription = ex["transcription"]
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
        enc = {k: v.cuda() for k, v in enc.items()}

        # Target forward
        out = target(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            input_features=enc["input_features"], feature_attention_mask=enc["feature_attention_mask"],
            output_hidden_states=True, use_cache=False,
        )
        num_hs = len(out.hidden_states)
        num_layers = num_hs - 1
        offset = 1
        aux_hidden = torch.cat((
            out.hidden_states[1 + offset],
            out.hidden_states[num_layers // 2 - 1 + offset],
            out.hidden_states[num_layers - 4 + offset],
        ), dim=-1)

        target_logits = out.logits
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]

        # Draft forward
        projected = draft.project_hidden_states(aux_hidden)
        inputs_embeds = draft.embed_input_ids(input_ids).to(projected.dtype)
        B, S = input_ids.shape
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        attn_mask = enc["attention_mask"]
        if hasattr(draft, "prepare_decoder_attention_mask"):
            attn_4d = draft.prepare_decoder_attention_mask(
                attention_mask=attn_mask, hidden_states=projected,
                batch_size=B, seq_length=S, past_key_values_length=0,
            )
        else:
            attn_4d = attn_mask

        draft_hidden = draft.backbone(
            input_embeds=inputs_embeds, hidden_states=projected,
            cache_hidden=[[], []], attention_mask=attn_4d,
            position_ids=position_ids, past_key_values=None, use_cache=False,
        )
        draft_logits = draft.compute_logits(draft_hidden)

        # Find assistant start
        assistant_start = 0
        for i in range(seq_len - len(assistant_header_ids)):
            if input_ids[0, i:i + len(assistant_header_ids)].tolist() == assistant_header_ids:
                assistant_start = i + len(assistant_header_ids)

        # Analyze each transcription position
        sample_matches = 0
        sample_positions = 0

        for pos in range(assistant_start, seq_len - 1):
            sample_positions += 1
            total_positions += 1

            target_pred = target_logits[0, pos].argmax(-1).item()
            target_tok = tokenizer.decode([target_pred])

            draft_probs = torch.softmax(draft_logits[0, pos].float(), dim=-1)
            topk_vals, topk_draft_ids = draft_probs.topk(args.top_k)
            topk_target_ids = topk_draft_ids + d2t[topk_draft_ids]
            draft_top1_target = topk_target_ids[0].item()
            draft_top1_match = (draft_top1_target == target_pred)

            if draft_top1_match:
                sample_matches += 1
                total_matches += 1
                continue

            # === FAILURE — record details ===
            total_failures += 1

            # Decoded so far (all tokens from assistant_start to pos)
            decoded_so_far = tokenizer.decode(input_ids[0, assistant_start:pos + 1].tolist())
            # Remaining GT
            remaining_gt = tokenizer.decode(input_ids[0, pos + 1:].tolist())

            # Draft top-k details
            draft_predictions = []
            for j in range(args.top_k):
                tid = topk_target_ids[j].item()
                tok = tokenizer.decode([tid])
                prob = topk_vals[j].item()
                is_correct = (tid == target_pred)
                draft_predictions.append({
                    "rank": j + 1, "token": tok, "token_id": tid,
                    "prob": round(prob, 4), "is_correct": is_correct
                })

            # What position in transcription is this?
            trans_pos = pos - assistant_start
            trans_total = seq_len - 1 - assistant_start

            # Categorize failure
            cur_tok = tokenizer.decode([input_ids[0, pos].item()])
            target_in_topk = target_pred in topk_target_ids.tolist()
            if target_in_topk:
                category = "in_topk_not_top1"
            elif draft_probs.max().item() > 0.9:
                category = "confident_wrong"
            else:
                category = "uncertain"
            failure_types[category] = failure_types.get(category, 0) + 1

            record = {
                "sample_idx": sample_idx,
                "gt": gt,
                "decoded_so_far": decoded_so_far,
                "remaining_gt": remaining_gt,
                "position_in_transcription": trans_pos,
                "total_transcription_positions": trans_total,
                "current_token": cur_tok,
                "target_next_token": target_tok,
                "target_next_token_id": target_pred,
                "draft_top_k": draft_predictions,
                "target_in_draft_topk": target_in_topk,
                "failure_category": category,
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Human-readable
            topk_str = " | ".join(
                f"{'→' if p['is_correct'] else ' '}{p['token']}({p['prob']:.2f})"
                for p in draft_predictions
            )
            f_txt.write(f"{'─'*80}\n")
            f_txt.write(f"Sample {sample_idx} | pos {trans_pos}/{trans_total} | [{category}]\n")
            f_txt.write(f"  GT:          {gt}\n")
            f_txt.write(f"  Decoded:     {decoded_so_far}█\n")
            f_txt.write(f"  Current tok: '{cur_tok}' → target expects: '{target_tok}'\n")
            f_txt.write(f"  Draft top-{args.top_k}: {topk_str}\n")
            f_txt.write(f"  Remaining:   {remaining_gt}\n\n")

        acc = sample_matches / max(sample_positions, 1)
        if (sample_idx + 1) % 20 == 0:
            print(f"  [{sample_idx+1}/{args.num_samples}] sample acc={acc:.0%}, "
                  f"cumulative acc={total_matches/max(total_positions,1):.1%}, "
                  f"failures={total_failures}", flush=True)

    f_jsonl.close()
    f_txt.close()

    # Write stats
    overall_acc = total_matches / max(total_positions, 1)
    with open(stats_txt, "w") as f:
        f.write(f"=== EAGLE3 Draft Failure Analysis ===\n")
        f.write(f"Target:   {args.target}\n")
        f.write(f"Draft:    {args.draft}\n")
        f.write(f"Samples:  {args.num_samples} (split={args.split})\n\n")
        f.write(f"Total transcription positions: {total_positions}\n")
        f.write(f"Matches (draft top-1 = target): {total_matches} ({overall_acc:.1%})\n")
        f.write(f"Failures: {total_failures} ({total_failures/max(total_positions,1):.1%})\n\n")
        f.write(f"Failure categories:\n")
        for cat, cnt in sorted(failure_types.items(), key=lambda x: -x[1]):
            f.write(f"  {cat:25s}: {cnt:5d} ({cnt/max(total_failures,1):.0%})\n")
        f.write(f"\n  in_topk_not_top1  = target token is in draft's top-{args.top_k} but not rank 1\n")
        f.write(f"  confident_wrong   = draft is >90% confident on a WRONG token\n")
        f.write(f"  uncertain         = draft is uncertain and wrong\n")

    print(f"\n{'='*60}")
    print(open(stats_txt).read())
    print(f"Output: {args.output_dir}/")
    print(f"  failures.jsonl  ({total_failures} failure records)")
    print(f"  failures.txt    (human-readable)")
    print(f"  stats.txt       (summary)")


if __name__ == "__main__":
    main()
