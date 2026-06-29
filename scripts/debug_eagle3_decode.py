#!/usr/bin/env python3
"""Offline per-position debug of EAGLE3 draft vs target predictions.

Teacher-forced analysis: at each decode position in the transcription, shows
the draft model's top-k predicted tokens vs the target model's actual token,
so you can see exactly WHERE and WHY the draft disagrees.

Usage (run in the SpecForge env):
    source ../.specforge_env/bin/activate
    export HF_HOME=.../.hf_cache TORCHDYNAMO_DISABLE=1 SPECFORGE_REFERENCE_LOSS=1
    python scripts/debug_eagle3_decode.py \
        --target Qwen/Qwen2-Audio-7B-Instruct \
        --draft outputs/qwen2-audio-7b-eagle3-clean-5e5/epoch_2_step_42000 \
        --vocab-mapping cache/vocab_mapping/da95a5b7d1444120451c5d0a516645e4.pt \
        --top-k 10
"""
import argparse
import io
import os

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from datasets.features import Audio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from specforge.modeling.auto import AutoEagle3DraftModel, AutoDraftModelConfig


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
        arr = np.interp(
            np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr
        ).astype(np.float32)
    return arr


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--draft", required=True, help="Draft checkpoint dir")
    ap.add_argument("--vocab-mapping", required=True, help="vocab mapping .pt")
    ap.add_argument("--top-k", type=int, default=10, help="Show top-k draft preds")
    ap.add_argument("--num-samples", type=int, default=3, help="AISHELL samples to debug")
    ap.add_argument("--ttt-depth", type=int, default=5, help="TTT positions to show (0..depth-1)")
    args = ap.parse_args()

    print("Loading target model...")
    target = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.target, torch_dtype=torch.bfloat16
    ).eval().cuda()
    processor = AutoProcessor.from_pretrained(args.target)
    tokenizer = processor.tokenizer

    print("Loading draft model (trained weights)...")
    draft = AutoEagle3DraftModel.from_pretrained(args.draft, torch_dtype=torch.bfloat16).cuda()
    draft.load_embedding(args.target, embedding_key="language_model.model.embed_tokens.weight")
    draft.load_vocab_mapping(args.vocab_mapping)
    draft.eval()

    d2t = draft.d2t  # draft_id → target_id offset
    t2d = draft.t2d  # target_id → in_draft_vocab (bool)

    print("Loading AISHELL samples...")
    ds = load_dataset("carlot/AIShell", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    instruction = "请将这段音频转写为文本。"

    for sample_idx, ex in enumerate(ds):
        if sample_idx >= args.num_samples:
            break

        transcription = ex["transcription"]
        if not transcription:
            continue
        transcription_clean = "".join(transcription.split())

        audio_arr = _to_16k_mono(ex["audio"])
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "x.wav"},
                {"type": "text", "text": instruction},
            ]},
            {"role": "assistant", "content": transcription_clean},
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        enc = processor(text=text, audio=[audio_arr], sampling_rate=16000, return_tensors="pt", padding=True)
        enc = {k: v.cuda() for k, v in enc.items()}

        # === Target forward ===
        out = target(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            input_features=enc["input_features"],
            feature_attention_mask=enc["feature_attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract 3 aux hidden states (same as training)
        num_hs = len(out.hidden_states)
        num_layers = num_hs - 1
        offset = 1
        low = out.hidden_states[1 + offset]
        mid = out.hidden_states[num_layers // 2 - 1 + offset]
        high = out.hidden_states[num_layers - 4 + offset]
        aux_hidden = torch.cat((low, mid, high), dim=-1)  # [1, seq, 3H]

        target_logits = out.logits  # [1, seq, vocab]
        input_ids = enc["input_ids"]  # [1, seq]
        seq_len = input_ids.shape[1]

        # Find transcription region (assistant tokens)
        # Simple: find where the assistant content starts in input_ids
        # Decode and find the transcription in the decoded text
        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)

        # === Draft forward (teacher-forced, position 0 = predict next token) ===
        projected = draft.project_hidden_states(aux_hidden)  # [1, seq, H]
        inputs_embeds = draft.embed_input_ids(input_ids).to(projected.dtype)

        # Run draft backbone over entire sequence (teacher-forced, no KV cache)
        B, S = input_ids.shape
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        attention_mask = enc["attention_mask"]
        if hasattr(draft, "prepare_decoder_attention_mask"):
            attn_4d = draft.prepare_decoder_attention_mask(
                attention_mask=attention_mask, hidden_states=projected,
                batch_size=B, seq_length=S, past_key_values_length=0,
            )
        else:
            attn_4d = attention_mask

        cache_hidden = [[], []]
        draft_hidden = draft.backbone(
            input_embeds=inputs_embeds,
            hidden_states=projected,
            cache_hidden=cache_hidden,
            attention_mask=attn_4d,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        draft_logits = draft.compute_logits(draft_hidden)  # [1, seq, draft_vocab]

        # === Per-position comparison ===
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx}: transcription = '{transcription_clean}'")
        print(f"Sequence length = {seq_len}, num_target_layers = {num_layers}")
        print(f"{'='*80}")

        # At position i: target_logits[i] → predicts input_ids[i+1] (the NEXT token)
        # draft_logits[i] → also predicts the next token (in draft vocab)
        # Compare: does draft's top-1 (mapped to target vocab via d2t) == target's top-1?

        matches_per_depth = {d: {"match": 0, "total": 0} for d in range(args.ttt_depth)}
        total_match = 0
        total_pos = 0

        # Find the assistant (transcription) start: look for tokens after the last
        # <|im_start|>assistant\n pattern
        assistant_header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        assistant_start = None
        for i in range(seq_len - len(assistant_header_ids)):
            if input_ids[0, i:i+len(assistant_header_ids)].tolist() == assistant_header_ids:
                assistant_start = i + len(assistant_header_ids)
        if assistant_start is None:
            print("  (could not find assistant start, showing all positions)")
            assistant_start = 0

        # Show a few prompt positions + all transcription positions
        show_start = max(0, assistant_start - 3)  # 3 prompt tokens for context

        print(f"\nPos  | Token@pos     | Target next-tok  | Draft top-{args.top_k} (in target vocab)  | Match?")
        print(f"{'─'*100}")

        for pos in range(show_start, seq_len - 1):  # -1 because we predict pos+1
            actual_next = input_ids[0, pos + 1].item()
            actual_tok_str = tokenizer.decode([actual_next]).replace('\n', '\\n')

            # Target's prediction at this position
            target_pred = target_logits[0, pos].argmax(-1).item()
            target_tok_str = tokenizer.decode([target_pred]).replace('\n', '\\n')
            target_agrees_with_actual = (target_pred == actual_next)

            # Draft's top-k
            draft_probs = torch.softmax(draft_logits[0, pos].float(), dim=-1)
            topk_vals, topk_draft_ids = draft_probs.topk(args.top_k)

            # Map draft IDs → target IDs via d2t
            topk_target_ids = topk_draft_ids + d2t[topk_draft_ids]
            draft_top1_target = topk_target_ids[0].item()
            draft_top1_match = (draft_top1_target == target_pred)

            # Is target_pred in draft's top-k?
            in_topk = target_pred in topk_target_ids.tolist()

            # Format top-k string
            topk_strs = []
            for j in range(min(args.top_k, 5)):  # show top-5 for readability
                tid = topk_target_ids[j].item()
                tok = tokenizer.decode([tid]).replace('\n', '\\n')
                prob = topk_vals[j].item()
                marker = "✓" if tid == target_pred else " "
                topk_strs.append(f"{marker}{tok}({prob:.2f})")

            is_transcription = pos >= assistant_start
            region = "AST" if is_transcription else "PRO"
            match_sym = "✓" if draft_top1_match else ("~" if in_topk else "✗")

            cur_tok = tokenizer.decode([input_ids[0, pos].item()]).replace('\n', '\\n')
            print(
                f"{pos:4d} | {cur_tok:12s} | tgt={target_tok_str:12s} "
                f"| {' '.join(topk_strs):60s} | {match_sym} [{region}]"
            )

            if is_transcription:
                total_pos += 1
                if draft_top1_match:
                    total_match += 1
                # Track by TTT depth (simplified: depth 0 = this pos)
                for d in range(args.ttt_depth):
                    shifted_pos = pos + d
                    if shifted_pos < seq_len - 1:
                        matches_per_depth[d]["total"] += 1
                        sp = target_logits[0, shifted_pos].argmax(-1).item()
                        dp = (draft_logits[0, shifted_pos].argmax(-1))
                        dp_target = (dp + d2t[dp]).item()
                        if dp_target == sp:
                            matches_per_depth[d]["match"] += 1

        # Summary
        acc = total_match / max(total_pos, 1)
        print(f"\n--- Summary (transcription region, {total_pos} positions) ---")
        print(f"Draft top-1 match (position 0): {total_match}/{total_pos} = {acc:.1%}")
        print(f"  (This approximates training 'acc' at position 0)")

    print("\nDone.")


if __name__ == "__main__":
    main()
