# DFlash Accept Length = 1.0 — SOLVED & VALIDATED

## VALIDATION (2026-07-01) — fix confirmed end-to-end
Retrained draft with the head fix (`outputs/qwen2-audio-sft-dflash-fixedhead/epoch_3_step_42000`,
only **3 of 50 epochs**, run stopped early) served in sglang DFLASH (fa3, block_size=4):
- **mean accept length = 2.30** (range 2.0–2.7), accept rate 0.43 — **up from 1.0** with the broken tied-head checkpoint.
- CER 0.65% / 94% exact on 80 AISHELL test utts (matches target ASR quality; spec decode lossless).
- Serve cmd: `python -m sglang.launch_server --model-path yuekai/qwen2_audio_aishell_sft --speculative-algorithm DFLASH --speculative-draft-model-path <ckpt> --speculative-num-draft-tokens 4 --speculative-num-steps 1 --speculative-eagle-topk 1 --attention-backend fa3 --mem-fraction-static 0.85` (needs `python3.12-dev` for Triton JIT).
More epochs should push accept length higher (toward the target self-consistency ceiling).



## Root cause (confirmed, triple-verified)
**Train-vs-serve LM-head mismatch.** The DFlash draft was trained against a HEAD TIED to the
embedding matrix, but sglang serves with the REAL untied `language_model.lm_head.weight`.

- Qwen2-Audio config: top-level `tie_word_embeddings=True`, but `text_config.tie_word_embeddings=False`.
- Real model ships an untied `language_model.lm_head.weight` (differs from `embed_tokens.weight`, max|Δ|=1.334).
- SpecForge `specforge/modeling/target/target_utils.py:77` read the **top-level** flag → tied
  `lm_head := embed_tokens` → the draft learned hidden states whose argmax is correct *through the
  embedding matrix*.
- sglang `python/sglang/srt/models/qwen2_audio.py:160` keys off `text_config.tie_word_embeddings=False`
  → loads the REAL untied head → every draft prediction is garbage → `accept_length=1.0`.
- **Numerical proof** (target model's OWN next-token accuracy on AISHELL, loss region):
  - REAL untied head: **0.47 – 0.56**
  - TIED (embed-as-head): **0.000**
- z-lab/Qwen3-8B-DFlash works in the same sglang build because Qwen3 genuinely ties (tied == real).

## Ruled out (evidence)
1. FSDP `FULL_STATE_DICT` save — complete: 58 keys, full unsharded, no NaN/Inf, prefix filter clean.
2. Model didn't train — false: early(step2000)-vs-late(step122000) L2 change 8–22% across layers.
3. Reload broken — false: `DFlashDraftModel.from_pretrained` reloads 0 missing / 0 unexpected, bit-identical.
4. Forward geometry (layer capture, fc concat order, mask token, positions/RoPE) — matches sglang.
5. `scripts/eval_dflash_offline.py` never runs the draft (only target self-consistency) — its "0%" was
   never evidence about the checkpoint.

## Fix (applied)
1. `specforge/modeling/target/target_utils.py`: prefer `config.text_config.tie_word_embeddings` when a
   `text_config` exists (composite/multimodal models).
2. `examples/run_qwen2_audio_dflash.sh`: add `--lm-head-key language_model.lm_head.weight`.
- Verified: fixed loader loads a head bit-identical to the real `lm_head` (not the embedding).
- The `epoch_9_step_122000` checkpoint is **unsalvageable** (optimized end-to-end against the wrong
  head; swapping the head at serve time does NOT recover it) → **retrain required**.

## Safety gate for future runs
Before long training, assert target's own next-token accuracy under the chosen head > 0.4. The tied
head would have failed this immediately (0.000).

## Environment note
The `.specforge_env` got transformers downgraded to 4.57.1 during sglang debugging; sglang HEAD needs
transformers 5.x (`PreTrainedConfig`), so `import specforge` (→ `specforge.args` → sglang) now crashes.
Training itself doesn't need that import to succeed for the DFlash path, but resume did. Either upgrade
transformers in `.specforge_env` or make `specforge.args`'s sglang import lazy before resuming.
