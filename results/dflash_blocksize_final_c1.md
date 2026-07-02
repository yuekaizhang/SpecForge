# DFlash block-size comparison (final checkpoints) — AISHELL test, 1×H100, batch=1

Same target (`yuekai/qwen2_audio_aishell_sft`), fa3, mem-fraction 0.85, concurrency=1, full test (7176 utts).
Each served with `--speculative-num-draft-tokens = block_size`.

| config | epoch | accept_len | CER | mean lat | RTF | utt/s | speedup |
|---|---|---|---|---|---|---|---|
| baseline (no draft) | – | 1.00 | 2.01% | 0.131 s | 0.0262 | 7.6 | 1.00× |
| **block-4** | 24 | **2.65** | 2.01% | **0.082 s** | 0.0164 | **12.1** | **1.60×** |
| block-6 | 8 | 2.38 | 2.01% | 0.085 s | 0.0171 | 11.6 | 1.54× |
| block-8 | 28 | 2.26 | 2.01% | 0.088 s | 0.0175 | 11.3 | 1.49× |

- **Accuracy identical** across all (CER 2.01%, exact 5827/7176) — lossless.
- **Block-4 wins**: highest accept length, lowest latency, best speedup (1.60×).
- **Bigger block ⇒ LOWER accept length here** (2.65 → 2.38 → 2.26) and slightly higher latency.
  The trend holds even at matched-high maturity: block-8 (epoch 28) < block-4 (epoch 24).
  For audio-conditioned ASR the far draft offsets (+4…) are rarely accepted (end-of-utterance is
  acoustically determined and unpredictable — see results/dflash_debug_failures), so a larger block
  just adds draft+verify overhead without raising acceptance. The accept-length ceiling for this task
  is ~2.6–2.7.
- Caveat: block-6 is only epoch 8 (vs 24/28); more training might lift it toward ~2.5, but the
  block-4>block-8 result at matched maturity indicates it won't overtake block-4.

**Recommendation: block_size=4 is the sweet spot for Qwen2-Audio ASR DFlash.**
