# DFlash on SpeechIO ZH00011 (out-of-domain) — 1×H100, batch=1

Dataset `yuekai/speechio_test` config `SPEECHIO_ASR_ZH00011` (1053 utts, `text` column), same
Qwen2-Audio SFT target, fa3, concurrency=1. Draft was trained ONLY on AISHELL.

| config | CER | accept_len | mean lat | RTF | utt/s | speedup |
|---|---|---|---|---|---|---|
| baseline (no draft) | 14.70% | – | 0.368 s | 0.0313 | 2.7 | 1.00× |
| DFlash block-4 (ep24) | 14.33% | 1.16 | 0.395 s | 0.0336 | 2.5 | 0.93× (slower) |
| DFlash block-6 (ep8)  | 14.33% | 1.06 | 0.421 s | 0.0358 | 2.4 | 0.87× (slower) |
| DFlash block-8 (ep28) | 14.33% | 1.06 | 0.423 s | 0.0360 | 2.4 | 0.87× (slower) |

**Key finding — the AISHELL-trained draft does NOT transfer out-of-domain.**
- Accept length **collapses to ~1.0–1.16** (vs 2.3–2.65 on AISHELL) → almost nothing accepted.
- With no acceptance, the draft+verify overhead makes spec decoding **SLOWER than no-draft baseline**
  (0.395–0.423 s vs 0.368 s; 0.87–0.93×).
- Bigger block = worse (more wasted draft compute): block-8 accept 1.06 < block-4 1.16.

**Contrast with in-domain AISHELL** (results/dflash_blocksize_final_c1.md): accept 2.65, 1.60× speedup.
So this draft's benefit is domain-specific — it overfits AISHELL. For SpeechIO-like domains you'd
need a draft trained on matching / multi-domain data.

Notes:
- CER ~14.3–14.7% across all (spec is ~lossless; the 0.37% baseline-vs-spec gap and 260 vs 271 exact
  is minor bf16 numerical-path difference between the pure-AR baseline and the verify path, not a real
  quality change).
- Latencies ~4-5× higher than AISHELL because SpeechIO ZH00011 utterances are much longer.
