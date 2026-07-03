# Qwen3-Omni-30B EAGLE3 — LibriSpeech clean/test, 1×H100, batch=1

Target `Qwen/Qwen3-Omni-30B-A3B-Instruct` (thinker), sglang tp=1, fa3,
`--mem-fraction-static 0.85 --cuda-graph-max-bs-decode 1`, greedy, 2620 utts,
prompt = training prompt: `Transcribe the English audio into text.`.
EAGLE3: num-steps 3, topk 1, draft-tokens 4, `--context-length 8192`
(draft max_position_embeddings=8192 < thinker 65536; sglang refuses otherwise).
WER = corpus-level on uppercase/punctuation-normalized text (`scripts/compute_wer.py`).

| config | draft ckpt | **WER** | CER | accept_len | mean lat | utt/s | speedup |
|---|---|---|---|---|---|---|---|
| baseline (no draft) | – | 1.70% | 0.48% | – | 0.195 s | 5.1 | 1.00× |
| EAGLE3 **ttt5** | 50ep/epoch_40_step_146000 | 1.68% | 0.47% | 2.23 | 0.166 s | 5.9 | 1.17× |
| EAGLE3 **ttt7** | 50ep-ttt7/epoch_37_step_134000 | **1.68%** | 0.47% | **2.38** | **0.159 s** | **6.2** | **1.23×** |

- **Lossless**: ttt5 and ttt7 produce byte-identical outputs (WER 881/52576, exact 2098 — identical),
  ≈ baseline (tiny diff is bf16 verify-path tie noise).
- **ttt7 > ttt5**: accept 2.38 vs 2.23 → 1.23× vs 1.17× — deeper TTT unrolling helps.
- **CUDA graphs dominate this setup**: no-graph baseline was 1.0 utt/s / 1.69 s/utt; `--cuda-graph-max-bs-decode 1`
  alone gave **5×** (48-layer MoE at batch=1 is kernel-launch-bound; bs=1 capture needs ~1 GB, took 1.6 s —
  the CI's `--disable-cuda-graph` is only needed for the default bs≤256 capture list).
- EAGLE3's end-to-end gain (1.17–1.23×) is diluted vs its accept length because (a) the graph-optimized
  baseline is already fast per token, (b) EAGLE3 draft steps run per decode step, and (c) ASR outputs are
  short (~50 tok) so audio-encode+prefill is a fixed cost per request.
- Absolute quality: WER 1.68–1.70% on test-clean.

Per-run data: `results/omni_libri_test_g1_{baseline,ttt5,ttt7}/`. Servers: `outputs/serve_omni_g1_*.log`.

## Round 3 — num-steps matched to training TTT depth (tp=1)

| config | steps/tokens | accept_len | WER | mean lat | utt/s |
|---|---|---|---|---|---|
| ttt5 | 5/6 | 2.31 (↑ vs 2.23) | 1.68% | 0.179 s | 5.5 (↓ vs 5.9) |
| ttt7 | 7/8 | 2.53 (↑ vs 2.38) | 1.68% | 0.180 s | 5.5 (↓ vs 6.2) |

**Deeper steps raise accept_length but LOWER end-to-end speed**: the extra 2–4 draft
forwards per decode step cost more than the ~0.15 extra accepted tokens pay back
(only ~2.5 of 8 drafted tokens accepted). **steps=3/tokens=4 is optimal here.**

## Round 4 — tp=2 (GPUs 0,1; steps=3 per the decision rule)

| config | accept_len | WER | mean lat | utt/s | speedup (vs tp2 baseline) |
|---|---|---|---|---|---|
| baseline | – | 1.68% | 0.181 s | 5.5 | 1.00× |
| ttt5 | 2.23 | 1.68% | 0.155 s | 6.4 | 1.17× |
| **ttt7** | 2.38 | 1.68% | **0.150 s** | **6.6** | **1.21×** |

- tp=2 baseline is only ~8% faster than tp=1 baseline (0.181 vs 0.195 s): batch=1 decode is
  latency-bound and per-layer NCCL sync eats most of the 2× FLOPs.
- accept_length identical across tp (2.23/2.38) — as expected, it's an algorithm property.
- EAGLE3's relative gain carries over to tp=2 (1.17×/1.21×).

**Deployment guidance:** best absolute latency = tp=2 + ttt7 + steps3 (0.150 s/utt, 6.6 utt/s);
best per-GPU efficiency = **tp=1 + ttt7 + steps3** (6.2 utt/s on ONE GPU ≈ 1.9× the per-GPU
throughput of tp=2). All configs lossless (WER 1.68%).

## Probe — train-split vs test-split accept (is the train-metric gap memorization?)

ttt7 (steps=3) decoding 300 utts of **train.100** (each seen ~37× during training):
**accept_len 2.27** vs test-clean 2.38 — identical within noise (train WER 0.70% — the
*target* is more accurate in-domain, but the *draft*'s acceptance doesn't move).

**Conclusion:** the training-time acceptance_rate ~0.97 vs serving accept ~2.4/4 gap is
NOT memorization/generalization. It is structural:
1. training metrics are teacher-forced (ground-truth prefix at every TTT step);
   serving accept requires *consecutive* correctness — accept_len ≈ 1 + p + p² + p³,
   so per-token p≈0.62 yields 2.38 while teacher-forced per-position ≈0.97;
2. deeper drafted tokens condition on the draft's own outputs (exposure bias TTT
   only partially removes);
3. metric units differ (per-step teacher-forced rate vs tokens-per-verify incl. bonus).
Implication: more epochs/data on the SAME distribution won't lift accept much; levers
are draft capacity (layers), deeper TTT (ttt7>ttt5 confirms), scheduled-sampling-style
self-feeding during training.

## Round 5 — long-form audio (earnings21, 41-min clip): can it run + does EAGLE3 pay off?

Clip: `hf-audio/asr-leaderboard-longform` earnings21/test idx=1 — 41.0 min, 6602 ref words.
Fed as ONE audio: processor does NOT truncate (mel 128×245,887 = full 2459 s;
**31,965 audio tokens ≈ 13 tok/s**). Total sequence ≈ 32k prompt + 7k generation ≈ 39k
of the 65536 context. Audio passed as a local file path in audio_url (server-side
load_audio reads it; avoids a ~105 MB base64 payload). Spec server needs
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` (draft config max_pos 8192).

| | baseline | EAGLE3 ttt7 (steps=3) |
|---|---|---|
| wall time | 531.9 s | 575.8 s (**0.92× — slower**) |
| completion tokens | 7092 | 7092 (byte-identical output, WER 9.19%) |
| steady decode speed | **~14.5 tok/s** | – |
| accept_len | – | **1.166** (vs 2.38 on LibriSpeech test) |
| accept vs position | – | flat: first-third 1.13 → last-third 1.20 |

Findings:
1. **Qwen3-Omni handles 41-min single-shot audio** end-to-end (transcript head/tail correct,
   WER 9.19% on spontaneous conference speech).
2. **Long context crushes decode speed ~10×**: ~14.5 tok/s at ~40k context vs ~140 tok/s on
   short clips — each step re-reads weights + ~4 GB KV. Decode is ~90% of wall (≈470/532 s).
   This IS the spec-decode-favorable regime: with accept 2.38 the wall would be ≈280 s (≈1.9×).
3. **But accept COLLAPSES out-of-domain: 2.38 → 1.17.** LibriSpeech-trained draft doesn't
   transfer to spontaneous financial-call speech — same pattern (and nearly same number) as
   the AISHELL DFlash draft on SpeechIO (2.39→1.16). Net: 8% slower than baseline.
4. **Rope extrapolation is NOT the problem**: accept is flat out to ~39k positions (40× the
   draft's training lengths). Long-form drafts don't need positional retraining — they need
   in-domain DATA (e.g. earnings21-train / GigaSpeech / SPGISpeech in the label-regen + training mix).

Artifacts: `outputs/longform/` (clip, GT, result_{baseline,ttt7}.json, serve logs).
