# Qwen2-Audio EAGLE3 Speculative Decoding — Results v1

## Setup

| Component | Value |
|-----------|-------|
| Target model | `yuekai/qwen2_audio_aishell_sft` (Qwen2-Audio-7B fine-tuned on AISHELL GT) |
| Draft model | `LlamaForCausalLMEagle3`, 1 layer, hidden 4096, draft_vocab 32000 |
| Draft checkpoint | `outputs/qwen2-audio-sft-eagle3/epoch_7_step_118000` (~7.8 epochs) |
| Training data | `carlot/AIShell` train split (~120k utterances, space-stripped Mandarin) |
| Training config | LR=1e-5, warmup_ratio=0.003, ttt_length=5, batch=1/GPU×8 DP, 10 epochs |
| Instruction prompt | `Detect the language and recognize the speech: <|zh|>` |
| Inference engine | sglang dev-HEAD with `qwen2_audio.py` EAGLE3 shim |
| Inference GPUs | 2× H100 (TP=2) |
| Client concurrency | 16 |
| Eval dataset | AISHELL test set (7176 utterances) |

### Key design choices
- **SFT target model**: the target was fine-tuned on AISHELL GT, so its output exactly matches the GT transcription (100% exact match verified on train split). This eliminates the train/inference distribution mismatch that plagued the original `Qwen2-Audio-7B-Instruct` target (which added wrappers like `这段音频的内容是：'...'`, extra punctuation, and different tokenization — only 0.2% GT match).
- **Space-stripped transcriptions**: AISHELL GT has word-segmentation spaces (`而 对 楼市`); we strip them to `而对楼市` to match natural Chinese text.
- **Standard 1D RoPE**: Qwen2-Audio uses standard RoPE (not M-RoPE like Qwen2.5-VL), simplifying the draft config.

---

## 1. Decoding Benchmark (AISHELL test, 7176 utterances)

### EAGLE3 vs Baseline — topk=1

| Metric | Baseline (no draft) | EAGLE3 (topk=1) | Delta |
|--------|-------------------|-----------------|-------|
| **CER** | 2.00% | 2.00% | **lossless** ✓ |
| **Exact match** | 5830/7176 (81%) | 5831/7176 (81%) | identical |
| **Wall time** | 157.1s | **131.1s** | **-16.5%** |
| **Overall RTF** | 0.0044 | **0.0038** | -14% |
| **Mean utt latency** | 0.336s | **0.278s** | -17% |
| **Mean utt RTF** | 0.0708 | **0.0610** | -14% |
| **Throughput** | 45.7 utt/s | **54.7 utt/s** | **+20%** |
| **Accept length** | — | **3.33** | — |

### topk sweep (num_steps=5, same draft)

| topk | num_draft_tokens | accept_len | wall_time | throughput | CER |
|------|-----------------|------------|-----------|------------|-----|
| **1** | 6 | 3.33 | **131.1s** | **54.7 utt/s** | 2.00% |
| 2 | 12 | 3.82 | 149.9s | 47.9 utt/s | 2.02% |
| 4 | 16 | 3.95 | 144.1s | 49.8 utt/s | 2.01% |
| 8 | 24 | 4.14 | 145.0s | 49.5 utt/s | 2.01% |
| baseline | — | — | 157.1s | 45.7 utt/s | 2.00% |

**Conclusion**: topk=1 (chain) is fastest end-to-end for this ASR task. Higher topk increases accept_len (3.33→4.14) but the tree verification overhead outweighs the gain — short-output ASR (5-15 decode tokens per utterance) doesn't benefit from wide trees.

---

## 2. Earlier draft experiments (for context)

### Original target (`Qwen2-Audio-7B-Instruct`, NOT SFT)

The original Instruct target has a massive train/inference distribution mismatch:

| Metric | GT (training labels) | Target free-running output |
|--------|---------------------|---------------------------|
| Format | Raw transcription: `而对楼市成交...` | Wrapped: `这段音频的内容是：'而对面楼市成交...。'` |
| GT match | — | **0.2%** (virtually zero) |
| Tokenization | `对` (single char) | `对面` (merged token) |
| Punctuation | None | Adds `。` |

This mismatch limited the best Instruct-target draft to **accept_len=1.46** (topk=1). Switching to the SFT target (100% GT match) raised it to **3.33** — a **2.3× improvement** from distribution alignment alone.

| Draft experiment | Target | accept_len (topk=1) | CER |
|-----------------|--------|---------------------|-----|
| Random init | Instruct | 1.00 | 2.00% |
| 300-step proof | Instruct | 1.03 | 2.00% |
| epoch_0 (spaced GT, LR 2e-4) | Instruct | 1.15 | 2.00% |
| epoch_2.8 (clean GT, LR 5e-5) | Instruct | **1.46** | 2.00% |
| **epoch_7 (clean GT, LR 1e-5)** | **SFT** | **3.33** | **2.00%** |

---

## 3. Draft failure analysis (100 AISHELL test clips)

Teacher-forced analysis: at each transcription position, compare draft top-1 vs target top-1.

### Overall stats

| Metric | Value |
|--------|-------|
| Total transcription positions | 841 |
| Draft top-1 matches target | 475 (**56.5%**) |
| Failures | 366 (43.5%) |

### Failure categories

| Category | Count | % of failures | Description |
|----------|-------|---------------|-------------|
| **in_topk_not_top1** | 242 | **66%** | Target token IS in draft's top-5 but not rank 1 |
| **confident_wrong** | 111 | **30%** | Draft >90% confident on a WRONG token |
| **uncertain** | 13 | **4%** | Draft uncertain and wrong |

### Dominant failure pattern: token offset (confident_wrong)

The draft "sees" the correct semantic structure but **skips one token ahead**:

```
GT: 但[因为]聚集了过多公共资源
Draft: 但 → 聚集(1.00)     actual: 因为    ← skipped "因为"

GT: 为了[规避]三四线城市...
Draft: 为了 → 三四(1.00)    actual: 规避    ← skipped "规避"

GT: 一线城市[土地]供应量减少
Draft: 一线城市 → 供应(1.00) actual: 土地    ← skipped "土地"

GT: 因此[土地储备]至关重要
Draft: 土地 → 至关重要(0.99) actual: 储备    ← skipped "储备"
```

The 1-layer draft predicts the right **phrase** but at the wrong **token boundary** — a fundamental capacity limitation of a single transformer layer tracking multi-token sequences.

### in_topk_not_top1 (66% of failures)

```
GT: 甚至[出现]交易几乎停滞...
Draft top-5: 交易(0.72) | →出现(0.28)     ← correct answer at rank 2

GT: 三四[线]城市...
Draft top-5: 城市(0.62) | →线(0.37)       ← correct answer at rank 2
```

The correct token is often rank 2-3 with meaningful probability. A topk≥2 tree would accept many of these, but the tree verification overhead makes it net-slower for short ASR outputs.

### Position bias

Failures concentrate at **position 0-1** (start of transcription, least context). Mid-to-end positions have much higher accuracy — the draft is strong once it has sufficient context.

---

## 4. Bottleneck analysis & next steps

### Current bottleneck: 1-layer draft capacity

The draft's 56.5% teacher-forced accuracy is the ceiling for accept length. The dominant failure mode (token offset, 30%) is a capacity issue — 1 transformer layer can't reliably track multi-token granularity.

### Potential improvements (not yet tried)

| Lever | Expected effect | Trade-off |
|-------|----------------|-----------|
| **More draft layers (1→2)** | Higher accuracy → higher accept_len | Slower draft inference (more FLOPs per step) |
| **More training epochs** | Diminishing returns (ep3→ep7 showed no gain: 3.46→3.33) | Compute cost |
| **topk>1 tree decoding** | Higher accept_len but slower e2e for short outputs | Only helps for longer outputs |
| **Larger draft_vocab** | Marginal (current 32k covers 100% of tokens) | Memory |

The most promising next step is **2-layer draft** — directly addresses the token-offset capacity limitation while keeping draft inference fast (2-layer is still very lightweight vs the 32-layer target).

---

## 5. File index

```
results/
  sft_eagle3_ep7_test/       — EAGLE3 (topk=1) full AISHELL test decode
    results.jsonl             (7176 per-utterance records)
    errors.txt                (CER>0 utterances)
    rtf.txt                   (RTF statistics)
    summary.txt               (overall summary)
  sft_baseline_ep7_test/     — Baseline (no draft) full AISHELL test decode
    (same structure)
  sft_eagle3_topkN_test/     — topk=1,2,4,8 sweep results (N=1,2,4,8)
  debug_failures_ep7/        — Per-position failure analysis (100 test clips)
    failures.jsonl            (366 structured failure records)
    failures.txt              (human-readable failure details)
    stats.txt                 (failure category statistics)

SpecForge/scripts/
  decode_sglang.py            — ASR decode benchmark (sglang server, with/without draft)
  debug_eagle3_decode.py      — Offline teacher-forced per-position debug
  debug_eagle3_failures.py    — Detailed failure analysis with categories
  regenerate_labels.py        — Target-model label regeneration (multi-server)
```

---

## 6. Out-of-domain evaluation: Speechio 00011

Speechio is a multi-domain Chinese ASR benchmark with longer utterances than AISHELL. Subset 00011 has the longest average sentences — a better test for speculative decoding acceleration since longer outputs mean more decode steps.

### Dataset stats

| Property | AISHELL test | Speechio 00011 |
|----------|-------------|----------------|
| Utterances | 7176 | 1053 |
| Mean duration | ~5s | **11.8s** |
| Max duration | ~15s | **22.2s** |
| Mean text (chars) | ~13 | **50** |
| Domain | Read speech | Law / news (OOD for AISHELL-SFT model) |

### EAGLE3 vs Baseline — Speechio 00011 (topk=1)

| Metric | Baseline | EAGLE3 | Delta |
|--------|----------|--------|-------|
| **CER** | 14.60% | **14.51%** | **lossless** ✓ |
| **Exact match** | 267/1053 (25%) | 270/1053 (26%) | ~identical |
| **Wall time** | 38.9s | **34.0s** | **-12.6%** |
| **Overall RTF** | 0.0031 | **0.0027** | -13% |
| **Mean utt latency** | 0.554s | **0.479s** | **-13.5%** |
| **Throughput** | 27.0 utt/s | **31.0 utt/s** | **+15%** |
| **Accept length** | — | **2.30** | — |

### Cross-dataset comparison

| Dataset | Domain | Mean utt len | Accept len | Wall speedup | CER delta |
|---------|--------|-------------|------------|-------------|-----------|
| AISHELL test | In-domain (train=AISHELL) | ~5s / 13 chars | **3.33** | **-16.5%** | 0% (lossless) |
| Speechio 00011 | **OOD** (law/news) | 11.8s / 50 chars | **2.30** | **-12.6%** | 0% (lossless) |

### Analysis

- **CER lossless on OOD data** — speculative decoding does not degrade accuracy even on out-of-domain inputs (14.51% vs 14.60% is within noise).
- **Accept length drops on OOD** (3.33→2.30) — the draft was trained on AISHELL, so its predictions are weaker on unseen law/news domain. This is expected: the draft's vocabulary patterns (trained on read speech) don't fully transfer to spontaneous legal discourse.
- **Speedup still meaningful** (~13-15%) — longer utterances provide more decode steps where the draft can contribute, partially compensating for the lower accept rate.
- **Higher CER (14.5% vs 2.0%)** is the target model's OOD generalization gap, not the draft's fault — the SFT model was fine-tuned only on AISHELL.
- **Improving OOD accept length** would require training the draft on multi-domain data (not just AISHELL), or using a target model that generalizes better across domains.

### Results files

```
results/
  speechio11_eagle3/     — EAGLE3 decode on Speechio 00011
  speechio11_baseline/   — Baseline decode on Speechio 00011
```
