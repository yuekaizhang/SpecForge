# DFlash block-size comparison — AISHELL test, 1×H100, batch=1

Same target, fa3, mem-fraction 0.85, concurrency/batch=1, full test (7176 utts).

| config | train epoch | accept_len | CER | mean lat | utt/s | speedup vs baseline |
|---|---|---|---|---|---|---|
| baseline (no draft) | – | 1.00 | 2.01% | 0.131 s | 7.6 | 1.00× |
| DFlash **block-4** | epoch 3 | **2.39** | 2.01% | 0.087 s | 11.4 | **1.51×** |
| DFlash **block-6** | epoch 1 | **2.00** | 2.01% | 0.096 s | 10.3 | **1.36×** |
| EAGLE3 (ref) | epoch 7 | – | 2.00% | 0.079 s | 12.5 | 1.66× |

**Accuracy identical (~2.0% CER)** everywhere — lossless.

**Block-6 (epoch 1) is currently WORSE than block-4 (epoch 3) — but the comparison is not yet fair:**
- Block-6 is only **epoch 1** vs block-4's epoch 3 → much less trained → lower accept rate.
- Block-6 drafts 6 tokens/step vs 4 → more draft+verify overhead. At epoch 1 the far offsets (+4,+5) are barely trained, so they're drafted but almost never accepted → wasted compute → both lower accept_len (2.0) AND higher latency (0.096 vs 0.087).

**Conclusion:** can't judge block size yet. Block-6 needs to train to at least comparable maturity (epoch ~3+) before comparing. It *can* accept more tokens in principle (6 slots), but only once offsets +4/+5 are trained. Resume block-6 training and re-benchmark at matched epochs.
