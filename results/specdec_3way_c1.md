# Speculative decoding on Qwen2-Audio — 3-way, AISHELL test, 1×H100, batch=1

Full AISHELL test (7176 utts), sglang, same SFT target (`yuekai/qwen2_audio_aishell_sft`),
fa3 backend, mem-fraction 0.85, concurrency/batch = 1.

| config | CER | exact | mean lat (s) | overall RTF | utt/s | wall (s) | speedup |
|---|---|---|---|---|---|---|---|
| baseline (no draft) | 2.01% | 5828/7176 | 0.131 | 0.0262 | 7.6 | 944.5 | 1.00× |
| **EAGLE3** (epoch_7_step_118000) | 2.00% | 5830/7176 | 0.079 | 0.0158 | 12.5 | 572.2 | **1.66×** |
| **DFlash** (epoch_3_step_42000) | 2.01% | 5827/7176 | 0.087 | 0.0174 | 11.4 | 628.0 | **1.51×** |

- **Accuracy identical (~2.0% CER) across all three** — both spec-decode methods are lossless; ±1–2 exact-match差异 is bf16 argmax tie noise.
- **EAGLE3 1.66× vs DFlash 1.51×** latency speedup at batch=1.
- **Not fully apples-to-apples on training:** EAGLE3 draft is fully trained (epoch 7, num-steps 3 chain), DFlash draft is only **epoch 3 of 50** (block_size 4). DFlash should close/overtake the gap with more training (accept length was still climbing).
- Serving args:
  - EAGLE3: `--speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
  - DFlash: `--speculative-algorithm DFLASH --speculative-num-draft-tokens 4 --speculative-num-steps 1 --speculative-eagle-topk 1`
- Per-utterance results under `results/bench_{baseline,eagle3,dflash}_c1/`.
