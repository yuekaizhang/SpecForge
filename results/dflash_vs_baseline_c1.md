# DFlash vs baseline — AISHELL test, 1 GPU (H100), concurrency/batch = 1

Full AISHELL test set (7176 utts), sglang, Qwen2-Audio SFT target, fa3 backend, mem-fraction 0.85.
DFlash draft = `outputs/qwen2-audio-sft-dflash-fixedhead/epoch_3_step_42000` (only 3/50 epochs trained).

| metric | baseline (no draft) | DFlash draft | speedup |
|---|---|---|---|
| CER | 2.01% | 2.01% | same (lossless) |
| Exact match | 5828/7176 (81%) | 5827/7176 (81%) | same (±1, bf16 tie) |
| Mean utt latency | 0.131 s | 0.087 s | **1.51×** |
| Overall RTF (wall/audio) | 0.0262 | 0.0174 | **1.51×** |
| Mean utt RTF | 0.0266 | 0.0180 | 1.48× |
| Throughput | 7.6 utt/s | 11.4 utt/s | **1.50×** |
| Wall time (7176 utts) | 944.5 s | 628.0 s | 1.50× |

- **Accuracy is identical** (2.01% CER) — speculative decoding is lossless w.r.t. greedy; the ±1 exact-match is a bf16 tie edge case.
- **~1.5× latency speedup at batch=1.** Accept length ≈ 2.3 (measured earlier) / 2.54 (implied by per-offset analysis); end-to-end speedup is lower than accept length because (a) ASR outputs are short so prefill/audio-encode dominates, and (b) draft-forward + verify overhead.
- Draft trained only **3/50 epochs** — more training should raise accept length and the speedup.

Repro:
```
# DFlash server
python -m sglang.launch_server --model-path yuekai/qwen2_audio_aishell_sft --served-model-name qwen2audio-sft \
  --trust-remote-code --port 31040 --mem-fraction-static 0.85 --dtype bfloat16 --attention-backend fa3 \
  --speculative-algorithm DFLASH --speculative-draft-model-path <ckpt> \
  --speculative-num-draft-tokens 4 --speculative-num-steps 1 --speculative-eagle-topk 1
# baseline server: same, drop the --speculative-* flags
# both: python scripts/decode_sglang.py --server http://127.0.0.1:31040 --model qwen2audio-sft \
#   --dataset carlot/AIShell --split test --concurrency 1 --num-samples 0 --output-dir results/bench_*
```
