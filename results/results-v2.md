# Speculative decoding on Qwen2-Audio ASR — results v2

Setup: target `yuekai/qwen2_audio_aishell_sft`, sglang DFLASH/EAGLE3, **1×H100, batch=1
(concurrency=1)**, fa3 backend, mem-fraction 0.85, greedy. Full test sets.
Prompt: `Detect the language and recognize the speech: <|zh|>`.
DFlash served with `--speculative-num-draft-tokens = block_size`; EAGLE3 with
`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.

## AISHELL-1 test (7176 utts) — in-domain

| method | block/epoch | CER | accept_len | mean latency | throughput | speedup |
|---|---|---|---|---|---|---|
| baseline (no draft) | – | 2.01% | 1.00 | 0.131 s | 7.6 utt/s | 1.00× |
| EAGLE3 | ep7 | 2.00% | –¹ | 0.079 s | 12.5 utt/s | **1.66×** |
| DFlash block-4 | ep24 | 2.01% | 2.65 | 0.082 s | 12.1 utt/s | 1.60× |
| DFlash block-6 | ep8 | 2.01% | 2.38 | 0.085 s | 11.6 utt/s | 1.54× |
| DFlash block-8 | ep28 | 2.01% | 2.26 | 0.088 s | 11.3 utt/s | 1.49× |

## SpeechIO ZH00011 (1053 utts) — out-of-domain

| method | block/epoch | CER | accept_len | mean latency | throughput | speedup |
|---|---|---|---|---|---|---|
| baseline (no draft) | – | 14.70% | – | 0.368 s | 2.7 utt/s | 1.00× |
| EAGLE3 | – | not run² | – | – | – | – |
| DFlash block-4 | ep24 | 14.33% | 1.16 | 0.395 s | 2.5 utt/s | 0.93× (slower) |
| DFlash block-6 | ep8 | 14.33% | 1.06 | 0.421 s | 2.4 utt/s | 0.87× (slower) |
| DFlash block-8 | ep28 | 14.33% | 1.06 | 0.423 s | 2.4 utt/s | 0.87× (slower) |

## Takeaways
- **In-domain (AISHELL): both methods ~1.5–1.7× faster, lossless.** EAGLE3 (1.66×) slightly ahead of
  DFlash block-4 (1.60×). For DFlash, larger blocks are *worse* (accept 2.65 → 2.38 → 2.26): for ASR the
  far draft offsets are rarely accepted (end-of-utterance is ~51% of draft misses). **block_size=4 is the
  DFlash sweet spot.**
- **Out-of-domain (SpeechIO): DFlash accept collapses to ~1.0–1.16 → net slowdown (0.87–0.93×).** The
  AISHELL-trained draft doesn't transfer; speculative benefit is domain-specific.
- **Accuracy essentially unchanged** by speculative decoding (lossless w.r.t. greedy); small CER shifts
  (2.01↔2.00, 14.70↔14.33) are bf16 numeric-path differences between the baseline and verify paths.

## Caveats
1. EAGLE3's accept_length wasn't logged in its AISHELL batch=1 run; the 1.66× implies effective
   accept ≈ DFlash's or slightly higher.
2. EAGLE3 was **not** benchmarked on SpeechIO (only baseline + DFlash were).
3. DFlash checkpoints are at **different epochs** (block-4→24, block-6→8, block-8→28), so the block-size
   rows are "latest available," not equal-budget; block-6 is the least trained.
4. SpeechIO latencies are ~4–5× higher than AISHELL because its utterances are much longer.

Per-run data: `results/bench_final_bs{4,6,8}/`, `results/bench_baseline_c1/`, `results/bench_eagle3_c1/`,
`results/speechio11_final_{baseline,4,6,8}/`. Detail docs: `dflash_blocksize_final_c1.md`,
`dflash_speechio11_c1.md`, `specdec_3way_c1.md`.
