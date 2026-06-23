# Qwen2-Audio EAGLE3 Draft Training (SpecForge + AISHELL)

Date: 2026-06-23
Status: Approved (design)

## Goal

Train an EAGLE3 draft model for `Qwen/Qwen2-Audio-7B-Instruct` on the
`carlot/AIShell` ASR dataset (train split), online mode, starting with a small
subset to get the pipeline working end-to-end. The trained draft must load in
the already-validated sglang dev-HEAD inference path (with the `qwen2_audio`
EAGLE3 shim) and yield acceptance length > 1 (better than a random draft).

This extends SpecForge's vision-language (Qwen2.5-VL) training path by analogy
to audio. The key data flow insight: EAGLE3 drafts are text-token predictors
conditioned on the target's (audio-aware) hidden states. Audio is consumed only
by the TARGET; the draft has no audio encoder. See memory
`qwen2audio-eagle3-sglang-validated`.

## Key simplifying fact

Qwen2-Audio's text backbone is a standard Qwen2 LM using **standard 1D RoPE,
not M-RoPE**. So the draft config needs no `mrope_section`, and `position_ids`
is a plain `arange`. This makes the audio path simpler than the VLM path.

## Decisions (locked)

- **Implementation route:** dedicated SpecForge env + extend SpecForge (not a
  self-contained trainer in the current env). SpecForge pins
  torch==2.9.1 / transformers==4.57.1 / sglang==0.5.9, which conflict with the
  current nemo_rl venv (torch 2.11 / transformers 5.8.1) and the dev-HEAD sglang
  used for inference. So training runs in an isolated env; inference stays in the
  validated dev-HEAD env. The draft checkpoint format
  (`LlamaForCausalLMEagle3`) is stable across sglang versions, so it bridges the
  two envs.
- **Hidden states:** online (computed on the fly from the target each step),
  mirroring `examples/run_qwen2.5_7b_vl_eagle3_online.sh`.
- **Data scale (first cut):** small subset of the train split (default 3000
  utterances) to validate end-to-end before scaling.
- **Target backend:** HF/custom (transformers ships Qwen2-Audio); does not rely
  on sglang to run the target.

## Architecture / components

Extends SpecForge by analogy to the Qwen2.5-VL path. Audio-specific code is
isolated in new functions/classes so the diff to the core training script stays
small and reviewable.

1. **Environment** — new isolated venv `specforge_env` via `pip install -e
   SpecForge`. One-time setup; documented in the example script / a setup note.

2. **Data prep** — `scripts/prepare_aishell.py`:
   - Loads `carlot/AIShell` train split (subset, default N=3000).
   - Decodes audio with **soundfile** from raw bytes (`Audio(decode=False)` ->
     bytes -> 16 kHz mono numpy), explicitly avoiding `torchcodec` (which fails
     to load its shared lib in these envs).
   - Emits examples `{audio_array, sampling_rate, transcription}` for the
     preprocessing step.

3. **Preprocessing** — `preprocess_audio_conversations()` in
   `specforge/data/preprocessing.py` (analogous to `preprocess_vlm_conversations`):
   - Builds a Qwen2-Audio chat: user = `[{audio}, {text: "请将这段音频转写为文本。"}]`,
     assistant = transcription.
   - Calls the Qwen2-Audio `AutoProcessor` -> `input_ids`, `input_features`
     (mel `[128, 3000]`, fixed 30 s), `feature_attention_mask`.
   - Builds `loss_mask` over the assistant (transcription) span only, using the
     processor offset mapping (same offset-based approach as the VLM path).

4. **Collation** — `AudioDataCollatorWithPadding` in `specforge/data/utils.py`:
   pads `input_ids` / `loss_mask` / `attention_mask`; stacks the fixed-size
   `input_features` and `feature_attention_mask` along the batch dim.

5. **EAGLE3 training model** — `QwenAudioOnlineEagle3Model` in
   `specforge/core/eagle3.py` (analogous to `QwenVLOnlineEagle3Model`):
   - `_prepare_data`: runs `Qwen2AudioForConditionalGeneration(..., input_features,
     feature_attention_mask, output_hidden_states=True)`; takes the 3 configured
     aux layers from `outputs.hidden_states` -> `cat` to `(B, seq, 3H)`; uses
     `outputs.logits` as the target distribution.
   - `_get_input_embeds`: draft text embedding via `draft.embed_input_ids`, then
     `masked_scatter` the target's audio features (audio_tower +
     multi_modal_projector) into the audio-placeholder token positions — the
     audio analog of the VLM image-embed fusion, keeping EAGLE inputs faithful.
   - `position_ids`: plain `arange` (no M-RoPE).
   - Reuses SpecForge's existing TTT multi-step loss, BF16 optimizer, and
     checkpoint saving.

6. **Draft config** — `configs/qwen2-audio-7b-eagle3.json`:
   `architectures: ["LlamaForCausalLMEagle3"]`, `target_model_type:
   "qwen2_audio"`, `num_hidden_layers: 1`, dims from the Qwen2-Audio LM
   (hidden 4096, 32 heads, 32 kv heads, head_dim 128, intermediate 11008,
   vocab 156032, draft_vocab_size 32000), **standard rope (no mrope_section)**.

7. **Training script** — add an `--is-audio` branch to
   `scripts/train_eagle3.py`, parallel to the `--is-vlm` branches (processor
   load, dataset build, collator select, eagle3-model select, forward call).
   Plus `examples/run_qwen2_audio_eagle3_online.sh` with small-subset hyperparams
   (batch-size 1, ttt-length 7, lr 1e-4, a few hundred steps).

## Data flow

```
AISHELL {audio, transcription}
  -> prepare_aishell.py (soundfile decode, 16 kHz)
  -> preprocess_audio_conversations (Qwen2-Audio processor)
       -> input_ids, input_features, feature_attention_mask, loss_mask
  -> AudioDataCollatorWithPadding (batch)
  -> QwenAudioOnlineEagle3Model
       _prepare_data:  target forward (output_hidden_states) -> 3 aux layers + logits
       _get_input_embeds: draft text embeds + audio feats scattered at audio tokens
       TTT loop (reused): K-step draft unroll, KL loss vs target on draft vocab
  -> draft checkpoint (LlamaForCausalLMEagle3 format)
  -> [cross-env] sglang dev-HEAD + qwen2_audio shim -> two-stage validation
       expect accept length > 1 on held-out AISHELL clip
```

## Testing / validation

- 1-batch smoke test of `_prepare_data` to confirm the target's
  `output_hidden_states` layer extraction works under transformers 4.57.1 for
  Qwen2-Audio (the main integration risk).
- End-to-end small-subset training run: loss decreases, draft checkpoint saved.
- Cross-env inference loop-back: load the trained draft in the validated
  sglang dev-HEAD path, run the two-stage validation
  (`qwen2audio_specdec/run_validation.sh`), confirm output is lossless-identical
  to baseline AND acceptance length > 1 (vs ~0 for the random draft).

## Risks / trade-offs

- Requires a second (isolated) env. Mitigated by the stable cross-version draft
  format bridging train (SpecForge env) and inference (dev-HEAD env).
- Need to confirm Qwen2-Audio `output_hidden_states` layer extraction under
  transformers 4.57.1 — covered by the 1-batch smoke test before full training.
- Qwen2-Audio processor fixes audio to mel `[128, 3000]` (30 s); longer audio is
  truncated. AISHELL utterances are short, so this is not a concern.

## Out of scope (first cut)

- Offline (pre-cached) hidden states.
- Full train split / large-scale training and acceptance-rate tuning.
- Qwen3-ASR / Qwen-Omni (the same shim + analogous training would follow once
  this path is proven).
