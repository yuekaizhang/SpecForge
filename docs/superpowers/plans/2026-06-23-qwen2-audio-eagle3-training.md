# Qwen2-Audio EAGLE3 Draft Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add audio-modality EAGLE3 draft training to SpecForge and train a draft for `Qwen/Qwen2-Audio-7B-Instruct` on a small subset of `carlot/AIShell` (online mode), producing a checkpoint that loads in the validated dev-HEAD sglang inference path.

**Architecture:** Extend SpecForge's Qwen2.5-VL path by analogy to audio. Audio-specific code is isolated in new functions/classes (`preprocess_audio_conversations`, `AudioDataCollatorWithPadding`, `QwenAudioOnlineEagle3Model`) plus an `--is-audio` branch threaded through `scripts/train_eagle3.py` parallel to `--is-vlm`. Qwen2-Audio's LM uses standard 1D RoPE (not M-RoPE), so the draft config has no `mrope_section` and `position_ids` is a plain cumsum — simpler than the VLM path.

**Tech Stack:** Python, PyTorch (bf16), HuggingFace transformers (Qwen2-Audio), SpecForge (EAGLE3 draft + TTT loss), `datasets`, `soundfile`. Training runs in an isolated env with SpecForge's pinned deps (torch 2.9.1 / transformers 4.57.1 / sglang 0.5.9).

**ENV/API FACTS established by Task 0 (apply to all later tasks):**
- The SpecForge env is `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/.specforge_env`, built on **Python 3.12** (3.13 lacks an `outlines_core==0.1.26` wheel). Activate: `source .specforge_env/bin/activate`. Versions confirmed: transformers 4.57.1 / torch 2.9.1+cu128 / sglang 0.5.9.
- Qwen2-Audio processor kwarg is **`audio=`** (singular), NOT `audios=`. Wrong kwarg is silently ignored → no `input_features`.
- `out.hidden_states` is a flat tuple of 33 (embedding + 32 LM layers), each `[1, seq, 4096]` — slice aux layers directly.
- `model.config.audio_token_index == 151646`.
- Always `export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/.hf_cache` (Qwen2-Audio-7B already cached).

**Spec:** `docs/superpowers/specs/2026-06-23-qwen2-audio-eagle3-training-design.md`

**Repo root for all paths below:** `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/SpecForge`

---

## File Structure

- Create `configs/qwen2-audio-7b-eagle3.json` — draft config (Qwen2-Audio LM dims, standard rope).
- Create `scripts/prepare_aishell.py` — download/decode AISHELL subset to a local arrow/jsonl with `{audio, transcription}`.
- Modify `specforge/data/preprocessing.py` — add `preprocess_audio_conversations()`; add `is_audio` branch in `build_eagle3_dataset`.
- Modify `specforge/data/utils.py` — add `AudioDataCollatorWithPadding`; add `is_audio` to `prepare_dp_dataloaders`.
- Modify `specforge/core/eagle3.py` — add `QwenAudioOnlineEagle3Model`.
- Modify `scripts/train_eagle3.py` — add `--is-audio` arg + parallel branches.
- Create `examples/run_qwen2_audio_eagle3_online.sh` — launch script (small subset).
- Create `scripts/smoke_audio_pipeline.py` — standalone smoke checks used by several tasks.

---

## Task 0: Isolated SpecForge env + Qwen2-Audio hidden-states smoke test

**Files:**
- Create: `scripts/smoke_audio_pipeline.py`

This de-risks the main integration unknown: that Qwen2-Audio under transformers 4.57.1 returns per-layer `hidden_states` we can slice for EAGLE3.

- [ ] **Step 1: Create the isolated env and install SpecForge**

Run:
```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
python -m venv .specforge_env
source .specforge_env/bin/activate
pip install -e SpecForge
```
Expected: install completes; `python -c "import specforge, transformers, torch; print(transformers.__version__, torch.__version__)"` prints `4.57.1 2.9.1` (or the versions SpecForge pins).

- [ ] **Step 2: Write the target hidden-states smoke test**

Create `scripts/smoke_audio_pipeline.py`:
```python
"""Standalone smoke checks for the Qwen2-Audio EAGLE3 audio pipeline.
Run pieces via: python scripts/smoke_audio_pipeline.py <check>
where <check> in {target, preprocess, collate, eagle3}.
"""
import sys
import numpy as np
import torch

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
INSTRUCTION = "请将这段音频转写为文本。"


def _dummy_audio(seconds=2.0, sr=16000):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr


def check_target():
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL)
    model = (
        Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16
        )
        .eval()
        .cuda()
    )
    audio, sr = _dummy_audio()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "x.wav"},
                {"type": "text", "text": INSTRUCTION},
            ],
        },
        {"role": "assistant", "content": "测试文本"},
    ]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=text, audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    hs = out.hidden_states
    print("num hidden_states (embed + layers):", len(hs))
    print("logits shape:", tuple(out.logits.shape))
    print("hidden[1] shape:", tuple(hs[1].shape))
    print("input_features shape:", tuple(inputs["input_features"].shape))
    print("audio_token_index:", model.config.audio_token_index)
    n_layers = len(hs) - 1
    for name, idx in [
        ("low", 1 + 1),
        ("mid", n_layers // 2 - 1 + 1),
        ("last", n_layers - 4 + 1),
    ]:
        assert 0 <= idx < len(hs), (name, idx, len(hs))
    print("OK: aux layer indices valid for", n_layers, "layers")


if __name__ == "__main__":
    {"target": check_target}[sys.argv[1]]()
```

- [ ] **Step 3: Run the target smoke test**

Run: `cd SpecForge && python scripts/smoke_audio_pipeline.py target`
Expected: prints `num hidden_states` (= LM layers + 1, e.g. 33), `logits shape (1, seq, 156032)`, `input_features shape (1, 128, 3000)`, an integer `audio_token_index`, and `OK: aux layer indices valid`. No exception.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_audio_pipeline.py
git commit -m "feat(audio): SpecForge env + Qwen2-Audio hidden-states smoke test"
```

---

## Task 1: Draft config for Qwen2-Audio

**Files:**
- Create: `configs/qwen2-audio-7b-eagle3.json`

- [ ] **Step 1: Write the config**

Create `configs/qwen2-audio-7b-eagle3.json` (dims confirmed from the live config: H=4096, 32 heads, 32 kv heads, head_dim=128, intermediate=11008, vocab=156032; standard rope, no `mrope_section`):
```json
{
  "architectures": ["LlamaForCausalLMEagle3"],
  "model_type": "llama",
  "target_model_type": "qwen2_audio",
  "num_hidden_layers": 1,
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "head_dim": 128,
  "hidden_act": "silu",
  "vocab_size": 156032,
  "draft_vocab_size": 32000,
  "max_position_embeddings": 8192,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "rope_scaling": null,
  "attention_bias": false,
  "tie_word_embeddings": false,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "torch_dtype": "bfloat16",
  "eagle_config": {
    "eagle_aux_hidden_state_layer_ids": [1, 15, 28]
  }
}
```

- [ ] **Step 2: Validate JSON + dims against the live target config**

Run:
```bash
cd SpecForge && python - <<'EOF'
import json
from transformers import AutoConfig
d = json.load(open("configs/qwen2-audio-7b-eagle3.json"))
cfg = AutoConfig.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
t = getattr(cfg, "text_config", cfg)
for k in ["hidden_size", "intermediate_size", "num_attention_heads", "vocab_size"]:
    assert d[k] == getattr(t, k), (k, d[k], getattr(t, k))
assert d["draft_vocab_size"] <= d["vocab_size"]
print("config OK; aux layers", d["eagle_config"]["eagle_aux_hidden_state_layer_ids"])
EOF
```
Expected: prints `config OK; aux layers [1, 15, 28]`. If a dim assert fires, edit the JSON to match the printed target value, then re-run.

- [ ] **Step 3: Commit**

```bash
git add configs/qwen2-audio-7b-eagle3.json
git commit -m "feat(audio): add Qwen2-Audio EAGLE3 draft config"
```

---

## Task 2: AISHELL data-prep script

**Files:**
- Create: `scripts/prepare_aishell.py`

Produces a local HF dataset on disk with columns `audio` (numpy float32 array via a plain dict `{"array","sampling_rate"}`) and `transcription`, decoded with soundfile to avoid torchcodec.

- [ ] **Step 1: Write the prep script**

Create `scripts/prepare_aishell.py`:
```python
"""Prepare a small subset of carlot/AIShell for Qwen2-Audio EAGLE3 training.

Decodes audio with soundfile (avoids torchcodec) to 16 kHz mono float32 and
saves a HF dataset to disk with columns: audio={"array","sampling_rate"}, transcription.
"""
import argparse
import io

import numpy as np
import soundfile as sf
from datasets import Dataset, load_dataset
from datasets.features import Audio


def decode_16k_mono(audio_field):
    # audio_field is {"bytes": ..., "path": ...} because we cast decode=False
    data = audio_field["bytes"]
    arr, sr = sf.read(io.BytesIO(data), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        n = int(round(len(arr) / sr * 16000))
        arr = np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr).astype(
            np.float32
        )
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset("carlot/AIShell", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    for i, ex in enumerate(ds):
        if i >= args.num_samples:
            break
        arr = decode_16k_mono(ex["audio"])
        rows.append({"array": arr, "sampling_rate": 16000, "transcription": ex["transcription"]})

    out = Dataset.from_list(
        [{"audio": {"array": r["array"], "sampling_rate": 16000}, "transcription": r["transcription"]} for r in rows]
    )
    out.save_to_disk(args.out)
    print(f"saved {len(out)} examples to {args.out}")
    print("example transcription:", out[0]["transcription"])
    print("audio len (samples):", len(out[0]["audio"]["array"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on a tiny count to verify decode works**

Run:
```bash
cd SpecForge && HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/.hf_cache \
  python scripts/prepare_aishell.py --num-samples 8 --out /tmp/aishell_smoke
```
Expected: `saved 8 examples to /tmp/aishell_smoke`, a non-empty Chinese transcription, and `audio len (samples)` > 0. No torchcodec error.

- [ ] **Step 3: Commit**

```bash
git add scripts/prepare_aishell.py
git commit -m "feat(audio): AISHELL subset prep (soundfile decode)"
```

---

## Task 3: `preprocess_audio_conversations` + dataset wiring

**Files:**
- Modify: `specforge/data/preprocessing.py` (add function after `preprocess_vlm_conversations`, ~line 294; add `is_audio` branch in `build_eagle3_dataset` near line 356)
- Modify: `scripts/smoke_audio_pipeline.py` (add `check_preprocess`)

- [ ] **Step 1: Add `preprocess_audio_conversations` to `specforge/data/preprocessing.py`**

Insert after `preprocess_vlm_conversations` (before `def build_eagle3_dataset`):
```python
def preprocess_audio_conversations(
    processor,
    examples,
    chat_template: ChatTemplate,
    max_length: int = 2048,
    instruction: str = "请将这段音频转写为文本。",
) -> Dict[str, List[torch.Tensor]]:
    """Preprocess AISHELL-style audio+transcription examples for Qwen2-Audio.

    examples columns:
        - audio: {"array": np.ndarray, "sampling_rate": int}
        - transcription: str
    Returns input_ids, loss_mask, attention_mask, input_features, feature_attention_mask.
    """
    results = {
        "input_ids": [],
        "loss_mask": [],
        "attention_mask": [],
        "input_features": [],
        "feature_attention_mask": [],
    }
    for i, audio in enumerate(examples["audio"]):
        transcription = examples["transcription"][i]
        if not transcription:
            continue
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "audio.wav"},
                    {"type": "text", "text": instruction},
                ],
            },
            {"role": "assistant", "content": transcription},
        ]
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        encoding = processor(
            text=text,
            audio=[audio["array"]],  # NOTE: transformers 4.57.1 uses `audio=` (singular), NOT `audios=`
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"][0]
        offsets = encoding["offset_mapping"][0]
        decoded_conversation = processor.tokenizer.decode(
            input_ids, skip_special_tokens=False
        )
        loss_mask = _apply_loss_mask_from_chat_template(
            decoded_conversation, offsets, chat_template
        )
        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
        results["input_features"].append(encoding["input_features"])
        results["feature_attention_mask"].append(encoding["feature_attention_mask"])
    return results
```

- [ ] **Step 2: Wire `is_audio` into `build_eagle3_dataset`**

In `specforge/data/preprocessing.py`, change the signature of `build_eagle3_dataset` to add `is_audio: Optional[bool] = False` (next to `is_vlm`). Then in `preprocess_function` (the `if is_vlm:` chain near line 358) add a branch FIRST:
```python
    def preprocess_function(examples):
        if is_audio:
            processed = preprocess_audio_conversations(
                processor,
                examples,
                template,
                max_length,
            )
        elif is_vlm:
            processed = preprocess_vlm_conversations(
                processor, examples, template, max_length
            )
        elif is_preformatted:
            ...
```
Also relax the early assert so audio uses the same processor guard:
```python
    if is_vlm or is_audio:
        assert processor is not None, "processor must be provided for is_vlm/is_audio"
```

- [ ] **Step 3: Confirm a Qwen2-Audio-compatible chat template is registered**

Run:
```bash
cd SpecForge && python - <<'EOF'
from specforge.data.template import TEMPLATE_REGISTRY
print(TEMPLATE_REGISTRY.get_all_template_names())
EOF
```
Expected: a list of template names. If a `qwen2-audio` template is NOT present but `qwen` (im_start/im_end based) is, use `--chat-template qwen` in later tasks. If neither has `<|im_start|>assistant` headers, register a `qwen2-audio` template mirroring the existing `qwen`/`qwen2-vl` entry (same `end_of_turn_token="<|im_end|>"`, assistant header `"<|im_start|>assistant\n"`, system prompt `"You are a helpful assistant."`) in `specforge/data/template.py`. Record the chosen template name for Task 6/7.

- [ ] **Step 4: Add `check_preprocess` to `scripts/smoke_audio_pipeline.py`**

Append to the file and to the dispatch dict:
```python
def check_preprocess():
    from datasets import load_from_disk
    from transformers import AutoProcessor
    from specforge.data.preprocessing import preprocess_audio_conversations
    from specforge.data.template import TEMPLATE_REGISTRY

    template = TEMPLATE_REGISTRY.get("qwen")  # or "qwen2-audio" if registered
    processor = AutoProcessor.from_pretrained(MODEL)
    ds = load_from_disk("/tmp/aishell_smoke")
    batch = {"audio": [ds[0]["audio"], ds[1]["audio"]],
             "transcription": [ds[0]["transcription"], ds[1]["transcription"]]}
    out = preprocess_audio_conversations(processor, batch, template, max_length=2048)
    ii = out["input_ids"][0]
    lm = out["loss_mask"][0]
    print("input_ids shape:", tuple(ii.shape), "loss_mask sum:", int(lm.sum()))
    print("input_features shape:", tuple(out["input_features"][0].shape))
    masked = processor.tokenizer.decode(ii[0][lm[0].bool()])
    print("loss-masked (assistant) text:", repr(masked))
    assert lm.sum() > 0, "loss mask is empty — assistant span not detected"
```
And update dispatch: `{"target": check_target, "preprocess": check_preprocess}[sys.argv[1]]()`.

- [ ] **Step 5: Run the preprocess smoke check**

Run: `cd SpecForge && python scripts/smoke_audio_pipeline.py preprocess`
Expected: non-zero `loss_mask sum`, `input_features shape (1, 128, 3000)`, and the printed "loss-masked (assistant) text" equals (approximately) the transcription. If the masked text is empty or includes the prompt, fix the chat-template name in Step 3 and re-run.

- [ ] **Step 6: Commit**

```bash
git add specforge/data/preprocessing.py scripts/smoke_audio_pipeline.py specforge/data/template.py
git commit -m "feat(audio): preprocess_audio_conversations + dataset wiring"
```

---

## Task 4: `AudioDataCollatorWithPadding` + dataloader wiring

**Files:**
- Modify: `specforge/data/utils.py` (add class after `VlmDataCollatorWithPadding` ~line 250; add `is_audio` to `prepare_dp_dataloaders`)
- Modify: `scripts/smoke_audio_pipeline.py` (add `check_collate`)

- [ ] **Step 1: Add `AudioDataCollatorWithPadding` to `specforge/data/utils.py`**

```python
class AudioDataCollatorWithPadding:
    """Pads text fields; stacks fixed-size Qwen2-Audio mel features."""

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        pad = torch.zeros(B, N - n, dtype=intensors.dtype)
        return torch.cat((intensors, pad), dim=1)

    def __call__(self, features):
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch = {
            "input_ids": torch.cat(
                [self.paddingtensor2D(f["input_ids"], max_length) for f in features]
            ),
            "attention_mask": torch.cat(
                [self.paddingtensor2D(f["attention_mask"], max_length) for f in features]
            ),
            "loss_mask": torch.cat(
                [self.paddingtensor2D(f["loss_mask"], max_length) for f in features]
            ),
            "input_features": torch.cat([f["input_features"] for f in features], dim=0),
            "feature_attention_mask": torch.cat(
                [f["feature_attention_mask"] for f in features], dim=0
            ),
            "hidden_state": None,
            "target": None,
        }
        return batch
```

- [ ] **Step 2: Wire `is_audio` into `prepare_dp_dataloaders`**

In `prepare_dp_dataloaders` add param `is_audio: Optional[bool] = False`, and where the collator is chosen (currently `VlmDataCollatorWithPadding if is_vlm else DataCollatorWithPadding`) make it:
```python
    if is_audio:
        collator = AudioDataCollatorWithPadding()
    elif is_vlm:
        collator = VlmDataCollatorWithPadding()
    else:
        collator = DataCollatorWithPadding()
```

- [ ] **Step 3: Add `check_collate` to the smoke script**

```python
def check_collate():
    from datasets import load_from_disk
    from transformers import AutoProcessor
    from specforge.data.preprocessing import preprocess_audio_conversations
    from specforge.data.template import TEMPLATE_REGISTRY
    from specforge.data.utils import AudioDataCollatorWithPadding

    template = TEMPLATE_REGISTRY.get("qwen")
    processor = AutoProcessor.from_pretrained(MODEL)
    ds = load_from_disk("/tmp/aishell_smoke")
    feats = []
    for k in range(2):
        b = {"audio": [ds[k]["audio"]], "transcription": [ds[k]["transcription"]]}
        o = preprocess_audio_conversations(processor, b, template, 2048)
        feats.append({kk: vv[0] for kk, vv in o.items()})
    batch = AudioDataCollatorWithPadding()(feats)
    for k, v in batch.items():
        print(k, None if v is None else tuple(v.shape))
    assert batch["input_features"].shape[0] == 2
```
Add `"collate": check_collate` to the dispatch dict.

- [ ] **Step 4: Run the collate smoke check**

Run: `cd SpecForge && python scripts/smoke_audio_pipeline.py collate`
Expected: prints batched shapes; `input_ids (2, N)`, `input_features (2, 128, 3000)`, `feature_attention_mask (2, 3000)`. No exception.

- [ ] **Step 5: Commit**

```bash
git add specforge/data/utils.py scripts/smoke_audio_pipeline.py
git commit -m "feat(audio): AudioDataCollatorWithPadding + dataloader wiring"
```

---

## Task 5: `QwenAudioOnlineEagle3Model`

**Files:**
- Modify: `specforge/core/eagle3.py` (add class after `QwenVLOnlineEagle3Model` ~line 791)
- Modify: `scripts/smoke_audio_pipeline.py` (add `check_eagle3`)

Differs from `QwenVLOnlineEagle3Model` only in: `_prepare_data` target inputs (audio instead of image), and `position_ids` computed as a standard cumsum (no `get_rope_index`). The TTT loop body is identical.

- [ ] **Step 1: Add the class to `specforge/core/eagle3.py`**

```python
class QwenAudioOnlineEagle3Model(Eagle3Model):
    """Online EAGLE3 training for Qwen2-Audio. Audio is consumed by the target
    only; the draft is a text-token predictor conditioned on the target's
    (audio-aware) aux hidden states. Standard 1D RoPE (no M-RoPE)."""

    def __init__(self, target_model, draft_model, processor, length=7,
                 attention_backend="sdpa", lk_loss_type=None, kl_scale=1.0, kl_decay=1.0):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.processor = processor
        self.length = length
        self.attention_backend = attention_backend
        self.lk_loss_type = lk_loss_type
        self.kl_scale = kl_scale
        self.kl_decay = kl_decay

    @torch.no_grad()
    def _prepare_data(self, input_ids, attention_mask, loss_mask,
                      input_features=None, feature_attention_mask=None, device=None):
        if device is None:
            device = input_ids.device
        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        num_hidden_states = len(outputs.hidden_states)
        offset = 1
        num_layers = num_hidden_states - 1
        low_aux_layer = 1 + offset
        mid_aux_layer = num_layers // 2 - 1 + offset
        last_aux_layer = num_layers - 4 + offset
        hidden_states = torch.cat(
            (outputs.hidden_states[low_aux_layer],
             outputs.hidden_states[mid_aux_layer],
             outputs.hidden_states[last_aux_layer]),
            dim=-1,
        )
        target = padding(outputs.logits, left=False)
        input_ids = padding(input_ids, left=False)
        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None].to(device)
        return hidden_states, target, loss_mask, input_ids

    def forward(self, input_ids, attention_mask, loss_mask,
                input_features=None, feature_attention_mask=None,
                past_key_values=None, position_ids=None):
        # Step 0: target forward -> aux hidden states + target logits
        hidden_states, target, loss_mask, input_ids = self._prepare_data(
            input_ids, attention_mask, loss_mask, input_features, feature_attention_mask
        )
        # Step 1: vocab handling
        (target_p_padded, target_p_on_draft_padded, target_token_ids_padded,
         position_mask) = _compute_target_p_padded(
            target=target, t2d=self.draft_model.t2d, loss_mask=loss_mask, length=self.length
        )
        del target
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0
        # Step 2: project 3*H -> H
        hidden_states = self.draft_model.project_hidden_states(hidden_states)
        # Step 3: standard 1D position ids from attention mask (no M-RoPE)
        if position_ids is None:
            am = attention_mask if attention_mask is not None else torch.ones(
                batch_size, seq_length, device=hidden_states.device, dtype=torch.long
            )
            position_ids = am.long().cumsum(-1) - 1
            position_ids.masked_fill_(am == 0, 1)
        # Step 4: attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool,
                                        device=hidden_states.device)
        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask, hidden_states=hidden_states,
                batch_size=batch_size, seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )
        # Step 5: TTT (identical to QwenVLOnlineEagle3Model)
        plosses, acceptance_rates, acces = [], [], []
        metric_corrects, metric_denoms, metric_losses, metric_loss_denoms = [], [], [], []
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        for idx in range(self.length):
            target_p = target_p_padded[:, idx: idx + seq_length, :].contiguous()
            target_p_on_draft = target_p_on_draft_padded[:, idx: idx + seq_length, :].contiguous()
            target_token_ids = target_token_ids_padded[:, idx: idx + seq_length].contiguous()
            is_last = idx == self.length - 1
            inputs_embeds = self.draft_model.embed_input_ids(input_ids).to(hidden_states.dtype)
            hidden_states = self.draft_model.backbone(
                input_embeds=inputs_embeds, hidden_states=hidden_states,
                cache_hidden=cache_hidden, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values, use_cache=True,
            )
            logits = self.draft_model.compute_logits(hidden_states)
            with torch.no_grad():
                correct, denom = _compute_metric_counts(
                    logits=logits, target_token_ids=target_token_ids,
                    loss_mask=loss_mask, d2t=self.draft_model.d2t,
                )
                acces.append(correct / denom)
                metric_corrects.append(correct)
                metric_denoms.append(denom)
            acceptance_rate, loss = _compute_loss_and_acceptance_rate(
                logits=logits, target_p=target_p, target_p_on_draft=target_p_on_draft,
                position_mask=position_mask, lk_loss_type=self.lk_loss_type,
                kl_scale=self.kl_scale, kl_decay=self.kl_decay,
            )
            acceptance_rates.append(acceptance_rate)
            plosses.append(loss)
            metric_losses.append(loss.detach())
            metric_loss_denoms.append(torch.tensor(
                logits.shape[0] * logits.shape[1], device=logits.device, dtype=torch.float32))
            if not is_last:
                input_ids = padding(input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
        return (plosses, acceptance_rates, acces, metric_corrects, metric_denoms,
                metric_losses, metric_loss_denoms)
```

- [ ] **Step 2: Add `check_eagle3` to the smoke script**

```python
def check_eagle3():
    from datasets import load_from_disk
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    from specforge.data.preprocessing import preprocess_audio_conversations
    from specforge.data.template import TEMPLATE_REGISTRY
    from specforge.data.utils import AudioDataCollatorWithPadding
    from specforge.modeling.auto import AutoEagle3DraftModel, AutoDraftModelConfig
    from specforge.core.eagle3 import QwenAudioOnlineEagle3Model

    template = TEMPLATE_REGISTRY.get("qwen")
    processor = AutoProcessor.from_pretrained(MODEL)
    target = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).eval().cuda()
    cfg = AutoDraftModelConfig.from_file("configs/qwen2-audio-7b-eagle3.json")
    draft = AutoEagle3DraftModel.from_config(cfg, torch_dtype=torch.bfloat16).cuda()
    draft.load_embedding(MODEL, embedding_key="language_model.model.embed_tokens.weight")
    draft.freeze_embedding()
    ds = load_from_disk("/tmp/aishell_smoke")
    feats = []
    for k in range(2):
        b = {"audio": [ds[k]["audio"]], "transcription": [ds[k]["transcription"]]}
        o = preprocess_audio_conversations(processor, b, template, 2048)
        feats.append({kk: vv[0] for kk, vv in o.items()})
    batch = AudioDataCollatorWithPadding()(feats)
    model = QwenAudioOnlineEagle3Model(target, draft, processor, length=3, attention_backend="sdpa")
    out = model(
        input_ids=batch["input_ids"].cuda(),
        attention_mask=batch["attention_mask"].cuda(),
        loss_mask=batch["loss_mask"].cuda(),
        input_features=batch["input_features"].cuda().to(torch.bfloat16),
        feature_attention_mask=batch["feature_attention_mask"].cuda(),
    )
    plosses = out[0]
    print("num TTT steps:", len(plosses), "step0 loss:", float(plosses[0]))
    assert torch.isfinite(plosses[0]).all()
```
Add `"eagle3": check_eagle3` to the dispatch dict.

NOTE: confirm the embedding key. Qwen2-Audio nests the LM, so the embedding weight is likely `language_model.model.embed_tokens.weight`. If `load_embedding` raises a KeyError, print the target `state_dict()` keys containing `embed_tokens` and use the matching key.

- [ ] **Step 3: Run the eagle3 forward smoke check**

Run: `cd SpecForge && python scripts/smoke_audio_pipeline.py eagle3`
Expected: `num TTT steps: 3`, a finite `step0 loss`. No exception. If `load_embedding` fails, adjust the embedding key per the NOTE and re-run.

- [ ] **Step 4: Commit**

```bash
git add specforge/core/eagle3.py scripts/smoke_audio_pipeline.py
git commit -m "feat(audio): QwenAudioOnlineEagle3Model"
```

---

## Task 6: `--is-audio` branches in `scripts/train_eagle3.py`

**Files:**
- Modify: `scripts/train_eagle3.py`

Thread `--is-audio` through the five VLM decision points (the exploration anchors: args ~91-149, target/processor build ~358-432, dataset build ~577-618, eagle3-model select ~1049-1064, run_forward ~718-841). Reuse everything else (optimizer, TTT weighting, checkpointing).

- [ ] **Step 1: Add the CLI arg**

Near the `--is-vlm` arg (~line 110), add:
```python
    parser.add_argument("--is-audio", action="store_true",
                        help="Set if the target is an audio model (e.g. Qwen2-Audio).")
```

- [ ] **Step 2: Load the audio processor + target**

In `build_target_model()` (the `if args.is_vlm ...` block ~358), add an audio branch:
```python
    if args.is_audio:
        from transformers import Qwen2AudioForConditionalGeneration
        target_model = (
            Qwen2AudioForConditionalGeneration.from_pretrained(
                args.target_model_path, torch_dtype=torch.bfloat16
            ).eval().cuda()
        )
```
And where the processor is loaded (`if args.is_vlm:` ~414):
```python
    if args.is_vlm or args.is_audio:
        processor = AutoProcessor.from_pretrained(args.target_model_path)
    else:
        processor = None
```
(Drop `min_pixels/max_pixels` for the audio case — they are image-only.)

- [ ] **Step 3: Pass `is_audio` to dataset + dataloader build**

In the `build_eagle3_dataset(...)` call (~585) add `is_audio=args.is_audio,` and in `prepare_dp_dataloaders(...)` (~600) add `is_audio=args.is_audio,`. Apply the same to the eval dataset/loader build if present.

- [ ] **Step 4: Select the audio eagle3 model**

In the eagle3-model selection (~1049), add an audio branch BEFORE the VLM branch:
```python
    if args.is_audio:
        from specforge.core.eagle3 import QwenAudioOnlineEagle3Model
        eagle3_model = QwenAudioOnlineEagle3Model(
            target_model=target_model, draft_model=draft_model, processor=processor,
            length=args.ttt_length, attention_backend=args.attention_backend,
            lk_loss_type=args.lk_loss_type, kl_scale=args.kl_scale, kl_decay=args.kl_decay,
        )
    elif args.is_vlm and ...:
        ...
```

- [ ] **Step 5: Route the forward call for audio**

In `run_forward()` (~718), add an audio branch parallel to the `if args.is_vlm and args.target_model_backend == "custom":` block:
```python
    if args.is_audio:
        plosses, acceptance_rates, acces, acc_corrects, acc_denoms, metric_losses, metric_loss_denoms = eagle3_model(
            input_ids=data["input_ids"].cuda(),
            attention_mask=data["attention_mask"].cuda(),
            loss_mask=data["loss_mask"].cuda(),
            input_features=data["input_features"].cuda().to(torch.bfloat16),
            feature_attention_mask=data["feature_attention_mask"].cuda(),
        )
    elif args.is_vlm and args.target_model_backend == "custom":
        ...
```

- [ ] **Step 6: Syntax check**

Run: `cd SpecForge && python -c "import ast; ast.parse(open('scripts/train_eagle3.py').read()); print('parse OK')"`
Expected: `parse OK`.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_eagle3.py
git commit -m "feat(audio): --is-audio branches in train_eagle3"
```

---

## Task 7: Launch script + small-subset end-to-end training

**Files:**
- Create: `examples/run_qwen2_audio_eagle3_online.sh`

- [ ] **Step 1: Prepare the 3000-example subset**

Run:
```bash
cd SpecForge && HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/.hf_cache \
  python scripts/prepare_aishell.py --num-samples 3000 --out /tmp/aishell_train3k
```
Expected: `saved 3000 examples to /tmp/aishell_train3k`.

- [ ] **Step 2: Write the launch script**

Create `examples/run_qwen2_audio_eagle3_online.sh` (replace `qwen` with the template chosen in Task 3 Step 3 if different):
```bash
#!/bin/bash
set -euo pipefail
export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/.hf_cache
torchrun --nproc_per_node 1 scripts/train_eagle3.py \
  --target-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --draft-model-config configs/qwen2-audio-7b-eagle3.json \
  --train-data-path /tmp/aishell_train3k \
  --is-audio \
  --chat-template qwen \
  --target-model-backend custom \
  --embedding-key language_model.model.embed_tokens.weight \
  --batch-size 1 \
  --max-length 1024 \
  --learning-rate 1e-4 \
  --ttt-length 7 \
  --num-epochs 1 \
  --max-num-steps 300 \
  --save-interval 300 \
  --output-dir outputs/qwen2-audio-7b-eagle3
```
NOTE: `--train-data-path` here points to a `save_to_disk` dataset dir. If `train_eagle3.py`'s loader only accepts JSONL, add a small `load_from_disk` branch in its dataset-loading code keyed on the path being a directory, OR have `prepare_aishell.py` also emit JSONL — confirm which the loader supports during this step and adjust.

- [ ] **Step 3: Run a short training job**

Run: `cd SpecForge && bash examples/run_qwen2_audio_eagle3_online.sh 2>&1 | tee /tmp/audio_train.log`
Expected: training starts, per-step logs show `loss` and `acc`/acceptance metrics, loss trends DOWN over 300 steps, and a checkpoint is written under `outputs/qwen2-audio-7b-eagle3/`. Capture the final checkpoint path.

- [ ] **Step 4: Sanity-check the saved checkpoint format**

Run:
```bash
cd SpecForge && python - <<'EOF'
import glob, json, os
ckpt = sorted(glob.glob("outputs/qwen2-audio-7b-eagle3/*/"))[-1]
cfg = json.load(open(os.path.join(ckpt, "config.json")))
assert cfg["architectures"] == ["LlamaForCausalLMEagle3"], cfg["architectures"]
assert any(f.endswith((".safetensors", ".bin")) for f in os.listdir(ckpt))
print("checkpoint OK:", ckpt)
EOF
```
Expected: `checkpoint OK: outputs/qwen2-audio-7b-eagle3/<...>/`.

- [ ] **Step 5: Commit**

```bash
git add examples/run_qwen2_audio_eagle3_online.sh
git commit -m "feat(audio): online training launch script for Qwen2-Audio EAGLE3"
```

---

## Task 8: Cross-env inference validation (acceptance > random)

**Files:**
- Use (no edit): `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/qwen2audio_specdec/run_validation.sh`

The trained draft must load in the validated dev-HEAD sglang path (deactivate the SpecForge env; use the env where sglang dev-HEAD + the `qwen2_audio` shim were validated).

- [ ] **Step 1: Point the validation at the trained draft**

Copy the trained checkpoint to a stable path and run the two-stage validation with `DRAFT` set to it:
```bash
deactivate 2>/dev/null || true
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
CKPT=$(ls -d SpecForge/outputs/qwen2-audio-7b-eagle3/*/ | tail -1)
export PYTHONPATH=$(pwd)/sglang/python:$PYTHONPATH
export HF_HOME=$(pwd)/.hf_cache
TARGET=Qwen/Qwen2-Audio-7B-Instruct DRAFT="$CKPT" PORT=31002 \
  bash qwen2audio_specdec/run_validation.sh 2>&1 | tee /tmp/audio_val.log
```
Expected: server boots with `type=LlamaForCausalLMEagle3`, the Stage-1 transcription is byte-identical to Stage-0 (lossless), and the run completes without error.

- [ ] **Step 2: Measure acceptance length vs the random draft**

Run an AISHELL clip through the spec-decode server with decode stats enabled and confirm mean accepted tokens/step > 1 (random draft ≈ 0/near-1). Use a longer Mandarin clip and add `--decode-log-interval 1` to the server args, or query `/get_server_info`:
```bash
# with the Stage-1 server still up on PORT=31002:
curl -s http://127.0.0.1:31002/get_server_info | python -c 'import sys,json;d=json.load(sys.stdin);print("spec_accept_length:", d.get("internal_states") or d)'
```
Expected: an acceptance-length / accept-rate figure meaningfully above the random-draft baseline. Record it in `/tmp/audio_val.log`. (If the field name differs in dev-HEAD, grep the server log for `accept` after sending several requests.)

- [ ] **Step 3: Update the memory file**

Append the result (trained-draft acceptance length, checkpoint path, chosen chat template, embedding key) to the memory `qwen2audio-eagle3-sglang-validated` so the next session has it.

---

## Self-Review notes (addressed)

- **Spec coverage:** env (T0), config (T1), data prep (T2), preprocessing (T3), collator (T4), eagle3 model (T5), train script (T6), launch+train (T7), cross-env validation (T8) — all spec sections covered.
- **Known confirm-at-runtime points (each has a guard/NOTE):** Qwen2-Audio chat-template name (T3 S3), `load_embedding` key (T5 S2 NOTE), `train-data-path` dir vs jsonl loader (T7 S2 NOTE), `get_server_info` field name (T8 S2). These are integration unknowns surfaced as explicit smoke checks rather than hidden assumptions.
- **Type consistency:** new field names (`input_features`, `feature_attention_mask`) are identical across preprocessing → collator → eagle3 model → train script. `is_audio` flag name consistent across `build_eagle3_dataset`, `prepare_dp_dataloaders`, `train_eagle3.py`.
