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
        text=text, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True
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


def check_preprocess():
    from datasets import load_from_disk
    from transformers import AutoProcessor
    from specforge.data.preprocessing import preprocess_audio_conversations
    from specforge.data.template import TEMPLATE_REGISTRY

    template = TEMPLATE_REGISTRY.get(
        "qwen"
    )  # use the template name confirmed in Step C
    processor = AutoProcessor.from_pretrained(MODEL)
    ds = load_from_disk("/tmp/aishell_smoke")
    batch = {
        "audio": [ds[0]["audio"], ds[1]["audio"]],
        "transcription": [ds[0]["transcription"], ds[1]["transcription"]],
    }
    out = preprocess_audio_conversations(processor, batch, template, max_length=2048)
    ii = out["input_ids"][0]
    lm = out["loss_mask"][0]
    print("input_ids shape:", tuple(ii.shape), "loss_mask sum:", int(lm.sum()))
    print("input_features shape:", tuple(out["input_features"][0].shape))
    masked = processor.tokenizer.decode(ii[0][lm[0].bool()])
    print("loss-masked (assistant) text:", repr(masked))
    assert lm.sum() > 0, "loss mask is empty — assistant span not detected"


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


def check_eagle3():
    from datasets import load_from_disk
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    from specforge.data.preprocessing import preprocess_audio_conversations
    from specforge.data.template import TEMPLATE_REGISTRY
    from specforge.data.utils import AudioDataCollatorWithPadding
    from specforge.modeling.auto import AutoEagle3DraftModel, AutoDraftModelConfig
    from specforge.core.eagle3 import QwenAudioOnlineEagle3Model
    from specforge.data.preprocessing import process_token_dict_to_mappings
    from collections import Counter

    template = TEMPLATE_REGISTRY.get("qwen")
    processor = AutoProcessor.from_pretrained(MODEL)
    target = (
        Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16
        )
        .eval()
        .cuda()
    )
    cfg = AutoDraftModelConfig.from_file("configs/qwen2-audio-7b-eagle3.json")
    draft = AutoEagle3DraftModel.from_config(cfg, torch_dtype=torch.bfloat16).cuda()
    draft.load_embedding(
        MODEL, embedding_key="language_model.model.embed_tokens.weight"
    )
    draft.freeze_embedding()
    ds = load_from_disk("/tmp/aishell_smoke")
    feats = []
    for k in range(2):
        b = {"audio": [ds[k]["audio"]], "transcription": [ds[k]["transcription"]]}
        o = preprocess_audio_conversations(processor, b, template, 2048)
        feats.append({kk: vv[0] for kk, vv in o.items()})
    batch = AudioDataCollatorWithPadding()(feats)

    # Build and load a vocab mapping (t2d/d2t) so the draft's draft_vocab_size
    # logits align with the target's reduced vocab. Without this the default
    # all-True t2d keeps target_p at the full target vocab and the loss errors.
    token_dict = Counter()
    for f in feats:
        ids = f["input_ids"].reshape(-1)
        lm = f["loss_mask"].reshape(-1).bool()
        masked = ids[lm]
        uids, cnts = masked.unique(return_counts=True)
        token_dict.update(dict(zip(uids.tolist(), cnts.tolist())))
    d2t, t2d = process_token_dict_to_mappings(
        token_dict, cfg.draft_vocab_size, cfg.vocab_size
    )
    draft.t2d.copy_(t2d.to(draft.t2d.device))
    draft.d2t.copy_(d2t.to(draft.d2t.device))
    # This container lacks CPython dev headers (/usr/include/python3.12/Python.h),
    # so Triton/torch-inductor JIT cannot build their C launcher stubs. For this
    # forward-only smoke check, route the KL loss through the mathematically
    # equivalent pure-PyTorch reference and disable dynamo. The production code
    # path (Triton LogSoftmaxLoss) is unchanged; this monkeypatch is test-local.
    import specforge.core.eagle3 as _e3
    from specforge.core.loss import _compute_loss as _ref_kl_loss

    def _patched_loss_and_acc(
        *,
        logits,
        target_p,
        target_p_on_draft,
        position_mask,
        lk_loss_type,
        kl_scale,
        kl_decay,
        **_,
    ):
        from specforge.core.lk_loss import compute_acceptance_rate

        loss = _ref_kl_loss(logits, target_p, position_mask)
        acceptance_rate, _ = compute_acceptance_rate(
            logits=logits,
            target_probs=target_p_on_draft,
            position_mask=position_mask,
            reduce_fn=None,
        )
        return acceptance_rate.detach(), loss

    _e3._compute_loss_and_acceptance_rate = _patched_loss_and_acc

    model = QwenAudioOnlineEagle3Model(
        target, draft, processor, length=3, attention_backend="sdpa"
    )
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


if __name__ == "__main__":
    {
        "target": check_target,
        "preprocess": check_preprocess,
        "collate": check_collate,
        "eagle3": check_eagle3,
    }[sys.argv[1]]()
