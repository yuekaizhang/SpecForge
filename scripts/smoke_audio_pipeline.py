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
