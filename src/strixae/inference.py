"""Inference CLI for the strixAE GRPO audio-reasoning model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from .outputs import parse_response
from .prompts import DEFAULT_PIPELINE, SYSTEM_PROMPT, build_restoration_prompt


DEFAULT_MODEL_ID = "wujunjiehhs/strixAE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run audio-restoration reasoning with strixAE."
    )
    parser.add_argument("audio", type=Path, help="Path to a local audio file.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--instruction",
        default="Analyze this audio and explain the proposed restoration pipeline.",
        help="Question or instruction about the audio.",
    )
    parser.add_argument(
        "--pipeline",
        nargs="+",
        default=DEFAULT_PIPELINE,
        metavar="TASK(MODEL)",
        help="Ordered restoration steps, for example: denoise(MPSENet).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print parsed THINK/response fields as JSON.",
    )
    return parser.parse_args()


def build_messages(audio_path: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def load_model(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, processor


def generate(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    audio_path: Path,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    messages = build_messages(str(audio_path), prompt)
    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    audio, _ = librosa.load(
        audio_path,
        sr=processor.feature_extractor.sampling_rate,
        mono=True,
    )
    inputs = processor(
        text=chat_text,
        audio=audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {name: value.to(model.device) for name, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main() -> None:
    args = parse_args()
    prompt = build_restoration_prompt(args.instruction, args.pipeline)
    model, processor = load_model(args.model_id)
    output = generate(
        model=model,
        processor=processor,
        audio_path=args.audio,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if args.json:
        result = {
            "audio": str(args.audio),
            "instruction": args.instruction,
            "pipeline": args.pipeline,
            **parse_response(output),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(output)


if __name__ == "__main__":
    main()
