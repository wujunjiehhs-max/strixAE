"""Prompt definitions for the strixAE GRPO inference workflow."""

from __future__ import annotations

from collections.abc import Sequence


SYSTEM_PROMPT = (
    "You are an audio deep-thinking model. Analyze the audio and explain the "
    "rationale for the given restoration pipeline. Respond in two parts: first, "
    "provide your reasoning in <THINK></THINK> tags; second, provide a structured "
    "analysis explaining why each task is needed and why the tasks are ordered "
    "this way."
)

DEFAULT_PIPELINE = [
    "denoise(MPSENet)",
    "dereverberate(SGMSE_16K)",
    "separate(SPMamba)",
    "super_resolution(MossFormer2_SR_48K)",
]

TASKS = {
    "denoise": "Remove background noise and improve signal-to-noise ratio.",
    "dereverberate": "Remove reverberation and echo effects.",
    "separate": "Separate overlapping speech or mixed audio sources.",
    "super_resolution": "Restore bandwidth and high-frequency content.",
}

MODELS = {
    "FRCRN_SE_16K": (
        "Complex-valued single-channel speech enhancement for wideband and "
        "full-band denoising."
    ),
    "MossFormerGAN_SE_16K": (
        "Adversarially trained 16 kHz single-channel speech enhancement model."
    ),
    "MPSENet": (
        "Time-frequency monaural enhancement with parallel magnitude and phase "
        "denoising, suitable for low-SNR 16 kHz speech."
    ),
    "VINP": (
        "Variational Bayesian dereverberation with a neural speech prior for "
        "far-field speech and unknown room impulse responses."
    ),
    "SGMSE_16K": (
        "Score-based generative speech enhancement and dereverberation at 16 kHz."
    ),
    "TIGER": (
        "Time-frequency interleaved speech separation designed for low parameter "
        "count and computational cost."
    ),
    "MossFormer2_SS_16K": (
        "16 kHz speech separation model capturing global structure and fine "
        "temporal patterns."
    ),
    "SPMamba": (
        "Long-sequence speech separation model suitable for low-latency and "
        "resource-constrained scenarios."
    ),
    "MossFormer2_SR_48K": (
        "Speech super-resolution model that reconstructs 48 kHz output from "
        "lower-resolution speech."
    ),
}


def build_restoration_prompt(instruction: str, pipeline: Sequence[str]) -> str:
    task_catalog = "\n".join(f"- {name}: {text}" for name, text in TASKS.items())
    model_catalog = "\n".join(f"- {name}: {text}" for name, text in MODELS.items())
    ordered_pipeline = "\n".join(
        f"{index}. {step}" for index, step in enumerate(pipeline, start=1)
    )

    return f"""Analyze the supplied audio for restoration planning.

Available tasks:
{task_catalog}

Available models:
{model_catalog}

Given restoration pipeline (execution order):
{ordered_pipeline}

User instruction:
{instruction}

Explain the given pipeline; do not replace it with a different pipeline.

Required response:
1. Put the audio analysis and reasoning inside <THINK></THINK> tags.
2. After </THINK>, include these sections:
   - Detected Audio Issues
   - Restoration Pipeline Analysis
   - Pipeline Execution Order Rationale
3. For every step, explain the target issue, why the named model is suitable,
   and why the step occurs at that position.
4. Avoid inventing precise measurements such as SNR or RT60 unless they were
   actually computed and provided in the prompt.
"""
