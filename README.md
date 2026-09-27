# strixAE

Inference code for [`wujunjiehhs/strixAE`](https://huggingface.co/wujunjiehhs/strixAE), an Audio-Reasoner checkpoint post-trained with Group Relative Policy Optimization (GRPO) for audio-restoration reasoning.

The model analyzes an audio file and explains a supplied restoration pipeline, including detected audio issues, the role of each restoration model, and the execution-order rationale. Model weights are hosted on Hugging Face and are not stored in this repository.

## Installation

Python 3.10 or newer is recommended. Install a CUDA-compatible PyTorch build for your system, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`requirements.txt` is also provided for environments that do not need an
editable package installation.

The checkpoint contains approximately 8.4B parameters. GPU inference is recommended; available VRAM and the selected PyTorch precision determine the actual memory requirement.

## Quick start

```bash
strixae-infer path/to/audio.wav
```

Specify a restoration pipeline in execution order:

```bash
strixae-infer path/to/audio.wav \
  --instruction "Analyze the degradation and explain this pipeline." \
  --pipeline \
    "denoise(MPSENet)" \
    "dereverberate(SGMSE_16K)" \
    "separate(SPMamba)" \
    "super_resolution(MossFormer2_SR_48K)"
```

Return parsed JSON:

```bash
strixae-infer path/to/audio.wav --json
```

Use a local checkpoint instead of downloading from Hugging Face:

```bash
strixae-infer path/to/audio.wav \
  --model-id /path/to/Audio-Reasoner-GRPO-merged
```

Without installing the package, run the source-checkout wrapper with
`PYTHONPATH=src python scripts/infer.py path/to/audio.wav`.

Run `strixae-infer --help` for all generation options.

## Project structure

```text
strixAE/
├── src/strixae/
│   ├── __init__.py
│   ├── inference.py       # Loading, preprocessing, generation, parsing
│   ├── outputs.py         # Structured response parsing
│   └── prompts.py         # Restoration catalog and GRPO prompt
├── scripts/
│   └── infer.py           # Source-checkout command wrapper
├── examples/
│   └── restoration_pipeline.json
├── tests/
│   └── test_prompts.py
├── pyproject.toml         # Installable Python package and CLI definition
└── requirements.txt       # Minimal runtime dependencies
```

## Notes

- Generated acoustic judgments may be inaccurate. Validate recommendations before using them in an automated workflow.
- The model may emit `<THINK>` reasoning. Avoid exposing sensitive audio or model output unintentionally.

## Acknowledgements

This checkpoint builds on [Audio-Reasoner](https://github.com/xzf-thu/Audio-Reasoner) and the Qwen2-Audio architecture. See the [model card](https://huggingface.co/wujunjiehhs/strixAE) for model details, limitations, and citation information.

## License

MIT. Users are responsible for complying with the licenses and terms of all upstream models, data, and dependencies.
