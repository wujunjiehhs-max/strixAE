from strixae.outputs import parse_response
from strixae.prompts import DEFAULT_PIPELINE, build_restoration_prompt


def test_prompt_contains_pipeline_and_output_contract():
    prompt = build_restoration_prompt("Inspect the recording.", DEFAULT_PIPELINE)

    assert "Inspect the recording." in prompt
    assert "denoise(MPSENet)" in prompt
    assert "Detected Audio Issues" in prompt
    assert "<THINK></THINK>" in prompt


def test_parse_tagged_response():
    parsed = parse_response("<THINK>analysis</THINK>final answer")

    assert parsed == {"think": "analysis", "response": "final answer"}


def test_parse_untagged_response():
    parsed = parse_response("final answer")

    assert parsed == {"think": "", "response": "final answer"}
