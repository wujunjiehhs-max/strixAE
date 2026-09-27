"""Helpers for parsing structured strixAE responses."""

from __future__ import annotations

import re


def parse_response(text: str) -> dict[str, str]:
    """Split a tagged reasoning response without failing on untagged output."""
    match = re.search(r"<THINK>(.*?)</THINK>", text, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        return {"think": "", "response": text.strip()}
    return {
        "think": match.group(1).strip(),
        "response": text[match.end() :].strip(),
    }
