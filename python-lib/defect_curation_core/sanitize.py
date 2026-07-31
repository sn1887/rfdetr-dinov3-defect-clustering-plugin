"""Sanitize row-level diagnostics before writing user-visible outputs."""

from __future__ import annotations

import re

_ABSOLUTE_PATH = re.compile(r"(?:(?:[A-Za-z]:)?[/\\](?:[^\s:'\"]+[/\\])+[^\s:'\"]*)")
_WHITESPACE = re.compile(r"\s+")


def sanitize_message(message: str, *, max_length: int = 500) -> str:
    redacted = _ABSOLUTE_PATH.sub("<redacted-path>", str(message))
    redacted = _WHITESPACE.sub(" ", redacted).strip()
    if len(redacted) > max_length:
        return redacted[: max_length - 1] + "…"
    return redacted
