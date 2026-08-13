"""Score parsing without silent fallback values."""

from __future__ import annotations

from dataclasses import dataclass
import re


_STRICT_SCORE = re.compile(r"^\s*Score:\s*(\d{1,3})\s*$", re.IGNORECASE)
_GPT_OSS_FINAL_CHANNEL = re.compile(
    r"assistant\s*final(?=\s*(?:\*\*)?(?:final\s+)?score)",
    re.IGNORECASE,
)
_LABELLED_SCORE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:final\s+)?score(?:\*\*)?\s*[:=]\s*(\d{1,3})(?:\s*/\s*100)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseResult:
    score: int | None
    ok: bool
    error: str | None = None


def _in_range(value: int) -> ParseResult:
    if 0 <= value <= 100:
        return ParseResult(score=value, ok=True)
    return ParseResult(None, False, f"score_out_of_range:{value}")


def parse_score(raw_output: str, *, strict: bool = True) -> ParseResult:
    """Parse a 0--100 score.

    Strict parsing is used for hard-constrained runs. Soft runs permit a
    labelled score in free text, but never guess from an arbitrary number and
    never replace a failure with 50.
    """

    if not isinstance(raw_output, str) or not raw_output.strip():
        return ParseResult(None, False, "empty_output")

    text = raw_output
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    final_channels = list(_GPT_OSS_FINAL_CHANNEL.finditer(text))
    if final_channels:
        text = text[final_channels[-1].end():]
    match = _STRICT_SCORE.fullmatch(text)
    if match:
        return _in_range(int(match.group(1)))
    if strict:
        return ParseResult(None, False, "expected_exact_score_format")

    matches = _LABELLED_SCORE.findall(text)
    if len(matches) == 1:
        return _in_range(int(matches[0]))
    if len(matches) > 1:
        return ParseResult(None, False, "multiple_labelled_scores")
    return ParseResult(None, False, "no_labelled_score")
