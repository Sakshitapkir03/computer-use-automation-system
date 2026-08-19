"""
PII redaction chokepoint.

Every string destined for /evidence/ or /artifact/store/ must pass through
redact() before touching disk. Two redaction layers run in order:

  1. Explicit sensitive values — caller supplies runtime param values that are
     marked sensitive=True in their ParamSpec.  Each non-empty value is
     replaced globally with [REDACTED].  Applied first so that a sensitive
     value that happens to look like a digit sequence is caught here rather
     than by the pattern layer.

  2. Structural PII patterns — three regex rules applied in sequence:
       • SSN              NNN-NN-NNNN                   → [REDACTED]
       • Grouped digits   NNNN[-/space]NNNN[-/space]…   → [REDACTED]
         Catches formatted account/card numbers (e.g. 5432-1098-76,
         4111-1111-1111-1111).  Requires 4+4 leading groups so ISO dates
         (YYYY-MM-DD = 4-2-2) are NOT matched.
       • Long digits       9 or more consecutive         → [REDACTED]
         Catches raw account/routing numbers not already masked above.

Fail-fast guard: redact() raises TypeError on non-string input so accidental
misuse (e.g. passing a dict instead of serialised JSON) surfaces immediately
rather than silently writing unredacted data.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

# SSN: three digits, hyphen, two digits, hyphen, four digits (word-bounded).
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Formatted account/card numbers: NNNN-NNNN-... or NNNN NNNN-...
# Requires 4-digit leading group + 4-digit second group (rules out ISO dates
# whose middle group is 2 digits), then one or more trailing groups of ≥ 2
# digits separated by hyphens or spaces.  Covers 10-digit and 16-digit formats.
_GROUPED_DIGITS_RE = re.compile(r"\b\d{4}[- ]\d{4}(?:[- ]\d{2,})+\b")

# Raw account / routing numbers: 9+ consecutive digits not already masked.
_LONG_DIGITS_RE = re.compile(r"\b\d{9,}\b")

_REDACTED = "[REDACTED]"


def redact(text: str, sensitive_values: Iterable[str] = ()) -> str:
    """
    Mask PII and sensitive values before writing to disk.

    Args:
        text:             The serialised string (JSON, log line, etc.) to clean.
        sensitive_values: Optional runtime values to redact by exact string
                          match.  Typical callers pass the values of all params
                          whose ParamSpec.sensitive is True.  Empty strings are
                          skipped (replacing "" would corrupt the entire output).

    Returns:
        A copy of text with all PII replaced by [REDACTED].

    Raises:
        TypeError: text is not a str (fail-fast guard against misuse).
    """
    if not isinstance(text, str):
        raise TypeError(f"redact() expects str, got {type(text).__name__}")

    # Layer 1 — explicit sensitive values (exact, case-sensitive).
    for val in sensitive_values:
        if val:  # skip empty strings
            text = text.replace(val, _REDACTED)

    # Layer 2 — structural PII patterns (order matters: grouped before long-digits
    # so a formatted number isn't partially consumed by the consecutive-digit rule).
    text = _SSN_RE.sub(_REDACTED, text)
    text = _GROUPED_DIGITS_RE.sub(_REDACTED, text)
    text = _LONG_DIGITS_RE.sub(_REDACTED, text)

    return text
