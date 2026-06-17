"""PII detection and masking.

Regex-based, dependency-free. Each category has a compiled pattern and a
redaction label. ``scan`` returns findings; ``mask`` returns redacted text
plus the findings it acted on. Credit-card matches are Luhn-validated to cut
false positives on arbitrary 16-digit numbers.
"""

from __future__ import annotations

import re

from ..models import Finding, Severity

# Each entry: category -> (regex, severity, redaction label).
# ORDER MATTERS: high-specificity secret patterns run first so a greedier
# pattern (e.g. phone matching a digit run) can't eat part of a token before
# its own rule fires. Phone (loosest) is therefore last.
_RAW_PATTERNS: dict[str, tuple[str, Severity, str]] = {
    # Common API/secret token shapes (sk-..., ghp_..., xoxb-...)
    "api_token": (
        r"(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})",
        Severity.CRITICAL,
        "API_TOKEN",
    ),
    # AWS access key id
    "aws_key": (
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        Severity.CRITICAL,
        "AWS_KEY",
    ),
    # US SSN-style ###-##-####
    "ssn": (
        r"\b\d{3}-\d{2}-\d{4}\b",
        Severity.HIGH,
        "SSN",
    ),
    # 13-19 digit card numbers, optionally separated by single spaces/dashes.
    # Greedy so the whole number is captured, then Luhn-validated.
    "credit_card": (
        r"\b\d(?:[ -]?\d){12,18}\b",
        Severity.HIGH,
        "CREDIT_CARD",
    ),
    "email": (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        Severity.MEDIUM,
        "EMAIL",
    ),
    # International-ish phone numbers (loose) — kept last on purpose.
    "phone": (
        r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?){2,4}\d{2,4}(?!\d)",
        Severity.LOW,
        "PHONE",
    ),
}

PATTERNS: dict[str, tuple[re.Pattern[str], Severity, str]] = {
    name: (re.compile(rx), sev, label) for name, (rx, sev, label) in _RAW_PATTERNS.items()
}


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _accept(category: str, match: str) -> bool:
    """Secondary validation to suppress obvious false positives."""
    if category == "credit_card":
        return _luhn_ok(match)
    if category == "phone":
        # Require at least 7 digits to count as a phone number.
        return sum(c.isdigit() for c in match) >= 7
    return True


def _active_categories(categories: list[str] | None) -> list[str]:
    if not categories:
        return list(PATTERNS)
    return [c for c in categories if c in PATTERNS]


def scan(text: str, *, categories: list[str] | None = None, location: str = "") -> list[Finding]:
    findings: list[Finding] = []
    for cat in _active_categories(categories):
        pattern, severity, label = PATTERNS[cat]
        for m in pattern.finditer(text):
            value = m.group(0)
            if not _accept(cat, value):
                continue
            findings.append(
                Finding(
                    detector="pii",
                    severity=severity,
                    title=f"PII detected: {cat}",
                    detail=f"Matched {label} pattern",
                    location=location,
                    evidence={"category": cat, "match": _preview(value)},
                )
            )
    return findings


def mask(text: str, *, categories: list[str] | None = None, location: str = "") -> tuple[str, list[Finding]]:
    """Return (redacted_text, findings)."""
    findings: list[Finding] = []
    masked = text
    for cat in _active_categories(categories):
        pattern, severity, label = PATTERNS[cat]

        def _repl(m: re.Match[str], _cat=cat, _sev=severity, _label=label) -> str:
            value = m.group(0)
            if not _accept(_cat, value):
                return value
            findings.append(
                Finding(
                    detector="pii",
                    severity=_sev,
                    title=f"PII masked: {_cat}",
                    detail=f"Redacted {_label}",
                    location=location,
                    evidence={"category": _cat, "match": _preview(value)},
                )
            )
            return f"[REDACTED_{_label}]"

        masked = pattern.sub(_repl, masked)
    return masked, findings


def _preview(value: str) -> str:
    """Never echo full secrets back into findings/logs."""
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]
