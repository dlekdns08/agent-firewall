"""Heuristic prompt-injection detection.

Scans untrusted text (tool results, retrieved documents) for instruction-
override patterns. Each matched signal carries a weight; weights sum to a
score, which the engine compares against review/block thresholds. This is a
heuristic layer, not a guarantee — it catches the common, well-documented
injection shapes.
"""

from __future__ import annotations

import re

from ..models import Finding, Severity

# (name, weight, severity, compiled pattern, human description)
_SIGNALS: list[tuple[str, int, Severity, re.Pattern[str], str]] = [
    (
        "ignore_previous",
        4,
        Severity.HIGH,
        re.compile(r"(?i)\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instruction|prompt|rule|context|message)s?\b"),
        "Attempt to override prior instructions",
    ),
    (
        "new_instructions",
        3,
        Severity.MEDIUM,
        re.compile(r"(?i)\b(new|updated|real|actual)\b.{0,15}\b(instruction|task|objective|directive)s?\b\s*:?"),
        "Injected replacement instructions",
    ),
    (
        "role_override",
        3,
        Severity.MEDIUM,
        re.compile(r"(?i)\byou are now\b|\bact as\b.{0,20}\b(admin|root|developer|dan|unrestricted)\b|\bswitch to\b.{0,15}\bmode\b"),
        "Attempt to redefine the assistant's role",
    ),
    (
        "system_prompt_exfil",
        4,
        Severity.HIGH,
        re.compile(r"(?i)\b(reveal|print|show|repeat|output|tell me)\b.{0,25}\b(system prompt|your instructions|initial prompt|the prompt above)\b"),
        "Attempt to exfiltrate the system prompt",
    ),
    (
        "fake_system_tag",
        4,
        Severity.HIGH,
        re.compile(r"(?i)<\s*/?\s*(system|assistant|im_start|im_end)\s*>|\[/?(INST|SYSTEM)\]|###\s*system"),
        "Spoofed system/role delimiter",
    ),
    (
        "exfil_action",
        4,
        Severity.HIGH,
        re.compile(r"(?i)\b(send|email|post|upload|exfiltrate|forward|transmit)\b.{0,40}\b(api[_ ]?key|password|secret|token|credential|/etc/passwd|env)\b"),
        "Instruction to exfiltrate secrets",
    ),
    (
        "tool_hijack",
        3,
        Severity.MEDIUM,
        re.compile(r"(?i)\b(call|invoke|use|execute)\b.{0,20}\b(the\s+)?(tool|function|command)\b.{0,30}\b(immediately|now|without asking|do not ask)\b"),
        "Attempt to force a tool call",
    ),
    (
        "override_safety",
        2,
        Severity.LOW,
        re.compile(r"(?i)\b(do not|don't|never)\b.{0,15}\b(ask|confirm|warn|refuse|mention)\b"),
        "Attempt to suppress confirmation/safety behavior",
    ),
]


def scan(text: str, *, location: str = "") -> tuple[int, list[Finding]]:
    """Return (total_score, findings)."""
    score = 0
    findings: list[Finding] = []
    for name, weight, severity, pattern, desc in _SIGNALS:
        m = pattern.search(text)
        if not m:
            continue
        score += weight
        findings.append(
            Finding(
                detector="injection",
                severity=severity,
                title=f"Prompt-injection signal: {name}",
                detail=desc,
                location=location,
                evidence={"signal": name, "weight": weight, "match": _clip(m.group(0))},
            )
        )
    return score, findings


def _clip(s: str, limit: int = 120) -> str:
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "…"
