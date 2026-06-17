"""Shared decision/finding types used across detectors, engine, and proxy."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    """What the firewall decided to do about a request/response."""

    ALLOW = "allow"
    MASK = "mask"          # content was modified (e.g. PII redacted) but allowed through
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


# Severity ordering helper — higher index == more severe.
_SEVERITY_ORDER = [
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
]


def max_severity(a: Severity, b: Severity) -> Severity:
    return a if _SEVERITY_ORDER.index(a) >= _SEVERITY_ORDER.index(b) else b


class Finding(BaseModel):
    """A single thing a detector noticed."""

    detector: str
    severity: Severity
    title: str
    detail: str = ""
    # Where it was found, e.g. "messages[2].content[0].tool_result".
    location: str = ""
    # Free-form evidence (matched span, tool name, arg path, ...).
    evidence: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """Aggregate verdict for one direction (input or output) of a request."""

    action: Action = Action.ALLOW
    severity: Severity = Severity.INFO
    findings: list[Finding] = Field(default_factory=list)
    # Human-readable reason summarizing why this action was chosen.
    reason: str = ""

    def merge(self, other: "Decision") -> "Decision":
        """Combine two decisions, taking the most restrictive action."""
        action = _most_restrictive(self.action, other.action)
        return Decision(
            action=action,
            severity=max_severity(self.severity, other.severity),
            findings=self.findings + other.findings,
            reason="; ".join(r for r in (self.reason, other.reason) if r),
        )


_ACTION_ORDER = [
    Action.ALLOW,
    Action.MASK,
    Action.REQUIRE_APPROVAL,
    Action.BLOCK,
]


def _most_restrictive(a: Action, b: Action) -> Action:
    return a if _ACTION_ORDER.index(a) >= _ACTION_ORDER.index(b) else b
