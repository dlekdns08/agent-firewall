"""Dangerous-action classification.

Given a tool name + input and the configured rules, decide whether the
tool call is allowed, needs approval, or must be blocked. Rules are matched
top-to-bottom; the most restrictive matching rule wins.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import ActionsConfig
from ..models import Action, Decision, Finding, Severity, _most_restrictive, max_severity

_SEV = {
    "info": Severity.INFO,
    "low": Severity.LOW,
    "medium": Severity.MEDIUM,
    "high": Severity.HIGH,
    "critical": Severity.CRITICAL,
}


def classify(tool_name: str, tool_input: Any, cfg: ActionsConfig, *, location: str = "") -> Decision:
    if not cfg.enabled:
        return Decision(action=Action.ALLOW)

    input_blob = _stringify(tool_input)

    # Allowlisted tools start at ALLOW even when default_action gates the rest
    # (allowlist mode). Dangerous-arg rules below can still escalate them.
    allowlisted = any(re.search(p, tool_name) for p in cfg.allowlist)
    base_action = Action.ALLOW if allowlisted else cfg.default_action
    decision = Decision(action=base_action)

    for rule in cfg.rules:
        if not re.search(rule.tool_pattern, tool_name):
            continue
        # If the rule scopes to argument patterns, all must be absent? No —
        # any matching arg pattern triggers. A rule with no arg_patterns
        # matches purely on tool name.
        if rule.arg_patterns:
            if not any(re.search(p, input_blob) for p in rule.arg_patterns):
                continue

        severity = _SEV.get(rule.severity.lower(), Severity.MEDIUM)
        finding = Finding(
            detector="actions",
            severity=severity,
            title=f"Dangerous action: {rule.name}",
            detail=rule.reason or rule.name,
            location=location,
            evidence={"tool": tool_name, "rule": rule.name},
        )
        decision = Decision(
            action=_most_restrictive(decision.action, rule.action),
            severity=max_severity(decision.severity, severity),
            findings=decision.findings + [finding],
            reason="; ".join(r for r in (decision.reason, rule.reason or rule.name) if r),
        )

    return decision


def _stringify(tool_input: Any) -> str:
    if isinstance(tool_input, str):
        return tool_input
    try:
        return json.dumps(tool_input, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(tool_input)
