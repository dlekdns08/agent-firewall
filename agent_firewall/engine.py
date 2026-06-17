"""Guardrail engine.

Walks Anthropic Messages API payloads and applies detectors:

* ``inspect_request``  — runs on the request *before* it reaches the model.
  Scans untrusted content (tool_result blocks) for prompt injection and
  masks PII in-place. Returns the (possibly mutated) body + a Decision.
* ``inspect_response`` — runs on the model's reply. Classifies every
  ``tool_use`` block against the action policy. Returns per-tool decisions.

The engine never performs I/O and never blocks; it only computes verdicts.
The proxy decides how to enforce them (error, approval prompt, pass-through).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import Config
from .detectors import actions as action_detector
from .detectors import injection, pii
from .models import Action, Decision, Finding, Severity, max_severity


def inspect_request(body: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], Decision]:
    """Scan + sanitize an outgoing request. Returns (new_body, decision)."""
    body = deepcopy(body)
    decision = Decision(action=Action.ALLOW)

    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, decision

    for mi, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        # String content (simple user/assistant turn).
        if isinstance(content, str):
            new_text, sub = _scan_text(
                content, role=role, is_tool_result=False, cfg=cfg,
                location=f"messages[{mi}].content",
            )
            message["content"] = new_text
            decision = decision.merge(sub)
            continue

        if not isinstance(content, list):
            continue

        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            loc = f"messages[{mi}].content[{bi}]"

            if btype == "text":
                new_text, sub = _scan_text(
                    block.get("text", ""), role=role, is_tool_result=False,
                    cfg=cfg, location=f"{loc}.text",
                )
                block["text"] = new_text
                decision = decision.merge(sub)

            elif btype == "tool_result":
                new_content, sub = _scan_tool_result(block.get("content"), cfg, loc)
                block["content"] = new_content
                decision = decision.merge(sub)

    return body, _finalize_request_action(decision, cfg)


def inspect_response(body: dict[str, Any], cfg: Config) -> tuple[Decision, list[dict[str, Any]]]:
    """Scan a model response. Returns (overall_decision, tool_calls).

    Each tool_call dict: {"id", "name", "input", "decision"}.
    """
    decision = Decision(action=Action.ALLOW)
    tool_calls: list[dict[str, Any]] = []

    content = body.get("content")
    if not isinstance(content, list):
        return decision, tool_calls

    for bi, block in enumerate(content):
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        tool_input = block.get("input", {})
        sub = action_detector.classify(
            name, tool_input, cfg.actions, location=f"content[{bi}]",
        )
        tool_calls.append(
            {"id": block.get("id"), "name": name, "input": tool_input, "decision": sub}
        )
        decision = decision.merge(sub)

    return decision, tool_calls


# --- internals -------------------------------------------------------------


def _scan_text(
    text: str, *, role: str | None, is_tool_result: bool, cfg: Config, location: str
) -> tuple[str, Decision]:
    decision = Decision(action=Action.ALLOW)
    if not text:
        return text, decision

    # PII masking applies to everything we send upstream.
    if cfg.pii.enabled:
        if cfg.pii.action == Action.MASK:
            text, findings = pii.mask(text, categories=cfg.pii.categories, location=location)
            if findings:
                decision = decision.merge(
                    Decision(
                        action=Action.MASK,
                        severity=_top_severity(findings),
                        findings=findings,
                        reason="Masked PII",
                    )
                )
        else:
            findings = pii.scan(text, categories=cfg.pii.categories, location=location)
            if findings:
                decision = decision.merge(
                    Decision(
                        action=cfg.pii.action,
                        severity=_top_severity(findings),
                        findings=findings,
                        reason="PII detected",
                    )
                )

    # Injection scanning. By default only untrusted (tool_result) content.
    if cfg.injection.enabled:
        scan_this = is_tool_result or not cfg.injection.scan_tool_results_only
        if scan_this:
            score, findings = injection.scan(text, location=location)
            if findings:
                action = Action.ALLOW
                if score >= cfg.injection.block_threshold:
                    action = Action.BLOCK
                elif score >= cfg.injection.review_threshold:
                    action = Action.REQUIRE_APPROVAL
                decision = decision.merge(
                    Decision(
                        action=action,
                        severity=_top_severity(findings),
                        findings=findings,
                        reason=f"Prompt-injection score {score}",
                    )
                )

    return text, decision


def _scan_tool_result(content: Any, cfg: Config, location: str) -> tuple[Any, Decision]:
    """tool_result content is a string or a list of blocks."""
    if isinstance(content, str):
        return _scan_text(content, role="user", is_tool_result=True, cfg=cfg, location=location)

    if isinstance(content, list):
        decision = Decision(action=Action.ALLOW)
        for bi, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "text":
                new_text, sub = _scan_text(
                    block.get("text", ""), role="user", is_tool_result=True,
                    cfg=cfg, location=f"{location}.content[{bi}].text",
                )
                block["text"] = new_text
                decision = decision.merge(sub)
        return content, decision

    return content, Decision(action=Action.ALLOW)


def _finalize_request_action(decision: Decision, cfg: Config) -> Decision:
    return decision


def _top_severity(findings: list[Finding]) -> Severity:
    sev = Severity.INFO
    for f in findings:
        sev = max_severity(sev, f.severity)
    return sev
