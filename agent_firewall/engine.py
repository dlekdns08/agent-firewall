"""Guardrail engine.

Walks Anthropic Messages API payloads and applies detectors:

* ``inspect_request``  — runs on the request *before* it reaches the model.
  Masks PII per block, then scores prompt injection across ALL untrusted
  blocks together (aggregate, so split payloads can't dodge the threshold).
  Returns the (possibly mutated) body + a Decision.
* ``inspect_response`` — runs on the model's reply. Optionally masks PII in
  the model's output text, and classifies every ``tool_use`` block against
  the action policy. Mutates the body in place; returns per-tool decisions.

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

    # Untrusted text collected for aggregate injection scoring.
    injection_targets: list[tuple[str, str]] = []  # (text, location)

    for mi, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str):
            new_text, sub = _mask_pii(content, cfg, location=f"messages[{mi}].content")
            message["content"] = new_text
            decision = decision.merge(sub)
            if _should_scan_injection(is_tool_result=False, cfg=cfg):
                injection_targets.append((new_text, f"messages[{mi}].content"))
            continue

        if not isinstance(content, list):
            continue

        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            loc = f"messages[{mi}].content[{bi}]"

            if btype == "text":
                new_text, sub = _mask_pii(block.get("text", ""), cfg, location=f"{loc}.text")
                block["text"] = new_text
                decision = decision.merge(sub)
                if _should_scan_injection(is_tool_result=False, cfg=cfg):
                    injection_targets.append((new_text, f"{loc}.text"))

            elif btype == "tool_result":
                sub, targets = _scan_tool_result(block, cfg, loc)
                decision = decision.merge(sub)
                injection_targets.extend(targets)

    decision = decision.merge(_score_injection(injection_targets, cfg))
    return body, decision


def extract_untrusted_texts(body: dict[str, Any], cfg: Config) -> list[tuple[str, str]]:
    """Return (text, location) spans considered untrusted, for LLM judging.

    Mirrors which content ``inspect_request`` scans for injection: tool_result
    text/document-text (always), plus user/assistant text when
    ``scan_tool_results_only`` is False. Call this on the body returned by
    ``inspect_request`` so it reflects post-masking content.
    """
    targets: list[tuple[str, str]] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return targets

    for mi, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, str):
            if _should_scan_injection(is_tool_result=False, cfg=cfg):
                targets.append((content, f"messages[{mi}].content"))
            continue
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            loc = f"messages[{mi}].content[{bi}]"
            if block.get("type") == "text":
                if _should_scan_injection(is_tool_result=False, cfg=cfg):
                    targets.append((block.get("text", ""), f"{loc}.text"))
            elif block.get("type") == "tool_result":
                _collect_tool_result_texts(block.get("content"), loc, targets)
    return [(t, l) for t, l in targets if t]


def _collect_tool_result_texts(content: Any, location: str, out: list[tuple[str, str]]) -> None:
    if isinstance(content, str):
        out.append((content, location))
    elif isinstance(content, list):
        for bi, sub in enumerate(content):
            if not isinstance(sub, dict):
                continue
            if sub.get("type") == "text":
                out.append((sub.get("text", ""), f"{location}.content[{bi}].text"))
            elif sub.get("type") == "document" and isinstance(sub.get("source"), dict) \
                    and sub["source"].get("type") == "text":
                out.append((sub["source"].get("data", ""), f"{location}.content[{bi}].source.data"))


def inspect_response(body: dict[str, Any], cfg: Config) -> tuple[Decision, list[dict[str, Any]]]:
    """Scan a model response in place. Returns (overall_decision, tool_calls).

    Each tool_call dict: {"id", "name", "input", "decision"}.
    """
    decision = Decision(action=Action.ALLOW)
    tool_calls: list[dict[str, Any]] = []

    content = body.get("content")
    if not isinstance(content, list):
        return decision, tool_calls

    for bi, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if btype == "text" and cfg.pii.enabled and cfg.pii.scan_output:
            new_text, sub = _mask_pii(block.get("text", ""), cfg, location=f"content[{bi}].text", output=True)
            block["text"] = new_text
            decision = decision.merge(sub)

        elif btype == "tool_use":
            name = block.get("name", "")
            tool_input = block.get("input", {})
            sub = action_detector.classify(name, tool_input, cfg.actions, location=f"content[{bi}]")
            tool_calls.append({"id": block.get("id"), "name": name, "input": tool_input, "decision": sub})
            decision = decision.merge(sub)

    return decision, tool_calls


# --- internals -------------------------------------------------------------


def _mask_pii(text: str, cfg: Config, *, location: str, output: bool = False) -> tuple[str, Decision]:
    """Mask (or flag) PII in a single text span. Pure for the text it gets."""
    if not text or not cfg.pii.enabled:
        return text, Decision(action=Action.ALLOW)

    if cfg.pii.action == Action.MASK or output:
        # Output PII is always masked (we can't ask the model to redo it).
        masked, findings = pii.mask(text, categories=cfg.pii.categories, location=location)
        if findings:
            return masked, Decision(
                action=Action.MASK,
                severity=_top_severity(findings),
                findings=findings,
                reason="Masked PII" + (" (output)" if output else ""),
            )
        return masked, Decision(action=Action.ALLOW)

    findings = pii.scan(text, categories=cfg.pii.categories, location=location)
    if findings:
        return text, Decision(
            action=cfg.pii.action,
            severity=_top_severity(findings),
            findings=findings,
            reason="PII detected",
        )
    return text, Decision(action=Action.ALLOW)


def _should_scan_injection(*, is_tool_result: bool, cfg: Config) -> bool:
    if not cfg.injection.enabled:
        return False
    return is_tool_result or not cfg.injection.scan_tool_results_only


def _scan_tool_result(block: dict[str, Any], cfg: Config, location: str) -> tuple[Decision, list[tuple[str, str]]]:
    """Mask PII in tool_result text and collect injection targets.

    Handles string content, lists of text blocks, and document blocks with a
    text source. Non-text payloads (images, base64 docs) cannot be scanned and
    are flagged as a coverage gap rather than silently passed.
    """
    decision = Decision(action=Action.ALLOW)
    targets: list[tuple[str, str]] = []
    content = block.get("content")

    def handle_text(text: str, loc: str) -> None:
        nonlocal decision
        masked, sub = _mask_pii(text, cfg, location=loc)
        decision = decision.merge(sub)
        if _should_scan_injection(is_tool_result=True, cfg=cfg):
            targets.append((masked, loc))
        return masked

    if isinstance(content, str):
        block["content"] = handle_text(content, location)
        return decision, targets

    if isinstance(content, list):
        for bi, sub_block in enumerate(content):
            if not isinstance(sub_block, dict):
                continue
            stype = sub_block.get("type")
            loc = f"{location}.content[{bi}]"
            if stype == "text":
                sub_block["text"] = handle_text(sub_block.get("text", ""), f"{loc}.text")
            elif stype == "document" and isinstance(sub_block.get("source"), dict) \
                    and sub_block["source"].get("type") == "text":
                src = sub_block["source"]
                src["data"] = handle_text(src.get("data", ""), f"{loc}.source.data")
            else:
                # image / base64 document / unknown — cannot be text-scanned.
                decision = decision.merge(_unscannable(stype or "unknown", loc))
        return decision, targets

    return decision, targets


def _score_injection(targets: list[tuple[str, str]], cfg: Config) -> Decision:
    if not cfg.injection.enabled or not targets:
        return Decision(action=Action.ALLOW)

    findings: list[Finding] = []
    if cfg.injection.aggregate_per_request:
        total = 0
        for text, loc in targets:
            score, fnd = injection.scan(text, location=loc)
            total += score
            findings.extend(fnd)
        return _injection_decision(total, findings, cfg)

    # Per-block scoring (most restrictive across blocks).
    decision = Decision(action=Action.ALLOW)
    for text, loc in targets:
        score, fnd = injection.scan(text, location=loc)
        decision = decision.merge(_injection_decision(score, fnd, cfg))
    return decision


def _injection_decision(score: int, findings: list[Finding], cfg: Config) -> Decision:
    if not findings:
        return Decision(action=Action.ALLOW)
    action = Action.ALLOW
    if score >= cfg.injection.block_threshold:
        action = Action.BLOCK
    elif score >= cfg.injection.review_threshold:
        action = Action.REQUIRE_APPROVAL
    return Decision(
        action=action,
        severity=_top_severity(findings),
        findings=findings,
        reason=f"Prompt-injection score {score}",
    )


def _unscannable(kind: str, location: str) -> Decision:
    return Decision(
        action=Action.ALLOW,
        severity=Severity.INFO,
        findings=[
            Finding(
                detector="coverage",
                severity=Severity.INFO,
                title=f"Unscannable tool_result content: {kind}",
                detail="Non-text payload could not be checked for prompt injection.",
                location=location,
                evidence={"content_type": kind},
            )
        ],
        reason="Unscannable content",
    )


def _top_severity(findings: list[Finding]) -> Severity:
    sev = Severity.INFO
    for f in findings:
        sev = max_severity(sev, f.severity)
    return sev
