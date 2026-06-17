"""Anthropic Messages API-compatible reverse proxy.

Point your agent's ``base_url`` at this server and it transparently forwards
to the real Anthropic API, applying guardrails on the way in and out:

  request  → [input guardrails: injection scan, PII mask] → upstream
  upstream → [output guardrails: PII mask, action policy + approval] → agent

Enforcement:
  * input BLOCK            → short-circuit with a synthetic refusal message
  * input REQUIRE_APPROVAL → ask human; deny → synthetic refusal
  * output tool_use BLOCK / denied approval → neutralize that tool_use block

Streaming responses are buffered, reconstructed, and run through the same
output guardrails before being (re)emitted, so dangerous tool calls cannot
slip through by setting ``stream: true``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__, judge, openai_shim
from .approvals import AuditLog, request_approval
from .config import Config
from .engine import extract_untrusted_texts, inspect_request, inspect_response
from .models import Action, Decision
from .streaming import parse_sse, reconstruct_message, serialize_message

# Headers we must not forward verbatim to upstream / back to client.
_HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "x-firewall-token"}


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config.load()
    app = FastAPI(title="agent-firewall", version=__version__)
    audit = AuditLog(cfg.audit_log)
    app.state.cfg = cfg
    app.state.audit = audit

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "approval_mode": cfg.approval.mode}

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        if (err := _check_auth(request, cfg)) is not None:
            return err

        raw = await request.body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return _error("invalid_request_error", "body is not valid JSON", 400)

        # --- input guardrails ------------------------------------------------
        body, in_decision = inspect_request(body, cfg)
        in_decision = await _judge_input(body, in_decision, cfg)
        _audit(audit, "request", in_decision, {"model": body.get("model")})

        if in_decision.action == Action.BLOCK:
            return _refusal(body, in_decision, "Request blocked by agent-firewall (input guardrail).")
        if in_decision.action == Action.REQUIRE_APPROVAL:
            ok = await request_approval(
                cfg.approval,
                summary="Suspicious content in request (possible prompt injection).",
                decision=in_decision,
                payload={"model": body.get("model")},
            )
            if not ok:
                return _refusal(body, in_decision, "Request denied by human reviewer (input guardrail).")

        is_stream = bool(body.get("stream"))
        upstream_headers = _forward_headers(dict(request.headers), cfg)
        url = cfg.upstream.base_url.rstrip("/") + "/v1/messages"

        if is_stream:
            return await _handle_stream(url, upstream_headers, body, cfg, audit)

        async with httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds) as client:
            upstream = await client.post(url, headers=upstream_headers, json=body)

        if upstream.status_code != 200:
            return Response(content=upstream.content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type", "application/json"))

        resp_body = upstream.json()
        out_decision, tool_calls = inspect_response(resp_body, cfg)
        out_decision = await _judge_actions(out_decision, tool_calls, cfg)
        _audit(audit, "response", out_decision, {"tools": [t["name"] for t in tool_calls]})
        resp_body = await _enforce_tool_calls(resp_body, tool_calls, cfg)
        return JSONResponse(resp_body)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        # Token counting is read-only; forward with the same auth + key handling.
        if (err := _check_auth(request, cfg)) is not None:
            return err
        raw = await request.body()
        headers = _forward_headers(dict(request.headers), cfg)
        url = cfg.upstream.base_url.rstrip("/") + "/v1/messages/count_tokens"
        async with httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds) as client:
            upstream = await client.post(url, headers=headers, content=raw)
        return Response(content=upstream.content, status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type", "application/json"))

    return app


async def _handle_stream(url, headers, body, cfg: Config, audit: AuditLog) -> Response:
    """Buffer the SSE stream, enforce output guardrails, then (re)emit."""
    async with httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds) as client:
        upstream = await client.post(url, headers=headers, json=body)
        if upstream.status_code != 200:
            return Response(content=upstream.content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type", "application/json"))
        raw_text = upstream.text

    events = parse_sse(raw_text)
    message = reconstruct_message(events)

    out_decision, tool_calls = inspect_response(message, cfg)
    out_decision = await _judge_actions(out_decision, tool_calls, cfg)
    _audit(audit, "response(stream)", out_decision, {"tools": [t["name"] for t in tool_calls]})
    sanitized = await _enforce_tool_calls(message, tool_calls, cfg)

    changed = out_decision.action != Action.ALLOW or bool(out_decision.findings)
    payload = serialize_message(sanitized) if changed else raw_text
    return StreamingResponse(iter([payload]), media_type="text/event-stream")


async def _judge_input(body: dict[str, Any], in_decision: Decision, cfg: Config) -> Decision:
    """Optional LLM injection judging on untrusted request content."""
    if not cfg.judge.enabled or cfg.judge.injection == "off":
        return in_decision
    # escalate: skip if heuristics already blocked (nothing left to escalate).
    if cfg.judge.injection == "escalate" and in_decision.action == Action.BLOCK:
        return in_decision
    targets = extract_untrusted_texts(body, cfg)
    if not targets:
        return in_decision
    return in_decision.merge(await judge.judge_injection(targets, cfg))


async def _judge_actions(out_decision: Decision, tool_calls: list[dict[str, Any]], cfg: Config) -> Decision:
    """Optional LLM judging of each tool call. Mutates call decisions in place."""
    if not cfg.judge.enabled or cfg.judge.actions == "off" or not tool_calls:
        return out_decision
    for call in tool_calls:
        # escalate: only judge tools the heuristics let through.
        if cfg.judge.actions == "escalate" and call["decision"].action != Action.ALLOW:
            continue
        verdict = await judge.judge_action(call["name"], call["input"], f"tool:{call['name']}", cfg)
        call["decision"] = call["decision"].merge(verdict)
        out_decision = out_decision.merge(verdict)
    return out_decision


async def _enforce_tool_calls(
    resp_body: dict[str, Any], tool_calls: list[dict[str, Any]], cfg: Config
) -> dict[str, Any]:
    """Neutralize any tool_use block that is blocked or denied."""
    blocked_ids: set[str] = set()

    for call in tool_calls:
        decision: Decision = call["decision"]
        if decision.action == Action.BLOCK:
            blocked_ids.add(call["id"])
        elif decision.action == Action.REQUIRE_APPROVAL:
            ok = await request_approval(
                cfg.approval,
                summary=f"Agent wants to call tool '{call['name']}'.",
                decision=decision,
                payload={"tool": call["name"], "input": call["input"]},
            )
            if not ok:
                blocked_ids.add(call["id"])

    if not blocked_ids:
        return resp_body

    new_content = []
    for block in resp_body.get("content", []):
        if block.get("type") == "tool_use" and block.get("id") in blocked_ids:
            new_content.append({
                "type": "text",
                "text": f"[agent-firewall] Blocked tool call '{block.get('name')}': "
                        f"this action was denied by policy/human review.",
            })
        else:
            new_content.append(block)
    resp_body["content"] = new_content

    if not any(b.get("type") == "tool_use" for b in new_content):
        resp_body["stop_reason"] = "end_turn"
    return resp_body


def _check_auth(request: Request, cfg: Config) -> Response | None:
    if not cfg.server.auth_token:
        return None
    if request.headers.get("x-firewall-token") == cfg.server.auth_token:
        return None
    return _error("authentication_error", "missing or invalid x-firewall-token", 401)


def _forward_headers(headers: dict[str, str], cfg: Config) -> dict[str, str]:
    out = {k: v for k, v in headers.items() if k.lower() not in _HOP_HEADERS}
    if cfg.upstream.api_key:
        out["x-api-key"] = cfg.upstream.api_key
    out.setdefault("anthropic-version", "2023-06-01")
    return out


def _refusal(body: dict[str, Any], decision: Decision, message: str) -> Response:
    """Return a well-formed Anthropic message that conveys the block.

    Honors stream mode so a streaming client still gets a valid SSE stream.
    """
    reasons = "; ".join(f.title for f in decision.findings) or decision.reason
    msg = {
        "id": "msg_firewall_block",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "unknown"),
        "content": [{"type": "text", "text": f"{message} ({reasons})"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    if body.get("stream"):
        return StreamingResponse(iter([serialize_message(msg)]), media_type="text/event-stream")
    return JSONResponse(msg)


def _error(etype: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": etype, "message": message}}, status_code=status)


def _audit(audit: AuditLog, phase: str, decision: Decision, extra: dict[str, Any]) -> None:
    if decision.action == Action.ALLOW and not decision.findings:
        return
    audit.write({
        "phase": phase,
        "action": decision.action.value,
        "severity": decision.severity.value,
        "reason": decision.reason,
        "findings": [f.model_dump() for f in decision.findings],
        **extra,
    })
