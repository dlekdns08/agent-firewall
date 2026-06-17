"""Anthropic Messages API-compatible reverse proxy.

Point your agent's ``base_url`` at this server and it transparently forwards
to the real Anthropic API, applying guardrails on the way in and out:

  request  → [input guardrails: injection scan, PII mask] → upstream
  upstream → [output guardrails: dangerous-action policy + approval] → agent

Enforcement:
  * input BLOCK            → short-circuit with a synthetic refusal message
  * input REQUIRE_APPROVAL → ask human; deny → synthetic refusal
  * output tool_use BLOCK / denied approval → neutralize that tool_use block
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .approvals import AuditLog, request_approval
from .config import Config
from .engine import inspect_request, inspect_response
from .models import Action, Decision

# Headers we must not forward verbatim to upstream / back to client.
_HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding"}


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
        raw = await request.body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": "body is not valid JSON"}}, status_code=400)

        # --- input guardrails ------------------------------------------------
        body, in_decision = inspect_request(body, cfg)
        _audit(audit, "request", in_decision, {"model": body.get("model")})

        if in_decision.action == Action.BLOCK:
            return _refusal(body, in_decision, "Request blocked by agent-firewall (input guardrail).")
        if in_decision.action == Action.REQUIRE_APPROVAL:
            ok = await request_approval(
                cfg.approval,
                summary="Suspicious content detected in request (possible prompt injection).",
                decision=in_decision,
                payload={"model": body.get("model")},
            )
            if not ok:
                return _refusal(body, in_decision, "Request denied by human reviewer (input guardrail).")

        is_stream = bool(body.get("stream"))

        # --- forward to upstream --------------------------------------------
        upstream_headers = _forward_headers(dict(request.headers), cfg)
        url = cfg.upstream.base_url.rstrip("/") + "/v1/messages"

        if is_stream:
            # Streaming: input guardrails already applied; output action
            # scanning is not enforced on the token stream (documented limit).
            return await _proxy_stream(url, upstream_headers, body, cfg)

        async with httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds) as client:
            upstream = await client.post(url, headers=upstream_headers, json=body)

        if upstream.status_code != 200:
            return Response(content=upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type", "application/json"))

        resp_body = upstream.json()

        # --- output guardrails ----------------------------------------------
        out_decision, tool_calls = inspect_response(resp_body, cfg)
        _audit(audit, "response", out_decision, {"tools": [t["name"] for t in tool_calls]})

        resp_body = await _enforce_tool_calls(resp_body, tool_calls, cfg)
        return JSONResponse(resp_body)

    return app


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

    # If no tool_use blocks remain, the turn is effectively done.
    if not any(b.get("type") == "tool_use" for b in new_content):
        resp_body["stop_reason"] = "end_turn"
    return resp_body


async def _proxy_stream(url: str, headers: dict[str, str], body: dict[str, Any], cfg: Config) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds)

    async def gen():
        try:
            async with client.stream("POST", url, headers=headers, json=body) as upstream:
                async for chunk in upstream.aiter_raw():
                    yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream")


def _forward_headers(headers: dict[str, str], cfg: Config) -> dict[str, str]:
    out = {k: v for k, v in headers.items() if k.lower() not in _HOP_HEADERS}
    if cfg.upstream.api_key:
        out["x-api-key"] = cfg.upstream.api_key
    out.setdefault("anthropic-version", "2023-06-01")
    return out


def _refusal(body: dict[str, Any], decision: Decision, message: str) -> JSONResponse:
    """Return a well-formed Anthropic message that conveys the block."""
    reasons = "; ".join(f.title for f in decision.findings) or decision.reason
    return JSONResponse(
        {
            "id": "msg_firewall_block",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "unknown"),
            "content": [{"type": "text", "text": f"{message} ({reasons})"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    )


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
