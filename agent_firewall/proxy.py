"""Multi-provider guardrail reverse proxy.

Point your agent's base URL at this server; it forwards to the real provider
and applies the same guardrails on the way in and out:

  request  → [input: injection scan, PII mask] → upstream
  upstream → [output: PII mask, action policy + approval] → agent

Endpoints:
  POST /v1/messages                      Anthropic Messages API
  POST /v1/messages/count_tokens         (passthrough)
  POST /v1/chat/completions              OpenAI Chat Completions (+ Ollama, etc.)
  POST /v1beta/models/{model}:generateContent | :streamGenerateContent   Gemini
  GET  /approvals  /approvals.json       human-in-the-loop approval UI
  POST /approvals/{id}/approve|deny
  GET  /dashboard  /metrics.json         audit metrics

Streaming responses are buffered, reconstructed, run through output guardrails,
then (re)emitted — ``stream: true`` cannot smuggle a dangerous tool call past.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import __version__, gemini_shim, judge, metrics, openai_shim
from .approvals import ApprovalManager, AuditLog
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
    manager = ApprovalManager(cfg)
    app.state.cfg = cfg
    app.state.audit = audit
    app.state.manager = manager

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "approval_mode": cfg.approval.mode}

    # --- Anthropic ---------------------------------------------------------
    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        if (err := _check_auth(request, cfg)) is not None:
            return err
        body = _json(await request.body())
        if isinstance(body, Response):
            return body

        body, in_decision = inspect_request(body, cfg)
        in_decision = await _judge_input(body, in_decision, cfg)
        _audit(audit, "request", in_decision, {"model": body.get("model")})

        refusal = await _gate_input(in_decision, manager, body, _refusal)
        if refusal is not None:
            return refusal

        upstream_headers = _forward_headers(dict(request.headers), cfg)
        url = cfg.upstream.base_url.rstrip("/") + "/v1/messages"

        if body.get("stream"):
            return await _handle_anthropic_stream(url, upstream_headers, body, cfg, audit, manager)

        async with httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds) as client:
            upstream = await client.post(url, headers=upstream_headers, json=body)
        if upstream.status_code != 200:
            return _passthrough(upstream)

        resp_body = upstream.json()
        out_decision, tool_calls = inspect_response(resp_body, cfg)
        out_decision = await _judge_actions(out_decision, tool_calls, cfg)
        _audit(audit, "response", out_decision, {"tools": [t["name"] for t in tool_calls]})
        resp_body = await _enforce_tool_calls(resp_body, tool_calls, manager)
        return JSONResponse(resp_body)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        if (err := _check_auth(request, cfg)) is not None:
            return err
        raw = await request.body()
        headers = _forward_headers(dict(request.headers), cfg)
        url = cfg.upstream.base_url.rstrip("/") + "/v1/messages/count_tokens"
        async with httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds) as client:
            upstream = await client.post(url, headers=headers, content=raw)
        return _passthrough(upstream)

    # --- OpenAI (and OpenAI-compatible: Ollama, vLLM, ...) -----------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        if (err := _check_auth(request, cfg)) is not None:
            return err
        body = _json(await request.body())
        if isinstance(body, Response):
            return body

        body, in_decision, targets = openai_shim.inspect_chat_request(body, cfg)
        in_decision = await _judge_input_targets(targets, in_decision, cfg)
        _audit(audit, "chat.request", in_decision, {"model": body.get("model")})

        refusal = await _gate_input(in_decision, manager, body, _chat_refusal)
        if refusal is not None:
            return refusal

        headers = _forward_openai_headers(dict(request.headers), cfg)
        url = cfg.openai_upstream.base_url.rstrip("/") + "/v1/chat/completions"
        async with httpx.AsyncClient(timeout=cfg.openai_upstream.timeout_seconds) as client:
            upstream = await client.post(url, headers=headers, json=body)
        if upstream.status_code != 200:
            return _passthrough(upstream)

        stream = bool(body.get("stream"))
        completion = (openai_shim.reconstruct_chat(openai_shim.parse_openai_sse(upstream.text))
                      if stream else upstream.json())
        out_decision, tool_calls = openai_shim.inspect_chat_response(completion, cfg)
        out_decision = await _judge_actions(out_decision, tool_calls, cfg)
        _audit(audit, "chat.response" + ("(stream)" if stream else ""), out_decision,
               {"tools": [t["name"] for t in tool_calls]})
        completion = await openai_shim.enforce_chat_tool_calls(completion, tool_calls, cfg, manager.request)
        if stream:
            changed = out_decision.action != Action.ALLOW or bool(out_decision.findings)
            payload = openai_shim.serialize_chat(completion) if changed else upstream.text
            return StreamingResponse(iter([payload]), media_type="text/event-stream")
        return JSONResponse(completion)

    # --- Gemini ------------------------------------------------------------
    @app.post("/v1beta/{rest:path}")
    async def gemini(rest: str, request: Request) -> Response:
        if (err := _check_auth(request, cfg)) is not None:
            return err
        method = rest.split(":")[-1].lower()
        if method not in ("generatecontent", "streamgeneratecontent"):
            return _error("invalid_request_error", f"unsupported gemini method: {rest}", 404)
        body = _json(await request.body())
        if isinstance(body, Response):
            return body
        stream = method == "streamgeneratecontent"

        body, in_decision, targets = gemini_shim.inspect_gemini_request(body, cfg)
        in_decision = await _judge_input_targets(targets, in_decision, cfg)
        _audit(audit, "gemini.request", in_decision, {"model": rest})

        def refuse(b, d, m):
            comp = gemini_shim.refusal_gemini(b, d, m)
            if stream:
                return StreamingResponse(iter([gemini_shim.serialize_gemini(comp)]), media_type="text/event-stream")
            return JSONResponse(comp)

        refusal = await _gate_input(in_decision, manager, body, refuse)
        if refusal is not None:
            return refusal

        headers = _forward_gemini_headers(dict(request.headers), cfg)
        url = cfg.gemini_upstream.base_url.rstrip("/") + "/v1beta/" + rest
        if request.url.query:
            url += "?" + request.url.query
        async with httpx.AsyncClient(timeout=cfg.gemini_upstream.timeout_seconds) as client:
            upstream = await client.post(url, headers=headers, json=body)
        if upstream.status_code != 200:
            return _passthrough(upstream)

        completion = (gemini_shim.reconstruct_gemini(gemini_shim.parse_gemini_sse(upstream.text))
                      if stream else upstream.json())
        out_decision, tool_calls = gemini_shim.inspect_gemini_response(completion, cfg)
        out_decision = await _judge_actions(out_decision, tool_calls, cfg)
        _audit(audit, "gemini.response" + ("(stream)" if stream else ""), out_decision,
               {"tools": [t["name"] for t in tool_calls]})
        completion = await gemini_shim.enforce_gemini_function_calls(completion, tool_calls, cfg, manager.request)
        if stream:
            changed = out_decision.action != Action.ALLOW or bool(out_decision.findings)
            payload = gemini_shim.serialize_gemini(completion) if changed else upstream.text
            return StreamingResponse(iter([payload]), media_type="text/event-stream")
        return JSONResponse(completion)

    # --- approval UI -------------------------------------------------------
    @app.get("/approvals.json")
    async def approvals_json() -> dict[str, Any]:
        return {"pending": [p.to_dict() for p in manager.list_pending()]}

    @app.get("/approvals", response_class=HTMLResponse)
    async def approvals_ui() -> str:
        return _render_approvals(manager)

    @app.api_route("/approvals/{approval_id}/approve", methods=["GET", "POST"], response_class=HTMLResponse)
    async def approve(approval_id: str) -> str:
        ok = manager.resolve(approval_id, True)
        return _resolved_html(approval_id, "approved" if ok else "not found / already decided")

    @app.api_route("/approvals/{approval_id}/deny", methods=["GET", "POST"], response_class=HTMLResponse)
    async def deny(approval_id: str) -> str:
        ok = manager.resolve(approval_id, False)
        return _resolved_html(approval_id, "denied" if ok else "not found / already decided")

    # --- metrics -----------------------------------------------------------
    @app.get("/metrics.json")
    async def metrics_json() -> dict[str, Any]:
        return metrics.aggregate(cfg.audit_log)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        return metrics.render_dashboard(metrics.aggregate(cfg.audit_log))

    return app


# --- streaming (Anthropic) -------------------------------------------------

async def _handle_anthropic_stream(url, headers, body, cfg: Config, audit: AuditLog, manager) -> Response:
    async with httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds) as client:
        upstream = await client.post(url, headers=headers, json=body)
        if upstream.status_code != 200:
            return _passthrough(upstream)
        raw_text = upstream.text

    message = reconstruct_message(parse_sse(raw_text))
    out_decision, tool_calls = inspect_response(message, cfg)
    out_decision = await _judge_actions(out_decision, tool_calls, cfg)
    _audit(audit, "response(stream)", out_decision, {"tools": [t["name"] for t in tool_calls]})
    sanitized = await _enforce_tool_calls(message, tool_calls, manager)

    changed = out_decision.action != Action.ALLOW or bool(out_decision.findings)
    payload = serialize_message(sanitized) if changed else raw_text
    return StreamingResponse(iter([payload]), media_type="text/event-stream")


# --- judge passes ----------------------------------------------------------

async def _judge_input(body: dict[str, Any], in_decision: Decision, cfg: Config) -> Decision:
    if not cfg.judge.enabled or cfg.judge.injection == "off":
        return in_decision
    if cfg.judge.injection == "escalate" and in_decision.action == Action.BLOCK:
        return in_decision
    return await _judge_input_targets(extract_untrusted_texts(body, cfg), in_decision, cfg)


async def _judge_input_targets(targets, in_decision: Decision, cfg: Config) -> Decision:
    if not cfg.judge.enabled or cfg.judge.injection == "off" or not targets:
        return in_decision
    if cfg.judge.injection == "escalate" and in_decision.action == Action.BLOCK:
        return in_decision
    return in_decision.merge(await judge.judge_injection(targets, cfg))


async def _judge_actions(out_decision: Decision, tool_calls: list[dict[str, Any]], cfg: Config) -> Decision:
    if not cfg.judge.enabled or cfg.judge.actions == "off" or not tool_calls:
        return out_decision
    for call in tool_calls:
        if cfg.judge.actions == "escalate" and call["decision"].action != Action.ALLOW:
            continue
        verdict = await judge.judge_action(call["name"], call["input"], f"tool:{call['name']}", cfg)
        call["decision"] = call["decision"].merge(verdict)
        out_decision = out_decision.merge(verdict)
    return out_decision


# --- enforcement -----------------------------------------------------------

async def _gate_input(in_decision: Decision, manager, body, refuse_fn):
    """Shared input gate: returns a refusal Response, or None to proceed."""
    if in_decision.action == Action.BLOCK:
        return refuse_fn(body, in_decision, "Request blocked by agent-firewall (input guardrail).")
    if in_decision.action == Action.REQUIRE_APPROVAL:
        ok = await manager.request(
            summary="Suspicious content in request (possible prompt injection).",
            decision=in_decision, payload={"model": body.get("model")},
        )
        if not ok:
            return refuse_fn(body, in_decision, "Request denied by human reviewer (input guardrail).")
    return None


async def _enforce_tool_calls(resp_body: dict[str, Any], tool_calls: list[dict[str, Any]], manager) -> dict[str, Any]:
    """Neutralize any Anthropic tool_use block that is blocked or denied."""
    blocked_ids: set[str] = set()
    for call in tool_calls:
        decision: Decision = call["decision"]
        if decision.action == Action.BLOCK:
            blocked_ids.add(call["id"])
        elif decision.action == Action.REQUIRE_APPROVAL:
            ok = await manager.request(
                summary=f"Agent wants to call tool '{call['name']}'.",
                decision=decision, payload={"tool": call["name"], "input": call["input"]},
            )
            if not ok:
                blocked_ids.add(call["id"])

    if not blocked_ids:
        return resp_body

    new_content = []
    for block in resp_body.get("content", []):
        if block.get("type") == "tool_use" and block.get("id") in blocked_ids:
            new_content.append({"type": "text",
                "text": f"[agent-firewall] Blocked tool call '{block.get('name')}': "
                        f"this action was denied by policy/human review."})
        else:
            new_content.append(block)
    resp_body["content"] = new_content
    if not any(b.get("type") == "tool_use" for b in new_content):
        resp_body["stop_reason"] = "end_turn"
    return resp_body


# --- headers / helpers -----------------------------------------------------

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


def _forward_openai_headers(headers: dict[str, str], cfg: Config) -> dict[str, str]:
    drop = _HOP_HEADERS | {"x-api-key", "anthropic-version"}
    out = {k: v for k, v in headers.items() if k.lower() not in drop}
    if cfg.openai_upstream.api_key:
        out["authorization"] = f"Bearer {cfg.openai_upstream.api_key}"
    return out


def _forward_gemini_headers(headers: dict[str, str], cfg: Config) -> dict[str, str]:
    drop = _HOP_HEADERS | {"x-api-key", "anthropic-version", "authorization"}
    out = {k: v for k, v in headers.items() if k.lower() not in drop}
    if cfg.gemini_upstream.api_key:
        out["x-goog-api-key"] = cfg.gemini_upstream.api_key
    return out


def _chat_refusal(body: dict[str, Any], decision: Decision, message: str) -> Response:
    completion = openai_shim.refusal_completion(body, decision, message)
    if body.get("stream"):
        return StreamingResponse(iter([openai_shim.serialize_chat(completion)]), media_type="text/event-stream")
    return JSONResponse(completion)


def _refusal(body: dict[str, Any], decision: Decision, message: str) -> Response:
    reasons = "; ".join(f.title for f in decision.findings) or decision.reason
    msg = {
        "id": "msg_firewall_block", "type": "message", "role": "assistant",
        "model": body.get("model", "unknown"),
        "content": [{"type": "text", "text": f"{message} ({reasons})"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    if body.get("stream"):
        return StreamingResponse(iter([serialize_message(msg)]), media_type="text/event-stream")
    return JSONResponse(msg)


def _json(raw: bytes):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _error("invalid_request_error", "body is not valid JSON", 400)


def _passthrough(upstream: httpx.Response) -> Response:
    return Response(content=upstream.content, status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"))


def _error(etype: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": etype, "message": message}}, status_code=status)


def _audit(audit: AuditLog, phase: str, decision: Decision, extra: dict[str, Any]) -> None:
    if decision.action == Action.ALLOW and not decision.findings:
        return
    audit.write({
        "phase": phase, "action": decision.action.value, "severity": decision.severity.value,
        "reason": decision.reason, "findings": [f.model_dump() for f in decision.findings], **extra,
    })


def _render_approvals(manager) -> str:
    pending = manager.list_pending()
    if not pending:
        rows = "<p class=dim>No pending approvals.</p>"
    else:
        rows = ""
        for p in pending:
            findings = ", ".join(f.title for f in p.decision.findings) or p.decision.reason
            rows += (f"<div class=item><div><b>{_h(p.summary)}</b> "
                     f"<span class=tag>{p.decision.action.value}</span><br>"
                     f"<span class=dim>{_h(findings)}</span><br>"
                     f"<code>{_h(json.dumps(p.payload, ensure_ascii=False)[:300])}</code></div>"
                     f"<div class=btns>"
                     f"<form method=post action='/approvals/{p.id}/approve'><button class=ok>Approve</button></form>"
                     f"<form method=post action='/approvals/{p.id}/deny'><button class=no>Deny</button></form>"
                     f"</div></div>")
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>agent-firewall · approvals</title><meta http-equiv=refresh content=5>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}}
 .item{{display:flex;justify-content:space-between;gap:1rem;background:#f5f5f7;border-radius:12px;padding:1rem;margin:.6rem 0}}
 .btns{{display:flex;gap:.5rem;align-items:center}} button{{padding:.5rem 1rem;border:0;border-radius:8px;cursor:pointer;color:#fff}}
 .ok{{background:#2e9e4f}} .no{{background:#c0362c}} .dim{{color:#999}} code{{font-size:12px}}
 .tag{{background:#ffe9b3;border-radius:6px;padding:.1rem .4rem;font-size:12px}}
</style></head><body>
<h1>🛡 pending approvals</h1>{rows}
<p class=dim>auto-refreshes every 5s</p></body></html>"""


def _resolved_html(approval_id: str, status: str) -> str:
    return (f"<!doctype html><meta charset=utf-8><body style='font:15px sans-serif;margin:3rem'>"
            f"<h2>🛡 agent-firewall</h2><p><code>{_h(approval_id)}</code> → <b>{_h(status)}</b></p>"
            f"<p><a href='/approvals'>← back to approvals</a></p></body>")


def _h(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
