"""Google Gemini (generateContent) compatibility.

Adapts Gemini's ``generateContent`` / ``streamGenerateContent`` shapes to the
shared detectors + policy + judge. Gemini differs from both other formats:

  * messages are ``contents[].parts[]`` where a part is ``{text}``,
    ``{functionCall:{name,args}}`` (model), or ``{functionResponse:{name,
    response}}`` (untrusted tool output)
  * function-call args are a JSON *object* (not a string), and calls have no id
  * the model's reply lives at ``candidates[].content.parts[]``
  * streaming is a sequence of GenerateContentResponse objects

We synthesize stable ids as ``"{candidate}:{part}"`` for enforcement.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .config import Config
from .detectors import actions as action_detector
from .engine import mask_pii, score_injection, should_scan_injection
from .models import Action, Decision


# --- request --------------------------------------------------------------

def inspect_gemini_request(body: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], Decision, list[tuple[str, str]]]:
    body = deepcopy(body)
    decision = Decision(action=Action.ALLOW)
    targets: list[tuple[str, str]] = []

    for ci, content in enumerate(body.get("contents", []) or []):
        if not isinstance(content, dict):
            continue
        for pi, part in enumerate(content.get("parts", []) or []):
            if not isinstance(part, dict):
                continue
            loc = f"contents[{ci}].parts[{pi}]"
            if "text" in part:
                masked, sub = mask_pii(part.get("text") or "", cfg, location=f"{loc}.text")
                part["text"] = masked
                decision = decision.merge(sub)
                if should_scan_injection(is_tool_result=False, cfg=cfg):
                    targets.append((masked, f"{loc}.text"))
            elif "functionResponse" in part:
                # Tool output — untrusted. Mask PII in string leaves and scan it.
                fr = part["functionResponse"]
                resp = fr.get("response")
                masked_resp, sub = _mask_struct(resp, cfg, f"{loc}.functionResponse.response")
                fr["response"] = masked_resp
                decision = decision.merge(sub)
                if should_scan_injection(is_tool_result=True, cfg=cfg):
                    targets.append((_stringify(masked_resp), f"{loc}.functionResponse"))

    # systemInstruction is author-controlled (trusted) — mask PII, don't scan.
    si = body.get("systemInstruction")
    if isinstance(si, dict):
        for pi, part in enumerate(si.get("parts", []) or []):
            if isinstance(part, dict) and "text" in part:
                masked, sub = mask_pii(part.get("text") or "", cfg, location=f"systemInstruction.parts[{pi}].text")
                part["text"] = masked
                decision = decision.merge(sub)

    decision = decision.merge(score_injection(targets, cfg))
    return body, decision, targets


# --- response -------------------------------------------------------------

def inspect_gemini_response(body: dict[str, Any], cfg: Config) -> tuple[Decision, list[dict[str, Any]]]:
    decision = Decision(action=Action.ALLOW)
    tool_calls: list[dict[str, Any]] = []

    for ci, cand in enumerate(body.get("candidates", []) or []):
        content = cand.get("content") if isinstance(cand, dict) else None
        if not isinstance(content, dict):
            continue
        for pi, part in enumerate(content.get("parts", []) or []):
            if not isinstance(part, dict):
                continue
            if "text" in part and cfg.pii.enabled and cfg.pii.scan_output:
                masked, sub = mask_pii(part["text"] or "", cfg,
                                       location=f"candidates[{ci}].parts[{pi}].text", output=True)
                part["text"] = masked
                decision = decision.merge(sub)
            elif "functionCall" in part:
                fc = part["functionCall"]
                name = fc.get("name", "")
                args = fc.get("args", {})
                sub = action_detector.classify(name, args, cfg.actions,
                                               location=f"candidates[{ci}].parts[{pi}]")
                tool_calls.append({"id": f"{ci}:{pi}", "name": name, "input": args,
                                   "decision": sub, "_cand": ci, "_part": pi})
                decision = decision.merge(sub)

    return decision, tool_calls


async def enforce_gemini_function_calls(body: dict[str, Any], tool_calls: list[dict[str, Any]],
                                        cfg: Config, approve) -> dict[str, Any]:
    blocked: set[str] = set()
    for call in tool_calls:
        decision: Decision = call["decision"]
        if decision.action == Action.BLOCK:
            blocked.add(call["id"])
        elif decision.action == Action.REQUIRE_APPROVAL:
            ok = await approve(summary=f"Agent wants to call tool '{call['name']}'.",
                               decision=decision, payload={"tool": call["name"], "input": call["input"]})
            if not ok:
                blocked.add(call["id"])

    if not blocked:
        return body

    for call in tool_calls:
        if call["id"] not in blocked:
            continue
        ci, pi = call["_cand"], call["_part"]
        try:
            parts = body["candidates"][ci]["content"]["parts"]
            parts[pi] = {"text": f"[agent-firewall] Blocked tool call '{call['name']}': "
                                 f"denied by policy/human review."}
        except (KeyError, IndexError, TypeError):
            continue
    return body


def refusal_gemini(body: dict[str, Any], decision: Decision, message: str) -> dict[str, Any]:
    reasons = "; ".join(f.title for f in decision.findings) or decision.reason
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": f"{message} ({reasons})"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
    }


# --- streaming ------------------------------------------------------------

def parse_gemini_sse(raw: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return chunks


def reconstruct_gemini(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge streamed GenerateContentResponse chunks into one response."""
    result: dict[str, Any] = {}
    cand_parts: dict[int, list[dict[str, Any]]] = {}
    cand_text: dict[int, str] = {}
    finish: dict[int, Any] = {}

    for chunk in chunks:
        if "usageMetadata" in chunk:
            result["usageMetadata"] = chunk["usageMetadata"]
        for cand in chunk.get("candidates", []):
            ci = cand.get("index", 0)
            content = cand.get("content", {})
            for part in content.get("parts", []) or []:
                if "text" in part:
                    cand_text[ci] = cand_text.get(ci, "") + (part.get("text") or "")
                else:
                    cand_parts.setdefault(ci, []).append(part)
            if cand.get("finishReason"):
                finish[ci] = cand["finishReason"]

    candidates = []
    for ci in sorted(set(cand_parts) | set(cand_text) | set(finish)):
        parts: list[dict[str, Any]] = []
        if cand_text.get(ci):
            parts.append({"text": cand_text[ci]})
        parts.extend(cand_parts.get(ci, []))
        candidates.append({"index": ci, "content": {"role": "model", "parts": parts},
                           "finishReason": finish.get(ci, "STOP")})
    result["candidates"] = candidates
    return result


def serialize_gemini(body: dict[str, Any]) -> str:
    """Emit a single SSE chunk carrying the full response (degenerate stream)."""
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


# --- helpers --------------------------------------------------------------

def _mask_struct(value: Any, cfg: Config, location: str) -> tuple[Any, Decision]:
    """Recursively mask PII in string leaves of a JSON-ish structure."""
    decision = Decision(action=Action.ALLOW)
    if isinstance(value, str):
        masked, sub = mask_pii(value, cfg, location=location)
        return masked, sub
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            mv, sub = _mask_struct(v, cfg, f"{location}.{k}")
            out[k] = mv
            decision = decision.merge(sub)
        return out, decision
    if isinstance(value, list):
        out_list = []
        for i, v in enumerate(value):
            mv, sub = _mask_struct(v, cfg, f"{location}[{i}]")
            out_list.append(mv)
            decision = decision.merge(sub)
        return out_list, decision
    return value, decision


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
