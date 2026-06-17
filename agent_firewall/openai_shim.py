"""OpenAI Chat Completions API compatibility.

Lets agents built on the OpenAI SDK (or any OpenAI-compatible endpoint) sit
behind the same firewall by pointing their base URL at ``/v1``. The detectors,
policy, and LLM judge are format-agnostic; this module only adapts OpenAI's
message / tool_call / SSE shapes to/from them.

OpenAI vs Anthropic shape, in brief:
  * untrusted tool output arrives as ``{"role": "tool", "content": "..."}``
  * the model's tool calls live at ``choices[].message.tool_calls[]`` with
    ``function.name`` and ``function.arguments`` (a JSON *string*)
  * streaming is ``data: {...}`` chunks terminated by ``data: [DONE]``
"""

from __future__ import annotations

import json
from typing import Any

from .config import Config
from .detectors import actions as action_detector
from .engine import mask_pii, score_injection, should_scan_injection, unscannable
from .models import Action, Decision


# --- request --------------------------------------------------------------

def inspect_chat_request(body: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], Decision, list[tuple[str, str]]]:
    """Mask PII + score injection on an OpenAI chat request.

    Returns (new_body, decision, injection_targets). Targets are returned so
    the caller can run the optional LLM injection judge.
    """
    from copy import deepcopy

    body = deepcopy(body)
    decision = Decision(action=Action.ALLOW)
    targets: list[tuple[str, str]] = []

    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, decision, targets

    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        is_tool = role == "tool"  # tool output is untrusted, like Anthropic tool_result
        content = msg.get("content")
        loc = f"messages[{mi}].content"

        if isinstance(content, str):
            masked, sub = mask_pii(content, cfg, location=loc)
            msg["content"] = masked
            decision = decision.merge(sub)
            if should_scan_injection(is_tool_result=is_tool, cfg=cfg):
                targets.append((masked, loc))

        elif isinstance(content, list):
            for bi, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                bloc = f"{loc}[{bi}]"
                if block.get("type") == "text":
                    masked, sub = mask_pii(block.get("text", ""), cfg, location=f"{bloc}.text")
                    block["text"] = masked
                    decision = decision.merge(sub)
                    if should_scan_injection(is_tool_result=is_tool, cfg=cfg):
                        targets.append((masked, f"{bloc}.text"))
                elif is_tool:
                    # Non-text content in a tool message can't be scanned.
                    decision = decision.merge(unscannable(block.get("type", "unknown"), bloc))

    decision = decision.merge(score_injection(targets, cfg))
    return body, decision, targets


# --- response -------------------------------------------------------------

def inspect_chat_response(body: dict[str, Any], cfg: Config) -> tuple[Decision, list[dict[str, Any]]]:
    """Mask output PII + classify tool calls on an OpenAI chat completion.

    Mutates body in place. Each returned tool_call: {id, name, input, decision,
    _choice, _idx} where _choice/_idx locate it for enforcement.
    """
    decision = Decision(action=Action.ALLOW)
    tool_calls: list[dict[str, Any]] = []

    choices = body.get("choices")
    if not isinstance(choices, list):
        return decision, tool_calls

    for ci, choice in enumerate(choices):
        msg = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(msg, dict):
            continue

        if isinstance(msg.get("content"), str) and cfg.pii.enabled and cfg.pii.scan_output:
            masked, sub = mask_pii(msg["content"], cfg, location=f"choices[{ci}].message.content", output=True)
            msg["content"] = masked
            decision = decision.merge(sub)

        for ti, tc in enumerate(msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            name = fn.get("name", "")
            tool_input = _parse_arguments(fn.get("arguments"))
            sub = action_detector.classify(name, tool_input, cfg.actions, location=f"choices[{ci}].tool_calls[{ti}]")
            tool_calls.append({"id": tc.get("id"), "name": name, "input": tool_input,
                               "decision": sub, "_choice": ci, "_idx": ti})
            decision = decision.merge(sub)

    return decision, tool_calls


async def enforce_chat_tool_calls(body: dict[str, Any], tool_calls: list[dict[str, Any]], cfg: Config,
                                  request_approval) -> dict[str, Any]:
    """Drop blocked/denied tool calls from the completion."""
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
        return body

    for choice in body.get("choices", []):
        msg = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(msg, dict) or not msg.get("tool_calls"):
            continue
        kept = [tc for tc in msg["tool_calls"] if tc.get("id") not in blocked_ids]
        dropped = [tc for tc in msg["tool_calls"] if tc.get("id") in blocked_ids]
        if not dropped:
            continue
        note = "; ".join(f"[agent-firewall] Blocked tool call "
                         f"'{tc.get('function', {}).get('name')}': denied by policy/human review."
                         for tc in dropped)
        msg["content"] = (msg.get("content") or "") + ("\n" if msg.get("content") else "") + note
        if kept:
            msg["tool_calls"] = kept
        else:
            msg.pop("tool_calls", None)
            choice["finish_reason"] = "stop"
    return body


def refusal_completion(body: dict[str, Any], decision: Decision, message: str) -> dict[str, Any]:
    reasons = "; ".join(f.title for f in decision.findings) or decision.reason
    return {
        "id": "chatcmpl-firewall-block",
        "object": "chat.completion",
        "model": body.get("model", "unknown"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"{message} ({reasons})"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# --- streaming ------------------------------------------------------------

def parse_openai_sse(raw: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return chunks


def reconstruct_chat(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a chat.completion object from streamed chunks."""
    completion: dict[str, Any] = {}
    # choice index -> assembled choice
    choices: dict[int, dict[str, Any]] = {}
    tool_acc: dict[int, dict[int, dict[str, Any]]] = {}  # choice -> tc index -> {id,name,args}

    for chunk in chunks:
        for k in ("id", "model", "system_fingerprint"):
            if k in chunk:
                completion[k] = chunk[k]
        for ch in chunk.get("choices", []):
            ci = ch.get("index", 0)
            c = choices.setdefault(ci, {"index": ci, "message": {"role": "assistant", "content": ""}})
            delta = ch.get("delta", {})
            if "role" in delta:
                c["message"]["role"] = delta["role"]
            if delta.get("content"):
                c["message"]["content"] += delta["content"]
            for tcd in delta.get("tool_calls", []):
                ti = tcd.get("index", 0)
                acc = tool_acc.setdefault(ci, {}).setdefault(ti, {"id": None, "name": "", "args": ""})
                if tcd.get("id"):
                    acc["id"] = tcd["id"]
                fn = tcd.get("function", {})
                if fn.get("name"):
                    acc["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["args"] += fn["arguments"]
            if ch.get("finish_reason"):
                c["finish_reason"] = ch["finish_reason"]

    for ci, tcs in tool_acc.items():
        built = [{"id": a["id"], "type": "function",
                  "function": {"name": a["name"], "arguments": a["args"]}}
                 for _, a in sorted(tcs.items())]
        if built:
            choices[ci]["message"]["tool_calls"] = built

    completion["object"] = "chat.completion"
    completion["choices"] = [choices[i] for i in sorted(choices)]
    return completion


def serialize_chat(completion: dict[str, Any]) -> str:
    """Serialize a chat.completion back into an OpenAI SSE chunk stream."""
    out: list[str] = []
    base = {"id": completion.get("id", "chatcmpl-firewall"),
            "object": "chat.completion.chunk",
            "model": completion.get("model", "unknown")}

    def emit(choices: list[dict[str, Any]]) -> None:
        out.append("data: " + json.dumps({**base, "choices": choices}, ensure_ascii=False) + "\n\n")

    for choice in completion.get("choices", []):
        ci = choice.get("index", 0)
        msg = choice.get("message", {})
        emit([{"index": ci, "delta": {"role": "assistant"}, "finish_reason": None}])
        if msg.get("content"):
            emit([{"index": ci, "delta": {"content": msg["content"]}, "finish_reason": None}])
        for ti, tc in enumerate(msg.get("tool_calls") or []):
            emit([{"index": ci, "delta": {"tool_calls": [{
                "index": ti, "id": tc.get("id"), "type": "function",
                "function": {"name": tc.get("function", {}).get("name"),
                             "arguments": tc.get("function", {}).get("arguments", "")},
            }]}, "finish_reason": None}])
        emit([{"index": ci, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}])

    out.append("data: [DONE]\n\n")
    return "".join(out)


def _parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments if arguments is not None else {}
