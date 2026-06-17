"""Minimal Anthropic SSE (server-sent events) parsing + reconstruction.

Used so the proxy can apply OUTPUT guardrails to streaming responses: buffer
the upstream stream, reconstruct the final message object, evaluate policy,
and then either replay the original bytes (nothing changed) or emit a freshly
serialized stream from the sanitized message (something was blocked/masked).
"""

from __future__ import annotations

import json
from typing import Any


def parse_sse(raw: str) -> list[dict[str, Any]]:
    """Return the list of decoded ``data:`` JSON objects, in order."""
    events: list[dict[str, Any]] = []
    for chunk in raw.split("\n\n"):
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return events


def reconstruct_message(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the final message dict from a parsed event list."""
    message: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    json_buffers: dict[int, str] = {}

    for ev in events:
        etype = ev.get("type")
        if etype == "message_start":
            message = dict(ev.get("message", {}))
            message["content"] = []
        elif etype == "content_block_start":
            idx = ev.get("index", 0)
            blocks[idx] = dict(ev.get("content_block", {}))
            json_buffers[idx] = ""
        elif etype == "content_block_delta":
            idx = ev.get("index", 0)
            delta = ev.get("delta", {})
            if delta.get("type") == "text_delta":
                blocks.setdefault(idx, {"type": "text", "text": ""})
                blocks[idx]["text"] = blocks[idx].get("text", "") + delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                json_buffers[idx] = json_buffers.get(idx, "") + delta.get("partial_json", "")
        elif etype == "content_block_stop":
            idx = ev.get("index", 0)
            buf = json_buffers.get(idx, "")
            if buf and blocks.get(idx, {}).get("type") == "tool_use":
                try:
                    blocks[idx]["input"] = json.loads(buf)
                except json.JSONDecodeError:
                    blocks[idx]["input"] = {}
        elif etype == "message_delta":
            delta = ev.get("delta", {})
            for k, v in delta.items():
                message[k] = v
            if "usage" in ev:
                message["usage"] = {**message.get("usage", {}), **ev["usage"]}

    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


def serialize_message(message: dict[str, Any]) -> str:
    """Serialize a message dict back into a valid Anthropic SSE stream."""
    out: list[str] = []

    def emit(event_type: str, data: dict[str, Any]) -> None:
        out.append(f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

    skeleton = {k: v for k, v in message.items() if k != "content"}
    skeleton["content"] = []
    skeleton.setdefault("type", "message")
    emit("message_start", {"type": "message_start", "message": skeleton})

    for idx, block in enumerate(message.get("content", [])):
        btype = block.get("type")
        if btype == "text":
            emit("content_block_start", {"type": "content_block_start", "index": idx,
                                         "content_block": {"type": "text", "text": ""}})
            emit("content_block_delta", {"type": "content_block_delta", "index": idx,
                                         "delta": {"type": "text_delta", "text": block.get("text", "")}})
            emit("content_block_stop", {"type": "content_block_stop", "index": idx})
        elif btype == "tool_use":
            emit("content_block_start", {"type": "content_block_start", "index": idx,
                                         "content_block": {"type": "tool_use", "id": block.get("id"),
                                                           "name": block.get("name"), "input": {}}})
            emit("content_block_delta", {"type": "content_block_delta", "index": idx,
                                         "delta": {"type": "input_json_delta",
                                                   "partial_json": json.dumps(block.get("input", {}), ensure_ascii=False)}})
            emit("content_block_stop", {"type": "content_block_stop", "index": idx})

    emit("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": message.get("stop_reason", "end_turn"),
                                     "stop_sequence": message.get("stop_sequence")},
                           "usage": message.get("usage", {"output_tokens": 0})})
    emit("message_stop", {"type": "message_stop"})
    return "".join(out)
