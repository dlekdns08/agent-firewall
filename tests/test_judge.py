"""Tests for the LLM-judge layer. The model call is monkeypatched — no network."""

import httpx
import pytest
from fastapi.testclient import TestClient

from agent_firewall import judge as judge_module
from agent_firewall import proxy as proxy_module
from agent_firewall.config import Config
from agent_firewall.models import Action


@pytest.fixture(autouse=True)
def clear_cache():
    judge_module._CACHE.clear()
    yield
    judge_module._CACHE.clear()


def _judge_cfg(**over):
    cfg = Config.load()
    cfg.judge.enabled = True
    cfg.upstream.api_key = "test-key"
    for k, v in over.items():
        setattr(cfg.judge, k, v)
    return cfg


def _patch_model(monkeypatch, verdict, counter=None):
    async def fake(cfg, system, content):
        if counter is not None:
            counter.append(content)
        # allow the verdict to depend on content
        return verdict(content) if callable(verdict) else verdict
    monkeypatch.setattr(judge_module, "_call_model", fake)


# --- direct judge API ------------------------------------------------------

async def test_judge_injection_escalates_novel_attack(monkeypatch):
    cfg = _judge_cfg()
    _patch_model(monkeypatch, {"recommended_action": "block", "severity": "high",
                               "confidence": 0.9, "reason": "obvious jailbreak"})
    d = await judge_module.judge_injection([("some sneaky novel attack", "loc")], cfg)
    assert d.action == Action.BLOCK
    assert d.findings[0].detector == "llm_judge:injection"


async def test_judge_allow_adds_no_findings(monkeypatch):
    cfg = _judge_cfg()
    _patch_model(monkeypatch, {"recommended_action": "allow", "reason": "benign"})
    d = await judge_module.judge_action("get_weather", {"city": "Seoul"}, "loc", cfg)
    assert d.action == Action.ALLOW
    assert d.findings == []


async def test_judge_caches_repeat_calls(monkeypatch):
    cfg = _judge_cfg()
    calls = []
    _patch_model(monkeypatch, {"recommended_action": "block", "reason": "x"}, counter=calls)
    await judge_module.judge_injection([("same text", "a")], cfg)
    await judge_module.judge_injection([("same text", "b")], cfg)
    assert len(calls) == 1  # second served from cache


async def test_judge_fail_open_vs_closed(monkeypatch):
    async def boom(cfg, system, content):
        raise httpx.ConnectError("upstream down")
    monkeypatch.setattr(judge_module, "_call_model", boom)

    open_cfg = _judge_cfg(fail_closed=False)
    d_open = await judge_module.judge_action("delete_file", {}, "loc", open_cfg)
    assert d_open.action == Action.ALLOW

    closed_cfg = _judge_cfg(fail_closed=True)
    d_closed = await judge_module.judge_action("delete_file", {}, "loc", closed_cfg)
    assert d_closed.action == Action.REQUIRE_APPROVAL


# --- proxy integration -----------------------------------------------------

class _Stub:
    def __init__(self, json_body):
        self.json_body = json_body

    def install(self, monkeypatch):
        body = self.json_body

        class FakeClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None, content=None):
                return httpx.Response(200, json=body, request=httpx.Request("POST", url))
        monkeypatch.setattr(proxy_module.httpx, "AsyncClient", FakeClient)


def test_proxy_action_judge_blocks_innocuous_named_tool(monkeypatch):
    """Heuristics allow 'do_stuff'; the LLM judge escalates it to a block."""
    cfg = _judge_cfg(actions="escalate")
    cfg.approval.mode = "auto_deny"
    _patch_model(monkeypatch, lambda c: (
        {"recommended_action": "block", "severity": "critical", "reason": "wipes the database"}
        if "do_stuff" in c else {"recommended_action": "allow", "reason": "ok"}
    ))
    _Stub({
        "id": "m", "type": "message", "role": "assistant", "model": "m",
        "content": [{"type": "tool_use", "id": "t", "name": "do_stuff",
                     "input": {"target": "prod database"}}],
        "stop_reason": "tool_use",
    }).install(monkeypatch)

    client = TestClient(proxy_module.create_app(cfg))
    r = client.post("/v1/messages", json={"model": "m",
                    "messages": [{"role": "user", "content": "go"}]})
    body = r.json()
    assert not any(b.get("type") == "tool_use" for b in body["content"])
    assert any("Blocked tool call" in b.get("text", "") for b in body["content"])


def test_proxy_escalate_skips_already_blocked(monkeypatch):
    """A fork bomb is already blocked by heuristics → judge must NOT be called."""
    cfg = _judge_cfg(actions="escalate")
    cfg.approval.mode = "auto_deny"
    calls = []
    _patch_model(monkeypatch, {"recommended_action": "allow", "reason": "n/a"}, counter=calls)
    _Stub({
        "id": "m", "type": "message", "role": "assistant", "model": "m",
        "content": [{"type": "tool_use", "id": "t", "name": "bash",
                     "input": {"command": "rm -rf /"}}],
        "stop_reason": "tool_use",
    }).install(monkeypatch)

    client = TestClient(proxy_module.create_app(cfg))
    client.post("/v1/messages", json={"model": "m",
                "messages": [{"role": "user", "content": "go"}]})
    assert calls == []  # escalate skipped the already-blocked tool
