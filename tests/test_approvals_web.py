"""Tests for web approval mode + the approval HTTP endpoints + metrics."""

import asyncio
import threading
import time

import httpx
from fastapi.testclient import TestClient

from agent_firewall import proxy as proxy_module
from agent_firewall.config import Config


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


def test_web_approval_approve_releases_request(monkeypatch):
    cfg = Config.load()
    cfg.approval.mode = "web"
    cfg.approval.timeout_seconds = 5
    cfg.upstream.api_key = "k"
    _Stub({"id": "x", "type": "message", "role": "assistant", "model": "m", "stop_reason": "tool_use",
           "content": [{"type": "tool_use", "id": "tu", "name": "delete_file", "input": {"path": "/tmp/x"}}]}
          ).install(monkeypatch)
    client = TestClient(proxy_module.create_app(cfg))

    # The request will block on approval; approve it from another thread.
    def approver():
        for _ in range(50):
            time.sleep(0.05)
            pend = client.get("/approvals.json").json()["pending"]
            if pend:
                client.post(f"/approvals/{pend[0]['id']}/approve")
                return
    t = threading.Thread(target=approver)
    t.start()
    r = client.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "go"}]})
    t.join()

    # Approved → the delete_file tool_use survives (not neutralized).
    assert any(b.get("type") == "tool_use" for b in r.json()["content"])


def test_web_approval_deny_neutralizes(monkeypatch):
    cfg = Config.load()
    cfg.approval.mode = "web"
    cfg.approval.timeout_seconds = 5
    cfg.upstream.api_key = "k"
    _Stub({"id": "x", "type": "message", "role": "assistant", "model": "m", "stop_reason": "tool_use",
           "content": [{"type": "tool_use", "id": "tu", "name": "delete_file", "input": {"path": "/tmp/x"}}]}
          ).install(monkeypatch)
    client = TestClient(proxy_module.create_app(cfg))

    def denier():
        for _ in range(50):
            time.sleep(0.05)
            pend = client.get("/approvals.json").json()["pending"]
            if pend:
                client.post(f"/approvals/{pend[0]['id']}/deny")
                return
    t = threading.Thread(target=denier)
    t.start()
    r = client.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "go"}]})
    t.join()
    assert not any(b.get("type") == "tool_use" for b in r.json()["content"])


def test_approvals_ui_and_dashboard_render():
    cfg = Config.load()
    client = TestClient(proxy_module.create_app(cfg))
    assert client.get("/approvals").status_code == 200
    assert "approvals" in client.get("/approvals").text.lower()
    assert client.get("/dashboard").status_code == 200
    assert client.get("/metrics.json").json()["total_events"] == 0


def test_metrics_aggregate(tmp_path):
    from agent_firewall import metrics

    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        '{"phase":"response","action":"block","severity":"critical","findings":[{"detector":"actions","title":"Dangerous action"}]}\n'
        '{"phase":"request","action":"mask","severity":"medium","findings":[{"detector":"pii","title":"PII masked"}]}\n'
        '{"phase":"request","action":"require_approval","severity":"high","findings":[]}\n'
    )
    s = metrics.aggregate(str(audit))
    assert s["total_events"] == 3
    assert s["blocked"] == 1
    assert s["enforced"] == 2
    assert s["by_detector"]["actions"] == 1
    assert "🛡" in metrics.render_dashboard(s)
