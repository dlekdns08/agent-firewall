"""Human-in-the-loop approval + audit logging.

Approval modes (``approval.mode``):
  * ``console``    — print details, block on stdin (y/N), serialized by a lock.
  * ``auto_allow`` — approve everything (staging/tests).
  * ``auto_deny``  — deny everything (safe headless default).
  * ``web``        — hold the request; a human approves/denies via the
                     ``/approvals`` HTTP UI. Works for concurrent server use.
  * ``slack``      — like ``web`` plus a Slack notification with action links.

All decisions can additionally be appended to a JSONL audit log.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel

from .config import Config
from .models import Decision

_console = Console(stderr=True)

# Serializes console prompts (single stdin can answer one question at a time).
_console_lock = asyncio.Lock()


class PendingApproval:
    """A request waiting for an out-of-band (web/slack) decision."""

    def __init__(self, approval_id: str, summary: str, decision: Decision, payload: dict[str, Any],
                 future: "asyncio.Future[bool]") -> None:
        self.id = approval_id
        self.summary = summary
        self.decision = decision
        self.payload = payload
        self.future = future

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "action": self.decision.action.value,
            "severity": self.decision.severity.value,
            "findings": [f.title for f in self.decision.findings],
            "payload": self.payload,
        }


class ApprovalManager:
    """Owns approval policy + the pending-request registry for web/slack modes."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pending: dict[str, PendingApproval] = {}
        self._counter = 0

    async def request(self, *, summary: str, decision: Decision, payload: dict[str, Any]) -> bool:
        mode = self.cfg.approval.mode
        if mode == "auto_allow":
            return True
        if mode == "auto_deny":
            return False
        if mode == "console":
            async with _console_lock:
                return await _console_prompt(self.cfg.approval, summary=summary, decision=decision, payload=payload)
        if mode in ("web", "slack"):
            return await self._wait_web(summary, decision, payload, notify_slack=(mode == "slack"))
        return False  # unknown mode → fail closed

    async def _wait_web(self, summary: str, decision: Decision, payload: dict[str, Any], *, notify_slack: bool) -> bool:
        self._counter += 1
        approval_id = f"ap_{self._counter}"
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        pending = PendingApproval(approval_id, summary, decision, payload, future)
        self.pending[approval_id] = pending

        _console.print(f"[yellow]🛡  approval pending[/yellow] [dim]{approval_id}[/dim]: {summary} "
                       f"→ {self._base_url()}/approvals")
        if notify_slack:
            await self._notify_slack(pending)

        try:
            return await asyncio.wait_for(future, timeout=self.cfg.approval.timeout_seconds)
        except asyncio.TimeoutError:
            return False
        finally:
            self.pending.pop(approval_id, None)

    def resolve(self, approval_id: str, approved: bool) -> bool:
        pending = self.pending.get(approval_id)
        if pending is None or pending.future.done():
            return False
        pending.future.set_result(approved)
        return True

    def list_pending(self) -> list[PendingApproval]:
        return list(self.pending.values())

    def _base_url(self) -> str:
        return self.cfg.approval.public_url or f"http://{self.cfg.server.host}:{self.cfg.server.port}"

    async def _notify_slack(self, pending: PendingApproval) -> None:
        webhook = self.cfg.approval.slack_webhook
        if not webhook:
            return
        base = self._base_url()
        findings = ", ".join(f.title for f in pending.decision.findings) or pending.decision.reason
        text = (
            f"*🛡 agent-firewall approval required* (`{pending.id}`)\n"
            f"{pending.summary}\n"
            f"_{findings}_\n"
            f"✅ Approve: {base}/approvals/{pending.id}/approve\n"
            f"⛔ Deny: {base}/approvals/{pending.id}/deny"
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(webhook, json={"text": text})
        except Exception as exc:  # notification is best-effort
            _console.print(f"[red]slack notify failed:[/red] {exc}")


async def _console_prompt(cfg, *, summary: str, decision: Decision, payload: dict[str, Any]) -> bool:
    findings = "\n".join(
        f"  • [{f.severity.value}] {f.title} — {f.detail}" for f in decision.findings
    )
    _console.print(
        Panel(
            f"[bold yellow]{summary}[/bold yellow]\n\n"
            f"{findings}\n\n"
            f"[dim]{json.dumps(payload, ensure_ascii=False)[:500]}[/dim]\n\n"
            f"[bold]Approve this action?[/bold] [y/N]",
            title="🛡  agent-firewall — approval required",
            border_style="yellow",
        )
    )
    try:
        answer = await asyncio.wait_for(_read_line(), timeout=cfg.timeout_seconds)
    except asyncio.TimeoutError:
        _console.print("[red]Approval timed out → denied[/red]")
        return False
    return answer.strip().lower() in {"y", "yes"}


async def _read_line() -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sys.stdin.readline)


class AuditLog:
    """Append-only JSONL sink for decisions."""

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
