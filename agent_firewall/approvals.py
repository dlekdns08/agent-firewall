"""Human-in-the-loop approval + audit logging.

When a decision is REQUIRE_APPROVAL the proxy calls ``request_approval``.
Modes:
  * ``console``    — print details, block on stdin (y/N) with a timeout.
  * ``auto_allow`` — approve everything (useful for staging/tests).
  * ``auto_deny``  — deny everything (safe default for headless runs).

All decisions can additionally be appended to a JSONL audit log.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from .config import ApprovalConfig
from .models import Decision

_console = Console(stderr=True)


async def request_approval(
    cfg: ApprovalConfig, *, summary: str, decision: Decision, payload: dict[str, Any]
) -> bool:
    """Return True if approved, False if denied/timed out."""
    if cfg.mode == "auto_allow":
        return True
    if cfg.mode == "auto_deny":
        return False
    if cfg.mode != "console":
        # Unknown mode → fail closed.
        return False

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
