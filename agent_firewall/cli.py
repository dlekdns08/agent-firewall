"""Command-line interface for agent-firewall."""

from __future__ import annotations

import json
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config
from .engine import inspect_request, inspect_response

app = typer.Typer(help="Runtime guardrail proxy for LLM agents.", no_args_is_help=True)
console = Console()


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Bind host (overrides config)."),
    port: Optional[int] = typer.Option(None, help="Bind port (overrides config)."),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to policy YAML."),
) -> None:
    """Start the guardrail proxy (Anthropic Messages API compatible)."""
    cfg = Config.load(config)
    host = host or cfg.server.host
    port = port or cfg.server.port
    console.print(f"[bold green]agent-firewall {__version__}[/bold green]")
    console.print(f"  upstream      : {cfg.upstream.base_url}")
    console.print(f"  approval mode : {cfg.approval.mode}")
    console.print(f"  auth token    : {'set' if cfg.server.auth_token else 'none (open on this host)'}")
    console.print(f"  listening on  : http://{host}:{port}")
    console.print(f"\n  Point your agent at [cyan]ANTHROPIC_BASE_URL=http://{host}:{port}[/cyan]\n")

    # Import here so the app picks up the resolved config.
    from .proxy import create_app

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


@app.command()
def check(
    file: str = typer.Argument(..., help="JSON file: an Anthropic request or response body."),
    kind: str = typer.Option("request", help="'request' or 'response'."),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to policy YAML."),
) -> None:
    """Dry-run the guardrails against a saved payload (no network)."""
    cfg = Config.load(config)
    body = json.loads(open(file, encoding="utf-8").read())

    if kind == "request":
        _, decision = inspect_request(body, cfg)
    elif kind == "response":
        decision, _ = inspect_response(body, cfg)
    else:
        raise typer.BadParameter("kind must be 'request' or 'response'")

    table = Table(title=f"agent-firewall :: {kind}")
    table.add_column("severity")
    table.add_column("detector")
    table.add_column("title")
    table.add_column("location")
    for f in decision.findings:
        table.add_row(f.severity.value, f.detector, f.title, f.location)
    console.print(table)
    color = {"allow": "green", "mask": "cyan", "require_approval": "yellow", "block": "red"}.get(
        decision.action.value, "white"
    )
    console.print(f"\nVerdict: [bold {color}]{decision.action.value.upper()}[/bold {color}]  "
                  f"(severity={decision.severity.value})  {decision.reason}")


@app.command()
def stats(
    audit: Optional[str] = typer.Option(None, help="Path to audit JSONL (defaults to config.audit_log)."),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to policy YAML."),
) -> None:
    """Summarize the audit log: block rate, actions, detectors, top findings."""
    from . import metrics

    cfg = Config.load(config)
    path = audit or cfg.audit_log
    if not path:
        console.print("[red]No audit log configured.[/red] Set audit_log in your policy or pass --audit.")
        raise typer.Exit(1)
    s = metrics.aggregate(path)
    console.print(f"[bold]{s['total_events']}[/bold] events · "
                  f"[red]{s['blocked']}[/red] blocked · "
                  f"{s['enforced']} enforced · block rate [bold]{int(s['block_rate']*100)}%[/bold]\n")

    def table(title, d):
        t = Table(title=title)
        t.add_column("key"); t.add_column("count", justify="right")
        for k, v in sorted(d.items(), key=lambda x: -x[1]):
            t.add_row(str(k), str(v))
        console.print(t)

    table("By action", s["by_action"])
    table("By detector", s["by_detector"])
    if s["top_findings"]:
        t = Table(title="Top findings")
        t.add_column("count", justify="right"); t.add_column("title")
        for title, v in s["top_findings"]:
            t.add_row(str(v), title)
        console.print(t)


@app.command()
def smoke(
    live: bool = typer.Option(False, "--live", help="Hit a running proxy + real provider (needs a key)."),
    base_url: str = typer.Option("http://127.0.0.1:8787", help="Running proxy URL (live mode)."),
    provider: str = typer.Option("anthropic", help="anthropic | openai (live mode)."),
) -> None:
    """Run an end-to-end smoke test (mock by default; --live for real calls)."""
    import os

    from . import smoke as smoke_mod

    if live:
        env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
        key = os.getenv(env) if env else None
        if not key:
            console.print(f"[red]--live needs {env} set.[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]Live smoke[/bold] → {base_url} ({provider})")
        results = smoke_mod.run_live(base_url, provider, key)
    else:
        console.print("[bold]Mock smoke[/bold] (no network; full proxy pipeline)")
        results = smoke_mod.run_mock()

    t = Table()
    t.add_column(""); t.add_column("check"); t.add_column("detail", overflow="fold")
    passed = 0
    for name, ok, detail in results:
        passed += ok
        t.add_row("[green]✓[/green]" if ok else "[red]✗[/red]", name, detail)
    console.print(t)
    total = len(results)
    color = "green" if passed == total else "red"
    console.print(f"\n[bold {color}]{passed}/{total} passed[/bold {color}]")
    if passed != total:
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
