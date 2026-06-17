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
def version() -> None:
    """Print version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
