"""Configuration loading for agent-firewall.

Config is a small YAML file. Anything omitted falls back to the bundled
defaults in ``default_policy.yaml`` so the proxy runs out-of-the-box.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .models import Action

_DEFAULT_POLICY_PATH = Path(__file__).with_name("default_policy.yaml")


class PIIConfig(BaseModel):
    enabled: bool = True
    # Which categories to scan for (keys of detectors.pii.PATTERNS).
    categories: list[str] = Field(default_factory=list)  # empty == all
    # What to do when PII is found in messages going to the model.
    action: Action = Action.MASK
    # Also mask PII in the model's OUTPUT text before it returns to the agent.
    scan_output: bool = True


class InjectionConfig(BaseModel):
    enabled: bool = True
    # Only scan content that arrived from outside the agent (tool results,
    # documents). User/system text is trusted by default.
    scan_tool_results_only: bool = True
    # Score at/above which we escalate. Each rule contributes points.
    block_threshold: int = 5
    review_threshold: int = 3
    # Sum injection score across all untrusted blocks in a request rather than
    # scoring each block independently (defeats split-payload evasion).
    aggregate_per_request: bool = True


class ActionRule(BaseModel):
    name: str
    # Regex matched (case-insensitive) against the tool name.
    tool_pattern: str = ".*"
    # Optional regexes matched against the JSON-encoded tool input.
    arg_patterns: list[str] = Field(default_factory=list)
    action: Action = Action.REQUIRE_APPROVAL
    severity: str = "high"
    reason: str = ""


class ActionsConfig(BaseModel):
    enabled: bool = True
    # Default for any tool that matches no rule and is not allowlisted.
    # Set to require_approval/block to run in "allowlist mode": unknown tools
    # are gated unless their name matches one of `allowlist`.
    default_action: Action = Action.ALLOW
    # Regexes for tool names that are always considered safe (start at ALLOW).
    # Dangerous-arg rules can still escalate an allowlisted tool.
    allowlist: list[str] = Field(default_factory=list)
    rules: list[ActionRule] = Field(default_factory=list)


class ApprovalConfig(BaseModel):
    # "console" prompts on stdin; "auto_deny"/"auto_allow" for headless/tests.
    mode: str = "console"
    timeout_seconds: float = 120.0


class UpstreamConfig(BaseModel):
    base_url: str = "https://api.anthropic.com"
    # If set, forces this key regardless of what the client sends.
    api_key: str | None = None
    timeout_seconds: float = 600.0


class JudgeConfig(BaseModel):
    """Optional LLM-based second-stage classifier for injection + actions.

    Costs an extra (cheap, fast) model call, so it is opt-in. The judge can
    only ESCALATE a heuristic verdict (never downgrade it), so a flaky judge
    can't silently weaken the firewall.
    """

    enabled: bool = False
    model: str = "claude-haiku-4-5"
    # off | escalate | always
    #   escalate — call the LLM only when heuristics did NOT already decide
    #              (catches novel attacks + trims false positives)
    #   always   — call the LLM on every item (max coverage, max cost)
    injection: str = "escalate"
    actions: str = "escalate"
    max_chars: int = 4000          # truncate content sent to the judge
    timeout_seconds: float = 20.0
    fail_closed: bool = False      # on judge error: require_approval (True) or allow (False)
    cache: bool = True


class OpenAIUpstreamConfig(BaseModel):
    """Upstream for the OpenAI-compatible shim (/v1/chat/completions)."""

    base_url: str = "https://api.openai.com"
    # If set, forces this key (sent as `Authorization: Bearer ...`).
    api_key: str | None = None
    timeout_seconds: float = 600.0


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    # If set, clients must send this value as the `x-firewall-token` header.
    # Protects the proxy (which holds the upstream API key) from local misuse.
    auth_token: str | None = None


class Config(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    openai_upstream: OpenAIUpstreamConfig = Field(default_factory=OpenAIUpstreamConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    # Append every decision as JSONL here when set.
    audit_log: str | None = None

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        data = _read_yaml(_DEFAULT_POLICY_PATH)
        if path is not None:
            user = _read_yaml(Path(path))
            data = _deep_merge(data, user)
        cfg = cls.model_validate(data)
        # Env overrides for the common deployment cases.
        if env_key := os.getenv("ANTHROPIC_API_KEY"):
            cfg.upstream.api_key = cfg.upstream.api_key or env_key
        if env_oai := os.getenv("OPENAI_API_KEY"):
            cfg.openai_upstream.api_key = cfg.openai_upstream.api_key or env_oai
        if env_token := os.getenv("FIREWALL_AUTH_TOKEN"):
            cfg.server.auth_token = cfg.server.auth_token or env_token
        return cfg


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
