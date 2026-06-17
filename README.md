# agent-firewall 🛡

**A runtime guardrail proxy for LLM agents.** It sits between your agent and
the model API as a drop-in reverse proxy and enforces safety policy on every
turn — no SDK, no framework lock-in. Change one environment variable and your
agent is firewalled.

```
agent ──▶ [ agent-firewall ] ──▶ Anthropic API
            │  input:  prompt-injection scan, PII masking
            │  output: dangerous-action policy + human approval
            ▼
        audit log (JSONL)
```

## Why

Once agents start *taking actions* — deleting files, sending email, moving
money, running shell — the LLM call itself becomes an attack surface:

- **Prompt injection** rides in on tool results / retrieved documents.
- **PII** leaks upward into the model and logs.
- **Dangerous tool calls** execute with no human in the loop.

`agent-firewall` is a single, framework-independent enforcement point for all
three. Because it speaks the Anthropic Messages API, *any* agent — LangChain,
a raw SDK loop, or one of your own frameworks — works unchanged.

## Install & run

```bash
cd agent-firewall
uv sync --extra dev          # create venv + install
uv run agent-firewall serve  # starts on http://127.0.0.1:8787
```

Point your agent at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY=sk-ant-...   # forwarded upstream
```

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8787")   # that's the whole change
```

## What it does

| Direction | Guardrail | Default action |
|-----------|-----------|----------------|
| Request (input) | **PII masking** — emails, SSNs, cards (Luhn-checked), API tokens, AWS keys | mask in place |
| Request (input) | **Prompt-injection scan** of tool-result/untrusted content | block (score ≥ 5) / approve (≥ 3) |
| Response (output) | **Dangerous-action policy** — fs-destructive, shell, money, email, secrets | require approval / block |

When an action needs approval, the proxy prompts on the console (`y/N`); deny
or timeout neutralizes the tool call before it ever reaches your agent. Set
`approval.mode: auto_deny` for headless/CI safety.

## LLM judge (optional second stage)

Heuristics are cheap but pattern-bound. Enable an LLM second stage to catch
what they miss — **novel prompt injections** and **dangerous tools with
innocuous names** — using a small, fast model:

```yaml
judge:
  enabled: true
  model: "claude-haiku-4-5"
  injection: escalate   # off | escalate | always
  actions: escalate
  fail_closed: false    # on judge error: require_approval (true) or allow (false)
```

- **escalate** (default) only calls the model when the heuristics didn't
  already decide — so clear positives cost nothing and the model resolves the
  gray zone. **always** judges every item.
- The judge is **escalate-only by construction**: verdicts merge with
  most-restrictive-wins, so it can tighten a decision but never weaken one. A
  flaky judge can't silently disable the firewall.
- Verdicts are cached by content hash; the judge calls the upstream API
  directly (not through the proxy), so there's no recursion.

## Configure

Everything is policy-driven. Copy the bundled defaults and edit:

```bash
cp agent_firewall/default_policy.yaml my_policy.yaml
uv run agent-firewall serve --config my_policy.yaml
```

See [`default_policy.yaml`](agent_firewall/default_policy.yaml) for the full
rule schema (tool-name regex + arg regex → action/severity).

## Dry-run a payload (no network)

```bash
uv run agent-firewall check examples/sample_request.json --kind request
uv run agent-firewall check examples/sample_response.json --kind response
```

## Test

```bash
uv run pytest -q
```

## Hardening (what's enforced)

- **Streaming** is fully covered: the SSE stream is buffered, the message
  reconstructed, output guardrails applied, then replayed — `stream: true`
  cannot smuggle a dangerous tool call past the policy.
- **Allowlist mode**: set `actions.default_action: require_approval` and list
  safe tools in `actions.allowlist` so unknown tools are gated by default.
- **Arg-based rules** catch dangerous intent regardless of tool name
  (e.g. `{"name":"fs_op","input":{"op":"delete"}}`).
- **Output PII** in the model's reply is masked (`pii.scan_output`).
- **Injection scoring is aggregated** across all untrusted blocks
  (`injection.aggregate_per_request`) so split payloads can't dodge the threshold.
- Non-text tool results (images, base64 docs) are flagged as a coverage gap in
  the audit log rather than silently passed.
- Concurrent **approvals are serialized** (no stdin races).
- Set `server.auth_token` (or `FIREWALL_AUTH_TOKEN`) to require an
  `x-firewall-token` header; the proxy binds to `127.0.0.1` by default.

## Scope & limits (honest)

- Injection detection is **heuristic** by default — it catches common
  documented shapes, not a guaranteed classifier. Enable the LLM judge (above)
  for novel-attack coverage. Treat it all as defense-in-depth.
- The `check` CLI runs heuristics only (no network); the LLM judge runs in the
  live proxy.
- Streaming responses are buffered (not incrementally relayed), so the client
  sees the reply once the full turn is available — a latency/UX trade-off for
  full output enforcement.
- Image / base64-document content can't be text-scanned (flagged, not blocked).
- Anthropic Messages API only for now; OpenAI-compatible shim is a natural next step.

## Layout

```
agent_firewall/
  proxy.py        FastAPI reverse proxy + enforcement
  engine.py       walks request/response payloads, applies detectors
  detectors/      pii.py · injection.py · actions.py
  judge.py        optional LLM second-stage classifier (escalate-only)
  streaming.py    SSE parse / reconstruct / serialize
  approvals.py    human-in-the-loop + JSONL audit log
  config.py       policy loading (YAML, deep-merged over defaults)
  cli.py          serve · check · version
```
