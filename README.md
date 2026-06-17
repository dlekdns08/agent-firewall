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
three. It speaks the **Anthropic Messages API**, the **OpenAI Chat Completions
API** (incl. Ollama/vLLM/LM Studio), and **Google Gemini** — so *any* agent
(LangChain, a raw SDK loop, or one of your own frameworks) works unchanged.

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

### OpenAI-compatible too

The same proxy exposes `/v1/chat/completions`, so OpenAI-SDK agents (and any
OpenAI-compatible endpoint) are firewalled identically — same detectors,
policy, and LLM judge:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="sk-...")
```

```bash
export OPENAI_API_KEY=sk-...     # forwarded upstream as a Bearer token
# point openai_upstream.base_url at any OpenAI-compatible server in your policy
```

### Ollama / vLLM / LM Studio

These all expose the OpenAI Chat Completions API, so they're firewalled via the
same `/v1/chat/completions` route — just repoint the upstream:

```yaml
openai_upstream:
  base_url: "http://localhost:11434"   # Ollama
```

### Gemini

Google Gemini's native `generateContent` API is supported directly:

```python
import google.generativeai as genai
genai.configure(api_key="...", transport="rest",
                client_options={"api_endpoint": "http://127.0.0.1:8787"})
```

```bash
export GEMINI_API_KEY=...   # forwarded upstream as x-goog-api-key
```

## Human-in-the-loop approval

`approval.mode` chooses how `require_approval` decisions are resolved:

| mode | behavior |
| --- | --- |
| `console` | prompt on stdin (y/N), serialized across concurrent requests |
| `web` | hold the request; approve/deny in the **`/approvals`** browser UI |
| `slack` | like `web`, plus a Slack notification with approve/deny links (`approval.slack_webhook`) |
| `auto_allow` / `auto_deny` | headless / CI |

`web`/`slack` work for concurrent server deployments — each pending request
waits on its own decision and is released the moment a human clicks.

## Metrics dashboard

Set `audit_log` in your policy, then:

- **`/dashboard`** — live HTML (block rate, actions, detectors, top findings)
- **`/metrics.json`** — the same aggregates as JSON
- **`agent-firewall stats`** — the same, in the terminal

## Smoke test

```bash
agent-firewall smoke                       # mock upstream, full proxy pipeline, no key
agent-firewall smoke --live --provider anthropic   # real provider via a running proxy
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
- Supports the Anthropic Messages, OpenAI Chat Completions, and Gemini
  generateContent APIs. Gemini streaming is reconstructed into a single chunk.

## Layout

```
agent_firewall/
  proxy.py        FastAPI reverse proxy + enforcement
  engine.py       walks request/response payloads, applies detectors
  detectors/      pii.py · injection.py · actions.py
  judge.py        optional LLM second-stage classifier (escalate-only)
  streaming.py    Anthropic SSE parse / reconstruct / serialize
  openai_shim.py  OpenAI Chat Completions adapter (request/response/SSE)
  gemini_shim.py  Gemini generateContent adapter (request/response/SSE)
  approvals.py    ApprovalManager (console/web/slack) + JSONL audit log
  metrics.py      audit-log aggregation + HTML dashboard
  smoke.py        end-to-end smoke test (mock + live)
  config.py       policy loading (YAML, deep-merged over defaults)
  cli.py          serve · check · stats · smoke · version
```
