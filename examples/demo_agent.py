"""Tiny demo: an Anthropic SDK client pointed at agent-firewall.

Run the proxy first:
    uv run agent-firewall serve

Then, in another shell:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python examples/demo_agent.py

The only difference from a normal agent is `base_url`. Try crafting a tool
that matches a dangerous-action rule (e.g. name it "delete_file") and watch
the proxy ask for approval on its console.
"""

import os

try:
    from anthropic import Anthropic
except ImportError:  # anthropic is not a hard dependency of the proxy itself
    raise SystemExit("pip install anthropic to run this demo")

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787"))

resp = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=512,
    tools=[
        {
            "name": "delete_file",
            "description": "Delete a file from disk.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ],
    messages=[
        {"role": "user", "content": "Delete the file /tmp/old_report.csv to free up space."}
    ],
)

for block in resp.content:
    if block.type == "text":
        print("TEXT:", block.text)
    elif block.type == "tool_use":
        print("TOOL_USE:", block.name, block.input)
