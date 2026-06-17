"""Aggregate the audit JSONL into stats + a tiny HTML dashboard.

The audit log (one JSON object per decision; see proxy._audit) is the source
of truth. ``aggregate`` reduces it to counts; ``render_dashboard`` produces a
dependency-free HTML page; both back the /metrics.json and /dashboard routes
and the ``stats`` CLI command.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

_ENFORCED = {"block", "require_approval"}


def aggregate(audit_path: str | None) -> dict[str, Any]:
    records = _read(audit_path)
    by_action: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    by_detector: Counter[str] = Counter()
    top_findings: Counter[str] = Counter()
    tools: Counter[str] = Counter()

    for rec in records:
        by_action[rec.get("action", "?")] += 1
        by_severity[rec.get("severity", "?")] += 1
        by_phase[rec.get("phase", "?")] += 1
        for f in rec.get("findings", []) or []:
            by_detector[f.get("detector", "?")] += 1
            top_findings[f.get("title", "?")] += 1
        for t in rec.get("tools", []) or []:
            tools[t] += 1

    total = len(records)
    enforced = sum(v for k, v in by_action.items() if k in _ENFORCED)
    blocked = by_action.get("block", 0)
    return {
        "total_events": total,
        "blocked": blocked,
        "enforced": enforced,
        "block_rate": round(blocked / total, 4) if total else 0.0,
        "enforce_rate": round(enforced / total, 4) if total else 0.0,
        "by_action": dict(by_action),
        "by_severity": dict(by_severity),
        "by_phase": dict(by_phase),
        "by_detector": dict(by_detector),
        "top_findings": top_findings.most_common(10),
        "top_tools": tools.most_common(10),
    }


def _read(audit_path: str | None) -> list[dict[str, Any]]:
    if not audit_path:
        return []
    path = Path(audit_path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def render_dashboard(stats: dict[str, Any]) -> str:
    def bars(d: dict[str, int]) -> str:
        if not d:
            return "<p class=dim>no data</p>"
        mx = max(d.values()) or 1
        rows = ""
        for k, v in sorted(d.items(), key=lambda x: -x[1]):
            pct = int(100 * v / mx)
            rows += (f"<div class=row><span class=label>{_esc(k)}</span>"
                     f"<span class=bar><i style='width:{pct}%'></i></span>"
                     f"<span class=num>{v}</span></div>")
        return rows

    findings = "".join(f"<li><span class=num>{v}</span> {_esc(k)}</li>" for k, v in stats["top_findings"]) \
        or "<li class=dim>none</li>"

    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>agent-firewall · dashboard</title>
<meta http-equiv=refresh content=10>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}}
 h1{{font-size:1.3rem}} h2{{font-size:1rem;margin-top:1.6rem;color:#444}}
 .cards{{display:flex;gap:.8rem;flex-wrap:wrap}}
 .card{{flex:1;min-width:120px;background:#f5f5f7;border-radius:12px;padding:1rem;text-align:center}}
 .card b{{display:block;font-size:1.8rem}} .card.bad b{{color:#c0362c}}
 .row{{display:flex;align-items:center;gap:.6rem;margin:.2rem 0}}
 .label{{width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
 .bar{{flex:1;background:#eee;border-radius:6px;height:14px;overflow:hidden}}
 .bar i{{display:block;height:100%;background:#3a7afe}} .num{{width:42px;text-align:right;color:#666}}
 ul{{list-style:none;padding:0}} li{{margin:.2rem 0}} .dim{{color:#999}}
</style></head><body>
<h1>🛡 agent-firewall dashboard</h1>
<div class=cards>
 <div class=card><b>{stats['total_events']}</b>events</div>
 <div class='card bad'><b>{stats['blocked']}</b>blocked</div>
 <div class=card><b>{stats['enforced']}</b>enforced</div>
 <div class=card><b>{int(stats['block_rate']*100)}%</b>block rate</div>
</div>
<h2>By action</h2>{bars(stats['by_action'])}
<h2>By detector</h2>{bars(stats['by_detector'])}
<h2>By phase</h2>{bars(stats['by_phase'])}
<h2>Top findings</h2><ul>{findings}</ul>
<p class=dim>auto-refreshes every 10s · source: audit log</p>
</body></html>"""


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
