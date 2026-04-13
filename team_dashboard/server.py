"""Real-time dashboard for the nmr-v2-revision agent team.

Serves a single HTML page on http://localhost:8765 that auto-refreshes every
2 seconds and shows:
- Team members + their current status
- Team task list
- Recent inter-agent messages (team room log)
- Running experiments (tailed from ~/nmr-ssl/experiments/*.log)
- Results summary (aggregated from results_2d/*.json)

Run:
    python3 team_dashboard/server.py

Then open http://localhost:8765 in a browser.
"""

from __future__ import annotations

import glob
import html
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM_NAME = "nmr-v2-revision"
TEAM_DIR = Path.home() / ".claude" / "teams" / TEAM_NAME
TASK_DIR = Path.home() / ".claude" / "tasks" / TEAM_NAME
RESULTS_DIR = ROOT / "experiments" / "results_2d"
LOGS = {
    "Reviewer experiments": ROOT / "experiments" / "reviewer_exp.log",
    "Batch 2 (stereo / err-decomp / label-sweep)": ROOT / "experiments" / "batch2.log",
    "Chemistry demo": ROOT / "experiments" / "chem_demo.log",
}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nmr-v2-revision · team dashboard</title>
<style>
    :root {
        --bg: #0e1116;
        --panel: #161b22;
        --panel2: #1f2630;
        --border: #2a323c;
        --fg: #c9d1d9;
        --muted: #7d8590;
        --accent: #2f81f7;
        --ok: #3fb950;
        --warn: #d29922;
        --err: #f85149;
        --pink: #db61a2;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body {
        background: var(--bg); color: var(--fg);
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", Helvetica, sans-serif;
        font-size: 13px; line-height: 1.5;
    }
    header {
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 14px 24px; border-bottom: 1px solid var(--border);
        background: linear-gradient(90deg, #161b22 0%, #1f2630 100%);
    }
    header h1 {
        margin: 0; font-size: 16px; font-weight: 600;
        letter-spacing: 0.3px;
    }
    header .badge {
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        background: var(--panel2); color: var(--accent); margin-left: 10px;
        font-size: 11px; font-weight: 500;
    }
    header .live {
        display: flex; align-items: center; gap: 6px;
        font-size: 11px; color: var(--muted);
    }
    header .live::before {
        content: ""; width: 8px; height: 8px; border-radius: 50%;
        background: var(--ok); animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.35; }
    }
    .grid {
        display: grid; gap: 14px; padding: 14px 24px;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: auto auto auto;
    }
    .card {
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 8px; padding: 14px 16px; overflow: hidden;
    }
    .card.wide { grid-column: 1 / span 2; }
    .card h2 {
        margin: 0 0 10px 0; font-size: 12px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted);
        display: flex; justify-content: space-between; align-items: center;
    }
    .card h2 .count {
        background: var(--panel2); padding: 1px 7px; border-radius: 10px;
        font-size: 10px; font-weight: 500; color: var(--fg);
    }
    .member {
        display: grid; grid-template-columns: 24px 1fr auto;
        gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--border);
        align-items: center;
    }
    .member:last-child { border-bottom: 0; }
    .member .avatar {
        width: 24px; height: 24px; border-radius: 50%;
        background: var(--accent); color: white;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 600;
    }
    .member .name { font-weight: 500; }
    .member .role { color: var(--muted); font-size: 11px; }
    .member .status {
        padding: 2px 8px; border-radius: 10px; font-size: 10px;
        font-weight: 500; text-transform: uppercase; letter-spacing: 0.4px;
    }
    .status.active { background: rgba(63, 185, 80, 0.15); color: var(--ok); }
    .status.idle { background: rgba(125, 133, 144, 0.15); color: var(--muted); }
    .status.error { background: rgba(248, 81, 73, 0.15); color: var(--err); }

    .task {
        display: grid; grid-template-columns: 18px 1fr; gap: 8px;
        padding: 6px 0; font-size: 12px;
    }
    .task .bullet {
        display: inline-block; width: 12px; height: 12px; border-radius: 2px;
        margin-top: 3px;
    }
    .task.pending .bullet { background: var(--panel2); border: 1px solid var(--muted); }
    .task.in_progress .bullet { background: var(--warn); }
    .task.completed .bullet { background: var(--ok); }
    .task.completed .content { color: var(--muted); text-decoration: line-through; }

    .log {
        background: var(--panel2); border-radius: 6px; padding: 10px 12px;
        max-height: 260px; overflow-y: auto;
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, monospace;
        font-size: 11px; line-height: 1.5; color: var(--fg);
    }
    .log-section { margin-bottom: 10px; }
    .log-section h3 {
        margin: 0 0 4px 0; font-size: 11px; color: var(--accent);
        font-weight: 500;
    }
    .log-line { white-space: pre; color: var(--fg); }
    .log-line.event-k { color: var(--accent); }
    .log-line.event-noise { color: var(--warn); }
    .log-line.event-wrong { color: var(--pink); }
    .log-line.event-ok { color: var(--ok); }
    .log-line.event-err { color: var(--err); }
    .log-line.event-wrote { color: var(--ok); font-weight: 600; }

    .result-kv {
        display: grid; grid-template-columns: 1fr auto; gap: 4px 12px;
        font-size: 12px;
    }
    .result-kv .k { color: var(--muted); }
    .result-kv .v { font-family: ui-monospace, monospace; font-weight: 500; color: var(--fg); }
    .result-kv .v.ok { color: var(--ok); }
    .result-kv .v.warn { color: var(--warn); }

    .pill {
        display: inline-block; padding: 2px 6px; background: var(--panel2);
        border-radius: 10px; font-size: 10px; color: var(--muted);
        font-family: ui-monospace, monospace;
    }
    footer {
        padding: 12px 24px; font-size: 11px; color: var(--muted);
        border-top: 1px solid var(--border);
        display: flex; justify-content: space-between;
    }
</style>
</head>
<body>
<header>
    <h1>nmr-v2-revision <span class="badge" id="team-badge">v2 preprint revision</span></h1>
    <div class="live">live · refresh 2s · <span id="clock"></span></div>
</header>

<div class="grid">
    <div class="card" id="members-card">
        <h2>Team members <span class="count" id="members-count">·</span></h2>
        <div id="members"></div>
    </div>

    <div class="card" id="tasks-card">
        <h2>Tasks <span class="count" id="tasks-count">·</span></h2>
        <div id="tasks"></div>
    </div>

    <div class="card wide">
        <h2>Live experiment output <span class="count" id="log-count">·</span></h2>
        <div class="log" id="logs"></div>
    </div>

    <div class="card">
        <h2>Results summary <span class="count" id="results-count">·</span></h2>
        <div id="results"></div>
    </div>

    <div class="card">
        <h2>Preprint status</h2>
        <div id="preprint"></div>
    </div>
</div>

<footer>
    <span>dashboard @ <code>/Users/ericdong/nmr-ssl/team_dashboard/server.py</code></span>
    <span id="uptime"></span>
</footer>

<script>
async function refresh() {
    try {
        const r = await fetch('/api/state?ts=' + Date.now());
        const d = await r.json();

        document.getElementById('clock').textContent = new Date().toLocaleTimeString();
        document.getElementById('uptime').textContent = 'uptime ' + d.uptime_s + 's';

        // Members
        const m = document.getElementById('members');
        m.innerHTML = '';
        document.getElementById('members-count').textContent = d.members.length;
        for (const mem of d.members) {
            const el = document.createElement('div');
            el.className = 'member';
            const initials = (mem.name || '?').slice(0, 2).toUpperCase();
            el.innerHTML = `
                <div class="avatar">${initials}</div>
                <div>
                    <div class="name">${mem.name}</div>
                    <div class="role">${mem.agentType || ''}</div>
                </div>
                <div class="status ${mem.status}">${mem.status}</div>
            `;
            m.appendChild(el);
        }

        // Tasks
        const t = document.getElementById('tasks');
        t.innerHTML = '';
        document.getElementById('tasks-count').textContent =
            d.tasks.filter(x => x.status === 'completed').length + ' / ' + d.tasks.length;
        for (const task of d.tasks) {
            const el = document.createElement('div');
            el.className = 'task ' + task.status;
            el.innerHTML = `
                <div class="bullet"></div>
                <div class="content">${task.content}</div>
            `;
            t.appendChild(el);
        }

        // Logs
        const logs = document.getElementById('logs');
        let html = '';
        let totalLines = 0;
        for (const section of d.log_sections) {
            html += `<div class="log-section"><h3>${section.name}</h3>`;
            for (const line of section.lines) {
                let cls = 'log-line';
                if (line.includes('[k-sweep]') || line.match(/K=\d+/)) cls += ' event-k';
                else if (line.includes('[noise]') || line.includes('sigma')) cls += ' event-noise';
                else if (line.includes('[wrong')) cls += ' event-wrong';
                else if (line.includes('Error') || line.includes('Traceback')) cls += ' event-err';
                else if (line.startsWith('wrote') || line.includes(' wrote ')) cls += ' event-wrote';
                else if (line.includes('MAE =') || line.includes('pass ')) cls += ' event-ok';
                html += `<div class="${cls}">${escapeHtml(line)}</div>`;
                totalLines++;
            }
            html += `</div>`;
        }
        logs.innerHTML = html || '<div class="log-line" style="color: var(--muted)">no live log output</div>';
        logs.scrollTop = logs.scrollHeight;
        document.getElementById('log-count').textContent = totalLines + ' lines';

        // Results
        const res = document.getElementById('results');
        res.innerHTML = '';
        const dl = document.createElement('div');
        dl.className = 'result-kv';
        let shownKeys = 0;
        for (const [k, v] of d.results) {
            const kv = document.createElement('div');
            kv.className = 'k';
            kv.textContent = k;
            const vv = document.createElement('div');
            vv.className = 'v';
            if (v.state === 'ok') vv.classList.add('ok');
            if (v.state === 'warn') vv.classList.add('warn');
            vv.textContent = v.value;
            dl.appendChild(kv);
            dl.appendChild(vv);
            shownKeys++;
        }
        res.appendChild(dl);
        document.getElementById('results-count').textContent = shownKeys;

        // Preprint
        const pp = document.getElementById('preprint');
        pp.innerHTML = `
            <div class="result-kv">
                <div class="k">Source .md</div>
                <div class="v">${d.preprint.md_lines} lines</div>
                <div class="k">Figures</div>
                <div class="v">${d.preprint.figure_count}</div>
                <div class="k">Last compiled .pdf</div>
                <div class="v ${d.preprint.pdf_age_s < 600 ? 'ok' : 'warn'}">${d.preprint.pdf_age_human}</div>
                <div class="k">.pdf pages</div>
                <div class="v">${d.preprint.pdf_pages || '—'}</div>
            </div>
        `;
    } catch (e) {
        document.getElementById('clock').textContent = 'error ' + (e.message || e);
    }
}

function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


START_TIME = time.time()


def tail(path, n=15):
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 8192)
            f.seek(size - chunk)
            data = f.read().decode("utf-8", errors="replace")
        return [ln for ln in data.splitlines() if ln.strip()][-n:]
    except Exception:
        return []


def load_team_members():
    config = TEAM_DIR / "config.json"
    if not config.exists():
        return []
    try:
        data = json.loads(config.read_text())
        members = data.get("members", [])
        return members
    except Exception:
        return []


def load_tasks():
    """Parse task list from the team task directory."""
    if not TASK_DIR.exists():
        return []
    tasks = []
    for p in sorted(TASK_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                tasks.extend(data)
            elif isinstance(data, dict) and "tasks" in data:
                tasks.extend(data["tasks"])
            elif isinstance(data, dict):
                tasks.append(data)
        except Exception:
            continue
    return tasks


def humanize_age(seconds):
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"


def pdf_pages(pdf_path):
    if not pdf_path.exists():
        return None
    try:
        import subprocess
        r = subprocess.run(
            ["pdfinfo", str(pdf_path)], capture_output=True, text=True, timeout=2,
        )
        for line in r.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        return None
    return None


def load_results():
    """Aggregate selected key numbers from results_2d JSON files."""
    out = []

    # Main experiment summary
    summary = RESULTS_DIR / "summary.json"
    if summary.exists():
        try:
            d = json.loads(summary.read_text())
            agg = d.get("aggregate", {})
            for v in ("supervised_1d", "sort_match_ssl_1d", "sort_match_ssl_2d"):
                if v in agg:
                    s = agg[v]
                    out.append((
                        f"{v} 13C",
                        {"value": f"{s['c_mean']:.2f} ± {s['c_std']:.2f}", "state": "ok" if v == "sort_match_ssl_2d" else ""},
                    ))
                    out.append((
                        f"{v} 1H",
                        {"value": f"{s['h_mean']:.2f} ± {s['h_std']:.2f}", "state": "ok" if v == "sort_match_ssl_2d" else ""},
                    ))
        except Exception:
            pass

    # Reviewer experiments
    rev = RESULTS_DIR / "reviewer_experiments.json"
    if rev.exists():
        try:
            d = json.loads(rev.read_text())
            ks = d.get("k_sweep", {})
            if ks:
                best_k = min(ks.keys(), key=lambda k: ks[k]["h_mae"])
                out.append(("K-sweep best H", {"value": f"K={best_k}: {ks[best_k]['h_mae']:.3f}", "state": "ok"}))
            noise = d.get("noise_sweep", {})
            if "high" in noise:
                out.append(("Noise robustness (high)", {"value": f"H={noise['high']['h_mae']:.3f}", "state": "ok"}))
            wrong = d.get("wrong_structure", {})
            if wrong:
                own = wrong.get("own", {})
                wr = wrong.get("wrong", {})
                out.append(("Correct-struct H pass", {"value": f"{own.get('h_rate', 0)*100:.1f}%", "state": "ok"}))
                out.append(("Wrong-struct H pass", {"value": f"{wr.get('h_rate', 0)*100:.1f}%", "state": "ok"}))
            sh = d.get("separate_heads", {})
            if sh:
                out.append(("Separate-encoder C", {"value": f"{sh['c_mae']:.3f}", "state": "warn"}))
        except Exception:
            pass

    # Chemistry demo
    cd = RESULTS_DIR / "chemistry_demo.json"
    if cd.exists():
        try:
            d = json.loads(cd.read_text())
            out.append(("Conformal q_C (ppm)", {"value": f"{d.get('c_quantile_ppm', 0):.2f}", "state": "warn"}))
            out.append(("Conformal q_H (ppm)", {"value": f"{d.get('h_quantile_ppm', 0):.2f}", "state": "ok"}))
            out.append(("Both-nuclei consistency", {"value": f"{d.get('structure_consistent_both', 0)}/{d.get('n_test_molecules', 0)}", "state": "ok"}))
        except Exception:
            pass

    # Error decomposition
    ed = RESULTS_DIR / "error_decomposition.json"
    if ed.exists():
        try:
            d = json.loads(ed.read_text())
            c_by = d.get("c_by_type", {})
            if "aromatic" in c_by:
                out.append(("C aromatic MAE", {"value": f"{c_by['aromatic']['mae']:.2f}", "state": "ok"}))
            if "olefinic" in c_by:
                out.append(("C olefinic MAE (tail)", {"value": f"{c_by['olefinic']['mae']:.2f}", "state": "warn"}))
        except Exception:
            pass

    # Label sweep
    ls = RESULTS_DIR / "label_sweep.json"
    if ls.exists():
        try:
            d = json.loads(ls.read_text())
            results = d.get("results", {})
            for frac, v in sorted(results.items(), key=lambda kv: float(kv[0])):
                var2 = v.get("variants", {}).get("sort_match_ssl_2d", {})
                if var2:
                    out.append((
                        f"2D SSL @ label={frac}",
                        {"value": f"C {var2.get('test_c_mae', 0):.2f}  H {var2.get('test_h_mae', 0):.2f}", "state": ""},
                    ))
        except Exception:
            pass

    return out


def build_state():
    members_raw = load_team_members()
    members = []
    for m in members_raw:
        name = m.get("name", "?")
        status = m.get("status", "idle")
        members.append({
            "name": name,
            "agentType": m.get("agentType", ""),
            "status": status,
            "agentId": m.get("agentId", "")[:12],
        })

    tasks_raw = load_tasks()
    tasks = []
    for t in tasks_raw:
        tasks.append({
            "content": t.get("content", t.get("text", "?")),
            "status": t.get("status", "pending"),
        })

    log_sections = []
    for name, path in LOGS.items():
        lines = tail(path, n=12)
        if lines:
            log_sections.append({"name": name, "lines": lines})

    results = load_results()

    # Preprint status
    md_path = ROOT / "docs" / "2d" / "preprint_2d_draft.md"
    pdf_path = ROOT / "docs" / "2d" / "preprint_2d.pdf"
    md_lines = sum(1 for _ in md_path.open()) if md_path.exists() else 0
    fig_dir = ROOT / "docs" / "2d" / "figures"
    fig_count = len(list(fig_dir.glob("*.png"))) if fig_dir.exists() else 0
    pdf_age = time.time() - pdf_path.stat().st_mtime if pdf_path.exists() else 1e9
    preprint = {
        "md_lines": md_lines,
        "figure_count": fig_count,
        "pdf_age_s": int(pdf_age),
        "pdf_age_human": humanize_age(pdf_age) if pdf_path.exists() else "not compiled",
        "pdf_pages": pdf_pages(pdf_path),
    }

    return {
        "team_name": TEAM_NAME,
        "uptime_s": int(time.time() - START_TIME),
        "members": members,
        "tasks": tasks,
        "log_sections": log_sections,
        "results": results,
        "preprint": preprint,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/state"):
            state = build_state()
            body = json.dumps(state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404); self.end_headers()


def main():
    port = int(os.environ.get("DASHBOARD_PORT", "8765"))
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"nmr-v2-revision dashboard: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
