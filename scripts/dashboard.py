"""
Pipeline Status Dashboard — http://localhost:7777
Auto-refreshes every 10 seconds. Shows which stages are COMPLETE / RUNNING / WAITING.
Run: python scripts/dashboard.py
"""
import http.server
import json
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent  # c:\AMAZON_ML_CHALLENGE

STAGES = [
    {
        "id": "rule_shards",
        "name": "Phase 1a — Blocking Rules",
        "description": "25 blocking rules generating raw candidate pairs (203M total)",
        "output": "data/advanced_hybrid/tmp_train_cands/rule_*.parquet",
        "check": lambda: all(
            (BASE / f"data/advanced_hybrid/tmp_train_cands/rule_{i:02d}.parquet").exists()
            and (BASE / f"data/advanced_hybrid/tmp_train_cands/rule_{i:02d}.parquet").stat().st_size > 0
            for i in range(25)
        ),
    },
    {
        "id": "aggregation",
        "name": "Phase 1b — Aggregate & Rank",
        "description": "Deduplicate pairs, compute evidence scores, top-120 per entity",
        "output": "data/advanced_hybrid/train_candidates.parquet",
        "check": lambda: (BASE / "data/advanced_hybrid/train_candidates.parquet").exists()
                         and (BASE / "data/advanced_hybrid/train_candidates.parquet").stat().st_size > 1_000_000,
    },
    {
        "id": "train_features",
        "name": "Phase 2 — Feature Engineering (Train)",
        "description": "80+ agreement/contradiction/margin features for LightGBM",
        "output": "data/advanced_hybrid/train_features.parquet",
        "check": lambda: (BASE / "data/advanced_hybrid/train_features.parquet").exists()
                         and (BASE / "data/advanced_hybrid/train_features.parquet").stat().st_size > 1_000_000,
    },
    {
        "id": "model_train",
        "name": "Phase 3 — LightGBM Training",
        "description": "5-fold cross-validation, 2 seeds, macro F0.5 threshold calibration",
        "output": "models/advanced_hybrid/*.txt",
        "check": lambda: (BASE / "models/advanced_hybrid").exists()
                         and any((BASE / "models/advanced_hybrid").glob("*.txt")),
    },
    {
        "id": "test_cands",
        "name": "Phase 4 — Test Candidate Blocking",
        "description": "Run all 25 blocking rules on the test set",
        "output": "data/advanced_hybrid/test_candidates.parquet",
        "check": lambda: (BASE / "data/advanced_hybrid/test_candidates.parquet").exists()
                         and (BASE / "data/advanced_hybrid/test_candidates.parquet").stat().st_size > 1_000_000,
    },
    {
        "id": "test_features",
        "name": "Phase 5 — Test Feature Engineering",
        "description": "80+ features for test candidate pairs",
        "output": "data/advanced_hybrid/test_features.parquet",
        "check": lambda: (BASE / "data/advanced_hybrid/test_features.parquet").exists()
                         and (BASE / "data/advanced_hybrid/test_features.parquet").stat().st_size > 1_000_000,
    },
    {
        "id": "scoring",
        "name": "Phase 6 — Ensemble Scoring",
        "description": "Score test pairs with LightGBM ensemble",
        "output": "data/advanced_hybrid/test_predictions.parquet",
        "check": lambda: (BASE / "data/advanced_hybrid/test_predictions.parquet").exists()
                         and (BASE / "data/advanced_hybrid/test_predictions.parquet").stat().st_size > 100_000,
    },
    {
        "id": "submission",
        "name": "Phase 7 — Final Submission",
        "description": "Format, deduplicate, validate -> matching_results.tsv",
        "output": "output/advanced_hybrid/matching_results.tsv",
        "check": lambda: (BASE / "output/advanced_hybrid/matching_results.tsv").exists()
                         and (BASE / "output/advanced_hybrid/matching_results.tsv").stat().st_size > 100_000,
    },
]


def get_status():
    results = []
    found_running = False
    for stage in STAGES:
        try:
            done = stage["check"]()
        except Exception:
            done = False
        if done:
            state = "done"
        elif not found_running:
            state = "running"
            found_running = True
        else:
            state = "pending"
        results.append(dict(stage, state=state, done=done))
    return results


def get_agg_progress():
    part_dir = BASE / "data/advanced_hybrid/tmp_train_cands/partitions"
    if not part_dir.exists():
        return 0, 32
    done = len(list(part_dir.glob("part_*.parquet")))
    return done, 32


def get_report():
    path = BASE / "output/advanced_hybrid/evaluation_report.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>Pipeline Dashboard</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:32px 28px;min-height:100vh}}
h1{{font-size:22px;font-weight:700;color:#f0f6fc;margin-bottom:4px}}
.sub{{font-size:13px;color:#7d8590;margin-bottom:28px}}
.stage{{display:flex;align-items:flex-start;gap:14px;padding:14px 18px;margin-bottom:10px;border-radius:10px;border:1px solid transparent}}
.done{{background:#0d2818;border-color:#1a4731}}
.running{{background:#111d2e;border-color:#388bfd;animation:pulse 2s infinite}}
.pending{{background:#161b22;border-color:#21262d}}
@keyframes pulse{{0%,100%{{border-color:#1f4068}}50%{{border-color:#388bfd}}}}
.icon{{font-size:22px;width:28px;text-align:center;flex-shrink:0;margin-top:1px}}
.body{{flex:1;min-width:0}}
.name{{font-size:14px;font-weight:600;margin-bottom:3px}}
.done .name{{color:#3fb950}}.running .name{{color:#58a6ff}}.pending .name{{color:#7d8590}}
.desc{{font-size:12px;color:#7d8590;margin-bottom:3px}}
.out{{font-size:11px;color:#484f58;font-family:Consolas,monospace}}
.badge{{margin-left:auto;font-size:11px;font-weight:700;padding:3px 12px;border-radius:20px;flex-shrink:0;align-self:center}}
.done .badge{{background:#1a4731;color:#3fb950}}
.running .badge{{background:#1f4068;color:#58a6ff}}
.pending .badge{{background:#21262d;color:#7d8590}}
.bar-wrap{{background:#21262d;border-radius:4px;height:5px;margin-top:8px;overflow:hidden}}
.bar{{background:#388bfd;height:100%;border-radius:4px;transition:width .5s}}
.bar-label{{font-size:11px;color:#388bfd;margin-top:4px}}
.result-box{{background:#0d2818;border:1px solid #1a4731;border-radius:10px;padding:20px 24px;margin-top:28px}}
.result-box h2{{color:#3fb950;font-size:16px;margin-bottom:14px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}}
.metric{{background:#0f2418;border:1px solid #1a4731;border-radius:8px;padding:12px 14px}}
.mlabel{{font-size:11px;color:#7d8590;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}}
.mval{{font-size:20px;font-weight:700;color:#3fb950;font-family:Consolas,monospace}}
.footer{{text-align:center;font-size:11px;color:#484f58;margin-top:20px}}
</style>
</head>
<body>
<h1>🚀 Advanced Hybrid Entity Resolution</h1>
<p class="sub">Amazon ML Challenge &nbsp;·&nbsp; Auto-refreshes every 10s &nbsp;·&nbsp; {ts}</p>
{stages}
{results}
<p class="footer">⟳ Refreshing every 10 seconds automatically</p>
</body>
</html>"""


def render_page(status_list, report):
    done_p, total_p = get_agg_progress()
    stages_html = ""
    for s in status_list:
        st = s["state"]
        icon = {"done": "✅", "running": "⚡", "pending": "⏳"}[st]
        badge = {"done": "COMPLETE", "running": "RUNNING", "pending": "WAITING"}[st]
        extra = ""
        if st == "running" and s["id"] == "aggregation":
            pct = int(done_p / total_p * 100)
            extra = (f'<div class="bar-wrap"><div class="bar" style="width:{pct}%"></div></div>'
                     f'<div class="bar-label">{done_p}/{total_p} partitions &nbsp;({pct}%)</div>')
        stages_html += (
            f'<div class="stage {st}">'
            f'<div class="icon">{icon}</div>'
            f'<div class="body">'
            f'<div class="name">{s["name"]}</div>'
            f'<div class="desc">{s["description"]}</div>'
            f'<div class="out">{s["output"]}</div>'
            f'{extra}'
            f'</div>'
            f'<div class="badge">{badge}</div>'
            f'</div>'
        )

    results_html = ""
    if report:
        def fmt(v):
            return f"{v:.5f}" if isinstance(v, float) else str(v)
        m = [
            ("Macro F0.5", fmt(report.get("measured_validation_macro_f0_5", "—"))),
            ("Macro F1",   fmt(report.get("measured_validation_macro_f1", "—"))),
            ("Precision",  fmt(report.get("measured_pair_precision", "—"))),
            ("Recall",     fmt(report.get("measured_pair_recall", "—"))),
            ("Elapsed",    f"{report.get('elapsed_minutes','—')}m"),
        ]
        cards = "".join(
            f'<div class="metric"><div class="mlabel">{lbl}</div><div class="mval">{val}</div></div>'
            for lbl, val in m
        )
        results_html = (
            f'<div class="result-box"><h2>📊 Measured Validation Results</h2>'
            f'<div class="metrics">{cards}</div></div>'
        )

    ts = datetime.now().strftime("%H:%M:%S")
    return HTML.format(stages=stages_html, results=results_html, ts=ts)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        status_list = get_status()
        report = get_report()
        html = render_page(status_list, report)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = 7777
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Dashboard -> {url}  (Ctrl+C to stop)", flush=True)
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
