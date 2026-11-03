from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from simple_rapidchiplet.evaluator import EvaluationResult
from simple_rapidchiplet.report import _graph_svg


GOALS = ("balanced", "latency", "power", "area")
METHODS = (("brute_force", "brute-force"), ("pareto_dp", "pareto-dp"))
GOAL_LABELS = {
    "balanced": "Balanced",
    "latency": "Low Latency",
    "power": "Low Power",
    "area": "Low Area",
}
GOAL_NOTES = {
    "balanced": "Latency, area, and power are weighted evenly.",
    "latency": "Latency has the highest PPA weight.",
    "power": "Power has the highest PPA weight.",
    "area": "Area has the highest PPA weight.",
}


@dataclass(frozen=True)
class ReportRow:
    method_dir: str
    method_label: str
    row: EvaluationResult


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Combine per-goal RapidChiplet reports.")
    parser.add_argument(
        "--results-root",
        action="append",
        default=None,
        help="Results root to include. Can be passed more than once. Default: results/latest.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for the combined report. Default: first results root.",
    )
    parser.add_argument("--title", default="RapidChiplet Goal Comparison Report")
    parser.add_argument(
        "--description",
        default="Best partition plus topology view for each user-selected goal.",
    )
    args = parser.parse_args()

    results_roots = [Path(item) for item in args.results_root] if args.results_root else [root / "results" / "latest"]
    output_root = Path(args.output_root) if args.output_root is not None else results_roots[0]
    output_root.mkdir(parents=True, exist_ok=True)

    best_rows = [row for results_root in results_roots for row in _load_rows(results_root, "best.json")]
    summary_rows = [row for results_root in results_roots for row in _load_rows(results_root, "summary.json")]
    html_text = _render_html(best_rows, summary_rows, title=args.title, description=args.description)
    output_path = output_root / "combined_report.html"
    output_path.write_text(html_text, encoding="utf-8")
    (output_root / "report.html").write_text(html_text, encoding="utf-8")
    print(f"Wrote {output_path}")


def _load_rows(results_root: Path, filename: str) -> list[ReportRow]:
    rows: list[ReportRow] = []
    for method_dir, method_label in METHODS:
        for goal in GOALS:
            path = results_root / method_dir / goal / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for raw in json.load(f):
                    if "canonical_flows" not in raw:
                        raise ValueError("Pre-Phase-1.5 results use obsolete traffic/FPS semantics; regenerate these results.")
                    rows.append(
                        ReportRow(
                            method_dir=method_dir,
                            method_label=method_label,
                            row=EvaluationResult(**raw),
                        )
                    )
    return rows


def _render_html(
    best_rows: list[ReportRow],
    summary_rows: list[ReportRow],
    title: str,
    description: str,
) -> str:
    side_by_side = _side_by_side_table(best_rows)
    goal_sections = _goal_sections(best_rows, summary_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5f6e7d;
      --line: #d8dee6;
      --header: #eef2f6;
      --bg: #f7f8fa;
      --ok: #117a65;
      --bad: #a93226;
      --accent: #2f6fad;
      --traffic: #c6422f;
      --link: #8a97a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: white;
    }}
    header {{
      padding: 20px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--bg);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    header p, .muted {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    main {{ padding: 22px 28px 36px; }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .goal-section {{
      margin: 26px 0 34px;
      padding-top: 4px;
    }}
    .goal-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 2px solid var(--line);
    }}
    .goal-header p {{
      margin: 0;
      color: var(--muted);
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      color: #26323f;
      background: var(--header);
      font-weight: 700;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .status {{ font-weight: 700; }}
    .met {{ color: var(--ok); }}
    .miss {{ color: var(--bad); }}
    .score {{ font-variant-numeric: tabular-nums; }}
    .design-cell {{
      min-width: 280px;
      white-space: normal;
      line-height: 1.4;
    }}
    .plan {{
      max-width: 560px;
      white-space: normal;
      line-height: 1.35;
    }}
    .chip {{
      display: inline-block;
      margin: 0 5px 5px 0;
      padding: 3px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fbfcfd;
      color: var(--muted);
      font-size: 12px;
    }}
    .same {{ color: var(--ok); font-weight: 700; }}
    .different {{ color: var(--bad); font-weight: 700; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 18px;
      align-items: start;
      margin-top: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: white;
    }}
    .card h3 {{
      margin: 0 0 6px;
      font-size: 17px;
      letter-spacing: 0;
    }}
    .card h4 {{
      margin: 0 0 6px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .card p {{
      margin: 0 0 10px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .partition-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(112px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .partition-chiplet {{
      min-height: 92px;
      border: 1px solid var(--line);
      border-top: 4px solid var(--stage-color);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfd;
    }}
    .partition-chiplet strong {{
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .partition-chiplet .stage {{
      min-height: 30px;
      color: var(--ink);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .partition-chiplet .metric {{
      color: var(--muted);
      font-size: 11px;
      margin-top: 4px;
    }}
    .load-bar {{
      height: 5px;
      margin-top: 7px;
      background: #e7edf3;
      border-radius: 999px;
      overflow: hidden;
    }}
    .load-bar span {{
      display: block;
      height: 5px;
      background: var(--accent);
    }}
    .topology-svg {{
      width: 100%;
      min-height: 260px;
      border: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .topology-link {{
      stroke: var(--link);
      stroke-width: 2;
      fill: none;
    }}
    .traffic-link {{
      stroke: var(--traffic);
      stroke-dasharray: 8 6;
      fill: none;
    }}
    .chiplet-node {{
      fill: #e8f2ff;
      stroke: #2f6fad;
      stroke-width: 2;
    }}
    .chiplet-label {{
      text-anchor: middle;
      font-size: 12px;
      font-weight: 700;
      dominant-baseline: central;
    }}
    .stage-label {{
      text-anchor: middle;
      fill: var(--muted);
      font-size: 10px;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      align-items: center;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 22px;
      height: 0;
      border-top: 2px solid var(--link);
      margin-right: 7px;
      vertical-align: middle;
    }}
    .legend .traffic::before {{
      border-top: 2px dashed var(--traffic);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(description)}</p>
  </header>
  <main>
    <h2>Search Method Check</h2>
    {side_by_side}
    {goal_sections}
  </main>
</body>
</html>
"""


def _side_by_side_table(rows: list[ReportRow]) -> str:
    grouped: dict[tuple[str, str], dict[str, EvaluationResult]] = defaultdict(dict)
    for item in rows:
        grouped[(item.row.ppa_goal, item.row.model)][item.method_label] = item.row

    body = []
    for goal in GOALS:
        for model in sorted(model for row_goal, model in grouped if row_goal == goal):
            methods = grouped[(goal, model)]
            brute = methods.get("brute-force")
            dp = methods.get("pareto-dp")
            same = _same_design(brute, dp) if brute and dp else False
            delta = abs(brute.ppa_score - dp.ppa_score) if brute and dp else math.nan
            body.append(
                f"""<tr>
  <td>{html.escape(goal)}</td>
  <td>{html.escape(model)}</td>
  <td class="design-cell">{_compact_design(brute)}</td>
  <td class="design-cell">{_compact_design(dp)}</td>
  <td class="{'same' if same else 'different'}">{'same' if same else 'different'}</td>
  <td class="score">{_fmt(delta, 6)}</td>
</tr>"""
            )
    return f"""<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Goal</th>
        <th>Model</th>
        <th>Brute-force</th>
        <th>Pareto-DP</th>
        <th>Same Design</th>
        <th>Score Delta</th>
      </tr>
    </thead>
    <tbody>
      {"".join(body)}
    </tbody>
  </table>
</div>"""


def _goal_sections(best_rows: list[ReportRow], summary_rows: list[ReportRow]) -> str:
    sections = []
    for goal in GOALS:
        label = GOAL_LABELS[goal]
        note = GOAL_NOTES[goal]
        sections.append(
            f"""<section class="goal-section" id="goal-{html.escape(goal)}">
  <div class="goal-header">
    <h2>User Choice: {html.escape(label)}</h2>
    <p>{html.escape(note)}</p>
  </div>
  <h3>Summary Table</h3>
  {_goal_summary_table(summary_rows, goal)}
  <h3>Partition And Connection Views</h3>
  <div class="cards">
    {_goal_design_cards(best_rows, goal)}
  </div>
</section>"""
        )
    return "\n".join(sections)


def _goal_summary_table(rows: list[ReportRow], goal: str) -> str:
    canonical_rows = [
        item.row
        for item in rows
        if item.row.ppa_goal == goal and item.method_label == "brute-force"
    ]
    body = []
    for row in sorted(canonical_rows, key=lambda item: (item.model, item.topology)):
        status_class = "valid" if row.architecture_feasible else "invalid"
        status_text = "valid" if row.architecture_feasible else "invalid"
        chiplets = f"used {row.selected_chiplets} + unused {row.unused_chiplets} = {row.available_chiplets}"
        body.append(
            f"""<tr>
  <td>{html.escape(row.model)}</td>
  <td>{html.escape(row.topology)}</td>
  <td class="status {status_class}">{status_text}</td>
  <td>{html.escape(chiplets)}</td>
  <td>{_fmt(row.achieved_fps, 1)}</td>
  <td>{_fmt(row.total_area_mm2, 2)}</td>
  <td>{_fmt(row.total_power_w, 2)}</td>
  <td class="score">{_fmt(row.ppa_score, 3)} ({html.escape(row.ppa_goal)})</td>
  <td>{_fmt(row.avg_latency_ns, 2)}</td>
  <td>{html.escape(row.bottleneck_link)}</td>
  <td class="plan">{html.escape(row.workload_plan)}</td>
</tr>"""
        )
    return f"""<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Topology</th>
        <th>Status</th>
        <th>Chiplets</th>
        <th>FPS</th>
        <th>Area mm^2</th>
        <th>Power W</th>
        <th>PPA Score</th>
        <th>Latency ns</th>
        <th>Bottleneck Link</th>
        <th>Workload Plan</th>
      </tr>
    </thead>
    <tbody>
      {"".join(body)}
    </tbody>
  </table>
</div>"""


def _goal_design_cards(rows: list[ReportRow], goal: str) -> str:
    canonical_rows = [
        item
        for item in rows
        if item.row.ppa_goal == goal and item.method_label == "brute-force"
    ]
    cards = []
    for index, item in enumerate(sorted(canonical_rows, key=lambda x: x.row.model)):
        dp_row = _matching_row(rows, goal, item.row.model, "pareto-dp")
        cards.append(_design_card(item.row, goal, index, dp_row))
    return "\n".join(cards)


def _matching_row(rows: list[ReportRow], goal: str, model: str, method_label: str) -> EvaluationResult | None:
    for item in rows:
        if item.row.ppa_goal == goal and item.row.model == model and item.method_label == method_label:
            return item.row
    return None


def _design_card(row: EvaluationResult, goal: str, index: int, dp_row: EvaluationResult | None) -> str:
    label = GOAL_LABELS[goal]
    status = "VALID" if row.architecture_feasible else "INVALID"
    status_class = "valid" if row.architecture_feasible else "invalid"
    dp_note = "Pareto-DP selects the same design." if _same_design(row, dp_row) else "Pareto-DP selects a different design."
    return f"""<section class="card">
  <h3>User Choice: {html.escape(label)}</h3>
  <h4>{html.escape(row.model)} / {html.escape(row.topology)} / <span class="status {status_class}">{status}</span></h4>
  <p>Selected by brute-force best row. {html.escape(dp_note)} Used {row.selected_chiplets}/{row.available_chiplets}, unused {row.unused_chiplets}, PPA score {_fmt(row.ppa_score, 3)}.</p>
  <p>Inverse-E2E diagnostic {_fmt(row.achieved_fps, 4)} FPS, area {_fmt(row.total_area_mm2, 2)} mm^2, power {_fmt(row.total_power_w, 2)} W, latency {_fmt(row.avg_latency_ns, 2)} ns.</p>
  {_partition_grid(row)}
  {_graph_svg(row, _goal_graph_index(goal, index))}
  <div class="legend"><span>topology link</span><span class="traffic">workload traffic</span></div>
</section>"""


def _goal_graph_index(goal: str, index: int) -> int:
    return GOALS.index(goal) * 10 + index


def _detail_table(rows: list[ReportRow], table_id: str) -> str:
    body = []
    for item in sorted(rows, key=lambda x: (x.row.ppa_goal, x.row.model, x.method_label, x.row.topology)):
        row = item.row
        status_class = "valid" if row.architecture_feasible else "invalid"
        status_text = "VALID" if row.architecture_feasible else "INVALID"
        body.append(
            f"""<tr>
  <td>{html.escape(row.ppa_goal)}</td>
  <td>{html.escape(item.method_label)}</td>
  <td>{html.escape(row.model)}</td>
  <td>{html.escape(row.topology)}</td>
  <td class="status {status_class}">{status_text}</td>
  <td>{row.selected_chiplets}/{row.available_chiplets}</td>
  <td>{row.unused_chiplets}</td>
  <td>{_fmt(row.achieved_fps, 4)}</td>
  <td>{_fmt(row.total_area_mm2, 4)}</td>
  <td>{_fmt(row.total_power_w, 4)}</td>
  <td>{_fmt(row.avg_latency_ns, 4)}</td>
  <td class="score">{_fmt(row.ppa_score, 6)}</td>
  <td class="plan">{html.escape(row.workload_plan)}</td>
</tr>"""
        )
    return f"""<div class="table-wrap" id="{html.escape(table_id)}">
  <table>
    <thead>
      <tr>
        <th>Goal</th>
        <th>Search</th>
        <th>Model</th>
        <th>Topology</th>
        <th>Status</th>
        <th>Used/Total</th>
        <th>Unused</th>
        <th>FPS</th>
        <th>Area mm^2</th>
        <th>Power W</th>
        <th>Latency ns</th>
        <th>PPA Score</th>
        <th>Workload Plan</th>
      </tr>
    </thead>
    <tbody>
      {"".join(body)}
    </tbody>
  </table>
</div>"""


def _unique_design_cards(rows: list[ReportRow]) -> str:
    grouped: dict[tuple[object, ...], list[ReportRow]] = defaultdict(list)
    for item in rows:
        row = item.row
        signature = (
            row.model,
            row.topology,
            row.selected_chiplets,
            round(row.achieved_fps, 6),
            round(row.total_area_mm2, 6),
            round(row.total_power_w, 6),
            round(row.avg_latency_ns, 6),
            row.workload_plan,
        )
        grouped[signature].append(item)

    cards = []
    for index, items in enumerate(grouped.values()):
        row = items[0].row
        used_by = " ".join(
            f'<span class="chip">{html.escape(item.row.ppa_goal)} / {html.escape(item.method_label)}</span>'
            for item in sorted(items, key=lambda x: (x.row.ppa_goal, x.method_label))
        )
        cards.append(
            f"""<section class="card">
  <h3>{html.escape(row.model)} / {html.escape(row.topology)} / {row.selected_chiplets} chiplets</h3>
  <p>{used_by}</p>
  <p>FPS {_fmt(row.achieved_fps, 4)}, area {_fmt(row.total_area_mm2, 2)} mm^2, power {_fmt(row.total_power_w, 2)} W, latency {_fmt(row.avg_latency_ns, 2)} ns, unused {row.unused_chiplets}.</p>
  {_partition_grid(row)}
  {_graph_svg(row, index)}
  <div class="legend"><span>topology link</span><span class="traffic">workload traffic</span></div>
</section>"""
        )
    return "\n".join(cards)


def _partition_grid(row: EvaluationResult) -> str:
    graph = row.connection_graph
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    traffic_edges = graph.get("traffic_edges", []) if isinstance(graph, dict) else []
    traffic_by_source = {int(edge["source"]): float(edge["mb"]) for edge in traffic_edges}
    max_ops = max((float(node.get("ops", 0.0)) for node in nodes), default=1.0)
    tiles = []
    for node in sorted(nodes, key=lambda item: int(item["id"])):
        node_id = int(node["id"])
        stage = str(node.get("stage", ""))
        ops = float(node.get("ops", 0.0))
        ops_g = ops / 1e9
        traffic_mb = traffic_by_source.get(node_id)
        traffic = "out -" if traffic_mb is None else f"out {traffic_mb:.2f} MB"
        width = _pct(ops, max_ops)
        color = _partition_color(_stage_key(stage))
        tiles.append(
            f"""<div class="partition-chiplet" style="--stage-color:{color}">
  <strong>C{node_id + 1}</strong>
  <div class="stage">{html.escape(stage)}</div>
  <div class="metric">{ops_g:.2f} GOP, {html.escape(traffic)}</div>
  <div class="load-bar"><span style="width:{width}%"></span></div>
</div>"""
        )
    return f'<div class="partition-strip">{"".join(tiles)}</div>'


def _compact_design(row: EvaluationResult | None) -> str:
    if row is None:
        return "-"
    status = "VALID" if row.architecture_feasible else "INVALID"
    return (
        f"<strong>{html.escape(row.topology)}</strong>, "
        f"<span class=\"status {'valid' if row.architecture_feasible else 'invalid'}\">{status}</span>, "
        f"used {row.selected_chiplets}/{row.available_chiplets}, unused {row.unused_chiplets}<br>"
        f"inverse-E2E diagnostic {_fmt(row.achieved_fps, 4)} FPS<br>"
        f"area {_fmt(row.total_area_mm2, 2)}, "
        f"power {_fmt(row.total_power_w, 2)}, latency {_fmt(row.avg_latency_ns, 2)}<br>"
        f"score <span class=\"score\">{_fmt(row.ppa_score, 6)}</span>"
    )


def _same_design(left: EvaluationResult | None, right: EvaluationResult | None) -> bool:
    if left is None or right is None:
        return False
    same_metrics = (
        left.model == right.model
        and left.topology == right.topology
        and left.selected_chiplets == right.selected_chiplets
        and left.unused_chiplets == right.unused_chiplets
        and _close(left.achieved_fps, right.achieved_fps)
        and _close(left.total_area_mm2, right.total_area_mm2)
        and _close(left.total_power_w, right.total_power_w)
        and _close(left.avg_latency_ns, right.avg_latency_ns)
    )
    if not same_metrics:
        return False
    if left.selected_chiplets == 1:
        return True
    return (
        left.workload_plan == right.workload_plan
        and left.connection_graph == right.connection_graph
    )


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6


def _fmt(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _stage_key(value: str) -> str:
    return value.split("[", 1)[0]


def _partition_color(value: str) -> str:
    palette = ["#2f6fad", "#117a65", "#b86b1b", "#7d4aa8", "#a94442", "#4d7f87"]
    return palette[sum(ord(char) for char in value) % len(palette)]


def _pct(value: float, maximum: float) -> int:
    if not math.isfinite(value) or maximum <= 0:
        return 0
    return max(1, min(100, int(round(value / maximum * 100))))


if __name__ == "__main__":
    main()
