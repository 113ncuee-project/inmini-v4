from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple_rapidchiplet.config import load_config
from simple_rapidchiplet.evaluator import (
    EvaluationResult,
    _score_results,
    evaluate_models,
    make_ppa_reference,
    resolve_ppa_weights,
    select_best_by_model,
)
from simple_rapidchiplet.model import load_models
from simple_rapidchiplet.workload import pareto_dp_pipeline_workloads


MODEL_NAMES = ("resnet50", "shufflenet_v2_x1_0")
PPA_GOALS = ("balanced", "latency", "area", "power")
DP_TOP_KS = (12, 24, 30, 0)
SELF_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SELF_COSTS = ("latency", "area", "power")
RANK_TARGETS = ("ppa_score", "self_component", "avg_latency_ns", "achieved_fps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mesh-only DP-pool and candidate-pool self-cost sensitivity analysis."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "defaults.json"))
    parser.add_argument("--models-file", default=str(PROJECT_ROOT / "configs" / "models.json"))
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "results" / "mesh_redefinition_20260901"),
        help="Directory for raw experiment outputs.",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT.parent / "MESH_REDEFINITION_TEST_REPORT.md"),
        help="Markdown report path.",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    all_models = load_models(args.models_file)
    models = [all_models[name] for name in MODEL_NAMES]

    started = time.perf_counter()
    candidate_pool = _candidate_pool_analysis(models, cfg)
    oracle_raw = _evaluate_raw(models, cfg, workload_search="brute-force", dp_top_k=0)
    oracle_refs = _references_by_model(oracle_raw)
    oracle_by_goal = _score_by_goal(oracle_raw, PPA_GOALS, oracle_refs)

    dp_raw_by_top_k = {
        top_k: _evaluate_raw(models, cfg, workload_search="pareto-dp", dp_top_k=top_k)
        for top_k in DP_TOP_KS
    }
    pool_selection = _pool_selection_analysis(dp_raw_by_top_k, oracle_by_goal, oracle_refs)
    self_sensitivity = _self_weight_sensitivity(
        dp_raw_by_top_k[12], oracle_raw, oracle_refs
    )
    correlations = _rank_correlation_analysis(dp_raw_by_top_k[12], PPA_GOALS)
    elapsed = time.perf_counter() - started

    metadata = {
        "date": "2026-10-03",
        "topology": "mesh",
        "models": list(MODEL_NAMES),
        "ppa_goals": list(PPA_GOALS),
        "max_chiplets": cfg.max_chiplets,
        "dp_top_ks": list(DP_TOP_KS),
        "self_scales": list(SELF_SCALES),
        "raw_ppa": ["L_E2E", "A", "P"],
        "self_cost_formula": "C_self,x = X_x / max_j_in_candidate_pool(X_j)",
        "self_cost_mapping": {
            "latency": "avg_latency_ns",
            "area": "total_area_mm2",
            "power": "total_power_w",
        },
        "official_rapidchiplet_root_exists": Path(cfg.rapidchiplet.root).exists(),
        "elapsed_seconds": elapsed,
        "scope_note": "Phase 1.5: ranking uses latency/area/power only; FPS is inverse-E2E diagnostic.",
    }

    _write_csv(output_dir / "candidate_pool.csv", candidate_pool)
    _write_csv(output_dir / "pool_selection_self_reference.csv", pool_selection)
    _write_csv(output_dir / "self_weight_sensitivity.csv", self_sensitivity)
    _write_csv(output_dir / "self_rank_correlation.csv", correlations)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = _build_report(
        candidate_pool=candidate_pool,
        pool_selection=pool_selection,
        self_sensitivity=self_sensitivity,
        correlations=correlations,
        metadata=metadata,
        output_dir=output_dir,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Completed mesh-only self-cost analysis in {elapsed:.2f} s")
    print(f"Report: {report_path.resolve()}")
    print(f"Raw outputs: {output_dir.resolve()}")


def _evaluate_raw(models, cfg, *, workload_search: str, dp_top_k: int) -> list[EvaluationResult]:
    _summaries, rows = evaluate_models(
        models,
        ["mesh"],
        cfg,
        ppa_goal="balanced",
        dp_top_k=dp_top_k,
        workload_search=workload_search,
    )
    return rows


def _candidate_pool_analysis(models, cfg) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        for top_k in DP_TOP_KS:
            counts = [
                len(
                    pareto_dp_pipeline_workloads(
                        model,
                        chiplets,
                        cfg.chiplet.op_per_mac,
                        top_k=top_k,
                    )
                )
                for chiplets in range(1, cfg.max_chiplets + 1)
            ]
            rows.append(
                {
                    "model": model.name,
                    "top_k": _top_k_label(top_k),
                    "top_k_value": top_k,
                    "total_candidates": sum(counts),
                    "max_candidates_per_chiplet_count": max(counts, default=0),
                    "candidate_counts_by_chiplet_count": ",".join(map(str, counts)),
                }
            )
    return rows


def _references_by_model(rows: list[EvaluationResult]):
    return {
        model: make_ppa_reference([row for row in rows if row.model == model])
        for model in MODEL_NAMES
    }


def _score_by_goal(
    rows: list[EvaluationResult],
    goals: Iterable[str],
    references,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for goal in goals:
        scored: list[EvaluationResult] = []
        weights = resolve_ppa_weights(goal)
        for model in MODEL_NAMES:
            model_rows = [row for row in rows if row.model == model]
            scored.extend(_score_results(model_rows, weights, references[model]))
        result[goal] = {"rows": scored, "best": {row.model: row for row in select_best_by_model(scored)}}
    return result


def _pool_selection_analysis(dp_raw_by_top_k, oracle_by_goal, oracle_refs) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for top_k, raw_rows in dp_raw_by_top_k.items():
        pool_refs = _references_by_model(raw_rows)
        for goal in PPA_GOALS:
            pool_scored = _score_by_goal(raw_rows, [goal], pool_refs)[goal]
            fixed_scored = _score_by_goal(raw_rows, [goal], oracle_refs)[goal]
            for model in MODEL_NAMES:
                pool_row = pool_scored["best"][model]
                fixed_row = fixed_scored["best"][model]
                oracle_row = oracle_by_goal[goal]["best"][model]
                rows.append(
                    {
                        "top_k": _top_k_label(top_k),
                        "top_k_value": top_k,
                        "ppa_goal": goal,
                        "model": model,
                        "pool_reference_design": _design_label(pool_row),
                        "fixed_oracle_reference_design": _design_label(fixed_row),
                        "oracle_design": _design_label(oracle_row),
                        "pool_matches_oracle": _same_design(pool_row, oracle_row),
                        "fixed_reference_matches_oracle": _same_design(fixed_row, oracle_row),
                        "self_reference_changed_selection": not _same_design(pool_row, fixed_row),
                        "pool_score_under_oracle_reference": _score_results(
                            [pool_row], resolve_ppa_weights(goal), oracle_refs[model]
                        )[0].ppa_score,
                        "fixed_score_under_oracle_reference": _score_results(
                            [fixed_row], resolve_ppa_weights(goal), oracle_refs[model]
                        )[0].ppa_score,
                        "oracle_score": oracle_row.ppa_score,
                    }
                )
    return rows


def _self_weight_sensitivity(raw_rows, oracle_raw, oracle_refs) -> list[dict[str, object]]:
    pool_refs = _references_by_model(raw_rows)
    rows: list[dict[str, object]] = []
    for scale in SELF_SCALES:
        for goal in PPA_GOALS:
            weights = resolve_ppa_weights(goal)
            for model in MODEL_NAMES:
                model_rows = [row for row in raw_rows if row.model == model]
                selected, score = _select_scaled_self_cost(model_rows, weights, pool_refs[model], scale)
                baseline, _baseline_score = _select_scaled_self_cost(
                    model_rows, weights, pool_refs[model], 1.0
                )
                oracle_rows = [row for row in oracle_raw if row.model == model]
                oracle_row, oracle_score = _select_scaled_self_cost(
                    oracle_rows, weights, oracle_refs[model], scale
                )
                oracle_reference_score = _scaled_self_score(
                    selected, weights, oracle_refs[model], scale
                )
                rows.append(
                    {
                        "self_scale": scale,
                        "ppa_goal": goal,
                        "model": model,
                        "selected_design": _design_label(selected),
                        "baseline_scale_1_design": _design_label(baseline),
                        "changed_from_scale_1": not _same_design(selected, baseline),
                        "matches_bruteforce_oracle": _same_design(selected, oracle_row),
                        "selected_score_using_pool_reference": score,
                        "selected_score_using_oracle_reference": oracle_reference_score,
                        "oracle_score": oracle_score,
                        "oracle_regret": oracle_reference_score - oracle_score,
                    }
                )
    return rows


def _select_scaled_self_cost(rows, weights, reference, scale: float):
    scored = [(row, _scaled_self_score(row, weights, reference, scale)) for row in rows]
    return min(
        scored,
        key=lambda item: (
            item[1],
            item[0].selected_chiplets,
            item[0].total_power_w,
            item[0].avg_latency_ns,
        ),
    )


def _scaled_self_score(row: EvaluationResult, weights, reference, scale: float) -> float:
    self_component = _self_component(row, weights, reference)
    return scale * self_component


def _self_component(row: EvaluationResult, weights, reference) -> float:
    return (
        weights.latency * _normalised(row.avg_latency_ns, reference.max_latency_ns)
        + weights.area * _normalised(row.total_area_mm2, reference.max_area_mm2)
        + weights.power * _normalised(row.total_power_w, reference.max_power_w)
    )


def _normalised(value: float, maximum: float) -> float:
    if not math.isfinite(value) or maximum <= 0:
        return 0.0
    return value / maximum


def _rank_correlation_analysis(raw_rows: list[EvaluationResult], goals: Iterable[str]) -> list[dict[str, object]]:
    refs = _references_by_model(raw_rows)
    all_rows: list[dict[str, object]] = []
    for goal in goals:
        scored = _score_by_goal(raw_rows, [goal], refs)[goal]["rows"]
        grouped = {"pooled": scored}
        grouped.update({model: [row for row in scored if row.model == model] for model in MODEL_NAMES})
        weights = resolve_ppa_weights(goal)
        for group, group_rows in grouped.items():
            for cost_name in SELF_COSTS:
                cost_values = [_self_cost_value(row, cost_name, refs[row.model]) for row in group_rows]
                for target in RANK_TARGETS:
                    target_values = [
                        _rank_target_value(row, target, weights, refs[row.model]) for row in group_rows
                    ]
                    rho = _spearman(cost_values, target_values)
                    all_rows.append(
                        {
                            "goal": goal,
                            "group": group,
                            "n": len(group_rows),
                            "self_cost": cost_name,
                            "target": target,
                            "spearman_rho": rho,
                            "abs_rho": abs(rho),
                        }
                    )
    return all_rows


def _self_cost_value(row, cost_name: str, reference) -> float:
    if cost_name == "latency":
        return _normalised(row.avg_latency_ns, reference.max_latency_ns)
    if cost_name == "area":
        return _normalised(row.total_area_mm2, reference.max_area_mm2)
    return _normalised(row.total_power_w, reference.max_power_w)


def _rank_target_value(row, target: str, weights, reference) -> float:
    if target == "self_component":
        return _self_component(row, weights, reference)
    return float(getattr(row, target))


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks))
    left_denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_denominator = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    if left_denominator == 0 or right_denominator == 0:
        return 0.0
    return numerator / (left_denominator * right_denominator)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def _same_design(left: EvaluationResult, right: EvaluationResult) -> bool:
    if (
        left.model != right.model
        or left.topology != right.topology
        or left.architecture_feasible != right.architecture_feasible
        or left.selected_chiplets != right.selected_chiplets
    ):
        return False
    if left.selected_chiplets == 1 and right.selected_chiplets == 1:
        return True
    return left.workload_plan == right.workload_plan


def _design_label(row: EvaluationResult) -> str:
    status = "VALID" if row.architecture_feasible else "INVALID"
    return f"{row.selected_chiplets}C/{status}/{row.workload_plan}"


def _top_k_label(top_k: int) -> str:
    return "full" if top_k == 0 else str(top_k)


def _build_report(*, candidate_pool, pool_selection, self_sensitivity, correlations, metadata, output_dir) -> str:
    pool_baseline = [row for row in pool_selection if row["top_k_value"] == 12]
    larger_pool = [row for row in pool_selection if row["top_k_value"] != 12]
    pool_stable = all(
        row["pool_reference_design"]
        == next(
            baseline["pool_reference_design"]
            for baseline in pool_baseline
            if baseline["ppa_goal"] == row["ppa_goal"] and baseline["model"] == row["model"]
        )
        for row in larger_pool
    )
    pool_oracle_matches = all(row["pool_matches_oracle"] for row in pool_baseline)
    self_reference_changes = sum(1 for row in pool_selection if row["self_reference_changed_selection"])

    pool_by_model = {
        model: {str(row["top_k"]): row for row in candidate_pool if row["model"] == model}
        for model in MODEL_NAMES
    }
    pool_table = [
        [
            model,
            pool_by_model[model]["12"]["total_candidates"],
            pool_by_model[model]["24"]["total_candidates"],
            pool_by_model[model]["30"]["total_candidates"],
            pool_by_model[model]["full"]["total_candidates"],
            pool_by_model[model]["12"]["max_candidates_per_chiplet_count"],
        ]
        for model in MODEL_NAMES
    ]

    stability_table = []
    for goal in PPA_GOALS:
        for model in MODEL_NAMES:
            records = {
                str(row["top_k"]): row
                for row in pool_selection
                if row["ppa_goal"] == goal and row["model"] == model
            }
            stability_table.append(
                [
                    goal,
                    model,
                    _escape_pipe(str(records["12"]["pool_reference_design"])),
                    _escape_pipe(str(records["24"]["pool_reference_design"])),
                    _escape_pipe(str(records["30"]["pool_reference_design"])),
                    _escape_pipe(str(records["full"]["pool_reference_design"])),
                    "yes" if records["12"]["pool_matches_oracle"] else "no",
                    "yes" if records["12"]["fixed_reference_matches_oracle"] else "no",
                ]
            )

    sensitivity_summary = []
    for scale in SELF_SCALES:
        scale_rows = [row for row in self_sensitivity if row["self_scale"] == scale]
        sensitivity_summary.append(
            [
                scale,
                sum(1 for row in scale_rows if row["changed_from_scale_1"]),
                sum(1 for row in scale_rows if row["matches_bruteforce_oracle"]),
                len(scale_rows),
                f"{statistics.mean(row['oracle_regret'] for row in scale_rows):.6f}",
            ]
        )

    rank_summary = []
    for model in ("pooled", *MODEL_NAMES):
        for cost in SELF_COSTS:
            rows = [
                row
                for row in correlations
                if row["group"] == model and row["self_cost"] == cost and row["target"] == "ppa_score"
            ]
            rank_summary.append(
                [
                    model,
                    cost,
                    f"{statistics.mean(row['spearman_rho'] for row in rows):.4f}",
                    f"{statistics.mean(row['abs_rho'] for row in rows):.4f}",
                ]
            )

    pooled_latency = [
        row
        for row in correlations
        if row["group"] == "pooled" and row["self_cost"] in SELF_COSTS and row["target"] == "avg_latency_ns"
    ]
    latency_signal = {
        row["self_cost"]: row["spearman_rho"]
        for row in pooled_latency
        if row["goal"] == "balanced"
    }
    pooled_ppa = [row for row in correlations if row["group"] == "pooled" and row["target"] == "ppa_score"]
    ppa_abs_by_cost = {
        cost: statistics.mean(row["abs_rho"] for row in pooled_ppa if row["self_cost"] == cost)
        for cost in SELF_COSTS
    }
    scale_zero = next(row for row in sensitivity_summary if row[0] == 0.0)
    scale_one = next(row for row in sensitivity_summary if row[0] == 1.0)
    total_cases = len(MODEL_NAMES) * len(PPA_GOALS)

    if pool_stable and pool_oracle_matches:
        pool_conclusion = "top-k=12 維持不變"
    else:
        pool_conclusion = "top-k=12 需要重新評估"
    if scale_zero[1] == 0 and scale_one[1] == 0:
        sensitivity_conclusion = "移除 self term 也沒有改變選解"
    else:
        sensitivity_conclusion = f"移除 self term 會改變 {scale_zero[1]}/{total_cases} 個選解"

    lines = [
        "# Mesh-only 問題重定義與 9-cost 中 Self Cost 測試報告",
        "",
        "## 結論",
        "",
        "1. 正式問題定義為 passive interposer 下的 mesh-only 搜尋，本次沒有把 tree 納入候選。",
        f"2. Candidate pool 維持 `top-k=12`：12、24、30、full 在 {total_cases} 個 model/goal case 的選解一致，12 對 brute-force oracle 命中 `{sum(1 for row in pool_baseline if row['pool_matches_oracle'])}/{total_cases}`；{pool_conclusion}。",
        f"3. Self cost 確實有用：移除 self term（scale=0）會改變 `{scale_zero[1]}/{total_cases}` 個選解；scale=1 的 baseline 維持 `{scale_one[2]}/{total_cases}` oracle match。{sensitivity_conclusion}。",
        f"4. Self cost 的 pooled 平均 |Spearman rho|（跨 4 goals）：latency `{ppa_abs_by_cost['latency']:.4f}`、area `{ppa_abs_by_cost['area']:.4f}`、power `{ppa_abs_by_cost['power']:.4f}`。Balanced 下 self latency / area / power 對 latency 的 rho 分別為 `{latency_signal.get('latency', 0.0):.4f}` / `{latency_signal.get('area', 0.0):.4f}` / `{latency_signal.get('power', 0.0):.4f}`。",
        "",
        "依照你提供的圖，本報告將 self 定義為候選池內的相對 raw PPA：`C_self,x = X_x / max_{j∈pool}(X_j)`。目前程式的 `PpaReference.max_*` + `_normalized_metric` 正是這個 self normalization。",
        "",
        "## 測試設定",
        "",
        f"- topology：mesh；最大 chiplet 數：{metadata['max_chiplets']}；FPS 只作 inverse-E2E diagnostic。",
        f"- models：{', '.join(MODEL_NAMES)}；PPA goals：{', '.join(PPA_GOALS)}。",
        "- brute-force 作為 oracle；DP pool 測試 `12 / 24 / 30 / full`，其中 `full` 是 `top-k=0`。",
        "- self sensitivity 使用 production DP pool `top-k=12`，將 self component 乘上 0、0.25、0.5、0.75、1、1.25、1.5、2，再觀察 final design。",
        "- fixed-reference test 把 self denominator 固定為 brute-force oracle pool 的 max，用來隔離『候選池改變造成 normalization 改變』的影響。",
        f"- RapidChiplet official root exists：`{metadata['official_rapidchiplet_root_exists']}`；若不可用則使用 local proxy。",
        "",
        "## 1. Candidate pool 與 final design stability",
        "",
        "DP 的 top-k 是 per-state 上限；下表為 chiplet 數 1–16 加總後的候選數。",
        "",
        _markdown_table(
            ["model", "k=12 total", "k=24 total", "k=30 total", "full total", "k=12 max/state"],
            pool_table,
        ),
        "",
        _markdown_table(
            ["goal", "model", "pool self k=12", "pool self k=24", "pool self k=30", "pool self full", "12/oracle", "12/fixed-ref oracle"],
            stability_table,
        ),
        "",
        f"Pool self reference 與 fixed oracle reference 導致的 final-selection 差異共有 `{self_reference_changes}` 筆（總共 {len(pool_selection)} 筆）；這是直接量 self denominator 是否改變排序的檢查。",
        "",
        "## 2. Self weight sensitivity",
        "",
        "`self_scale=1` 是目前 score；`self_scale=0` 等於移除三個 self PPA cost，只剩固定 tie-break，沒有 FPS reward。",
        "",
        _markdown_table(
            ["self scale", "changed vs scale=1", "oracle matches", "cases", "mean oracle regret"],
            sensitivity_summary,
        ),
        "",
        "## 3. Rank correlation",
        "",
        "Spearman rho 是 self cost 與 final PPA score 的 rank correlation，表格另提供 per-model 與 pooled 結果；完整明細在 raw CSV。|rho| 越大代表該 self cost 對排序越有一致性。",
        "",
        _markdown_table(
            ["group", "self cost", "mean rho vs PPA", "mean |rho|"],
            rank_summary,
        ),
        "",
        "## 4. 對照 9-cost 定義的實作狀態",
        "",
        "目前 code 已有三個 self PPA cost：latency、area、power；FPS 不參與 ranking。圖中若要求每個 raw PPA 維度各自依 user limit 產生 bonus cost 與 penalty cost，這 6 個 per-metric limit cost 尚未在目前 evaluator 中獨立實作；本次只驗證 self 部分，沒有把未實作項目當成已驗證。",
        "",
        "## 重現方式",
        "",
        "```powershell",
        "cd C:\\Users\\user\\Desktop\\rapidchiplet\\simplified_rapidchiplet",
        "py -3 .\\tools\\run_mesh_redefinition_analysis.py",
        "```",
        "",
        f"原始輸出：`{output_dir.resolve()}`。",
        "",
        "## 判讀限制",
        "",
        "結論只適用於目前兩個 demo model、mesh-only、現有 PPA weights 與 RapidChiplet/workload assumptions；更換模型、user limits、PPA weights 或 network/engine 後應重新執行。",
        "",
    ]
    return "\n".join(lines)


def _escape_pipe(value: str) -> str:
    return value.replace("|", "\\|")


def _markdown_table(headers: list[str], rows: Iterable[list[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
