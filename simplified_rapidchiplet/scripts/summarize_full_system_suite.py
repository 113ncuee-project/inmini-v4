"""Regenerate an overnight text/numeric report from completed case evidence."""
import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev
from run_local_rl_diagnostics import read, write


def summarize(root):
    root = Path(root).resolve()
    rows = []
    for path in sorted(root.glob('*/*/summary.json')):
        if not (path.parent / 'effective_config.json').exists():
            continue
        model, seed = path.parent.parent.name, path.parent.name
        for method, result in read(path, {}).items():
            if not isinstance(result, dict) or result.get('method') != method:
                continue
            rows.append(dict(result, model=model, rng=seed,
                             case=str(path.parent.relative_to(root))))
    lines = ['# INMIN Full System 夜間實驗數值報告', '',
        '只列已完成方法；正在執行的案例請查看各 status.json／runner.log。', '',
        '|Model|RNG|Method|Full PPA|Cost改善%|篩選後改善%|Latency改善%|Energy改善%|Search s|架構|Fusion深度|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    keys = ('model', 'rng', 'method', 'full_ppa', 'gain_vs_seed_pct', 'gain_after_shared_screen_pct',
            'latency_improvement_pct', 'energy_improvement_pct', 'search_wall_s', 'best_architecture', 'best_fusion_depth')
    for row in rows:
        lines.append('|' + '|'.join(f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k]) for k in keys) + '|')
    aggregates = []
    lines += ['', '|Model|Method|完成RNG數|Cost改善平均%|中位數%|標準差%|', '|---|---|---:|---:|---:|---:|']
    for model, method in sorted({(r['model'], r['method']) for r in rows}):
        values = [r['gain_vs_seed_pct'] for r in rows if r['model'] == model and r['method'] == method]
        stats = dict(model=model, method=method, n=len(values), mean=mean(values), median=median(values),
                     std=stdev(values) if len(values) > 1 else None)
        aggregates.append(stats)
        lines.append(f'|{model}|{method}|{len(values)}|{stats["mean"]:.4f}|{stats["median"]:.4f}|{stats["std"] if stats["std"] is not None else "NA"}|')
    lines += ['', 'Cost改善包含共同架構／內層篩選收益；篩選後改善另列。所有PPA均為解析估計。',
        '相同查詢／相同時間比較請讀各案例 analysis/equal_budget.json；configured ceiling不能代替實際完整查詢數。',
        'SA使用同一套INMIN完整空間與Oracle，不是原版Gemini binary。最佳設計請看各方法 recommendation.json；硬限制失敗時 state為null。',
        '單一RNG或有限樣本不足以宣稱RL穩定優於對照或已達到全域最佳。', '', '## 完整案例', '']
    for case in sorted({r['case'] for r in rows}):
        lines.append(f'- [{case}]({case}/REPORT.md) · [公平預算]({case}/analysis/equal_budget.json) · [審計]({case}/analysis/audit.json)')
    root.mkdir(exist_ok=True, parents=True)
    (root / 'FULL_NIGHT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(root / 'suite_summary.json', dict(completed_methods=rows, aggregates=aggregates))
    print(json.dumps(dict(report=str(root / 'FULL_NIGHT_REPORT.md'), completed_methods=len(rows))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    summarize(p.parse_args().out)
