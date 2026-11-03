"""Export a completed shared screen; never imports an oracle or a policy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    common = read(args.case / 'common.json')
    screen = read(args.case / 'screening.json')
    config = read(args.case / 'effective_config.json')
    if config['objective'] != 'edp' or any(config.get(k) is not None for k in ('max_area', 'max_power', 'max_latency_ms')):
        raise ValueError('this export expects the unconstrained, predeclared EDP study')
    if not screen['completed']:
        raise ValueError('shared phase must finish before exporting')
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = common['metrics']
    rows, valid = [], []
    for item in screen['rows']:
        metrics = item.get('metrics', {})
        data = dict(item['trial']['state'])
        data.pop('evaluation_report', None)
        data.pop('remaining_budget', None)
        state_hash = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        row = dict(architecture=item['architecture'], status=item['status'], feasible=metrics.get('feasible'),
            objective=config['objective'], cost=item.get('cost'), state_hash=state_hash)
        for key in ('latency_s', 'energy_j', 'power_w', 'area_mm2', 'executed_macs', 'recompute_macs',
                    'dram_read_bytes', 'dram_write_bytes', 'noc_hop_bytes', 'nop_hop_bytes'):
            row[key] = metrics.get(key)
        rows.append(row)
        if metrics.get('feasible') and item.get('cost') is not None:
            if metrics['executed_macs'] != baseline['executed_macs'] + metrics['recompute_macs']:
                raise ValueError('MAC conservation failure')
            valid.append(item)
    with (args.out / 'r18_shared_screen.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min(valid, key=lambda r: r['cost'])
    bounded = [r for r in valid if all(r['metrics'][k] <= baseline[k] * (1 + 1e-12)
               for k in ('latency_s', 'energy_j', 'area_mm2'))]
    choices = [('Best shared EDP', best)]
    if bounded:
        choices.append(('Post-hoc area/latency/energy <= baseline', min(bounded, key=lambda r: r['cost'])))
    all_ppa = [r for r in bounded if r['metrics']['power_w'] <= baseline['power_w'] * (1 + 1e-12)]
    if all_ppa:
        choices.append(('Post-hoc all four metrics <= baseline', min(all_ppa, key=lambda r: r['cost'])))
    summary = dict(scope='pre-RL architecture + fixed inner mapping screen; no policy-search gain claimed',
        source_identity=read(args.case / 'source_snapshot_manifest.json')['identity'], objective=config['objective'],
        common_full_ppa=common['full_ppa'], screen_full_ppa=screen['full_ppa'], feasible_screen_points=len(valid),
        complete_domain_attempted=screen['complete_domain_attempted'], baseline_state=common['initial_state'],
        baseline_metrics=baseline, seed_cost=common['seed_cost'], choices=[])
    lines = ['# ResNet18 完整架構篩選：已完成的前置階段', '',
        '這是 RL 精修前的共享架構／內層結果。主研究 RL／SA／Random 當時仍在執行；此表不能當作 RL policy 的收益。', '',
        f'完整架構 PPA：{screen["full_ppa"]}；可行：{len(valid)}；另有 {common["full_ppa"]} 次原始 reference／最佳 seed 重驗。', '',
        '|Point|Architecture|Latency ms|Energy mJ|Area mm²|Power W|EDP改善%|Latency改善%|Energy改善%|Area改善%|Power改善%|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        f'|Original best Tangram-style seed|{common["initial_state"]["architecture_candidate_id"]}|{baseline["latency_s"]*1000:.6f}|{baseline["energy_j"]*1000:.6f}|{baseline["area_mm2"]:.6f}|{baseline["power_w"]:.6f}|0|0|0|0|0|']
    for label, item in choices:
        m = item['metrics']
        gains = [100 * (1 - item['cost'] / common['seed_cost'])] + [100 * (1 - m[k] / baseline[k]) for k in ('latency_s', 'energy_j', 'area_mm2', 'power_w')]
        lines.append(f'|{label}|{item["architecture"]}|{m["latency_s"]*1000:.6f}|{m["energy_j"]*1000:.6f}|{m["area_mm2"]:.6f}|{m["power_w"]:.6f}|' + '|'.join(f'{v:.4f}' for v in gains) + '|')
        summary['choices'].append(dict(label=label, state=item['trial']['state'], metrics=m, cost=item['cost'], gains=gains))
    lines += ['', '後兩點是從已測 archive 事後選出、滿足明列指標上限的完整設計，沒有額外訓練或額外 PPA，不能稱為另一次有硬限制的RL訓練。第一點主要目標只有 EDP，面積及Power會增加，必須同表呈現。',
        '全部是未校準的解析 PPA。MAC守恆已核對；完整50點見CSV，各點完整mapping見JSON。',
        '比較起點是專案的bounded Tangram-style DP/beam，不是原版Tangram或Gemini二進位。來源與研究設定見夜間協定。']
    (args.out / 'R18_SHARED_SCREEN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (args.out / 'r18_shared_screen_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(full_ppa=screen['full_ppa'], feasible=len(valid), best_architecture=best['architecture'],
                         best_gain_pct=100 * (1 - best['cost'] / common['seed_cost']))))


if __name__ == '__main__':
    main()
