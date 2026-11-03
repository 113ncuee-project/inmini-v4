"""Independent arithmetic/accounting checks on completed v0.2 results.

This does not run an oracle or modify a checkpoint. Its report explicitly
distinguishes audited completed cases from experiments that are still running.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cost(metrics, reference, objective, penalized=False):
    weights = [('latency_s', 'latency_weight'), ('area_mm2', 'area_weight'),
               ('power_w', 'power_weight'), ('energy_j', 'energy_weight')]
    total = sum(objective[name] for _, name in weights)
    value = sum(objective[name] / total * metrics[key] / reference[key]
                for key, name in weights if objective[name])
    violation = 0.
    for key, suffix in [('latency_s', 'latency_s'), ('area_mm2', 'area_mm2'), ('power_w', 'power_w')]:
        goal = objective.get('soft_' + suffix)
        if goal:
            value += max(0., metrics[key] / goal - 1.)
        limit = objective.get('max_' + suffix)
        if limit:
            violation += max(0., metrics[key] / limit - 1.)
    return value + (objective['violation_penalty'] * violation if penalized else 0.)


def audit_run(folder, snapshot):
    config, source = read(folder / 'effective_config.json'), read(folder / 'source_identity.json')
    errors, arms = [], []
    def check(condition, message):
        if not condition:
            errors.append(message)
    def near(actual, expected, message):
        check(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-14), message)
    for relative, expected in source['sources'].items():
        check(hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() == expected,
              'source mismatch: ' + relative)
    seeds = read(folder / 'seed_attempts.json')
    seed_ledger = read(folder / 'seed_worker/accounting.json')
    seed_full = sum(row.get('full_evaluations', 0) for row in seed_ledger if row['operation'] == 'evaluate')
    check(seed_full == sum(bool(row.get('ppa_completed')) for row in seeds['attempts']), 'seed full PPA ledger mismatch')
    references = []
    for result_file in folder.glob('*/joint_search.json'):
        arm = result_file.parent.name
        result, checkpoint = read(result_file), read(result_file.parent / 'checkpoint.json')
        summary, reference, objective = result['summary'], checkpoint['reference'], result['objective']
        references.append({key: reference[key] for key in ('latency_s', 'area_mm2', 'power_w', 'energy_j')})
        ledger = read(result_file.parent / 'worker/accounting.json')
        full = sum(row.get('full_evaluations', 0) for row in ledger if row['operation'] == 'evaluate')
        check(seed_full + full == summary['full_ppa_completed'], arm + ': full PPA ledger mismatch')
        check(summary['full_ppa_completed'] <= config['queries'], arm + ': full query ceiling exceeded')
        check(summary['cache_hits'] == result['requests'] - result['evaluations'] - result.get('oracle_timeouts', 0),
              arm + ': cache/timeout accounting mismatch')
        check(all(not row.get('full_evaluations') for row in ledger if row['operation'] != 'evaluate'),
              arm + ': artifact export charged as full PPA')
        # Do not infer a duration from process start times; use the cumulative
        # measured ledgers and permit only the documented reporting tail.
        check(summary['wall_s_including_common_seed_charge'] <= config['wall_ceiling_per_arm_including_common_seed_charge_s'] + 60,
              arm + ': wall ledger exceeds ceiling plus reporting tail')
        by_state = {}
        for metrics in checkpoint['cache'].values():
            if not all(key in metrics for key in ('latency_s', 'power_w', 'energy_j')):
                continue
            near(metrics['power_w'] * metrics['latency_s'], metrics['energy_j'], arm + ': P*T != E')
            near(sum(metrics['energy_components'].values()), metrics['energy_j'], arm + ': component energy != total')
            check(all(value >= 0 and math.isfinite(value) for value in metrics['energy_components'].values()),
                  arm + ': negative/nonfinite energy')
            check(metrics['physical_cores'] == 0 and metrics['pes'] == 16 * metrics['compute_chiplets'],
                  arm + ': PE/core resource mismatch')
            check(metrics['native_rapidchiplet_executed'] is False, arm + ': backend provenance mismatch')
            if metrics['feasible']:
                check(not metrics['violations'], arm + ': feasible state has physical violations')
            by_state[metrics['design_state_hash']] = metrics
        for row in result['trace']:
            p = row['behavior_probability']
            check(0 < p <= 1 and math.isfinite(p), arm + ': behavior probability out of range')
            near(math.log(p), row['behavior_log_probability'], arm + ': behavior log probability mismatch')
            evidence = row.get('candidate_shortlist')
            if evidence:
                path = Path(evidence['path'])
                payload = path.read_bytes()
                check(hashlib.sha256(payload).hexdigest() == evidence['sha256'], arm + ': shortlist hash mismatch')
                check(len(json.loads(gzip.decompress(payload))['plans']) == row['candidate_count'], arm + ': shortlist count mismatch')
            if row.get('candidate_probabilities') is not None:
                probs = row['candidate_probabilities']
                check(len(probs) == row['candidate_count'], arm + ': probability vector size mismatch')
                near(sum(probs), 1., arm + ': probability sum mismatch')
                check(len(row['observation']['state_features']) == 24 and len(row['observation']['selected_candidate_features']) == 283, arm + ': schema3 feature dimensions')
            if row['valid']:
                before = by_state.get(row['before_state_hash'])
                after = row['current_metrics']
                check(before is not None, arm + ': before state missing from exact cache')
                if before is not None:
                    expected = cost(before, reference, objective, True) - cost(after, reference, objective, True)
                    near(row['reward'], expected, arm + ': reward != physical cost difference')
            if row.get('physical_validity') == 'unknown':
                check(row['reward'] is None and row['advantage'] is None, arm + ': timeout used for training')
                check(not row['cache_hit'] and row['before_state_hash'] == row['after_state_hash'],
                      arm + ': timeout changed state or became a cache hit')
        expected_updates = sum(row['valid'] for row in result['trace']) if arm in ('atomic_rl', 'bottleneck_rl') else 0
        check(result['policy_updates'] == expected_updates, arm + ': policy update count mismatch')
        if result['best_metrics']:
            near(result['best_cost'], cost(result['best_metrics'], reference, objective), arm + ': archive cost mismatch')
        coverage = read(result_file.parent / 'fusion_coverage.json')
        best = summary.get('best') or {}
        seed_cost = cost(result['initial_metrics'], reference, objective)
        previous, improved_steps = seed_cost, []
        for row in result['trace']:
            if row['best_cost'] < previous - 1e-12:
                improved_steps.append(row['step'])
                previous = row['best_cost']
        tail_reference = result['trace'][-11]['best_cost'] if len(result['trace']) > 10 else seed_cost
        arms.append({'model': config['model'], 'fusion': config['fusion'], 'arm': arm,
            'full_ppa': summary['full_ppa_completed'], 'search_full_ppa': full,
            'policy_updates': result['policy_updates'], 'oracle_timeouts': summary['oracle_timeouts'],
            'wall_s': summary['wall_s_including_common_seed_charge'], 'best_cost': summary['best_cost'],
            'seed_improvement_pct': summary['improvement_over_best_seed_pct'],
            'latency_ms': best.get('latency_s', 0.) * 1000, 'area_mm2': best.get('area_mm2'),
            'power_w': best.get('power_w'), 'compute': best.get('compute_chiplets'), 'io': best.get('io_chiplets'),
            'max_full_fusion_depth': coverage['max_full_evaluated_depth'],
            'selected_steps': len(result['trace']), 'best_seed_cost': seed_cost,
            'first_improvement_step': improved_steps[0] if improved_steps else None,
            'last_improvement_step': improved_steps[-1] if improved_steps else None,
            'last_ten_selected_steps_cost_gain': tail_reference - result['best_cost'], 'output': str(folder)})
    check(not references or all(item == references[0] for item in references), 'methods used different references')
    return {'folder': str(folder), 'completed_arms': len(arms), 'passed': not errors, 'errors': errors,
            'cached_metrics_checked': sum(len(read(p.parent / 'checkpoint.json')['cache'])
                                         for p in folder.glob('*/joint_search.json'))}, arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    snapshot = root / 'source_snapshot'
    manifest = read(snapshot / 'snapshot_manifest.json')
    mismatches = [relative for relative, expected in manifest['files'].items()
                  if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != expected]
    runs, rows = [], []
    for state_path in sorted(root.rglob('run_state.json')):
        case_status = read(state_path).get('status')
        if not any(state_path.parent.glob('*/joint_search.json')):
            continue
        run, items = audit_run(state_path.parent, snapshot)
        run['case_status'] = case_status
        runs.append(run)
        rows.extend(items)
    extra = []
    finite = root / 'finite_toy'
    if (finite / 'summary.json').exists():
        summary = read(finite / 'summary.json')
        states = {row['state_hash']: row for row in read(finite / 'results.json')}
        pairs = read(finite / 'paired_fusion.json')
        errors = []
        for pair in pairs:
            on, off = states[pair['on_state_hash']], states[pair['off_state_hash']]
            if on['paired_off_state_hash'] != off['state_hash'] or off['fusion'] != 'off':
                errors.append('incorrect pair identity')
            if not math.isclose(on['cost'] - off['cost'], pair['cost_delta_on_minus_off'], abs_tol=1e-12):
                errors.append('incorrect paired cost difference')
            for key in ('latency_s', 'energy_j', 'executed_macs', 'dram_read_bytes', 'dram_write_bytes', 'noc_hop_bytes', 'unit_count'):
                if not math.isclose(on[key] - off[key], pair['delta_' + key], rel_tol=1e-9, abs_tol=1e-14):
                    errors.append('incorrect paired difference: ' + key)
        if len(states) != summary['unique_domain_states'] or len(pairs) != summary['paired_fusion_count']:
            errors.append('finite-domain coverage mismatch')
        extra.append({'scope': 'finite_toy', 'states_checked': len(states), 'pairs_checked': len(pairs),
                      'passed': not errors, 'errors': errors})
    for path in root.glob('counterfactual*/summary.json'):
        summary = read(path)
        if summary['status'] != 'completed_matched_pair':
            extra.append({'scope': str(path.parent), 'status': summary['status'], 'passed': True,
                          'note': 'not a completed pair; no benefit conclusion'})
            continue
        selected = read(path.parent / 'selected_state.json')
        on = dict(selected['state']['mapping'])
        on['fusion_stacks'] = []
        errors = []
        if on != selected['off_mapping']:
            errors.append('counterfactual changed fields other than fusion_stacks')
        if not summary['replayed_on_matches_original']:
            errors.append('on replay changed the original physical PPA')
        for intervention in ('on', 'off'):
            metrics = read(path.parent / intervention / 'evaluation.json')['metrics']
            if not math.isclose(metrics['power_w'] * metrics['latency_s'], metrics['energy_j'], rel_tol=1e-9):
                errors.append(intervention + ': P*T != E')
            if not math.isclose(sum(metrics['energy_components'].values()), metrics['energy_j'], rel_tol=1e-9):
                errors.append(intervention + ': energy component sum mismatch')
        extra.append({'scope': str(path.parent), 'passed': not errors, 'errors': errors,
                      'extra_full_ppa': summary['extra_full_ppa']})
    output = {'passed_completed_runs': not mismatches and all(run['passed'] for run in runs + extra),
              'snapshot_identity': manifest['identity'], 'snapshot_mismatches': mismatches, 'runs': runs,
              'additional_checks': extra,
              'scope': 'completed arms only; a containing case may still be running; not a declaration that the research suite is complete'}
    (root / 'audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if rows:
        with (root / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    modern = any(row['output'] and read(Path(row['output'])/'effective_config.json')['hardware_profile'].endswith('_3') for row in rows)
    lines = ['# INMIN ' + ('v0.3' if modern else 'v0.2') + ' 研究結果（持續更新）', '',
        '更新 UTC：' + datetime.now(timezone.utc).isoformat(), '',
        '本檔只呈現已完成並審計的實驗；整個研究尚未完成時，不把部分結果當成最終結論。報告僅文字與數字表格。', '',
        '來源 snapshot：`' + manifest['identity'] + '`。新 profile 每 Compute 16PE、L1 2.75MiB、共享L2 1MiB、無實體Core/額外UB；Batch1、Seed7、balanced、無外加PPA硬限制。', '',
        '新 oracle 是 Python 顯式網路/資源 calendar，沒有執行 native RapidChiplet。PPA 未校準，不是晶片量測或timing signoff。舊硬體結果不混入本表。', '',
        '## 完整模型與搜尋對照', '',
        '各case上限與engine版本見effective_config；v0.3 seed與search有獨立時間預算，common seed實際成本仍向各方法計費。Timeout/reject/export另列帳。相同上限不保證相同完成次數；未達相同完成查詢時，不能宣稱query efficiency勝出。', '',
        '|模型|Fusion|方法|Full PPA|Policy更新|Timeout|Wall s|最佳成本|種子後改善 %|Latency ms|Area mm²|Power W|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if 'snapshot_smoke' in row['output']:
            continue
        values = [row[key] for key in ('model', 'fusion', 'arm', 'full_ppa', 'policy_updates', 'oracle_timeouts', 'wall_s',
                   'best_cost', 'seed_improvement_pct', 'latency_ms', 'area_mm2', 'power_w')]
        lines.append('|' + '|'.join(f'{value:.8g}' if isinstance(value, float) else str(value) for value in values) + '|')
    status_path = root / 'results/suite_status.json'
    if status_path.exists():
        suite = read(status_path)
        lines.extend(['', '整個正式佇列狀態：`' + suite['status'] + '`。', '', '|Case|狀態|', '|---|---|'])
        observed = {item['case']: item['status'] for item in suite['cases']}
        for model in ('toy', 'resnet18', 'resnet50', 'mobilenet_v2'):
            for mode in ('fusion_on', 'fusion_off') if model != 'toy' else ('fusion_on',):
                key = model + '/' + mode
                lines.append('|' + key + '|' + observed.get(key, 'queued_not_run') + '|')
    lines.extend(['', '## 種子與搜尋軌跡', '',
        '這裡的step是一個所選交易；Greedy的lookahead與失敗/timeout成本另在查詢帳本，不能把step視為完整PPA次數。最後10步沒有改善只表示這段已觀察軌跡停滯，不能證明全域收斂。', '',
        '|模型|Fusion|方法|最佳種子成本|選擇步數|首個改善step|最後改善step|最後10選擇步的成本下降|最高實評DF深度|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for row in rows:
        if 'snapshot_smoke' in row['output']:
            continue
        values = [row[key] for key in ('model', 'fusion', 'arm', 'best_seed_cost', 'selected_steps',
                  'first_improvement_step', 'last_improvement_step', 'last_ten_selected_steps_cost_gain', 'max_full_fusion_depth')]
        lines.append('|' + '|'.join('none_observed' if value is None else f'{value:.8g}' if isinstance(value, float) else str(value) for value in values) + '|')
    if (finite / 'summary.json').exists():
        summary, domain = read(finite / 'summary.json'), read(finite / 'domain.json')
        all_rows, pairs = read(finite / 'results.json'), read(finite / 'paired_fusion.json')
        lines.extend(['', '## 固定硬體的有限 Toy 枚舉', '',
            f"固定1 Compute/2 IO，380個去重狀態，{summary['full_evaluations']}次完整PPA（含baseline），耗時{summary['elapsed_s']:.3f}秒。範圍是Temporal/Spatial、PE配額1/2/4、K/W分割、contiguous/spread placement、IO0/interleave、off及三種cache策略、2/4/7方形tile。這是有限域最佳，不是全域最佳，也不與可變架構搜尋直接算regret。", '',
            '|有限域最佳|成本（baseline=1）|Latency µs|Energy µJ|Power W|DRAM讀 B|重算 MAC|Units|',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for flag, label in ((False, 'Fusion off'), (True, 'Fusion on')):
            best = min((row for row in all_rows if (row['fusion'] != 'off') == flag), key=lambda row: row['cost'])
            values = [label, best['cost'], best['latency_s'] * 1e6, best['energy_j'] * 1e6, best['power_w'],
                      best['dram_read_bytes'], best['recompute_macs'], best['unit_count']]
            lines.append('|' + '|'.join(f'{value:.10g}' if isinstance(value, float) else str(value) for value in values) + '|')
        lines.extend(['', '|同Mapping Fusion on−off 指標|下降組數|持平組數|增加組數|', '|---|---:|---:|---:|'])
        for key, label in [('cost_delta_on_minus_off', '綜合成本'), ('delta_latency_s', '延遲'),
                           ('delta_energy_j', '能量'), ('delta_dram_read_bytes', 'DRAM讀取'), ('delta_executed_macs', '執行MAC')]:
            counts = [sum(row[key] < 0 for row in pairs), sum(row[key] == 0 for row in pairs), sum(row[key] > 0 for row in pairs)]
            lines.append('|' + label + '|' + '|'.join(map(str, counts)) + '|')
        lines.extend(['', '綜合成本包含平均功耗E/T，因此分數改善可能同時伴隨延遲和能量變差。此Toy的31組綜合成本改善，不能寫成31組加速；342組裡沒有延遲或能量下降。此有限域最佳是無Fusion，不能推論所有DNN/Mapping的Fusion都無效。'])
    lines.extend(['', '## 固定Mapping的Fusion介入', '',
        '每組只移除fusion_stacks，保留硬體/groups/PE配額與placement/partition/DRAM等全部欄位；重新執行on和off，額外PPA單獨計費。負差表示Fusion on降低該指標；綜合cost降低不必然更快。這是選中的樣本，不代表全部候選。', '',
        '|Case|狀態|額外Full PPA|架構|On回放一致|Cost差|Latency差 µs|Energy差 µJ|DRAM讀差 B|執行MAC差|',
        '|---|---|---:|---|---|---:|---:|---:|---:|---:|'])
    for path in sorted(root.glob('counterfactual*/summary.json')):
        if path.parent.name == 'counterfactual_smoke':
            continue
        pair = read(path)
        delta = pair.get('delta_on_minus_off') or {}
        values = [path.parent.name, pair['status'], pair['extra_full_ppa'], pair.get('architecture', 'not_run'),
                  pair.get('replayed_on_matches_original'), pair.get('cost_delta_on_minus_off'),
                  delta.get('latency_s', 0.) * 1e6 if delta else None,
                  delta.get('energy_j', 0.) * 1e6 if delta else None,
                  delta.get('dram_read_bytes'), delta.get('executed_macs')]
        lines.append('|' + '|'.join('not_run' if value is None else f'{value:.9g}' if isinstance(value, float) else str(value) for value in values) + '|')
    lines.extend(['', '## 證據與限制', '',
        ('已通過113項來源穩定驗收（含50布局、A12、target features、memory引導與舊回歸）' if modern else '已通過103項固定來源驗收（舊版回歸亦包含）') + '。獨立審計檢查P×T=E、energy分項總和、reward物理差值、機率/shortlist hash、query/export/timeout帳本、PE資源與來源hash；詳細結果見audit.json。審計完成不代表尚在排隊的實驗已完成。', '',
        '所有M/T實值見各case hardware_profile.json；技術設計與修改清單見workspace docs/INMIN_V02_IMPLEMENTATION.md。4×4 PE mesh/four corner gateways、queues/protocol、INT24 rounding/saturation、共同scale Add、16nm成本、DRAM面積/idle、VPU/IO/register proxy均為分析假設。Whole-flow reservation保守，SRAM bank/word-tail、cycle-accurate VC、DDR refresh、量化準確率、實體校準未完成。MAC .024pJ的op/MAC convention有歧義，2×敏感度尚未做。', '',
        '只有Seed7；不能作跨seed統計保證。未實作C10、內部C-panel、standalone BatchNorm；提案Fusion depth≤4。完整per-layer mapping、traffic、深度覆蓋、未選/拒絕/timeout原因與best archive見各case REPORT.md及JSON。', '',
        '最終需分別回答：RL是否在最佳已驗證seed之外找到更好的設計；是否優於同預算Random/Greedy/atomic RL。正式三模型未完成前，兩者均不預設正面答案。'])
    (root / 'RESEARCH_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False))
    if not output['passed_completed_runs']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
