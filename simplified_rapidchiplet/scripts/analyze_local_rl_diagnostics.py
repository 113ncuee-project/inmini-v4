"""Reconstruct text/numeric evidence without executing a workload or oracle.

Missing evidence stays unknown. Never invent rewards, throughput or causal labels.
"""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

from run_local_rl_diagnostics import read, write, sha, verify_snapshot

def ledger(path):
    data = read(path, [])
    return data if isinstance(data, list) else []

def jsonl(path, warnings):
    rows = []
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except ValueError:
                warnings.append(f'{path}:{line_no}: incomplete/corrupt JSONL retained')
    return rows

def save_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)

def remaining_budget(path, out):
    full=sum(r.get('full_evaluations',0) for r in ledger(path))
    if (path.parents[2]/'effective_config.json').exists():
        seed=sum(bool(r.get('ppa_completed')) for r in read(path.parents[2]/'seed_attempts.json',{}).get('attempts',[]))
        return max(0,read(out/'effective_config.json',{}).get('budgets',{}).get('queries',0)-seed-full)
    return max(0,(10 if path.parents[1].name=='action_pairs' else 2)-full)


def status(out):
    out = Path(out)
    state = read(out/'suite_status.json', {'status':'not_started'})
    records = [r for path in (out/'cases').rglob('accounting.json') for r in ledger(path)]
    return dict(state=state, unique_completed_full_ppa=sum(r.get('full_evaluations',0) for r in records),
        timeouts=sum(r.get('status')=='timeout' for r in records),
        in_progress=sum(r.get('status')=='running' for r in records),
        stop_requested=(out/'STOP_REQUESTED').exists(),
        arms=[dict(path=str(path.parent.relative_to(out)), summary=read(path,{})) for path in (out/'cases').rglob('summary.json')],
        checkpoints=[dict(path=str(path.parent.relative_to(out)),status=data.get('status'),steps=data.get('steps'),
            best_cost=data.get('best_cost'),elapsed_search_s=data.get('elapsed_s'),policy_updates=data.get('policy',{}).get('updates'))
            for path in (out/'cases').rglob('checkpoint.json') if (data:=read(path,{}))],
        remaining_full_ppa_by_worker={str(path.parent.relative_to(out)): remaining_budget(path,out) for path in (out/'cases').rglob('accounting.json') if path.parent.name=='worker'},
        note='Stored state only; running after a host crash may be stale. Resume acquires OS lock before work.')

def physical_checks(metrics, label, errors):
    if not metrics or not metrics.get('feasible'):
        return
    try:
        t,e,p = [metrics[k] for k in ('latency_s','energy_j','power_w')]
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (t,e,p)):
            raise ValueError('non-positive/non-finite PPA')
        if not math.isclose(t*p,e,rel_tol=1e-9,abs_tol=1e-18):
            errors.append(label+': P*T != E (rtol 1e-9, atol 1e-18 J)')
        components=metrics.get('energy_components')
        if components and not math.isclose(sum(components.values()),e,rel_tol=1e-9,abs_tol=1e-18):
            errors.append(label+': energy component sum mismatch')
        if all(k in metrics for k in ('logical_macs','executed_macs','recompute_macs','pruned_macs')) and metrics['executed_macs'] != metrics['logical_macs'] + metrics['recompute_macs'] - metrics['pruned_macs']:
            errors.append(label+': MAC conservation mismatch')
        for pool in metrics.get('diagnostics',{}).get('memory',[]):
            if pool['peak_live_bytes'] > pool['capacity_bytes']:
                errors.append(label+': feasible SRAM overflow '+pool['pool_id'])
    except (KeyError, TypeError, ValueError) as error:
        errors.append(label+': invalid physical evidence '+str(error))

def analyze(out):
    out=Path(out); report=out/'reports'; report.mkdir(parents=True,exist_ok=True)
    errors=[]; warnings=[]; comparisons=[]; curves=[]; funnel=[]; policy=[]; effects=[]; failures=[]; pairs=[]; action_pairs=[]
    plan=read(out/'effective_config.json',{}); manifest=read(out/'manifest.json',{})
    try:
        verify_snapshot(out)
    except ValueError as error:
        errors.append(str(error))
    if plan and manifest.get('plan_hash') != sha(plan):
        errors.append('immutable effective configuration mismatch')
    references={}; seeds_by_group={}
    for case in plan.get('cases',[]):
        folder=out/'cases'/case['id']
        if case['kind']=='action_pairs':
            pair_data=read(folder/'summary.json',{})
            for pair in pair_data.get('pairs',[]):
                for label,m in pair.get('metrics',{}).items():physical_checks(m,case['id']+':'+label,errors)
                action_pairs.append(dict(case=case['id'],id=pair['id'],category=pair['category'],status=pair.get('status','partial'),
                    cost_delta=pair.get('cost_delta'),**(pair.get('delta') or {})))
            continue
        if case['kind']=='pair':
            pair=read(folder/'summary.json',{})
            pairs.append(dict(case=case['id'],status=pair.get('status','not_run'),full_ppa=pair.get('extra_full_ppa',0),
                **(pair.get('delta_on_minus_off') or {})))
            continue
        effective=read(folder/'effective_config.json',{})
        source=read(folder/'source_identity.json',{})
        seed=read(folder/'seed_attempts.json',{})
        if not seed:
            warnings.append(case['id']+': not started/no seed evidence'); continue
        seed_count=sum(bool(r.get('ppa_completed')) for r in seed.get('attempts',[]))
        if not seed.get('shared_from'):
            actual_seed=sum(r.get('full_evaluations',0) for r in ledger(folder/'seed_worker/accounting.json'))
            if actual_seed!=seed_count:
                errors.append(case['id']+': seed Full PPA ledger mismatch (may need resume recovery)')
        seed_bootstrap=seed.get('bootstrap',[])
        common_best=read(folder/'seed_only/joint_search.json',{}).get('summary',{}).get('best_cost')
        group=(case['model'],case['seed'])
        seed_hash=sha([(r['state'],{k:r['metrics'].get(k) for k in ('latency_s','energy_j','area_mm2','power_w')}) for r in seed_bootstrap])
        if group in seeds_by_group and seeds_by_group[group]!=seed_hash:
            errors.append(case['id']+': incomparable common seed archive')
        seeds_by_group[group]=seed_hash
        for r in seed_bootstrap:
            physical_checks(r['metrics'],case['id']+':seed',errors)
        for attempt in seed.get('attempts',[]):
            if attempt.get('status') in ('timeout','rejected'):
                failures.append(dict(case=case['id'],arm='seed',stage='seed',reason=attempt.get('reason'),count=1))
        for arm in case['arms'].split(','):
            arm_folder=folder/arm
            data=read(arm_folder/'joint_search.json') or read(arm_folder/'checkpoint.json',{})
            summary=data.get('summary',{})
            worker=ledger(arm_folder/'worker/accounting.json')
            count=sum(r.get('full_evaluations',0) for r in worker)
            comparable=True
            reference=data.get('reference',seed_bootstrap[0]['metrics'] if seed_bootstrap else {})
            ref_hash=sha({k:reference.get(k) for k in ('latency_s','area_mm2','power_w','energy_j')})
            signature=(source.get('identity'),ref_hash,seed_hash,effective.get('hardware_profile'))
            if group in references and references[group]!=signature:
                comparable=False; errors.append(case['id']+': incomparable source/reference/profile')
            references[group]=signature
            if summary and summary.get('search_full_ppa',0)!=count:
                errors.append(case['id']+'/'+arm+': search Full PPA ledger mismatch')
            if summary and summary.get('common_seed_full_ppa')!=seed_count:
                errors.append(case['id']+'/'+arm+': charged seed Full PPA mismatch')
            best=summary.get('best_cost',data.get('best_cost'))
            if arm=='seed_only' and best is not None:
                common_best=best
            if common_best is None and summary.get('improvement_over_best_seed_pct') is not None and best is not None:
                divisor=1-summary['improvement_over_best_seed_pct']/100
                common_best=best/divisor if divisor else None
            if common_best is None and seed_bootstrap:
                obj=effective.get('objective',{})
                weights=[obj.get(k,0) for k in ('latency_weight','area_weight','power_weight','energy_weight')]
                keys=('latency_s','area_mm2','power_w','energy_j'); total=sum(weights)
                if total:
                    common_best=min(sum(w/total*r['metrics'][k]/reference[k] for k,w in zip(keys,weights) if w) for r in seed_bootstrap)
            start_info=read(folder/'common_start.json',{})
            row=dict(initial_state_hash=start_info.get('initial_state_hash'),start_rank=start_info.get('start_rank',0),case=case['id'],model=case['model'],seed=case['seed'],space=case['space'],fusion=case['fusion'],arm=arm,
                generator=case['generator'],features=case['features'],comparable=comparable,seed_hash=seed_hash,reference_hash=ref_hash,
                common_seed_cost=common_best,best_cost=best,improvement_pct=(1-best/common_best)*100 if best is not None and common_best else None,
                charged_seed_full_ppa=seed_count,search_full_ppa=count,total_full_ppa=seed_count+count,
                timeout=sum(r.get('status')=='timeout' for r in worker),cache_hits=sum(r.get('cache_hits',0) for r in worker),
                policy_updates=data.get('policy_updates',data.get('policy',{}).get('updates',0)),
                charged_wall_s=summary.get('wall_s_including_common_seed_charge',seed.get('elapsed_s',0)+data.get('elapsed_s',0)),
                best_depth=max([len(st['layer_ids']) for st in (data.get('best_state') or {}).get('mapping',{}).get('fusion_stacks',[])] or [0]),
                max_evaluated_depth=max([max(r.get('fusion_depths') or [0]) for r in worker if r.get('full_evaluations')] or [0]),
                status=summary.get('status',data.get('status','not_started')))
            row.update({k:(data.get('best_metrics') or {}).get(k,(summary.get('best') or {}).get(k)) for k in (
                'latency_s','energy_j','area_mm2','power_w','compute_chiplets','io_chiplets','dram_read_bytes','dram_write_bytes',
                'noc_hop_bytes','nop_hop_bytes','executed_macs','recompute_macs','l1_l2_service_bytes')})
            comparisons.append(row)
            physical_checks(data.get('best_metrics'),case['id']+'/'+arm+':best',errors)
            if best is not None and arm!='seed_only' and not data.get('best_state'):
                errors.append(case['id']+'/'+arm+': best has no replayable state')
            events=jsonl(arm_folder/'evaluation_events.jsonl',warnings)
            running=common_best
            curves.append(dict(case=case['id'],arm=arm,completed_search_full_ppa=0,charged_full_ppa=seed_count,
                search_wall_s=0,charged_wall_s=seed.get('elapsed_s',0),best_cost=running))
            for event in events:
                if event.get('cost') is not None:
                    running=min(running,event['cost']) if running is not None else event['cost']
                curves.append(dict(case=case['id'],arm=arm,completed_search_full_ppa=event.get('full_ppa_completed'),
                    charged_full_ppa=seed_count+event.get('full_ppa_completed',0),search_wall_s=event.get('search_wall_s'),
                    charged_wall_s=seed.get('elapsed_s',0)+event.get('search_wall_s',0),best_cost=running))
            # Includes all evaluated Greedy lookahead states, not just committed steps.
            if count and not events:
                warnings.append(case['id']+'/'+arm+': per-query curves unavailable (legacy/partial log)')
            family=Counter(); shortlisted=Counter(); legal_counts=Counter(); reject_counts=Counter(); last_global=common_best
            for step in data.get('trace',[]):
                label=case['id']+'/'+arm+':'+str(step.get('step'))
                action=step.get('action_plan',{}); f=action.get('family_id','unknown')
                family[f]+=1
                if not step.get('valid') and step.get('policy_updated'):
                    errors.append(label+': invalid/timeout updated policy')
                if step.get('reward') is None and step.get('advantage') is not None:
                    errors.append(label+': unknown PPA has training advantage')
                observation=step.get('observation',{})
                if observation and (len(observation.get('state_features',[]))!=24 or len(observation.get('selected_candidate_features',[]))!=283):
                    errors.append(label+': feature schema dimensions differ')
                probabilities=step.get('candidate_probabilities',[])
                if probabilities and (any(p<0 or not math.isfinite(p) for p in probabilities) or not math.isclose(sum(probabilities),1,abs_tol=1e-9)):
                    errors.append(label+': invalid probabilities')
                if case['space']!='joint' and any(e.get('action_id')=='A9' for e in action.get('ordered_edits',[])):
                    errors.append(label+': architecture edit crossed search-space mask')
                physical_checks(step.get('current_metrics'),label,errors)
                before=step.get('before_metrics',{}); after=step.get('current_metrics',{})
                if step.get('valid') and before and reference:
                    weights=effective.get('objective',{})
                    names=('latency','area','power','energy'); keys=('latency_s','area_mm2','power_w','energy_j')
                    total=sum(weights.get(n+'_weight',0) for n in names)
                    if total and all(before.get(k) and after.get(k) and reference.get(k) for k in keys):
                        expected=sum(weights.get(n+'_weight',0)/total*(before[k]-after[k])/reference[k] for n,k in zip(names,keys))
                        if not math.isclose(expected,step.get('reward',0),rel_tol=1e-8,abs_tol=1e-10):
                            errors.append(label+': physical reward mismatch')
                policy.append(dict(case=case['id'],arm=arm,step=step.get('step'),family=f,
                    probability=step.get('behavior_probability'),entropy=step.get('policy_normalized_entropy'),
                    policy_updated=step.get('policy_updated'),reward=step.get('reward'),advantage=step.get('advantage')))
                effect=dict(case=case['id'],arm=arm,step=step.get('step'),family=f,targets=';'.join(action.get('target_ids',[])),
                    valid=step.get('valid'),evidence_quality=step.get('observation',{}).get('evidence_quality','unknown'))
                for key in ('latency_s','energy_j','dram_read_bytes','dram_write_bytes','noc_hop_bytes','nop_hop_bytes','executed_macs','recompute_macs','l1_l2_service_bytes'):
                    effect['delta_'+key]=after[key]-before[key] if step.get('valid') and key in after and key in before else None
                targets=set(action.get('target_ids',[]))
                for label,metrics in (('before',before),('after',after)):
                    layers=[r for r in metrics.get('diagnostics',{}).get('layers',[]) if r.get('layer_id') in targets]
                    for key in ('compute_busy_s','l1_service_s','vector_busy_s','max_job_queue_wait_s','max_transfer_queue_wait_s','criticality'):
                        effect[label+'_'+key]=max((r.get(key,0) or 0 for r in layers),default=None)
                    effect[label+'_max_pool_pressure']=max((r.get('peak_live_bytes',0)/max(r.get('capacity_bytes',1),1) for r in metrics.get('diagnostics',{}).get('memory',[])),default=None)
                effect['attribution']='target layers + shared pool max; partial, not unique hardware cause'
                effects.append(effect)
                if not step.get('valid'):
                    failures.append(dict(case=case['id'],arm=arm,stage='selected',reason=step.get('rejection_reason'),count=1))
                shortlist=step.get('candidate_shortlist') or {}
                file=Path(shortlist.get('path',''))
                if shortlist and not file.is_file():
                    file=arm_folder/'shortlists'/Path(shortlist['path']).name
                if shortlist and file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()!=shortlist.get('sha256'):
                    errors.append(label+': shortlist hash mismatch')
                if shortlist and file.is_file():
                    try:
                        recovered_shortlist=json.loads(gzip.decompress(file.read_bytes()))
                        shortlisted.update(p['family_id'] for p in recovered_shortlist['plans'])
                    except (OSError,ValueError,KeyError) as error:
                        warnings.append(label+': shortlist parse failed '+str(error))
                legal_counts.update(step.get('proposal_statistics',{}).get('structurally_legal_by_family',{}))
                reject_counts.update(step.get('proposal_statistics',{}).get('structurally_rejected_by_family',{}))
                for reason,n in step.get('proposal_statistics',{}).get('space_rejected',{}).items():
                    failures.append(dict(case=case['id'],arm=arm,stage='space_filter',reason=reason,count=n))
            queried={}
            for event in jsonl(arm_folder/'query_events.jsonl',warnings):
                f=event.get('action_plan',{}).get('family_id','unknown')
                stage=queried.setdefault(f,Counter())
                stage['oracle_attempted']+=event.get('status')!='cache_hit'
                stage['cache_hits']+=event.get('status')=='cache_hit'
                stage['completed_full_ppa']+=event.get('new_full_ppa',0)
                stage['completed_feasible']+=event.get('new_full_ppa',0)*int(event.get('feasible',False))
                if event.get('cost') is not None and event.get('new_full_ppa'):
                    stage['exceeds_seed']+=event['cost']<event['common_seed_cost']-1e-12
                    if event.get('best_cost_before') is not None:
                        stage['exceeds_global_best']+=event['cost']<event['best_cost_before']-1e-12
            for f,stats in data.get('family_statistics',{}).items():
                funnel.append(dict(case=case['id'],arm=arm,family=f,**stats,**dict(queried.get(f,{})),
                    shortlisted=shortlisted.get(f,0),structurally_legal=legal_counts.get(f),structurally_rejected_native=reject_counts.get(f),
                    note='native proposals/commits; query events include every Greedy lookahead; missing fields unknown'))
            for record in worker:
                if record.get('status') not in ('complete',):
                    failures.append(dict(case=case['id'],arm=arm,stage='oracle',reason=record.get('error',record.get('status')),count=1))
    tables={
        'compare_methods':(comparisons,list(comparisons[0]) if comparisons else ['case','arm','best_cost']),
        'best_vs_full_ppa':(curves,['case','arm','completed_search_full_ppa','charged_full_ppa','search_wall_s','charged_wall_s','best_cost']),
        'best_vs_wall':(curves,['case','arm','search_wall_s','charged_wall_s','completed_search_full_ppa','best_cost']),
        'action_funnel':(funnel,['case','arm','family','proposed','structurally_legal','structurally_rejected_native','shortlisted','selected','valid','improved','timeouts','oracle_attempted','cache_hits','completed_full_ppa','completed_feasible','exceeds_seed','exceeds_global_best','evaluation_elapsed_s','note']),
        'policy_learning':(policy,['case','arm','step','family','probability','entropy','policy_updated','reward','advantage']),
        'bottleneck_effects':(effects,list(effects[0]) if effects else ['case','arm','step','evidence_quality']),
        'failure_reasons':(failures,['case','arm','stage','reason','count']),
        'chosen_action_pairs':(action_pairs,list(action_pairs[0]) if action_pairs else ['case','id','category','status','cost_delta']),
        'fusion_pairs':(pairs,list(pairs[0]) if pairs else ['case','status','full_ppa']),
    }
    for name,(rows,fields) in tables.items():
        save_csv(report/(name+'.csv'),rows,fields)
    findings=[]
    no_seed=any(r['common_seed_cost'] is None for r in comparisons)
    if no_seed:
        findings.append('沒有完成且可行的共同 Seed；搜尋尚未開始。先區分 seed timeout、compiler reject 與 SRAM pool overflow，不能推論 RL 失敗或不存在可行設計。')
    for row in comparisons:
        label=row['case']+'/'+row['arm']
        if row['improvement_pct'] is not None and row['improvement_pct']<=1e-9 and row['arm']!='seed_only':
            findings.append(label+': 未超越共同 Seed；檢查同 state shortlist 的可達改善，再分離候選與 policy。')
        if row['timeout']>row['search_full_ppa']:
            findings.append(label+': timeout 多於完成 PPA；先測單一既有狀態的 oracle 耗時，不能判為方案較差。')
    if policy and all((p['entropy'] or 0)>.99 for p in policy if p['arm']=='bottleneck_rl'):
        findings.append('RL entropy 接近均勻；固定 shortlist replay、比較特徵差與多 RNG，不能僅憑更新數判定有效。')
    for group in {(r['model'],r['seed'],r['fusion'],r['generator'],r['features']) for r in comparisons}:
        members=[r for r in comparisons if (r['model'],r['seed'],r['fusion'],r['generator'],r['features'])==group]
        best_by_arm={r['arm']:r for r in members if r['best_cost'] is not None}
        rl=best_by_arm.get('bottleneck_rl');greedy=best_by_arm.get('greedy');rand=best_by_arm.get('random')
        if rl and greedy and greedy['best_cost'] < rl['best_cost']-1e-9:
            findings.append(str(group)+': Greedy 優於 RL；證據支持候選可能含改善而 policy 未取得同品質。先固定 shortlist replay；若實際查詢不同，仍不能歸因為 policy 劣勢。')
        if rl and rand and math.isclose(rl['best_cost'],rand['best_cost'],rel_tol=1e-6,abs_tol=1e-9):
            findings.append(str(group)+': Random 與 RL 最佳 cost 相近；尚無學習優勢證據，下一步 P2 與多 RNG，不能把 composite 候選增加算成 policy 效益。')
        for r in members:
            if r['cache_hits']>r['search_full_ppa'] and r['arm']!='greedy':
                findings.append(r['case']+'/'+r['arm']+': cache hit 多於新完整 PPA；檢查反向動作、重啟與 unique state 覆蓋。')
    for pair in pairs:
        if pair.get('dram_read_bytes',0)>0 and pair.get('recompute_macs',0)>0:
            findings.append(pair['case']+': 固定 Mapping 的 Fusion 增加 DRAM 與重算；證據支持 tile/halo/cache 路徑成本，下一步逐 stack/tensor 分解，尚不能斷言硬體模型錯誤。')
    replayed=read(report/'fixed_candidate_replay.json',{})
    if replayed.get('steps') and all(r['max_probability_delta']<1e-8 for r in replayed['steps']):
        findings.append('固定候選 replay 的機率差小；需分開查看零更新 actor 與已訓練 actor，再檢查 feature 區辨力／飽和。')
    if errors:
        findings.append('來源／帳本／物理一致性存在審計錯誤；在修復前不可合併勝負結論。')
    lines=['# 本機 RL 診斷（文字與數字）','',f"套件狀態：{read(out/'suite_status.json',{}).get('status','unknown')}。",
        '', '|Case|Arm|Seed cost|Best cost|改善 %|Seed PPA|Search PPA|Timeout|更新|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append('|'+ '|'.join(str(r.get(k)) for k in ('case','arm','common_seed_cost','best_cost','improvement_pct','charged_seed_full_ppa','search_full_ppa','timeout','policy_updates'))+'|')
    lines += ['', 'Seed 的改善不算 RL 收益；不同實際查詢數僅為相同上限下的完成結果。Atomic 與 Composite 動作空間不同。',
        '同模型來源／起點／參考不一致時不可合併勝負；無完成 PPA 的候選不算較差。',
        '分析式 PPA 未經 silicon／SRAM macro 校準；whole-flow reservation 等待歸因為 partial。',
        'CSV 中空白代表未知；曲線包含所有已保存的 Greedy lookahead 完成查詢。',
        f"實際 unique 已完成 Full PPA：{status(out)['unique_completed_full_ppa']}；共同 Seed 按方法計費但不重複宣稱實際 CPU 工作量。"]
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    diagnosis=['# 失敗歸因與下一個測試','', '可確定的事實：數字與帳本見 summary.md／CSV；未完成實驗保留 partial 狀態。','',
        '證據支持的候選原因（不是根因定論）：',*['- '+s for s in findings],
        '' ,('下一個最小鑑別測試：先讀 seed_attempts／compile_evidence 的容量與 timeout 原因，再由使用者固定一個候選檢查逐 pool peak/lifetime；沒有可行 state 時不啟動 P2 或 replay。' if no_seed else '下一個最小鑑別測試：固定同一 state／shortlist，執行 replay（0 PPA）；若有可達改善仍未選中，再由使用者啟動 P2。'),
        'P3：一組固定 Mapping Fusion On/Off，另選最多 5 個 chosen actions 於各自 before state 做雙 oracle 配對（最多 10 PPA）；缺少類型／預算截斷如實保存。',
        '已核對位置：hybrid/rl_search.py（policy/trace）、rl_actions.py（候選）、search_space.py（交易限制）、single_array_proposals.py（聯合提案）。',
        '不能推論：全域最佳、普遍 RL 優勢、精確硬體瓶頸因果或晶片實測。','', '審計錯誤：',*['- '+s for s in errors], '', '資料缺口：',*['- '+s for s in warnings]]
    (report/'diagnostics.md').write_text('\n'.join(diagnosis)+'\n',encoding='utf-8')
    result=dict(schema_version=1,errors=errors,warnings=warnings,arms=len(comparisons),physical_tolerance='rtol=1e-9; atol=1e-18 J',
        status='failed' if errors else 'checked_available_evidence',coverage='partial evidence is not full experimental acceptance')
    write(report/'audit.json',result)
    return result

def replay(out):
    """One-step fixed candidate set intervention; no oracle/model construction."""
    out=Path(out); verify_snapshot(out)
    snapshot=out/'source_snapshot'; sys.path.insert(0,str(snapshot))
    from simple_rapidchiplet.hybrid.rl_search import candidate_features, state_features, JointObjective
    from simple_rapidchiplet.hybrid.rl_schema import DesignState, ActionPlan
    from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
    from simple_rapidchiplet.hybrid.search import LinearActorCritic
    from simple_rapidchiplet.hybrid.workload import Workload, TensorSpec, LayerSpec
    import numpy as np
    candidates={c.id:c for c in enumerate_single_array_architectures_v3()}; rows=[]
    for file in sorted((out/'cases').rglob('shortlists/*.json.gz')):
        case=file.parents[2]; config=read(case/'effective_config.json',{})
        if not config:
            continue
        raw=json.loads(gzip.decompress(file.read_bytes()))
        if 'policy' not in raw:
            continue
        w=read(case/'workload.json'); tuple_keys={'inputs','kernel','stride','padding','dilation','weight_shape','post_ops'}
        workload=Workload(w['name'],tuple(TensorSpec(**{**t,'shape':tuple(t['shape'])}) for t in w['tensors']),
            tuple(LayerSpec(**{k:tuple(v) if k in tuple_keys else v for k,v in l.items()}) for l in w['layers']),
            tuple(w['input_ids']),tuple(w['output_ids']),w['batch_size'],w['source'])
        state=DesignState.from_dict(raw['before_state']); plans=[ActionPlan.from_dict(p) for p in raw['plans']]
        objective=JointObjective(**config['objective']); policy=LinearActorCritic.from_dict(raw['policy'])
        sf=state_features(state,raw['before_metrics'],raw['reference'],objective,candidates[state.architecture_candidate_id],
            max(len(c.package.mapping_core_ids) for c in candidates.values()) if not config.get('fixed_architecture') else len(candidates[state.architecture_candidate_id].package.mapping_core_ids), raw['episode_progress'])
        on=policy.probabilities(candidate_features(state,plans,sf,candidates,workload,raw['before_metrics']))
        sf[10:13]=0; sf[16:]=0
        off=policy.probabilities(candidate_features(state,plans,sf,candidates,workload,{}))
        rows.append(dict(shortlist=str(file.relative_to(out)),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            candidate_count=len(plans),policy_updates=raw['policy'].get('updates'),
            max_probability_delta=float(np.max(np.abs(on-off))),top_on=plans[int(np.argmax(on))].id,
            top_off=plans[int(np.argmax(off))].id,probabilities_on=on.tolist(),probabilities_off=off.tolist(),
            entropy_on=float(-np.dot(on,np.log(np.maximum(on,1e-300)))/max(math.log(len(on)),1e-30)),
            entropy_off=float(-np.dot(off,np.log(np.maximum(off,1e-300)))/max(math.log(len(off)),1e-30))))
    result=dict(oracle_calls=0,scope='same pre-update actor/state/candidate set; feature mask only, no retraining',steps=rows)
    write(out/'reports/fixed_candidate_replay.json',result)
    return dict(oracle_calls=0,steps=len(rows),report=str(out/'reports/fixed_candidate_replay.json'))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(); print(json.dumps(analyze(args.out),indent=2))
