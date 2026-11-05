"""Text/numeric development snapshot from audited physical results; no search."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def curve(case,method='rl'):
    shared=read(case/'shared_start.json')
    events=[json.loads(s) for s in (case/method/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    return [dict(q=shared['full_ppa'],seconds=shared['wall_s'],cost=shared['cost'])]+[
        dict(q=e['charged_full_ppa'],seconds=shared['wall_s']+e['search_wall_s'],cost=e['best_cost']) for e in events]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'output/inmin_v14_20261105/readiness_study_v1')
    args=parser.parse_args();study=args.study.resolve()
    case=study/'development/residual/7';old=ROOT/'output/inmin_v08_20261104/study_v3/development/stratified/7'
    if read(case/'status.json')['status']!='completed' or not read(case/'analysis/audit.json')['passed']:
        raise ValueError('Completed audited development case required')
    earlier,current=curve(old),curve(case);matched={}
    for axis in ('q','seconds'):
        floor=min(max(r[axis] for r in earlier),max(r[axis] for r in current))
        before=min(r['cost'] for r in earlier if r[axis]<=floor);after=min(r['cost'] for r in current if r[axis]<=floor)
        matched[axis]=dict(budget=floor,old_RL=before,new_RL=after,improvement_pct=100*(1-after/before))
    rows=[]
    for label,path,method in (('old_RL',old,'rl'),('new_RL',case,'rl'),
        ('old_Random_reference',ROOT/'output/inmin_v08_20261104/study_v3/development/controls/7','random')):
        rec=read(path/method/'recommendation.json');metrics=rec['metrics'];summary=read(path/'summary.json')[method]
        rows.append(dict(method=label,EDP=rec['cost'],new_PPA=summary['search_full_ppa'],charged_PPA=summary['full_ppa'],
            search_seconds=summary['search_wall_s'],architecture=metrics['architecture_candidate_id'],
            latency_ms=metrics['latency_s']*1000,energy_mJ=metrics['energy_j']*1000,area_mm2=metrics['area_mm2'],power_W=metrics['power_w']))
    learning_path=study.parent/'development_learning_seed7_context_v2.json'
    if not learning_path.exists():learning_path=study.parent/'development_learning_seed7.json'
    learning=read(learning_path)
    pending={name:read(study/f'development/{name}/7/status.json').get('status') if (study/f'development/{name}/7/status.json').exists() else 'not_started'
        for name in ('no_residual','myopic')}
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),source_identity=read(case/'source_snapshot_manifest.json')['identity'],
        random_seed=7,scope='one adaptive development RNG; old-controller references; fresh confirmation and causal controls pending',
        rows=rows,matched=matched,controls=pending,learning_diagnostic=learning,new_PPA_by_report=0,new_SGD_by_report=0,images=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (study/'DEVELOPMENT_SNAPSHOT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# 修正後RL：已完成的開發結果','',
        '2026-11-05；一個開發RNG7，Batch1／224×224 ResNet18，EDP。新來源b771、218檔，完整V3有效空間、同一物理模型與已支付100次PPA／2230.010633秒的共同起點。舊控制器資料是開發參考；新確認仍在等待控制完成。','',
        '|方法|新／累計PPA|EDP|搜尋秒|架構|T ms|E mJ|Area mm²|平均W|','|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['method']}|{r['new_PPA']}／{r['charged_PPA']}|{r['EDP']:.9f}|{r['search_seconds']:.3f}|{r['architecture']}|{r['latency_ms']:.6f}|{r['energy_mJ']:.6f}|{r['area_mm2']:.3f}|{r['power_W']:.6f}|")
    lines += ['', '|匹配方式|含共同階段的預算floor|舊RL EDP|新RL EDP|改善%|','|---|---:|---:|---:|---:|']
    for axis,row in matched.items():lines.append(f"|{axis}|{row['budget']:.3f}|{row['old_RL']:.9f}|{row['new_RL']:.9f}|{row['improvement_pct']:.4f}|")
    warm,final=learning['models']['after_shared'],learning['models']['final']
    lines += ['',
        '新RL相對共同強起點降低26.4827% EDP。這個種子仍未超過原Random最佳值；不能因為勝過舊RL，就宣稱勝過Random、SA或原論文。舊RL的12顆2×6布局有4個IO，新RL的12顆3×4布局有6個IO；面積和平均功率上升，延遲與能量下降，屬於完整架構／mapping取捨。','',
        f"未參與該RL訓練的109個舊控制設計上，成本排序Spearman為{warm['after_cost_spearman']:.6f}→{final['after_cost_spearman']:.6f}，potential MAE為{warm['after_potential_MAE']:.6f}→{final['after_potential_MAE']:.6f}；同架構／分組pair concordance為{warm['same_architecture_and_group']['concordance']:.6f}→{final['same_architecture_and_group']['concordance']:.6f}。最佳控制方案的預測rank反而由22變26，仍保留。這是開發診斷，不是新獨立泛化確認。",'',
        '最終H在該corpus的標準差1.673，對48組已存nonterminal候選中的36組改變argmax，表示多步分支確實參與排序。這不證明排序更有用；必須等待gamma0、無殘差、凍結policy及新RNG搜尋對照。R的最終選擇為after basis／ridge1，僅以自己的真實PPA作訓練內LOO選擇。','',
        f"目前控制：{json.dumps(pending,ensure_ascii=False)}。原協定須至少5%同query／同time門檻才能進入fresh197／207／217確認；當前主臂資料通過該數值門檻，但仍依協定先完成開發消融。",'',
        '有一個可行global_mapping完整評估耗時481.407秒，成本0.257494，真實負reward−17.653；23,552units，沒有Fusion。來源與負面結果完整保留。另有精確通訊bitset元件原型通過163組合成資料逐序相等檢查，但尚未採用，不能宣稱完整PPA或DSE加速。','',
        '所有PPA為未經晶片校準的解析資源calendar模型；本報告新增0次PPA／SGD，不產生圖片。','']
    if 'same_before_state' in final:
        lines += ['|固定條件下的成本排序|比較群數|非同成本pair數|共同訓練後一致率|搜尋訓練後一致率|',
                  '|---|---:|---:|---:|---:|']
        for key,label in (('same_before_state','相同起點／起點成本'),
                          ('same_before_and_after_architecture_group','相同起點＋結果架構／分組')):
            w,f=warm[key],final[key]
            lines.append(f"|{label}|{f['groups']}|{f['non_tied_pairs']}|{w['concordance']:.6f}|{f['concordance']:.6f}|")
        lines += ['', '這些比較再固定before-state，避免已知起點成本造成排序相關性的假象。它們是控制軌跡的回顧子集合，並非原始完整候選池的反事實測試；pair也不是獨立PPA查詢或獨立隨機種子。僅對V0＋R的實際成本排序評分，未把H的多步值當成已知的真實未來回報。','']
    (ROOT/'docs/INMIN_RESIDUAL_RL_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(new_EDP=rows[1]['EDP'],matched=matched,controls=pending,new_PPA=0,new_SGD=0)))


if __name__=='__main__':main()
