"""Export immutable completed runs; no new PPA or policy update."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    base = workspace / 'output/inmin_v05_20261103'
    target = Path(__file__).resolve().parent
    cases = [(base / 'night_edp/resnet18' / str(seed),str(seed)) for seed in (7,17)]
    cases += [(base / 'fixed_mapping_fusion/resnet18/7','fixed_c2/7'),
              (base / 'crossmodel_smoke/resnet50/7','crossmodel_smoke/resnet50/7'),
              (base / 'crossmodel_smoke/mobilenet_v2/7','crossmodel_smoke/mobilenet_v2/7')]
    for case,relative in cases:
        if read(case / 'status.json')['status'] != 'completed' or not read(case / 'analysis/audit.json')['passed']:
            raise ValueError('refusing incomplete or unaudited source case')
        destination = target / relative
        destination.mkdir(exist_ok=True,parents=True)
        names = ['status.json','summary.json', 'common.json', 'shared_start.json', 'effective_config.json', 'environment.json',
                 'seed_input_manifest.json', 'source_snapshot_manifest.json', 'design_space.json',
                 'analysis/audit.json', 'analysis/equal_budget.json', 'analysis/curves.csv',
                 'analysis/held_out_policy.json', 'analysis/held_out_policy_strata.json']
        for method in read(case / 'summary.json'):
            names += [f'{method}/recommendation.json', f'{method}/summary.json']
            for name in ('learning.jsonl', 'shared_warmstart.json'):
                if (case / method / name).exists():
                    names.append(f'{method}/{name}')
            rec = read(case / method / 'recommendation.json')
            metrics = {k:rec['metrics'][k] for k in ('latency_s','energy_j','area_mm2','power_w','dram_read_bytes',
                'dram_write_bytes','noc_hop_bytes','nop_hop_bytes','executed_macs','recompute_macs','unit_count')}
            compact = dict(state=rec['state'],metrics=metrics,cost=rec['cost'],strict_feasible=rec['strict_feasible'],
                source_identity=read(case / 'source_snapshot_manifest.json')['identity'],
                scope='one complete physical design; compact review view, original recommendation retains full diagnostics')
            review = destination / method / 'mapping_review.json'
            review.parent.mkdir(parents=True,exist_ok=True)
            review.write_text(json.dumps(compact,indent=2),encoding='utf-8')
            with (case / method / 'events.jsonl').open('rb') as src, (destination / (method + '_events.jsonl.gz')).open('wb') as dst:
                # Fixed timestamp for a reproducible gzip byte representation.
                with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as compressed:
                    shutil.copyfileobj(src, compressed)
        for name in names:
            if not (case / name).is_file():
                continue
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(case / name, path)
    derived = workspace / 'output/inmin_v06_20261104/completed_v05_evidence'
    for name in ('REPORT.md', 'summary.json', 'results.csv','inactive_fd.json'):
        shutil.copyfile(derived / name, target / name)
    for folder,relative in ((base / 'fixed_inner_controls','inner_only_controls'),
            (workspace / 'output/inmin_v06_20261104/fixed_random_best_fusion_pair','fusion_pair')):
        for name in ('REPORT.md','result.json'):
            destination = target / relative / name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(folder / name,destination)
    manifest = {str(p.relative_to(target)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(target.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    (target / 'manifest.json').write_text(json.dumps(dict(files=manifest, new_ppa=0, policy_updates=0,
        scope='Completed/audited original source336 runs, including fixed-hardware negative controls and cross-model execution checks; independent Fusion-Off diagnostic has its own source/evaluation ledger. Prediction-only original neural weights are exported separately; full optimizer/replay continuation checkpoints remain local.'), indent=2), encoding='utf-8')
    print(json.dumps(dict(files=len(manifest), bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()), new_ppa=0)))


if __name__ == '__main__':
    main()
