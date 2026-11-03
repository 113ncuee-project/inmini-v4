"""Export immutable completed runs; no new PPA or policy update."""
import argparse
import csv
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
    source = args.workspace.resolve() / 'output/inmin_v05_20261103/night_edp/resnet18'
    target = Path(__file__).resolve().parent
    for seed in (7, 17):
        case = source / str(seed)
        if read(case / 'status.json')['status'] != 'completed' or not read(case / 'analysis/audit.json')['passed']:
            raise ValueError('refusing incomplete or unaudited source case')
        destination = target / str(seed)
        destination.mkdir(exist_ok=True)
        names = ['summary.json', 'common.json', 'shared_start.json', 'effective_config.json', 'environment.json',
                 'seed_input_manifest.json', 'source_snapshot_manifest.json', 'design_space.json',
                 'analysis/audit.json', 'analysis/equal_budget.json', 'analysis/curves.csv',
                 'analysis/held_out_policy.json', 'analysis/held_out_policy_strata.json']
        for method in ('rl','sa','random'):
            names += [f'{method}/recommendation.json', f'{method}/summary.json']
            for name in ('learning.jsonl', 'shared_warmstart.json'):
                if (case / method / name).exists():
                    names.append(f'{method}/{name}')
            with (case / method / 'events.jsonl').open('rb') as src, (destination / (method + '_events.jsonl.gz')).open('wb') as dst:
                # Fixed timestamp for a reproducible gzip byte representation.
                with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as compressed:
                    shutil.copyfileobj(src, compressed)
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(case / name, path)
    derived = args.workspace.resolve() / 'output/inmin_v06_20261104/completed_v05_evidence'
    for name in ('REPORT.md', 'summary.json', 'results.csv'):
        shutil.copyfile(derived / name, target / name)
    manifest = {str(p.relative_to(target)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(target.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    (target / 'manifest.json').write_text(json.dumps(dict(files=manifest, new_ppa=0, policy_updates=0,
        scope='Completed/audited original source336 runs; compressed events retain full mapping/behavior/reward evidence; no trained policy tensor checkpoint.'), indent=2), encoding='utf-8')
    print(json.dumps(dict(files=len(manifest), bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()), new_ppa=0)))


if __name__ == '__main__':
    main()
