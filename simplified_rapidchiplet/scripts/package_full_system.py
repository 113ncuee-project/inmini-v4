"""Portable full-system code and curated numeric test/results packages."""
import argparse
from pathlib import Path
import json

from run_local_rl_diagnostics import APP, ROOT, sources, sha, write
from package_replay_v4 import archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'output/uploads')
    p.add_argument('--tag', default='20261103_full')
    p.add_argument('--evidence', type=Path, action='append', default=[])
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    source_files = sources()
    code = [(APP / rel, 'simplified_rapidchiplet/' + rel) for rel in source_files]
    for relative in ('README.md', '.gitattributes', 'run_inmin_full_overnight.ps1', 'run_inmin_full_overnight.cmd', 'start_inmin_full_gui.cmd',
        'run_inmin_v04_overnight.ps1', 'run_inmin_v04_overnight.cmd', 'docs/INMIN_FULL_SYSTEM_GUIDE.md',
        'docs/INMIN_FULL_SYSTEM_VALIDATION.md', 'docs/inmin_v05_progress.md', 'docs/INMIN_V03_IMPLEMENTATION.md',
        'docs/INMIN_V04_LOCAL_GUIDE.md', 'docs/INMIN_V04_RESULTS.md', 'docs/specs/Chiplet_Hardware_Spec_v0.2_reviewed.md'):
        path = ROOT / relative
        if path.exists():
            code.append((path, relative))
    for path in (ROOT / 'inputs/replay_v04').rglob('*.json'):
        code.append((path, path.relative_to(ROOT).as_posix()))
    result = dict(source_identity=sha(source_files), code=archive(args.out / f'INMIN_Full_Code_{args.tag}.zip', code))
    evidence = []
    selected_names = {'result.json', 'tests.log', 'summary.json', 'REPORT.md', 'status.json', 'effective_config.json',
        'parent_invocations.jsonl', 'source_snapshot_manifest.json', 'seed_input_manifest.json', 'common.json',
        'shared_start.json', 'screening.json', 'design_space.json', 'audit.json', 'equal_budget.json',
        'recommendation.json', 'events.jsonl', 'learning.jsonl', 'shared_warmstart.json', 'accounting.json'}
    for folder in args.evidence:
        folder = folder.resolve()
        if not folder.is_relative_to((ROOT / 'output').resolve()):
            raise ValueError('evidence must be in project output')
        for path in folder.rglob('*'):
            if not path.is_file() or 'source_snapshot' in path.parts or '__pycache__' in path.parts:
                continue
            if path.name in selected_names or path.suffix == '.csv':
                evidence.append((path, folder.name + '/' + path.relative_to(folder).as_posix()))
    if evidence:
        result['results'] = archive(args.out / f'INMIN_Full_Results_{args.tag}.zip', evidence)
    write(args.out / f'INMIN_Full_{args.tag}_package.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
