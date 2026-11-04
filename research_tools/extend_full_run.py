"""Continue a completed full-system run in a NEW folder with larger budgets.

This is a continuation of the same policies/RNGs and observations, not an
independent replicate. Only cumulative query/time ceilings change. Original
results stay byte-for-byte unchanged; old queries and time remain charged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'inputs').is_dir() and (p / 'simplified_rapidchiplet').is_dir())


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def files(folder):
    return {str(p.relative_to(folder)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and not p.name.endswith('.lock') and p.name!='STOP_REQUESTED'}


def prepare(source, out, extra_queries, extra_seconds):
    source, out = source.resolve(), out.resolve()
    allowed = (ROOT / 'output').resolve()
    if not source.is_relative_to(allowed) or not out.is_relative_to(allowed):
        raise ValueError('source and new output must stay under the real workspace output/')
    if source==out or out.is_relative_to(source) or source.is_relative_to(out) or out.exists():
        raise ValueError('choose a new sibling output, never a source ancestor/descendant')
    if read(source / 'status.json').get('status')!='completed':
        raise ValueError('only a completed run can be extended; resume an interrupted run normally first')
    config, summaries = read(source / 'effective_config.json'),read(source / 'summary.json')
    methods = config['methods'].split(',')
    if any(m not in summaries for m in methods):
        raise ValueError('one or more comparator arms are incomplete')
    started = time.perf_counter()
    before = files(source)
    checkpoints = {m:read(source / m / 'checkpoint.json') for m in methods}
    if any(checkpoints[m].get('pending_transition') for m in methods):
        raise ValueError('completed source has an uncommitted transition; inspect it first')
    shutil.copytree(source,out,ignore=shutil.ignore_patterns('__pycache__','*.lock','STOP_REQUESTED'))
    metadata = out / 'budget_extension'
    # An earlier phase's lineage is kept as a nested ancestor in each fork.
    metadata = metadata / ('phase_' + str(len(list(metadata.glob('phase_*')))+1))
    metadata.mkdir(parents=True)
    for name in ('effective_config.json','summary.json','status.json','REPORT.md'):
        if (source / name).exists():
            shutil.copyfile(source / name,metadata / name)
    for m in methods:
        shutil.copyfile(source / m / 'checkpoint.json',metadata / (m + '_checkpoint.json'))
        policy = checkpoints[m].get('policy_file')
        if policy:
            shutil.copyfile(source / m / policy,metadata / (m + '_inherited_policy.pt'))
    if files(source)!=before:
        raise ValueError('source changed during copy; preserve this failed fork and inspect the source')
    copied = files(out)
    mismatches = [n for n,v in before.items() if copied.get(n)!=v]
    if mismatches:
        raise ValueError('copy mismatch before continuation: ' + ', '.join(mismatches[:8]))
    admin = time.perf_counter()-started
    config.update(out=str(out),seed_case=str(out / 'seed_input_snapshot'),
                  queries=config['queries']+extra_queries,seconds=config['seconds']+extra_seconds)
    write(out / 'effective_config.json',config)
    write(out / 'summary.json',{})
    for m, old in checkpoints.items():
        checkpoint = dict(old)
        # A new phase starts at the best measured design. Keep the learned
        # online/target weights, replay, optimizer, action RNG and all ledgers.
        checkpoint.update(status='budget_extension_ready',current=old['elite'],current_metrics=old['elite_metrics'],
            elapsed_s=max(old['elapsed_s'],summaries[m]['search_wall_s'])+admin,
            epoch_steps=0,episode=old['episode']+1,candidate_set=[],pending_transition=None)
        write(out / m / 'checkpoint.json',checkpoint)
    write(out / 'status.json',dict(status='interrupted',phase='budget_extension_ready',return_code=130))
    lineage = dict(created_utc=datetime.now(timezone.utc).isoformat(),parent=str(source),output=str(out),
        source_identity=read(source / 'source_snapshot_manifest.json')['identity'],parent_file_hashes=before,
        original_query_ceiling=config['queries']-extra_queries,new_query_ceiling=config['queries'],
        original_time_ceiling=config['seconds']-extra_seconds,new_time_ceiling=config['seconds'],
        inherited_charged_full_ppa={m:summaries[m]['full_ppa'] for m in methods},
        inherited_search_s={m:summaries[m]['search_wall_s'] for m in methods},
        copy_and_source_verification_wall_s=admin,copy_time_charge='added identically at the phase boundary to each method elapsed clock',
        statistical_scope='same policies and RNG continuations; never count as independent RNGs',
        trajectory_scope='prior episode remains terminal; new phase begins at elite; new ceiling changes epsilon annealing',
        original_unchanged=True)
    write(metadata / 'lineage.json',lineage)
    return lineage,metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extra-queries',type=int,default=256)
    p.add_argument('--extra-seconds',type=float,default=3600)
    p.add_argument('--launcher',type=Path,default=ROOT / 'simplified_rapidchiplet/full_system_dse.py')
    p.add_argument('--prepare-only',action='store_true')
    args = p.parse_args()
    if args.extra_queries<0 or not math.isfinite(args.extra_seconds) or args.extra_seconds<0 or not args.extra_queries+args.extra_seconds>0:
        p.error('nonnegative ceiling increments, with at least one positive value, are required')
    lineage,metadata = prepare(args.source,args.out,args.extra_queries,args.extra_seconds)
    print(json.dumps(dict(status='prepared_continuation',out=lineage['output'],
        inherited_full_ppa=lineage['inherited_charged_full_ppa'],new_query_ceiling=lineage['new_query_ceiling'],
        new_time_ceiling=lineage['new_time_ceiling'],new_ppa=0)),flush=True)
    if args.prepare_only:
        return
    began = time.perf_counter()
    result = subprocess.run([sys.executable,'-u','-X','utf8',str(args.launcher.resolve()),'--out',lineage['output'],'--resume'],cwd=ROOT)
    unchanged = files(args.source.resolve())==lineage['parent_file_hashes']
    execution = dict(return_code=result.returncode,wall_s=time.perf_counter()-began,original_unchanged=unchanged)
    if result.returncode==0:
        summary = read(Path(lineage['output']) / 'summary.json')
        execution['additional_full_ppa'] = {m:r['full_ppa']-lineage['inherited_charged_full_ppa'][m] for m,r in summary.items()}
    write(metadata / 'execution.json',execution)
    print(json.dumps(execution),flush=True)
    if not unchanged:
        raise RuntimeError('source mutated outside this continuation; investigate before using the fork')
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
