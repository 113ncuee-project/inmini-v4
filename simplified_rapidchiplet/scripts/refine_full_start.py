"""Core CLI startup refinement, using the same paid wider-DP stage as research.

No prior night-study output is needed. The current case's frozen common/input
archive is captured once, so restart cannot append the shared observations
twice or change the architecture shortlist mid-refinement.
"""
from pathlib import Path
import json
import shutil
import time
from types import SimpleNamespace

from run_local_rl_diagnostics import read, write


def _refine_start_impl(out, args, prior_elapsed=0., call_start=None):
    out = Path(out)
    saved = read(out / 'seed_refinement.json')
    if saved and saved['status'] in ('completed', 'skipped_budget'):
        return
    workspace = out / 'seed_refinement'
    backup = workspace / 'original_shared'
    original = backup if (backup / 'captured.json').exists() else out
    common, screen, shared = (read(original / name) for name in ('common.json', 'screening.json', 'shared_start.json'))
    reserve = max(4, args.queries // 2)
    budget = min(args.seed_queries, max(0, args.queries - shared['full_ppa'] - reserve))
    count = len({common['initial_state']['architecture_candidate_id'], shared['state']['architecture_candidate_id']})
    if budget < count:
        write(out / 'seed_refinement.json', dict(status='skipped_budget', full_ppa=0,
            reason='keep at least half the query ceiling (or 4 queries) available for real RL/search; raise queries for wider initialization'))
        return
    started = call_start if call_start is not None else time.perf_counter()
    if not (backup / 'captured.json').exists():
        backup.mkdir(parents=True, exist_ok=True)
        for name in ('common.json', 'screening.json', 'shared_start.json', 'source_snapshot_manifest.json',
                     'seed_input_manifest.json', 'effective_config.json', 'environment.json'):
            shutil.copyfile(out / name, backup / name)
        if (out/'objective.json').exists():
            shutil.copyfile(out/'objective.json',backup/'objective.json')
        for name in ('source_snapshot', 'seed_input_snapshot'):
            shutil.copytree(out / name, backup / name, dirs_exist_ok=True)
        write(backup / 'captured.json', dict(scope='immutable before wider-start refinement; copied inputs/source/common observations, zero new PPA'))
    baseline = workspace / 'wide_dp'
    from run_strong_seed_comparison import run
    if not read(baseline / 'summary.json'):
        run(SimpleNamespace(donor=str(backup.resolve()), out=str(baseline.resolve()), queries=budget // count,
            dp_seconds=args.seed_dp_seconds, seconds=args.seed_seconds / count, eval_timeout=args.eval_timeout,
            objective=args.objective, max_area=args.max_area, max_power=args.max_power, max_latency_ms=args.max_latency_ms,
            stop_file=str(out / 'STOP_REQUESTED')))
    from build_strong_shared_start import build
    combined = workspace / 'combined'
    if not read(combined / 'summary.json'):
        build(backup, baseline, combined, seeds=(args.seed,), model_id='input_model')
    merged = read(combined / f'input_model/{args.seed}/screening.json')
    total_elapsed = prior_elapsed + time.perf_counter() - started
    merged['wall_s'] = screen['wall_s'] + total_elapsed
    merged['refinement_time_scope'] = 'complete local refinement calls, including cooperative resumes; outer CLI startup separately logged'
    write(out / 'screening.json', merged)
    result = read(combined / 'summary.json')
    write(out / 'seed_refinement.json', dict(status='completed', full_ppa=result['extra_full_ppa'],
        source='same frozen compiler/oracle; wider DP + declared inner mapping proposals',
        new_full_query_budget=budget, reserved_search_queries=reserve, wall_s=total_elapsed,
        charged_total_shared_ppa=result['charged_full_ppa'], charged_shared_wall_s=common['wall_s'] + merged['wall_s'],
        baseline=str(baseline), no_policy_or_search_arm_results_imported=True))


def refine_start(out, args):
    out = Path(out)
    if read(out / 'seed_refinement.json', {}).get('status') in ('completed', 'skipped_budget'):
        return
    journal = out / 'seed_refinement_invocations.jsonl'
    previous = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()] if journal.exists() else []
    started, began, status = time.perf_counter(), time.time(), 'completed'
    try:
        _refine_start_impl(out, args, sum(r['wall_s'] for r in previous), started)
    except InterruptedError:
        status = 'interrupted'
        raise
    except Exception:
        status = 'failed'
        raise
    finally:
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(status=status, started_unix=began, finished_unix=time.time(), wall_s=time.perf_counter() - started)) + '\n')
