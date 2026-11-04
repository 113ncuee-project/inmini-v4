"""Wider Tangram-style DP/beam baseline, with every score from full PPA.

Runs both whole-output-screened and unscreened proposals over all contiguous
segment lengths; also evaluates declared inner q8 refinements. This is an INMIN
adapter comparison, not an execution of original Tangram or Gemini binaries.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / 'scripts'))
from run_local_rl_diagnostics import read, write, sources, sha, lock
from profile_mapping_oracle import load_workload
from replay_mapping_dse import compact, append
from full_system_dse import objective_cost
from simple_rapidchiplet.hybrid.full_space import FullDesignSpace, DesignTrial
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.single_array_runtime import SingleArrayOraclePool
from simple_rapidchiplet.hybrid.single_array_seed_v3 import generate_seeds_v3
from simple_rapidchiplet.hybrid.shared_evidence import verify_physical_sources


def run(args):
    donor, out = Path(args.donor).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = sha(sources(APP))
    saved = read(out / 'protocol.json')
    protocol = dict(source_identity=identity, arguments=vars(args), source_scope='frozen enclosing graph study engine')
    if saved and saved != protocol:
        raise ValueError('strong baseline source/protocol changed')
    write(out / 'protocol.json', protocol)
    def stopped():
        path = getattr(args, 'stop_file', None)
        return bool(path and Path(path).exists())
    def progress(folder, label, extension, **status):
        write(folder / 'status.json', dict(status='seed_generation', variant=label, extension=extension, **status))
        if stopped():
            raise InterruptedError('cooperative stop after a durable DP prefix checkpoint')
    verify_physical_sources(read(donor / 'source_snapshot_manifest.json')['files'], sources(APP))
    common, shared = read(donor / 'common.json'), read(donor / 'shared_start.json')
    w = load_workload(donor / 'seed_input_snapshot/workload.json')
    initial = DesignState.from_dict(common['initial_state'])
    candidates = enumerate_single_array_architectures_v3()
    lookup = {c.id: c for c in candidates}
    space = FullDesignSpace(w, candidates, boundary_features=False, resource_projection=False)
    objective = args
    ids = tuple(dict.fromkeys((initial.architecture_candidate_id, shared['state']['architecture_candidate_id'])))
    results = []
    for architecture in ids:
        folder = out / architecture
        folder.mkdir(exist_ok=True)
        if read(folder / 'summary.json'):
            results.append(read(folder / 'summary.json'))
            continue
        c = lookup[architecture]
        base = space.architecture_trial(initial, c).state
        tick = time.perf_counter()
        proposals = read(folder / 'proposals.json')
        generation = []
        if not proposals:
            rows, seen = [], set()
            for screen in (True, False):
                label = 'whole_output_screen' if screen else 'no_whole_output_prune'
                last_prefix, extensions = -1, 0
                for extensions in range(3):
                    generated = generate_seeds_v3(w, c.hardware, max_candidates=4, beam_width=16,
                        max_group_layers=len(w.layers), timeout_s=args.dp_seconds, capacity_screen=screen,
                        checkpoint=folder / (label + '_dp_checkpoint.json'),
                        update=lambda **status: progress(folder, label, extensions, **status))
                    if stopped():
                        raise InterruptedError('cooperative stop during seed generation')
                    prefix = generated[0].provenance.get('prefix_completed', 0)
                    if generated[0].provenance.get('DP_complete') or prefix <= last_prefix:
                        break
                    last_prefix = prefix
                generation.append(dict(variant=label, extension_attempts=extensions,
                                       provenance=[p.provenance for p in generated]))
                for p in generated:
                    state = replace(base, mapping=p.plan)
                    for variant, trial in (('raw_dp', DesignTrial(state, 'tangram_wide_dp', (), p.to_dict())),
                        ('inner_q8', None)):
                        if trial is None:
                            try:
                                t = space.bank(state).build((8, 'balanced', 'spread', 'interleave', 'auto'), base=state)
                                trial = DesignTrial(t.state, 'tangram_wide_dp_inner', t.targets,
                                    dict(dp_provenance=p.provenance, declared_inner=(8, 'balanced', 'spread', 'interleave', 'auto')))
                            except ValueError:
                                continue
                        if trial.fingerprint in seen:
                            continue
                        seen.add(trial.fingerprint)
                        rows.append(dict(variant=variant, screen=screen, trial=trial.to_dict()))
            proposals = dict(rows=rows, generation=generation, generation_wall_s=time.perf_counter() - tick)
            # Additional announced inner directions on each distinct DP group
            # structure. No measured winner is used to choose these proposals.
            inner_states = [DesignTrial.from_dict(r['trial']).state for r in rows if r['variant'] == 'inner_q8']
            for state in inner_states:
                for params in ((8, 'P', 'spread', 'interleave', 'auto'), (8, 'K', 'spread', 'interleave', 'auto'),
                               (16, 'balanced', 'spread', 'interleave', 'auto'), (32, 'balanced', 'spread', 'interleave', 'auto')):
                    try:
                        t = space.bank(state).build(params, base=state)
                    except ValueError:
                        continue
                    trial = DesignTrial(t.state, 'tangram_wide_dp_inner', t.targets,
                        dict(declared_inner=params, group_provenance='widened DP; unchanged boundaries'))
                    if trial.fingerprint not in seen:
                        seen.add(trial.fingerprint)
                        rows.append(dict(variant='inner_direction_quota', screen=None, trial=trial.to_dict()))
            # Expensive raw full-quota plans remain visible, but cannot consume
            # the entire wall budget before the declared fitting refinements.
            rows.sort(key=lambda r: (r['variant'] == 'raw_dp', len(r['trial']['state']['mapping']['groups']), r['variant']))
            proposals['generation_wall_s'] = time.perf_counter() - tick
            write(folder / 'proposals.json', proposals)
        events_path = folder / 'events.jsonl'
        events = [json.loads(l) for l in events_path.read_text(encoding='utf-8').splitlines()] if events_path.exists() else []
        done = {r['state_hash'] for r in events}
        elapsed = sum(e.get('wall_s', 0) for e in events)
        pool = SingleArrayOraclePool(w, folder / 'worker', deadline=time.monotonic() + max(.01, args.seconds - elapsed),
                                    evaluation_timeout_s=args.eval_timeout)
        try:
            for row in proposals['rows']:
                if stopped():
                    raise InterruptedError('cooperative stop after the current bounded initialization evaluation')
                trial = DesignTrial.from_dict(row['trial'])
                if trial.fingerprint in done:
                    continue
                if pool.full_evaluations >= args.queries or time.monotonic() >= pool.deadline:
                    break
                tick = time.perf_counter()
                event = dict(state_hash=trial.fingerprint, variant=row['variant'], screen=row['screen'], trial=trial.to_dict())
                try:
                    m = pool.evaluate(c, trial.state.mapping, phase='wide_tangram_style_baseline')['metrics']
                    value = objective_cost(m, common['reference'], objective)
                    event.update(status='full_completed', cost=value, metrics=compact(m), feasible=m['feasible'])
                except (ValueError, RuntimeError, TimeoutError) as error:
                    event.update(status='timeout' if isinstance(error, TimeoutError) else 'rejected', cost=None, reason=str(error))
                event['wall_s'] = time.perf_counter() - tick
                append(events_path, event)
                events.append(event)
                done.add(trial.fingerprint)
                print(json.dumps(dict(architecture=architecture, tested=len(events), cost=event['cost'], status=event['status'])), flush=True)
        finally:
            pool.close()
        best = min((e for e in events if e.get('cost') is not None), key=lambda e: e['cost'], default=None)
        original = architecture == initial.architecture_candidate_id
        inherited = common['full_ppa'] if original else shared['full_ppa']
        inherited_time = common['wall_s'] if original else shared['wall_s']
        summary = dict(architecture=architecture, new_full_ppa=pool.full_evaluations,
            charged_full_ppa=inherited + pool.full_evaluations, inherited_full_ppa=inherited,
            generation_wall_s=proposals['generation_wall_s'], search_wall_s=sum(e['wall_s'] for e in events),
            charged_wall_s=inherited_time + proposals['generation_wall_s'] + sum(e['wall_s'] for e in events),
            best_cost=best['cost'] if best else None, best=best,
            source_identity=identity, proposal_count=len(proposals['rows']),
            complete_proposal_evaluation=len(done) == len(proposals['rows']),
            comparison_scope='widened INMIN Tangram-style adapter + explicit inner refinement; not original executable')
        write(folder / 'summary.json', summary)
        results.append(summary)
    stable = sha(sources(APP)) == identity
    if not stable:
        raise ValueError('strong baseline engine source changed during execution')
    write(out / 'summary.json', dict(source_stable=stable, architectures=results))
    lines = ['# Wider Tangram-style seed comparison', '',
        'All-segment DP/beam=16, screened and no-whole-output-prune variants, raw and declared q8 inner refinement. Complete INMIN oracle decides feasibility and EDP. This is an adapted baseline, not the original executable.', '',
        '|Architecture|New full PPA|Charged full PPA|Best normalized EDP|Generation s|Search s|', '|---|---:|---:|---:|---:|---:|']
    for r in results:
        value = f"{r['best_cost']:.8f}" if r['best_cost'] is not None else 'no feasible design'
        lines.append(f"|{r['architecture']}|{r['new_full_ppa']}|{r['charged_full_ppa']}|{value}|{r['generation_wall_s']:.3f}|{r['search_wall_s']:.3f}|")
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--queries', type=int, default=24)
    p.add_argument('--dp-seconds', type=float, default=600)
    p.add_argument('--seconds', type=float, default=1200)
    p.add_argument('--eval-timeout', type=float, default=240)
    p.add_argument('--objective', choices=('edp', 'balanced', 'latency', 'energy'), default='edp')
    p.add_argument('--max-area', type=float)
    p.add_argument('--max-power', type=float)
    p.add_argument('--max-latency-ms', type=float)
    p.add_argument('--stop-file')
    args = p.parse_args()
    workspace = next((x for x in APP.parents if (x / '.venv/Scripts/python.exe').is_file() and (x / 'inputs').is_dir()), APP.parent)
    with lock(workspace / 'output/.local_rl_diagnostics.lock'):
        run(args)
