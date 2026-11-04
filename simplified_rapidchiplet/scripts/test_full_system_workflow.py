"""Actual CLI stop/resume and full-domain screening integration checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_local_rl_diagnostics import APP, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-case', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cost-value-q', action='store_true')
    p.add_argument('--episode-cost-rise',type=float)
    p.add_argument('--architecture-refresh-interval',type=int,default=0)
    p.add_argument('--service-features',action='store_true')
    p.add_argument('--cost-rank-weight',type=float,default=0.)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError('new validation folder required')
    out.mkdir(parents=True)
    script = APP / 'full_system_dse.py'
    start = time.perf_counter()
    def configured(arguments):
        if args.cost_value_q and '--resume' not in arguments and '--cost-value-q' not in arguments:
            arguments = [*arguments, '--cost-value-q']
        if args.episode_cost_rise is not None and '--resume' not in arguments and '--episode-cost-rise' not in arguments:
            arguments = [*arguments, '--episode-cost-rise', str(args.episode_cost_rise)]
        if '--resume' not in arguments:
            if args.architecture_refresh_interval and '--operators' not in arguments and '--architecture-refresh-interval' not in arguments:
                arguments=[*arguments,'--architecture-refresh-interval',str(args.architecture_refresh_interval)]
            if args.service_features and '--service-features' not in arguments: arguments=[*arguments,'--service-features']
            if args.cost_rank_weight and '--cost-rank-weight' not in arguments: arguments=[*arguments,'--cost-rank-weight',str(args.cost_rank_weight)]
        return arguments

    def run(arguments, expected=0):
        arguments=configured(arguments)
        result = subprocess.run([sys.executable, '-u', '-X', 'utf8', str(script), *map(str, arguments)], capture_output=True, text=True, encoding='utf-8')
        with (out / 'commands.log').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(list(map(str, arguments))) + '\n' + result.stdout + result.stderr + '\n')
        if result.returncode != expected:
            raise AssertionError(result.stdout + result.stderr)

    def command(target):
        return configured(['--seed-case', str(Path(args.seed_case).resolve()), '--out', str(target), '--queries', '14', '--seconds', '240',
                '--architecture-screen', 'off', '--methods', 'rl,sa,random', '--eval-timeout', '20', '--shortlist', '24',
                '--horizon', '4', '--warmup', '3', '--replay-updates', '8', *(['--cost-value-q'] if args.cost_value_q else []),
                *(['--episode-cost-rise',str(args.episode_cost_rise)] if args.episode_cost_rise is not None else [])])
    full, partial = out / 'full', out / 'resumed'
    run(command(full))
    refinement = read(full / 'seed_refinement.json')
    if not refinement or refinement['status'] != 'completed' or refinement['full_ppa'] <= 0:
        raise AssertionError('core CLI did not execute and charge wider initialization')
    if any(s['search_full_ppa'] < 7 for s in read(full / 'summary.json').values()):
        raise AssertionError('initializer consumed the query budget reserved for real search')
    with (out / 'interrupted.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', str(script), *command(partial)], stdout=log, stderr=log)
        deadline = time.monotonic() + 120
        while process.poll() is None and time.monotonic() < deadline:
            if read(partial / 'rl/checkpoint.json', {}).get('steps', 0) >= 3:
                (partial / 'STOP_REQUESTED').touch()
                break
            time.sleep(.05)
        code = process.wait(timeout=60)
        if code != 130 or read(partial / 'status.json')['status'] != 'interrupted':
            raise AssertionError('cooperative stop was not recorded')
    run(['--out', partial, '--resume'])
    for method in ('rl', 'sa', 'random'):
        a, b = read(full / method / 'checkpoint.json'), read(partial / method / 'checkpoint.json')
        def events(root):
            return [json.loads(line) for line in (root / method / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        if [r['state_hash'] for r in events(full)] != [r['state_hash'] for r in events(partial)]:
            raise AssertionError(method + ': resumed action sequence differs')
        if a['elite'] != b['elite'] or a['policy_updates'] != b['policy_updates']:
            raise AssertionError(method + ': resumed best state/updates differ')
        if a.get('architecture_refresh') != b.get('architecture_refresh'):
            raise AssertionError(method + ': architecture schedule changed after resume')
        if args.architecture_refresh_interval and not a['architecture_refresh']['history']:
            raise AssertionError(method + ': actual CLI never evaluated a scheduled architecture transfer')
        ledger = read(partial / method / 'worker/accounting.json')
        keys = [r['durable_key'] for r in ledger if r.get('full_evaluations')]
        if len(set(keys)) != len(keys):
            raise AssertionError(method + ': duplicate PPA charged')
    before = (partial / 'rl/worker/accounting.json').read_bytes()
    run(['--out', partial, '--resume'])
    if before != (partial / 'rl/worker/accounting.json').read_bytes():
        raise AssertionError('completed resume changed PPA ledger')
    from analyze_full_system import analyze
    for folder in (full, partial):
        audit = analyze(folder)
        if not audit['passed']:
            raise AssertionError(audit)
    # Impossible strict area must produce no fake recommendation.
    constrained = out / 'infeasible'
    run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', constrained, '--queries', '6', '--seconds', '60',
         '--architecture-screen', 'off', '--methods', 'rl', '--max-area', '.01', '--replay-updates', '4'])
    rec = read(constrained / 'rl/recommendation.json')
    if rec['strict_feasible'] or rec['state'] is not None:
        raise AssertionError('impossible constraint fabricated a feasible recommendation')
    screened = out / 'shared_screen'
    run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', screened, '--queries', '58', '--seconds', '120',
         '--screen-seconds', '120', '--methods', 'rl,sa,random', '--shortlist', '24', '--replay-updates', '4'])
    screen = read(screened / 'screening.json')
    if not screen['complete_domain_attempted'] or len(screen['rows']) != 50 or screen['full_ppa'] != 50:
        raise AssertionError('complete 50-layout physical screen not verified')
    total = read(screened / 'summary.json')
    if any(s['common_full_ppa'] != 52 or s['full_ppa'] != 58 for s in total.values()):
        raise AssertionError('shared 50-layout charge differs across methods')
    if not analyze(screened)['passed']:
        raise AssertionError('shared-screen audit failed')
    reused = out / 'shared_reused'
    run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', reused, '--queries', '58', '--seconds', '120',
         '--shared-case', screened, '--methods', 'rl', '--shortlist', '24', '--replay-updates', '4'])
    reused_summary = read(reused / 'rl/summary.json')
    provenance = read(reused / 'imported_shared.json')
    if reused_summary['common_full_ppa'] != 52 or reused_summary['full_ppa'] != 58 or provenance['additional_full_ppa'] != 0:
        raise AssertionError('historical common queries not charged exactly')
    if (reused / 'common_worker').exists() or (reused / 'screen_worker').exists():
        raise AssertionError('shared history was physically recomputed')
    if not analyze(reused)['passed']:
        raise AssertionError('reused shared history audit failed')
    lp5 = out / 'lp5'
    run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', lp5, '--queries', '58', '--seconds', '120',
         '--shared-case', screened, '--methods', 'sa', '--operators', 'gemini5', '--no-boundary-features',
         '--no-resource-projection', '--warmup', '1', '--horizon', '2', '--sa-elite-restart',
         '--sa-temperature-reference', 'shared', '--sa-temperature', '10', '--shortlist', '24'])
    lp_events = [json.loads(line) for line in (lp5 / 'sa/events.jsonl').read_text(encoding='utf-8').splitlines()]
    if any(e['action']['family'] not in ('A2', 'A4', 'A5', 'A6', 'A8') for e in lp_events):
        raise AssertionError('Gemini5 arm escaped its declared LP comparison scope')
    if read(lp5 / 'sa/checkpoint.json')['episode'] < 1 or not analyze(lp5)['passed']:
        raise AssertionError('SA elite restart/LP5 workflow was not exercised')
    uniform = out / 'uniform_horizon'
    run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', uniform, '--queries', '10', '--seconds', '120',
         '--methods', 'rl', '--architecture-screen', 'off', '--warmup', '0', '--horizon', '2',
         '--uniform-policy', '--freeze-search-policy', '--no-shared-warmstart', '--shortlist', '24'])
    uniform_summary = read(uniform / 'rl/summary.json')
    uniform_events = [json.loads(line) for line in (uniform / 'rl/events.jsonl').read_text(encoding='utf-8').splitlines()]
    if uniform_summary['policy_updates'] or uniform_summary['algorithm'] != 'uniform_horizon_control':
        raise AssertionError('uniform causal control secretly trained or was labelled as learned RL')
    if any(max(e['probabilities']) - min(e['probabilities']) > 1e-10 for e in uniform_events):
        raise AssertionError('uniform control used non-uniform learned choices')
    frozen = out / 'frozen_shared_q'
    run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', frozen, '--queries', '58', '--seconds', '120',
         '--shared-case', screened, '--methods', 'rl', '--freeze-search-policy', '--warmup', '0',
         '--shortlist', '24', '--replay-updates', '4'])
    if read(frozen / 'rl/summary.json')['policy_updates'] != read(frozen / 'rl/shared_warmstart.json')['policy_updates']:
        raise AssertionError('frozen Q secretly performed search-policy updates')
    if not analyze(uniform)['passed'] or not analyze(frozen)['passed']:
        raise AssertionError('causal control ledger/source audit failed')
    init_stopped = out / 'initializer_stopped'
    with (out / 'initializer_interrupted.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', str(script), *command(init_stopped)], stdout=log, stderr=log)
        deadline = time.monotonic() + 120
        while process.poll() is None and time.monotonic() < deadline:
            if read(init_stopped / 'status.json', {}).get('phase') == 'wide_seed_refinement':
                (init_stopped / 'STOP_REQUESTED').touch()
                break
            time.sleep(.02)
        code = process.wait(timeout=60)
        if code != 130 or read(init_stopped / 'status.json').get('phase') != 'wide_seed_refinement':
            raise AssertionError('initializer cooperative stop was not recorded')
    run(['--out', init_stopped, '--resume'])
    if read(init_stopped / 'summary.json')['rl']['best_state_hash'] != read(full / 'summary.json')['rl']['best_state_hash']:
        raise AssertionError('initializer stop/resume changed the deterministic best mapping')
    if not analyze(init_stopped)['passed']:
        raise AssertionError('initializer stop/resume accounting audit failed')
    if args.episode_cost_rise is not None:
        restart = out / 'measured_restart'
        run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', restart, '--queries', '20', '--seconds', '120',
             '--architecture-screen', 'off', '--seed-refinement', 'off', '--methods', 'rl', '--warmup', '0', '--horizon', '8',
             '--cost-value-q', '--episode-cost-rise', '0', '--episode-restart-probability', '1', '--replay-updates', '4'])
        from simple_rapidchiplet.hybrid.full_space import decision_hash
        from simple_rapidchiplet.hybrid.rl_schema import DesignState
        records = [json.loads(line) for line in (restart / 'rl/events.jsonl').read_text(encoding='utf-8').splitlines()]
        best = decision_hash(DesignState.from_dict(read(restart / 'shared_start.json')['state']))
        cuts = 0
        require_before = None
        for row in records:
            if require_before is not None and row['before_state_hash'] != require_before:
                raise AssertionError('measured episode cut did not return to the verified elite')
            require_before = None
            if row.get('improved_global_best'):
                best = row['state_hash']
            if row.get('episode_cut_reason') == 'measured_cost_rise':
                if not row.get('terminal') or not row.get('feasible') or row.get('reward') is None:
                    raise AssertionError('measured cut lacks a real feasible PPA reward/terminal')
                cuts += 1
                require_before = best
        if not cuts or not analyze(restart)['passed']:
            raise AssertionError('actual workflow did not exercise a measured-cost episode cut')
    if args.cost_value_q:
        import torch
        checkpoint = read(full / 'rl/checkpoint.json')
        trained = torch.load(full / 'rl' / checkpoint['policy_file'], map_location='cpu', weights_only=False)
        if trained.get('representation') != 'physical_cost_factored_double_q_v1' or not trained['cost_replay']:
            raise AssertionError('CLI did not train the declared physical cost representation')
        nonterminal_source='full'
        if not any(not row[3] for row in trained['replay']):
            # Strict cost resets plus a tiny query ceiling can legitimately
            # terminate every episode. Exercise continuation explicitly rather
            # than claiming those terminal transitions tested a future target.
            continuation=out/'physical_continuation'
            run(['--seed-case',str(Path(args.seed_case).resolve()),'--out',continuation,
                 '--queries','10','--seconds','90','--architecture-screen','off','--seed-refinement','off',
                 '--methods','rl','--warmup','0','--horizon','8','--episode-restart-probability','0',
                 '--replay-updates','4'])
            cp=read(continuation/'rl/checkpoint.json')
            data=torch.load(continuation/'rl'/cp['policy_file'],map_location='cpu',weights_only=False)
            if not any(not row[3] for row in data['replay']) or not analyze(continuation)['passed']:
                raise AssertionError('explicit actual CLI failed to exercise nonterminal Bellman continuation')
            nonterminal_source='physical_continuation: measured reset probability0, same real oracle'
    write(out / 'result.json', dict(passed=True, cost_value_q=args.cost_value_q, episode_cost_rise=args.episode_cost_rise,
        architecture_refresh_interval=args.architecture_refresh_interval,service_features=args.service_features,
        cost_rank_weight=args.cost_rank_weight,nonterminal_source=nonterminal_source if args.cost_value_q else None,
        elapsed_s=time.perf_counter() - start,
        checks=['3 methods exact-query CLI', 'same trajectory after cooperative stop/resume', 'same Q update count',
                'no duplicate durable PPA', 'completed resume no new PPA', 'physical/reward/source audits', 'strict infeasible archive',
                '50 actual architecture PPA', 'identical shared-screen charges for all 3 methods',
                'shared physical history reuse with exact inherited charge and zero new common PPA',
                'Gemini-style LP5 masking and SA elite restart',
                'uniform choices with identical RL horizon and zero SGD',
                'frozen shared Q with zero online updates',
                'core CLI wider initialization with query reserve and charged shared PPA',
                'cooperative initialization stop/resume and source/accounting audit']))
    print(json.dumps(read(out / 'result.json')))


if __name__ == '__main__':
    main()
