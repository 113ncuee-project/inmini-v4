"""Re-evaluate one published ResNet18 best mapping using the full physical oracle.

Run from the repository root with the documented Python environment. This
executes one real PPA, does not train RL, and requires the project experiment
lock so it cannot silently overlap a formal timed run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP = ROOT / 'simplified_rapidchiplet'
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / 'scripts'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=(7,17), default=7)
    p.add_argument('--method', choices=('rl','sa','random'), default='rl')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=600)
    args = p.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / 'output').resolve()) or out.exists():
        raise ValueError('choose a new output directory under repository output/')
    from run_local_rl_diagnostics import lock
    from profile_mapping_oracle import load_workload
    from simple_rapidchiplet.hybrid.rl_schema import DesignState
    from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
    from simple_rapidchiplet.hybrid.single_array_runtime import SingleArrayOraclePool
    published = HERE / str(args.seed)
    manifest = read(published / 'source_snapshot_manifest.json')
    controller = {'full_space.py','multi_step_q.py','single_array_seed_v3.py','mapping_proposals_v4.py','rl_actions.py'}
    physical = {name: value for name,value in manifest['files'].items()
                if (name.startswith('simple_rapidchiplet/') and Path(name).name not in controller)
                or name == 'scripts/profile_mapping_oracle.py'}
    mismatches = [name for name,value in physical.items() if not (APP / name).is_file()
                  or hashlib.sha256((APP / name).read_bytes()).hexdigest() != value]
    if mismatches:
        raise ValueError('physical source bytes changed; use repository commit82fd95e or original ZIP: ' + ', '.join(mismatches))
    rec = read(published / args.method / 'recommendation.json')
    if not rec['strict_feasible']:
        raise ValueError('published recommendation is not strict-feasible')
    state = DesignState.from_dict(rec['state'])
    workload = load_workload(ROOT / 'inputs/replay_v04/resnet18/workload.json')
    if workload.fingerprint != state.model_hash:
        raise ValueError('workload graph differs from the published mapping')
    candidate = next(c for c in enumerate_single_array_architectures_v3() if c.id == state.architecture_candidate_id)
    out.mkdir(parents=True)
    started = time.monotonic()
    with lock(ROOT / 'output/.local_rl_diagnostics.lock'):
        pool = SingleArrayOraclePool(workload, out / 'worker', deadline=time.monotonic()+args.timeout+30,
                                     evaluation_timeout_s=args.timeout)
        try:
            actual = pool.evaluate(candidate, state.mapping, phase='published_best_replay')['metrics']
            count = pool.full_evaluations
        finally:
            pool.close()
    keys = ('latency_s','energy_j','area_mm2','power_w','executed_macs','recompute_macs',
            'dram_read_bytes','dram_write_bytes','noc_hop_bytes','nop_hop_bytes')
    expected = rec['metrics']
    compared = {k: dict(expected=expected[k], actual=actual[k],
        equal=math.isclose(actual[k],expected[k],rel_tol=1e-12,abs_tol=1e-18)) for k in keys}
    result = dict(passed=all(v['equal'] for v in compared.values()), new_full_ppa=count, policy_updates=0,
                  physical_files_verified=len(physical), elapsed_s=time.monotonic()-started,
                  seed=args.seed, method=args.method, analytical_ppa=True, compared=compared)
    (out / 'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    raise SystemExit(not result['passed'])


if __name__ == '__main__':
    main()
