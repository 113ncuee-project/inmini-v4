"""Create a self-contained, actually evaluated two-point Toy input archive."""
import argparse
from pathlib import Path
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from run_local_rl_diagnostics import write, sources, sha, lock
from simple_rapidchiplet.hybrid.mapping import initial_mapping
from simple_rapidchiplet.hybrid.mapping_proposals_v4 import StructuredMappings
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.single_array_runtime import SingleArrayOraclePool
from simple_rapidchiplet.hybrid.workload import make_toy_workload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if (out / 'seed_attempts.json').exists():
        raise ValueError('Toy archive already exists')
    w = make_toy_workload(batch=1, channels=5, spatial=7)
    c = next(c for c in enumerate_single_array_architectures_v3() if c.id == 'single_array_c2_2x1')
    s = DesignState(c.id, initial_mapping(w, c.hardware.compute_slots, c.package.memory_ids, max_group_layers=1),
                    w.fingerprint, c.hardware.profile_id, version=3)
    t = StructuredMappings(w, c, s).build((4, 'balanced', 'spread', 'interleave', 'auto')).state
    pool = SingleArrayOraclePool(w, out / 'fixture_worker', deadline=time.monotonic() + 120, evaluation_timeout_s=30)
    try:
        bootstrap = [dict(state=state.to_dict(), metrics=pool.evaluate(c, state.mapping, phase='qa_fixture')['metrics']) for state in (s, t)]
        if not all(row['metrics']['feasible'] for row in bootstrap):
            raise ValueError('Toy fixture must be physically executable')
        write(out / 'workload.json', w.to_dict())
        write(out / 'seed_attempts.json', dict(bootstrap=bootstrap))
        write(out / 'source_identity.json', dict(workload=w.fingerprint, source=sha(sources(APP)),
            full_ppa=pool.full_evaluations, scope='QA Toy fixture only; no DNN improvement claim'))
    finally:
        pool.close()


if __name__ == '__main__':
    workspace = next((x for x in APP.parents if (x / '.venv/Scripts/python.exe').is_file() and (x / 'inputs').is_dir()), APP.parent)
    with lock(workspace / 'output/.local_rl_diagnostics.lock'):
        main()
