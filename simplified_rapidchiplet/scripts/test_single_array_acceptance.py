"""Run the new engine and required historical regressions on frozen sources."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ('test_tensor_locality.py', 'test_graph_edit.py', 'test_full_system.py', 'test_calendar_index.py', 'test_replay_mapping_v4.py', 'test_local_rl_diagnostics.py', 'test_single_array*.py', 'test_joint_rl.py', 'test_rl_acceptance.py', 'test_rl_integration.py',
            'test_tangram_seed.py', 'test_gemini_hardware_diagnostics.py', 'test_gui04.py', 'test_gui04_v3.py')


def manifest():
    paths = list((ROOT / 'simple_rapidchiplet' / 'hybrid').glob('*.py'))
    paths += [ROOT / name for name in ('single_array_dse.py', 'single_array_gui.py', 'gui_request.py', 'gui_request_v3.py', 'gui_request_v4.py', 'hybrid_gui.py')]
    paths += list((ROOT / 'gui').glob('*'))
    paths += list((ROOT / 'scripts').glob('*local*.py'))
    paths += [ROOT / 'replay_mapping_dse.py', ROOT / 'full_system_dse.py', ROOT / 'scripts' / 'analyze_full_system.py', ROOT / 'scripts' / 'build_native_calendar.py']
    paths += list((ROOT / 'simple_rapidchiplet' / 'hybrid').glob('*.cpp'))
    paths += list((ROOT / 'simple_rapidchiplet' / 'hybrid' / '_native').glob('*'))
    for pattern in PATTERNS:
        paths += list((ROOT / 'tests').glob(pattern))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths)) if path.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='../output/inmin_v02_20261101/acceptance_final')
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if not output.is_relative_to(ROOT.parent.resolve()):
        raise ValueError('acceptance output must remain in the workspace')
    output.mkdir(parents=True, exist_ok=True)
    temporary_root = (ROOT.parent / 'tmp').resolve()
    if not temporary_root.is_relative_to(ROOT.parent.resolve()):
        raise ValueError('test temporary root escapes the workspace')
    temporary_root.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary_root)
    import sys
    sys.path.insert(0, str(ROOT))
    before = manifest()
    started = time.monotonic()
    suite = unittest.TestSuite()
    for pattern in PATTERNS:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
    with (output / 'tests.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = manifest()
    data = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
            'elapsed_s': time.monotonic() - started, 'source_stable': before == after,
            'source_before': before, 'source_after': after,
            'passed': result.wasSuccessful() and before == after, 'patterns': PATTERNS}
    (output / 'result.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in data.items() if not key.startswith('source_')}))
    raise SystemExit(not data['passed'])


if __name__ == '__main__':
    main()
