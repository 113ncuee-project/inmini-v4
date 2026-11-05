"""Actual OS lock interoperability and bounded admission, zero PPA."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP/'scripts'))
from full_cohort import shared_lock,validate_lease
from run_local_rl_diagnostics import lock
from run_parallel_full_suite import shared_ready


class FullCohortTests(unittest.TestCase):
    def setUp(self):
        folder = APP.parent/'tmp'
        folder.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root/'global.lock'
        self.path.write_bytes(b'0')

    def test_multiple_shared_holders_block_legacy_serial_until_last_close(self):
        with ExitStack() as stack:
            stack.enter_context(shared_lock(self.path))
            inner = stack.enter_context(ExitStack())
            inner.enter_context(shared_lock(self.path))
            with self.assertRaises(OSError):
                with lock(self.path): pass
            inner.close()
            with self.assertRaises(OSError):
                with lock(self.path): pass
        with lock(self.path): pass

    def test_existing_serial_owner_blocks_managed_shared_admission(self):
        with lock(self.path):
            with self.assertRaises(OSError):
                with shared_lock(self.path): pass
        with shared_lock(self.path): pass

    def test_exclusive_slot_still_allows_only_one_case(self):
        with lock(self.path):
            with self.assertRaises(OSError):
                with lock(self.path): pass

    def test_declared_outputs_source_status_and_slot_are_checked(self):
        output = self.root/'output/cohort/cases/7'
        lease_path = self.root/'output/cohort/cohort_lease.json'
        lease_path.parent.mkdir(parents=True)
        lease = dict(status='active',workspace=str(self.root.resolve()),source_identity='expected',jobs=2,
                     output_root=str(lease_path.parent.resolve()),allowed_outputs=[str(output.resolve())])
        lease_path.write_text(json.dumps(lease),encoding='utf-8')
        validate_lease(self.root,output,lease_path,1,'expected')
        for path,slot,source in ((output,2,'expected'),(self.root/'elsewhere',0,'expected'),(output,0,'changed')):
            with self.assertRaises(ValueError):
                validate_lease(self.root,path,lease_path,slot,source)
        lease['status']='completed'
        lease_path.write_text(json.dumps(lease),encoding='utf-8')
        with self.assertRaises(ValueError):
            validate_lease(self.root,output,lease_path,0,'expected')

    def test_other_process_shared_lock_blocks_original_exclusive_lock(self):
        ready,stop = self.root/'ready',self.root/'stop'
        code = """import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from full_cohort import shared_lock
with shared_lock(Path(sys.argv[2])):
    Path(sys.argv[3]).write_text('ready')
    end=time.monotonic()+15
    while not Path(sys.argv[4]).exists() and time.monotonic()<end:time.sleep(.01)
"""
        process = subprocess.Popen([sys.executable,'-X','utf8','-c',code,str(APP/'scripts'),str(self.path),str(ready),str(stop)],
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            end = time.monotonic()+10
            while not ready.exists() and process.poll() is None and time.monotonic()<end:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            with self.assertRaises(OSError):
                with lock(self.path): pass
        finally:
            stop.touch()
            stdout,stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode,0,stderr.decode(errors='replace'))
        with lock(self.path): pass

    def test_screen_off_initializer_is_shareable_without_a_screen_file(self):
        case=self.root/'case';case.mkdir()
        for name,value in (('effective_config.json',dict(architecture_screen='off')),
                           ('status.json',dict(status='running',method='rl')),
                           ('common.json',{}),('shared_start.json',{})):
            (case/name).write_text(json.dumps(value),encoding='utf-8')
        self.assertFalse((case/'screening.json').exists())
        self.assertTrue(shared_ready(case))

    def test_screen_on_and_incomplete_phase_do_not_share_early(self):
        case=self.root/'case';case.mkdir()
        for name,value in (('effective_config.json',dict(architecture_screen='all')),
                           ('status.json',dict(status='running',method='rl')),
                           ('common.json',{}),('shared_start.json',{})):
            (case/name).write_text(json.dumps(value),encoding='utf-8')
        self.assertFalse(shared_ready(case))
        (case/'screening.json').write_text('{}',encoding='utf-8')
        self.assertTrue(shared_ready(case))
        (case/'status.json').write_text(json.dumps(dict(status='running',method='initialization')),encoding='utf-8')
        self.assertFalse(shared_ready(case))


if __name__ == '__main__': unittest.main()
