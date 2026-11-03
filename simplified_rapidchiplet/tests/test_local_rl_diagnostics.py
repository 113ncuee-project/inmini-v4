"""Local runner controls and real transaction boundaries (no large model runs)."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

APP=Path(__file__).resolve().parents[1]
TEMP=APP.parent/'tmp/local_diagnostics_tests'
TEMP.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(APP),str(APP/'scripts')]
from run_local_rl_diagnostics import build_plan, worst_case, command, write, sha, sources, verify_snapshot, lock, main
from analyze_local_rl_diagnostics import analyze, physical_checks, status
from simple_rapidchiplet.hybrid.search_space import check_transition
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.mapping import initial_mapping
from simple_rapidchiplet.hybrid.workload import make_toy_workload
from simple_rapidchiplet.hybrid.rl_schema import DesignState, make_action_plan
from simple_rapidchiplet.hybrid.rl_actions import generate_candidates, apply_transaction
from simple_rapidchiplet.hybrid.single_array_proposals import SingleArrayJointProposals

class LocalDiagnostics(unittest.TestCase):
    def args(self,**changes):
        return argparse.Namespace(**dict(profile='diagnostic',phase='p1',architecture='single_array_c2_2x1',seed=7,arms=None,**changes))

    def fixture(self):
        workload=make_toy_workload(batch=1,channels=5,spatial=7)
        candidates=enumerate_single_array_architectures_v3()
        candidate=next(c for c in candidates if c.id=='single_array_c2_2x1')
        mapping=initial_mapping(workload,candidate.package.mapping_core_ids,candidate.package.memory_ids,max_group_layers=1)
        state=DesignState(candidate.id,mapping,workload.fingerprint,candidate.hardware.profile_id,version=3)
        return workload,candidates,candidate,state

    def test_diagnostic_budget_and_shared_seed(self):
        plan=build_plan(self.args())
        self.assertEqual(worst_case(plan)['charged_wall_upper_per_search_arm_s'],1800)
        self.assertEqual(plan['cases'][1]['common'],plan['cases'][0]['id'])
        self.assertEqual(plan['cases'][1]['space'],'mapping_only')

    def test_distinct_start_keeps_best_seed_as_gain_reference(self):
        from types import SimpleNamespace
        from single_array_dse import select_seed_start
        from simple_rapidchiplet.hybrid.rl_search import JointObjective
        reference=dict(latency_s=1,energy_j=1,power_w=1,area_mm2=1,feasible=True)
        bootstrap=[(SimpleNamespace(fingerprint=str(i)),{**reference,'latency_s':i+1}) for i in range(3)]
        best,start=select_seed_start(bootstrap,JointObjective(),reference,2)
        self.assertEqual(best[0].fingerprint,'0')
        self.assertEqual(start[0].fingerprint,'2')
        with self.assertRaises(ValueError):select_seed_start(bootstrap,JointObjective(),reference,3)

    def test_two_by_two_is_four_generators_feature_masks(self):
        args=self.args();args.phase='p2';p=build_plan(args)
        self.assertEqual({(c['generator'],c['features']) for c in p['cases'][1:]},
            {('generic','off'),('generic','on'),('guided','off'),('guided','on')})

    def test_long_budget_requires_research(self):
        args=self.args();args.seconds=1801
        with self.assertRaises(ValueError):build_plan(args)

    def test_smoke_cannot_launch_resnet(self):
        args=self.args();args.profile='smoke'
        with self.assertRaises(ValueError):build_plan(args)

    def test_primitive_or_nested_architecture_rejected_whole(self):
        w,cs,c,state=self.fixture(); target=next(x for x in cs if x.id!=c.id)
        for family in ('A9','C2'):
            plan=make_action_plan(state,family,[('A9',{'architecture_candidate_id':target.id,'mapping':state.mapping.to_dict()})])
            with self.assertRaisesRegex(ValueError,'entire transaction'):
                check_transition(state,state,'mapping_only',plan)

    def test_mask_filters_before_policy(self):
        w,cs,c,state=self.fixture()
        plans,stats=generate_candidates(w,state,{},cs,rng=random.Random(7),search_space='mapping_only',fusion_enabled=True)
        self.assertTrue(plans)
        for plan in plans:
            check_transition(state,apply_transaction(w,state,plan,cs),'mapping_only',plan)
        self.assertTrue(any('search_space' in reason for reason in stats['hard_rejected']))

    def test_generic_ignores_measured_diagnosis(self):
        w,cs,c,state=self.fixture()
        def generate(report):
            plans,_=generate_candidates(w,state,report,[c],rng=random.Random(7),search_space='mapping_only',generator_guidance=False)
            hook=SingleArrayJointProposals(w,[c],generator_guidance=False,fusion_enabled=False,timeout_s=10)
            plans,stats=hook(state,report,plans,{'proposed':{}})
            return [p.to_dict() for p in plans]
        self.assertEqual(generate({}),generate({'diagnostics':{'layers':[{'layer_id':w.layers[0].id,'criticality':1,'compute_busy_s':1000}]}}))

    def test_mapping_joint_proposals_do_not_require_fusion_to_compile(self):
        w,_,c,state=self.fixture()
        from simple_rapidchiplet.hybrid import single_array_proposals as module
        original=module.apply_transaction
        def reject_fusion(workload,before,plan,candidates):
            if any(e.action_id=='A1' for e in plan.ordered_edits):
                raise ValueError('deliberate fusion-only failure')
            return original(workload,before,plan,candidates)
        hook=SingleArrayJointProposals(w,[c],fusion_enabled=False,timeout_s=10)
        with patch.object(module,'apply_transaction',side_effect=reject_fusion):
            plans,_=hook(state,{},(),{'proposed':{}})
        self.assertTrue(any(p.family_id=='C9' for p in plans))

    def test_fusion_difference_rejected(self):
        _,_,_,state=self.fixture()
        trial=replace(state,mapping=replace(state.mapping,fusion_stacks=('synthetic_changed_stack',)))
        with self.assertRaisesRegex(ValueError,'fusion'):check_transition(state,trial,'mapping_only')
        check_transition(state,trial,'mapping_fusion')

    def test_physical_consistency_detects_errors(self):
        errors=[]
        physical_checks(dict(feasible=True,latency_s=2,energy_j=6,power_w=3,energy_components={'x':6}), 'ok',errors)
        self.assertFalse(errors)
        physical_checks(dict(feasible=True,latency_s=2,energy_j=7,power_w=3), 'bad',errors)
        self.assertTrue(errors)

    def test_admin_partial_missing_evidence_no_subprocess(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            root=Path(tmp); write(root/'effective_config.json',{})
            with patch('subprocess.Popen',side_effect=AssertionError('must not run oracle')):
                self.assertEqual(status(root)['unique_completed_full_ppa'],0)
                result=analyze(root)
                self.assertTrue((root/'reports/summary.md').exists())
                self.assertTrue(result['errors']) # missing snapshot explicitly detected

    def test_atomic_metadata_survives_transient_windows_lock(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            path=Path(tmp)/'state.json';write(path,{'status':'old'})
            original=Path.replace;attempts=[]
            def transient(source,target):
                attempts.append(1)
                if len(attempts)==1:raise PermissionError('simulated scanner read lock')
                return original(source,target)
            with patch.object(Path,'replace',transient):write(path,{'status':'new'})
            self.assertEqual(json.loads(path.read_text()),{'status':'new'})
            self.assertEqual(len(attempts),2)

    def test_zero_feasible_timeout_case_still_reports_unknown(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            root=Path(tmp);snapshot=root/'source_snapshot';snapshot.mkdir()
            write(root/'source_snapshot_manifest.json',{'files':{}})
            case=dict(id='toy/7/seed',kind='seed',model='toy',seed=7,space='mapping_only',fusion='off',
                arms='seed_only',generator='guided',features='on')
            plan={'cases':[case]};write(root/'effective_config.json',plan);write(root/'manifest.json',{'plan_hash':sha(plan)})
            write(root/'cases/toy/7/seed/seed_attempts.json',{'attempts':[{'status':'timeout','reason':'simulated timeout'}],
                'bootstrap':[],'elapsed_s':1})
            with patch('subprocess.Popen',side_effect=AssertionError('no oracle allowed')):
                result=analyze(root)
            self.assertFalse(result['errors'])
            rows=(root/'reports/compare_methods.csv').read_text(encoding='utf-8-sig')
            self.assertIn('seed_only',rows)
            self.assertIn('simulated timeout',(root/'reports/failure_reasons.csv').read_text(encoding='utf-8-sig'))
            self.assertIn('容量',(root/'reports/diagnostics.md').read_text(encoding='utf-8'))

    def test_manifest_tamper_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            root=Path(tmp);snapshot=root/'source_snapshot';snapshot.mkdir()
            (snapshot/'oracle.py').write_text('original')
            write(root/'source_snapshot_manifest.json',{'files':sources(snapshot)})
            verify_snapshot(root)
            (snapshot/'oracle.py').write_text('changed')
            with self.assertRaises(ValueError):verify_snapshot(root)

    def test_lock_excludes_second_runner(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            path=Path(tmp)/'run.lock'
            with lock(path):
                with self.assertRaises(OSError):
                    with lock(path):pass

    def test_feature_columns_match_actor_critic_sizes(self):
        from local_rl_feature_schema import schema
        data=schema()
        self.assertEqual(len(data['state']),24)
        self.assertEqual(len(data['candidate']),283)
        self.assertEqual(len(data['target']),12)

    def test_command_is_fixed_explicit_and_offline(self):
        plan=build_plan(self.args());cmd=command(plan['cases'][1],plan,Path('output/example').resolve())
        self.assertIn('--fixed-architecture',cmd);self.assertIn('--common-seed',cmd)
        self.assertIn('--search-space',cmd);self.assertIn('--stop-file',cmd)
        self.assertNotIn('7200',cmd)

    def test_durable_completion_recovered_without_oracle(self):
        from simple_rapidchiplet.hybrid.single_array_runtime import SingleArrayOraclePool
        import time
        w,_,c,state=self.fixture()
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            folder=Path(tmp)
            digest=hashlib.sha256(json.dumps({'architecture':c.id,'hardware_profile':c.hardware.profile_id,
                'mapping':state.mapping.to_dict()},sort_keys=True).encode()).hexdigest()
            write(folder/'completed'/ (digest+'.json'),dict(status='complete',full_evaluations=1,
                cache_hits=0,metrics={'feasible':True},architecture=c.id,mapping=state.mapping.to_dict(),evaluation_elapsed_s=.01))
            pool=SingleArrayOraclePool(w,folder,deadline=time.monotonic()+5,evaluation_timeout_s=1)
            self.assertEqual(pool.full_evaluations,1)
            with patch.object(pool,'request',side_effect=AssertionError('must not recompute')):
                response=pool.evaluate(c,state.mapping)
            self.assertEqual(response['full_evaluations'],0)
            self.assertEqual(pool.full_evaluations,1)
            self.assertEqual(sum(r.get('full_evaluations',0) for r in pool.records),1)
            pool.close()

if __name__=='__main__':unittest.main()
