"""GUI v1/v2, bounded multi-start search and explicit local DP acceptance."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json,tempfile,time,unittest

import hybrid_gui
from gui_request import SEARCH_PROFILES
from simple_rapidchiplet.hybrid.architecture import enumerate_gemini_architectures
from simple_rapidchiplet.hybrid.workload import make_toy_workload,Workload,TensorSpec
from simple_rapidchiplet.hybrid.mapping import initial_mapping,compile_mapping
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.rl_schema import stable_hash
from simple_rapidchiplet.hybrid.rl_search import JointSearchConfig,JointObjective,run_joint_search
from simple_rapidchiplet.hybrid.auto_proposals import AutoProposals
from simple_rapidchiplet.hybrid.auto_runtime import BoundedOraclePool,EvaluationTimeout,atomic_json
from simple_rapidchiplet.hybrid.rl_actions import apply_transaction
from simple_rapidchiplet.hybrid.ondie import core_service

class GUI04ContractTests(unittest.TestCase):
    def test_v1_defaults_and_exact_numeric_inputs_are_unchanged(self):
        c,p=hybrid_gui.normalize_gui_request({})
        self.assertEqual({k:c[k] for k in ['batch','budget','seed','fusion','fusion_depth','fusion_budget']},
            dict(batch=1,budget=8,seed=7,fusion='dp',fusion_depth=4,fusion_budget=6))
        raw=dict(models=['toy'],batch=2,budget=12,seed=3,fusion='off',fusion_depth=2)
        self.assertEqual(hybrid_gui.normalize_gui_request(raw)[0],hybrid_gui.validate_config(raw))
        self.assertNotIn('engine_profile',c);self.assertFalse(p['auto_resolved'])
    def test_default_v2_has_numeric_effective_bounds_and_preserves_PPA(self):
        raw=dict(ui_schema_version=2,models=['resnet18'],max_power=6.3,max_area=150.,max_latency_ms=.8,
            strict_power=True,strict_area=True,strict_latency=False,preference='low_power')
        before=json.loads(json.dumps(raw));c,p=hybrid_gui.normalize_gui_request(raw)
        self.assertEqual(raw,before);self.assertEqual(c['search_mode'],'standard');self.assertEqual(c['batch'],1)
        self.assertEqual(c['seed'],7);self.assertEqual(c['budget_mode'],'auto');self.assertIs(type(c['budget']),int)
        for k in ['max_power','max_area','max_latency_ms','strict_latency','preference']:self.assertEqual(c[k],raw[k])
        self.assertEqual(len(c['architecture_templates']),15);self.assertEqual(p['microbatch_candidates'],[1])
        self.assertFalse(p['capabilities']['C10_local_partition_DP'])
    def test_manual_budget_is_used_for_every_profile(self):
        for mode in SEARCH_PROFILES:
            c,p=hybrid_gui.normalize_gui_request(dict(ui_schema_version=2,search_mode=mode,budget_mode='manual',budget=13,seed=17))
            self.assertEqual(c['budget'],13);self.assertEqual(c['seed'],17)
    def test_malformed_intents_are_rejected(self):
        for change in [dict(ui_schema_version=True),dict(ui_schema_version=5),dict(workload_mode='throughput'),
            dict(batch=2),dict(fusion_mode='on'),dict(search_mode='magic'),dict(budget='Auto'),
            dict(budget_mode='manual',budget=True),dict(budget_mode='manual',budget=0),dict(seed=-1),dict(fusion_depth=99)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                hybrid_gui.normalize_gui_request({'ui_schema_version':2,**change})
    def test_resume_uses_saved_effective_config_without_resolving_auto(self):
        raw=dict(ui_schema_version=2,models=['toy']);c,p=hybrid_gui.normalize_gui_request(raw)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);(path/'requested_ui.json').write_text(json.dumps(raw));(path/'config.json').write_text(json.dumps(c))
            with patch('hybrid_gui.normalize_gui_request',side_effect=AssertionError('Auto must not rerun')),patch('auto_dse.run_auto_suite',return_value={'ok':True}) as run:
                self.assertEqual(hybrid_gui.run_suite(raw,path,resume=True),{'ok':True})
                self.assertEqual(run.call_args.args[0],c)
    def test_frontend_has_v2_defaults_and_no_exposed_depth_input(self):
        html=(Path(hybrid_gui.STATIC)/'index.html').read_text(encoding='utf-8')
        self.assertIn('value="standard" selected',html);self.assertIn('value="auto" selected',html)
        self.assertIn('<details id="advanced">',html);self.assertNotIn('id="fusion_depth"',html)
        self.assertNotIn('id="batch"',html);self.assertNotIn('id="fusion"',html)

class GUI04EngineTests(unittest.TestCase):
    def setUp(self):
        self.w=make_toy_workload(batch=1,channels=8,spatial=8)
        self.candidates=enumerate_gemini_architectures(compute_counts=(1,2))
        self.c=self.candidates[0]
        self.plan=initial_mapping(self.w,self.c.package.mapping_core_ids,self.c.package.memory_ids,max_group_layers=3)
        self.s=DesignState(self.c.id,self.plan,self.w.fingerprint)
    def test_multistart_keeps_completed_different_archives_without_fake_RL_calls(self):
        states=[]
        for i,c in enumerate(self.candidates[:2]):
            plan=initial_mapping(self.w,c.package.mapping_core_ids,c.package.memory_ids,max_group_layers=3)
            m={'latency_s':2.-i,'energy_j':2.,'area_mm2':10.,'power_w':2./(2.-i),'feasible':True}
            states.append((DesignState(c.id,plan,self.w.fingerprint),m))
        r=run_joint_search(self.w,states[0][0],self.candidates,
            lambda c:lambda p:(_ for _ in ()).throw(AssertionError('must reuse paid seed PPA')),
            objective=JointObjective(latency_weight=1,area_weight=0,power_weight=0),
            config=JointSearchConfig(budget=4),bootstrap_evaluations=states,
            resource_guard=lambda _: 'wall_time_budget_exhausted')
        self.assertEqual(r.evaluations,0);self.assertEqual(len(r.architecture_archive),2)
        self.assertEqual(r.best_state.architecture_candidate_id,states[1][0].architecture_candidate_id)
        self.assertEqual(r.status,'wall_time_budget_exhausted')
    def test_local_dp_exposes_real_depth_two_and_three_explicit_transactions(self):
        cfg={'fusion':'dp','local_fusion_dp':True,'fusion_depth':4,'fusion_depth_max':8,'shortlist':48}
        hook=AutoProposals(self.w,self.candidates,cfg)
        plans,stats=hook(self.s,{},(),{'proposed':{}})
        depths=set()
        for p in plans:
            trial=apply_transaction(self.w,self.s,p,self.candidates)
            depths.update(len(stack.layer_ids) for stack in trial.mapping.fusion_stacks)
            self.assertEqual(trial.mapping.microbatch_size,1)
            for stack in trial.mapping.fusion_stacks:
                self.assertIn(tuple(stack.layer_ids),[tuple(g.layer_ids) for g in trial.mapping.groups])
        self.assertTrue({2,3}<=depths,depths)
        self.assertTrue(all(x['full_evaluator_calls']==0 for x in stats['gui04_proposals']['local_dp']))
    def test_off_never_exposes_fusion_and_microbatch_remains_one(self):
        hook=AutoProposals(self.w,self.candidates,{'fusion':'off','fusion_depth':4,'fusion_depth_max':8})
        plans,stats=hook(self.s,{},(),{'proposed':{}})
        self.assertEqual(stats['gui04_proposals']['local_dp'],[])
        self.assertFalse(any(p.family_id in ['A1','A3','A7','C6','C7','C8'] for p in plans))
    def test_isolated_oracle_timeout_has_no_partial_PPA_and_stops_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            pool=BoundedOraclePool(self.w,temp,deadline=time.monotonic()+5,evaluation_timeout_s=.0001)
            try:
                with self.assertRaises(EvaluationTimeout):pool.evaluate(self.c,self.plan,phase='test')
                self.assertEqual(pool.full_evaluations,0);self.assertEqual(pool.records[-1]['status'],'timeout')
                self.assertIsNone(pool.process)
            finally:pool.close()
    def test_exact_same_microkernel_reuses_cache_across_system_geometry(self):
        core_service.cache_clear()
        first=compile_mapping(self.w,self.plan,self.c.hardware).units[0];layer=self.w.layer_map[first.layer_id]
        a=core_service(first,layer,self.w,self.c.hardware);before=core_service.cache_info()
        other=self.candidates[1];b=core_service(first,layer,self.w,other.hardware)
        self.assertEqual(a.to_dict(),b.to_dict());self.assertEqual(core_service.cache_info().hits,before.hits+1)
    def test_omitting_derived_report_before_copy_preserves_legacy_decision_hash(self):
        state=replace(self.s,evaluation_report={'large_diagnostic':[{'layer':i,'value':i*2} for i in range(1000)]},remaining_budget=19)
        legacy=state.to_dict();legacy.pop('evaluation_report');legacy.pop('remaining_budget')
        self.assertEqual(state.fingerprint,stable_hash(legacy))
        self.assertEqual(state.fingerprint,self.s.fingerprint)
    def test_deep_window_survives_prefix_budget_and_passes_global_oracle(self):
        from simple_rapidchiplet.hybrid.evaluator import HybridEvaluator
        tensors=[TensorSpec('image',(1,8,8,8))];layers=[];previous='image'
        for i in range(6):
            lid='chain'+str(i);out='t'+str(i)
            layers.append(replace(self.w.layers[0],id=lid,inputs=(previous,),output=out))
            tensors.append(TensorSpec(out,(1,8,8,8),producer=lid));previous=out
        w=Workload('six_layer_depth_probe',tuple(tensors),tuple(layers),('image',),(previous,),1)
        p=initial_mapping(w,self.c.package.mapping_core_ids,self.c.package.memory_ids,max_group_layers=1)
        state=DesignState(self.c.id,p,w.fingerprint)
        hook=AutoProposals(w,self.candidates,{'fusion':'dp','fusion_depth':6,'fusion_depth_max':8,'shortlist':48})
        plans,stats=hook(state,{},(),{'proposed':{}})
        trials=[apply_transaction(w,state,plan,self.candidates) for plan in plans]
        deep=next((t for t in trials if any(len(s.layer_ids)==6 for s in t.mapping.fusion_stacks)),None)
        self.assertIsNotNone(deep,'A displayed cap is insufficient: a legal full-depth transaction is required')
        e=HybridEvaluator(w,self.c.hardware,trace=False).evaluate(deep.mapping)
        self.assertTrue(e.metrics['feasible']);self.assertEqual(e.metrics['executed_macs'],w.total_macs)
    def test_atomic_json_recovers_transient_windows_replace_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'accounting.json';atomic_json(path,{'old':True})
            original=Path.replace;attempts=[]
            def transient(source,target):
                attempts.append(1)
                if len(attempts)<3:raise PermissionError('simulated Windows temporary file lock')
                return original(source,target)
            with patch.object(Path,'replace',transient):atomic_json(path,{'complete':True,'queries':3})
            self.assertEqual(json.loads(path.read_text()),{'complete':True,'queries':3})
            self.assertEqual(len(attempts),3);self.assertEqual(list(Path(temp).glob('*.tmp')),[])
    def test_published_response_recovers_transient_read_lock_and_obeys_deadline(self):
        from simple_rapidchiplet.hybrid.auto_runtime import read_completed_response
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'response.json';atomic_json(path,{'status':'complete','metrics':{'latency_s':1}})
            original=Path.read_text;attempts=[]
            def transient(p,*args,**kwargs):
                attempts.append(1)
                if len(attempts)<3:raise PermissionError('simulated published-file scanner lock')
                return original(p,*args,**kwargs)
            with patch.object(Path,'read_text',transient):
                value=read_completed_response(path,time.monotonic()+1)
            self.assertEqual(value['metrics']['latency_s'],1);self.assertEqual(len(attempts),3)
            with patch.object(Path,'read_text',side_effect=PermissionError('persistent lock')):
                with self.assertRaises(EvaluationTimeout):read_completed_response(path,time.monotonic()-.01)

if __name__=='__main__':unittest.main()
