from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np

from simple_rapidchiplet.hybrid.single_array_hardware import (
    SingleArrayHardware, SingleArrayHardwareV3, PROFILE_V3, enumerate_single_array_architectures_v3)
from simple_rapidchiplet.hybrid.mapping import initial_mapping, FusionStack
from simple_rapidchiplet.hybrid.rl_schema import DesignState, make_action_plan
from simple_rapidchiplet.hybrid.rl_actions import apply_transaction, resized_mapping, generate_candidates
from simple_rapidchiplet.hybrid.rl_search import (candidate_features, state_features, JointObjective,
    ACTION_SIZE_V3, JointSearchConfig, run_joint_search)
from simple_rapidchiplet.hybrid.single_array_evaluator import SingleArrayEvaluator
from simple_rapidchiplet.hybrid.single_array_seed_v3 import generate_seeds_v3
from simple_rapidchiplet.hybrid.target_bottleneck import memory_actions
from simple_rapidchiplet.hybrid.workload import make_toy_workload


class SingleArrayV3Tests(unittest.TestCase):
    def setUp(self):
        self.w = make_toy_workload(batch=1, channels=5, spatial=7)
        self.cs = enumerate_single_array_architectures_v3()
        self.h = self.cs[0].hardware
        self.p = initial_mapping(self.w, self.h.compute_slots, ('memory:0', 'memory:1'), max_group_layers=3)
        self.s = DesignState(self.cs[0].id, self.p, self.w.fingerprint, PROFILE_V3, version=3)

    def test_all_50_rectangles_and_io_cost_are_real(self):
        self.assertEqual(len(self.cs), 50)
        self.assertEqual({c.hardware.compute_count for c in self.cs}, set(range(1, 17)))
        for c in self.cs:
            h = c.hardware
            self.assertEqual(h.io_count, 2 * h.rows)
            self.assertEqual(len(h.compute_slots), 16 * h.compute_count)
            self.assertEqual(SingleArrayHardware.from_dict(h.to_dict()), h)
        a, b = SingleArrayHardwareV3(1, 16), SingleArrayHardwareV3(16, 1)
        self.assertEqual((a.io_count, b.io_count), (2, 32))
        self.assertGreater(b.area_estimate['total_silicon_mm2'], a.area_estimate['total_silicon_mm2'])
        with self.assertRaises(ValueError): SingleArrayHardwareV3(1, 17)
        with self.assertRaises(ValueError): SingleArrayHardware(1, 3)

    def test_every_architecture_is_reachable_in_one_transaction(self):
        for c in self.cs[1:]:
            mapping = resized_mapping(self.w, self.s, c)
            plan = make_action_plan(self.s, 'A9', [('A9', {'architecture_candidate_id': c.id, 'mapping': mapping.to_dict()})])
            trial = apply_transaction(self.w, self.s, plan, self.cs)
            self.assertEqual(trial.architecture_candidate_id, c.id)
            self.assertEqual(len(trial.mapping.memory_ids), c.hardware.io_count)

    def test_dataflow_action_changes_paid_traffic_and_conserves_macs(self):
        oracle = SingleArrayEvaluator(self.w, self.h)
        base = oracle(self.p).metrics
        plan = make_action_plan(self.s, 'A12', [('A12', {'layer_id': 'conv1', 'dataflow': 'weight_reuse'})], targets=('conv1',))
        trial = apply_transaction(self.w, self.s, plan, self.cs)
        changed = oracle(trial.mapping).metrics
        self.assertEqual(base['executed_macs'], changed['executed_macs'])
        self.assertNotEqual(base['energy_components']['sram_j'], changed['energy_components']['sram_j'])
        bad = make_action_plan(self.s, 'A12', [('A12', {'layer_id': 'conv1', 'dataflow': 'free_bypass'})])
        with self.assertRaises(ValueError): apply_transaction(self.w, self.s, bad, self.cs)

    def test_same_action_family_distinguishes_target_bottlenecks(self):
        a = make_action_plan(self.s, 'A12', [('A12', {'layer_id':'conv0','dataflow':'weight_reuse'})], targets=('conv0',))
        b = make_action_plan(self.s, 'A12', [('A12', {'layer_id':'conv1','dataflow':'weight_reuse'})], targets=('conv1',))
        report = {'latency_s':1.,'diagnostics':{'layers':[
            {'layer_id':'conv0','criticality':1.,'mac_service_s':.8,'memory_pressure':.95},
            {'layer_id':'conv1','criticality':0.,'mac_service_s':.01,'memory_pressure':.1}]}}
        vectors = candidate_features(self.s, [a,b], np.zeros(24), {c.id:c for c in self.cs}, self.w, report)
        self.assertEqual(vectors.shape, (2, ACTION_SIZE_V3))
        self.assertFalse(np.array_equal(vectors[0], vectors[1]))
        blind = candidate_features(self.s, [a,b], np.zeros(24), {c.id:c for c in self.cs}, self.w, {})
        self.assertEqual(blind[0, 180], 0.)
        self.assertEqual(blind[0, 186], 0.)
        self.assertNotEqual(vectors[0, 186], blind[0, 186])

    def test_target_wait_retains_severity_without_parallel_sum_clipping(self):
        from simple_rapidchiplet.hybrid.target_bottleneck import target_features
        report={'latency_s':1.,'diagnostics':{'layers':[
            {'layer_id':'conv0','criticality':1.,'mac_service_s':len(self.p.mapping_map['conv0'].chiplets),
             'transfer_queue_wait_s':10000.,'max_transfer_queue_wait_s':.8},
            {'layer_id':'conv1','criticality':1.,'mac_service_s':len(self.p.mapping_map['conv1'].chiplets)/2,
             'transfer_queue_wait_s':20000.,'max_transfer_queue_wait_s':.2}]}}
        values=[]
        for lid in ('conv0','conv1'):
            plan=make_action_plan(self.s,'A12',[('A12',{'layer_id':lid,'dataflow':'weight_reuse'})],targets=(lid,))
            values.append(target_features(self.s,plan,report,self.w))
        self.assertAlmostEqual(values[0][4],.8)
        self.assertAlmostEqual(values[1][4],.2)
        self.assertAlmostEqual(values[0][1],1.)
        self.assertAlmostEqual(values[1][1],.5)
        jump=make_action_plan(self.s,'A9',[('A9',{'architecture_candidate_id':self.cs[-1].id,'mapping':self.p.to_dict()})],targets=(self.cs[-1].id,))
        self.assertAlmostEqual(target_features(self.s,jump,report,self.w)[4],.8)

    def test_memory_pressure_targets_real_stack_and_headroom_allows_C8(self):
        ids = tuple(layer.id for layer in self.w.layers)
        state = replace(self.s, mapping=replace(self.p, fusion_stacks=(FusionStack(ids,(4,4),'full_cache'),)))
        report = {'diagnostics':{'memory':[{'pool_id':'compute:0:OL2','peak_live_bytes':95,'capacity_bytes':100}]}}
        plans = memory_actions(self.w,state,report)
        self.assertEqual(plans[0].family_id,'C7')
        trial = apply_transaction(self.w,state,plans[0],self.cs)
        self.assertEqual(trial.mapping.fusion_stacks[0].tile_hw,(2,2))
        self.assertEqual(trial.mapping.fusion_stacks[0].overlap,'recompute')
        report['diagnostics']['memory'][0]['pool_id']='compute:99:OL2'
        self.assertFalse(memory_actions(self.w,state,report))
        report={'recompute_macs':100,'diagnostics':{'memory':[{'pool_id':'compute:0:OL2','peak_live_bytes':10,'capacity_bytes':100}]}}
        self.assertEqual(memory_actions(self.w,state,report)[0].family_id,'C8')

    def test_seed_beam_completes_and_has_executable_nonbaseline(self):
        proposals = generate_seeds_v3(self.w,self.h,timeout_s=30.)
        self.assertGreater(len(proposals),1)
        feasible = [SingleArrayEvaluator(self.w,self.h)(p.plan).metrics['feasible'] for p in proposals[1:]]
        self.assertTrue(any(feasible))
        self.assertTrue(all(not p.proxy['formal_PPA'] for p in proposals))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'seed.json'
            first=generate_seeds_v3(self.w,self.h,checkpoint=path)
            replay=generate_seeds_v3(self.w,self.h,checkpoint=path)
            self.assertTrue(replay[0].provenance['DP_complete'])
            self.assertEqual([p.plan.to_dict() for p in first],[p.plan.to_dict() for p in replay])

    def test_observe_act_evaluate_and_update_with_schema3(self):
        report = SingleArrayEvaluator(self.w,self.h)(self.p).metrics
        plans,_ = generate_candidates(self.w,self.s,report,self.cs,shortlist=64)
        self.assertTrue(any(p.family_id=='A12' for p in plans))
        with tempfile.TemporaryDirectory() as tmp:
            result=run_joint_search(self.w,self.s,self.cs,lambda c:SingleArrayEvaluator(self.w,c.hardware),
                config=JointSearchConfig(budget=4,max_steps=3,shortlist=12,method='rl'),
                checkpoint_path=Path(tmp)/'checkpoint.json')
        self.assertTrue(result.trace)
        self.assertEqual(len(result.trace[0]['observation']['selected_candidate_features']), ACTION_SIZE_V3)
        self.assertAlmostEqual(sum(result.trace[0]['candidate_probabilities']),1.)
        self.assertTrue(0 <= result.trace[0]['policy_normalized_entropy'] <= 1.00000001)
        self.assertGreater(result.policy_updates,0)
        self.assertTrue(all(0 < row['behavior_probability'] <= 1 for row in result.trace))

    def test_GUI4_completed_resume_preserves_queries_and_recommendation(self):
        import hybrid_gui
        from single_array_gui import run_suite
        config,_=hybrid_gui.normalize_gui_request({'ui_schema_version':4,'models':['toy'],'search_mode':'fast','budget_mode':'manual','budget':3})
        config.update(seed_wall_time_s=20.,max_wall_time_s=20.,evaluation_timeout_s=15.,seed_dp_seconds=2.)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            first=run_suite(config,folder)
            before={str(p):p.read_bytes() for p in folder.rglob('accounting.json')}
            replay=run_suite(config,folder,resume=True)
            after={str(p):p.read_bytes() for p in folder.rglob('accounting.json')}
            self.assertEqual(before,after)
            self.assertTrue(any(row['metrics'].get('feasible') for row in replay['runs']))
            self.assertEqual(first['best_architecture'],replay['best_architecture'])

    def test_GUI_schema4_reserves_seed_and_search_and_preserves_old(self):
        import hybrid_gui
        new,_=hybrid_gui.normalize_gui_request({'ui_schema_version':4,'models':['toy']})
        old,_=hybrid_gui.normalize_gui_request({'ui_schema_version':3,'models':['toy']})
        self.assertEqual(len(new['architecture_templates']),50)
        self.assertEqual(len(old['architecture_templates']),15)
        self.assertEqual(new['seed_wall_time_s'],7200.)
        self.assertEqual(new['max_wall_time_s'],7200.)
        self.assertEqual(new['evaluation_timeout_s'],1800.)


if __name__ == '__main__': unittest.main()
