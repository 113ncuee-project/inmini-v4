"""Graph edit semantics and conserving projection, independent of search wins."""
from dataclasses import replace
from itertools import product
import random
import tempfile
from pathlib import Path
import unittest

import numpy as np

from simple_rapidchiplet.hybrid.full_space import FullDesignSpace, DesignTrial, decision_hash
from simple_rapidchiplet.hybrid.mapping import initial_mapping, LayerGroup, FusionStack
from simple_rapidchiplet.hybrid.multi_step_q import MultiStepQ
from simple_rapidchiplet.hybrid.lp_five import propose_five
from simple_rapidchiplet.hybrid.resource_projection import integer_projection, ResourceProjector
from simple_rapidchiplet.hybrid.temporal_projection import TemporalProjector
from simple_rapidchiplet.hybrid.rl_schema import ActionEdit, make_action_plan
from simple_rapidchiplet.hybrid.rl_actions import apply_transaction
from simple_rapidchiplet.hybrid.shared_evidence import verify_physical_sources,common_query_count
from simple_rapidchiplet.hybrid.rl_actions import validate_state
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.single_array_evaluator import SingleArrayEvaluator
from simple_rapidchiplet.hybrid.tensor_boundary import TensorBoundaryEncoder, NAMES
from simple_rapidchiplet.hybrid.workload import make_toy_workload, TensorSpec, LayerSpec, Workload


class GraphEditTests(unittest.TestCase):
    def setUp(self):
        self.w = make_toy_workload(batch=1, channels=5, spatial=7)
        self.cs = enumerate_single_array_architectures_v3()
        self.c = next(c for c in self.cs if c.id == 'single_array_c2_2x1')
        mapping = initial_mapping(self.w, self.c.hardware.compute_slots, self.c.package.memory_ids, max_group_layers=1)
        self.s = DesignState(self.c.id, mapping, self.w.fingerprint, self.c.hardware.profile_id, version=3)
        self.encoder = TensorBoundaryEncoder(self.w, {c.id: c for c in self.cs})

    def test_cut_reads_and_writes_are_distinct_and_merge_removes_both(self):
        before, cuts = self.encoder.sketch(self.s)
        amount = self.w.tensor_map['conv0'].bytes
        self.assertEqual(set(cuts), {('conv0', 'conv1'), ('conv1', 'add')})
        groups = (LayerGroup('all', tuple(self.w.layer_map), 'temporal'),)
        after = replace(self.s, mapping=replace(self.s.mapping, groups=groups))
        encoded = self.encoder.encode(self.s, after)
        self.assertEqual(len(encoded), len(self.encoder.names))
        self.assertAlmostEqual(encoded[-3], 2 * amount / self.encoder.activation_bytes)
        self.assertEqual(encoded[-2], 0.)
        self.assertEqual(encoded[-1], 1.)
        b, _ = self.encoder.sketch(after)
        self.assertLess(b[NAMES.index('boundary_write')], before[NAMES.index('boundary_write')])
        self.assertEqual(b[NAMES.index('cross_group_read')], 0.)

    def test_shared_residual_has_one_write_and_separate_remote_read_edges(self):
        t = TensorSpec
        l = LayerSpec
        w = Workload('fanout', (t('x', (1, 2, 3, 3)), t('p', (1, 2, 3, 3), producer='p'),
            t('a', (1, 2, 3, 3), producer='a'), t('b', (1, 2, 3, 3), producer='b'),
            t('y', (1, 2, 3, 3), producer='join')), (
            l('p', 'Identity', ('x',), 'p'), l('a', 'Identity', ('p',), 'a'),
            l('b', 'Identity', ('p',), 'b'), l('join', 'Add', ('a', 'b'), 'y')), ('x',), ('y',), 1)
        w.validate()
        m = initial_mapping(w, self.c.hardware.compute_slots, self.c.package.memory_ids, max_group_layers=1)
        s = replace(self.s, model_hash=w.fingerprint, mapping=m)
        e = TensorBoundaryEncoder(w, {self.c.id: self.c})
        v, cuts = e.sketch(s)
        self.assertEqual(set(cuts), {('p', 'a'), ('p', 'b'), ('a', 'join'), ('b', 'join')})
        self.assertAlmostEqual(v[NAMES.index('cross_group_read')], 4 / 5)
        self.assertAlmostEqual(v[NAMES.index('boundary_write')], 4 / 5)
        self.assertAlmostEqual(v[NAMES.index('residual_cross_read')], 2 / 5)
        # Merge only p/a: p still materializes once for remote b; local a edge
        # must survive as internal traffic instead of disappearing from the DAG.
        m2 = replace(m, groups=(LayerGroup('pa', ('p', 'a'), 'temporal'), *m.groups[2:]))
        v2, c2 = e.sketch(replace(s, mapping=m2))
        self.assertEqual(set(c2), {('p', 'b'), ('a', 'join'), ('b', 'join')})
        self.assertEqual(v2[NAMES.index('boundary_write')], v[NAMES.index('boundary_write')])
        self.assertGreater(v2[NAMES.index('internal_edge_bytes')], 0)

    def test_features_are_unchanged_by_group_id_rename_and_identity_has_zero_delta(self):
        renamed = replace(self.s, mapping=replace(self.s.mapping, groups=tuple(
            replace(g, id='renamed_' + str(i)) for i, g in enumerate(self.s.mapping.groups))))
        a, _ = self.encoder.sketch(self.s)
        b, _ = self.encoder.sketch(renamed)
        np.testing.assert_array_equal(a, b)
        encoded = self.encoder.encode(self.s, renamed)
        np.testing.assert_array_equal(encoded[2 * len(NAMES):], np.zeros(len(NAMES) + 3))
        self.assertEqual(decision_hash(self.s), decision_hash(renamed))
        self.assertNotEqual(self.s.fingerprint, renamed.fingerprint)

    def test_group_labels_do_not_change_physical_ppa(self):
        oracle = SingleArrayEvaluator(self.w, self.c.hardware)
        for state in (self.s, replace(self.s, mapping=replace(self.s.mapping, groups=(
                LayerGroup('first', ('conv0', 'conv1'), 'temporal'), self.s.mapping.groups[-1])))):
            before = oracle(state.mapping).metrics
            for names in (('z', 'a', 'b'), ('group19', 'group2', 'group1')):
                renamed = replace(state.mapping, groups=tuple(replace(g, id=names[i]) for i, g in enumerate(state.mapping.groups)))
                after = oracle(renamed).metrics
                for key in ('latency_s', 'energy_j', 'power_w', 'area_mm2', 'dram_read_bytes', 'dram_write_bytes', 'noc_hop_bytes', 'nop_hop_bytes', 'executed_macs'):
                    self.assertEqual(before[key], after[key], (key, names))

    def test_integer_projection_matches_exhaustive_and_allows_idle(self):
        rng = random.Random(7)
        for _ in range(25):
            choices = [tuple((q, rng.randint(0, 100) / 7, str(q)) for q in range(1, 5)) for _ in range(3)]
            result = integer_projection(choices, 7)
            expected = min((sum(x[1] for x in row), sum(x[0] for x in row)) for row in product(*choices) if sum(x[0] for x in row) <= 7)
            self.assertAlmostEqual(result[1], expected[0])
            self.assertEqual(result[2], expected[1])
        idle = integer_projection([((1, 1., 'one'), (3, 1., 'three'))], 8)
        self.assertEqual(idle[2], 1)
        prime = integer_projection([((3, 0., 'a'),), ((5, 0., 'b'),)], 8)
        self.assertEqual(prime[2], 8)
        self.assertIsNone(integer_projection([((3, 0., None),), ((5, 0., None),)], 7))

    def test_projection_repairs_overlapping_shadow_and_preserves_fd_flow(self):
        groups = (LayerGroup('joint', tuple(self.w.layer_map), 'spatial'),)
        shadow = replace(self.s, mapping=replace(self.s.mapping, groups=groups))
        p = ResourceProjector(self.w, {self.c.id: self.c})
        projected, evidence = p.project(shadow, ('joint',))
        self.assertTrue(validate_state(self.w, projected, self.cs))
        owners = [s for m in projected.mapping.layer_mappings for s in m.chiplets]
        self.assertEqual(len(owners), len(set(owners)))
        self.assertLessEqual(len(owners), len(self.c.hardware.compute_slots))
        self.assertFalse(evidence['proxy_is_full_ppa'])
        for before, after in zip(self.s.mapping.layer_mappings, projected.mapping.layer_mappings):
            self.assertEqual(before.fd, after.fd)
            self.assertEqual(before.dataflow, after.dataflow)
            self.assertEqual(before.weight_policy, after.weight_policy)
            self.assertEqual(len(after.chiplets), after.partition.size)

    def test_full_space_and_separate_feature_proposal_ablations(self):
        base = FullDesignSpace(self.w, self.cs, resource_projection=False, boundary_features=False)
        graph = FullDesignSpace(self.w, self.cs, resource_projection=False, boundary_features=True)
        a, _ = base.propose(self.s, {}, random.Random(8))
        b, _ = graph.propose(self.s, {}, random.Random(8))
        self.assertEqual([t.fingerprint for t in a], [t.fingerprint for t in b])
        metrics = dict(latency_s=1., energy_j=1., power_w=1., area_mm2=1.)
        plain = base.features(self.s, metrics, metrics, a)
        encoded = graph.features(self.s, metrics, metrics, b)
        self.assertEqual(encoded.shape[1] - plain.shape[1], len(graph.graph_feature_names))
        self.assertTrue(np.isfinite(encoded).all())
        np.testing.assert_array_equal(encoded[:, :plain.shape[1]], plain)
        projected = graph._projected_groups(self.s, random.Random(7))
        self.assertTrue(projected)
        for trial in projected:
            self.assertTrue(validate_state(self.w, trial.state, self.cs))
        self.assertEqual(len(graph.domain()['architectures']), 50)
        base_policy = MultiStepQ(plain.shape[1], seed=7)
        graph_policy = MultiStepQ(encoded.shape[1], seed=7, base_size=plain.shape[1])
        np.testing.assert_array_equal(base_policy.predict(plain), graph_policy.predict(encoded))

    def test_shared_reuse_rejects_changed_physical_model(self):
        before = {'simple_rapidchiplet/hybrid/single_array_evaluator.py': 'physical',
                  'simple_rapidchiplet/hybrid/mapping.py': 'ownership',
                  'simple_rapidchiplet/hybrid/full_space.py': 'controller-v5'}
        after = {**before, 'simple_rapidchiplet/hybrid/full_space.py': 'controller-v6',
                 'simple_rapidchiplet/hybrid/tensor_boundary.py': 'label-free'}
        self.assertEqual(len(verify_physical_sources(before, after)), 2)
        for key in ('simple_rapidchiplet/hybrid/mapping.py', 'simple_rapidchiplet/hybrid/single_array_evaluator.py'):
            with self.assertRaisesRegex(ValueError, 'physical source mismatch'):
                verify_physical_sources(before, {**after, key: 'changed'})

    def test_common_reference_and_seed_cache_hit_is_charged_once(self):
        initial = self.s.to_dict()
        self.assertEqual(common_query_count(initial,initial),1)
        labelled = replace(self.s,evaluation_report={'derived':'does not create another physical request'}).to_dict()
        self.assertEqual(common_query_count(initial,labelled),1)
        changed = replace(self.s,mapping=self.s.mapping.replace_layer(replace(
            self.s.mapping.mapping_map['conv0'],dataflow='weight_reuse'))).to_dict()
        self.assertEqual(common_query_count(initial,changed),2)

    def test_graph_ablation_has_identical_initial_scores_then_learns(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(8, 149)).astype(np.float32)
        graph_features = rng.normal(size=(8, 75)).astype(np.float32)
        augmented = np.concatenate((x, graph_features), axis=1)
        base = MultiStepQ(149, seed=7)
        graph = MultiStepQ(224, seed=7, base_size=149)
        np.testing.assert_allclose(base.predict(x), graph.predict(augmented), rtol=0, atol=0)
        np.testing.assert_allclose(base.predict_reward(x), graph.predict_reward(augmented), rtol=0, atol=0)
        before = graph.model.graph[-1].weight.detach().numpy().copy()
        graph.observe(augmented[0], 1., augmented, updates=4)
        self.assertGreater(np.max(np.abs(graph.model.graph[-1].weight.detach().numpy() - before)), 0)
        self.assertTrue(np.isfinite(graph.predict(augmented)).all())
        temporary_root = Path(__file__).resolve().parents[1].parent / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            path = Path(folder) / 'graph.pt'
            graph.save(path)
            restored = MultiStepQ(224, seed=37)
            restored.load(path)
            np.testing.assert_array_equal(graph.predict(augmented), restored.predict(augmented))
            graph.observe(augmented[1], -.5, augmented, updates=2)
            restored.observe(augmented[1], -.5, augmented, updates=2)
            np.testing.assert_array_equal(graph.predict(augmented), restored.predict(augmented))

    def test_gemini_five_comparison_holds_other_decisions_fixed(self):
        groups = (LayerGroup('all', tuple(self.w.layer_map), 'spatial'),)
        shadow = replace(self.s, mapping=replace(self.s.mapping, groups=groups))
        projected, _ = ResourceProjector(self.w, {self.c.id: self.c}).project(shadow, ('all',))
        rows = propose_five(self.w, projected, self.c, random.Random(17))
        self.assertEqual({r[3]['mapping_action']['operator'] for r in rows}, {'OP1', 'OP2', 'OP3', 'OP4', 'OP5'})
        for state, _, _, action in rows:
            self.assertTrue(validate_state(self.w, state, self.cs))
            self.assertEqual(state.mapping.groups, projected.mapping.groups)
            self.assertEqual(state.mapping.fusion_stacks, projected.mapping.fusion_stacks)
            self.assertEqual(state.architecture_candidate_id, projected.architecture_candidate_id)
            for a, b in zip(state.mapping.layer_mappings, projected.mapping.layer_mappings):
                self.assertEqual(a.dataflow, b.dataflow)
                self.assertEqual(a.weight_policy, b.weight_policy)
            if action['mapping_action']['operator'] == 'OP4':
                self.assertEqual({p for m in state.mapping.layer_mappings for p in m.chiplets},
                                 {p for m in projected.mapping.layer_mappings for p in m.chiplets})

    def test_temporal_projection_reuses_paid_pes_without_spatial_quota_rule(self):
        groups = (LayerGroup('serial', tuple(self.w.layer_map), 'temporal'),)
        base = replace(self.s, mapping=replace(self.s.mapping, groups=groups))
        space = FullDesignSpace(self.w, self.cs)
        projector = TemporalProjector(self.w, {self.c.id: self.c})
        result, evidence = projector.project(base, 'serial', space.bank(base), dict(latency_s=.01, energy_j=.01), cap=16)
        self.assertTrue(validate_state(self.w, result, self.cs))
        self.assertFalse(evidence['proxy_is_full_ppa'])
        self.assertEqual(result.mapping.groups, base.mapping.groups)
        self.assertEqual(projector.oracle(self.c.id).evaluations, 0)
        for a, b in zip(base.mapping.layer_mappings, result.mapping.layer_mappings):
            self.assertEqual(a.fd, b.fd)
            self.assertLessEqual(len(b.chiplets), 16)
            self.assertEqual(len(b.chiplets), b.partition.size)
        self.assertTrue(all(r['time_s'] > 0 and r['energy_j'] > 0 for r in evidence['allocations']))

    def test_internal_fd_edit_cannot_move_unrelated_external_residual_input(self):
        before = self.s.mapping.mapping_map['add'].input_dram
        memory = next(m for m in self.s.mapping.memory_ids if m != before)
        action = make_action_plan(self.s, 'A8', [ActionEdit('A8', dict(tensor_id='conv1', configuration_id=memory))])
        state = apply_transaction(self.w, self.s, action, self.cs)
        self.assertEqual(state.mapping.mapping_map['conv1'].output_dram, memory)
        self.assertEqual(state.mapping.mapping_map['add'].input_dram, before)

    def test_fusion_tile_can_reach_odd_and_rectangular_integer_shapes(self):
        state = replace(self.s, mapping=replace(self.s.mapping,
            groups=(LayerGroup('fused', ('conv0', 'conv1'), 'temporal'), self.s.mapping.groups[-1]),
            fusion_stacks=(FusionStack(('conv0', 'conv1'), (4, 6), 'recompute'),)))
        validate_state(self.w, state, self.cs)
        space = FullDesignSpace(self.w, self.cs, resource_projection=False)
        shapes = set()
        for action in space._sample_transactions(state, {}, random.Random(7)):
            if action.family_id != 'A3':
                continue
            result = apply_transaction(self.w, state, action, self.cs)
            shape = result.mapping.fusion_stacks[0].tile_hw
            self.assertTrue(all(1 <= size <= 7 for size in shape))
            shapes.add(shape)
        self.assertTrue({(3, 6), (5, 6), (4, 5), (4, 7)}.issubset(shapes))

    def test_observed_context_is_from_before_schedule_only(self):
        after = replace(self.s, mapping=self.s.mapping.replace_layer(replace(self.s.mapping.mapping_map['conv0'], dataflow='weight_reuse')))
        report = dict(latency_s=.01, diagnostics=dict(layers=[dict(layer_id='conv0', criticality=1., mac_service_s=.02,
            l1_service_s=.01, vector_busy_s=.002, memory_pressure=.9, max_job_queue_wait_s=.001)]))
        values = self.encoder.observed_edit(self.s, report, after, ('conv0',))
        self.assertEqual(values[0], 1.)
        self.assertGreater(values[1], 0.)
        self.assertGreater(values[11], 0.)
        self.assertEqual(values[7], 0.)


if __name__ == '__main__':
    unittest.main()
