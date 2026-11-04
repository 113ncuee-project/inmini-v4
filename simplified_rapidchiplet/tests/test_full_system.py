"""Full-domain semantics and actual multi-step replay, independent of DNN wins."""
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from simple_rapidchiplet.hybrid.full_space import FullDesignSpace, DesignTrial
from simple_rapidchiplet.hybrid.mapping import initial_mapping
from simple_rapidchiplet.hybrid.multi_step_q import MultiStepQ
from simple_rapidchiplet.hybrid.rl_actions import apply_transaction, validate_state
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.single_array_evaluator import SingleArrayEvaluator
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.workload import make_toy_workload


class FullSystemTests(unittest.TestCase):
    def setUp(self):
        self.w = make_toy_workload(batch=1, channels=5, spatial=7)
        self.cs = enumerate_single_array_architectures_v3()
        self.c = next(c for c in self.cs if c.id == 'single_array_c2_2x1')
        mapping = initial_mapping(self.w, self.c.hardware.compute_slots, self.c.package.memory_ids, max_group_layers=2)
        self.s = DesignState(self.c.id, mapping, self.w.fingerprint, self.c.hardware.profile_id, version=3)
        self.sp = FullDesignSpace(self.w, self.cs, max_depth=4)

    def test_every_architecture_one_step_preserves_resources_and_contract(self):
        self.assertEqual(len(self.cs), 50)
        self.assertEqual({c.hardware.compute_count for c in self.cs}, set(range(1, 17)))
        for c in self.cs:
            trial = self.sp.architecture_trial(self.s, c)
            self.assertTrue(validate_state(self.w, trial.state, self.cs))
            self.assertEqual(len(trial.state.mapping.memory_ids), 2 * c.hardware.rows)
            self.assertEqual(trial.state.model_hash, self.s.model_hash)
            self.assertEqual(sum(len(g.layer_ids) for g in trial.state.mapping.groups), len(self.w.layers))

    def test_lazy_transactions_use_original_v3_validator(self):
        families = set()
        for seed in range(4):
            plans = self.sp._sample_transactions(self.s, {}, random.Random(seed))
            for p in plans:
                try:
                    state = apply_transaction(self.w, self.s, p, self.cs)
                except ValueError:
                    continue
                self.assertTrue(validate_state(self.w, state, self.cs))
                families.add(p.family_id)
        self.assertTrue({'A2', 'A4', 'A5', 'A6', 'A8', 'A10', 'A12'}.issubset(families), families)

    def test_proposals_are_deterministic_and_have_no_oracle_labels(self):
        a, stats = self.sp.propose(self.s, {}, random.Random(8))
        other = FullDesignSpace(self.w, self.cs)
        b, _ = other.propose(self.s, {}, random.Random(8))
        self.assertEqual([t.fingerprint for t in a], [t.fingerprint for t in b])
        self.assertEqual(stats['full_ppa'], 0)
        self.assertTrue({'architecture_mapping', 'global_mapping', 'fusion_beam'}.issubset({t.family for t in a}))
        self.assertTrue(all(validate_state(self.w, t.state, self.cs) for t in a))

    def test_fusion_tile_and_cache_are_live_and_costed(self):
        self.sp.calls = 1
        trials = self.sp._fusion_beam(self.s, random.Random(7), 20)
        self.assertTrue(trials)
        state = trials[0].state
        self.assertTrue(state.mapping.fusion_stacks)
        families = set()
        for p in self.sp._sample_transactions(state, {}, random.Random(3)):
            try:
                next_state = apply_transaction(self.w, state, p, self.cs)
                families.add(p.family_id)
                validate_state(self.w, next_state, self.cs)
            except ValueError:
                pass
        self.assertTrue({'A1', 'A3', 'A7'}.issubset(families))
        oracle = SingleArrayEvaluator(self.w, self.c.hardware)
        off = oracle(self.s.mapping).metrics
        on = oracle(state.mapping).metrics
        self.assertGreater(on['executed_macs'], 0)
        self.assertGreater(on['energy_j'], 0)
        self.assertGreaterEqual(on['executed_macs'], off['executed_macs'])
        self.assertIn('recompute_macs', on)

    def test_features_describe_different_architecture_fusion_and_placement(self):
        trials, _ = self.sp.propose(self.s, {}, random.Random(11))
        metrics = dict(latency_s=1., energy_j=1., area_mm2=1., power_w=1.)
        f = self.sp.features(self.s, metrics, metrics, trials)
        self.assertEqual(f.shape[0], len(trials))
        self.assertTrue(np.isfinite(f).all())
        self.assertGreater(np.unique(f, axis=0).shape[0], len(trials) // 2)
        a, b = tuple(self.w.layer_map)[:2]
        context = self.sp.features(self.s, metrics, metrics, [DesignTrial(self.s, 'A2', (a,)), DesignTrial(self.s, 'A2', (b,))])
        self.assertFalse(np.array_equal(context[0], context[1]))
        selected = min(3, len(trials)-1)
        single = self.sp.features(self.s,metrics,metrics,[trials[selected]])
        np.testing.assert_array_equal(f[selected],single[0])

    def test_batch_and_unsupported_routing_keep_v3_limits(self):
        with self.assertRaises(ValueError):
            validate_state(self.w, replace(self.s, mapping=replace(self.s.mapping, microbatch_size=2)), self.cs)
        with self.assertRaises(ValueError):
            validate_state(self.w, replace(self.s, routing_config={'algorithm': 'free'}), self.cs)

    def test_generation_survives_diverse_full_domain_walk(self):
        rng = random.Random(51)
        state = self.s
        for _ in range(16):
            trials, _ = self.sp.propose(state, {}, rng)
            self.assertTrue(trials)
            state = rng.choice(trials).state
            self.assertTrue(validate_state(self.w, state, self.cs))

    def test_double_q_trains_target_and_recovers_exactly(self):
        model = MultiStepQ(4, seed=7, gamma=.9)
        a, b = np.array([1, 0, 0, 0], np.float32), np.array([0, 1, 0, 0], np.float32)
        for _ in range(4):
            model.observe(a, 2., np.stack([a, b]), terminal=False, updates=24)
            model.observe(b, -2., np.stack([a, b]), terminal=True, updates=24)
        p, q = model.probabilities(np.stack([a, b]), .1)
        self.assertGreater(q[0], q[1])
        self.assertGreater(p[0], .9)
        self.assertTrue(any(not row[3] for row in model.replay))
        self.assertTrue(any(row[3] for row in model.replay))
        temporary_root = Path(__file__).resolve().parents[2] / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            path = Path(folder) / 'policy.pt'
            model.save(path)
            restored = MultiStepQ(4, seed=1)
            restored.load(path)
            np.testing.assert_array_equal(model.predict(np.stack([a, b])), restored.predict(np.stack([a, b])))
            model.observe(b, 3., np.stack([a]), False, 4)
            restored.observe(b, 3., np.stack([a]), False, 4)
            np.testing.assert_array_equal(model.predict(np.stack([a, b])), restored.predict(np.stack([a, b])))


if __name__ == '__main__':
    unittest.main()
