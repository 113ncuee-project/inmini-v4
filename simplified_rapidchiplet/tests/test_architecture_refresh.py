"""No-label count coverage and shared-resource observations, not DNN wins."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from simple_rapidchiplet.hybrid.architecture_refresh import ArchitectureRefresh
from simple_rapidchiplet.hybrid.full_space import FullDesignSpace, DesignTrial, decision_hash
from simple_rapidchiplet.hybrid.mapping import initial_mapping
from simple_rapidchiplet.hybrid.rl_actions import validate_state
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.service_features import NAMES
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.workload import make_toy_workload


class ArchitectureRefreshTests(unittest.TestCase):
    def setUp(self):
        self.w = make_toy_workload(batch=1, channels=5, spatial=7)
        self.cs = enumerate_single_array_architectures_v3()
        self.c = next(c for c in self.cs if c.id == 'single_array_c8_2x4')
        mapping = initial_mapping(self.w, self.c.hardware.compute_slots, self.c.package.memory_ids, max_group_layers=2)
        self.s = DesignState(self.c.id, mapping, self.w.fingerprint, self.c.hardware.profile_id, version=3)
        self.space = FullDesignSpace(self.w, self.cs, service_features=True)

    def test_all_integer_counts_and_rotations_receive_valid_paired_frames(self):
        refresh = ArchitectureRefresh(4)
        architectures, counts, seen = set(), [], {decision_hash(self.s)}
        for step in range(16):
            rows, evidence = refresh.proposals(self.space, self.s, seen)
            self.assertTrue(rows)
            count = evidence['count']
            counts.append(count)
            for trial in rows:
                self.assertTrue(validate_state(self.w, trial.state, self.cs))
                architecture = self.space.candidates[trial.state.architecture_candidate_id]
                self.assertEqual(architecture.hardware.compute_count, count)
                self.assertEqual(len(trial.state.mapping.memory_ids), 2*architecture.hardware.rows)
                self.assertNotIn(trial.fingerprint, seen)
                architectures.add(architecture.id)
            refresh.finish(step=step, state_hash=rows[0].fingerprint, status='full_completed')
        self.assertEqual(counts[:3], [16, 1, 12])  # maximin over count, not measured costs
        self.assertEqual(set(counts), set(range(1,17)))
        self.assertGreaterEqual(len(architectures), 49)  # exact incumbent may be the only removed no-op
        self.assertEqual(len(refresh.history), 16)

    def test_pending_frame_and_cycle_survive_json_restore_including_timeout(self):
        refresh = ArchitectureRefresh(3)
        rows, _ = refresh.proposals(self.space, self.s)
        restored = ArchitectureRefresh.restore(3, json.loads(json.dumps(refresh.to_dict())))
        restored.finish(step=3, state_hash=rows[0].fingerprint, status='timeout')
        refresh.finish(step=3, state_hash=rows[0].fingerprint, status='timeout')
        self.assertEqual(restored.to_dict(), refresh.to_dict())
        a, _ = refresh.proposals(self.space, self.s)
        b, _ = restored.proposals(self.space, self.s)
        self.assertEqual([r.fingerprint for r in a], [r.fingerprint for r in b])
        self.assertFalse(ArchitectureRefresh().due(999))
        self.assertFalse(refresh.due(1)); self.assertTrue(refresh.due(2))
        with self.assertRaises(ValueError): ArchitectureRefresh.restore(4, restored.to_dict())

    def test_local_service_after_observation_is_before_invariant_and_label_free(self):
        target = self.space.architecture_trial(self.s, self.cs[0])
        metrics = dict(latency_s=.003, energy_j=.01, power_w=3., area_mm2=100.)
        with patch('simple_rapidchiplet.hybrid.single_array_evaluator.SingleArrayEvaluator.__call__',
                   side_effect=AssertionError('full PPA cannot run inside features')):
            a = self.space.features(self.s, metrics, metrics, [target])
            other = {key: value*3 for key,value in metrics.items()}
            b = self.space.features(target.state, other, metrics,
                [DesignTrial(target.state, 'A5', ('conv',))], .8)
        idx = list(self.space.after_design_indices)
        self.assertEqual(len(idx), 81)
        self.assertEqual(a.shape[1], 308)
        np.testing.assert_array_equal(a[:,idx], b[:,idx])
        self.assertTrue(np.isfinite(a).all())
        # Disabling the old boundary reduction must not shift the service
        # after indices onto before/progress/delta observations.
        compact = FullDesignSpace(self.w, self.cs, boundary_features=False, service_features=True)
        self.assertEqual(len(compact.after_design_indices), 50)
        self.assertEqual(compact.features(self.s, metrics, metrics, [target]).shape[1], 197)

    def test_unknown_fitting_is_masked_without_pruning_fusion_or_labels(self):
        encoder = self.space.service_encoder
        with patch.object(encoder.projector, 'geometry', side_effect=ValueError('unknown local fit')):
            values = encoder.sketch(self.s)
        self.assertEqual(values[NAMES.index('geometry_unknown_fraction')], 1.)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(validate_state(self.w, self.s, self.cs))


if __name__ == '__main__': unittest.main()
