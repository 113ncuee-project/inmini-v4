"""Engineering tests for experimental ordered-shard features, no search wins."""
from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest
import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))
from simple_rapidchiplet.hybrid.tensor_locality import TensorLocalityEncoder,NAMES
from simple_rapidchiplet.hybrid.tensor_boundary import TensorBoundaryEncoder
from simple_rapidchiplet.hybrid.mapping import initial_mapping,LayerGroup,Partition
from simple_rapidchiplet.hybrid.workload import Workload,LayerSpec,TensorSpec,make_toy_workload
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.rl_actions import validate_state
from simple_rapidchiplet.hybrid.actions import partition_options


class LocalityTests(unittest.TestCase):
    def setUp(self):
        self.c = next(c for c in enumerate_single_array_architectures_v3() if c.id=='single_array_c2_2x1')
        self.w = Workload('ordered_identity',
            (TensorSpec('x',(1,4,3,3)),TensorSpec('p',(1,4,3,3),producer='p'),TensorSpec('y',(1,4,3,3),producer='q')),
            (LayerSpec('p','Identity',('x',),'p',vector_ops_per_element=1),
             LayerSpec('q','Identity',('p',),'y',vector_ops_per_element=1)),('x',),('y',),1)
        m = initial_mapping(self.w,self.c.hardware.compute_slots,self.c.package.memory_ids)
        slots = ('compute:0/pe:0','compute:1/pe:0')
        m = replace(m,groups=(LayerGroup('all',('p','q'),'temporal'),),layer_mappings=tuple(
            replace(item,chiplets=slots,partition=Partition(1,2,1,1)) for item in m.layer_mappings))
        self.before = DesignState(self.c.id,m,self.w.fingerprint,self.c.hardware.profile_id,version=3)
        self.after = replace(self.before,mapping=m.replace_layer(replace(m.mapping_map['q'],chiplets=tuple(reversed(slots)))))
        validate_state(self.w,self.before,(self.c,))
        validate_state(self.w,self.after,(self.c,))
        self.encoder = TensorLocalityEncoder(self.w,{self.c.id:self.c})

    def test_ordered_ownership_separates_placement_that_old_graph_sketch_aliases(self):
        old = TensorBoundaryEncoder(self.w,{self.c.id:self.c})
        np.testing.assert_array_equal(old.encode(self.before,self.before),old.encode(self.before,self.after))
        before,_,b = self.encoder.sketch(self.before)
        after,_,a = self.encoder.sketch(self.after)
        self.assertEqual(b['raw']['logical_remote_peer_demand'],0)
        self.assertEqual(a['raw']['logical_remote_peer_demand'],36)
        self.assertEqual(sum(a['link_demand']),a['raw']['logical_peer_hop_bytes'])
        self.assertGreater(after[NAMES.index('logical_peer_hop_bytes')],before[NAMES.index('logical_peer_hop_bytes')])
        features = self.encoder.encode(self.before,{},self.after)
        self.assertEqual(len(features),len(self.encoder.names))
        self.assertTrue(np.isfinite(features).all())

    def test_box_flow_matches_scalar_ownership_with_destination_deduplication(self):
        w = make_toy_workload(batch=1,channels=5,spatial=7)
        encoder = TensorLocalityEncoder(w,{self.c.id:self.c})
        rng = random.Random(7)
        for _ in range(16):
            q = rng.choice((1,2,3,4,5,6,8))
            left = rng.choice(partition_options(w,'conv0',q,1))
            right = rng.choice(partition_options(w,'conv1',q,1))
            pslots = tuple(rng.sample(self.c.hardware.compute_slots,q))
            cslots = tuple(rng.sample(self.c.hardware.compute_slots,q))
            flow,demand = encoder.edge(self.c.id,'conv0',left,pslots,'conv1',right,cslots,'conv0')
            expected = np.zeros((2,2))
            seen = set()
            for index,bounds in encoder.requirements('conv1',right,'conv0'):
                destination = int(cslots[index].split('/')[0].split(':')[1])
                if (destination,bounds) in seen:
                    continue
                seen.add((destination,bounds))
                from simple_rapidchiplet.hybrid.mapping import TensorRegion
                request = TensorRegion(bounds)
                for slot,owned in zip(pslots,encoder.owners('conv0',left)[0]):
                    overlap = request.intersection(owned)
                    if overlap:
                        source = int(slot.split('/')[0].split(':')[1])
                        expected[source,destination] += overlap.elements
            np.testing.assert_array_equal(flow.reshape(2,2),expected)
            self.assertEqual(demand,expected.sum())

    def test_group_boundary_and_after_report_do_not_create_fake_peer_or_labels(self):
        groups = (LayerGroup('p_only',('p',),'temporal'),LayerGroup('q_only',('q',),'temporal'))
        split = replace(self.after,mapping=replace(self.after.mapping,groups=groups))
        self.assertEqual(self.encoder.sketch(split)[2]['raw']['logical_peer_demand'],0)
        renamed = replace(self.after,mapping=replace(self.after.mapping,groups=(LayerGroup('renamed',('p','q'),'temporal'),)))
        np.testing.assert_array_equal(self.encoder.sketch(self.after)[0],self.encoder.sketch(renamed)[0])
        attached = replace(self.after,evaluation_report={'latency_s':1e99,'energy_j':1e-99})
        np.testing.assert_array_equal(self.encoder.encode(self.before,{},self.after),self.encoder.encode(self.before,{},attached))
        report = dict(diagnostics=dict(layers=[dict(layer_id='q',criticality=1)]))
        self.assertGreater(self.encoder.encode(self.before,report,self.after)[-1],0)

    def test_order_change_has_a_real_nop_traffic_effect(self):
        from simple_rapidchiplet.hybrid.single_array_evaluator import SingleArrayEvaluator
        oracle = SingleArrayEvaluator(self.w,self.c.hardware)
        before,after = oracle(self.before.mapping).metrics,oracle(self.after.mapping).metrics
        self.assertTrue(before['feasible'] and after['feasible'])
        self.assertGreater(after['nop_hop_bytes'],before['nop_hop_bytes'])
        self.assertEqual(before['executed_macs'],after['executed_macs'])


if __name__=='__main__':
    unittest.main(verbosity=2)
