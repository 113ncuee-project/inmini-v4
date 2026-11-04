"""Experimental ordered-Part/CG tensor ownership features; never a PPA model.

Pure shard-box algebra exposes which producer die owns each requested value.
Identical region requests to the same die are deduplicated; overlapping unequal
requests remain separate demands. This ignores executable tiling, generation,
cache/multicast lifetimes and scheduling, so it cannot supply a cost or reward.
"""
from functools import lru_cache
import math
import numpy as np

from simple_rapidchiplet.hybrid.mapping import Partition,TensorRegion,_shards,required_input_regions

NAMES = ('logical_peer_demand','logical_remote_peer_demand','logical_peer_hop_bytes',
         'logical_peak_nop_link_demand','logical_mac_input_gateway_demand',
         'logical_mac_output_gateway_demand','logical_mac_weight_gateway_demand')


class TensorLocalityEncoder:
    version = 'ordered_shard_tensor_locality_experimental_v1'

    def __init__(self,workload,candidates):
        if workload.batch_size!=1:
            raise ValueError('logical locality features follow the schema3 Batch=1 contract')
        self.w,self.candidates = workload,candidates
        self.bytes = max(1,sum(t.bytes for t in workload.tensors))
        self.weights = max(1,sum(l.weight_bytes for l in workload.layers))
        self.cache = {}

    @lru_cache(maxsize=2048)
    def owners(self,lid,part):
        layer = self.w.layer_map[lid]
        regions = _shards(self.w.tensor_map[layer.output].shape,Partition(*part),0,1)
        return regions,np.asarray([r.bounds for r in regions],dtype=np.int64)

    @lru_cache(maxsize=4096)
    def requirements(self,lid,part,tid):
        rows = []
        for index,region in enumerate(self.owners(lid,part)[0]):
            rows.extend((index,req.region.bounds) for req in required_input_regions(self.w,self.w.layer_map[lid],region)
                        if req.tensor_id==tid)
        return tuple(rows)

    @lru_cache(maxsize=50)
    def routes(self,architecture):
        c = self.candidates[architecture]
        count = c.hardware.compute_count
        routes = {(a,b):tuple(zip(path,path[1:])) for a in range(count) for b in range(count)
                  for path in (c.package.route(f'compute:{a}',f'compute:{b}'),)}
        edges = sorted({edge for path in routes.values() for edge in path})
        lookup = {edge:i for i,edge in enumerate(edges)}
        incidence = np.zeros((count*count,len(edges)),dtype=np.float64)
        for (a,b),path in routes.items():
            for edge in path:
                incidence[a*count+b,lookup[edge]] += 1
        return incidence,tuple(edges)

    @staticmethod
    def dies(slots):
        return np.asarray([int(s.split('/')[0].split(':')[1]) for s in slots],dtype=np.int64)

    @lru_cache(maxsize=4096)
    def edge(self,architecture,producer,p_part,p_slots,consumer,c_part,c_slots,tid):
        h = self.candidates[architecture].hardware
        p_regions = self.owners(producer,p_part)[1]
        destinations = self.dies(c_slots)
        # Deduplicate identical AL2 destination demands, not different tensors,
        # source owners or unrelated partial overlaps.
        unique = tuple(dict.fromkeys((int(destinations[index]),bounds)
                        for index,bounds in self.requirements(consumer,c_part,tid)))
        if not unique:
            return np.zeros(h.compute_count*h.compute_count),0.
        requested = np.asarray([bounds for _,bounds in unique],dtype=np.int64)
        low = np.maximum(requested[:,None,:,0],p_regions[None,:,:,0])
        high = np.minimum(requested[:,None,:,1],p_regions[None,:,:,1])
        amounts = np.prod(np.maximum(0,high-low),axis=-1)*self.w.tensor_map[tid].bytes_per_element
        expected = np.prod(requested[:,:,1]-requested[:,:,0],axis=-1)*self.w.tensor_map[tid].bytes_per_element
        if not np.array_equal(amounts.sum(axis=1),expected):
            raise ValueError('logical producer ownership did not cover a required tensor box')
        flow = np.zeros((h.compute_count,h.compute_count),dtype=np.float64)
        sources = self.dies(p_slots)
        for index,(destination,_) in enumerate(unique):
            np.add.at(flow[:,destination],sources,amounts[index])
        return flow.reshape(-1),float(amounts.sum())

    @lru_cache(maxsize=4096)
    def local(self,architecture,lid,part,slots):
        layer = self.w.layer_map[lid]
        # Vector jobs execute in the shared die VPU, not fictitious PE L1s.
        if not layer.weight_shape:
            return (0.,0.,0.)
        h = self.candidates[architecture].hardware
        gateways = h.assumptions.l2_gateway_pes
        inputs,outputs,weights = 0.,0.,0.
        regions = self.owners(lid,part)[0]
        for slot,region in zip(slots,regions):
            pe = int(slot.split(':')[-1])
            r,c = divmod(pe,4)
            distance = min(abs(r-g//4)+abs(c-g%4) for g in gateways)
            inputs += sum(req.region.elements*self.w.tensor_map[req.tensor_id].bytes_per_element
                          for req in required_input_regions(self.w,layer,region))*distance
            outputs += region.elements*h.assumptions.psum_storage_bytes*distance
            weights += region.shape[1]*math.prod(layer.weight_shape[1:])*(layer.weight_bits//8)*distance
        return inputs,outputs,weights

    def sketch(self,state):
        key = state.fingerprint
        if key in self.cache:
            return self.cache[key]
        architecture = state.architecture_candidate_id
        h = self.candidates[architecture].hardware
        maps,groups = state.mapping.mapping_map,state.mapping.group_map
        incidence,_ = self.routes(architecture)
        flows = np.zeros(h.compute_count*h.compute_count)
        total = 0.
        edges = {}
        local = np.zeros(3)
        for layer in self.w.layers:
            m = maps[layer.id]
            local += self.local(architecture,layer.id,m.partition.as_tuple(),m.chiplets)
            for tid in layer.inputs:
                tensor = self.w.tensor_map[tid]
                producer = tensor.producer
                # A materialized edge uses its separate DRAM-boundary features;
                # it is not falsely modelled as an unpriced direct peer route.
                if not producer or groups[producer].id!=groups[layer.id].id:
                    continue
                previous = maps[producer]
                flow,demand = self.edge(architecture,producer,previous.partition.as_tuple(),previous.chiplets,
                    layer.id,m.partition.as_tuple(),m.chiplets,tid)
                loads = flow@incidence
                edges[(producer,layer.id,tid)] = float(loads.sum())
                flows += flow
                total += demand
        matrix = flows.reshape(h.compute_count,h.compute_count)
        link_loads = flows@incidence
        raw = (total,total-float(np.trace(matrix)),float(link_loads.sum()),
               float(link_loads.max()) if len(link_loads) else 0.,*local)
        # Named positive scales, with log compression for replicated logical
        # demand. These are observations/features, never predicted PPA labels.
        scales = np.asarray((self.bytes,self.bytes,self.bytes*8,self.bytes,
                             self.bytes*8,self.bytes*8,self.weights*8))
        values = np.log1p(np.asarray(raw)/scales)
        result = values,edges,dict(raw=dict(zip(NAMES,raw)),link_demand=tuple(link_loads))
        if len(self.cache)>1024:
            self.cache.clear()
        self.cache[key] = result
        return result

    @property
    def names(self):
        return tuple(prefix+name for prefix in ('before_','after_','delta_') for name in NAMES)+('critical_peer_hop_delta',)

    def encode(self,before,report,after):
        b,be,_ = self.sketch(before)
        a,ae,_ = self.sketch(after)
        evidence = {r['layer_id']:r for r in report.get('diagnostics',{}).get('layers',())}
        critical = sum((ae.get(edge,0.)-be.get(edge,0.))*max(
            float(evidence.get(edge[0],{}).get('criticality',0) or 0),
            float(evidence.get(edge[1],{}).get('criticality',0) or 0)) for edge in be.keys()|ae.keys())/(self.bytes*8)
        return np.clip(np.asarray((*b,*a,*(a-b),critical),dtype=np.float32),-8,8)
