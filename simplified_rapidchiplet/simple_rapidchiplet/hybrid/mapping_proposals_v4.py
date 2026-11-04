"""Structured mapping-only actions with a bounded inner loop-order optimizer.

No candidate has a PPA score before the full oracle. Global and critical-window
edits use the same paid hardware, buffer pools and transfer semantics.
"""
from dataclasses import dataclass,replace
from functools import lru_cache
from itertools import product
import math
from types import SimpleNamespace
import numpy as np
from .mapping import LayerGroup,Partition,TensorRegion,enumerate_partitions,validate_plan
from .rl_schema import stable_hash
from .single_array_compiler import _fit_full_channel_tile,tile_requirements
from .single_array_evaluator import SingleArrayEvaluator

FAMILIES=('partition','resource','placement','dram','joint','critical_window')
FLOWS=('output_stationary','weight_reuse','activation_reuse')

@dataclass(frozen=True)
class MappingTrial:
    state: object
    parameters: tuple
    family: str
    targets: tuple
    @property
    def fingerprint(self):return self.state.fingerprint
    def to_dict(self):
        return dict(state=self.state.to_dict(),parameters=self.parameters,family=self.family,targets=self.targets,
            scope='mapping-only; whole ordered mapping transaction; physical evaluation pending')

class StructuredMappings:
    version='structured_mapping_inner_v4_2'
    def __init__(self,workload,candidate,seed,inner=True):
        self.w,self.c,self.seed,self.inner=workload,candidate,seed,inner
        self.h=candidate.hardware;self.slots=self.h.compute_slots;self.memo={};self.geometry_cache={}
        self.local_oracle=SingleArrayEvaluator(workload,self.h)
        qs=sorted({q for q in (1,2,4,8,16,32,64,128,256,len(self.slots)) if q<=len(self.slots)})
        self.domain=list(product(qs,('K','P','balanced'),('contiguous','spread','gateway'),('interleave','nearest','rotate'),('cold','auto','persistent')))

    @lru_cache(maxsize=8192)
    def partition(self,lid,quota,direction):
        layer=self.w.layer_map[lid];shape=self.w.tensor_map[layer.output].shape
        while quota:
            options=enumerate_partitions(self.w,lid,quota,1)
            if options:break
            quota-=1
        if not quota:raise ValueError('no nonempty output partition')
        if direction=='K':key=lambda p:(-p.k,p.h*p.w,p.h,p.w)
        elif direction=='P':key=lambda p:(p.k,shape[2]/p.h+shape[3]/p.w,p.h,p.w)
        else:key=lambda p:(shape[2]/p.h+shape[3]/p.w+shape[1]/p.k,p.h*p.w,p.h,p.w)
        return min(options,key=key)

    @lru_cache(maxsize=8192)
    def inner_flow(self,lid,part):
        layer=self.w.layer_map[lid]
        if not layer.weight_shape:return 'output_stationary'
        shape=self.w.tensor_map[layer.output].shape
        region=TensorRegion(tuple((0,math.ceil(n/p)) for n,p in zip(shape,part)))
        tile=_fit_full_channel_tile(self.w,layer,region,self.h)
        fragment=TensorRegion(tuple((0,n) for n in tile));req=tile_requirements(self.w,layer,fragment)
        scores=[]
        for flow in FLOWS:
            unit=SimpleNamespace(macs=fragment.elements*math.prod(layer.weight_shape[1:]),output_region=fragment,
                weight_bytes=req['weight_bytes'],dataflow=flow)
            k=self.local_oracle._kernel(unit,layer)
            time=max(k['mac_cycles']/self.h.compute_clock_hz,(k['l1_read_bytes']+k['l1_write_bytes'])/self.h.l1_bandwidth_Bps)
            energy=self.local_oracle._sram_j('WL1',read=k['weight_read_bytes'])+self.local_oracle._sram_j('AL1',read=k['activation_read_bytes'])
            energy+=self.local_oracle._sram_j('OL1',read=k['partial_sum_rw_bytes']//2,write=k['partial_sum_rw_bytes']//2)
            scores.append((time*energy,flow))
        return min(scores)[1]

    def order(self,placement):
        if placement=='contiguous':return self.slots
        if placement=='spread':return tuple(f'compute:{d}/pe:{p}' for p in range(16) for d in range(self.h.compute_count))
        gateways=self.h.assumptions.l2_gateway_pes
        def distance(slot):
            pe=int(slot.split(':')[-1]);r,c=divmod(pe,4)
            return min(abs(r-g//4)+abs(c-g%4) for g in gateways),pe,slot
        return tuple(sorted(self.slots,key=distance))

    def retention_policy(self,lid,part,allocated,retention):
        layer=self.w.layer_map[lid]
        if not layer.weight_shape or retention=='cold':return 'cold'
        if retention=='persistent':return 'persistent'
        largest=math.ceil(self.w.tensor_map[layer.output].shape[1]/part.k)*math.prod(layer.weight_shape[1:])
        max_per_die=max(sum(s.split('/')[0]==die for s in allocated) for die in {s.split('/')[0] for s in allocated})
        # This only proposes retention. The complete oracle still checks the
        # actual simultaneous WL2 arrivals and last-reader lifetime peaks.
        panel=min(largest*max_per_die,2*self.h.l1_weight_bytes*max_per_die)
        return 'persistent' if panel<=.8*self.h.l2_weight_bytes else 'cold'

    def build(self,params,base=None,targets=(),family='joint'):
        q,direction,placement,memory,retention=params;base=base or self.seed
        key=(base.fingerprint,params,targets)
        if key in self.memo:return self.memo[key]
        selected=set(targets) if targets else set(self.w.layer_map)
        groups=base.mapping.groups;slots=self.order(placement);changed={}
        for gi,group in enumerate(groups):
            if not selected.intersection(group.layer_ids):continue
            # Spatial groups must be reallocated as a whole: no hidden overlap.
            ids=group.layer_ids if group.mode=='spatial' else tuple(l for l in group.layer_ids if l in selected)
            per=min(q,len(slots)//len(group.layer_ids)) if group.mode=='spatial' else min(q,len(slots))
            cursor=0
            for lid in ids:
                old=base.mapping.mapping_map[lid];layer=self.w.layer_map[lid]
                part=self.partition(lid,per,direction);quota=part.size
                allocated=tuple(slots[cursor:cursor+quota] if group.mode=='spatial' else slots[:quota]);cursor+=quota
                if memory=='interleave':entry='interleave'
                elif memory=='rotate':entry=base.mapping.memory_ids[gi%len(base.mapping.memory_ids)]
                else:
                    rows=[int(s.split('/')[0].split(':')[1])//self.h.cols for s in allocated]
                    row=min(range(self.h.rows),key=lambda r:sum(abs(r-x) for x in rows))
                    entry=f'memory:{2*row}'
                flow=self.inner_flow(lid,part.as_tuple()) if self.inner else old.dataflow
                policy=self.retention_policy(lid,part,allocated,retention)
                changed[lid]=replace(old,chiplets=allocated,partition=part,input_dram=entry,weight_dram=entry,output_dram=entry,
                    dataflow=flow,weight_policy=policy)
        mapping=replace(base.mapping,layer_mappings=tuple(changed.get(m.layer_id,m) for m in base.mapping.layer_mappings))
        validate_plan(self.w,mapping,self.slots)
        state=replace(base,mapping=mapping,evaluation_report=None,remaining_budget=None)
        trial=MappingTrial(state,params,family,tuple(sorted(changed)))
        self.memo[key]=trial
        return trial

    def global_pool(self):
        rows=[];seen=set()
        for params in self.domain:
            try:trial=self.build(params)
            except ValueError:continue
            if trial.fingerprint not in seen and trial.fingerprint!=self.seed.fingerprint:
                rows.append(trial);seen.add(trial.fingerprint)
        return tuple(rows)

    def local_pool(self,base,report,rng,limit=24,guidance=True):
        layers=report.get('diagnostics',{}).get('layers',[])
        ranked=sorted(layers,key=lambda r:-(r.get('criticality',0)*(r.get('max_transfer_queue_wait_s',0)+r.get('compute_busy_s',0)+r.get('l1_service_s',0))))
        order=[l.id for l in self.w.layers];out=[];seen=set()
        if not guidance:ranked=[{'layer_id':lid} for lid in rng.sample(order,len(order))]
        for i in range(limit*2):
            if not ranked:break
            lid=ranked[i%(min(8,len(ranked)) if guidance else len(ranked))]['layer_id'];index=order.index(lid)
            targets=tuple(order[max(0,index-1):min(len(order),index+2)])
            params=rng.choice(self.domain)
            try:trial=self.build(params,base,targets,'critical_window')
            except ValueError:continue
            if trial.fingerprint not in seen and trial.fingerprint!=base.fingerprint:
                out.append(trial);seen.add(trial.fingerprint)
            if len(out)>=limit:break
        # Bound exact mapping memo growth; global proposals remain in caller's bank.
        if len(self.memo)>4096:self.memo.clear()
        return tuple(out)

    def features(self,trial,report,reference,before_cost):
        p=trial.parameters;q,direction,placement,memory,retention=p
        state=trial.state;rows=report.get('diagnostics',{}).get('layers',[]);lat=max(report.get('latency_s',1),1e-30)
        targets=set(trial.targets);measured=[r for r in rows if r['layer_id'] in targets]
        mappings=state.mapping.layer_mappings
        vector=[1.,math.log2(q+1)/6,*[float(direction==v) for v in ('K','P','balanced')],
            *[float(placement==v) for v in ('contiguous','spread','gateway')],*[float(memory==v) for v in ('interleave','nearest','rotate')],
            float(retention=='auto'),float(trial.family=='critical_window'),len(targets)/len(mappings),before_cost,
            float(retention=='persistent')]
        vector += [math.log(max(report.get(k,reference[k]),1e-30)/reference[k]) for k in ('latency_s','energy_j','power_w','area_mm2')]
        for key in ('criticality','memory_pressure','pe_utilization'):
            vector += [max((r.get(key,0) or 0 for r in measured),default=0)]
        allocated={p for row in measured for p in row.get('pe_allocations',())}
        for key in ('mac_service_s','l1_service_s','vector_busy_s','max_transfer_queue_wait_s','max_job_queue_wait_s'):
            if key.startswith('max_'):value=max((r.get(key,0) or 0 for r in measured),default=0.)/lat
            else:
                parallel=max(1,len({p.split('/')[0] for p in allocated}) if key=='vector_busy_s' else len(allocated))
                value=sum(r.get(key,0) or 0 for r in measured)/(lat*parallel)
            vector += [min(5.,value)]
        quotas=[len(m.chiplets) for m in mappings]
        vector += [sum(quotas)/(len(quotas)*len(self.slots)),min(quotas)/len(self.slots),max(quotas)/len(self.slots)]
        vector += [sum(m.partition.k>1 for m in mappings)/len(mappings),sum(m.partition.h*m.partition.w>1 for m in mappings)/len(mappings),
            sum(m.weight_policy=='persistent' for m in mappings)/len(mappings)]
        # Explicit interactions let the learned Q distinguish a quota's effect
        # under partition/placement/retention; no hidden candidate score enters.
        vector += [vector[1]*v for v in vector[2:13]]
        return np.clip(np.asarray(vector,dtype=np.float32),-5,5)
