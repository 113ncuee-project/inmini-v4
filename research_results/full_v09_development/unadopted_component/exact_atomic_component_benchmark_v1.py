"""Exact communication partition prototype; synthetic component QA, zero PPA."""
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simplified_rapidchiplet'))
from simple_rapidchiplet.hybrid.communication import _atomic_demands,_Demand
from simple_rapidchiplet.hybrid.mapping import TensorRegion


def atomic_bitset(demands):
    groups={}
    for d in demands:
        key=(d.src,d.tensor,d.bits,d.microbatch,d.producer_units,d.kind,d.dependencies,d.reload_cohort)
        groups.setdefault(key,[]).append(d)
    result=[]
    for key,selected in groups.items():
        src,tensor,bits,microbatch,producers,kind,dependencies,_=key
        def emit(bounds,indices):
            remote=[selected[i] for i in indices if selected[i].dst!=src]
            if remote:
                region=TensorRegion(tuple(bounds))
                result.append((src,tuple(dict.fromkeys(d.dst for d in remote)),region.elements*bits//8,
                    producers,tuple(dict.fromkeys(d.consumer_unit for d in remote)),kind,tensor,region,microbatch,dependencies))
        if len(selected)==1:
            emit(selected[0].region.bounds,[0]);continue
        cuts=[sorted({v for d in selected for v in d.region.bounds[a]}) for a in range(4)]
        axes=[]
        for a,values in enumerate(cuts):
            spans=[]
            for span in zip(values[:-1],values[1:]):
                mask=sum(1<<i for i,d in enumerate(selected) if d.region.bounds[a][0]<=span[0] and span[1]<=d.region.bounds[a][1])
                spans.append((span,mask))
            axes.append(spans)
        def visit(axis,mask,bounds):
            if axis==4:
                indices=[]
                while mask:
                    bit=mask&-mask;indices.append(bit.bit_length()-1);mask-=bit
                emit(bounds,indices);return
            for span,coverage in axes[axis]:
                covered=mask&coverage
                if covered:visit(axis+1,covered,(*bounds,span))
        visit(0,(1<<len(selected))-1,())
    return result


def demand(index,region,*,src='memory:0',dst=None,tensor='a',bits=8,microbatch=0,producer=(),deps=(),cohort=0):
    return _Demand(src,dst or 'compute:'+str(index%8),tensor,TensorRegion(region),bits,microbatch,
        producer,'unit'+str(index),'dram_input',deps,cohort)


def main():
    rng=random.Random(20261105)
    examples=[]
    for i in range(160):
        values=[]
        for j in range(rng.randint(1,48)):
            bounds=tuple(tuple(sorted(rng.sample(range(5),2))) for _ in range(4))
            values.append(demand(j,bounds,src=rng.choice(('memory:0','compute:0')),dst=rng.choice(('compute:0','compute:1','compute:2')),
                tensor=rng.choice(('a','b')),bits=rng.choice((8,16)),microbatch=rng.randrange(2),
                producer=() if i%2 else ('p',),deps=() if i%3 else ('dep',),cohort=j%2 if i%4 else 0))
        examples.append(values)
    # Stress the exact grid, duplicates and local/remote consumers, with
    # deterministic original output order included in equality.
    examples += [[demand(i,((0,1),(0,16),(0,32),(0,32))) for i in range(512)],
        [demand(i,((0,1),(0,16),(i%32,i%32+2),(i%32,i%32+2))) for i in range(128)],
        [demand(i,((0,1),(i, i+1),(0,8),(0,8)),src='compute:0',dst='compute:0') for i in range(64)]]
    for values in examples:
        old,new=_atomic_demands(values),atomic_bitset(values)
        if old!=new:raise AssertionError('Exact ordered physical demand atoms differ')
    timings=[]
    for label,values in (('singletons',[demand(i,((0,1),(0,8),(0,8),(0,8)),cohort=i) for i in range(1024)]),
                         ('duplicates',examples[-3]),('diagonal',examples[-2])):
        times={}
        for name,function in (('original',_atomic_demands),('bitset',atomic_bitset)):
            started=time.perf_counter()
            for _ in range(12):function(values)
            times[name]=time.perf_counter()-started
        timings.append(dict(case=label,demands=len(values),iterations=12,seconds=times,component_speed_ratio=times['original']/times['bitset']))
    result=dict(ordered_outputs_identical=True,tested_inputs=len(examples),timings=timings,new_PPA=0,new_SGD=0,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adopted=False,scope='synthetic exact atomic-demand partition component only; not full PPA equivalence, full oracle or DSE acceleration; existing physics/source guards unchanged')
    out=ROOT/'output/inmin_v14_20261105/exact_atomic_component_benchmark.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
