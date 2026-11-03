"""Conservative, capacity-checked INMIN v0.2 resource-calendar evaluator.

This is a new analytical oracle, not a retagged Gemini result. Whole-flow
reservations and full-channel PE tiles trade overlap for auditable resource
conservation. Queue storage is finite; one flow reserves each routed resource
atomically, so split queue-wait attribution is explicitly unsupported.
"""
from __future__ import annotations

from collections import defaultdict,OrderedDict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

from .communication import Transfer, build_communication
from .costs import sram_bank
from .diagnostics import _schedule_slack
from .mapping import TensorRegion,validate_plan
from .reuse import _union_elements, apply_weight_reuse, dram_residency
from .scheduler import Job, peak_interval_bytes, schedule_jobs
from .single_array_compiler import compile_single_array, tile_requirements
from .single_array_communication import VERSION as COHORT_VERSION, split_read_cohorts
from .single_array_hardware import (PROFILE, SingleArrayHardware, build_single_array_noc,
                                    build_single_array_package)


def physical_die(endpoint):
    return endpoint.split('/')[0]


@dataclass(frozen=True)
class SingleArrayEvaluation:
    metrics: dict
    compiled: object
    communication: object
    schedule: object
    diagnostics: dict
    transfer_statistics: tuple
    buffer_statistics: dict

    def to_dict(self):
        return {"metrics": self.metrics, "mapping": self.compiled.plan.to_dict(),
                "compute_slots": sorted({u.chiplet for u in self.compiled.units}),
                "pe_allocations": {m.layer_id: list(m.chiplets) for m in self.compiled.plan.layer_mappings},
                "units": [asdict(u) for u in self.compiled.units],
                "events": [e.to_dict() for e in self.schedule.events],
                "transfers": list(self.transfer_statistics), "diagnostics": self.diagnostics,
                "buffer_statistics": self.buffer_statistics}


class SingleArrayEvaluator:
    def __init__(self, workload, hardware=None, *, trace=True, deadline=None, compile_hook=None, calendar_backend='auto'):
        self.workload = workload
        self.hardware = hardware or SingleArrayHardware()
        self.trace, self.deadline = trace, deadline
        self.compile_hook, self.compile_evidence = compile_hook, None
        from .native_schedule import available,source_hash
        self.calendar_backend=('native' if available() else 'coalesced') if calendar_backend=='auto' else calendar_backend
        self.package = build_single_array_package(self.hardware)
        self.nocs = {f'compute:{die}': build_single_array_noc(self.hardware, die)
                     for die in range(self.hardware.compute_count)}
        self.cache, self.cache_hits, self.evaluations = {}, 0, 0
        self._path_cache = {}
        self._transfer_service_cache = {}
        self._kernel_cache = {}
        self._geometry_cache = OrderedDict()
        self._geometry_items = 0
        self._package_paths = {}
        self._link_map = {frozenset((link.source, link.destination)): link for link in self.package.links}
        self.sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.glob('*.py')}
        self.identity = hashlib.sha256(json.dumps({"workload": workload.fingerprint,
            "hardware": self.hardware.to_dict(), "sources": self.sources,
            "scheduler": self.calendar_backend, "native_source":source_hash()}, sort_keys=True).encode()).hexdigest()
        self.banks = self.hardware.sram_costs

    def _check_time(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('v0.2 oracle deadline reached; no partial result is executable')

    def _sram_j(self, name, read=0, write=0):
        bank = self.banks[name]
        return 8e-12 * (read * bank['read_pj_per_bit'] + write * bank['write_pj_per_bit'])

    def _noc_path(self, slot, direction):
        key = (slot, direction)
        if key in self._path_cache:
            return self._path_cache[key]
        graph = self.nocs[physical_die(slot)]
        gateways = [name for name, _ in graph.coordinates if '/l2_gateway:' in name]
        path = min((graph.route(slot, gateway) for gateway in gateways), key=lambda path: (len(path), path))
        self._path_cache[key] = path if direction == 'write' else tuple(reversed(path))
        return self._path_cache[key]

    def _local_service(self, slot, amount, direction):
        if amount == 0:
            return 0., (), (), 0.
        spec = self.hardware
        path = self._noc_path(slot, direction)
        wire = spec.assumptions.wire_bytes(amount)
        loads = tuple((a, b, wire, 'noc') for a, b in zip(path, path[1:]))
        resources = tuple(f'link:{a}>{b}' for a, b, _, _ in loads)
        duration = max(amount / spec.l2_bandwidth_Bps, amount / spec.l1_bandwidth_Bps,
                       wire / spec.noc_bandwidth_Bps) + (len(path) - 1) * spec.assumptions.router_latency_cycles / spec.noc_clock_hz
        return duration, resources, loads, sum(row[2] for row in loads) * 8 * .4e-12

    def transfer_service(self, transfer):
        key=(transfer.src,transfer.dsts,transfer.bytes,transfer.kind)
        cached=self._transfer_service_cache.get(key)
        if cached is None:
            cached=self._transfer_service_uncached(transfer)
            if len(self._transfer_service_cache)>=32768:self._transfer_service_cache.clear()
            self._transfer_service_cache[key]=cached
        return (*cached[:3],dict(cached[3]))

    def _transfer_service_uncached(self, transfer):
        """Only endpoints access L2/DMA; transit compute routers never do."""
        spec = self.hardware
        source = physical_die(transfer.src)
        destinations = tuple(dict.fromkeys(physical_die(dst) for dst in transfer.dsts))
        paths = []
        for dst in destinations:
            if dst != source:
                key = (source, dst)
                if key not in self._package_paths:
                    self._package_paths[key] = self.package.route(source, dst)
                paths.append(self._package_paths[key])
        edges = {(a, b) for path in paths for a, b in zip(path, path[1:])}
        wire = spec.assumptions.wire_bytes(transfer.bytes)
        loads, resources, services = [], [], []
        components = defaultdict(float)
        link_map = self._link_map
        for a, b in sorted(edges):
            link = link_map[frozenset((a, b))]
            memory = link.kind == 'dram_service'
            amount = transfer.bytes if memory else wire
            loads.append((a, b, amount, 'dram' if memory else 'nop'))
            resources.append(f'link:{a}>{b}')
            services.append(amount / link.bandwidth_Bps)
            components['dram_j' if memory else 'nop_j'] += amount * 8 * (8.75e-12 if memory else 1.17e-12)
        memory_nodes = {name for path in paths for name in path if name.startswith('memory:')}
        for memory in sorted(memory_nodes):
            resources.append(f'{dict(self.package.memory_owners)[memory]}:controller')
            services.append(transfer.bytes / spec.dram_bandwidth_Bps + spec.assumptions.dram_access_latency_s)
        routers = {name for path in paths for name in path if not name.startswith('memory:')}
        for router in sorted(routers):
            fanout = max(1, sum(a == router for a, _ in edges))
            resources.append(f'{router}:nop_switch')
            services.append(wire * fanout / spec.router_bandwidth_Bps)
            components['router_j'] += wire * fanout * 8 * .1e-12
        endpoints = {name for name in (source, *destinations) if name.startswith('compute:')}
        remote = bool(paths)
        if remote or transfer.kind == 'df_cache_copy':
            for die in sorted(endpoints):
                resources.extend((f'{die}:dma', f'{die}:l2_rw'))
                services.extend((transfer.bytes / spec.dma_bandwidth_Bps,
                                 transfer.bytes / spec.l2_bandwidth_Bps,
                                 spec.assumptions.dma_descriptor_cycles / spec.compute_clock_hz))
                write = die in destinations and die != source
                bank = 'WL2' if transfer.kind == 'dram_weight' else 'AL2' if write else 'OL2'
                components['l2_j'] += self._sram_j(bank, write=transfer.bytes if write else 0,
                                                  read=0 if write else transfer.bytes)
        if transfer.kind == 'df_cache_copy':
            # Explicit L2 gather, including both directions of a local copy.
            die = source
            services.append(2 * transfer.bytes / spec.l2_bandwidth_Bps)
            components['l2_j'] += self._sram_j('AL2', write=transfer.bytes)
        duration = max(services, default=0.)
        if paths:
            duration += max(len(path) - 1 for path in paths) * spec.assumptions.router_latency_cycles / spec.nop_clock_hz
        return duration, tuple(dict.fromkeys(resources)), tuple(loads), dict(components)

    def _kernel(self, unit, layer):
        key=(layer.id,unit.output_region.shape,unit.output_region.bounds[1],unit.dataflow)
        if key not in self._kernel_cache:
            self._kernel_cache[key]=self._kernel_uncached(unit,layer)
        return dict(self._kernel_cache[key])

    def _kernel_uncached(self, unit, layer):
        """16 output lanes x16 input MACs; tails and grouped conv are explicit."""
        if not unit.macs:
            return {"mac_cycles": 0, "l1_read_bytes": 0, "l1_write_bytes": 0, "lane_utilization": 0.}
        p = unit.output_region.shape[0] * unit.output_region.shape[2] * unit.output_region.shape[3]
        c, kernel = layer.weight_shape[1], math.prod(layer.kernel)
        total_k = self.workload.tensor_map[layer.output].shape[1]
        per_group = total_k // layer.groups
        lo, hi = unit.output_region.bounds[1]
        blocks = sum(math.ceil((min(hi, (group + 1) * per_group) - max(lo, group * per_group)) / 16)
                     for group in range(lo // per_group, (hi - 1) // per_group + 1))
        cycles = p * blocks * math.ceil(c / 16) * kernel
        weights = unit.weight_bytes * (1 if unit.dataflow == 'weight_reuse' else p)
        activations = p * blocks * c * kernel
        if layer.kind == 'Conv2d' and layer.groups == total_k and c == 1:
            # Independent depthwise channels occupy independent output lanes.
            # Each lane uses one of its sixteen input-MAC positions; sixteen
            # channel operands fit the explicit 16B activation working set.
            cycles = p * math.ceil((hi - lo) / 16) * kernel
            activations = unit.macs
        psum_extra = 0
        if unit.dataflow == 'activation_reuse':
            activations = p * c * kernel * (1 + (hi - 1) // per_group - lo // per_group)
            psum_extra = max(0, 2 * math.ceil(c / 16) * kernel - 2) * unit.output_region.elements * self.hardware.assumptions.psum_storage_bytes
        elif unit.dataflow == 'weight_reuse':
            psum_extra = max(0, 2 * math.ceil(c / 16) * kernel - 2) * unit.output_region.elements * self.hardware.assumptions.psum_storage_bytes
        return {"mac_cycles": cycles, "l1_read_bytes": weights + activations + psum_extra // 2,
                "l1_write_bytes": psum_extra - psum_extra // 2,
                "weight_read_bytes": weights, "activation_read_bytes": activations,
                "partial_sum_rw_bytes": psum_extra, "lane_utilization": unit.macs / (cycles * 256)}

    def evaluate(self, plan, *, retain_trace=None):
        self._check_time()
        key = hashlib.sha256(json.dumps(plan.to_dict(), sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        wall_started = time.monotonic()
        cpu_started = time.process_time()
        spec, workload = self.hardware, self.workload
        geometry=plan.to_dict()
        for mapping in geometry['layer_mappings']:
            mapping['weight_policy']='cold';mapping['dataflow']='output_stationary'
        geometry_key=hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()
        cached=self._geometry_cache.get(geometry_key)
        if cached is not None:
            # Canonical geometry intentionally ignores these physical policy
            # fields. Reusing compilation must not bypass their validation.
            validate_plan(workload,plan,spec.compute_slots)
            base,communication=cached
            flow={m.layer_id:m.dataflow for m in plan.layer_mappings}
            compiled=replace(base,plan=plan,units=tuple(replace(u,dataflow=flow[u.layer_id]) for u in base.units))
            self._geometry_cache.move_to_end(geometry_key)
        else:
            compiled = compile_single_array(workload, plan, spec)
        compiled_wall = time.monotonic()
        self.compile_evidence = {'fusion_depths': sorted({len(stack.layer_ids) for stack in plan.fusion_stacks}),
                                 'unit_count': len(compiled.units), 'executed_macs': sum(u.macs for u in compiled.units)}
        if self.compile_hook is not None:
            self.compile_hook(self.compile_evidence)
        self._check_time()
        if cached is None:
            communication=split_read_cohorts(compiled,build_communication(workload,compiled,plan.collective))
            if len(communication.transfers)<=250000:
                while self._geometry_cache and (len(self._geometry_cache)>=8 or self._geometry_items+len(communication.transfers)>250000):
                    _,old=self._geometry_cache.popitem(last=False);self._geometry_items-=len(old[1].transfers)
                self._geometry_cache[geometry_key]=(compiled,communication)
                self._geometry_items+=len(communication.transfers)
        communication, reuse = apply_weight_reuse(compiled, communication)
        # Quantization scale/shift/zero point parameters are paid cold loads,
        # using the selected external weight IO (interleave -> explicit IO0).
        qparams = []
        for unit in compiled.units:
            if not unit.macs:
                continue
            mapping = plan.mapping_map[unit.layer_id]
            memory = mapping.weight_dram if mapping.weight_dram in plan.memory_ids else plan.memory_ids[0]
            lo, hi = unit.output_region.bounds[1]
            qparams.append(Transfer(f'quant:{unit.id}', memory, (unit.chiplet,), (hi - lo) * 8,
                (), (unit.id,), 'quantization_parameters', f'quant:{unit.layer_id}',
                TensorRegion(((0, 1), (lo, hi), (0, 1), (0, 8))), 0))
        dependencies = {name: list(deps) for name, deps in communication.unit_dependencies}
        communication = replace(communication, transfers=(*communication.transfers, *qparams),
                                unit_dependencies=tuple((name, tuple(deps)) for name, deps in dependencies.items()))
        units = compiled.unit_map
        group_index = {group.id: i for i, group in enumerate(plan.groups)}
        layer_index = {layer.id: i for i, layer in enumerate(workload.layers)}
        def priority(unit):
            order = unit.execution_order if unit.execution_order >= 0 else layer_index[unit.layer_id]
            return (group_index[unit.group_id], order, unit.tile_index, unit.shard_index)
        previous, before, bank_before, previous_vector, vector_before = {}, {}, {}, {}, {}
        for unit in sorted(compiled.units, key=priority):
            predecessor = previous.get(unit.chiplet)
            if predecessor:
                before[unit.id] = predecessor
                if predecessor in before:
                    bank_before[unit.id] = before[predecessor]
            previous[unit.chiplet] = unit.id
            if unit.physical_die_id in previous_vector:
                vector_before[unit.id] = previous_vector[unit.physical_die_id]
            previous_vector[unit.physical_die_id] = unit.id
        jobs, service, local_loads, energy_components = [], {}, {}, defaultdict(float)
        def add_job(name, deps, resources, duration, kind, components, pri, metadata=None, loads=()):
            jobs.append(Job(name, tuple(dict.fromkeys(deps)), tuple(dict.fromkeys(resources)), duration,
                            kind, sum(components.values()), pri, metadata))
            for component, value in components.items():
                energy_components[component] += value
            local_loads[name] = loads
        for unit in compiled.units:
            self._check_time()
            layer, die, slot = workload.layer_map[unit.layer_id], unit.physical_die_id, unit.chiplet
            pri, req = priority(unit), tile_requirements(workload, workload.layer_map[unit.layer_id], unit.output_region,
                                                        spec.assumptions.psum_storage_bytes)
            deps = list(dependencies[unit.id])
            if group_index[unit.group_id]:
                deps.append(f'group_done:{plan.groups[group_index[unit.group_id] - 1].id}')
            if unit.id in bank_before:
                deps.append(bank_before[unit.id])
            kernel = self._kernel(unit, layer)
            service[unit.id] = kernel
            if unit.macs:
                input_bytes = unit.input_bytes + unit.weight_bytes
                read_s, links, loads, noc_j = self._local_service(slot, input_bytes, 'read')
                read_j = self._sram_j('WL2', read=unit.weight_bytes) + self._sram_j('AL2', read=unit.input_bytes)
                read_j += self._sram_j('WL1', write=unit.weight_bytes) + self._sram_j('AL1', write=unit.input_bytes)
                add_job(f'read:{unit.id}', deps, (f'{die}:l2_rw', f'{slot}:l1_rw', *links), read_s,
                        'pe_read', {'sram_j': read_j, 'noc_j': noc_j}, (*pri, 0), {'unit_id': unit.id}, loads)
                compute_deps = [f'read:{unit.id}']
                if unit.id in before:
                    compute_deps.append(f'compute:{before[unit.id]}')
                l1_time = (kernel['l1_read_bytes'] + kernel['l1_write_bytes'] + req['psum_bytes']) / spec.l1_bandwidth_Bps
                l1_j = self._sram_j('WL1', read=kernel['weight_read_bytes'])
                l1_j += self._sram_j('AL1', read=kernel['activation_read_bytes'])
                l1_j += self._sram_j('OL1', read=kernel['partial_sum_rw_bytes'] // 2,
                                   write=kernel['partial_sum_rw_bytes'] - kernel['partial_sum_rw_bytes'] // 2 + req['psum_bytes'])
                compute_s = max(kernel['mac_cycles'] / spec.compute_clock_hz, l1_time)
                service[unit.id].update({'compute_s': kernel['mac_cycles'] / spec.compute_clock_hz,
                                         'l1_service_s': l1_time, 'scheduled_compute_s': compute_s})
                add_job(f'compute:{unit.id}', compute_deps, (f'{slot}:mac', f'{slot}:l1_rw'), compute_s,
                        'pe_compute', {'mac_j': unit.macs * .024e-12, 'sram_j': l1_j}, (*pri, 1), {'unit_id': unit.id})
                write_s, links, loads, noc_j = self._local_service(slot, req['psum_bytes'], 'write')
                add_job(f'psum:{unit.id}', (f'compute:{unit.id}',), (f'{die}:l2_rw', f'{slot}:l1_rw', *links),
                        write_s, 'psum_write', {'sram_j': self._sram_j('OL1', read=req['psum_bytes']) +
                        self._sram_j('OL2', write=req['psum_bytes']), 'noc_j': noc_j}, (*pri, 2), {'unit_id': unit.id}, loads)
                vector_deps, vector_input = (f'psum:{unit.id}', f'quant:{unit.id}'), req['psum_bytes']
            else:
                # No PE compute/fabric is invented for the die-level VPU.
                add_job(f'read:{unit.id}', deps, (), 0., 'vector_ready', {}, (*pri, 0), {'unit_id': unit.id})
                add_job(f'compute:{unit.id}', (f'read:{unit.id}',), (), 0., 'vector_marker', {}, (*pri, 1), {'unit_id': unit.id})
                vector_deps, vector_input = (f'compute:{unit.id}',), unit.input_bytes
            elements = unit.output_region.elements
            vector_ops = unit.vector_ops + (elements * dict(spec.assumptions.vector_cycles_per_element)['requant'] if unit.macs else 0)
            # Average/reduction normalization is explicitly three lane cycles
            # in addition to the extracted additions; an M fixed-point scale.
            if layer.kind in ('AvgPool2d', 'AdaptiveAvgPool2d'):
                vector_ops += elements * 3
            qbytes = unit.output_region.shape[1] * 8 if unit.macs else 0
            if unit.id in vector_before:
                vector_deps = (*vector_deps, vector_before[unit.id])
            moved = vector_input + unit.output_bytes + qbytes
            vector_s = max(math.ceil(vector_ops / spec.vector_lanes) / spec.compute_clock_hz,
                           moved / spec.l2_bandwidth_Bps)
            add_job(unit.id, vector_deps, (f'{die}:vpu', f'{die}:l2_rw'), vector_s, 'vector_execute',
                    {'vector_j': vector_ops * .024e-12, 'sram_j': self._sram_j('OL2' if unit.macs else 'AL2', read=vector_input) +
                     self._sram_j('AL2', read=qbytes) + self._sram_j('OL2', write=unit.output_bytes)},
                    (*pri, 3), {'unit_id': unit.id, 'vector_lane_ops': vector_ops, 'requantized': bool(unit.macs)})
        transfer_stats = []
        for transfer in communication.transfers:
            self._check_time()
            consumers = [units[name] for name in transfer.consumer_units]
            producers = [units[name] for name in transfer.producer_units]
            reference = min(consumers or producers, key=priority)
            deps = [*transfer.producer_units, *transfer.dependencies]
            if transfer.kind == 'quantization_parameters':
                # The 4KiB AL2 parameter region has one explicit live batch.
                # Load after this tile's psums exist and the preceding VPU
                # consumer has released its parameters, avoiding free staging.
                owner = transfer.consumer_units[0]
                deps.append(f'psum:{owner}')
                if owner in vector_before:
                    deps.append(vector_before[owner])
            # Source data remains in L2 until transfer completion. Limit A/B
            # staging to the first recipient's bank, never a future same-flow
            # consumer (which would create a self-dependency).
            first_by_slot = {}
            for unit in sorted(consumers, key=priority):
                first_by_slot.setdefault(unit.chiplet, unit)
            # Shared L2 holds one pending read cohort per PE; its space can
            # be reused after the preceding tile has refilled L1, without
            # waiting for that tile's compute/VPU completion. L1 A/B remains
            # independently protected by bank_before on the PE read stage.
            deps.extend(f'read:{before[u.id]}' if units[before[u.id]].macs else before[u.id]
                        for u in first_by_slot.values() if u.id in before)
            if group_index[reference.group_id]:
                deps.append(f'group_done:{plan.groups[group_index[reference.group_id] - 1].id}')
            duration, resources, loads, components = self.transfer_service(transfer)
            add_job(transfer.id, deps, resources, duration, transfer.kind, components,
                    (*priority(reference), -1), {'transfer_id': transfer.id}, loads)
            transfer_stats.append({'id': transfer.id, 'kind': transfer.kind, 'bytes': transfer.bytes,
                                   'source': transfer.src, 'destinations': transfer.dsts,
                                   'link_loads': loads, 'energy_components': components})
        for group in plan.groups:
            members = [job.id for job in jobs if (job.metadata or {}).get('unit_id') in units and
                       units[job.metadata['unit_id']].group_id == group.id]
            members.extend(t.id for t in communication.transfers if t.producer_units and
                           units[t.producer_units[0]].group_id == group.id and not t.consumer_units)
            add_job(f'group_done:{group.id}', members, (), 0., 'barrier', {}, (group_index[group.id], 10**9))
        self._check_time()
        jobs_wall = time.monotonic()
        schedule = schedule_jobs(jobs, availability=self.calendar_backend)
        scheduled_wall = time.monotonic()
        self._check_time()
        buffers = self._capacity(compiled, communication, schedule)
        memory = dram_residency(compiled, communication, schedule, spec)
        parameter_regions = defaultdict(list)
        for transfer in qparams:
            parameter_regions[(transfer.src, transfer.tensor)].append(transfer.region)
        parameter_bytes = defaultdict(int)
        for (memory_id, _), regions in parameter_regions.items():
            parameter_bytes[memory_id] += _union_elements(regions)
        for memory_id, amount in parameter_bytes.items():
            memory['initial_resident_bytes'][memory_id] = memory['initial_resident_bytes'].get(memory_id, 0) + amount
            memory['peak_by_memory'][memory_id] += amount
        memory['quantization_parameter_bytes_by_memory'] = dict(parameter_bytes)
        memory['violations'] = [f'{memory_id} DRAM needs {amount} B > {spec.dram_capacity_bytes} B'
                                for memory_id, amount in memory['peak_by_memory'].items() if amount > spec.dram_capacity_bytes]
        buffers['dram'] = memory
        violations = [*buffers['violations'], *memory['violations']]
        area = spec.area_estimate
        idle_w = .02 * area['compute_total_mm2'] + .008 * area['io_total_mm2']
        idle_w += area['external_dram_area_mm2'] * spec.assumptions.external_dram_idle_w_per_mm2
        energy_components['idle_j'] = idle_w * schedule.makespan_s
        energy = sum(energy_components.values())
        executed = sum(u.macs for u in compiled.units)
        stats = compiled.execution_statistics or {}
        metrics = {'model': workload.name, 'hardware_profile': spec.profile_id, 'evaluator_identity': self.identity,
            'backend': 'single_array_python_resource_calendar_v1', 'native_rapidchiplet_executed': False,
            'architecture_candidate_id': spec.architecture_id,
            'feasible': not violations, 'executable': not violations, 'violations': violations,
            'latency_s': schedule.makespan_s, 'energy_j': energy, 'power_w': energy / schedule.makespan_s,
            'area_mm2': area['total_silicon_mm2'], 'area_definition': 'compute+IO+external DRAM silicon proxy; external device/area uncalibrated M, not physical measurement',
            'power_definition': 'modeled compute/IO/DRAM idle plus workload dynamic energy divided by batch1 E2E latency; not peak/TDP',
            'energy_components': dict(energy_components), 'compute_chiplets': spec.compute_count, 'io_chiplets': spec.io_count,
            'physical_cores': 0, 'pes': spec.compute_count * 16, 'compute_slots': spec.compute_slots,
            'logical_macs': workload.total_macs, 'executed_macs': executed,
            'recompute_macs': stats.get('recompute_macs', 0), 'pruned_macs': stats.get('pruned_macs', 0),
            'dram_read_bytes': communication.dram_read_bytes, 'dram_write_bytes': communication.dram_write_bytes,
            'nop_hop_bytes': sum(amount for loads in local_loads.values() for _, _, amount, kind in loads if kind == 'nop'),
            'noc_hop_bytes': sum(amount for loads in local_loads.values() for _, _, amount, kind in loads if kind == 'noc'),
            'l1_l2_service_bytes': sum(u.input_bytes + u.weight_bytes + u.output_region.elements * spec.assumptions.psum_storage_bytes for u in compiled.units if u.macs),
            'unit_count': len(compiled.units), 'transfer_count': len(communication.transfers), 'weight_reuse': reuse,
            'communication_cohort_version': COHORT_VERSION,
            'resource_busy_s': schedule.resource_busy_s, 'buffer_peak_bytes': buffers['peak_by_pool'],
            'compute_utilization': executed / (spec.compute_count * 4096 * spec.compute_clock_hz * schedule.makespan_s),
            'calibration_status': 'single_array_v0_2_uncalibrated', 'physical_limit_certified': False,
            'oracle_source_sha256': self.sources, 'timing_model': 'finite-buffer atomic routed flow reservations; full-channel PE tiles; serialized L2 RW/VPU; data-dependent stalls not modeled'}
        physical_wall = time.monotonic()
        diagnostics = self._diagnostics(compiled, communication, schedule, service, local_loads, buffers, metrics)
        metrics['diagnostics'] = self.compact_diagnostics(diagnostics)
        metrics['calendar_backend'] = self.calendar_backend
        metrics['oracle_process_cpu_s'] = time.process_time() - cpu_started
        metrics['exact_geometry_cache_hit'] = cached is not None
        metrics['oracle_phase_wall_s'] = {'compile': compiled_wall - wall_started,
            'communication_and_job_build': jobs_wall - compiled_wall, 'schedule': scheduled_wall - jobs_wall,
            'capacity_and_metrics': physical_wall - scheduled_wall, 'diagnostics': time.monotonic() - physical_wall}
        result = SingleArrayEvaluation(metrics, compiled, communication, schedule, diagnostics,
                                       tuple(transfer_stats), buffers)
        self._check_time()
        self.evaluations += 1
        self.cache[key] = result
        return result

    __call__ = evaluate

    @staticmethod
    def compact_diagnostics(diagnostics):
        """Keep exact aggregates; bound trace samples in policy/checkpoints.

        The full verified artifact retains every resource window and flow.
        Truncation is explicit, so an omitted flow is never invented as a
        targeted placement/storage owner by the proposal generator.
        """
        return {**diagnostics, 'critical_path': diagnostics['critical_path'][:128],
                'critical_path_count': len(diagnostics['critical_path']),
                'critical_path_truncated': len(diagnostics['critical_path']) > 128,
                'network': [{**row, 'time_windows': row['time_windows'][:4],
                             'time_window_count': len(row['time_windows']),
                             'time_windows_truncated': len(row['time_windows']) > 4} for row in diagnostics['network']],
                'flows': diagnostics['flows'][:128], 'flows_truncated': len(diagnostics['flows']) > 128}

    def _capacity(self, compiled, communication, schedule):
        spec, events = self.hardware, schedule.event_map
        intervals, capacities = defaultdict(list), {}
        for die in range(spec.compute_count):
            for bank, capacity in (('WL2', spec.l2_weight_bytes), ('AL2', spec.l2_activation_bytes - spec.assumptions.quantization_parameter_bytes), ('OL2', spec.l2_output_bytes)):
                capacities[f'compute:{die}:{bank}'] = capacity
            for pe in range(spec.pe_count):
                for bank, capacity in (('WL1', spec.l1_weight_bytes), ('AL1', spec.l1_activation_bytes), ('OL1', spec.l1_output_bytes)):
                    capacities[f'compute:{die}/pe:{pe}:{bank}'] = 2 * capacity
        # Reserve the quantization parameter region inside AL2, never extra SRAM.
        for die in range(spec.compute_count):
            capacities[f'compute:{die}:quant_params'] = spec.assumptions.quantization_parameter_bytes
        outgoing, readers = defaultdict(list), defaultdict(list)
        for consumer in compiled.units:
            for name in consumer.dependencies:
                producer = compiled.unit_map[name]
                if producer.group_id == consumer.group_id and producer.physical_die_id == consumer.physical_die_id:
                    # The communication builder omits a same-PE local flow.
                    # Its L2 result still lives until the actual refill/VPU
                    # read, including a residual consumed much later.
                    readers[name].append(consumer.id)
        for transfer in communication.transfers:
            if physical_die(transfer.src).startswith('compute:'):
                for name in transfer.producer_units:
                    outgoing[name].append(transfer.id)
                    readers[name].extend(user for user in transfer.consumer_units
                                         if compiled.unit_map[user].physical_die_id == physical_die(transfer.src))
            for destination in set(physical_die(dst) for dst in transfer.dsts if dst.startswith('compute:')):
                if destination == physical_die(transfer.src) and transfer.kind != 'df_cache_copy':
                    continue
                users = [name for name in transfer.consumer_units if compiled.unit_map[name].physical_die_id == destination]
                if not users:
                    continue
                finish = max(events[name if transfer.kind == 'quantization_parameters' or not compiled.unit_map[name].macs else f'read:{name}'].finish_s for name in users)
                pool = 'WL2' if transfer.kind == 'dram_weight' else 'quant_params' if transfer.kind == 'quantization_parameters' else 'AL2'
                intervals[f'{destination}:{pool}'].append((events[transfer.id].start_s, finish, transfer.bytes))
        for unit in compiled.units:
            req = tile_requirements(self.workload, self.workload.layer_map[unit.layer_id], unit.output_region,
                                    spec.assumptions.psum_storage_bytes)
            start, finish = events[f'read:{unit.id}'].start_s, events[unit.id].finish_s
            for pool, amount, capacity in (('WL1', req['weight_bytes'], spec.l1_weight_bytes),
                                           ('AL1', req['activation_bytes'], spec.l1_activation_bytes),
                                           ('OL1', req['psum_bytes'], spec.l1_output_bytes)):
                if not unit.macs:
                    continue
                name = f'{unit.chiplet}:{pool}'
                # Two physical A/B slots; lifetimes must fit their combined pool
                # and each compiler-checked tile fits a single slot.
                capacities[name] = 2 * capacity
                intervals[name].append((start, finish, amount))
            if unit.macs:
                intervals[f'{unit.physical_die_id}:OL2'].append((events[f'psum:{unit.id}'].start_s, finish, req['psum_bytes']))
            last = max([events[name].finish_s for name in outgoing[unit.id]] +
                       [events[f'read:{name}' if compiled.unit_map[name].macs else name].finish_s for name in readers[unit.id]] + [finish])
            intervals[f'{unit.physical_die_id}:OL2'].append((events[unit.id].start_s, last, unit.output_bytes))
        peak = {name: peak_interval_bytes(intervals[name]) for name in capacities}
        violations = [f'{name} needs {amount} B > {capacities[name]} B' for name, amount in peak.items() if amount > capacities[name]]
        return {'peak_by_pool': peak, 'capacity_by_pool': capacities, 'violations': violations,
                'allocator': 'actual conservative arrival-to-last-reader intervals; shared die pools, separate W/A/O; per-PE two-slot L1; no implicit spill'}

    def _diagnostics(self, compiled, communication, schedule, services, loads, buffers, metrics):
        events, slack = schedule.event_map, _schedule_slack(schedule)
        waits = {name: max(0., event.start_s - max((events[dep].finish_s for dep in event.job.dependencies), default=0.))
                 for name, event in events.items()}
        layers = []
        tolerance = max(1e-15, schedule.makespan_s * 1e-8)
        units_by_layer, jobs_by_layer, transfers_by_layer = defaultdict(list), defaultdict(list), defaultdict(list)
        for unit in compiled.units:
            units_by_layer[unit.layer_id].append(unit)
        for event in schedule.events:
            owner = (event.job.metadata or {}).get('unit_id')
            if owner in compiled.unit_map:
                jobs_by_layer[compiled.unit_map[owner].layer_id].append(event.job.id)
        for transfer in communication.transfers:
            for layer_id in {compiled.unit_map[name].layer_id for name in (*transfer.producer_units, *transfer.consumer_units)}:
                transfers_by_layer[layer_id].append(transfer)
        for layer in self.workload.layers:
            units = units_by_layer[layer.id]
            names = jobs_by_layer[layer.id]
            compute_s = sum(services[u.id].get('scheduled_compute_s', 0.) for u in units)
            critical = min((slack[name] for name in names if slack[name] is not None), default=math.inf) <= tolerance
            pe_wait = sum(waits[name] for name in names)
            vector_s = sum(events[u.id].job.duration_s for u in units)
            mac_s = sum(services[u.id].get('compute_s', 0.) for u in units)
            l1_s = sum(services[u.id].get('l1_service_s', 0.) for u in units)
            relevant = transfers_by_layer[layer.id]
            transfer_wait = sum(waits[t.id] for t in relevant)
            used_pools = {name for unit in units for name in (unit.physical_die_id, unit.chiplet)}
            evidence = []
            if critical and mac_s >= l1_s and mac_s > 2 * vector_s and mac_s >= pe_wait:
                evidence.append({'kind': 'compute', 'confidence': 'partial', 'signal': 'critical calendar path, MAC service dominates L1 and VPU'})
            if critical and l1_s > mac_s:
                evidence.append({'kind': 'sram_bandwidth', 'confidence': 'partial', 'signal': 'modeled L1 service exceeds arithmetic service on critical layer'})
            if critical and transfer_wait > compute_s:
                has_dram = any(t.src.startswith('memory:') or any(dst.startswith('memory:') for dst in t.dsts) for t in relevant)
                evidence.append({'kind': 'dram_dma_or_network' if has_dram else 'network', 'confidence': 'partial',
                                 'signal': 'observed route reservation queue; unique per-resource attribution unsupported'})
            if critical and vector_s > compute_s:
                evidence.append({'kind': 'vector_or_shared_l2', 'confidence': 'partial', 'signal': 'die VPU/L2 reservation dominates local MAC stages'})
            label = 'compute' if any(row['kind'] == 'compute' for row in evidence) else 'network' if any('network' in row['kind'] for row in evidence) else 'memory' if any(row['kind'] == 'sram_bandwidth' for row in evidence) else 'unknown'
            layers.append({'layer_id': layer.id, 'pe_allocations': sorted({u.chiplet for u in units}),
                'compute_busy_s': compute_s, 'mac_service_s': mac_s, 'l1_service_s': l1_s,
                'vector_busy_s': vector_s, 'resource_wait_s': pe_wait, 'transfer_queue_wait_s': transfer_wait,
                'max_job_queue_wait_s': max((waits[name] for name in names), default=0.),
                'max_transfer_queue_wait_s': max((waits[t.id] for t in relevant), default=0.),
                'transfer_count': len(relevant),
                'criticality': float(critical), 'slack_s': min((slack[n] for n in names if slack[n] is not None), default=None),
                'pe_utilization': sum(u.macs for u in units) / (compute_s * 256 * self.hardware.compute_clock_hz) if compute_s else 0.,
                'bottleneck': label, 'bottlenecks': evidence or [{'kind': 'unknown', 'confidence': 'unknown', 'signal': 'insufficient dominant-path evidence'}],
                'memory_pressure': max((buffers['peak_by_pool'].get(pool, 0) / capacity for pool, capacity in buffers['capacity_by_pool'].items()
                    if pool.rsplit(':', 1)[0] in used_pools), default=0.),
                'memory_pressure_scope': 'shared physical pools; not unique per-layer causality',
                'has_parallelism': math.prod(self.workload.tensor_map[layer.output].shape) > len({u.chiplet for u in units})})
        windows = defaultdict(list)
        for name, rows in loads.items():
            event = events[name]
            for source, destination, amount, kind in rows:
                windows[(source, destination, kind)].append({'flow_id': name, 'start_s': event.start_s,
                    'finish_s': event.finish_s, 'transmitted_bytes': amount, 'queue_wait_s': waits[name]})
        network = [{'link_id': f'{a}>{b}', 'source': a, 'destination': b, 'kind': kind,
                    'bytes': sum(row['transmitted_bytes'] for row in rows), 'time_windows': rows,
                    'utilization': sum(row['finish_s'] - row['start_s'] for row in rows) / schedule.makespan_s,
                    'queue_wait_s': sum(row['queue_wait_s'] for row in rows)}
                   for (a, b, kind), rows in sorted(windows.items())]
        memory = [{'pool_id': name, 'capacity_bytes': buffers['capacity_by_pool'][name], 'peak_live_bytes': amount,
                   'spill_bytes': max(0, amount - buffers['capacity_by_pool'][name]),
                   'spill_status': 'rejected, not silently spilled' if amount > buffers['capacity_by_pool'][name] else 'none'}
                  for name, amount in buffers['peak_by_pool'].items()]
        return {'diagnostics_version': 'bottleneck_single_array_v1', 'evidence_quality': 'partial',
            'critical_path': [name for name in events if slack[name] is not None and slack[name] <= tolerance],
            'layers': layers, 'network': network, 'memory': memory,
            'flows': [{'flow_id': t.id, 'tensor_id': t.tensor, 'producer_units': t.producer_units,
                       'consumer_units': t.consumer_units, 'source': t.src, 'destinations': t.dsts} for t in communication.transfers],
            'fusion': [dict(row) for row in (compiled.execution_statistics or {}).get('stacks', ())],
            'unsupported': {'split_wait_attribution': 'atomic flow reservations cannot assign one queue wait to a unique resource',
                            'bank_conflicts': 'aggregate single RW ledger; physical address bank conflicts unsupported',
                            'packet_cycle_timing': 'finite-buffer conservative reservation, not cycle-accurate protocol'},
            'latency_breakdown': 'resource busy and queue times overlap and must not be added as E2E latency'}
