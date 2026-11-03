"""INMIN v0.2 physical contract, isolated from both historical engines.

This module describes resources and routes, not a completed PPA evaluator.
The provisional topology and protocol choices are versioned M assumptions.
No historical core/GLB hardware is instantiated by this contract.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import ClassVar

from .costs import NN_BATON, grs_port, sram_bank


PROFILE = "chiplet_single_pe_array_v0_2"
ENGINE_SCHEMA = "inmin_single_array_v0_2"
COUNTS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class ExperimentalAssumptions:
    """Explicit executable choices for previously T protocol parameters."""

    version: str = "single_array_assumptions_v2"
    on_die_topology: str = "pe_mesh_4x4"
    l2_gateway_pes: tuple[int, ...] = (0, 3, 12, 15)
    io_vertical_links: bool = True
    psum_storage_bytes: int = 3
    rounding: str = "nearest_ties_to_even"
    accumulator_overflow: str = "signed_int24_saturate_per_mac"
    activation_saturation: str = "signed_int8"
    router_inputs: int = 5
    virtual_channels: int = 2
    flits_per_vc: int = 8
    flit_bytes: int = 16
    packet_payload_bytes: int = 256
    packet_header_bytes: int = 16
    dma_queue_entries: int = 16
    dma_descriptor_bytes: int = 64
    io_queue_entries: int = 16
    io_descriptor_bytes: int = 64
    control_storage_bytes: int = 4096
    working_register_bytes_per_pe: int = 320
    vector_register_bytes: int = 192
    quantization_parameter_bytes: int = 4096
    dram_access_latency_s: float = 50e-9
    router_latency_cycles: int = 1
    dma_descriptor_cycles: int = 1
    external_dram_area_mm2_per_io: float = 5.0
    external_dram_idle_w_per_mm2: float = .005
    vector_cycles_per_element: tuple[tuple[str, int], ...] = (
        ("relu", 1), ("add", 1), ("max", 1), ("pool_compare", 1),
        ("reduction_add", 1), ("requant", 3),
    )

    def __post_init__(self):
        if self.on_die_topology != "pe_mesh_4x4":
            raise ValueError("unimplemented on-die topology; supply a routed implementation")
        gateways = self.l2_gateway_pes
        if not gateways or len(set(gateways)) != len(gateways) or any(
            type(pe) is not int or not 0 <= pe < 16 for pe in gateways
        ):
            raise ValueError("L2 gateways must be unique physical PE router indices")
        if self.psum_storage_bytes not in (3, 4):
            raise ValueError("INT24 storage requires packed 3B or aligned 4B")
        if (self.rounding, self.accumulator_overflow, self.activation_saturation) != (
            "nearest_ties_to_even", "signed_int24_saturate_per_mac", "signed_int8"
        ):
            raise ValueError("unsupported quantization semantics")
        for name in ("router_inputs", "virtual_channels", "flits_per_vc", "flit_bytes",
                     "packet_payload_bytes", "packet_header_bytes", "dma_queue_entries",
                     "dma_descriptor_bytes", "io_queue_entries", "io_descriptor_bytes",
                     "control_storage_bytes", "working_register_bytes_per_pe", "vector_register_bytes",
                     "quantization_parameter_bytes", "router_latency_cycles", "dma_descriptor_cycles"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.router_inputs != 5:
            raise ValueError("this implementation provisions five-port PE/NoP routers")
        if any(pe // 4 not in (0, 3) and pe % 4 not in (0, 3) for pe in gateways):
            raise ValueError('L2 gateway requires a spare boundary router port')
        if self.flit_bytes != 16 or self.quantization_parameter_bytes >= 128 * 1024:
            raise ValueError("unsupported flit width or quantization allocation")
        if not math.isfinite(self.dram_access_latency_s) or self.dram_access_latency_s <= 0:
            raise ValueError("DRAM latency must be finite and positive")
        if any(not math.isfinite(value) or value <= 0 for value in (
            self.external_dram_area_mm2_per_io, self.external_dram_idle_w_per_mm2
        )):
            raise ValueError("external DRAM analytical area/idle assumptions must be positive")
        operations = dict(self.vector_cycles_per_element)
        expected = {"relu", "add", "max", "pool_compare", "reduction_add", "requant"}
        if set(operations) != expected or len(operations) != len(self.vector_cycles_per_element) or any(
            type(cycles) is not int or cycles <= 0 for cycles in operations.values()
        ):
            raise ValueError("explicit positive cycle counts required for each supported vector operation")

    @property
    def fifo_bytes_per_router(self):
        return self.router_inputs * self.virtual_channels * self.flits_per_vc * self.flit_bytes

    def wire_bytes(self, payload_bytes):
        if type(payload_bytes) is not int or payload_bytes < 0:
            raise ValueError("payload byte count must be a nonnegative integer")
        packets = math.ceil(payload_bytes / self.packet_payload_bytes)
        return math.ceil(payload_bytes / self.flit_bytes) * self.flit_bytes + packets * self.packet_header_bytes


@dataclass(frozen=True)
class SingleArrayHardware:
    rows: int = 2
    cols: int = 2
    assumptions: ExperimentalAssumptions = field(default_factory=ExperimentalAssumptions)
    compute_clock_hz: float = 1e9
    noc_clock_hz: ClassVar[float] = 1e9
    nop_clock_hz: ClassVar[float] = 1e9
    sram_clock_hz: ClassVar[float] = 1e9
    profile_id: ClassVar[str] = PROFILE
    activation_bits: ClassVar[int] = 8
    weight_bits: ClassVar[int] = 8
    psum_bits: ClassVar[int] = 24
    # Fixed physical resources, deliberately not constructor parameters.
    pe_count: ClassVar[int] = 16
    lanes: ClassVar[int] = 16
    vector_size: ClassVar[int] = 16
    physical_cores_per_die: ClassVar[int] = 0
    vector_units_per_die: ClassVar[int] = 1
    vector_lanes: ClassVar[int] = 16
    dma_engines_per_die: ClassVar[int] = 1
    control_units_per_die: ClassVar[int] = 1
    prefetch_slots: ClassVar[int] = 2
    l1_weight_bytes: ClassVar[int] = 64 * 1024  # per PE per slot
    l1_activation_bytes: ClassVar[int] = 16 * 1024
    l1_output_bytes: ClassVar[int] = 8 * 1024
    l2_weight_bytes: ClassVar[int] = 512 * 1024  # total A+B, no second multiplication
    l2_activation_bytes: ClassVar[int] = 256 * 1024
    l2_output_bytes: ClassVar[int] = 256 * 1024
    unified_buffer_bytes: ClassVar[int] = 0
    core_global_buffer_bytes: ClassVar[int] = 0  # cost-helper compatibility only
    l1_bandwidth_Bps: ClassVar[float] = 64e9  # aggregate RW per PE
    l2_bandwidth_Bps: ClassVar[float] = 64e9  # aggregate RW per die
    noc_bandwidth_Bps: ClassVar[float] = 16e9  # each directed link
    nop_bandwidth_Bps: ClassVar[float] = 12.5e9
    dma_bandwidth_Bps: ClassVar[float] = 16e9
    router_bandwidth_Bps: ClassVar[float] = 64e9
    dram_bandwidth_Bps: ClassVar[float] = 8.192e9  # per IO, one controller ledger
    dram_capacity_bytes: ClassVar[int] = 512 * 1024**2
    d2d_ports_per_die: ClassVar[int] = 4
    # One explicitly uncalibrated 16nm reference coefficient table. Reuses the
    # existing bank interpolation, without creating any old physical hardware.
    cost_profile: ClassVar[str] = "literature16_v1"
    reference_process_nm: ClassVar[int] = 16
    sram_word_bits: ClassVar[int] = 128
    sram_rw_ports: ClassVar[int] = 1
    sram_max_bank_bytes: ClassVar[int] = 32 * 1024
    sram_l1_anchor_pj_per_bit: ClassVar[float] = .30
    sram_energy_pj_per_bit: ClassVar[float] = .81
    sram_width_energy_exponent: ClassVar[float] = .15
    sram_extra_port_energy_factor: ClassVar[float] = .3
    sram_energy_scale: ClassVar[float] = 1.0
    sram_write_energy_factor: ClassVar[float] = 1.0
    sram_area_mm2_per_byte: ClassVar[float] = .55 / 1e6
    sram_bank_overhead_mm2: ClassVar[float] = .001
    sram_area_scale: ClassVar[float] = 1.0
    grs_data_lanes: ClassVar[int] = 8
    grs_lane_rate_bps: ClassVar[float] = 25e9
    grs_phy_area_mm2: ClassVar[float] = .387590
    grs_active_area_mm2: ClassVar[float] = .081406

    def __post_init__(self):
        if type(self.rows) is not int or type(self.cols) is not int or self.rows <= 0 or self.cols <= 0:
            raise ValueError("rectangular layout requires positive integer dimensions")
        if self.compute_count not in COUNTS:
            raise ValueError("legal compute counts are 1/2/4/8/16")
        if self.compute_clock_hz not in (.5e9, 1e9):
            raise ValueError("only versioned 0.5GHz sensitivity and 1GHz target are supported")

    @property
    def compute_count(self):
        return self.rows * self.cols

    @property
    def io_count(self):
        return 2 * self.rows

    @property
    def macs_per_cycle(self):
        return self.pe_count * self.lanes * self.vector_size

    @property
    def l1_bytes_per_die(self):
        return self.pe_count * self.prefetch_slots * (self.l1_weight_bytes + self.l1_activation_bytes + self.l1_output_bytes)

    @property
    def l2_bytes_per_die(self):
        return self.l2_weight_bytes + self.l2_activation_bytes + self.l2_output_bytes

    @property
    def compute_slots(self):
        return tuple(f"compute:{die}/pe:{pe}" for die in range(self.compute_count) for pe in range(16))

    @property
    def architecture_id(self):
        return f"single_array_c{self.compute_count}_{self.rows}x{self.cols}"

    @property
    def sram_costs(self):
        # MAC/VPU frequency sensitivity does not create extra SRAM macros.
        # SRAM/NoC remain at a fixed 1GHz M clock; no CDC signoff is claimed.
        reference = self if self.compute_clock_hz == 1e9 else type(self)(self.rows, self.cols, self.assumptions, 1e9)
        return {name: sram_bank(reference, name) for name in ('WL1', 'AL1', 'OL1', 'WL2', 'AL2', 'OL2')}

    @property
    def resource_totals(self):
        return {
            "compute_dies": self.compute_count, "io_dies": self.io_count,
            "physical_cores": 0, "pes": 16 * self.compute_count,
            "vector_units": self.compute_count, "dma_engines": self.compute_count,
            "l1_bytes": self.compute_count * self.l1_bytes_per_die,
            "l2_bytes": self.compute_count * self.l2_bytes_per_die, "additional_ub_bytes": 0,
            "main_data_sram_bytes": self.compute_count * (self.l1_bytes_per_die + self.l2_bytes_per_die),
            "peak_tops": 2 * self.compute_count * self.macs_per_cycle * self.compute_clock_hz / 1e12,
            "dram_capacity_bytes": self.io_count * self.dram_capacity_bytes,
            "dram_bandwidth_Bps": self.io_count * self.dram_bandwidth_Bps,
            "reserved_d2d_ports": 4 * (self.compute_count + self.io_count),
        }

    @property
    def area_estimate(self):
        """Costs include unused PHYs and separate FIFO/register storage.

        Buffer storage uses an explicit M area slope, including registers as a
        conservative 4x SRAM proxy. This is not a fitted PDK register macro.
        """
        a = self.assumptions
        banks = self.sram_costs
        storage_cost = lambda amount: amount * self.sram_area_mm2_per_byte + math.ceil(amount / 1024) * self.sram_bank_overhead_mm2
        # 16 PE routers plus one separate NoP router. Gateway ports are already
        # among the five provisioned PE-router inputs, not extra free switches.
        compute = {
            "l1_sram": 32 * sum(banks[name]["area_mm2"] for name in ("WL1", "AL1", "OL1")),
            "l2_sram": sum(banks[name]["area_mm2"] for name in ("WL2", "AL2", "OL2")),
            "scalar_macs": self.macs_per_cycle * 135.1e-6,
            "vector_unit": self.vector_lanes * 135.1e-6,
            "dma": .012, "control": .109,
            "pe_router_switches": 16 * .012 * self.noc_bandwidth_Bps / 8.5e9,
            "nop_router_switch": .042 * self.router_bandwidth_Bps / 50e9,
            "router_fifos": 17 * storage_cost(a.fifo_bytes_per_router),
            "dma_descriptors": storage_cost(a.dma_queue_entries * a.dma_descriptor_bytes),
            "control_storage": storage_cost(a.control_storage_bytes),
            "pe_working_registers": 16 * 4 * storage_cost(a.working_register_bytes_per_pe),
            "vector_working_registers": 4 * storage_cost(a.vector_register_bytes),
            "reserved_d2d_phys": 4 * grs_port(self)["footprint_mm2"],
        }
        io = {
            "dram_controller_phy": 10.5 * self.dram_bandwidth_Bps / 18.75e9,
            "nop_router_switch": .042 * self.router_bandwidth_Bps / 50e9,
            "router_fifos": storage_cost(a.fifo_bytes_per_router),
            "request_descriptors": storage_cost(a.io_queue_entries * a.io_descriptor_bytes),
            "reserved_d2d_phys": 4 * grs_port(self)["footprint_mm2"],
        }
        return {"compute_per_die_mm2": compute, "io_per_die_mm2": io,
                "compute_total_mm2": self.compute_count * sum(compute.values()),
                "io_total_mm2": self.io_count * sum(io.values()),
                "logic_total_mm2": self.compute_count * sum(compute.values()) + self.io_count * sum(io.values()),
                "external_dram_area_mm2": self.io_count * a.external_dram_area_mm2_per_io,
                "total_silicon_mm2": self.compute_count * sum(compute.values()) + self.io_count * sum(io.values()) + self.io_count * a.external_dram_area_mm2_per_io,
                "external_dram_status": "M: 5mm2 per 512MiB default linear capacity proxy from historical 20mm2/2GiB assumption; actual device remains T",
                "status": "M: uncalibrated 16nm analytical estimate; no timing or physical signoff"}

    def to_dict(self):
        return {
            "profile_id": self.profile_id, "engine_schema": getattr(self, "engine_schema", ENGINE_SCHEMA),
            "architecture_id": self.architecture_id, "layout": [self.rows, self.cols],
            "compute_clock_hz": self.compute_clock_hz,
            "noc_clock_hz": self.noc_clock_hz, "nop_clock_hz": self.nop_clock_hz, "sram_clock_hz": self.sram_clock_hz,
            "physical_resources": {name: getattr(self, name) for name in (
                "pe_count", "lanes", "vector_size", "physical_cores_per_die", "vector_units_per_die",
                "vector_lanes", "dma_engines_per_die", "control_units_per_die", "prefetch_slots",
                "l1_weight_bytes", "l1_activation_bytes", "l1_output_bytes", "l2_weight_bytes",
                "l2_activation_bytes", "l2_output_bytes", "unified_buffer_bytes", "d2d_ports_per_die")},
            "services_Bps": {name: getattr(self, name) for name in (
                "l1_bandwidth_Bps", "l2_bandwidth_Bps", "noc_bandwidth_Bps", "nop_bandwidth_Bps",
                "dma_bandwidth_Bps", "router_bandwidth_Bps", "dram_bandwidth_Bps")},
            "precision": {"activation_bits": 8, "weight_bits": 8, "psum_bits": 24,
                          "psum_storage_bytes": self.assumptions.psum_storage_bytes,
                          "non_MAC_scale_assumption": "M: common fixed input scale; INT8 Add is a saturating lane operation; quantized numerical accuracy not validated"},
            "resource_totals": self.resource_totals,
            "area_estimate": self.area_estimate,
            "cost_coefficients": {
                "classification": "M: anchor adaptations and uncalibrated analytical assumptions",
                "reference_process_nm": 16, "scalar_mac_area_mm2": 135.1e-6,
                "scalar_mac_energy_pj": .024,
                "MAC_energy_convention": "assume one reported NN-Baton op per scalar MAC; source ambiguity, 2x sensitivity not yet run; TOPS still counts two arithmetic ops/MAC",
                "vector_area_and_energy": "16 scalar-MAC area proxies; .024pJ per modeled lane operation/cycle",
                "noc_pj_per_bit": .4, "nop_pj_per_bit": 1.17, "router_pj_per_bit": .1, "dram_pj_per_bit": 8.75,
                "compute_idle_w_per_mm2": .02, "io_idle_w_per_mm2": .008,
                "sram_area_mm2_per_byte": self.sram_area_mm2_per_byte,
                "sram_bank_overhead_mm2": self.sram_bank_overhead_mm2,
                "SRAM_model": "existing versioned literature16_v1 bank interpolation; fixed 1GHz physical provisioning for frequency sensitivity",
                "SRAM_transaction_scope": "M: packed logical data-byte energy/service, shared aggregate RW reservation; physical byte-enable, word-tail and bank-address effects not calibrated",
                "GRS_endpoint": grs_port(self),
                "register_area": "M: four times the explicit SRAM storage-area proxy; not a register macro measurement",
                "IO_controller_area": "M: 10.5mm2 * effective GB/s / 18.75GB/s linear proxy",
            },
            "experimental_assumptions": {"classification": "M", **asdict(self.assumptions)},
            "provenance": {
                "main_spec": "Chiplet_Hardware_Spec_v0.2_reviewed.md",
                "I_D": "16 PE, 16x16 scalar MAC/PE; INT8/INT24; 12.5GB/s D2D reference",
                "M": "L1 double buffers; 1MiB shared L2; L1/L2 service; vector/DMA; gateway; packet/queue/latency/rounding; IO geometry/DRAM; all uncalibrated component scaling",
                "cost_anchor": NN_BATON, "cost_reference_process_nm": 16,
                "cost_model": "single_array_cost_assumptions_v2",
                "T": ["physical DRAM device/area and pinout", "DDR channel timing/refresh", "PHY geometry and package signoff", "SRAM macro banking/timing", "cycle-accurate VC deadlock proof"],
                "scope": "Batch1 inference; host/model loading excluded; analytical nominal resource contract",
            },
        }

    @classmethod
    def from_dict(cls, data):
        if data.get('profile_id') == PROFILE_V3 and cls is SingleArrayHardware:
            return SingleArrayHardwareV3.from_dict(data)
        if data.get('profile_id') != cls.profile_id or data.get('engine_schema') != getattr(cls, 'engine_schema', ENGINE_SCHEMA):
            raise ValueError('not a supported single-array hardware profile')
        assumptions = dict(data['experimental_assumptions'])
        assumptions.pop('classification', None)
        assumptions['l2_gateway_pes'] = tuple(assumptions['l2_gateway_pes'])
        assumptions['vector_cycles_per_element'] = tuple(tuple(row) for row in assumptions['vector_cycles_per_element'])
        spec = cls(*data['layout'], ExperimentalAssumptions(**assumptions), data['compute_clock_hz'])
        # Fixed resources, services, costs and derived totals are evidence, not
        # ignored hints. Reject edited or incompatible snapshots explicitly.
        if json.loads(json.dumps(spec.to_dict(), sort_keys=True)) != json.loads(json.dumps(data, sort_keys=True)):
            raise ValueError('hardware snapshot differs from the current versioned contract')
        return spec

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PhysicalLink:
    source: str
    destination: str
    source_port: str
    destination_port: str
    bandwidth_Bps: float
    kind: str


@dataclass(frozen=True)
class SingleArrayNetwork:
    coordinates: tuple[tuple[str, tuple[int, int]], ...]
    links: tuple[PhysicalLink, ...]
    memory_owners: tuple[tuple[str, str], ...] = ()
    compute_slots: tuple[str, ...] = ()

    @property
    def mapping_core_ids(self):
        """Historical parser alias, carrying physical PE endpoints only."""
        return self.compute_slots

    @property
    def memory_ids(self):
        return tuple(name for name, _ in self.memory_owners)

    @property
    def compute_ids(self):
        return tuple(name for name, _ in self.coordinates if name.startswith('compute:') and '/' not in name)

    @property
    def io_ids(self):
        return tuple(name for name, _ in self.coordinates if name.startswith('io:'))

    def route(self, source, destination):
        adjacency = {name: [] for name, _ in self.coordinates}
        for link in self.links:
            adjacency[link.source].append(link.destination)
            adjacency[link.destination].append(link.source)
        if source not in adjacency or destination not in adjacency:
            raise ValueError("route endpoint is not a physical network node")
        # Deterministic shortest path. This finite, whole-flow reservation model
        # makes no claim to a cycle-accurate VC protocol or deadlock proof.
        queue, previous = deque([source]), {source: None}
        while queue:
            current = queue.popleft()
            if current == destination:
                path = []
                while current is not None:
                    path.append(current)
                    current = previous[current]
                return tuple(reversed(path))
            for neighbor in sorted(adjacency[current]):
                if neighbor not in previous:
                    previous[neighbor] = current
                    queue.append(neighbor)
        raise ValueError("no connected physical path")


def enumerate_single_array_hardware(compute_counts=COUNTS):
    candidates = []
    for count in compute_counts:
        if type(count) is not int or count not in COUNTS:
            raise ValueError("legal compute counts are 1/2/4/8/16")
        candidates.extend(SingleArrayHardware(rows, count // rows)
                          for rows in range(1, count + 1) if count % rows == 0)
    if not candidates or len({item.architecture_id for item in candidates}) != len(candidates):
        raise ValueError("architecture set must be nonempty and unique")
    return tuple(candidates)


@dataclass(frozen=True)
class SingleArrayCandidate:
    hardware: SingleArrayHardware

    @property
    def id(self):
        return self.hardware.architecture_id

    @property
    def package(self):
        return build_single_array_package(self.hardware)

    def to_dict(self):
        return {'id': self.id, 'hardware': self.hardware.to_dict(),
                'chiplet_profile': self.hardware.profile_id, 'resource_totals': self.hardware.resource_totals}


def enumerate_single_array_architectures(compute_counts=COUNTS):
    return tuple(SingleArrayCandidate(hardware) for hardware in enumerate_single_array_hardware(compute_counts))


def build_single_array_package(spec):
    coordinates, links, owners = {}, [], []
    for row in range(spec.rows):
        for col in range(spec.cols):
            coordinates[f"compute:{row * spec.cols + col}"] = (row, col + 1)
        for side, col in enumerate((0, spec.cols + 1)):
            io = f"io:{2 * row + side}"
            coordinates[io] = (row, col)
            memory = f"memory:{2 * row + side}"
            owners.append((memory, io))
    by_coordinate = {position: name for name, position in coordinates.items()}
    for name, (row, col) in coordinates.items():
        for dr, dc, port, opposite in ((0, 1, "east", "west"), (1, 0, "south", "north")):
            other = by_coordinate.get((row + dr, col + dc))
            if other is None:
                continue
            if name.startswith("io:") and other.startswith("io:") and not spec.assumptions.io_vertical_links:
                continue
            links.append(PhysicalLink(name, other, port, opposite, spec.nop_bandwidth_Bps, "d2d_single_full_duplex"))
    for memory, io in owners:
        coordinates[memory] = coordinates[io]
        links.append(PhysicalLink(memory, io, "memory", "controller", spec.dram_bandwidth_Bps, "dram_service"))
    return SingleArrayNetwork(tuple(coordinates.items()), tuple(links), tuple(owners), spec.compute_slots)


def build_single_array_noc(spec, die=0):
    if type(die) is not int or not 0 <= die < spec.compute_count:
        raise ValueError("unknown compute die")
    pe_name = lambda pe: f"compute:{die}/pe:{pe}"
    coordinates = {pe_name(pe): divmod(pe, 4) for pe in range(16)}
    links = []
    for pe in range(16):
        row, col = divmod(pe, 4)
        for other, valid, port, opposite in ((pe + 1, col < 3, "east", "west"),
                                             (pe + 4, row < 3, "south", "north")):
            if valid:
                links.append(PhysicalLink(pe_name(pe), pe_name(other), port, opposite, spec.noc_bandwidth_Bps, "pe_mesh"))
    for index, pe in enumerate(spec.assumptions.l2_gateway_pes):
        gateway = f"compute:{die}/l2_gateway:{index}"
        coordinates[gateway] = divmod(pe, 4)
        links.append(PhysicalLink(pe_name(pe), gateway, "l2_gateway", f"gateway:{index}", spec.noc_bandwidth_Bps, "l2_gateway"))
    return SingleArrayNetwork(tuple(coordinates.items()), tuple(links))


PROFILE_V3 = "chiplet_single_pe_array_v0_3"

@dataclass(frozen=True)
class SingleArrayHardwareV3(SingleArrayHardware):
    """Same priced resources, versioned integer-count architecture domain."""
    profile_id: ClassVar[str] = PROFILE_V3
    engine_schema: ClassVar[str] = "inmin_single_array_v0_3"

    def __post_init__(self):
        if type(self.rows) is not int or type(self.cols) is not int or self.rows < 1 or self.cols < 1 or not 1 <= self.compute_count <= 16:
            raise ValueError("v0.3 requires a complete rectangle with 1..16 Compute dies")
        if self.compute_clock_hz not in (.5e9, 1e9):
            raise ValueError("unsupported compute clock")

def enumerate_single_array_architectures_v3():
    return tuple(SingleArrayCandidate(SingleArrayHardwareV3(rows, count // rows))
                 for count in range(1, 17) for rows in range(1, count + 1) if count % rows == 0)
