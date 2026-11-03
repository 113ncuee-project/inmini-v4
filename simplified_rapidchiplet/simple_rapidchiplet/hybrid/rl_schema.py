"""Versioned, explicit design decisions for candidate-based chiplet RL.

The user-facing historical list has eleven labels but ten transitions: A11 is
an import alias of A9 and never receives a second policy class or probability.
Derived schedules/flows are evaluator outputs, never independent decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Any

from .mapping import MappingPlan
from .search import plan_from_dict


SCHEMA_VERSION = 1
ATOMIC_ACTIONS = tuple(f"A{index}" for index in range(1, 11))
COMPOSITE_ACTIONS = tuple(f"C{index}" for index in range(1, 11))


def canonical_action_id(action_id: str) -> str:
    value = str(action_id).upper()
    value = "A9" if value == "A11" else value
    if value not in (*ATOMIC_ACTIONS, "A12"):
        raise ValueError(f"unknown atomic action {action_id}")
    return value


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DesignState:
    architecture_candidate_id: str
    mapping: MappingPlan
    model_hash: str
    hardware_template_version: str = "gemini-2x2-v1"
    version: int = SCHEMA_VERSION
    precision_config: dict = field(default_factory=lambda: {"activation_bits": 8, "weight_bits": 8, "psum_bits": 24})
    routing_config: dict = field(default_factory=lambda: {"algorithm": "template_deterministic"})
    scheduler_config: dict = field(default_factory=lambda: {"algorithm": "resource_calendar"})
    preferences: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    workload_definition: dict = field(default_factory=dict)
    evaluation_report: dict | None = None
    remaining_budget: int | None = None

    def to_dict(self):
        data = asdict(self)
        data["mapping"] = self.mapping.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        copied = dict(data)
        copied["mapping"] = plan_from_dict(copied["mapping"])
        return cls(**copied)

    @property
    def fingerprint(self):
        """Identity of decisions; reports/budget cannot invalidate exact PPA."""
        # Derived reports may be megabytes. Omit them before asdict performs
        # recursive copies, preserving the exact historical decision hash.
        data = replace(self,evaluation_report=None,remaining_budget=None).to_dict()
        data.pop("evaluation_report")
        data.pop("remaining_budget")
        if self.version == 2:
            # Human group labels do not alter resources, dependencies or PPA.
            # Keep the historical v1 identity byte-for-byte unchanged.
            for index, group in enumerate(data['mapping']['groups']):
                group['id'] = f'group{index}'
        return stable_hash(data)

    def with_report(self, report, remaining_budget=None):
        return replace(self, evaluation_report=report, remaining_budget=remaining_budget)


@dataclass(frozen=True)
class ActionEdit:
    action_id: str
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "action_id", canonical_action_id(self.action_id))
        object.__setattr__(self, "parameters", json.loads(json.dumps(self.parameters)))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(data["action_id"], dict(data.get("parameters", {})))


@dataclass(frozen=True)
class ActionPlan:
    id: str
    family_id: str
    base_state_hash: str
    ordered_edits: tuple[ActionEdit, ...]
    target_ids: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    dependency_closure: tuple[str, ...] = ("mapping", "memory", "flows", "network", "schedule", "ppa")
    proposal_reason: str = "generic exploration"
    source_tags: tuple[str, ...] = ()
    estimated_proxy_delta: float | None = None
    proxy_confidence: str = "unknown"
    evaluation_cost_estimate: int = 1

    def __post_init__(self):
        family = "A9" if self.family_id == "A11" else self.family_id
        if family not in (*ATOMIC_ACTIONS, "A12", *COMPOSITE_ACTIONS):
            raise ValueError(f"unknown action family {family}")
        if not self.ordered_edits:
            raise ValueError("action plan requires explicit ordered edits")
        object.__setattr__(self, "family_id", family)

    @property
    def primitive_or_composite(self):
        return "composite" if self.family_id.startswith("C") else "primitive"

    def to_dict(self):
        return json.loads(json.dumps({**asdict(self), "primitive_or_composite": self.primitive_or_composite}))

    @classmethod
    def from_dict(cls, data):
        copied = dict(data)
        copied.pop("primitive_or_composite", None)
        copied["ordered_edits"] = tuple(ActionEdit.from_dict(edit) for edit in copied["ordered_edits"])
        for key in ("target_ids", "preconditions", "dependency_closure", "source_tags"):
            if key in copied:
                copied[key] = tuple(copied[key])
        return cls(**copied)


def make_action_plan(state, family_id, edits, *, targets=(), reason="generic exploration", sources=()):
    edits = tuple(edit if isinstance(edit, ActionEdit) else ActionEdit(*edit) for edit in edits)
    family_id = "A9" if family_id == "A11" else family_id
    identity = stable_hash({"base": state.fingerprint, "edits": [edit.to_dict() for edit in edits]})[:20]
    return ActionPlan(identity, family_id, state.fingerprint, edits, tuple(targets),
                      proposal_reason=reason, source_tags=tuple(sources))


CAPABILITIES = {
    "A1": {"supported": True, "scope": "closed, contiguous single-exit fused stack occupying one pipeline group"},
    "A2": {"supported": True, "scope": "B/K/H/W output partition; uneven nonempty integer shards"},
    "A3": {"supported": True, "scope": "sink H/W DF tile; upstream regions derived by dependency compiler"},
    "A4": {"supported": True, "scope": "same spatial group transfer or legal idle resource acquisition/release"},
    "A5": {"supported": True, "scope": "ordered within-layer placement/idle relocation with unchanged quota"},
    "A6": {"supported": True, "scope": "equal-set swap between simultaneous stages of one spatial group"},
    "A7": {"supported": True, "scope": "(H,V)=(0,0),(1,0),(1,1); vertical-only unsupported"},
    "A8": {"supported": True, "scope": "paid external tensor/weight assignment including backend interleave"},
    "A9": {"supported": True, "scope": "design-time legal architecture template switch with explicit evacuation mapping"},
    "A10": {"supported": True, "scope": "topological pipeline groups and fixed-workload microbatch divisors"},
    "A11": {"supported": True, "alias_of": "A9", "independent_policy_class": False},
    "C10": {"supported": False, "reason": "optional communication-aware local partition DP is disabled"},
    "input_channel_partition": {"supported": False, "reason": "no package partial-sum reduction model"},
    "arbitrary_floorplan_or_routing": {"supported": False, "reason": "architecture/routing comes from legal templates"},
    "peak_power": {"supported": False, "reason": "only workload-window average power is modeled"},
}
