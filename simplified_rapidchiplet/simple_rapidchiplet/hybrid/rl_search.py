"""Joint architecture/mapping actor-critic with exact transactions and replay.

Policy selection is a masked softmax over stable candidate features. It learns
from physical cost differences; diagnostic priors affect proposals only. The
behavior probability is stored for each sample, including random baselines.
All legal evaluated trials commit even when worse. Best feasible designs live
in a separate archive, and no infeasible design is called a recommendation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import gzip
import hashlib
import math
from pathlib import Path
import random
import time

import numpy as np

from .rl_actions import (_candidate_map, _package, apply_transaction, generate_candidates,
                         validate_state)
from .rl_schema import ATOMIC_ACTIONS, CAPABILITIES, DesignState, stable_hash
from .search import LinearActorCritic, _jsonable, _rng_tuple


CHECKPOINT_SCHEMA = 1
STATE_SIZE = 16
ACTION_SIZE = 86
STATE_SIZE_V2 = 24
ACTION_SIZE_V2 = 180
ACTION_SIZE_V3 = 283


def _positive(metrics, key):
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"physical report requires positive finite {key}")
    return float(value)


@dataclass(frozen=True)
class JointObjective:
    latency_weight: float = 1.0
    area_weight: float = 1.0
    power_weight: float = 1.0
    energy_weight: float = 0.0
    max_area_mm2: float | None = None
    max_power_w: float | None = None
    max_latency_s: float | None = None
    soft_area_mm2: float | None = None
    soft_power_w: float | None = None
    soft_latency_s: float | None = None
    violation_penalty: float = 10.0
    allow_infeasible_exploration: bool = True

    def __post_init__(self):
        weights = (self.latency_weight, self.area_weight, self.power_weight, self.energy_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
            raise ValueError("objective needs nonnegative weights and a positive total")
        for value in (self.max_area_mm2, self.max_power_w, self.max_latency_s,
                      self.soft_area_mm2, self.soft_power_w, self.soft_latency_s):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("PPA limits/goals must be positive finite values")
        if not math.isfinite(self.violation_penalty) or self.violation_penalty < 0:
            raise ValueError("violation penalty must be finite and nonnegative")

    def violations(self, metrics):
        return {key: max(0.0, _positive(metrics, key) / limit - 1)
                for key, limit in (("latency_s", self.max_latency_s), ("area_mm2", self.max_area_mm2),
                                   ("power_w", self.max_power_w)) if limit is not None}

    def violation(self, metrics):
        return sum(self.violations(metrics).values())

    def strict_feasible(self, metrics):
        return executable(metrics) and not any(self.violations(metrics).values())

    def cost(self, metrics, reference):
        weights = (("latency_s", self.latency_weight), ("area_mm2", self.area_weight),
                   ("power_w", self.power_weight), ("energy_j", self.energy_weight))
        total = sum(weight for _, weight in weights)
        cost = sum(weight / total * _positive(metrics, key) / _positive(reference, key)
                   for key, weight in weights if weight)
        # Soft goals remain soft. They do not enter the hard-feasible mask.
        for key, goal in (("latency_s", self.soft_latency_s), ("area_mm2", self.soft_area_mm2),
                          ("power_w", self.soft_power_w)):
            if goal is not None:
                cost += max(0.0, _positive(metrics, key) / goal - 1)
        return cost

    def penalized_cost(self, metrics, reference):
        return self.cost(metrics, reference) + self.violation_penalty * self.violation(metrics)


@dataclass(frozen=True)
class JointSearchConfig:
    search_space: str = "joint"
    generator_guidance: bool = True
    bottleneck_features: bool = True
    seed: int = 7
    budget: int = 128
    max_steps: int = 128
    horizon: int = 16
    shortlist: int = 48
    random_quota: int = 12
    max_per_family: int = 12
    proposal: str = "composite"
    method: str = "rl"
    actor_lr: float = .03
    value_lr: float = .03
    entropy: float = .005
    checkpoint_every: int = 8
    fusion_enabled: bool = True
    max_fusion_depth: int = 4
    tile_sizes: tuple[tuple[int, int], ...] = ((8, 8), (16, 16))
    max_wall_time_s: float | None = None

    def __post_init__(self):
        from .search_space import SPACES
        if self.search_space not in SPACES:
            raise ValueError("unknown search space")
        if any(type(value) is not int or value < 1 for value in
               (self.budget, self.max_steps, self.horizon, self.shortlist, self.random_quota,
                self.max_per_family, self.checkpoint_every, self.max_fusion_depth)):
            raise ValueError("search budgets and candidate bounds must be positive integers")
        if self.method not in {"rl", "random", "greedy"}:
            raise ValueError("method must be rl, random, or greedy")
        if self.proposal not in {"atomic", "guided_atomic", "composite"}:
            raise ValueError("unknown candidate proposal ablation")
        if self.proposal == "composite" and self.shortlist < 2:
            raise ValueError("composite shortlist must be at least 2 to reserve generic and macro candidates")
        if self.actor_lr <= 0 or self.value_lr <= 0 or self.entropy < 0:
            raise ValueError("invalid actor/critic learning parameters")
        if not self.tile_sizes or any(len(tile) != 2 or any(type(v) is not int or v < 1 for v in tile)
                                      for tile in self.tile_sizes):
            raise ValueError("DF tile bounds must contain positive H/W pairs")
        if self.max_wall_time_s is not None and self.max_wall_time_s <= 0:
            raise ValueError("wall time budget must be positive")


def executable(metrics):
    if metrics.get("executable", metrics.get("feasible", True)) is False:
        return False
    try:
        for key in ("latency_s", "energy_j", "area_mm2", "power_w"):
            _positive(metrics, key)
    except ValueError:
        return False
    return True


def compute_reward(old_metrics, new_metrics, reference, objective):
    """gamma=1 fixed-horizon physical improvement, with no action bonuses."""
    return objective.penalized_cost(old_metrics, reference) - objective.penalized_cost(new_metrics, reference)


@dataclass(frozen=True)
class StepResult:
    new_state: DesignState
    reward: float
    terminated: bool
    truncated: bool
    valid: bool
    rejection_reason: str | None
    realized_action_plan: dict
    evaluation_report: dict
    best_feasible_solution_id: str | None
    cache_hit: bool = False
    evaluation_elapsed_s: float = 0.0


class ExactEvaluationCache:
    """Full-system cache: hardware, workload, precision, mapper and all edits.

    No incremental result is reused across placement/routing/schedule changes.
    A9 always gets the complete target architecture in its exact key.
    """
    def __init__(self, candidates, evaluator_factory, evaluator_identity, *, cache=None,
                 evaluations=0, requests=0, elapsed_s=0.0, timeouts=0):
        self.candidates = _candidate_map(candidates)
        self.factory, self.evaluator_identity = evaluator_factory, evaluator_identity
        self.cache = dict(cache or {})
        self.invalid_design_hashes = {metrics["design_state_hash"] for metrics in self.cache.values()
                                      if not executable(metrics) and "design_state_hash" in metrics}
        self.evaluations, self.requests, self.elapsed_s = evaluations, requests, elapsed_s
        self.timeouts = timeouts
        self.oracles = {}
        self.observer = None
        self.context = {}

    def key(self, state):
        return stable_hash({"design": state.fingerprint,
                            "hardware": self.candidates[state.architecture_candidate_id].hardware.to_dict(),
                            "evaluator": self.evaluator_identity, "full_system": True})

    def get(self, state, budget):
        key = self.key(state)
        if key in self.cache:
            self.requests += 1
            if self.observer:
                self.observer(self.context, state, self.cache[key], "cache_hit", 0.)
            return self.cache[key], True, 0.0
        if self.evaluations >= budget:
            return None, False, 0.0
        self.requests += 1
        started = time.perf_counter()
        candidate_id = state.architecture_candidate_id
        try:
            if candidate_id not in self.oracles:
                self.oracles[candidate_id] = self.factory(self.candidates[candidate_id])
            oracle = self.oracles[candidate_id]
            result = oracle(state.mapping) if callable(oracle) else oracle.evaluate(state.mapping)
            metrics = dict(result if isinstance(result, dict) else result.metrics)
            metrics = _jsonable(metrics)
            metrics["executable"] = executable(metrics)
        except ValueError as error:
            metrics = {"executable": False, "feasible": False, "reason": str(error)}
        except TimeoutError:
            if self.observer:
                self.observer(self.context, state, None, "timeout", time.perf_counter() - started)
            self.timeouts += 1
            self.elapsed_s += time.perf_counter() - started
            raise
        metrics["design_state_hash"] = state.fingerprint
        if not executable(metrics):
            self.invalid_design_hashes.add(state.fingerprint)
        elapsed = time.perf_counter() - started
        self.elapsed_s += elapsed
        self.evaluations += 1
        self.cache[key] = metrics
        if self.observer:
            self.observer(self.context, state, metrics, "oracle_response", elapsed)
        return metrics, False, elapsed


class JointDesignEnvironment:
    def __init__(self, workload, initial_state, architecture_candidates, evaluator_factory, *,
                 objective=None, budget=128, evaluator_identity="unspecified", cache=None, reference=None, search_space="joint"):
        self.search_space = search_space
        self.workload, self.candidates = workload, _candidate_map(architecture_candidates)
        self.objective, self.budget = objective or JointObjective(), budget
        validate_state(workload, initial_state, self.candidates)
        self.oracle = cache or ExactEvaluationCache(self.candidates, evaluator_factory, evaluator_identity)
        metrics, _, _ = self.oracle.get(initial_state, budget)
        if metrics is None or not executable(metrics):
            raise ValueError("initial design must be executable with valid physical PPA; " + str((metrics or {}).get("reason", "evaluation budget exhausted")))
        if not self.objective.allow_infeasible_exploration and not self.objective.strict_feasible(metrics):
            raise ValueError("initial design violates strict PPA limits and infeasible exploration is disabled")
        self.reference = dict(reference or metrics)
        self.current_state, self.current_metrics = initial_state.with_report(metrics), metrics
        self.best_state, self.best_metrics, self.best_cost = None, None, None
        self.closest_state, self.closest_metrics, self.closest_rank = None, None, None
        self.architecture_archive = {}
        self.update_archive(initial_state, metrics)

    def update_archive(self, state, metrics):
        if not executable(metrics):
            return
        cost = self.objective.cost(metrics, self.reference)
        violation = self.objective.violation(metrics)
        strict = self.objective.strict_feasible(metrics)
        state = state.with_report(metrics)
        rank = (violation, cost)
        if self.closest_rank is None or rank < self.closest_rank:
            self.closest_state, self.closest_metrics, self.closest_rank = state, metrics, rank
        if strict and (self.best_cost is None or cost < self.best_cost):
            self.best_state, self.best_metrics, self.best_cost = state, metrics, cost
        candidate_id = state.architecture_candidate_id
        old = self.architecture_archive.get(candidate_id)
        if old is None or (not strict, violation if not strict else 0.0, cost) < (
                not old["strict_feasible"], old["violation"] if not old["strict_feasible"] else 0.0, old["cost"]):
            self.architecture_archive[candidate_id] = {"state": state.to_dict(), "metrics": metrics,
                                                       "cost": cost, "violation": violation,
                                                       "strict_feasible": strict}

    def step(self, plan):
        old_state, old_metrics = self.current_state, self.current_metrics
        try:
            trial = apply_transaction(self.workload, old_state, plan, self.candidates)
            from .search_space import check_transition
            check_transition(old_state, trial, self.search_space, plan)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            return StepResult(old_state, 0.0, False, False, False, str(error), plan.to_dict(), old_metrics,
                              self.best_state.fingerprint if self.best_state else None)
        self.oracle.context = {"stage": "selected", "action_plan": plan.to_dict(), "before_state_hash": old_state.fingerprint,
            "before_cost": self.objective.cost(old_metrics, self.reference), "best_cost_before": self.best_cost}
        metrics, hit, elapsed = self.oracle.get(trial, self.budget)
        if metrics is None:
            return StepResult(old_state, 0.0, False, True, False, "evaluation_budget_exhausted", plan.to_dict(), old_metrics,
                              self.best_state.fingerprint if self.best_state else None)
        if not executable(metrics):
            return StepResult(old_state, 0.0, False, False, False,
                              metrics.get("reason", "physical execution or memory plan is invalid"),
                              plan.to_dict(), metrics, self.best_state.fingerprint if self.best_state else None, hit, elapsed)
        self.update_archive(trial, metrics)
        if not self.objective.allow_infeasible_exploration and not self.objective.strict_feasible(metrics):
            return StepResult(old_state, 0.0, False, False, False, "PPA exploration outside hard limits is disabled",
                              plan.to_dict(), metrics, self.best_state.fingerprint if self.best_state else None, hit, elapsed)
        reward = compute_reward(old_metrics, metrics, self.reference, self.objective)
        # An executable worse state commits. There is no SA acceptance rule.
        trial = trial.with_report(metrics, self.budget - self.oracle.evaluations)
        self.current_state, self.current_metrics = trial, metrics
        return StepResult(trial, reward, False, False, True, None, plan.to_dict(), metrics,
                          self.best_state.fingerprint if self.best_state else None, hit, elapsed)


def state_features(state, metrics, reference, objective, candidate, max_cores, episode_progress):
    ppa = [math.log(max(_positive(metrics, key) / _positive(reference, key), 1e-30))
           for key in ("latency_s", "area_mm2", "power_w", "energy_j")]
    count = len(_package(candidate.hardware).mapping_core_ids)
    diagnostics = metrics.get("diagnostics", {})
    layers = diagnostics.get("layers", [])
    if isinstance(layers, dict):
        layers = list(layers.values())
    kinds = [row.get("bottleneck", "unknown") for row in layers]
    mapping = state.mapping
    vector = [1.0, *ppa, objective.violation(metrics), count / max_cores,
                               len(mapping.groups) / max(len(mapping.layer_mappings), 1),
                               len(mapping.fusion_stacks) / max(len(mapping.groups), 1),
                               mapping.microbatch_size / max(state.workload_definition.get("batch_size", mapping.microbatch_size), 1),
                               kinds.count("compute") / max(len(kinds), 1),
                               kinds.count("network") / max(len(kinds), 1),
                               kinds.count("memory") / max(len(kinds), 1),
                               float(episode_progress), objective.cost(metrics, reference),
                               float(bool(layers))]
    if state.version >= 2:
        network = diagnostics.get('network', [])
        memory = diagnostics.get('memory', [])
        resources = metrics.get('resource_busy_s', {})
        latency = _positive(metrics, 'latency_s')
        vector.extend([
            sum(row.get('pe_utilization', 0.) or 0. for row in layers) / max(1, len(layers)),
            sum(row.get('criticality', 0.) or 0. for row in layers) / max(1, len(layers)),
            max((row.get('queue_wait_s', 0.) / latency for row in network), default=0.),
            max((row.get('utilization', 0.) for row in network), default=0.),
            max((row.get('peak_live_bytes', 0.) / max(row.get('capacity_bytes', 1), 1) for row in memory), default=0.),
            max((busy / latency for resource, busy in resources.items() if resource.endswith(':controller')), default=0.),
            max((busy / latency for resource, busy in resources.items() if resource.endswith(':vpu')), default=0.),
            max((busy / latency for resource, busy in resources.items() if resource.endswith(':l2_rw')), default=0.),
        ])
    return np.clip(np.asarray(vector, dtype=np.float64), -8, 8)


def candidate_features(state, plans, features, candidates, workload, report=None):
    """Stable action embeddings: no candidate index, architecture-ID table,
    random ordering, or hidden heuristic score enters the learned policy.
    """
    source = candidates[state.architecture_candidate_id]
    source_count = len(_package(source.hardware).mapping_core_ids)
    all_macs = max(sum(layer.macs_per_sample for layer in workload.layers), 1)
    vectors = []
    for plan in plans:
        one_hot = np.asarray([float(any(edit.action_id == action for edit in plan.ordered_edits)) for action in ATOMIC_ACTIONS])
        macro = np.asarray([float(plan.family_id == f"C{index}") for index in range(1, 10)])
        target_count = source_count
        partition_delta = [0.0] * 4
        tile_delta, cache_delta, transfer = [0.0, 0.0], [0.0, 0.0], 0.0
        for edit in plan.ordered_edits:
            params = edit.parameters
            if edit.action_id == "A9":
                target_count = len(_package(candidates[params["architecture_candidate_id"]].hardware).mapping_core_ids)
            elif edit.action_id == "A2":
                previous = state.mapping.mapping_map[params["layer_id"]].partition.as_tuple()
                partition_delta = [math.log(a / b) for a, b in zip(params["partition"], previous)]
            elif edit.action_id == "A3":
                old = next((stack for stack in state.mapping.fusion_stacks if tuple(stack.layer_ids) == tuple(params["layer_ids"])), None)
                if old:
                    tile_delta = [math.log(a / b) for a, b in zip(params["tile_hw"], old.tile_hw)]
            elif edit.action_id == "A7":
                cache_delta = [float(params["cache_h"]), float(params["cache_v"])]
            elif edit.action_id == "A4":
                transfer += params.get("mapping_action", {}).get("core_delta", 0) / max(source_count, 1)
            elif state.version >= 2 and edit.action_id == 'A1' and params.get('operation', 'merge') == 'merge':
                cache_delta = [float(params.get('cache_h', False)), float(params.get('cache_v', False))]
        targets = [target for target in plan.target_ids if target in workload.layer_map]
        target_work = sum(workload.layer_map[layer].macs_per_sample for layer in set(targets)) / all_macs
        local = np.asarray([target_count / max(source_count, 1) - 1, target_work,
                            len(plan.ordered_edits) / 4, transfer, *partition_delta, *tile_delta, *cache_delta,
                            len(targets) / max(len(workload.layers), 1), float(plan.primitive_or_composite == "composite"),
                            float(plan.proxy_confidence != "unknown"), 1.0])
        # 10 atomic + 9 macro + 16 local + 5x10 state/action interactions + bias.
        vector = np.concatenate((one_hot, macro, local, np.outer(features[1:6], one_hot).ravel(), [1.0]))
        expected_size = ACTION_SIZE
        if state.version >= 2:
            allocations = [allocation for edit in plan.ordered_edits if edit.action_id == 'A4'
                           for allocation in edit.parameters.get('allocations', {}).values()]
            groups = [group for edit in plan.ordered_edits if edit.action_id == 'A10'
                      for group in edit.parameters.get('groups', ())]
            parts = [row['partition'] for row in allocations]
            k_fraction = sum(part[1] / math.prod(part) for part in parts) / max(1, len(parts))
            p_fraction = sum(part[2] * part[3] / math.prod(part) for part in parts) / max(1, len(parts))
            tile_fraction = [0., 0.]
            target_hardware = source.hardware
            for edit in plan.ordered_edits:
                if edit.action_id == 'A9':
                    target_hardware = candidates[edit.parameters['architecture_candidate_id']].hardware
                elif edit.action_id in ('A1', 'A3') and edit.parameters.get('tile_hw'):
                    sink = edit.parameters['layer_ids'][-1]
                    shape = workload.tensor_map[workload.layer_map[sink].output].shape
                    tile_fraction = [value / extent for value, extent in zip(edit.parameters['tile_hw'], shape[2:])]
            explicit = [len(allocations) / max(1, len(workload.layers)),
                        sum(len(row.get('cores', ())) for row in allocations) / max(1, source_count),
                        len(groups) / max(1, len(workload.layers)),
                        sum(group.get('mode') == 'temporal' for group in groups) / max(1, len(groups)),
                        k_fraction, p_fraction, *tile_fraction]
            physical = [math.log(target_hardware.rows / source.hardware.rows),
                        math.log(target_hardware.cols / source.hardware.cols),
                        target_hardware.io_count / source.hardware.io_count - 1,
                        math.prod(tile_fraction) * features[20], k_fraction * features[18], p_fraction * features[16]]
            vector = np.concatenate((vector, np.outer(features[16:], one_hot).ravel(), explicit, physical))
            expected_size = ACTION_SIZE_V2
        if state.version >= 3:
            from .target_bottleneck import target_features
            local_target = target_features(state, plan, (report if report is not None else state.evaluation_report or {}), workload)
            family = np.concatenate((one_hot, macro))
            dataflows = np.asarray([float(any(edit.action_id == 'A12' and edit.parameters.get('dataflow') == flow
                for edit in plan.ordered_edits)) for flow in ('output_stationary', 'weight_reuse', 'activation_reuse')])
            severity = local_target[[1, 2, 4, 6]]
            vector = np.concatenate((vector, local_target, np.outer(severity, family).ravel(),
                                     dataflows, np.outer(severity, dataflows).ravel()))
            expected_size = ACTION_SIZE_V3
        if len(vector) != expected_size:
            raise RuntimeError(f"candidate feature schema mismatch {len(vector)}")
        vectors.append(np.clip(vector, -8, 8))
    return np.asarray(vectors, dtype=np.float64)


@dataclass
class JointSearchResult:
    method: str
    seed: int
    best_state: DesignState | None
    best_metrics: dict | None
    best_cost: float | None
    current_state: DesignState
    current_metrics: dict
    closest_state: DesignState
    closest_metrics: dict
    initial_metrics: dict
    architecture_archive: dict
    trace: list[dict]
    evaluations: int
    requests: int
    policy_updates: int
    status: str
    elapsed_s: float
    evaluation_elapsed_s: float
    proposal_elapsed_s: float
    config: dict
    objective: dict
    checkpoint_path: str | None = None
    family_statistics: dict = field(default_factory=dict)

    def to_dict(self):
        data = {**asdict(self), "best_state": self.best_state.to_dict() if self.best_state else None,
                          "current_state": self.current_state.to_dict(), "closest_state": self.closest_state.to_dict(),
                          "cache_hits": self.requests - self.evaluations - (getattr(self, 'oracle_timeouts', 0) if self.current_state.version >= 2 else 0), "strict_feasible": self.best_state is not None,
                          "terminated": self.status in {"no_legal_nonduplicate_actions", "no_executable_candidates"},
                          "truncated": self.status.endswith("budget_exhausted"),
                          "algorithm": "linear softmax actor-critic; gamma=1; physical difference reward" if self.method == "rl" else self.method,
                          "capabilities": CAPABILITIES}
        if self.current_state.version >= 2:
            data['oracle_timeouts'] = getattr(self, 'oracle_timeouts', 0)
        if self.current_state.version >= 3:
            data['capabilities'] = {**CAPABILITIES, 'A12': {'supported': True, 'scope': 'priced per-layer loop-order dataflow; schema3'}}
            data['feature_schema'] = 'target_local_v3_283'
        return _jsonable(data)


def _atomic_save(path, data):
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    from .auto_runtime import atomic_json
    atomic_json(target,_jsonable(data))


def run_joint_search(workload, initial_state, architecture_candidates, evaluator_factory, *,
                     objective=None, config=None, checkpoint_path=None, resume=False,
                     evaluator_identity="unspecified", bootstrap_evaluations=(), proposal_hook=None,
                     resource_guard=None, reference_metrics=None, recovered_evaluations=(), query_observer=None):
    objective, config = objective or JointObjective(), config or JointSearchConfig()
    candidates = _candidate_map(architecture_candidates)
    state_size, action_size = (STATE_SIZE_V2, ACTION_SIZE_V2) if initial_state.version >= 2 else (STATE_SIZE, ACTION_SIZE)
    if initial_state.version >= 3:
        action_size = ACTION_SIZE_V3
    settings = asdict(config)
    # Cumulative resource ceilings may increase when resuming. The horizon,
    # proposal policy, learning rates and behavior distribution cannot change.
    identity_settings = {key: value for key, value in settings.items()
                         if key not in {"budget", "max_steps", "max_wall_time_s", "checkpoint_every"}}
    identity_payload={"workload": workload.fingerprint, "initial": initial_state.fingerprint,
                            "architectures": {key: candidate.hardware.to_dict() for key, candidate in candidates.items()},
                            "objective": asdict(objective), "settings": identity_settings,
                            "evaluator": evaluator_identity, "feature_schema": [state_size, action_size]}
    if reference_metrics is not None:
        identity_payload['reference_metrics'] = {key: reference_metrics[key] for key in ('latency_s', 'area_mm2', 'power_w', 'energy_j')}
    if bootstrap_evaluations or proposal_hook is not None:
        identity_payload['multi_start']=[s.fingerprint for s,m in bootstrap_evaluations]
        identity_payload['proposal_extension']=getattr(proposal_hook,'version',None)
    identity=stable_hash(identity_payload)
    started, elapsed_before, proposal_elapsed = time.perf_counter(), 0.0, 0.0
    rng = random.Random(config.seed)
    policy = LinearActorCritic(action_size, state_size, config.actor_lr, config.value_lr,
                              discount=1.0, entropy=config.entropy)
    trace, steps, episode_step, family_stats = [], 0, 0, {}
    timed_out_states = set()
    oracle = ExactEvaluationCache(candidates, evaluator_factory, evaluator_identity)
    saved = None
    if bootstrap_evaluations and not resume:
        for state,metrics in bootstrap_evaluations:
            validate_state(workload,state,candidates)
            if not executable(metrics):raise ValueError('bootstrap requires completed executable full PPA')
            oracle.cache[oracle.key(state)]={**_jsonable(metrics),'design_state_hash':state.fingerprint}
    if resume:
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise ValueError("resume requires an existing joint RL checkpoint")
        saved = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        if saved.get("schema") != CHECKPOINT_SCHEMA or saved.get("identity") != identity:
            raise ValueError("checkpoint workload/hardware/objective/policy identity differs")
        rng.setstate(_rng_tuple(saved["rng_state"]))
        policy = LinearActorCritic.from_dict(saved["policy"])
        oracle = ExactEvaluationCache(candidates, evaluator_factory, evaluator_identity, cache=saved["cache"],
                                      evaluations=saved["evaluations"], requests=saved["requests"],
                                      elapsed_s=saved.get("evaluation_elapsed_s", 0.0), timeouts=saved.get('oracle_timeouts', 0))
        steps, episode_step, trace = saved["steps"], saved["episode_step"], saved["trace"]
        if trace:
            trace[-1]["budget_truncated"] = False
        elapsed_before, proposal_elapsed = saved["elapsed_s"], saved.get("proposal_elapsed_s", 0.0)
        family_stats = saved.get("family_statistics", {})
        timed_out_states = set(saved.get('timed_out_state_hashes', ()))
        if proposal_hook is not None and saved.get('proposal_extension_state'):
            proposal_hook.load_state_dict(saved['proposal_extension_state'])
    oracle.observer = query_observer
    # Recover completed physical evidence without inventing a policy update.
    for state, metrics in recovered_evaluations:
        validate_state(workload, state, candidates)
        oracle.cache[oracle.key(state)] = {**_jsonable(metrics), 'design_state_hash': state.fingerprint}
    # On resume constructing the environment reads a known cache entry. Keep
    # accounting identical to uninterrupted execution by removing that read.
    previous_requests = oracle.requests
    env = JointDesignEnvironment(workload, initial_state, candidates, evaluator_factory, objective=objective,
                                 budget=config.budget, evaluator_identity=evaluator_identity, cache=oracle,
                                 reference=saved["reference"] if saved else reference_metrics, search_space=config.search_space)
    initial_metrics = dict(env.current_metrics)
    if bootstrap_evaluations and not saved:
        for state,metrics in bootstrap_evaluations:env.update_archive(state,metrics)
    if saved:
        oracle.requests = previous_requests
        env.current_state = DesignState.from_dict(saved["current_state"])
        env.current_metrics = saved["current_metrics"]
        env.best_state = DesignState.from_dict(saved["best_state"]) if saved["best_state"] else None
        env.best_metrics, env.best_cost = saved["best_metrics"], saved["best_cost"]
        env.closest_state = DesignState.from_dict(saved["closest_state"])
        env.closest_metrics, env.closest_rank = saved["closest_metrics"], tuple(saved["closest_rank"])
        env.architecture_archive = saved["architecture_archive"]
    for state, metrics in recovered_evaluations:
        env.update_archive(state, metrics)
    max_cores = max(len(_package(candidate.hardware).mapping_core_ids) for candidate in candidates.values())

    def save(status):
        payload={"schema": CHECKPOINT_SCHEMA, "identity": identity,
            "settings": settings, "reference": env.reference, "policy": policy.to_dict(),
            "rng_state": rng.getstate(), "steps": steps, "episode_step": episode_step,
            "current_state": env.current_state.to_dict(), "current_metrics": env.current_metrics,
            "best_state": env.best_state.to_dict() if env.best_state else None,
            "best_metrics": env.best_metrics, "best_cost": env.best_cost,
            "closest_state": env.closest_state.to_dict(), "closest_metrics": env.closest_metrics,
            "closest_rank": env.closest_rank, "architecture_archive": env.architecture_archive,
            "cache": oracle.cache, "evaluations": oracle.evaluations, "requests": oracle.requests,
            "evaluation_elapsed_s": oracle.elapsed_s, "proposal_elapsed_s": proposal_elapsed,
            "elapsed_s": elapsed_before + time.perf_counter() - started,
            "trace": trace, "family_statistics": family_stats, "status": status}
        if proposal_hook is not None:payload['proposal_extension_state']=proposal_hook.state_dict()
        if initial_state.version >= 2:
            payload['timed_out_state_hashes'] = sorted(timed_out_states)
            payload['oracle_timeouts'] = oracle.timeouts
        _atomic_save(checkpoint_path,payload)

    status = "step_budget_exhausted"
    while steps < config.max_steps:
        if resource_guard is not None:
            reason=resource_guard(oracle)
            if reason:status=reason;break
        if oracle.evaluations >= config.budget:
            status = "evaluation_budget_exhausted"
            break
        if config.max_wall_time_s is not None and elapsed_before + time.perf_counter() - started >= config.max_wall_time_s:
            status = "wall_time_budget_exhausted"
            break
        proposal_started = time.perf_counter()
        plans, proposal_stats = generate_candidates(workload, env.current_state, env.current_metrics, candidates,
            rng=rng, shortlist=config.shortlist, random_quota=config.random_quota,
            max_per_family=config.max_per_family, proposal=config.proposal,
            fusion_enabled=config.fusion_enabled, max_fusion_depth=config.max_fusion_depth, tile_sizes=config.tile_sizes,
            excluded_state_hashes=oracle.invalid_design_hashes, budget_excluded_state_hashes=timed_out_states,
            search_space=config.search_space, generator_guidance=config.generator_guidance)
        if not plans and (oracle.invalid_design_hashes or timed_out_states):
            # Retry a broader generic proposal pool before declaring a stalled
            # shortlist. Missing a sample is never reported as proof of no
            # executable architecture or an infeasible user's PPA request.
            plans, proposal_stats = generate_candidates(workload, env.current_state, env.current_metrics, candidates,
                rng=rng, shortlist=config.shortlist * 4, random_quota=config.random_quota * 4,
                max_per_family=config.max_per_family * 4, proposal=config.proposal,
                fusion_enabled=config.fusion_enabled, max_fusion_depth=config.max_fusion_depth, tile_sizes=config.tile_sizes,
                excluded_state_hashes=oracle.invalid_design_hashes, budget_excluded_state_hashes=timed_out_states,
            search_space=config.search_space, generator_guidance=config.generator_guidance)
        proposal_duration = time.perf_counter() - proposal_started
        if proposal_hook is not None:
            if initial_state.version >= 2:
                proposal_stats['excluded_state_hashes'] = sorted(oracle.invalid_design_hashes)
                proposal_stats['budget_excluded_state_hashes'] = sorted(timed_out_states)
            plans,proposal_stats=proposal_hook(env.current_state,env.current_metrics,plans,proposal_stats)
            proposal_duration=time.perf_counter()-proposal_started
        from .search_space import check_transition
        allowed = []
        for candidate_plan in plans:
            try:
                check_transition(env.current_state, apply_transaction(workload, env.current_state, candidate_plan, candidates), config.search_space, candidate_plan)
                allowed.append(candidate_plan)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                rejects = proposal_stats.setdefault('space_rejected', {})
                rejects[str(error)] = rejects.get(str(error), 0) + 1
        plans = tuple(allowed)
        proposal_stats['search_space'] = config.search_space
        proposal_stats['generator_guidance'] = config.generator_guidance
        proposal_elapsed += proposal_duration
        if initial_state.version >= 2:
            for family, count in proposal_stats['proposed'].items():
                family_stats.setdefault(family, {'proposed': 0, 'selected': 0, 'valid': 0,
                    'improved': 0, 'evaluation_elapsed_s': 0.})['proposed'] += count
        if not plans:
            status = ('no_candidate_within_current_timeout_budget' if timed_out_states else
                      "candidate_generation_stalled" if oracle.invalid_design_hashes else "no_legal_nonduplicate_actions")
            break
        shortlist_evidence = None
        if initial_state.version >= 2:
            # A budget-truncated greedy lookahead must not always evaluate the
            # architecture-reservation prefix before any local candidate.
            plans = list(plans)
            rng.shuffle(plans)
            plans = tuple(plans)
            if checkpoint_path:
                target = Path(checkpoint_path).parent / 'shortlists' / f'{steps + 1:06d}_{env.current_state.fingerprint[:16]}.json.gz'
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = json.dumps({'base_state_hash': env.current_state.fingerprint,
                    'plans': [plan.to_dict() for plan in plans], 'rng_state_after_proposals': rng.getstate(),
                    'before_state': replace(env.current_state, evaluation_report=None).to_dict(), 'before_metrics': env.current_metrics,
                    'state_features': None, 'policy': policy.to_dict(), 'reference': env.reference,
                    'episode_progress': episode_step / config.horizon},
                    sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
                compressed = gzip.compress(raw, compresslevel=1, mtime=0)
                temporary = target.with_suffix('.tmp')
                temporary.write_bytes(compressed)
                temporary.replace(target)
                shortlist_evidence = {'path': str(target.resolve()), 'sha256': hashlib.sha256(compressed).hexdigest(),
                                      'candidate_count': len(plans), 'encoding': 'gzip JSON; full ordered edits including architecture mappings'}
        before_state, before_metrics = env.current_state, env.current_metrics
        sf = state_features(before_state, before_metrics, env.reference, objective,
                            candidates[before_state.architecture_candidate_id], max_cores, episode_step / config.horizon)
        if initial_state.version >= 3 and not config.bottleneck_features:
            sf[10:13] = 0.
            sf[16:] = 0.
        policy_report = before_metrics if config.bottleneck_features or initial_state.version < 3 else {}
        af = candidate_features(before_state, plans, sf, candidates, workload, report=policy_report)
        if config.method == "rl":
            probabilities = policy.probabilities(af)
        elif config.method == "random":
            probabilities = np.ones(len(plans)) / len(plans)
        else:
            # Same candidates and exact oracle; every lookahead evaluation is
            # charged. This baseline cannot claim one-query greedy steps.
            values = []
            lookahead_stop = None
            for plan in plans:
                if initial_state.version >= 2:
                    lookahead_stop = resource_guard(oracle) if resource_guard is not None else None
                    if lookahead_stop:
                        break
                    if config.max_wall_time_s is not None and elapsed_before + time.perf_counter() - started >= config.max_wall_time_s:
                        lookahead_stop = 'wall_time_budget_exhausted'
                        break
                trial = apply_transaction(workload, before_state, plan, candidates)
                oracle.context = {"stage": "greedy_lookahead", "action_plan": plan.to_dict(), "before_state_hash": before_state.fingerprint,
                    "before_cost": objective.cost(before_metrics, env.reference), "best_cost_before": env.best_cost}
                try:
                    metrics, _, _ = oracle.get(trial, config.budget)
                except TimeoutError:
                    if initial_state.version < 2:
                        raise
                    timed_out_states.add(trial.fingerprint)
                    lookahead_stop = 'evaluation_timeout_budget_exhausted'
                    values.append(math.inf)
                    stop = resource_guard(oracle) if resource_guard is not None else None
                    if stop:
                        lookahead_stop = stop
                        break
                    continue
                if metrics is None:
                    break
                if executable(metrics):
                    env.update_archive(trial, metrics)
                values.append(objective.penalized_cost(metrics, env.reference) if executable(metrics) else math.inf)
            if not values or not any(math.isfinite(value) for value in values):
                status = lookahead_stop or ("evaluation_budget_exhausted" if oracle.evaluations >= config.budget else "no_executable_candidates")
                break
            chosen = min(range(len(values)), key=values.__getitem__)
            probabilities = np.zeros(len(plans)); probabilities[chosen] = 1.0
        if config.method != "greedy":
            draw, running, chosen = rng.random(), 0.0, len(plans) - 1
            for index, probability in enumerate(probabilities):
                running += float(probability)
                if draw < running:
                    chosen = index
                    break
        plan = plans[chosen]
        decision_evidence = {'search_elapsed_s': elapsed_before + time.perf_counter() - started,
            'before_metrics': before_metrics, 'completed_evaluations_before': oracle.evaluations,
            'generator_guidance': config.generator_guidance}
        if initial_state.version >= 3:
            from .target_bottleneck import target_features
            decision_evidence.update({'observation': {'feature_schema': 'target_local_v3_normalized_v2',
                'bottleneck_features_enabled': config.bottleneck_features,
                'state_features': sf.tolist(), 'selected_candidate_features': af[chosen].tolist(),
                'target_bottleneck_features': target_features(before_state, plan, policy_report, workload).tolist(),
                'evidence_quality': before_metrics.get('diagnostics', {}).get('evidence_quality', 'unknown')},
                'candidate_probabilities': probabilities.tolist(),
                'policy_normalized_entropy': float(-np.dot(probabilities, np.log(np.maximum(probabilities, 1e-300))) / max(math.log(len(probabilities)), 1e-30))})
        attempt_started = time.perf_counter()
        try:step = env.step(plan)
        except TimeoutError as error:
            if initial_state.version < 2:
                status='evaluation_timeout_budget_exhausted'
                break
            trial = apply_transaction(workload, before_state, plan, candidates)
            timed_out_states.add(trial.fingerprint)
            steps += 1
            episode_step += 1
            duration = time.perf_counter() - attempt_started
            stats = family_stats.setdefault(plan.family_id, {'proposed': 0, 'selected': 0, 'valid': 0,
                'improved': 0, 'evaluation_elapsed_s': 0.})
            stats['selected'] += 1
            stats['timeouts'] = stats.get('timeouts', 0) + 1
            stats['evaluation_elapsed_s'] += duration
            terminal = episode_step >= config.horizon
            trace.append({'step': steps, 'episode_step': episode_step, 'evaluations': oracle.evaluations,
                'requests': oracle.requests, 'action_plan': plan.to_dict(),
                'behavior_probability': float(probabilities[chosen]),
                'behavior_log_probability': float(math.log(max(probabilities[chosen], 1e-300))),
                'candidate_count': len(plans), 'candidate_shortlist': shortlist_evidence,
                'proposal_statistics': proposal_stats, 'reward': None, 'advantage': None,
                'valid': False, 'physical_validity': 'unknown', 'rejection_reason': str(error) + '; no PPA or policy update',
                'cache_hit': False, 'attempted_state_hash': trial.fingerprint,
                'before_state_hash': before_state.fingerprint, 'after_state_hash': before_state.fingerprint,
                'architecture_candidate_id': before_state.architecture_candidate_id,
                'current_metrics': before_metrics, 'current_cost': objective.cost(before_metrics, env.reference),
                'violation': objective.violation(before_metrics),
                'best_feasible_solution_id': env.best_state.fingerprint if env.best_state else None,
                'best_cost': env.best_cost, 'best_metrics': env.best_metrics,
                'episode_terminated': terminal, 'budget_truncated': False,
                'evaluation_elapsed_s': duration, 'proposal_elapsed_s': proposal_duration})
            trace[-1].update(decision_evidence)
            trace[-1]["search_elapsed_s"] = elapsed_before + time.perf_counter() - started
            trace[-1]["policy_updated"] = False
            if terminal:
                reset_state, reset_metrics = bootstrap_evaluations[(steps // config.horizon) % len(bootstrap_evaluations)] if bootstrap_evaluations else (initial_state, initial_metrics)
                env.current_state, env.current_metrics = reset_state.with_report(reset_metrics), reset_metrics
                episode_step = 0
            save('running')
            stop = resource_guard(oracle) if resource_guard is not None else None
            if stop:
                status = stop
                break
            continue
        if step.truncated:
            status = "evaluation_budget_exhausted"
            break
        steps += 1
        episode_step += 1
        terminal = episode_step >= config.horizon
        next_sf = state_features(env.current_state, env.current_metrics, env.reference, objective,
                                 candidates[env.current_state.architecture_candidate_id], max_cores,
                                 min(episode_step / config.horizon, 1.0))
        if initial_state.version >= 3 and not config.bottleneck_features:
            next_sf[10:13] = 0.
            next_sf[16:] = 0.
        # Invalid execution gives no fictitious PPA/reward and is excluded from
        # physical-cost training. Its rejection remains in the hard-mask log.
        advantage = None
        if config.method == "rl" and step.valid:
            advantage = policy.update(af, chosen, probabilities, sf, next_sf, step.reward, terminal)
        stats = family_stats.setdefault(plan.family_id, {"proposed": 0, "selected": 0, "valid": 0,
                                                         "improved": 0, "evaluation_elapsed_s": 0.0})
        if initial_state.version == 1:
            for family, count in proposal_stats["proposed"].items():
                family_stats.setdefault(family, {"proposed": 0, "selected": 0, "valid": 0,
                                                 "improved": 0, "evaluation_elapsed_s": 0.0})["proposed"] += count
        stats["selected"] += 1
        stats["valid"] += int(step.valid)
        stats["improved"] += int(step.valid and step.reward > 0)
        stats["evaluation_elapsed_s"] += step.evaluation_elapsed_s
        trace.append({"step": steps, "episode_step": episode_step, "evaluations": oracle.evaluations,
                      "requests": oracle.requests, "action_plan": step.realized_action_plan,
                      "behavior_probability": float(probabilities[chosen]), "behavior_log_probability": float(math.log(max(probabilities[chosen], 1e-300))),
                      "candidate_count": len(plans), "proposal_statistics": proposal_stats,
                      "reward": step.reward, "advantage": advantage, "valid": step.valid,
                      "rejection_reason": step.rejection_reason, "cache_hit": step.cache_hit,
                      "before_state_hash": before_state.fingerprint, "after_state_hash": env.current_state.fingerprint,
                      "architecture_candidate_id": env.current_state.architecture_candidate_id,
                      "current_metrics": env.current_metrics, "current_cost": objective.cost(env.current_metrics, env.reference),
                      "violation": objective.violation(env.current_metrics),
                      "best_feasible_solution_id": step.best_feasible_solution_id,
                      "best_cost": env.best_cost, "best_metrics": env.best_metrics,
                      "episode_terminated": terminal, "budget_truncated": False,
                      "evaluation_elapsed_s": step.evaluation_elapsed_s, "proposal_elapsed_s": proposal_duration})
        trace[-1].update(decision_evidence)
        trace[-1]["search_elapsed_s"] = elapsed_before + time.perf_counter() - started
        trace[-1]["policy_updated"] = bool(step.valid and config.method == "rl")
        if initial_state.version >= 2:
            trace[-1]['candidate_shortlist'] = shortlist_evidence
        if terminal:
            # Fixed-horizon episodes restart at the same executable Tangram
            # seed, while the learned policy and global feasible archive stay.
            if bootstrap_evaluations:
                reset_state,reset_metrics=bootstrap_evaluations[(steps//config.horizon)%len(bootstrap_evaluations)]
                env.current_state=reset_state.with_report(reset_metrics);env.current_metrics=reset_metrics
            else:
                env.current_state = initial_state.with_report(initial_metrics)
                env.current_metrics = initial_metrics
            episode_step = 0
        if steps % config.checkpoint_every == 0:
            save("running")
    if trace and status.endswith("budget_exhausted"):
        trace[-1]["budget_truncated"] = True
    save(status)
    result = JointSearchResult(config.method, config.seed, env.best_state, env.best_metrics, env.best_cost,
                             env.current_state, env.current_metrics, env.closest_state, env.closest_metrics,
                             initial_metrics, env.architecture_archive, trace, oracle.evaluations, oracle.requests,
                             policy.updates, status, elapsed_before + time.perf_counter() - started,
                             oracle.elapsed_s, proposal_elapsed, settings, asdict(objective),
                             str(checkpoint_path) if checkpoint_path else None, family_stats)
    if initial_state.version >= 2:
        result.oracle_timeouts = oracle.timeouts
    return result
