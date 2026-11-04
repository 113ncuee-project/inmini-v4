"""Budgeted, count-stratified transfers of the verified elite mapping.

This is an outer proposal schedule, NOT a PPA estimator or an RL reward.
Neural, random and annealing selectors receive the same scheduled count and
the same legal rectangle/quota alternatives. Every selected transfer still
needs a fully charged physical evaluation. All original local/global moves
remain available between frames; no chiplet count or layout is pruned.
"""
from dataclasses import dataclass, field


@dataclass
class ArchitectureRefresh:
    interval: int = 0
    tried: list = field(default_factory=list)
    cycle: int = 0
    history: list = field(default_factory=list)
    pending_count: int | None = None

    def __post_init__(self):
        if type(self.interval) is not int or self.interval < 0:
            raise ValueError('architecture refresh interval must be a nonnegative integer')

    def due(self, completed_decisions):
        return bool(self.interval and (completed_decisions + 1) % self.interval == 0)

    def proposals(self, space, elite, excluded=()):
        """One farthest-count frame, all its real rectangles and two quotas.

        Maximin distance on compute count is label-free. It is not ordered
        using the development control's best architecture. A cycle visits
        every count, including the elite count's other layout/quota options.
        Exhausted duplicate frames cost zero PPA and are explicitly logged.
        """
        counts = sorted({c.hardware.compute_count for c in space.candidates.values()})
        if not counts:
            return (), dict(reason='no_architecture_domain', full_ppa=0)
        anchor = space.candidates[elite.architecture_candidate_id].hardware.compute_count
        from .full_space import decision_hash
        seen = set(excluded) | {decision_hash(elite)}
        skipped = []
        for _ in range(len(counts)):
            remaining = [c for c in counts if c not in self.tried]
            if not remaining:
                self.tried = []
                self.cycle += 1
                remaining = counts
            points = [anchor, *self.tried]
            count = max(remaining, key=lambda c: (min(abs(c-v) for v in points), c))
            rows, keys, rejects = [], set(), 0
            for candidate in space.candidates.values():
                if candidate.hardware.compute_count != count:
                    continue
                for proportional in (False, True):
                    try:
                        trial = space.architecture_trial(elite, candidate, legacy=proportional)
                    except ValueError:
                        rejects += 1
                        continue
                    if trial.fingerprint not in seen and trial.fingerprint not in keys:
                        rows.append(trial)
                        keys.add(trial.fingerprint)
            if rows:
                self.pending_count = count
                return tuple(rows), dict(proposal_kind='architecture_refresh', count=count,
                    cycle=self.cycle, anchor_count=anchor, previous_counts=list(self.tried),
                    candidates=len(rows), skipped_duplicate_counts=skipped, rejected=rejects,
                    full_ppa=0, scope='verified elite template transfer; no cost label or pruning')
            self.tried.append(count)
            skipped.append(count)
        self.pending_count = None
        return (), dict(proposal_kind='architecture_refresh', reason='all_transfer_states_already_seen',
                        skipped_duplicate_counts=skipped, full_ppa=0)

    def finish(self, *, step, state_hash, status):
        if self.pending_count is None:
            raise ValueError('no pending architecture frame')
        count = self.pending_count
        self.tried.append(count)
        self.history.append(dict(cycle=self.cycle, count=count, step=step,
                                 state_hash=state_hash, status=status))
        self.pending_count = None

    def to_dict(self):
        return dict(interval=self.interval, tried=list(self.tried), cycle=self.cycle,
                    history=list(self.history), pending_count=self.pending_count)

    @classmethod
    def restore(cls, interval, data=None):
        if data and data['interval'] != interval:
            raise ValueError('cannot change architecture schedule inside a resumed run')
        return cls(**data) if data else cls(interval)
