"""Validated scheduling tools composed from the existing numerical planners.

Model plans choose an objective. They cannot invent an action or modify scores.
Each proposal is computed against the same received snapshot without committing
its predicted observation to the planners' real feedback state.
"""
from __future__ import annotations

from collections import Counter
import math
import os

from anomaly_detection import AnomalyDetector
from formal_optimizer import FormalOptimizer
from practice_long_horizon import LongHorizonPlanner
from rolling_planner import stamp
from scoring_preview import PROGRAMS, preview_actions


TOOLS = {
    "balanced": "Balance present science, published later opportunities, quotas and requests.",
    "deadline_first": "Prioritize currently observable required tiles and active request visits, earliest deadline first.",
    "coverage_first": "Prioritize useful flexible observations in the least complete regions below their quota.",
}


def action_key(decision):
    return tuple(decision.get(k, "") for k in ("action", "tile_id", "program", "request_id"))


def wait(reason):
    return {"action": "wait", "tile_id": "", "program": "", "request_id": "", "reason": reason}


class SchedulingTools:
    def __init__(self, initial):
        self.initial = initial
        self.contract = initial["scoring_contract"]
        self.catalog = {r["tile_id"]: r for r in initial["tile_catalog"]["tiles"]}
        self.scheduler = os.getenv('OBSERVER_SCHEDULER', 'legacy')
        if self.scheduler not in ('legacy', 'opportunity', 'window-search'):
            raise ValueError('Unknown OBSERVER_SCHEDULER')
        if self.scheduler == 'window-search':
            from window_search import WindowSearchPlanner
            self.practice = WindowSearchPlanner(initial)
        elif self.scheduler == 'opportunity':
            from opportunity_planner import OpportunityPlanner
            self.practice = OpportunityPlanner(initial)
        else:
            self.practice = LongHorizonPlanner(initial)
        self.formal = FormalOptimizer(initial)
        self.detector = AnomalyDetector(initial)
        self.snapshot = None
        self.previews = []
        self.reports = []
        self.formal_estimates = {}
        self.mechanics = False
        self.plan_guard = os.getenv('OBSERVER_PLAN_GUARD', '0') == '1'
        self.guard_margin = float(os.getenv('OBSERVER_GUARD_MARGIN', '0'))
        if not math.isfinite(self.guard_margin) or not 0 <= self.guard_margin <= 1:
            raise ValueError('Invalid guard improvement margin')
        self.offline_portfolio = os.getenv('OBSERVER_OFFLINE_PORTFOLIO', '0') == '1'
        self.proposal_diagnostics = {}
        self.selection_diagnostics = {}
        self.guard_rejections = 0

    def update(self, snapshot):
        self.mechanics = snapshot["schema_version"] == "decision-snapshot-v3"
        self.reports = []
        if self.mechanics:
            if self.scheduler != 'legacy':
                self.practice.update(snapshot)
            self.formal.update(snapshot)
            self.reports = self.detector.process_snapshot(snapshot)
            self.snapshot = self.detector.filter_fault_scope(snapshot)
            bests = self.detector.bests
        else:
            self.practice.update(snapshot)
            self.snapshot = snapshot
            bests = None
        self.previews = preview_actions(self.snapshot, self.contract, bests)

    def _propose(self, rows):
        fallback = wait("no useful legal exposure")
        if self.scheduler != 'legacy':
            decision = self.practice.choose(self.snapshot, rows, fallback,
                                           self.formal if self.mechanics else None,
                                           self.detector if self.mechanics else None)
            self.formal_estimates.update(self.practice.estimates)
            return decision
        if self.mechanics:
            decision = self.formal.choose(self.snapshot, rows, fallback, self.detector)
            self.formal_estimates.update(self.formal.estimates)
            return decision
        pending = self.practice.pending
        try:
            return self.practice.choose(self.snapshot, rows, fallback)
        finally:
            self.practice.pending = pending

    def proposals(self, all_tools=True):
        self.formal_estimates = {}
        balanced = self._propose(self.previews)
        self.proposal_diagnostics = {'balanced': dict(getattr(self.practice, 'last_diagnostics', {}))}
        result = {"balanced": balanced}
        if all_tools:
            tasks = [r for r in self.previews if r.request_id or
                     (r.scheduling_class == "REQUIRED" and r.terminal_penalty_avoidance > 0)]
            requests = {r["request_id"]: r for r in self.snapshot.get("active_requests", [])}
            if tasks:
                def deadline(row):
                    req = requests.get(row.request_id)
                    return stamp(req["deadline_utc"] if req else self.catalog[row.tile_id]["available_until_utc"])
                earliest = min(map(deadline, tasks))
                urgent = [r for r in tasks if deadline(r) == earliest]
                choice = self._propose(urgent)
                result["deadline_first"] = choice if choice["action"] == "observe" else balanced
                self.proposal_diagnostics['deadline_first'] = dict(self.practice.last_diagnostics) if choice['action'] == 'observe' and self.scheduler != 'legacy' else dict(self.proposal_diagnostics['balanced'])
            else:
                result["deadline_first"] = balanced
                self.proposal_diagnostics['deadline_first'] = dict(self.proposal_diagnostics['balanced'])
            counts = Counter(self.catalog[t]["region_id"] for t in
                             self.snapshot.get("progress", {}).get("completed_tile_ids", [])
                             if t in self.catalog and self.catalog[t]["scheduling_class"] == "FLEXIBLE")
            flexible = [r for r in self.previews if r.scheduling_class == "FLEXIBLE"
                        and r.terminal_penalty_avoidance > 0]
            if flexible:
                least = min(counts[r.region_id] for r in flexible)
                choice = self._propose([r for r in flexible if counts[r.region_id] == least])
                result["coverage_first"] = choice if choice["action"] == "observe" else balanced
                self.proposal_diagnostics['coverage_first'] = dict(self.practice.last_diagnostics) if choice['action'] == 'observe' and self.scheduler != 'legacy' else dict(self.proposal_diagnostics['balanced'])
            else:
                result["coverage_first"] = balanced
                self.proposal_diagnostics['coverage_first'] = dict(self.proposal_diagnostics['balanced'])
        # Every tool, including the reused baseline, passes the same last check.
        return {key: value if self.legal(value) else wait("tool proposal failed legality check")
                for key, value in result.items()}

    def legal(self, decision):
        if decision.get("action") == "wait":
            return all(not decision.get(k) for k in ("tile_id", "program", "request_id"))
        if decision.get("action") != "observe":
            return False
        key = (decision.get("tile_id"), decision.get("program"), decision.get("request_id", ""))
        # The numerical exposure estimate can choose a different program from
        # the start-of-exposure preview band. All published programs are legal.
        if key[1] not in PROGRAMS or (key[0], key[2]) not in {(r.tile_id, r.request_id) for r in self.previews}:
            return False
        raw = next(r for r in self.snapshot["candidate_tiles"] if r["tile_id"] == key[0])
        now = stamp(self.snapshot["cursor"]["timestamp_utc"])
        end = now + float(raw["nominal_exptime_seconds"])
        if not raw["effective_weather"]["is_observable"] or not stamp(raw["window_start_utc"]) <= now < end <= stamp(raw["window_end_utc"]):
            return False
        if key[2]:
            req = next((r for r in self.snapshot.get("active_requests", []) if r["request_id"] == key[2]), None)
            if req is None or not stamp(req["available_from_utc"]) <= now < end <= stamp(req["deadline_utc"]):
                return False
        return True

    def schedule_value(self, tool):
        detail = self.proposal_diagnostics.get(tool, {})
        if detail.get('search_fallback') or detail.get('scheduler') != 'window-search-v1':
            return None
        return detail.get('predicted_schedule_utility')

    def select_tool(self, requested, proposals, portfolio=False):
        selected, reason = requested, 'guard_disabled'
        baseline = self.schedule_value('balanced')
        proposed = self.schedule_value(requested)
        minimum_improvement = self.guard_margin*max(abs(baseline or 0), abs(proposed or 0), 1)
        if portfolio and baseline is not None:
            choices = [(self.schedule_value(name), name) for name in proposals]
            selected = max(((v, name) for v, name in choices if v is not None),
                           key=lambda pair: (pair[0], pair[1] == 'balanced'))[1]
            reason = 'best_completed_schedule'
        elif self.plan_guard:
            if action_key(proposals[requested]) == action_key(proposals['balanced']):
                reason = 'same_action'
            elif baseline is None or proposed is None:
                selected, reason = 'balanced', 'missing_comparable_schedule'
            elif proposed <= baseline+1e-6:
                selected, reason = 'balanced', 'no_predicted_improvement'
            elif proposed-baseline <= minimum_improvement:
                selected, reason = 'balanced', 'improvement_below_uncertainty_margin'
            else:
                reason = 'predicted_improvement'
            if selected != requested:
                self.guard_rejections += 1
        self.selection_diagnostics = {'guard_enabled': self.plan_guard, 'requested_tool': requested,
            'executed_tool': selected, 'reason': reason, 'balanced_schedule_value': baseline,
            'requested_schedule_value': proposed, 'minimum_improvement': minimum_improvement,
            'relative_margin': self.guard_margin, 'guard_rejections': self.guard_rejections}
        return selected

    def estimate(self, decision):
        if decision["action"] != "observe":
            return None
        key = (decision["tile_id"], decision["program"], decision.get("request_id", ""))
        if self.scheduler != 'legacy':
            prediction = self.formal_estimates.get(key)
            return sum(prediction) if prediction is not None else None
        if self.mechanics:
            prediction = self.formal_estimates.get(key)
            return sum(prediction) if prediction is not None else None
        raw = next(r for r in self.snapshot["candidate_tiles"] if r["tile_id"] == key[0])
        now = stamp(self.snapshot["cursor"]["timestamp_utc"])
        prediction = self.practice.estimate(raw, now, now-int(self.snapshot["cursor"]["slot_offset_seconds"]), {})
        return prediction[0] if prediction else None

    def audit(self):
        result = dict(self.practice.last_diagnostics) if self.scheduler != 'legacy' else {'scheduler': 'legacy'}
        result['plan_guard'] = dict(self.selection_diagnostics)
        return result

    def commit(self, decision):
        if not self.legal(decision):
            raise ValueError("Cannot commit an illegal tool proposal")
        tile = decision.get("tile_id") if decision["action"] == "observe" else None
        if self.mechanics:
            self.formal.estimates = self.formal_estimates
            expected = self.formal.note(decision, self.snapshot, self.detector)
            self.detector.note_observation(tile, expected,
                                           under_cold_wave=self.detector.under_cold_wave(self.snapshot))
        else:
            self.practice.pending = tile
        return {**decision, **({"reports": self.reports} if self.reports else {})}
