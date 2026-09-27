"""Bounded, receding-horizon search using only public candidates and feedback.

Weather and program choices use FormalOptimizer's existing predictions. The
model contains only currently visible candidates, not unpublished future ones.
Request partial credit and information credit are explicitly heuristic values;
they are not claims about earned official points or optimality in real weather.
"""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import heapq
import os

from formal_optimizer import coverage_delta, stamp


@dataclass(frozen=True)
class SearchAction:
    tile: int
    program: str
    request: int = -1
    requirement: int = -1


@dataclass(frozen=True)
class SearchState:
    elapsed: int
    bests: tuple
    bases: tuple
    completed: int
    visits: tuple
    informed: int
    value: float = 0.0
    first: int = -2  # -2: no action yet; -1: wait; >=0: action index


class SearchProblem:
    """Finite public-state model shared with an evaluator-only exact oracle."""

    def __init__(self, tiles, actions, requests, predict, *, horizon, offset=0,
                 regions=(), counts=(), flexible=(), quota=4, required_penalty=1000,
                 flexible_penalty=100, wait_cost=.001, coverage_weight=0,
                 total_base=0):
        self.tiles, self.actions, self.requests = tiles, actions, requests
        self.predict = predict
        self.horizon, self.offset = int(horizon), int(offset)
        self.regions, self.counts, self.flexible = regions, counts, flexible
        self.quota = quota
        self.required_penalty, self.flexible_penalty = required_penalty, flexible_penalty
        self.wait_cost, self.coverage_weight, self.total_base = wait_cost, coverage_weight, total_base
        self.region_masks = [sum(1 << i for i, t in enumerate(tiles) if t['region'] == r) for r in regions]
        self.flex_masks = [sum(1 << i for i, t in enumerate(tiles) if t['region'] == r and t['class'] == 'FLEXIBLE') for r in regions]
        completed = sum(1 << i for i, t in enumerate(tiles) if t['completed'])
        visits = tuple(tuple(item['done'] for item in req['items']) for req in requests)
        self.initial = SearchState(0, tuple(t['best'] for t in tiles), tuple(t['base'] for t in tiles), completed, visits, 0)
        self.predictions = {}

    def prediction(self, action, elapsed):
        key = (action.tile, action.program, elapsed)
        if key not in self.predictions:
            self.predictions[key] = self.predict(action, elapsed)
        return self.predictions[key]

    def request_value(self, visits, elapsed):
        value = 0.0
        for req, counts in zip(self.requests, visits):
            fractions = sorted((min(1, count / item['required']) for item, count in zip(req['items'], counts)), reverse=True)
            needed = req['count']
            # Top-k prevents optional targets from multiplying one request's reward.
            complete = sum(f >= 1 for f in fractions) >= needed
            if complete:
                value += req['value']
            elif elapsed < req['deadline']:
                # Terminal potential for unfinished requests, removed at expiry.
                value += req['value'] * sum(fractions[:needed]) / needed
        return value

    def advance(self, state, index):
        action = self.actions[index]
        tile = self.tiles[action.tile]
        end = state.elapsed + tile['duration']
        if state.elapsed < tile['start'] or end > min(self.horizon, tile['end']):
            return None
        if action.request >= 0:
            req = self.requests[action.request]
            if state.elapsed < req['start'] or end > req['deadline']:
                return None
            if state.visits[action.request][action.requirement] >= req['items'][action.requirement]['required']:
                return None
            if sum(c >= r['required'] for c, r in zip(state.visits[action.request], req['items'])) >= req['count']:
                return None
        prediction = self.prediction(action, state.elapsed)
        if prediction is None:
            return None
        base, total = prediction
        bit = 1 << action.tile
        first_visit = not state.completed & bit
        bests, bases = list(state.bests), list(state.bases)
        science = max(0, total - bests[action.tile])
        base_delta = base - bases[action.tile] if total > bests[action.tile] else 0
        if total > bests[action.tile]:
            bests[action.tile], bases[action.tile] = total, base
        terminal = 0.0
        added = state.completed & ~self.initial.completed
        region = self.regions.index(tile['region'])
        if first_visit and tile['class'] == 'REQUIRED':
            terminal = self.required_penalty
        elif first_visit and tile['class'] == 'FLEXIBLE':
            n = self.flexible[region] + (added & self.flex_masks[region]).bit_count()
            if n < self.quota:
                terminal = self.flexible_penalty
        counts = [n + (added & mask).bit_count() for n, mask in zip(self.counts, self.region_masks)]
        coverage = coverage_delta(self.coverage_weight,
                                  self.total_base + sum(state.bases) - sum(self.initial.bases),
                                  base_delta, counts, region, first_visit) if self.coverage_weight else 0.0
        visits = state.visits
        if action.request >= 0:
            visits = list(visits)
            row = list(visits[action.request])
            row[action.requirement] += 1
            visits[action.request] = tuple(row)
            visits = tuple(visits)
        request_delta = self.request_value(visits, end) - self.request_value(state.visits, state.elapsed)
        # Pay the inherited information heuristic at most once per tile per plan.
        info = tile['information'] if not state.informed & bit else 0.0
        value = state.value + science + terminal + coverage + request_delta + info
        return SearchState(end, tuple(bests), tuple(bases), state.completed | bit,
                           visits, state.informed | bit, value,
                           index if state.first == -2 else state.first)

    def wait(self, state):
        end = min(self.horizon, state.elapsed + 900 - (self.offset + state.elapsed) % 900)
        delta = self.request_value(state.visits, end) - self.request_value(state.visits, state.elapsed)
        return replace(state, elapsed=end, value=state.value + delta - (end-state.elapsed)*self.wait_cost,
                       first=-1 if state.first == -2 else state.first)

    def finish(self, state):
        """A legal idle completion, giving an anytime feasible model solution."""
        delta = self.request_value(state.visits, self.horizon) - self.request_value(state.visits, state.elapsed)
        return replace(state, elapsed=self.horizon,
                       value=state.value + delta - (self.horizon-state.elapsed)*self.wait_cost,
                       first=-1 if state.first == -2 else state.first)


def beam_search(problem, width=12, max_states=4000):
    """Compare equal-time states; stop deterministically at an expansion budget."""
    buckets = {0: [problem.initial]}
    times = [0]
    best = problem.finish(problem.initial)
    expanded = 0
    while times and expanded < max_states:
        elapsed = heapq.heappop(times)
        unique = {}
        for state in buckets.pop(elapsed):
            key = (state.bests, state.bases, state.completed, state.visits, state.informed, state.first)
            if key not in unique or state.value > unique[key].value:
                unique[key] = state
        states = sorted(unique.values(), key=lambda s: (-s.value, s.first))[:width]
        for state in states:
            terminal = problem.finish(state)
            if terminal.value > best.value:
                best = terminal
            if elapsed >= problem.horizon or expanded >= max_states:
                continue
            expanded += 1
            children = [problem.wait(state)]
            children.extend(child for i in range(len(problem.actions)) if (child := problem.advance(state, i)) is not None)
            for child in children:
                terminal = problem.finish(child)
                if terminal.value > best.value:
                    best = terminal
                if child.elapsed not in buckets:
                    buckets[child.elapsed] = []
                    heapq.heappush(times, child.elapsed)
                buckets[child.elapsed].append(child)
    return best, {'expanded_states': expanded, 'budget_exhausted': bool(times), 'predictions': len(problem.predictions)}


class FormalSearch:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.horizon = max(900, min(7200, int(os.getenv('SAC_FORMAL_SEARCH_HORIZON', '3600'))))
        self.width = max(1, min(64, int(os.getenv('SAC_FORMAL_SEARCH_WIDTH', '12'))))
        self.options = max(4, min(40, int(os.getenv('SAC_FORMAL_SEARCH_OPTIONS', '16'))))
        self.budget = max(1, min(20000, int(os.getenv('SAC_FORMAL_SEARCH_BUDGET', '4000'))))

    def choose(self, snapshot, ranked, default, detector):
        opt = self.optimizer
        now = stamp(snapshot['cursor']['timestamp_utc'])
        offset = int(snapshot['cursor']['slot_offset_seconds'])
        raws = {r['tile_id']: r for r in snapshot['candidate_tiles']}
        # Include urgent and request options as well as high immediate utility.
        ordered = sorted(ranked, key=lambda r: (-r[0], -r[1], r[2].tile_id, r[2].request_id))
        urgent = sorted(ordered, key=lambda r: stamp(raws[r[2].tile_id]['window_end_utc']) - now - r[2].nominal_exptime_seconds)
        pool = ordered[:self.options//2] + urgent[:self.options//4] + [r for r in ordered if r[2].request_id][:self.options//4] + ordered
        selected = []
        seen = set()
        for item in pool:
            row = item[2]
            key = (row.tile_id, row.program, row.request_id)
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) >= self.options:
                break
        if not selected:
            return default
        completed = set(snapshot.get('progress', {}).get('completed_tile_ids', []))
        tiles = []
        tile_index = {}
        for row in selected:
            if row.tile_id in tile_index:
                continue
            raw = raws[row.tile_id]
            tile_index[row.tile_id] = len(tiles)
            tiles.append({'id': row.tile_id, 'region': row.region_id, 'class': row.scheduling_class,
                          'duration': row.nominal_exptime_seconds,
                          'start': int(stamp(raw['window_start_utc']) - now), 'end': int(stamp(raw['window_end_utc']) - now),
                          'completed': row.tile_id in completed or bool(raw.get('already_completed')),
                          'best': opt.bests.get(row.tile_id, detector.bests.get(row.tile_id, 0)),
                          'base': opt.base_bests.get(row.tile_id, 0),
                          'information': opt.information_value(row, snapshot, detector)})
        requests = []
        request_index = {}
        for req in snapshot.get('active_requests', []):
            if req.get('is_complete') or not any(r.request_id == req['request_id'] for r in selected):
                continue
            request_index[req['request_id']] = len(requests)
            requests.append({'id': req['request_id'], 'start': int(stamp(req['available_from_utc']) - now),
                             'deadline': int(stamp(req['deadline_utc']) - now), 'count': int(req['required_tile_count']),
                             'value': float(req['completion_reward']) + float(req['miss_penalty']),
                             'items': [{'id': r['tile_id'], 'required': int(r['required_visits']),
                                        'done': int(r.get('completed_visits', int(r['required_visits'])-int(r.get('remaining_visits', r['required_visits']))))}
                                       for r in req['tile_requirements']]})
        actions = []
        for row in selected:
            ri = request_index.get(row.request_id, -1)
            if row.request_id and ri < 0:
                continue
            requirement = next((i for i, r in enumerate(requests[ri]['items']) if r['id'] == row.tile_id), -1) if ri >= 0 else -1
            if ri >= 0 and requirement < 0:
                continue
            action = SearchAction(tile_index[row.tile_id], row.program, ri, requirement)
            if action not in actions:
                actions.append(action)
            # A fulfilled/expired request must not remove the ordinary repeat option.
            ordinary = SearchAction(action.tile, action.program)
            if ordinary not in actions:
                actions.append(ordinary)

        def predict(action, elapsed):
            tile_id = tiles[action.tile]['id']
            raw = raws[tile_id]
            if elapsed == 0:
                prediction = next((v for (t, p, _), v in opt.estimates.items() if t == tile_id and p == action.program), None)
            else:
                cursor = dict(snapshot['cursor'], timestamp_utc=datetime.fromtimestamp(now+elapsed, timezone.utc).isoformat(),
                              slot_offset_seconds=(offset+elapsed) % 900)
                prediction = opt.exposure(raw, action.program, {'cursor': cursor})
            if prediction is None:
                return None
            base, bonus = prediction
            factor = opt.factor(tile_id)
            return base*factor, (base+bonus)*factor

        counts = [sum(opt.catalog[t]['region_id'] == r for t in completed if t in opt.catalog) for r in opt.regions]
        flex = snapshot.get('progress', {}).get('flexible_completed_by_region', {})
        penalties = opt.config['penalties']
        horizon = min(self.horizon, max(t['end'] for t in tiles))
        problem = SearchProblem(tiles, actions, requests, predict, horizon=horizon, offset=offset,
                                regions=opt.regions, counts=counts, flexible=[int(flex.get(r, 0)) for r in opt.regions],
                                quota=int(opt.config['flexible_quota_per_region']),
                                required_penalty=float(penalties['required_miss']),
                                flexible_penalty=float(penalties['flexible_shortfall_per_tile']),
                                wait_cost=float(penalties.get('avoidable_wait_per_second', .001)),
                                coverage_weight=float(opt.config.get('coverage_bonus_weight', 0)) if opt.coverage else 0,
                                total_base=sum(opt.base_bests.values()))
        best, diagnostics = beam_search(problem, self.width, self.budget)
        opt.last_diagnostics = {**diagnostics, 'model_value': best.value, 'search_horizon': horizon,
                                'search_actions': len(actions), 'weather_model': 'current conditions held constant'}
        if best.first == -1 and best.value > 0:
            return {'action': 'wait', 'tile_id': '', 'program': '', 'request_id': '',
                    'reason': 'formal rolling beam waits for a better public geometry opportunity',
                    'decision_source': 'formal_search'}
        if best.first < 0:
            # Keep the reliable current-action fallback when search finds no gain.
            return default
        action = actions[best.first]
        tile_id = tiles[action.tile]['id']
        request_id = requests[action.request]['id'] if action.request >= 0 else ''
        key = (tile_id, action.program, request_id)
        if key not in opt.estimates:
            opt.estimates[key] = next(v for (t, p, _), v in list(opt.estimates.items()) if t == tile_id and p == action.program)
        return {'action': 'observe', 'tile_id': tile_id, 'program': action.program, 'request_id': request_id,
                'reason': f'formal rolling beam horizon={horizon} states={diagnostics["expanded_states"]}',
                'decision_source': 'formal_search'}
