"""Bounded practice rollout using received publications, never scenario files.

The stopping tail is an independent-target approximation, not a claim of a
globally optimal schedule. Only the short rollout enforces shared telescope time.
"""
from collections import Counter, deque
from dataclasses import dataclass, replace
import json
import math
import os
import random

from practice_long_horizon import LongHorizonPlanner
from rolling_planner import stamp


@dataclass(frozen=True)
class Weather:
    quality: float
    efficiency: float = 1.0
    opened: bool = True


def public_weather(row):
    if not row.get('is_observable', False):
        # Closed-site snapshots legitimately publish null numerical conditions.
        return Weather(0.0, 1.0, False)
    return Weather(float(row.get('transparency', 1)) * float(row.get('sky_quality', 1)) /
                   max(.01, float(row.get('seeing_arcsec', 1))),
                   float(row.get('instrument_efficiency', 1)), bool(row.get('is_observable', False)))


def stopping_value(opportunities):
    """Bellman recursion: observe after seeing the outcome, or keep waiting.

Each chronological opportunity is a finite reward distribution; None means
closed. There is no observation reward on a closed opportunity.
"""
    value = 0.0
    for rewards in reversed(opportunities):
        value = sum(value if reward is None else max(reward, value)
                    for reward in rewards) / len(rewards)
    return value


@dataclass(frozen=True)
class Action:
    tile: int
    program: str = ''
    request: int = -1
    item: int = -1


WAIT = Action(-1)


@dataclass(frozen=True)
class State:
    time: int
    done: int
    visits: tuple
    score: float = 0.0


class PublicProblem:
    def __init__(self, planner, snapshot):
        self.planner = planner
        self.snapshot = snapshot
        self.now = stamp(snapshot['cursor']['timestamp_utc'])
        self.offset = int(snapshot['cursor']['slot_offset_seconds'])
        self.origin = self.now - self.offset
        self.horizon = max(1, min(planner.horizon, int(planner.night_end - self.now)))
        self.catalog = list(planner.catalog.values())
        self.ids = [r['tile_id'] for r in self.catalog]
        self.indices = {t: i for i, t in enumerate(self.ids)}
        self.config = planner.config
        self.penalties = self.config['penalties']
        self.wait_cost = float(self.penalties['avoidable_wait_per_second'])
        self.quota = int(self.config['flexible_quota_per_region'])
        self.regions = sorted({r['region_id'] for r in self.catalog})
        self.flex_masks = {region: sum(1 << i for i, r in enumerate(self.catalog)
                                     if r['region_id'] == region and r['scheduling_class'] == 'FLEXIBLE')
                           for region in self.regions}
        self.windows = [[] for _ in self.ids]
        for w in planner.windows.values():
            if w['end'] >= self.now and w['tile_id'] in self.indices:
                self.windows[self.indices[w['tile_id']]].append(w)
        for windows in self.windows:
            windows.sort(key=lambda w: (w['start'], w['end']))
        self.raws = {r['tile_id']: r for r in snapshot['candidate_tiles']}
        self.current = public_weather(snapshot['current_site_weather'])
        self.requests = []
        visits = []
        self.tile_requests = [[] for _ in self.ids]
        for req in snapshot.get('active_requests', []):
            if req.get('is_complete') or stamp(req['deadline_utc']) <= self.now:
                continue
            items = []
            for item in req['tile_requirements']:
                required = int(item['required_visits'])
                done = int(item.get('completed_visits', required - int(item.get('remaining_visits', required))))
                tile = self.indices.get(item['tile_id'])
                items.append((tile, required, len(visits)))
                if tile is not None:
                    self.tile_requests[tile].append((len(self.requests), len(items)-1))
                visits.append(done)
            self.requests.append({**req, 'items': items, 'start': stamp(req['available_from_utc'])-self.now,
                                  'deadline': stamp(req['deadline_utc'])-self.now,
                                  'value': float(req['completion_reward']) + float(req['miss_penalty'])})
        done = sum(1 << self.indices[t] for t in snapshot['progress']['completed_tile_ids'] if t in self.indices)
        self.initial = State(0, done, tuple(visits))
        self.geometry_cache = {}
        self.score_cache = {}
        self.tail_cache = {}
        self.distribution = planner.distribution()
        self.forecasts = list(planner.forecasts.values())
        self.expanded = 0

    def window(self, tile, elapsed):
        t = self.now + elapsed
        duration = int(self.catalog[tile]['nominal_exptime_seconds'])
        return next((w for w in self.windows[tile] if w['start'] <= t and t+duration <= w['end']), None)

    def pieces(self, tile, elapsed):
        key = tile, elapsed
        if key in self.geometry_cache:
            return self.geometry_cache[key]
        row = self.catalog[tile]
        duration = int(row['nominal_exptime_seconds'])
        if self.window(tile, elapsed) is None or elapsed+duration > self.horizon:
            self.geometry_cache[key] = None
            return None
        cursor, end = self.now + elapsed, self.now + elapsed + duration
        result = []
        interface = self.planner.contract['weather_score_interface']
        while cursor < end:
            boundary = self.origin + (math.floor((cursor-self.origin)/900)+1)*900
            stop = min(end, boundary)
            start = self.planner.geometry(self.ids[tile], cursor)
            middle = self.planner.geometry(self.ids[tile], (cursor+stop)/2)
            finish = self.planner.geometry(self.ids[tile], stop-1e-3)
            if min(start['altitude_deg'], middle['altitude_deg'], finish['altitude_deg']) < 30:
                result = None
                break
            result.append((int(cursor-self.now), int(stop-self.now),
                           middle['airmass'] ** float(interface['airmass_exponent']),
                           middle['lunar_quality_factor']))
            cursor = stop
        self.geometry_cache[key] = result
        return result

    def weather_at(self, tile, elapsed, path):
        # Path is accessible to the simulator; the policy receives only this one
        # current sample. It never receives the remaining sampled weather path.
        index = min(len(path)-1, (self.offset+elapsed)//900)
        weather = path[index]
        if self.offset+elapsed < 900 and self.ids[tile] in self.raws:
            return public_weather(self.raws[self.ids[tile]]['effective_weather'])
        return weather

    def forecast_close(self, tile, elapsed, draws):
        when = self.now + elapsed
        row = self.catalog[tile]
        for forecast, draw in zip(self.forecasts, draws):
            if forecast['condition'] not in ('rainy', 'tornado', 'rocket_launch'):
                continue
            if not forecast['start'] <= when < forecast['end'] or draw >= forecast['probability']:
                continue
            scope = forecast.get('spatial_scope_type')
            payload = forecast.get('spatial_scope_payload') or {}
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    continue
            # Unknown directional scopes are deliberately not treated as an
            # all-sky closure. Their present effects still arrive in snapshots.
            if scope == 'ALL' or (scope == 'REGION_SET' and row['region_id'] in payload.get('region_ids', [])):
                return True
        return False

    def visible_weather(self, tile, elapsed, path, draws):
        weather = self.weather_at(tile, elapsed, path)
        if self.offset+elapsed >= 900 and self.forecast_close(tile, elapsed, draws):
            weather = replace(weather, opened=False)
        return weather

    def estimate(self, tile, elapsed, weather):
        """Policy estimate: current sample held constant, never future path."""
        key = tile, elapsed, weather
        if key in self.score_cache:
            return self.score_cache[key]
        pieces = self.pieces(tile, elapsed)
        if not weather.opened or pieces is None:
            self.score_cache[key] = None
            return None
        row = self.catalog[tile]
        cap = float(self.planner.contract['weather_score_interface']['maximum_weather_quality'])
        rewards = {p: 0.0 for p in ('DARK', 'BRIGHT', 'BACKUP')}
        for start, stop, airmass, lunar in pieces:
            science_q = min(cap, weather.quality*weather.efficiency/airmass)*lunar
            band_q = science_q  # Legacy practice bands include instrument efficiency.
            thresholds = self.config['quality_thresholds']
            band = 'DARK' if band_q >= thresholds['dark'] else 'BRIGHT' if band_q >= thresholds['bright'] else 'BACKUP'
            base = float(row['tile_science_value'])*science_q*(stop-start)/int(row['nominal_exptime_seconds'])
            for program in rewards:
                rewards[program] += base*(1+float(self.config['program_bonus'][program]) if program == band else 1)
        result = max(rewards.values()), max(rewards, key=rewards.get), rewards
        self.score_cache[key] = result
        return result

    def request_potential(self, state):
        total = 0.0
        for req in self.requests:
            fractions = sorted((min(1.0, state.visits[index]/required) for _, required, index in req['items']), reverse=True)
            count = int(req['required_tile_count'])
            if sum(f >= 1 for f in fractions) >= count:
                total += req['value']
            elif state.time < req['deadline']:
                total += req['value']*sum(fractions[:count])/count
        return total

    def credit(self, state, tile):
        if state.done & (1 << tile):
            return 0.0
        row = self.catalog[tile]
        if row['scheduling_class'] == 'REQUIRED':
            return float(self.penalties['required_miss'])
        if row['scheduling_class'] == 'FLEXIBLE' and (state.done & self.flex_masks[row['region_id']]).bit_count() < self.quota:
            return float(self.penalties['flexible_shortfall_per_tile'])
        return 0.0

    def visit(self, state, action):
        if action.request < 0:
            return state.visits
        index = self.requests[action.request]['items'][action.item][2]
        visits = list(state.visits)
        visits[index] += 1
        return tuple(visits)

    def actions(self, state, visible):
        actions = []
        for tile in range(len(self.ids)):
            weather = visible(tile)
            estimate = self.estimate(tile, state.time, weather)
            if estimate is None:
                continue
            program = estimate[1]
            if not state.done & (1 << tile):
                actions.append(Action(tile, program))
            duration = int(self.catalog[tile]['nominal_exptime_seconds'])
            for request, item in self.tile_requests[tile]:
                req = self.requests[request]
                _, required, index = req['items'][item]
                satisfied = sum(state.visits[j] >= n for _, n, j in req['items'])
                if (req['start'] <= state.time and state.time+duration <= req['deadline'] and
                        state.visits[index] < required and satisfied < int(req['required_tile_count'])):
                    actions.append(Action(tile, program, request, item))
        return actions

    def immediate(self, state, action, visible):
        estimate = self.estimate(action.tile, state.time, visible(action.tile))
        if estimate is None:
            return -math.inf
        science = 0.0 if state.done & (1 << action.tile) else estimate[2][action.program]
        after = replace(state, visits=self.visit(state, action))
        return science+self.credit(state, action.tile)+self.request_potential(after)-self.request_potential(state)

    def target_tail(self, tile, elapsed, credit):
        key = tile, elapsed, credit
        if key in self.tail_cache:
            return self.tail_cache[key]
        row = self.catalog[tile]
        duration = int(row['nominal_exptime_seconds'])
        nights = {}
        cap = float(self.planner.contract['weather_score_interface']['maximum_weather_quality'])
        power = float(self.planner.contract['weather_score_interface']['airmass_exponent'])
        for window in self.windows[tile]:
            earliest = max(self.now+elapsed, window['start'])
            if earliest+duration > window['end']:
                continue
            moment = max(earliest+duration/2, min(window['end']-duration/2, stamp(window['best_time_utc'])))
            geom = self.planner.geometry(self.ids[tile], moment)
            if geom['altitude_deg'] < 30:
                continue
            rewards = []
            for weather in self.distribution:
                if not weather.opened:
                    rewards.append(None)
                    continue
                science_q = min(cap, weather.quality*weather.efficiency/geom['airmass']**power)*geom['lunar_quality_factor']
                band_q = science_q
                thresholds = self.config['quality_thresholds']
                band = 'DARK' if band_q >= thresholds['dark'] else 'BRIGHT' if band_q >= thresholds['bright'] else 'BACKUP'
                rewards.append(float(row['tile_science_value'])*science_q*(1+float(self.config['program_bonus'][band]))+credit)
            merit = sum(r or 0 for r in rewards)
            previous = nights.get(window['night_id'])
            if previous is None or merit > previous[1]:
                nights[window['night_id']] = (moment, merit, rewards)
        value = stopping_value([r[2] for r in sorted(nights.values())])
        self.tail_cache[key] = value
        return value

    def tail(self, state):
        if not self.planner.tail_enabled:
            return 0.0
        total = 0.0
        flex = {region: [] for region in self.regions}
        for tile, row in enumerate(self.catalog):
            if state.done & (1 << tile):
                continue
            credit = float(self.penalties['required_miss']) if row['scheduling_class'] == 'REQUIRED' else 0.0
            value = self.target_tail(tile, state.time, credit)
            total += value
            if row['scheduling_class'] == 'FLEXIBLE':
                extra = self.target_tail(tile, state.time, float(self.penalties['flexible_shortfall_per_tile']))-value
                flex[row['region_id']].append(extra)
        for region, extras in flex.items():
            missing = max(0, self.quota-(state.done & self.flex_masks[region]).bit_count())
            total += sum(sorted(extras, reverse=True)[:missing])
        return total

    def value(self, state):
        return state.score+self.request_potential(state)+self.tail(state)

    def advance(self, state, action, path, draws):
        self.expanded += 1
        visible = lambda tile: self.visible_weather(tile, state.time, path, draws)
        if action.tile < 0:
            end = min(self.horizon, state.time+900-(self.offset+state.time)%900)
            cost = self.wait_cost*(end-state.time) if self.actions(state, visible) else 0.0
            return replace(state, time=end, score=state.score-cost)
        pieces = self.pieces(action.tile, state.time)
        if pieces is None:
            raise ValueError('Cannot simulate an infeasible action')
        earned = 0.0
        row = self.catalog[action.tile]
        for start, stop, airmass, lunar in pieces:
            weather = self.weather_at(action.tile, start, path)
            # Root current-slot weather is known and overrides uncertain forecasts.
            closed = self.offset+start >= 900 and self.forecast_close(action.tile, start, draws)
            if not weather.opened or closed:
                return replace(state, time=start)
            cap = float(self.planner.contract['weather_score_interface']['maximum_weather_quality'])
            science_q = min(cap, weather.quality*weather.efficiency/airmass)*lunar
            band_q = science_q
            threshold = self.config['quality_thresholds']
            band = 'DARK' if band_q >= threshold['dark'] else 'BRIGHT' if band_q >= threshold['bright'] else 'BACKUP'
            earned += float(row['tile_science_value'])*science_q*(stop-start)/int(row['nominal_exptime_seconds'])*(1+float(self.config['program_bonus'][band]) if action.program == band else 1)
        first = not state.done & (1 << action.tile)
        gain = (earned if first else 0.0)+self.credit(state, action.tile)
        return State(pieces[-1][1], state.done | (1 << action.tile), self.visit(state, action), state.score+gain)

    def policy(self, state, visible):
        actions = self.actions(state, visible)
        if not actions:
            return WAIT
        def priority(action):
            duration = int(self.catalog[action.tile]['nominal_exptime_seconds'])
            reserve = self.target_tail(action.tile, state.time+duration+1, self.credit(state, action.tile)) if self.planner.tail_enabled and not state.done & (1 << action.tile) else 0.0
            return ((self.immediate(state, action, visible)-reserve)/duration,
                    -action.tile, -action.request)
        best = max(actions, key=priority)
        return best if priority(best)[0] > 0 else WAIT

    def rollout(self, actions, paths):
        scores = [0.0]*len(actions)
        # Advance all action/sample branches in lockstep. If the budget ends,
        # every candidate has the same number of rollout steps and all samples.
        states = [[self.initial for _ in paths] for _ in actions]
        rounds = 0
        while True:
            active = sum(s.time < self.horizon for row in states for s in row)
            if not active or self.expanded+active > self.planner.budget:
                break
            for i, action in enumerate(actions):
                for j, (path, draws) in enumerate(paths):
                    state = states[i][j]
                    if state.time >= self.horizon:
                        continue
                    choice = action if not rounds else self.policy(state, lambda tile: self.visible_weather(tile, state.time, path, draws))
                    states[i][j] = self.advance(state, choice, path, draws)
            rounds += 1
        # Finish truncated branches with idle and the same terminal horizon.
        for i, row in enumerate(states):
            for state in row:
                end = replace(state, time=self.horizon,
                              score=state.score-self.wait_cost*(self.horizon-state.time))
                scores[i] += self.value(end)/len(paths)
        return scores, bool(active), rounds


class RolloutPlanner(LongHorizonPlanner):
    def __init__(self, initial):
        super().__init__(initial)
        if not self.config.get('one_ordinary_credit_per_tile') or self.config.get('repeat_observation'):
            raise ValueError('Rollout v1 requires original practice scoring')
        self.variant = os.getenv('SAC_ROLLOUT_VARIANT', 'weather')
        if self.variant not in ('windows', 'tail', 'weather'):
            raise ValueError('Unknown rollout variant')
        self.tail_enabled = self.variant != 'windows'
        self.horizon, self.options, self.samples, self.budget = 7200, 12, 8, 4096
        self.history = deque(maxlen=512)
        self.forecasts = {}
        self.night_end = 0
        self.night_id = None
        self.public_geometry_cache = {}
        self.last_weather_slot = None
        self.diagnostics = Counter()
        self.last_prediction = None

    def geometry(self, tile, timestamp):
        key = tile, timestamp
        cache = self.public_geometry_cache
        if key not in cache:
            if len(cache) > 200000:
                cache.clear()
            cache[key] = super().geometry(tile, timestamp)
        return cache[key]

    def update(self, snapshot):
        if snapshot['schema_version'] != 'decision-snapshot-v2':
            raise ValueError('Rollout practice path requires snapshot v2')
        now = stamp(snapshot['cursor']['timestamp_utc'])
        if self.night_id != snapshot['cursor']['night_id']:
            self.public_geometry_cache.clear()
            self.night_id = snapshot['cursor']['night_id']
        publication = snapshot.get('night_start')
        if publication:
            self.night_end = stamp(publication['night']['observing_end_utc'])
        super().update(snapshot)
        if self.last_weather_slot != snapshot['cursor']['slot_id']:
            self.history.append(public_weather(snapshot['current_site_weather']))
            self.last_weather_slot = snapshot['cursor']['slot_id']
        weekly = snapshot.get('weekly') or {}
        for forecast in weekly.get('weather_forecast', []):
            if stamp(forecast['issued_at_utc']) > now:
                continue
            key = forecast['event_id']
            previous = self.forecasts.get(key)
            if previous is None or int(forecast['revision']) >= int(previous['revision']):
                self.forecasts[key] = {**forecast, 'start': stamp(forecast['predicted_start_utc']),
                                      'end': stamp(forecast['predicted_end_utc']),
                                      'probability': max(0.0, min(1.0, float(forecast['probability'])))}
        self.forecasts = {k: f for k, f in self.forecasts.items() if f['end'] > now}

    def distribution(self):
        # Fixed quantile quadrature of past open AND closed slots. Bootstrap
        # cold start with neutral historical-baseline priors, not future data.
        history = list(self.history) + [Weather(.7), Weather(.7), Weather(.7, opened=False)]
        history.sort(key=lambda w: w.quality*w.efficiency if w.opened else -1)
        return [history[min(len(history)-1, int((i+.5)*len(history)/8))] for i in range(8)]

    def weather_paths(self, problem):
        count = 1 if self.variant != 'weather' else self.samples
        rng = random.Random(314159 + int(problem.snapshot.get('decision_sequence', 0)))
        length = math.ceil((problem.offset+problem.horizon)/900)+1
        history = list(self.history)
        paths = []
        for _ in range(count):
            path = [problem.current]
            start = rng.randrange(max(1, len(history)))
            for step in range(1, length):
                if count == 1:
                    weather = problem.current
                elif len(history) >= 4:
                    if (step-1) % 4 == 0:
                        start = rng.randrange(len(history)-3)
                    weather = history[start+(step-1) % 4]
                else:
                    weather = problem.current
                path.append(weather)
            draws = [rng.random() if count > 1 else .5 for _ in problem.forecasts]
            paths.append((tuple(path), tuple(draws)))
        return paths

    def choose(self, snapshot, previews, default):
        self.diagnostics['calls'] += 1
        self.last_prediction = {'kind': 'heuristic_future_value', 'chosen_value': None,
                                'comparisons': [], 'fallback_reason': None}
        # The inherited policy contributes one incumbent action, not a hidden
        # alternative runtime path. Real pending feedback state is not modified.
        incumbent = super().choose(snapshot, previews, default)
        self.pending = None
        problem = PublicProblem(self, snapshot)
        visible = lambda tile: problem.weather_at(tile, 0, (problem.current,))
        available = problem.actions(problem.initial, visible)
        # Root submissions must be present in the received candidate catalogue.
        available = [a for a in available if problem.ids[a.tile] in problem.raws]
        if not available:
            self.diagnostics['no_legal_observation'] += 1
            self.last_prediction['fallback_reason'] = 'no_legal_observation'
            return self.decision(WAIT, problem)
        def matches(a):
            rid = problem.requests[a.request]['request_id'] if a.request >= 0 else ''
            return incumbent.get('action') == 'observe' and (problem.ids[a.tile], rid) == (incumbent.get('tile_id'), incumbent.get('request_id', ''))
        old = next((replace(a, program=incumbent['program']) for a in available if matches(a)), WAIT)
        self.diagnostics['incumbent_rejected'] += incumbent.get('action') == 'observe' and old == WAIT
        urgency = sorted(available, key=lambda a: (problem.window(a.tile, 0)['end'], a.tile, a.request))
        ranked = sorted(available, key=lambda a: (-problem.immediate(problem.initial, a, visible)/int(problem.catalog[a.tile]['nominal_exptime_seconds']), a.tile, a.request))
        candidates = [old, WAIT] + urgency[:4] + [a for a in urgency if a.request >= 0][:2] + ranked
        actions = list(dict.fromkeys(candidates))[:self.options]
        paths = self.weather_paths(problem)
        scores, exhausted, rounds = problem.rollout(actions, paths)
        winner = max(range(len(actions)), key=lambda i: (scores[i], -i))
        self.diagnostics['planned'] += 1
        self.diagnostics['states'] += problem.expanded
        self.diagnostics['budget_exhausted'] += exhausted
        self.diagnostics['changed_from_incumbent'] += actions[winner] != old
        self.diagnostics['rollout_rounds'] += rounds
        self.diagnostics['future_window_tiles'] += sum(any(w['start'] > problem.now for w in windows) for windows in problem.windows)
        self.last_prediction = {
            'kind': 'heuristic_future_value', 'chosen_value': scores[winner],
            'chosen_index': winner, 'horizon_seconds': problem.horizon,
            'state_advances': problem.expanded, 'weather_paths': len(paths),
            'fallback_reason': 'state_budget_exhausted' if exhausted else None,
            'comparisons': [dict(self.decision(a, problem), predicted_value=v)
                            for a, v in zip(actions, scores)],
        }
        return self.decision(actions[winner], problem)

    def decision(self, action, problem):
        if action.tile < 0:
            return {'action': 'wait', 'tile_id': '', 'program': '', 'request_id': '',
                    'reason': 'research rollout '+self.variant+' '+
                              ('no legal observation' if self.last_prediction and self.last_prediction.get('fallback_reason') == 'no_legal_observation'
                               else 'waits after public opportunity comparison')}
        return {'action': 'observe', 'tile_id': problem.ids[action.tile], 'program': action.program,
                'request_id': problem.requests[action.request]['request_id'] if action.request >= 0 else '',
                'reason': 'research rollout '+self.variant+' with public windows and joint task state'}
