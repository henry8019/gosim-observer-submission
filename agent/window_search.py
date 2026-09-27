"""Receding-horizon scheduling from received windows and current conditions.

Beam search enforces one telescope, first-credit uniqueness, dynamic quotas and
request visit counts. It is approximate and does not know future weather.
"""
from dataclasses import replace
import heapq
import json
import math
import os

from observed_weather import ObservedWeather
from public_geometry import _angular_separation_deg
from opportunity_planner import OpportunityPlanner
from rollout_planner import PublicProblem, State, public_weather
from rolling_planner import stamp


class SegmentWeatherProblem(PublicProblem):
    """Integrate the causal prediction at each crossed weather time slot."""
    def __init__(self, planner, snapshot):
        super().__init__(planner, snapshot)
        self.weather_scales = {}
        self.base_scores = {}

    def scale(self, elapsed):
        step = int((self.offset+elapsed)//900)
        if step not in self.weather_scales:
            model = self.planner.weather_model
            prediction = model.predict(step, self.planner.weather_blend)
            self.weather_scales[step] = prediction/model.current if model.current else 1.0
        return self.weather_scales[step]

    def estimate(self, tile, elapsed, weather):
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
        total_base = 0.0
        for start, stop, airmass, lunar in pieces:
            q = min(cap, weather.quality*self.scale(start)/self.scale(elapsed)*weather.efficiency/airmass)*lunar
            threshold = self.config['quality_thresholds']
            band = 'DARK' if q >= threshold['dark'] else 'BRIGHT' if q >= threshold['bright'] else 'BACKUP'
            base = float(row['tile_science_value'])*q*(stop-start)/int(row['nominal_exptime_seconds'])
            total_base += base
            for program in rewards:
                rewards[program] += base*(1+self.config['program_bonus'][program] if program == band else 1)
        self.base_scores[key] = total_base
        self.score_cache[key] = max(rewards.values()), max(rewards, key=rewards.get), rewards
        return self.score_cache[key]


class WindowSearchPlanner(OpportunityPlanner):
    def __init__(self, initial):
        self.geometry_cache = {}
        super().__init__(initial)
        self.horizon = int(os.getenv('SAC_WINDOW_HORIZON', '7200'))
        self.width = int(os.getenv('SAC_WINDOW_WIDTH', '12'))
        self.budget = int(os.getenv('SAC_WINDOW_BUDGET', '50000'))
        if not 900 <= self.horizon <= 21600 or not 1 <= self.width <= 64 or not 100 <= self.budget <= 200000:
            raise ValueError('Window search budget outside supported bounds')
        self.tail_mode = os.getenv('SAC_WINDOW_TAIL', 'threshold')
        if self.tail_mode not in ('threshold', 'stopping', 'seasonal-cost', 'current-window'):
            raise ValueError('Unknown window reservation model')
        self.wait_share = float(os.getenv('SAC_WINDOW_WAIT_SHARE', '.5'))
        if not math.isfinite(self.wait_share) or not 0 <= self.wait_share <= 1:
            raise ValueError('Invalid waiting-cost sharing exponent')
        if self.tail_mode == 'seasonal-cost' and not self.seasonal:
            self.seasonal = self.seasonal_opportunities()
        self.weather_blend = float(os.getenv('SAC_WINDOW_WEATHER', '0'))
        if not math.isfinite(self.weather_blend) or not 0 <= self.weather_blend <= 1:
            raise ValueError('Invalid causal weather blend')
        self.weather_model = ObservedWeather()
        self.keep_first_choices = bool(int(os.getenv('SAC_WINDOW_DIVERSITY', '0')))
        self.use_forecasts = bool(int(os.getenv('SAC_WINDOW_FORECAST', '0')))
        self.segment_weather = bool(int(os.getenv('SAC_WINDOW_WEATHER_SEGMENTS', '0')))
        self.tail_forecasts = bool(int(os.getenv('SAC_WINDOW_TAIL_FORECAST', '0')))
        self.request_reserve = bool(int(os.getenv('SAC_WINDOW_REQUEST_RESERVE', '0')))
        self.request_reservation_weight = float(os.getenv('SAC_WINDOW_REQUEST_WEIGHT', '1'))
        if not math.isfinite(self.request_reservation_weight) or not 0 <= self.request_reservation_weight <= 1:
            raise ValueError('Invalid request reservation weight')
        self.night_end = 0
        self.forecasts = {}
        self.geometry_night = None
        self.search_calls = 0
        self.budget_hits = 0

    def geometry(self, tile, timestamp):
        key = tile, timestamp
        if key not in self.geometry_cache:
            if len(self.geometry_cache) > 200000:
                self.geometry_cache.clear()
            self.geometry_cache[key] = super().geometry(tile, timestamp)
        return self.geometry_cache[key]

    def update(self, snapshot):
        if self.geometry_night != snapshot['cursor']['night_id']:
            self.geometry_cache.clear()
            self.geometry_night = snapshot['cursor']['night_id']
        publication = snapshot.get('night_start')
        if publication:
            self.night_end = stamp(publication['night']['observing_end_utc'])
        super().update(snapshot)
        now = stamp(snapshot['cursor']['timestamp_utc'])
        weather = public_weather(snapshot['current_site_weather'])
        self.weather_model.observe(now-int(snapshot['cursor']['slot_offset_seconds']), weather.quality if weather.opened else None)
        if self.use_forecasts:
            for publication in (snapshot.get('weekly'), snapshot.get('night_start')):
                for row in (publication or {}).get('weather_forecast', []):
                    issued = stamp(row['issued_at_utc'])
                    key, order = row['event_id'], (int(row['revision']), issued)
                    if issued > now or (key in self.forecasts and order <= self.forecasts[key]['order']):
                        continue
                    payload = row.get('spatial_scope_payload') or {}
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    self.forecasts[key] = {**row, 'payload': payload, 'order': order,
                        'start': stamp(row['predicted_start_utc']), 'end': stamp(row['predicted_end_utc'])}

    def scope_applies(self, forecast, tile, moment):
        kind, payload = forecast['spatial_scope_type'], forecast['payload']
        if kind == 'ALL':
            return True
        if kind == 'REGION_SET':
            return self.catalog[tile]['region_id'] in payload.get('region_ids', [])
        if kind == 'TILE_SET':
            return tile in payload.get('tile_ids', [])
        if kind == 'SKY_CAP_ICRS':
            coord = self.coords[tile]
            return _angular_separation_deg(coord.ra_deg, coord.dec_deg, float(payload['ra_deg']), float(payload['dec_deg'])) <= float(payload['radius_deg'])
        if kind == 'HORIZON_SECTOR':
            geom = self.geometry(tile, moment)
            lo, hi = float(payload['azimuth_start_deg']), float(payload['azimuth_end_deg'])
            az = geom['azimuth_deg']
            az_ok = lo <= az <= hi if lo <= hi else az >= lo or az <= hi
            return az_ok and float(payload['min_altitude_deg']) <= geom['altitude_deg'] <= float(payload['max_altitude_deg'])
        return False

    def survival(self, tile, start, end, origin):
        if not self.use_forecasts:
            return 1.0
        probability = 1.0
        for forecast in self.forecasts.values():
            if forecast['condition'] not in ('rainy', 'tornado', 'rocket_launch'):
                continue
            cursor = max(start, origin+900, forecast['start'])
            stop = min(end, forecast['end'])
            while cursor < stop:
                edge = min(stop, origin+(math.floor((cursor-origin)/900)+1)*900)
                if self.scope_applies(forecast, tile, (cursor+edge)/2):
                    probability *= 1-max(0, min(1, float(forecast['probability'])))
                    break
                cursor = edge
        return probability

    def distribution(self):
        # The reusable public problem expects this hook. The threshold variant
        # needs no sampled weather and never loads a hidden weather trajectory.
        return []

    def future_values(self, problem):
        reserve, risks = [], []
        target = self.weather_target()
        outstanding = sum(not problem.initial.done & (1 << i) and stamp(r['available_until_utc']) > problem.now
                          for i, r in enumerate(problem.catalog))
        cost_share = max(1, outstanding**self.wait_share)
        values = sorted(self.weather_history) or [self.prior]
        samples = [values[min(len(values)-1, int((j+.5)*len(values)/8))] for j in range(8)]
        for tile, row in enumerate(problem.catalog):
            duration = int(row['nominal_exptime_seconds'])
            moments = {}
            for window in problem.windows[tile]:
                cutoff = 900 if self.tail_mode == 'current-window' else problem.horizon
                earliest = max(problem.now+cutoff, window['start'])
                if earliest+duration > window['end']:
                    continue
                moment = max(earliest+duration/2, min(window['end']-duration/2, stamp(window['best_time_utc'])))
                g = self.geometry(row['tile_id'], moment)
                if g['altitude_deg'] < 30:
                    continue
                factor = g['lunar_quality_factor']/g['airmass']**float(self.contract['weather_score_interface']['airmass_exponent'])
                probability = self.survival(row['tile_id'], moment-duration/2, moment+duration/2, problem.origin) if self.tail_forecasts else 1.0
                key = window['night_id']
                if key not in moments or factor*probability > moments[key][1]*moments[key][2]:
                    moments[key] = moment, factor, probability
            if self.tail_mode == 'seasonal-cost':
                later = [(ts, factor) for ts, factor in self.seasonal.get(row['tile_id'], [])
                         if ts >= problem.now+problem.horizon+duration/2]
            else:
                later = []
            risk = self.failure**max(len(moments), len(later))
            if self.tail_forecasts and self.tail_mode != 'seasonal-cost':
                risk = math.prod(1-(1-self.failure)*p for _, _, p in moments.values())
            if self.tail_mode == 'current-window':
                current_night = problem.snapshot['cursor']['night_id']
                current_end = max((w['end'] for w in problem.windows[tile] if w['night_id'] == current_night), default=problem.now)
                later_nights = sum(night != current_night for night in moments)
                hours = max(0, (current_end-problem.now-duration)/3600)
                risk = self.failure**(later_nights+min(.8, hours/6)) if moments else 1.0
            days_left = max(0, (stamp(row['available_until_utc'])-problem.now)/86400)
            risks.append(max(risk, max(0, 1-days_left/max(.01, self.deadline_days))))
            def reward(quality, factor):
                q = min(float(self.contract['weather_score_interface']['maximum_weather_quality']), quality*factor)
                threshold = self.config['quality_thresholds']
                band = 'DARK' if q >= threshold['dark'] else 'BRIGHT' if q >= threshold['bright'] else 'BACKUP'
                return float(row['tile_science_value'])*q*(1+self.config['program_bonus'][band])
            if self.tail_mode == 'seasonal-cost':
                lengths = [min(43200, max(0, w['end']-w['start'])) for w in problem.windows[tile]]
                visible_seconds = sum(lengths)/len(lengths) if lengths else 21600
                # Shared waiting is charged once globally; splitting its public
                # cost discourages keeping the telescope idle for tiny gains.
                choices = [max(0, reward(target, factor)-(i+1)*visible_seconds*problem.wait_cost/cost_share)
                           for i, (_, factor) in enumerate(sorted(later))]
                reserve.append(max(choices, default=0)*(1-risk))
            elif self.tail_mode in ('threshold', 'current-window'):
                reserve.append(max((reward(target, factor) for _, factor, p in moments.values() if p > 0), default=0)*(1-risk))
            else:
                continuation = 0.0
                for _, factor, probability in sorted(moments.values(), reverse=True):
                    success = (1-self.failure)*probability
                    continuation = (1-success)*continuation + success*sum(max(continuation, reward(q, factor)) for q in samples)/len(samples)
                reserve.append(continuation)
        return reserve, risks

    def reserve_request_rewards(self, problem):
        """Approximate request urgency from published later completion chances.

        One weather opportunity per night is conservative for repeat visits.
        The Poisson-binomial recurrence handles multi-tile and multi-visit tasks.
        Only the hypothetical problem changes, never received request progress.
        """
        if not self.request_reserve:
            return
        for req in problem.requests:
            item_success = []
            for tile, required, index in req['items']:
                remaining = max(0, required-problem.initial.visits[index])
                probabilities = {}
                if tile is not None and remaining:
                    duration = int(problem.catalog[tile]['nominal_exptime_seconds'])
                    for window in problem.windows[tile]:
                        start = max(window['start'], problem.now+problem.horizon, problem.now+req['start'])
                        end = min(window['end'], problem.now+req['deadline'])
                        if start+duration <= end:
                            p = (1-self.failure)*self.survival(problem.ids[tile], start, start+duration, problem.origin)
                            key = window['night_id']
                            probabilities[key] = max(probabilities.get(key, 0), p)
                distribution = [1.0]+[0.0]*remaining
                for p in probabilities.values():
                    after = [0.0]*len(distribution)
                    for count, mass in enumerate(distribution):
                        after[count] += mass*(1-p)
                        after[min(remaining, count+1)] += mass*p
                    distribution = after
                item_success.append(distribution[-1])
            count = int(req['required_tile_count'])
            distribution = [1.0]+[0.0]*count
            for p in item_success:
                after = [0.0]*len(distribution)
                for complete, mass in enumerate(distribution):
                    after[complete] += mass*(1-p)
                    after[min(count, complete+1)] += mass*p
                distribution = after
            # Future requests compete for telescope time and share weather.
            # Shrink the independent-opportunity estimate towards full urgency
            # when that approximation is given less confidence.
            urgency = 1-self.request_reservation_weight*(1-max(.001, 1-distribution[-1]))
            req['undiscounted_value'] = req['value']
            req['value'] *= urgency
            req['planning_urgency'] = urgency

    def choose(self, snapshot, previews, default, formal=None, detector=None):
        fallback = super().choose(snapshot, previews, default, formal, detector)
        self.last_diagnostics.update(scheduler='window-search-v1', search_calls=self.search_calls,
                                     budget_exhaustions=self.budget_hits, horizon_seconds=self.horizon,
                                     beam_width=self.width, reservation_model=self.tail_mode, expansions=0)
        self.last_diagnostics['weather_blend'] = self.weather_blend
        self.last_diagnostics['wait_share'] = self.wait_share
        self.last_diagnostics['keep_first_choices'] = self.keep_first_choices
        self.last_diagnostics['use_forecasts'] = self.use_forecasts
        self.last_diagnostics['segment_weather'] = self.segment_weather
        self.last_diagnostics['tail_forecasts'] = self.tail_forecasts
        self.last_diagnostics['request_reserve'] = self.request_reserve
        self.last_diagnostics['request_reservation_weight'] = self.request_reservation_weight
        if self.config.get('repeat_observation'):
            self.last_diagnostics['search_fallback'] = 'repeat_mechanics_use_marginal_planner'
            return fallback
        if not previews:
            self.last_diagnostics['search_fallback'] = 'no_executable_first_observation'
            return fallback
        now = stamp(snapshot['cursor']['timestamp_utc'])
        if self.night_end <= now:
            self.night_end = max(stamp(r['window_end_utc']) for r in snapshot['candidate_tiles'])
        problem = (SegmentWeatherProblem if self.segment_weather else PublicProblem)(self, snapshot)
        # Current candidates may accompany a partial or absent publication.
        for tile, windows in enumerate(problem.windows):
            raw = problem.raws.get(problem.ids[tile])
            if raw and not any(w['start'] <= now and now+int(raw['nominal_exptime_seconds']) <= w['end'] for w in windows):
                windows.append({**raw, 'start': stamp(raw['window_start_utc']), 'end': stamp(raw['window_end_utc']),
                                'night_id': snapshot['cursor']['night_id'], 'best_time_utc': raw.get('best_time_utc', raw['window_end_utc'])})
        reserve, risks = self.future_values(problem)
        self.reserve_request_rewards(problem)
        weather = [public_weather(problem.raws.get(t, {}).get('effective_weather', snapshot['current_site_weather'])) for t in problem.ids]
        weather_scale = {}
        def weather_at(tile, elapsed):
            steps = int((problem.offset+elapsed)//900)
            if steps not in weather_scale:
                predicted = self.weather_model.predict(steps, self.weather_blend)
                weather_scale[steps] = predicted/self.weather_model.current if self.weather_model.current else 1.0
            return replace(weather[tile], quality=weather[tile].quality*weather_scale[steps])
        permitted = {(r.tile_id, r.request_id) for r in previews}
        queues = {0: [(problem.initial, None)]}
        timeline = [0]
        finished = []
        expansions = 0
        survival_cache = {}
        self.search_calls += 1

        def value(state):
            return state.score+problem.request_potential(state)

        def enqueue(state, first):
            if state.time == problem.horizon:
                finished.append((value(state), first))
                return
            if state.time not in queues:
                queues[state.time] = []
                heapq.heappush(timeline, state.time)
            queues[state.time].append((state, first))

        while timeline and expansions < self.budget:
            elapsed = heapq.heappop(timeline)
            unique = {}
            for state, first in queues.pop(elapsed):
                key = state.done, state.visits, first if self.keep_first_choices else None
                if key not in unique or value(state) > value(unique[key][0]):
                    unique[key] = state, first
            ordered = sorted(unique.values(), key=lambda pair: value(pair[0]), reverse=True)
            if self.keep_first_choices:
                representatives = {}
                for pair in ordered:
                    representatives.setdefault(pair[1], pair)
                # Keep waiting as a competing first decision even when its
                # prefix score is smaller before a valuable window opens.
                protected = sorted(representatives.values(), key=lambda pair: (pair[1] == ('wait',), value(pair[0])), reverse=True)
                states = protected[:self.width]
                retained = {id(pair) for pair in states}
                states.extend(pair for pair in ordered if id(pair) not in retained)
                states = states[:self.width]
            else:
                states = ordered[:self.width]
            # Physical opportunities depend on time/weather; task progress is
            # filtered independently inside each hypothetical state.
            for state, first in states:
                actions = problem.actions(state, lambda tile: weather_at(tile, elapsed))
                ordinary_available = any(not state.done & (1 << a.tile) for a in actions)
                if elapsed == 0:
                    actions = [a for a in actions if (problem.ids[a.tile], problem.requests[a.request]['request_id'] if a.request >= 0 else '') in permitted]
                end_wait = min(problem.horizon, elapsed+900-(problem.offset+elapsed)%900)
                cost = (end_wait-elapsed)*problem.wait_cost if ordinary_available else 0.0
                enqueue(replace(state, time=end_wait, score=state.score-cost), first if first is not None else ('wait',))
                expansions += 1
                for action in actions:
                    estimate = problem.estimate(action.tile, elapsed, weather_at(action.tile, elapsed))
                    first_credit = not state.done & (1 << action.tile)
                    gain = estimate[0]-reserve[action.tile] if first_credit else 0.0
                    gain += problem.credit(state, action.tile)*risks[action.tile]
                    visits = problem.visit(state, action)
                    if self.use_forecasts:
                        key = action.tile, elapsed
                        if key not in survival_cache:
                            survival_cache[key] = self.survival(problem.ids[action.tile], now+elapsed,
                                  now+elapsed+int(problem.catalog[action.tile]['nominal_exptime_seconds']), problem.origin)
                        probability = survival_cache[key]
                        request_delta = problem.request_potential(replace(state, visits=visits))-problem.request_potential(state)
                        gain = probability*gain-(1-probability)*request_delta
                    child = State(elapsed+int(problem.catalog[action.tile]['nominal_exptime_seconds']),
                                  state.done | (1 << action.tile), visits, state.score+gain)
                    root_action = first if first is not None else (problem.ids[action.tile], action.program,
                                      problem.requests[action.request]['request_id'] if action.request >= 0 else '')
                    enqueue(child, root_action)
                    expansions += 1
        if timeline:
            self.budget_hits += 1
        self.last_diagnostics.update(search_calls=self.search_calls, expansions=expansions,
                                     budget_exhaustions=self.budget_hits, completed_paths=len(finished))
        if not finished:
            self.last_diagnostics['search_fallback'] = 'no_complete_path_within_budget'
            return fallback
        predicted, first = max(finished, key=lambda item: item[0])
        self.last_diagnostics.update(search_fallback=None, predicted_schedule_utility=predicted,
                                     outcome='wait' if first[0] == 'wait' else 'observe')
        if first[0] == 'wait':
            return {'action': 'wait', 'tile_id': '', 'program': '', 'request_id': '',
                    'reason': 'joint public-window schedule preserves a better opportunity', 'decision_source': 'window-search-v1'}
        tile, program, request = first
        raw = problem.raws[tile]
        prediction = self.predict(raw, now, problem.origin, {})
        if prediction:
            self.estimates[(tile, program, request)] = prediction[2:]
        if self.segment_weather:
            index = problem.indices[tile]
            observed = weather_at(index, 0)
            estimate = problem.estimate(index, 0, observed)
            base = problem.base_scores[index, 0, observed]
            self.estimates[(tile, program, request)] = base, estimate[2][program]-base
        return {'action': 'observe', 'tile_id': tile, 'program': program, 'request_id': request,
                'reason': 'joint public-window schedule with dynamic request visits and quotas', 'decision_source': 'window-search-v1'}
