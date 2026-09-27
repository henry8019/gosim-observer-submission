"""Causal marginal-value scheduling shared by both public scoring contracts.

The lookahead is an opportunity-cost approximation, not an optimal schedule.
It holds observed weather fixed and uses only received windows and history.
"""
from collections import Counter, defaultdict
import math
import os

from formal_optimizer import coverage_delta
from practice_long_horizon import LongHorizonPlanner
from rolling_planner import stamp


class OpportunityPlanner(LongHorizonPlanner):
    def __init__(self, initial):
        super().__init__(initial)
        self.lookahead_weight = float(os.getenv('SAC_OPPORTUNITY_LOOKAHEAD', '.5'))
        if not math.isfinite(self.lookahead_weight) or not 0 <= self.lookahead_weight <= 1:
            raise ValueError('SAC_OPPORTUNITY_LOOKAHEAD must be between zero and one')
        self.calls = 0
        self.last_diagnostics = {}
        self.estimates = {}

    def predict(self, raw, start, origin, cache):
        """Integrate public geometry across the complete exposure; choose one program."""
        key = raw['tile_id'], start
        if key in cache:
            return cache[key]
        duration = int(raw['nominal_exptime_seconds'])
        if start < stamp(raw['window_start_utc']) or start+duration > stamp(raw['window_end_utc']):
            cache[key] = None
            return None
        weather = raw['effective_weather']
        if not weather['is_observable']:
            cache[key] = None
            return None
        interface = self.contract['weather_score_interface']
        multiplier = float(weather['transparency'])*float(weather['sky_quality'])/float(weather['seeing_arcsec'])
        efficiency = float(weather.get('instrument_efficiency', 1))
        formal = bool(self.config.get('repeat_observation'))
        totals = {program: 0.0 for program in self.config['program_bonus']}
        base = 0.0
        cursor, end = start, start+duration
        while cursor < end:
            stop = min(end, origin+(math.floor((cursor-origin)/900)+1)*900)
            g = self.geometry(raw['tile_id'], (cursor+stop)/2)
            if g['altitude_deg'] < 30 or self.geometry(raw['tile_id'], cursor)['altitude_deg'] < 30:
                cache[key] = None
                return None
            atmosphere = multiplier/g['airmass']**float(interface['airmass_exponent'])
            cap = float(interface['maximum_weather_quality'])
            quality = min(cap, atmosphere*efficiency)*g['lunar_quality_factor']
            band_quality = min(cap, atmosphere)*g['lunar_quality_factor'] if formal else quality
            thresholds = self.config['quality_thresholds']
            band = 'DARK' if band_quality >= thresholds['dark'] else 'BRIGHT' if band_quality >= thresholds['bright'] else 'BACKUP'
            part = float(raw['tile_science_value'])*quality*(stop-cursor)/duration
            base += part
            for program in totals:
                totals[program] += part*(1+self.config['program_bonus'][program] if program == band else 1)
            cursor = stop
        program = max(totals, key=totals.get)
        result = totals[program], program, base, totals[program]-base
        cache[key] = result
        return result

    @staticmethod
    def request_credit(request, tile, preview_credit):
        if not request or request.get('is_complete'):
            return 0.0
        requirement = next((r for r in request.get('tile_requirements', []) if r['tile_id'] == tile), None)
        if requirement is None:
            return 0.0
        remaining = int(requirement.get('remaining_visits', requirement.get('required_visits', 0)))
        return preview_credit/remaining if remaining > 0 else 0.0

    def choose(self, snapshot, previews, default, formal=None, detector=None):
        self.calls += 1
        self.estimates = {}
        now = stamp(snapshot['cursor']['timestamp_utc'])
        origin = now-int(snapshot['cursor']['slot_offset_seconds'])
        repeats = self.config.get('repeat_observation', {}).get('tile_score_aggregation') == 'max'
        completed = set(snapshot.get('progress', {}).get('completed_tile_ids', []))
        raws = {r['tile_id']: r for r in snapshot['candidate_tiles']}
        requests = {r['request_id']: r for r in snapshot.get('active_requests', [])}
        windows = defaultdict(list)
        for key, window in self.windows.items():
            if window['end'] > now:
                windows[window['tile_id']].append((window, self.window_geometry[key]))
        counts = Counter(self.catalog[t]['region_id'] for t in completed if t in self.catalog)
        regions = sorted({r['region_id'] for r in self.catalog.values()})
        vector = [counts[r] for r in regions]
        total_base = sum(formal.base_bests.values()) if formal is not None else 0.0
        cache, options = {}, []
        target = self.weather_target()
        for row in previews:
            raw = raws[row.tile_id]
            request = requests.get(row.request_id)
            duration = row.nominal_exptime_seconds
            if row.request_id and (request is None or request.get('is_complete') or
                    not stamp(request['available_from_utc']) <= now < now+duration <= stamp(request['deadline_utc'])):
                continue
            first = row.tile_id not in completed
            if not first and not repeats and not request:
                continue
            predicted = self.predict(raw, now, origin, cache)
            if predicted is None:
                continue
            score, program, base, bonus = predicted
            self.estimates[(row.tile_id, program, row.request_id)] = (base, bonus)
            factor = formal.factor(row.tile_id) if formal is not None else 1.0
            old = formal.bests.get(row.tile_id, detector.bests.get(row.tile_id, 0)) if formal is not None else self.bests.get(row.tile_id, 0)
            earned = score*factor if first else max(0, score*factor-old) if repeats else 0.0
            future = [(w, g) for w, g in windows[row.tile_id]
                      if w['end'] >= now+duration+900 and w['start'] <= stamp(self.catalog[row.tile_id]['available_until_utc'])]
            later_nights = {w['night_id'] for w, _ in future if w['night_id'] != snapshot['cursor']['night_id']}
            hours = max(0, (stamp(raw['window_end_utc'])-now-duration)/3600)
            risk = self.failure**(len(later_nights)+min(.8, hours/6)) if future else 1.0
            remaining_days = max(0, (stamp(self.catalog[row.tile_id]['available_until_utc'])-now)/86400)
            urgency = max(risk, max(0, 1-remaining_days/max(.01, self.deadline_days)))
            reserve = 0.0
            if first and not repeats:
                quality = min(3, target*max((g for _, g in future), default=0))
                band = 'DARK' if quality >= self.config['quality_thresholds']['dark'] else 'BRIGHT' if quality >= self.config['quality_thresholds']['bright'] else 'BACKUP'
                reserve = float(raw['tile_science_value'])*quality*(1+self.config['program_bonus'][band])*(1-risk)
            # Waiting can improve geometry, but consumes telescope time and has
            # a published cost. This estimate never reads future weather.
            later_gain = 0.0
            for delay in (900, 1800, 3600):
                if not self.lookahead_weight:
                    break
                later = self.predict(raw, now+delay, origin, cache)
                if later is None:
                    continue
                value = later[0]*factor
                value = value if first else max(0, value-old) if repeats else 0.0
                waiting_cost = delay*float(self.config['penalties'].get('avoidable_wait_per_second', 0))
                later_gain = max(later_gain, value-earned-waiting_cost)
            base_gain = base*factor-(formal.base_bests.get(row.tile_id, 0) if formal is not None else 0)
            coverage = coverage_delta(float(self.config.get('coverage_bonus_weight', 0)), total_base,
                                      base_gain if first or earned > 0 else 0, vector,
                                      regions.index(row.region_id), first) if formal is not None else 0.0
            info = formal.information_value(row, snapshot, detector) if formal is not None else 0.0
            credit = self.request_credit(request, row.tile_id, row.request_policy_value)
            value = earned-reserve+row.terminal_penalty_avoidance*urgency+credit+coverage+info
            value -= self.lookahead_weight*later_gain
            value += duration*float(self.config['penalties'].get('avoidable_wait_per_second', 0))
            detail = {'science_gain': earned, 'reservation': reserve, 'deadline_risk': urgency,
                      'request_credit': credit, 'coverage_gain': coverage,
                      'geometry_opportunity': later_gain, 'utility': value}
            options.append((value/duration, earned/duration, row.tile_id, program, row.request_id, detail))
        self.last_diagnostics = {'scheduler': 'opportunity-v1', 'planner_calls': self.calls,
                                 'lookahead_weight': self.lookahead_weight, 'valid_options': len(options)}
        if not options or max(options, key=lambda o: o[:5])[0] <= 0:
            self.last_diagnostics['outcome'] = 'wait'
            return {'action': 'wait', 'tile_id': '', 'program': '', 'request_id': '',
                    'reason': 'causal opportunity comparison prefers waiting', 'decision_source': 'opportunity-v1'}
        _, _, tile, program, request, detail = max(options, key=lambda o: o[:5])
        self.last_diagnostics.update(detail, outcome='observe')
        return {'action': 'observe', 'tile_id': tile, 'program': program, 'request_id': request,
                'reason': 'marginal science and deadline risk with public geometry opportunity cost',
                'decision_source': 'opportunity-v1'}
