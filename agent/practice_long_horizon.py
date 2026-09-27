"""Causal long-horizon practice policy; only initialization and received snapshots."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import os

from rolling_planner import RollingPlanner, stamp
from public_geometry import Tile, _sun_equatorial_deg, geometry_sample_without_lunar


class LongHorizonPlanner(RollingPlanner):
    def __init__(self, initial):
        super().__init__(initial)
        self.policy = os.getenv('SAC_LONG_POLICY', 'threshold')
        self.scale = float(os.getenv('SAC_LONG_SCALE', '1'))
        self.prior_target = float(os.getenv('SAC_LONG_TARGET', '1'))
        self.request_weight = float(os.getenv('SAC_LONG_REQUEST', '1'))
        self.deadline_days = float(os.getenv('SAC_LONG_DEADLINE', '2'))
        self.repeat_fill = bool(int(os.getenv('SAC_LONG_REPEAT_FILL', '0')))
        self.rate_power = float(os.getenv('SAC_LONG_RATE_POWER', '1'))
        self.repeat_reserve = float(os.getenv('SAC_LONG_REPEAT_RESERVE', '0'))
        self.bests = {}
        self.pending = None
        self.seasonal = self.seasonal_opportunities() if bool(int(os.getenv('SAC_LONG_SEASONAL', '0'))) else {}

    def seasonal_opportunities(self):
        """Approximate future geometry from public dates/site/coordinates only.

        Half-hour samples model astronomy, not future weather or unpublished
        requests. They are reservation estimates, never executable candidates.
        """
        result = defaultdict(list)
        local_zone = timezone(timedelta(hours=float(self.initial['site']['utc_offset_hours'])))
        first = datetime.fromisoformat(self.initial['calendar']['first_night']).replace(tzinfo=local_zone)
        for day in range(int(self.initial['calendar']['night_count'])):
            best = {}
            for half_hour in range(48):
                moment = first+timedelta(days=day, hours=12, minutes=30*half_hour)
                sun = Tile(*_sun_equatorial_deg(moment))
                altitude = geometry_sample_without_lunar(sun, moment, self.calendar_config)['altitude_deg']
                if altitude > float(self.initial['site']['sun_altitude_limit_deg']):
                    continue
                ts = moment.timestamp()
                for tile, row in self.catalog.items():
                    if not stamp(row['available_from_utc']) <= ts < stamp(row['available_until_utc']):
                        continue
                    g = self.geometry(tile, ts)
                    if g['altitude_deg'] < 30:
                        continue
                    factor = g['lunar_quality_factor']/g['airmass']
                    if tile not in best or factor > best[tile][1]:
                        best[tile] = (ts, factor)
            for tile, pair in best.items():
                result[tile].append(pair)
        return dict(result)

    def update(self, snapshot):
        super().update(snapshot)
        feedback = snapshot.get('tile_last_finished')
        if self.pending and feedback and feedback.get('tile_id') == self.pending:
            self.bests[self.pending] = max(self.bests.get(self.pending, 0), float(feedback.get('score', 0)))
        self.pending = None

    def choose(self, snapshot, previews, default):
        now = stamp(snapshot['cursor']['timestamp_utc'])
        origin = now-int(snapshot['cursor']['slot_offset_seconds'])
        completed = set(snapshot['progress'].get('completed_tile_ids', []))
        repeat = snapshot['schema_version'] == 'decision-snapshot-v3'
        raws = {r['tile_id']: r for r in snapshot['candidate_tiles']}
        requests = {r['request_id']: r for r in snapshot.get('active_requests', [])}
        available = defaultdict(list)
        for key, window in self.windows.items():
            if window['end'] > now:
                available[window['tile_id']].append((window, self.window_geometry[key]))
        target = self.prior_target if self.policy == 'fixed' else self.weather_target()
        options = []
        cache = {}
        for row in previews:
            raw = raws[row.tile_id]
            request = requests.get(row.request_id)
            if row.request_id and (request is None or now < stamp(request['available_from_utc']) or
                                   now+row.nominal_exptime_seconds > stamp(request['deadline_utc'])):
                continue
            prediction = self.estimate(raw, now, origin, cache)
            if prediction is None:
                continue
            score, program = prediction
            first = row.tile_id not in completed
            if not first and not repeat and not row.request_id:
                continue
            earned = score if first else max(0, score-self.bests.get(row.tile_id, 0)) if repeat else 0
            future = [(w, g) for w, g in available[row.tile_id]
                      if w['end'] >= now+row.nominal_exptime_seconds+900]
            later_nights = {w['night_id'] for w, g in future
                            if w['night_id'] != snapshot['cursor']['night_id']}
            hours = max(0, (stamp(raw['window_end_utc'])-now-row.nominal_exptime_seconds)/3600)
            chances = len(later_nights)+min(.8, hours/6)
            risk = self.failure**chances if future else 1.0
            best_geometry = max((g for w, g in future), default=0)
            seasonal = [(ts, g) for ts, g in self.seasonal.get(row.tile_id, [])
                        if ts > now+row.nominal_exptime_seconds+900]
            if seasonal:
                best_geometry = max(best_geometry, max(g for ts, g in seasonal))
                risk = min(risk, self.failure**len(seasonal))
            q = min(3, target*best_geometry)
            band = 'DARK' if q >= self.config['quality_thresholds']['dark'] else 'BRIGHT' if q >= self.config['quality_thresholds']['bright'] else 'BACKUP'
            reserve = float(raw['tile_science_value'])*q*(1+self.config['program_bonus'][band])*(1-risk)*self.scale if first else 0
            if repeat and self.repeat_fill:
                reserve = 0
            request_credit = 0
            if request:
                requirement = next(r for r in request['tile_requirements'] if r['tile_id'] == row.tile_id)
                remaining = max(1, int(requirement.get('remaining_visits', 1)))
                request_credit = row.request_policy_value/remaining*self.request_weight
            # Known catalogue deadlines reduce deferral near expiration. No future weather.
            remaining_days = max(0, (stamp(self.catalog[row.tile_id]['available_until_utc'])-now)/86400)
            urgency = max(risk, max(0, 1-remaining_days/max(.01, self.deadline_days)))
            value = earned-reserve+row.terminal_penalty_avoidance*urgency+request_credit
            if repeat and self.repeat_reserve:
                later = [self.estimate(raw, now+delay, origin, cache) for delay in (900, 1800, 3600)]
                later_gain = max(0, max((v[0] for v in later if v is not None), default=score)-self.bests.get(row.tile_id, 0))
                value -= self.repeat_reserve*max(0, later_gain-earned)
            denominator = row.nominal_exptime_seconds**self.rate_power
            rate = value/denominator
            if repeat and not first and not row.request_id:
                rate = max(0, value)/denominator
            if repeat and self.repeat_fill:
                # A legal repeat never reduces the banked score. It can also avoid
                # waiting cost; no truth-based prediction of that cost is used.
                rate += float(self.config.get('penalties', {}).get('avoidable_wait_per_second', .001))
            options.append((rate, earned/row.nominal_exptime_seconds, row.tile_id, program, row.request_id))
        if not options or max(options)[0] <= 0:
            return {'action': 'wait', 'tile_id': '', 'program': '', 'request_id': '',
                    'reason': 'long horizon preserves a higher expected public opportunity'}
        _, _, tile, program, request = max(options)
        self.pending = tile
        return {'action': 'observe', 'tile_id': tile, 'program': program, 'request_id': request,
                'reason': 'long horizon marginal science with request visits and public-window urgency'}
