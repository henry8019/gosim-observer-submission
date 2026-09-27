"""First-credit stopping experiments using public astronomy and past weather only."""
import math
import os

from practice_long_horizon import LongHorizonPlanner
from rolling_planner import stamp


class FirstCreditPlanner(LongHorizonPlanner):
    def __init__(self, initial):
        super().__init__(initial)
        if initial['scoring_contract']['score_config'].get('repeat_observation'):
            raise ValueError('First-credit experiments require the original practice rules')
        if not self.seasonal:
            self.seasonal = self.seasonal_opportunities()
        self.cost = float(os.getenv('SAC_STOP_COST', '1'))
        self.share_power = float(os.getenv('SAC_STOP_SHARE_POWER', '.5'))
        self.future_days = float(os.getenv('SAC_STOP_DAYS', '180'))
        self.short_reserve = float(os.getenv('SAC_STOP_SHORT', '1'))
        self.min_target = float(os.getenv('SAC_STOP_TARGET', '.9'))
        self.empirical = bool(int(os.getenv('SAC_STOP_EMPIRICAL', '0')))
        self.request_factor = float(os.getenv('SAC_STOP_REQUEST', '1'))
        self.request_delay = float(os.getenv('SAC_STOP_REQUEST_DELAY', '0'))
        self.day_hours = float(os.getenv('SAC_STOP_HOURS', '8'))
        self.beam = bool(int(os.getenv('SAC_STOP_BEAM', '0')))
        self.beam_hours = int(os.getenv('SAC_STOP_BEAM_SECONDS', '7200'))
        self.beam_width = int(os.getenv('SAC_STOP_BEAM_WIDTH', '16'))

    def reservation(self, tile, now, value, terminal, outstanding, target):
        points = [(ts, g) for ts, g in self.seasonal.get(tile, []) if ts > now+900]
        if not points:
            return 0.0, 1.0
        best = 0.0
        count = 0
        rate = float(self.config['penalties'].get('avoidable_wait_per_second', .001))
        share = max(1, outstanding ** self.share_power)
        for ts, g in points:
            days = (ts-now)/86400
            if days > self.future_days:
                continue
            count += 1
            q = min(3, target*g)
            band = 'DARK' if q >= self.config['quality_thresholds']['dark'] else 'BRIGHT' if q >= self.config['quality_thresholds']['bright'] else 'BACKUP'
            science = value*q*(1+self.config['program_bonus'][band])
            waiting_cost = days*self.day_hours*3600*rate*self.cost/share
            best = max(best, science-waiting_cost)
        risk = self.failure**len(points)
        return best*(1-risk), risk

    def choose(self, snapshot, previews, default):
        if snapshot['schema_version'] != 'decision-snapshot-v2':
            raise ValueError('First-credit policy only accepts pre-repeat snapshots')
        now = stamp(snapshot['cursor']['timestamp_utc'])
        origin = now-int(snapshot['cursor']['slot_offset_seconds'])
        completed = set(snapshot['progress'].get('completed_tile_ids', []))
        outstanding = sum(t not in completed and stamp(r['available_until_utc']) > now
                          for t, r in self.catalog.items())
        raws = {r['tile_id']: r for r in snapshot['candidate_tiles']}
        requests = {r['request_id']: r for r in snapshot.get('active_requests', [])}
        target = max(self.min_target, self.weather_target()) if self.empirical else self.min_target
        options, cache, plans = [], {}, []
        for row in previews:
            raw = raws[row.tile_id]
            first = row.tile_id not in completed
            request = requests.get(row.request_id)
            if not first and not request:
                continue
            if row.request_id and (request is None or request.get('is_complete') or
                    now < stamp(request['available_from_utc']) or
                    now+row.nominal_exptime_seconds > stamp(request['deadline_utc'])):
                continue
            predicted = self.estimate(raw, now, origin, cache)
            if predicted is None:
                continue
            science, program = predicted
            reserve, risk = self.reservation(row.tile_id, now, float(raw['tile_science_value']),
                    row.terminal_penalty_avoidance, outstanding, target) if first else (0.0, 0.0)
            # Same-night improvement uses current weather held constant, never future weather.
            later = [self.estimate(raw, now+d, origin, cache) for d in (900, 1800, 3600, 7200, 10800)] if first and self.short_reserve and not self.beam else []
            short = max((p[0] for p in later if p is not None), default=0)
            reserve = max(reserve, short*self.short_reserve)
            request_credit = 0.0
            if request:
                need = next(r for r in request['tile_requirements'] if r['tile_id'] == row.tile_id)
                visits = max(1, int(need.get('remaining_visits', 1)))
                request_credit = row.request_policy_value/visits*self.request_factor
                if self.request_delay:
                    days = max(0, (stamp(request['deadline_utc'])-now)/86400)
                    request_credit *= math.exp(-self.request_delay*days)
            gain = (science-reserve+row.terminal_penalty_avoidance*risk if first else 0)+request_credit
            plans.append((row, raw, first, gain-(science if first else 0)))
            # Waiting has an immediate published cost when useful starts exist.
            gain += float(self.config['penalties'].get('avoidable_wait_per_second', .001))*row.nominal_exptime_seconds
            options.append((gain/row.nominal_exptime_seconds**self.rate_power,
                            science if first else 0, row.tile_id, program, row.request_id))
        if self.beam and plans:
            return self.choose_beam(snapshot, plans, cache)
        if not options or max(options)[0] <= 0:
            return {'action': 'wait', 'tile_id': '', 'program': '', 'request_id': '',
                    'reason': 'preserve first credit after charging public estimated waiting cost'}
        _, _, tile, program, request = max(options)
        return {'action': 'observe', 'tile_id': tile, 'program': program, 'request_id': request,
                'reason': 'first credit opportunity with seasonal geometry and waiting cost'}

    def choose_beam(self, snapshot, plans, cache):
        now = stamp(snapshot['cursor']['timestamp_utc'])
        origin = now-int(snapshot['cursor']['slot_offset_seconds'])
        requests = {r['request_id']: r for r in snapshot.get('active_requests', [])}
        plans.sort(key=lambda p: (((self.estimate(p[1], now, origin, cache) or (0, ''))[0]
                                  if p[2] else 0)+p[3])/p[0].nominal_exptime_seconds, reverse=True)
        plans = plans[:20]
        horizon = self.beam_hours
        buckets = {0: [(0.0, frozenset(), None)]}
        finished = []
        for elapsed in range(0, horizon+1, 150):
            unique = {}
            for state in buckets.pop(elapsed, []):
                key = state[1], state[2]
                if key not in unique or state[0] > unique[key][0]:
                    unique[key] = state
            states = sorted(unique.values(), key=lambda s: s[0], reverse=True)[:self.beam_width]
            for value, used, first_action in states:
                if elapsed == horizon:
                    finished.append((value, first_action))
                    continue
                moment = now+elapsed
                stop = min(horizon, elapsed+int(900-(moment-origin)%900))
                cost = (stop-elapsed)*float(self.config['penalties'].get('avoidable_wait_per_second', .001))
                buckets.setdefault(stop, []).append((value-cost, used, first_action or ('wait', '')))
                for i, (row, raw, first_credit, adjustment) in enumerate(plans):
                    end = elapsed+row.nominal_exptime_seconds
                    if row.tile_id in used or end > horizon:
                        continue
                    request = requests.get(row.request_id)
                    if request and now+end > stamp(request['deadline_utc']):
                        continue
                    predicted = self.estimate(raw, moment, origin, cache)
                    if predicted is None:
                        continue
                    gain = (predicted[0] if first_credit else 0)+adjustment
                    if gain <= 0:
                        continue
                    buckets.setdefault(end, []).append((value+gain, used|{row.tile_id}, first_action or (i, predicted[1])))
        first = max(finished, key=lambda s: s[0])[1] if finished else None
        if first is None or first[0] == 'wait':
            return {'action': 'wait', 'tile_id': '', 'program': '', 'request_id': '',
                    'reason': 'first credit beam preserves a better public opportunity'}
        row = plans[first[0]][0]
        return {'action': 'observe', 'tile_id': row.tile_id, 'program': first[1], 'request_id': row.request_id,
                'reason': 'first credit beam with seasonal reservation and waiting cost'}
