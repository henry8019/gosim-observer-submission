"""Rolling decisions from public state only; no scenario files or future weather.

Weather is held constant inside the short search horizon. Published geometry is
predictable; longer-term weather quality is estimated from past visible samples.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import os

from public_geometry import Tile, geometry_sample


def stamp(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


class RollingPlanner:
    def __init__(self, initial):
        self.initial = initial
        self.catalog = {r["tile_id"]: r for r in initial["tile_catalog"]["tiles"]}
        self.coords = {k: Tile(float(r["ra_deg"]), float(r["dec_deg"])) for k, r in self.catalog.items()}
        self.contract = initial["scoring_contract"]
        self.config = self.contract["score_config"]
        self.tile_config = {"lunar_model": self.contract["lunar_model"]}
        self.calendar_config = {"site": initial["site"]}
        self.windows = {}
        self.window_geometry = {}
        self.weather_history = []
        self.last_slot = None
        self.mode = os.environ.get("SAC_PLANNER_MODE", "beam")
        self.quantile = float(os.environ.get("SAC_PLANNER_QUANTILE", ".8"))
        self.prior = float(os.environ.get("SAC_PLANNER_PRIOR", ".7"))
        self.reserve_weight = float(os.environ.get("SAC_PLANNER_RESERVE", "1"))
        self.failure = float(os.environ.get("SAC_PLANNER_FAILURE", ".6"))
        self.horizon = int(os.environ.get("SAC_PLANNER_HORIZON", "7200"))
        self.width = int(os.environ.get("SAC_PLANNER_WIDTH", "16"))

    def geometry(self, tile, timestamp):
        return geometry_sample(self.coords[tile], datetime.fromtimestamp(timestamp, timezone.utc), self.tile_config, self.calendar_config)

    def update(self, snapshot):
        # Called even on closed slots, before the wrapper's no-candidate shortcut.
        for publication in (snapshot.get("weekly"), snapshot.get("night_start")):
            for row in (publication or {}).get("tile_windows", []):
                key = row["window_id"]
                if key not in self.windows:
                    start, end = stamp(row["window_start_utc"]), stamp(row["window_end_utc"])
                    exposure = int(row["nominal_exptime_seconds"])
                    best = max(start + exposure / 2, min(end - exposure / 2, stamp(row["best_time_utc"])))
                    geom = self.geometry(row["tile_id"], best)
                    self.windows[key] = {**row, "start": start, "end": end}
                    self.window_geometry[key] = geom["lunar_quality_factor"] / geom["airmass"]
        slot = snapshot["cursor"]["slot_id"]
        weather = snapshot["current_site_weather"]
        if slot != self.last_slot and weather.get("is_observable"):
            q = float(weather["transparency"]) * float(weather["sky_quality"]) / float(weather["seeing_arcsec"])
            q *= float(weather.get("instrument_efficiency", 1))
            self.weather_history.append(q)
            if len(self.weather_history) > 1024:
                del self.weather_history[:256]
        self.last_slot = slot

    def weather_target(self):
        if not self.weather_history:
            return self.prior
        values = sorted(self.weather_history)
        empirical = values[min(len(values)-1, int(self.quantile * (len(values)-1)))]
        confidence = min(1, len(values) / 40)
        return max(.15, (1-confidence) * self.prior + confidence * empirical)

    def estimate(self, raw, start, slot_origin, cache):
        key = (raw["tile_id"], start)
        if key in cache:
            return cache[key]
        exposure = int(raw["nominal_exptime_seconds"])
        end = start + exposure
        if start < stamp(raw["window_start_utc"]) or end > stamp(raw["window_end_utc"]):
            return None
        weather = raw["effective_weather"]
        if not weather["is_observable"]:
            return None
        multiplier = float(weather["transparency"]) * float(weather["sky_quality"]) / float(weather["seeing_arcsec"])
        multiplier *= float(weather.get("instrument_efficiency", 1))
        pieces = []
        cursor = start
        while cursor < end:
            boundary = slot_origin + (math.floor((cursor-slot_origin)/900)+1)*900
            stop = min(end, boundary)
            geometry = self.geometry(raw["tile_id"], (cursor+stop)/2)
            if geometry["altitude_deg"] < 30:
                cache[key] = None
                return None
            q = min(multiplier / geometry["airmass"], 3) * geometry["lunar_quality_factor"]
            thresholds = self.config["quality_thresholds"]
            band = "DARK" if q >= thresholds["dark"] else "BRIGHT" if q >= thresholds["bright"] else "BACKUP"
            base = float(raw["tile_science_value"]) * q * (stop-cursor)/exposure
            pieces.append((base, band))
            cursor = stop
        scores = {p: sum(base*(1+self.config["program_bonus"][p] if p==band else 1) for base,band in pieces) for p in ("DARK","BRIGHT","BACKUP")}
        program = max(scores, key=scores.get)
        cache[key] = (scores[program], program)
        return cache[key]

    def choose(self, snapshot, previews, default):
        if not previews:
            return default
        now = stamp(snapshot["cursor"]["timestamp_utc"])
        slot_origin = now - int(snapshot["cursor"]["slot_offset_seconds"])
        raw_by_id = {r["tile_id"]: r for r in snapshot["candidate_tiles"]}
        options = []
        cache = {}
        available = defaultdict(list)
        for key, window in self.windows.items():
            if window["end"] > now:
                available[window["tile_id"]].append((window, self.window_geometry[key]))
        target = self.weather_target()
        for row in previews:
            raw = raw_by_id[row.tile_id]
            estimate = self.estimate(raw, now, slot_origin, cache)
            if estimate is None:
                continue
            score, program = estimate
            if self.mode == "accurate":
                reservation, risk = 0, 1
            else:
                future = [(w,g) for w,g in available[row.tile_id] if w["end"] >= now + row.nominal_exptime_seconds + 900]
                later_nights = {w["night_id"] for w,g in future if w["night_id"] != snapshot["cursor"]["night_id"]}
                remaining_hours = max(0, (stamp(raw["window_end_utc"])-now-row.nominal_exptime_seconds)/3600)
                chances = len(later_nights) + min(.8, remaining_hours/6)
                risk = self.failure ** chances if future else 1
                best_geometry = max((g for w,g in future), default=0)
                q = target * best_geometry
                thresholds = self.config["quality_thresholds"]
                band = "DARK" if q >= thresholds["dark"] else "BRIGHT" if q >= thresholds["bright"] else "BACKUP"
                reservation = float(raw["tile_science_value"]) * q * (1+self.config["program_bonus"][band]) * (1-risk) * self.reserve_weight
            credit = row.terminal_penalty_avoidance * risk + row.request_policy_value
            adjustment = credit - reservation
            options.append((row, raw, adjustment, score, program))
        if not options:
            return default
        options.sort(key=lambda x: (x[3]+x[2])/x[0].nominal_exptime_seconds, reverse=True)

        def observe(index, program):
            row = options[index][0]
            return {"action":"observe", "tile_id":row.tile_id,"program":program,"request_id":row.request_id,
                    "reason":"rolling public-geometry plan with earned science and future opportunity value","decision_source":"rolling_planner"}

        def wait():
            return {"action":"wait","tile_id":"","program":"","request_id":"",
                    "reason":"preserve a later public opportunity with higher expected science","decision_source":"rolling_planner"}

        if self.mode in ("accurate", "reserve"):
            if options[0][3]+options[0][2] <= 0:
                return wait()
            return observe(0, options[0][4])

        # Time-indexed beam search: compare plans at the same elapsed time,
        # rather than rewarding paths merely for containing more short actions.
        options = options[:20]
        horizon = self.horizon
        buckets = {0: [(0.0, frozenset(), None)]}
        finished = []
        for elapsed in range(0, horizon+1, 150):
            states = buckets.pop(elapsed, [])
            unique = {}
            for state in states:
                key = (state[1], state[2])
                if key not in unique or state[0] > unique[key][0]:
                    unique[key] = state
            states = sorted(unique.values(), key=lambda x:x[0], reverse=True)[:self.width]
            for value, used, first in states:
                moment = now + elapsed
                if elapsed == horizon:
                    finished.append((value, first))
                    continue
                duration = int(900-((moment-slot_origin)%900))
                new_elapsed = min(horizon, elapsed+duration)
                wait_value = value - (new_elapsed-elapsed)*float(self.config["penalties"].get("avoidable_wait_per_second", .001))
                buckets.setdefault(new_elapsed,[]).append((wait_value,used,first or ("wait",None)))
                for index, (row, raw, adjustment, _, _) in enumerate(options):
                    if row.tile_id in used:
                        continue
                    end = elapsed + row.nominal_exptime_seconds
                    if end > horizon:
                        continue
                    predicted = self.estimate(raw,moment,slot_origin,cache)
                    if predicted is None:
                        continue
                    score, program = predicted
                    if score+adjustment <= 0:
                        continue
                    buckets.setdefault(end,[]).append((value+score+adjustment,used|{row.tile_id},first or (index,program)))
        if not finished:
            return wait()
        _, first = max(finished,key=lambda x:x[0])
        return wait() if first is None or first[0]=="wait" else observe(*first)
