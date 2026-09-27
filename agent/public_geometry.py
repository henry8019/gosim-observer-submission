"""Pure public geometry formula copied from the official starter kit.
Only coordinates, site, lunar model and timestamps are inputs; no file access.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Mapping
from functools import lru_cache

@dataclass(frozen=True)
class Tile:
    ra_deg: float
    dec_deg: float

def _julian_date(moment: datetime) -> float:
    return moment.timestamp() / 86400.0 + 2440587.5


def _local_sidereal_deg(moment: datetime, longitude_deg: float) -> float:
    days = _julian_date(moment) - 2451545.0
    return (280.46061837 + 360.98564736629 * days + longitude_deg) % 360.0


@lru_cache(maxsize=4096)
def _sun_equatorial_deg(moment: datetime) -> tuple[float, float]:
    days = _julian_date(moment) - 2451545.0
    mean_longitude = (280.460 + 0.9856474 * days) % 360.0
    mean_anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    longitude = math.radians(
        (mean_longitude + 1.915 * math.sin(mean_anomaly) + 0.020 * math.sin(2 * mean_anomaly))
        % 360.0
    )
    obliquity = math.radians(23.439 - 0.0000004 * days)
    return (
        math.degrees(math.atan2(math.cos(obliquity) * math.sin(longitude), math.cos(longitude)))
        % 360.0,
        math.degrees(math.asin(math.sin(obliquity) * math.sin(longitude))),
    )


@lru_cache(maxsize=4096)
def _moon_equatorial_deg(moment: datetime) -> tuple[float, float]:
    days = _julian_date(moment) - 2451545.0
    mean_longitude = math.radians((218.316 + 13.176396 * days) % 360.0)
    mean_anomaly = math.radians((134.963 + 13.064993 * days) % 360.0)
    argument_latitude = math.radians((93.272 + 13.229350 * days) % 360.0)
    longitude = mean_longitude + math.radians(6.289) * math.sin(mean_anomaly)
    latitude = math.radians(5.128) * math.sin(argument_latitude)
    obliquity = math.radians(23.439 - 0.0000004 * days)
    x = math.cos(longitude) * math.cos(latitude)
    y = (
        math.sin(longitude) * math.cos(latitude) * math.cos(obliquity)
        - math.sin(latitude) * math.sin(obliquity)
    )
    z = (
        math.sin(longitude) * math.cos(latitude) * math.sin(obliquity)
        + math.sin(latitude) * math.cos(obliquity)
    )
    return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(z))


def _angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1r, dec1r, ra2r, dec2r = map(math.radians, (ra1, dec1, ra2, dec2))
    cosine = (
        math.sin(dec1r) * math.sin(dec2r)
        + math.cos(dec1r) * math.cos(dec2r) * math.cos(ra1r - ra2r)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _normalized_airmass(altitude_deg: float) -> float:
    if altitude_deg <= 0.0:
        return float("inf")
    zenith_deg = 90.0 - altitude_deg
    raw = 1.0 / (
        math.cos(math.radians(zenith_deg))
        + 0.50572 * (96.07995 - zenith_deg) ** -1.6364
    )
    zenith_raw = 1.0 / (1.0 + 0.50572 * 96.07995**-1.6364)
    return raw / zenith_raw


def geometry_sample(
    tile: Tile,
    moment: datetime,
    tile_config: Mapping,
    calendar_config: Mapping,
) -> dict[str, float]:
    site = calendar_config["site"]
    latitude = math.radians(float(site["latitude_deg"]))
    longitude = float(site["longitude_deg"])
    declination = math.radians(tile.dec_deg)
    hour_angle_deg = (_local_sidereal_deg(moment, longitude) - tile.ra_deg + 180.0) % 360.0 - 180.0
    hour_angle = math.radians(hour_angle_deg)
    sin_altitude = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    altitude = math.asin(max(-1.0, min(1.0, sin_altitude)))
    cos_altitude = max(1e-12, math.cos(altitude))
    sin_azimuth = -math.sin(hour_angle) * math.cos(declination) / cos_altitude
    cos_azimuth = (
        math.sin(declination) - math.sin(altitude) * math.sin(latitude)
    ) / (cos_altitude * max(1e-12, math.cos(latitude)))
    azimuth = math.degrees(math.atan2(sin_azimuth, cos_azimuth)) % 360.0
    altitude_deg = math.degrees(altitude)

    sun_ra, sun_dec = _sun_equatorial_deg(moment)
    moon_ra, moon_dec = _moon_equatorial_deg(moment)
    sun_moon_separation = _angular_separation_deg(sun_ra, sun_dec, moon_ra, moon_dec)
    illumination = (1.0 - math.cos(math.radians(sun_moon_separation))) / 2.0
    tile_moon_separation = _angular_separation_deg(tile.ra_deg, tile.dec_deg, moon_ra, moon_dec)
    moon_tile = replace(tile, ra_deg=moon_ra, dec_deg=moon_dec)
    moon_altitude = geometry_sample_without_lunar(moon_tile, moment, calendar_config)["altitude_deg"]
    lunar = tile_config["lunar_model"]
    altitude_weight = math.sin(math.radians(max(0.0, moon_altitude))) ** float(
        lunar["altitude_exponent"]
    )
    angular_weight = math.exp(
        -tile_moon_separation / float(lunar["angular_decay_scale_deg"])
    )
    lunar_quality = (
        1.0
        - float(lunar["maximum_penalty"])
        * illumination
        * altitude_weight
        * angular_weight
    )
    return {
        "altitude_deg": altitude_deg,
        "azimuth_deg": azimuth,
        "hour_angle_deg": hour_angle_deg,
        "airmass": _normalized_airmass(altitude_deg),
        "moon_separation_deg": tile_moon_separation,
        "lunar_quality_factor": max(0.0, min(1.0, lunar_quality)),
    }


def geometry_sample_without_lunar(
    tile: Tile, moment: datetime, calendar_config: Mapping
) -> dict[str, float]:
    site = calendar_config["site"]
    latitude = math.radians(float(site["latitude_deg"]))
    declination = math.radians(tile.dec_deg)
    hour_angle_deg = (
        _local_sidereal_deg(moment, float(site["longitude_deg"])) - tile.ra_deg + 180.0
    ) % 360.0 - 180.0
    hour_angle = math.radians(hour_angle_deg)
    sin_altitude = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    altitude = math.asin(max(-1.0, min(1.0, sin_altitude)))
    cos_altitude = max(1e-12, math.cos(altitude))
    sin_azimuth = -math.sin(hour_angle) * math.cos(declination) / cos_altitude
    cos_azimuth = (
        math.sin(declination) - math.sin(altitude) * math.sin(latitude)
    ) / (cos_altitude * max(1e-12, math.cos(latitude)))
    return {
        "altitude_deg": math.degrees(altitude),
        "azimuth_deg": math.degrees(math.atan2(sin_azimuth, cos_azimuth)) % 360.0,
    }
