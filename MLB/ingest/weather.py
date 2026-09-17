"""Ingesta de clima desde Open-Meteo para partidos MLB proximos.

Open-Meteo es una API gratuita que no requiere key. Devuelve pronostico por hora
con temperatura, humedad, viento (velocidad + direccion) y precipitacion.

El resultado se guarda en la tabla `weather` que ya existe en el esquema MLB.
Se marca `is_forecast=True` y `available_at=utcnow()`.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import timedelta

from sqlalchemy import select

from MLB.database.models import Game, Weather
from shared.contextual import ROOF_CLOSED, VENUE_COORDS
from shared.timeutil import utcnow

log = logging.getLogger(__name__)

SOURCE = "open_meteo"
BASE_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 30


def _fetch_forecast(lat: float, lon: float, dt_utc) -> dict | None:
    """Consulta Open-Meteo para un punto y hora especifica.

    Devuelve {temperature_c, humidity, wind_speed_kmh, wind_direction_deg,
    precipitation_mm} o None si falla.
    """
    date_str = dt_utc.strftime("%Y-%m-%d")
    hour = dt_utc.hour

    url = (
        f"{BASE_URL}?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"wind_direction_10m,precipitation"
        f"&start_date={date_str}&end_date={date_str}"
        f"&timezone=UTC"
    )

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Sports-Prediction-Center/0.2",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log.warning("Open-Meteo fallo para (%.2f, %.2f): %s", lat, lon, e)
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    # Buscar el indice de la hora mas cercana al inicio del partido
    target = f"{date_str}T{hour:02d}:00"
    idx = None
    for i, t in enumerate(times):
        if t == target:
            idx = i
            break
    if idx is None and times:
        idx = min(range(len(times)),
                  key=lambda i: abs(int(times[i].split("T")[1][:2]) - hour))
    if idx is None:
        return None

    def _safe(arr, i):
        if arr and i < len(arr):
            return arr[i]
        return None

    return {
        "temperature_c": _safe(hourly.get("temperature_2m"), idx),
        "humidity": _safe(hourly.get("relative_humidity_2m"), idx),
        "wind_speed_kmh": _safe(hourly.get("wind_speed_10m"), idx),
        "wind_direction_deg": _safe(hourly.get("wind_direction_10m"), idx),
        "precipitation_mm": _safe(hourly.get("precipitation"), idx),
    }


def fetch_for_upcoming(session, hours_ahead: int = 24) -> dict:
    """Descarga pronostico de clima para partidos MLB en las proximas horas.

    Guarda en la tabla `weather` existente. Devuelve un resumen.
    """
    now = utcnow()
    cutoff = now + timedelta(hours=hours_ahead)

    games = session.execute(
        select(Game).where(
            Game.start_utc >= now,
            Game.start_utc <= cutoff,
            Game.status.notin_(["Final", "Game Over", "Completed Early",
                                "Postponed"]),
        )
    ).scalars().all()

    fetched = 0
    skipped_roof = 0
    skipped_coords = 0
    errors = 0

    for game in games:
        venue = game.venue_name
        if not venue:
            skipped_coords += 1
            continue

        if venue in ROOF_CLOSED:
            skipped_roof += 1
            continue

        coords = VENUE_COORDS.get(venue)
        if not coords:
            skipped_coords += 1
            log.debug("Sin coordenadas para %s (game %d)", venue, game.id)
            continue

        lat, lon, _ = coords
        forecast = _fetch_forecast(lat, lon, game.start_utc)
        if not forecast:
            errors += 1
            continue

        session.add(Weather(
            game_id=game.id,
            temperature_c=forecast.get("temperature_c"),
            humidity=forecast.get("humidity"),
            wind_speed_kmh=forecast.get("wind_speed_kmh"),
            wind_direction_deg=forecast.get("wind_direction_deg"),
            precipitation_mm=forecast.get("precipitation_mm"),
            is_forecast=True,
            available_at=now,
            source=SOURCE,
        ))
        fetched += 1

        # Rate limit: Open-Meteo permite ~100 req/min en tier gratuito.
        # Con 15 partidos por dia esto no es problema, pero no esta de mas.
        time.sleep(0.3)

    session.flush()

    return {
        "games_checked": len(games),
        "forecasts_saved": fetched,
        "skipped_roof": skipped_roof,
        "skipped_no_coords": skipped_coords,
        "errors": errors,
    }
