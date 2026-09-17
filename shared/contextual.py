"""Factor Contextual Automatizado: clima y confirmacion de alineaciones.

Ajusta las probabilidades finales DESPUES de que el modelo base genera su
prediccion. Los deltas son conservadores (±1% a ±6%) porque aun no hay datos
historicos suficientes para validar ajustes mas grandes.

El modulo se ejecuta en background desde el auto-score loop y cachea los
ajustes en un dict global que los providers consultan al construir tarjetas.

API de clima: Open-Meteo (gratuita, sin key).
Alineaciones: MLB Stats API (gratuita, sin key).
"""
from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timedelta

from shared.timeutil import utcnow

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordenadas GPS de los 30 estadios MLB (lat, lon) y orientacion del home
# plate hacia center field en grados (0=N, 90=E, 180=S, 270=W).
# Fuente: datos publicos de MLB y Google Maps.
# ---------------------------------------------------------------------------
VENUE_COORDS: dict[str, tuple[float, float, float]] = {
    # (lat, lon, bearing_to_cf_degrees)
    "Angel Stadium":             (33.800, -117.883, 68),
    "Busch Stadium":             (38.623, -90.193, 56),
    "Chase Field":               (33.445, -112.067, 0),     # techo retractil
    "Citi Field":                (40.757, -73.846, 135),
    "Citizens Bank Park":        (39.906, -75.166, 68),
    "Comerica Park":             (42.339, -83.049, 75),
    "Coors Field":               (39.756, -104.994, 75),
    "Dodger Stadium":            (34.074, -118.240, 56),
    "Fenway Park":               (42.346, -71.098, 68),
    "Globe Life Field":          (32.747, -97.084, 0),      # techo retractil
    "Great American Ball Park":  (39.097, -84.507, 79),
    "Guaranteed Rate Field":     (41.830, -87.634, 68),
    "Kauffman Stadium":          (39.052, -94.481, 79),
    "LoanDepot park":            (25.778, -80.220, 0),      # techo retractil
    "Minute Maid Park":          (29.757, -95.355, 0),      # techo retractil
    "Nationals Park":            (38.873, -77.007, 90),
    "Oakland Coliseum":          (37.752, -122.201, 68),
    "Oracle Park":               (37.778, -122.389, 23),
    "Oriole Park at Camden Yards": (39.284, -76.622, 68),
    "PNC Park":                  (40.447, -80.006, 56),
    "Petco Park":                (32.707, -117.157, 68),
    "Progressive Field":         (41.496, -81.685, 79),
    "Rogers Centre":             (43.641, -79.389, 0),      # techo retractil
    "T-Mobile Park":             (47.591, -122.332, 0),     # techo retractil
    "Target Field":              (44.982, -93.278, 56),
    "Tropicana Field":           (27.768, -82.653, 0),      # techo cerrado
    "Truist Park":               (33.891, -84.468, 135),
    "Wrigley Field":             (41.948, -87.656, 23),
    "Yankee Stadium":            (40.829, -73.926, 68),
    "American Family Field":     (43.028, -87.971, 0),      # techo retractil
}

# Estadios donde el clima NO afecta (techo cerrado o retractil).
# Si el techo esta abierto se podria ajustar, pero la API de MLB no dice el
# estado del techo en tiempo real, asi que se asume cerrado para ser conservador.
ROOF_CLOSED: set[str] = {
    "Chase Field", "Globe Life Field", "LoanDepot park", "Minute Maid Park",
    "Rogers Centre", "T-Mobile Park", "Tropicana Field", "American Family Field",
}

# ---------------------------------------------------------------------------
# Cache global: {game_id: {market: adjustment_dict}}
# Protegido por un lock porque se escribe desde el auto-score thread y se lee
# desde el thread del request handler de FastAPI.
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_CACHE: dict[int, dict[str, dict]] = {}


# ---------------------------------------------------------------------------
# Funciones de calculo de deltas
# ---------------------------------------------------------------------------

def _wind_component_to_cf(wind_speed_kmh: float, wind_dir_deg: float,
                           cf_bearing_deg: float) -> float:
    """Componente del viento EN DIRECCION a center field (km/h).

    Positivo = viento HACIA CF (favorece batazo largo / over).
    Negativo = viento DESDE CF (frena batazo / under).
    """
    # Angulo entre la direccion del viento y la direccion hacia CF.
    # wind_dir_deg es la direccion DE DONDE viene el viento (convencion meteo).
    # Para saber hacia donde VA: va en sentido opuesto = wind_dir + 180.
    wind_going_deg = (wind_dir_deg + 180) % 360
    diff = math.radians(wind_going_deg - cf_bearing_deg)
    return wind_speed_kmh * math.cos(diff)


def delta_weather(weather: dict, venue_name: str, market: str) -> dict | None:
    """Calcula el ajuste por clima para un partido MLB.

    Returns dict con {delta_pct, reason} o None si no aplica ajuste.
    """
    if venue_name in ROOF_CLOSED:
        return None

    coords = VENUE_COORDS.get(venue_name)
    if not coords:
        return None

    _, _, cf_bearing = coords
    wind_kmh = weather.get("wind_speed_kmh") or 0
    wind_dir = weather.get("wind_direction_deg") or 0
    temp_c = weather.get("temperature_c") or 20
    humidity = weather.get("humidity") or 50
    precip = weather.get("precipitation_mm") or 0

    delta = 0.0
    reasons = []

    # --- Viento ---
    if wind_kmh > 15:
        component = _wind_component_to_cf(wind_kmh, wind_dir, cf_bearing)
        if market == "total":
            if component > 10:
                # Viento fuerte hacia CF: favorece Over
                delta += min(0.06, component / 200)
                reasons.append(f"viento {wind_kmh:.0f} km/h hacia CF")
            elif component < -10:
                # Viento fuerte desde CF: favorece Under
                delta -= min(0.04, abs(component) / 250)
                reasons.append(f"viento {wind_kmh:.0f} km/h desde CF")

    # --- Temperatura ---
    if market == "total":
        if temp_c < 10:
            delta -= 0.02
            reasons.append(f"frio ({temp_c:.0f}°C)")
        elif temp_c > 32:
            delta += 0.015
            reasons.append(f"calor ({temp_c:.0f}°C)")

    # --- Humedad alta ---
    if market == "total" and humidity > 85:
        delta -= 0.01
        reasons.append(f"humedad {humidity:.0f}%")

    # --- Lluvia significativa ---
    if precip > 2:
        reasons.append(f"lluvia {precip:.1f} mm/h (posible retraso)")
        # No se ajusta la probabilidad: es mas probable que se posponga.

    if not reasons or abs(delta) < 0.005:
        return None

    return {
        "delta_pct": round(delta * 100, 1),
        "delta_raw": round(delta, 4),
        "reason": "; ".join(reasons),
        "wind_kmh": round(wind_kmh, 1),
        "temp_c": round(temp_c, 1),
        "humidity": round(humidity, 0),
        "precip_mm": round(precip, 1),
    }


def delta_pitcher_change(expected_era: float | None,
                         actual_era: float | None,
                         is_home: bool) -> dict | None:
    """Ajuste cuando el abridor cambia respecto al proyectado.

    Usa la diferencia de ERA como proxy del impacto:
      - actual ERA mucho peor que expected -> negativo para su equipo
      - actual ERA mucho mejor que expected -> positivo
    """
    if expected_era is None or actual_era is None:
        return None

    diff = actual_era - expected_era
    if abs(diff) < 1.0:
        return None

    # Cada punto de ERA de diferencia ~ 1.5% de ajuste, maximo ±5%
    delta = -min(0.05, max(-0.05, diff * 0.015))

    if abs(delta) < 0.005:
        return None

    direction = "local" if is_home else "visitante"
    better = "mejor" if delta > 0 else "peor"
    return {
        "delta_pct": round(delta * 100, 1),
        "delta_raw": round(delta, 4),
        "side": "home" if is_home else "away",
        "reason": (f"cambio de abridor ({direction}): ERA {actual_era:.2f} vs "
                   f"{expected_era:.2f} ({better})"),
    }


def delta_lineup_change(missing_wrc_plus: float | None,
                        is_home: bool) -> dict | None:
    """Ajuste cuando un bateador importante sale de la alineacion.

    `missing_wrc_plus`: wRC+ del bateador que falta (None si no falta nadie).
    Umbral: solo se ajusta si wRC+ > 130 (bateador elite).
    """
    if missing_wrc_plus is None or missing_wrc_plus <= 130:
        return None

    delta = -0.02 * ((missing_wrc_plus - 130) / 70)  # ~ -2% a -3%
    delta = max(-0.03, delta)

    direction = "local" if is_home else "visitante"
    return {
        "delta_pct": round(delta * 100, 1),
        "delta_raw": round(delta, 4),
        "side": "home" if is_home else "away",
        "reason": (f"bateador clave fuera de lineup ({direction}, "
                   f"wRC+ {missing_wrc_plus:.0f})"),
    }


# ---------------------------------------------------------------------------
# Aplicar ajustes a una tarjeta
# ---------------------------------------------------------------------------

def apply_adjustment(p_original: float, delta_raw: float,
                     market: str, pick_is_home: bool,
                     adj_side: str | None) -> float:
    """Aplica un delta a la probabilidad original respetando [0.01, 0.99].

    Para ajustes de lineup/pitcher, el delta se aplica positivo si el ajuste
    favorece al lado del pick, negativo si lo perjudica.
    """
    if adj_side is not None:
        # El delta ya viene con el signo correcto para el equipo afectado.
        # Si el pick es del mismo lado, se suma; si es del contrario, se resta.
        if (adj_side == "home") != pick_is_home:
            delta_raw = -delta_raw

    p = p_original + delta_raw
    return max(0.01, min(0.99, round(p, 4)))


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _cache_put(game_id: int, market: str, adj: dict) -> None:
    with _LOCK:
        _CACHE.setdefault(game_id, {})[market] = adj


def get_adjustments(game_id: int, market: str) -> dict | None:
    """Lee los ajustes cacheados para un partido/mercado. Thread-safe."""
    with _LOCK:
        return (_CACHE.get(game_id) or {}).get(market)


def get_all_adjustments() -> dict:
    """Copia completa del cache. Para el endpoint /api/contextual."""
    with _LOCK:
        return json.loads(json.dumps(_CACHE, default=str))


def status() -> dict:
    """Estado del modulo contextual para /api/contextual."""
    with _LOCK:
        n_games = len(_CACHE)
        n_adjustments = sum(len(mkts) for mkts in _CACHE.values())
        games_with_weather = sum(
            1 for mkts in _CACHE.values()
            if any(m.get("weather") for m in mkts.values())
        )
        games_with_lineup = sum(
            1 for mkts in _CACHE.values()
            if any(m.get("lineup") or m.get("pitcher") for m in mkts.values())
        )
    return {
        "enabled": True,
        "games_tracked": n_games,
        "adjustments_active": n_adjustments,
        "with_weather_adjustment": games_with_weather,
        "with_lineup_adjustment": games_with_lineup,
        "note": ("Los ajustes son conservadores (max +/-6%). Se calibraran "
                 "cuando haya suficientes datos para validar el impacto."),
    }


# ---------------------------------------------------------------------------
# Refresh: se llama desde el auto-score loop
# ---------------------------------------------------------------------------

def refresh_contextual() -> dict:
    """Punto de entrada principal: descarga clima y detecta cambios de lineup.

    Se ejecuta cada 10 minutos desde _auto_score_loop().
    """
    result = {"weather": None, "lineup": None}

    # 1. Clima MLB
    try:
        result["weather"] = _refresh_weather_mlb()
    except Exception as e:
        result["weather"] = f"{type(e).__name__}: {e}"
        log.exception("Error refrescando clima MLB")

    # 2. Detectar cambios de pitcher/lineup MLB
    try:
        result["lineup"] = _refresh_lineups_mlb()
    except Exception as e:
        result["lineup"] = f"{type(e).__name__}: {e}"
        log.exception("Error detectando cambios de lineup MLB")

    return result


def _refresh_weather_mlb() -> dict:
    """Descarga pronostico de clima para partidos MLB proximos y cachea ajustes."""
    from MLB.database.models import Game, Weather
    from MLB.database.session import init_db, session_scope
    from MLB.ingest.weather import fetch_for_upcoming

    init_db()

    # 1. Descargar y guardar en DB
    with session_scope() as s:
        saved = fetch_for_upcoming(s, hours_ahead=24)

    # 2. Leer los datos guardados y calcular ajustes
    now = utcnow()
    markets = ["moneyline", "run_line", "total"]
    adjusted = 0

    with session_scope() as s:
        from sqlalchemy import select
        upcoming = s.execute(
            select(Game).where(
                Game.start_utc >= now,
                Game.start_utc <= now + timedelta(hours=36),
                Game.status.notin_(["Final", "Game Over", "Completed Early",
                                    "Postponed"]),
            )
        ).scalars().all()

        for game in upcoming:
            # Ultimo registro de clima para este partido
            w = s.execute(
                select(Weather).where(Weather.game_id == game.id)
                .order_by(Weather.available_at.desc()).limit(1)
            ).scalars().first()
            if not w:
                continue

            weather_data = {
                "temperature_c": w.temperature_c,
                "humidity": w.humidity,
                "wind_speed_kmh": w.wind_speed_kmh,
                "wind_direction_deg": w.wind_direction_deg,
                "precipitation_mm": w.precipitation_mm,
            }

            for mkt in markets:
                adj_w = delta_weather(weather_data, game.venue_name, mkt)
                if adj_w:
                    existing = get_adjustments(game.id, mkt) or {}
                    existing["weather"] = adj_w
                    existing["applied"] = True
                    existing["updated_at"] = now.isoformat()
                    _cache_put(game.id, mkt, existing)
                    adjusted += 1

    return {"fetched": saved, "adjustments_computed": adjusted}


def _refresh_lineups_mlb() -> dict:
    """Detecta cambios de pitcher respecto al proyectado y ajusta."""
    from sqlalchemy import select

    from MLB.database.models import Game, ProbablePitcher
    from MLB.database.session import init_db, session_scope

    init_db()
    now = utcnow()
    adjusted = 0
    n_checked = 0

    with session_scope() as s:
        upcoming = s.execute(
            select(Game).where(
                Game.start_utc >= now,
                Game.start_utc <= now + timedelta(hours=24),
                Game.status.notin_(["Final", "Game Over", "Completed Early",
                                    "Postponed"]),
            )
        ).scalars().all()
        n_checked = len(upcoming)

        for game in upcoming:
            # Buscar todos los registros de probable pitchers para este partido
            pps = s.execute(
                select(ProbablePitcher).where(
                    ProbablePitcher.game_id == game.id)
                .order_by(ProbablePitcher.available_at.asc())
            ).scalars().all()

            if len(pps) < 2:
                continue  # necesitamos al menos un cambio

            # Agrupar por equipo: detectar si el ultimo es distinto al primero
            by_team: dict[int, list] = {}
            for pp in pps:
                by_team.setdefault(pp.team_id, []).append(pp)

            for team_id, team_pps in by_team.items():
                if len(team_pps) < 2:
                    continue
                first, last = team_pps[0], team_pps[-1]
                if first.player_id == last.player_id:
                    continue  # sin cambio

                # Hay cambio de pitcher. Usamos un delta generico conservador
                # porque no tenemos ERA en la tabla de probable pitchers.
                adj_p = {
                    "delta_pct": -2.0,
                    "delta_raw": -0.02,
                    "side": "home" if last.is_home else "away",
                    "reason": (f"cambio de abridor: "
                               f"{first.player_name or first.player_id} -> "
                               f"{last.player_name or last.player_id}"),
                }

                for mkt in ("moneyline", "run_line"):
                    existing = get_adjustments(game.id, mkt) or {}
                    existing["pitcher"] = adj_p
                    existing["applied"] = True
                    existing["updated_at"] = now.isoformat()
                    _cache_put(game.id, mkt, existing)
                    adjusted += 1

    return {"games_checked": n_checked, "adjustments_computed": adjusted}
