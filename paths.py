"""Rutas del proyecto y guardas de aislamiento.

Regla dura: cada deporte solo puede escribir dentro de su propia carpeta.
`assert_writable(path, sport)` levanta una excepcion si un proceso de un deporte
intenta escribir en el arbol de otro. Los tests de aislamiento se apoyan en esto.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

CONFIG_DIR = ROOT / "config"
LOG_DIR = ROOT / "logs"
SHARED_DIR = ROOT / "shared"
DASHBOARD_DIR = ROOT / "dashboard"

SPORTS = ("NFL", "MLB", "NBA", "TENIS", "SOCCER")

# --- MLB (motor propio, dentro del proyecto) --------------------------------
MLB_DIR = ROOT / "MLB"
MLB_DB_DIR = MLB_DIR / "database"
MLB_DATA_DIR = MLB_DIR / "data"
MLB_RAW_DIR = MLB_DATA_DIR / "raw"
MLB_PROCESSED_DIR = MLB_DATA_DIR / "processed"
MLB_SNAPSHOT_DIR = MLB_DATA_DIR / "snapshots"
MLB_MODELS_DIR = MLB_DIR / "models"
MLB_BACKTEST_DIR = MLB_DIR / "backtests"
MLB_PREDICTIONS_DIR = MLB_DIR / "predictions"

# --- NFL (sistema existente; se ENVUELVE, no se mueve) ----------------------
# NFL_HOME apunta a la instalacion real de nfl-prediction-app. El centro solo la LEE.
NFL_HOME = Path(os.getenv("NFL_HOME", str(ROOT.parent / "nfl-prediction-app"))).expanduser()

for _d in (LOG_DIR, CONFIG_DIR, MLB_DB_DIR, MLB_RAW_DIR, MLB_PROCESSED_DIR,
           MLB_SNAPSHOT_DIR, MLB_MODELS_DIR, MLB_BACKTEST_DIR, MLB_PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MLB_DATABASE_URL = os.getenv("MLB_DATABASE_URL", f"sqlite:///{MLB_DB_DIR / 'mlb.sqlite3'}")


class IsolationError(RuntimeError):
    """Un deporte intento tocar el arbol de otro deporte."""


# Arbol NFL PROPIO del centro (mercados total/spread, base y modelos propios).
# NFL_HOME (nfl-prediction-app) es SOLO LECTURA: nadie escribe ahi.
NFL_DIR = ROOT / "NFL"
NFL_MARKETS_DB_DIR = NFL_DIR / "database"
NFL_MARKETS_MODELS_DIR = NFL_DIR / "models"


PARLAY_DIR = ROOT / "PARLAY"
PARLAY_DB_DIR = PARLAY_DIR / "database"
# CHAT no es un deporte: como PARLAY, solo LEE a los demas a traves de su API
# interna. No tiene base de datos ni modelos propios a proposito, para que no
# pueda convertirse en una via lateral de escritura entre deportes.
CHAT_DIR = ROOT / "CHAT"
TENIS_DIR = ROOT / "TENIS"
TENIS_DB_DIR = TENIS_DIR / "database"
TENIS_MODELS_DIR = TENIS_DIR / "models"
TENIS_OUT_DIR = TENIS_DIR / "markets" / "out"
NBA_DIR = ROOT / "NBA"
NBA_DB_DIR = NBA_DIR / "database"
NBA_MODELS_DIR = NBA_DIR / "models"
NBA_OUT_DIR = NBA_DIR / "markets" / "out"

# --- SOCCER (modulo propio y aislado) ---------------------------------------
SOCCER_DIR = ROOT / "SOCCER"
SOCCER_DB_DIR = SOCCER_DIR / "database"
SOCCER_DATA_DIR = SOCCER_DIR / "data"
SOCCER_MODELS_DIR = SOCCER_DIR / "models"
SOCCER_FEATURES_DIR = SOCCER_DIR / "features"
SOCCER_REPORTS_DIR = SOCCER_DIR / "reports"
SOCCER_OUT_DIR = SOCCER_DIR / "evaluation" / "out"
SOCCER_DATABASE_URL = os.getenv(
    "SOCCER_DATABASE_URL", f"sqlite:///{SOCCER_DB_DIR / 'soccer_markets.sqlite3'}")


def sport_root(sport: str) -> Path:
    s = sport.upper()
    if s == "MLB":
        return MLB_DIR
    if s == "NFL":
        return NFL_DIR
    if s == "NBA":
        return NBA_DIR
    if s in ("TENIS", "TENNIS"):
        return TENIS_DIR
    if s in ("SOCCER", "FUTBOL"):
        return SOCCER_DIR
    if s == "PARLAY":                      # no es un deporte: solo lee a los demas
        return PARLAY_DIR
    if s == "CHAT":                        # tampoco: solo lee, via CHAT/api.py
        return CHAT_DIR
    raise ValueError(f"deporte desconocido: {sport}")


def assert_writable(path, sport: str) -> Path:
    """Permite escribir solo dentro de la carpeta del deporte (o en logs/)."""
    p = Path(path).resolve()
    try:                       # el motor NFL original nunca se escribe, desde ningun deporte
        p.relative_to(NFL_HOME.resolve())
        raise IsolationError(f"intento de escribir en el motor NFL original (solo lectura): {p}")
    except ValueError:
        pass
    allowed = [sport_root(sport).resolve(), LOG_DIR.resolve()]
    for base in allowed:
        try:
            p.relative_to(base)
            return p
        except ValueError:
            continue
    others = [sport_root(s).resolve() for s in SPORTS if s.upper() != sport.upper()]
    for o in others:
        try:
            p.relative_to(o)
            raise IsolationError(
                f"{sport} intento escribir en el arbol de otro deporte: {p}")
        except ValueError:
            continue
    raise IsolationError(f"{sport} intento escribir fuera de su arbol: {p}")
