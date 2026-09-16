"""Que queda sin calificar, deporte por deporte, y por que.

Existe porque el dashboard mostraba ceros silenciosos: el auto-scoring decia
`calificadas: 0` en todos los deportes y no habia forma de saber si eso era
"no hay nada que calificar" o "algo esta roto". Son cosas muy distintas.

OJO con las versiones superseded. Una prediccion reemplazada por otra mas nueva
NO debe calificarse, y contarla como pendiente da un falso positivo enorme: en
MLB salian 114 pendientes cuando las de verdad eran 0 (104 eran superseded y el
resto, partidos aun en juego).

Solo lectura: todas las bases se abren con mode=ro.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from shared.paths import (MLB_DB_DIR, NBA_DB_DIR, NFL_MARKETS_DB_DIR,
                          SOCCER_DB_DIR, TENIS_DB_DIR)

# (nombre, ruta, columna de fecha preferida)
BASES = [
    ("SOCCER", SOCCER_DB_DIR / "soccer_markets.sqlite3", "kickoff_utc"),
    ("MLB", MLB_DB_DIR / "mlb.sqlite3", "game_start_utc"),
    ("NFL", NFL_MARKETS_DB_DIR / "nfl_markets.sqlite3", "kickoff_utc"),
    ("NBA", NBA_DB_DIR / "nba_markets.sqlite3", "tipoff_utc"),
    ("TENIS", TENIS_DB_DIR / "tenis_markets.sqlite3", "start_utc"),
]
FECHAS = ("kickoff_utc", "game_start_utc", "start_utc", "tipoff_utc", "commence_utc")


def _cols(c, t):
    return {r[1] for r in c.execute(f"PRAGMA table_info({t})")}


def _vigente(cp: set[str]) -> str:
    """Solo cuenta la version VIGENTE de cada prediccion.

    Cada motor marca lo obsoleto a su manera: unos con `record_status`, otros
    con `status='superseded'`. Sin este filtro, las versiones viejas inflan los
    pendientes y parece que el sistema no califica cuando si lo hace.
    """
    if "record_status" in cp:
        return " AND record_status='active'"
    if "status" in cp:
        return " AND status != 'superseded'"
    return ""


def estado_de(nombre: str, ruta, col_pref: str) -> dict:
    if not ruta.exists():
        return {"deporte": nombre, "error": f"no existe {ruta.name}"}
    try:
        c = sqlite3.connect(f"file:{ruta}?mode=ro", uri=True)
        tablas = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        pred = ("predictions" if "predictions" in tablas
                else next((t for t in sorted(tablas) if "prediction" in t), None))
        if pred is None:
            return {"deporte": nombre, "error": "sin tabla de predicciones"}
        cp = _cols(c, pred)
        ts = col_pref if col_pref in cp else next((x for x in FECHAS if x in cp), None)
        if ts is None:
            return {"deporte": nombre, "error": "sin columna de fecha"}
        ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        viv = _vigente(cp)
        jugadas = c.execute(
            f"SELECT COUNT(*) FROM {pred} WHERE {ts} < ?{viv}", (ahora,)).fetchone()[0]
        pend = c.execute(
            f"SELECT COUNT(*) FROM {pred} WHERE {ts} < ?{viv} "
            f"AND (result IS NULL OR result='')", (ahora,)).fetchone()[0]
        return {"deporte": nombre, "ya_jugadas": jugadas,
                "calificadas": jugadas - pend, "pendientes": pend,
                "al_dia": pend == 0}
    except Exception as e:                        # un fallo se reporta, no se oculta
        return {"deporte": nombre, "error": f"{type(e).__name__}: {e}"}


def resumen() -> dict:
    """Estado de la calificacion de los 5 deportes, listo para la API."""
    partes = [estado_de(n, r, c) for n, r, c in BASES]
    pendientes = sum(p.get("pendientes", 0) for p in partes)
    return {
        "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pendientes_totales": pendientes,
        "por_deporte": partes,
        "nota": ("Solo cuenta la version vigente de cada prediccion: una "
                 "reemplazada por otra mas nueva no se califica."),
    }


if __name__ == "__main__":
    r = resumen()
    print(f"corte: {r['generado_en']}   pendientes totales: {r['pendientes_totales']}\n")
    for p in r["por_deporte"]:
        if "error" in p:
            print(f"  {p['deporte']:8} | ERROR: {p['error']}")
        else:
            marca = "ok " if p["al_dia"] else "!! "
            print(f"  {marca}{p['deporte']:8} | ya jugadas={p['ya_jugadas']:5} "
                  f"calificadas={p['calificadas']:5} pendientes={p['pendientes']:5}")
