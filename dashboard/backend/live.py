"""Marcador en vivo de MLB. SOLO LECTURA: consulta la MLB Stats API y no escribe
en la base. Se cachea 45 segundos para no golpear la API con cada recarga.

Estados que devuelve la API (abstractGameState): Preview | Live | Final.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, timedelta

_CACHE: dict = {"at": 0.0, "data": None, "key": None}
TTL = 45
UA = {"User-Agent": "Sports-Prediction-Center/1.0"}

ESTADO = {"Preview": "programado", "Live": "en_vivo", "Final": "finalizado"}


def _fetch(d0: str, d1: str) -> dict:
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={d0}"
           f"&endDate={d1}&hydrate=linescore,team")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def live(days_back: int = 1, days_ahead: int = 1) -> dict:
    hoy = date.today()
    key = (str(hoy - timedelta(days=days_back)), str(hoy + timedelta(days=days_ahead)))
    now = time.time()
    if _CACHE["data"] is not None and _CACHE["key"] == key and now - _CACHE["at"] < TTL:
        return _CACHE["data"]
    try:
        raw = _fetch(*key)
    except Exception as e:                                   # noqa: BLE001
        stale = _CACHE["data"] or {"games": {}, "error": None}
        stale["error"] = f"no se pudo consultar el marcador en vivo: {type(e).__name__}"
        return stale
    games = {}
    for day in raw.get("dates", []):
        for g in day.get("games", []):
            st = g.get("status") or {}
            ls = g.get("linescore") or {}
            abstract = st.get("abstractGameState")
            estado = ESTADO.get(abstract, "programado")
            detalle = st.get("detailedState") or ""
            if "Postponed" in detalle or "Suspended" in detalle or "Cancel" in detalle:
                estado = "pospuesto"
            entrada = None
            if estado == "en_vivo":
                mitad = (ls.get("inningHalf") or "").lower()
                mitad_es = {"top": "alta", "bottom": "baja", "middle": "cambio",
                            "end": "fin"}.get(mitad, mitad)
                n = ls.get("currentInning")
                entrada = f"{mitad_es} {n}ª" if n else None
            games[g["gamePk"]] = {
                "estado": estado, "detalle": detalle,
                "home": g["teams"]["home"].get("score"),
                "away": g["teams"]["away"].get("score"),
                "entrada": entrada,
                "outs": ls.get("outs") if estado == "en_vivo" else None,
                "entradas_jugadas": ls.get("currentInning") if estado == "finalizado" else None,
                "home_abbr": (g["teams"]["home"].get("team") or {}).get("abbreviation"),
                "away_abbr": (g["teams"]["away"].get("team") or {}).get("abbreviation"),
            }
    data = {"games": games, "actualizado": time.strftime("%H:%M:%S"), "error": None,
            "rango": list(key)}
    _CACHE.update({"at": now, "data": data, "key": key})
    return data
