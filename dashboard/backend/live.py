"""Marcador en vivo unificado para todos los deportes. SOLO LECTURA.

- MLB: usa statsapi.mlb.com (cacheado 45s)
- NFL, NBA, SOCCER: usan ESPN API oculta (cacheado 60s)
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, timedelta
from sqlalchemy import select

from SOCCER.db import SoccerFixture, session_scope as soccer_session
from SOCCER.ingestion.teams import Resolver
from SOCCER.ingestion.cerrar import _clave, ESPN_LEAGUES

# -----------------------------------------------------------------------------
# Caches independientes por deporte para no bloquear si uno falla
# -----------------------------------------------------------------------------
_CACHES = {
    "MLB": {"at": 0.0, "data": None, "key": None},
    "ESPN": {"at": 0.0, "data": None, "key": None}
}
TTL = 45
UA = {"User-Agent": "Mozilla/5.0"}

ESTADO = {"Preview": "programado", "Live": "en_vivo", "Final": "finalizado"}
ESTADO_ESPN = {"pre": "programado", "in": "en_vivo", "post": "finalizado"}


def _fetch_mlb(d0: str, d1: str) -> dict:
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={d0}"
           f"&endDate={d1}&hydrate=linescore,team")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _fetch_espn(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def live(days_back: int = 1, days_ahead: int = 1) -> dict:
    hoy = date.today()
    key = (str(hoy - timedelta(days=days_back)), str(hoy + timedelta(days=days_ahead)))
    now = time.time()
    
    games = {}
    errores = []

    # =========================================================================
    # 1. MLB (statsapi)
    # =========================================================================
    if _CACHES["MLB"]["data"] is not None and _CACHES["MLB"]["key"] == key and now - _CACHES["MLB"]["at"] < TTL:
        games.update(_CACHES["MLB"]["data"])
    else:
        try:
            raw = _fetch_mlb(*key)
            mlb_games = {}
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
                    mlb_games[g["gamePk"]] = {
                        "estado": estado, "detalle": detalle,
                        "home": g["teams"]["home"].get("score"),
                        "away": g["teams"]["away"].get("score"),
                        "entrada": entrada,
                        "outs": ls.get("outs") if estado == "en_vivo" else None,
                        "entradas_jugadas": ls.get("currentInning") if estado == "finalizado" else None,
                        "home_abbr": (g["teams"]["home"].get("team") or {}).get("abbreviation"),
                        "away_abbr": (g["teams"]["away"].get("team") or {}).get("abbreviation"),
                    }
            _CACHES["MLB"].update({"at": now, "data": mlb_games, "key": key})
            games.update(mlb_games)
        except Exception as e:
            errores.append(f"MLB: {type(e).__name__}")
            if _CACHES["MLB"]["data"]:
                games.update(_CACHES["MLB"]["data"])

    # =========================================================================
    # 2. Otros Deportes (ESPN)
    # =========================================================================
    if _CACHES["ESPN"]["data"] is not None and _CACHES["ESPN"]["key"] == key and now - _CACHES["ESPN"]["at"] < 60:
        games.update(_CACHES["ESPN"]["data"])
    else:
        espn_games = {}
        
        # -- NFL --
        try:
            TEAM_MAP = {"WSH": "WAS", "LAR": "LA"}
            raw_nfl = _fetch_espn("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard")
            season = raw_nfl.get("season", {}).get("year")
            week = raw_nfl.get("week", {}).get("number")
            if season and week:
                for ev in raw_nfl.get("events", []):
                    comps = ev.get("competitions", [])[0].get("competitors", [])
                    h = [c for c in comps if c.get("homeAway") == "home"][0]
                    a = [c for c in comps if c.get("homeAway") == "away"][0]
                    h_abbr = TEAM_MAP.get(h["team"].get("abbreviation", ""), h["team"].get("abbreviation", ""))
                    a_abbr = TEAM_MAP.get(a["team"].get("abbreviation", ""), a["team"].get("abbreviation", ""))
                    gid = f"{season}_{week:02d}_{a_abbr}_{h_abbr}"
                    st = ev.get("status", {})
                    estado = ESTADO_ESPN.get(st.get("type", {}).get("state"), "programado")
                    entrada = st.get("displayClock")
                    if st.get("period") and estado == "en_vivo":
                        entrada = f"Q{st['period']} {entrada}"
                    espn_games[gid] = {
                        "estado": estado, "detalle": "",
                        "home": int(h.get("score", 0)), "away": int(a.get("score", 0)),
                        "entrada": entrada if estado == "en_vivo" else None
                    }
        except Exception as e:
            errores.append(f"NFL: {type(e).__name__}")

        # -- SOCCER --
        try:
            # Primero buscamos los event_id de los partidos de SOCCER activos en la BD
            soccer_map = {}
            res = Resolver()
            with soccer_session() as s:
                fxs = s.execute(select(SoccerFixture).where(
                    SoccerFixture.commence_utc >= (hoy - timedelta(days=2))
                )).scalars().all()
                for f in fxs:
                    k = _clave(res, f.league_code, f.home_team, f.away_team)
                    if k:
                        soccer_map[k] = f.event_id

            # Consultamos las ligas en ESPN
            dt_str = hoy.strftime("%Y%m%d")
            for league_code, espn_code in ESPN_LEAGUES.items():
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard?dates={dt_str}"
                raw_soc = _fetch_espn(url)
                for ev in raw_soc.get("events", []):
                    comps = ev.get("competitions", [])[0].get("competitors", [])
                    h = [c for c in comps if c.get("homeAway") == "home"][0]
                    a = [c for c in comps if c.get("homeAway") == "away"][0]
                    h_name = h["team"].get("name", "")
                    a_name = a["team"].get("name", "")
                    
                    k = _clave(res, league_code, h_name, a_name)
                    if not k:
                        # Fallback a nombre parcial
                        h_res, _ = res.resolver(league_code, h_name)
                        a_res, _ = res.resolver(league_code, a_name)
                        k = (league_code, h_res or h_name.lower(), a_res or a_name.lower())
                    
                    # Fuzzy match por si no dio exacto
                    matched_id = None
                    if k in soccer_map:
                        matched_id = soccer_map[k]
                    else:
                        for sm_k, sm_v in soccer_map.items():
                            if sm_k[0] == league_code:
                                if (sm_k[1] in k[1] or k[1] in sm_k[1]) and (sm_k[2] in k[2] or k[2] in sm_k[2]):
                                    matched_id = sm_v
                                    break
                    
                    if matched_id:
                        st = ev.get("status", {})
                        estado = ESTADO_ESPN.get(st.get("type", {}).get("state"), "programado")
                        entrada = st.get("displayClock")
                        if st.get("period") and estado == "en_vivo":
                            entrada = f"{st['period']}T {entrada}"
                        espn_games[matched_id] = {
                            "estado": estado, "detalle": "",
                            "home": int(h.get("score", 0)), "away": int(a.get("score", 0)),
                            "entrada": entrada if estado == "en_vivo" else None
                        }
        except Exception as e:
            import traceback
            errores.append(f"SOCCER: {traceback.format_exc()}")
            
        _CACHES["ESPN"].update({"at": now, "data": espn_games, "key": key})
        games.update(espn_games)

    data = {"games": games, "actualizado": time.strftime("%H:%M:%S"), "error": ", ".join(errores) if errores else None,
            "rango": list(key)}
    return data
