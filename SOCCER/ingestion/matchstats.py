"""Estadisticas de partido: CORNERS, tiros, faltas y tarjetas.

Fuente: `datasets/football-datasets` en GitHub, que replica las columnas completas
de football-data.co.uk (HC, AC, HS, AS, HST, AST, HF, AF, HY, AY). Se eligio
despues de comprobar que football-data.co.uk responde 503 y que el espejo
`footballcsv` esta reducido a marcadores.

Cobertura verificada: 5 ligas europeas, 2012-13 a 2025-26, corners al 100%.
NO incluye Liga MX ni Saudi Pro League. Esas dos quedan sin mercado de corners y
se marcan INSUFFICIENT DATA para ese mercado; no se les inventa una estimacion.

Este modulo ADEMAS da de alta partidos que falten, asi que sirve tambien como
fuente de resultados: llega hasta 2025-26, mas lejos que el espejo footballcsv.
"""
from __future__ import annotations

import csv
import io
import urllib.request
from datetime import date, datetime

from sqlalchemy import select

from SOCCER.db import SoccerMatch, SoccerSourceLog, init_db, session_scope
from SOCCER.ingestion.historical import _mid
from shared.timeutil import utcnow

BASE = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/{lg}/season-{t}.csv"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124"}

# liga interna -> carpeta del repositorio
REPO = {"EPL": "premier-league", "LALIGA": "la-liga", "SERIEA": "serie-a",
        "BUNDES": "bundesliga", "LIGUE1": "ligue-1"}
SIN_CORNERS = ("LIGAMX", "SAUDI", "UCL")   # sin fuente verificada: se dira asi

# Fuente secundaria SOLO para Liga MX. football-data.co.uk si publica sus corners
# (HC/AC), pero lleva dias devolviendo 503. Se intenta en cada `train`: el dia que
# vuelva, los corners de Liga MX entran solos y el gating la activa sin tocar nada.
# Si falla, se registra el error y el resto del proceso sigue igual.
FD_MEX = "https://www.football-data.co.uk/new/MEX.csv"
FD_COLS = {"home_corners": "HC", "away_corners": "AC", "home_shots": "HS", "away_shots": "AS",
           "home_sot": "HST", "away_sot": "AST", "home_fouls": "HF", "away_fouls": "AF",
           "home_yellow": "HY", "away_yellow": "AY"}

ENTEROS = {"home_corners": "HC", "away_corners": "AC", "home_shots": "HS", "away_shots": "AS",
           "home_sot": "HST", "away_sot": "AST", "home_fouls": "HF", "away_fouls": "AF",
           "home_yellow": "HY", "away_yellow": "AY"}


def temporadas(desde: int = 2012, hasta: int | None = None) -> list[str]:
    fin = hasta if hasta is not None else utcnow().year
    return ["%02d%02d" % (y % 100, (y + 1) % 100) for y in range(desde, fin + 1)]


def _get(url, timeout=40):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore"), None
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _fecha(txt: str) -> date | None:
    t = str(txt or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


def _ent(v):
    """Entero o None. Una celda vacia NO es cero."""
    s = str(v or "").strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _log(s, scope, url, status, n=0, err=None):
    s.add(SoccerSourceLog(source="football-datasets", scope=scope, url=url, status=status,
                          records=n, error=err, retrieved_at=utcnow()))


def ingest(codes: list[str] | None = None, desde: int = 2012, progress=None) -> dict:
    """Rellena estadisticas en partidos existentes y da de alta los que falten."""
    init_db()
    ahora = utcnow()
    resumen: dict[str, dict] = {}
    with session_scope() as s:
        por_id = {m.match_id: m for m in s.execute(select(SoccerMatch)).scalars()}
        for code, carpeta in REPO.items():
            if codes and code not in [c.upper() for c in codes]:
                continue
            r = {"filas": 0, "actualizados": 0, "nuevos": 0, "sin_corners": 0, "temporadas": 0}
            for t in temporadas(desde):
                url = BASE.format(lg=carpeta, t=t)
                txt, err = _get(url)
                if txt is None:
                    _log(s, f"{code} {t}", url, "error", 0, err)
                    continue
                filas = list(csv.DictReader(io.StringIO(txt)))
                _log(s, f"{code} {t}", url, "ok", len(filas))
                if not filas:
                    continue
                r["temporadas"] += 1
                temp = f"20{t[:2]}-{t[2:]}"
                for row in filas:
                    f = _fecha(row.get("Date"))
                    home, away = (row.get("HomeTeam") or "").strip(), (row.get("AwayTeam") or "").strip()
                    hg, ag = _ent(row.get("FTHG")), _ent(row.get("FTAG"))
                    if f is None or not home or not away or hg is None or ag is None:
                        continue
                    r["filas"] += 1
                    stats = {k: _ent(row.get(col)) for k, col in ENTEROS.items()}
                    if stats["home_corners"] is None or stats["away_corners"] is None:
                        r["sin_corners"] += 1
                    mid = _mid(code, f, home, away)
                    m = por_id.get(mid)
                    if m is None:
                        m = SoccerMatch(match_id=mid, league_code=code, season=temp,
                                        match_date=f, kickoff_utc=None,
                                        home_team=home, away_team=away,
                                        home_goals=hg, away_goals=ag,
                                        ht_home_goals=_ent(row.get("HTHG")),
                                        ht_away_goals=_ent(row.get("HTAG")),
                                        source="football-datasets", source_url=url,
                                        fetched_at=ahora)
                        s.add(m)
                        por_id[mid] = m
                        r["nuevos"] += 1
                    else:
                        r["actualizados"] += 1
                    for k, v in stats.items():
                        setattr(m, k, v)
                    m.stats_source = "football-datasets"
                if progress:
                    progress(f"{code} {temp}: {len(filas)} filas")
                s.flush()
            resumen[code] = r
        for code in SIN_CORNERS:
            resumen[code] = {"nota": "sin fuente verificada de corners: mercado no disponible"}
    return resumen


def liga_mx_corners(progress=None) -> dict:
    """Intento oportunista de traer los corners de Liga MX. Puede fallar y no pasa nada."""
    init_db()
    ahora = utcnow()
    txt, err = _get(FD_MEX)
    with session_scope() as s:
        if txt is None:
            _log(s, "LIGAMX corners", FD_MEX, "error", 0, err)
            return {"ok": False, "motivo": err, "actualizados": 0,
                    "nota": "football-data.co.uk no responde; se reintenta en el proximo train"}
        filas = list(csv.DictReader(io.StringIO(txt)))
        _log(s, "LIGAMX corners", FD_MEX, "ok", len(filas))
        por_clave = {}
        for m in s.execute(select(SoccerMatch).where(
                SoccerMatch.league_code == "LIGAMX")).scalars():
            por_clave[m.match_id] = m
        n = 0
        for row in filas:
            f = _fecha(row.get("Date"))
            home, away = (row.get("Home") or row.get("HomeTeam") or "").strip(), \
                         (row.get("Away") or row.get("AwayTeam") or "").strip()
            if f is None or not home or not away:
                continue
            m = por_clave.get(_mid("LIGAMX", f, home, away))
            if m is None or m.home_corners is not None:
                continue
            vals = {k: _ent(row.get(col)) for k, col in FD_COLS.items()}
            if vals["home_corners"] is None:
                continue
            for k, v in vals.items():
                setattr(m, k, v)
            m.stats_source = "football-data.co.uk"
            n += 1
        s.flush()
    if progress:
        progress(f"Liga MX: {n} partidos con corners incorporados")
    return {"ok": True, "actualizados": n, "filas_fuente": len(filas)}


def cobertura() -> dict:
    """Cuantos partidos tienen corners, por liga. Alimenta Data Health."""
    init_db()
    out = {}
    with session_scope() as s:
        for (code,) in s.execute(select(SoccerMatch.league_code).distinct()).all():
            ms = s.execute(select(SoccerMatch).where(SoccerMatch.league_code == code)).scalars().all()
            con = [m for m in ms if m.home_corners is not None and m.away_corners is not None]
            fechas = sorted(m.match_date for m in con) if con else []
            out[code] = {"partidos": len(ms), "con_corners": len(con),
                         "cobertura": round(len(con) / len(ms), 4) if ms else 0.0,
                         "desde": fechas[0].isoformat() if fechas else None,
                         "hasta": fechas[-1].isoformat() if fechas else None}
    return out
