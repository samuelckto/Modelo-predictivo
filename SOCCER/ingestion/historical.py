"""Historico real de partidos jugados. Solo para entrenar, validar y calibrar.

Fuente: espejo `footballcsv/cache.footballdata` en GitHub, formato uniforme
`Date, Team 1, FT, HT, Team 2`. Se eligio despues de comprobar en vivo que
football-data.co.uk devuelve 503, y que FBref y Understat bloquean (403). El
espejo cubre las 6 ligas con historico, Liga MX incluida.

Lo que NO trae y por tanto no existe en v1: xG, tiros, corners, tarjetas y
cuotas historicas. No se inventan.

Tampoco trae hora de saque. `kickoff_utc` queda en NULL a proposito, y el corte
anti-leakage usa 00:00 del dia del partido, que es la opcion conservadora.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
import urllib.request
from datetime import date, datetime

from sqlalchemy import select

from SOCCER.db import SoccerLeague, SoccerMatch, SoccerSourceLog, SoccerTeam, init_db, session_scope
from SOCCER.leagues import LIGAS, MIN_PARTIDOS_ENTRENAR, liga
from shared.timeutil import utcnow

BASE = "https://raw.githubusercontent.com/footballcsv/cache.footballdata/master/{temp}/{f}.csv"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124",
     "Accept": "text/plain"}
MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def temporadas(desde: int = 2012, hasta: int | None = None) -> list[str]:
    fin = hasta if hasta is not None else utcnow().year
    return ["%d-%02d" % (y, (y + 1) % 100) for y in range(desde, fin + 1)]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _fecha(txt: str) -> date | None:
    """'Fri Jun 30 2023' -> date. Devuelve None si no se puede leer (no se adivina)."""
    p = str(txt or "").replace(",", " ").split()
    for i, t in enumerate(p):
        if t[:3] in MESES and i + 2 < len(p) + 1:
            try:
                return date(int(p[i + 2]), MESES[t[:3]], int(p[i + 1]))
            except (ValueError, IndexError):
                return None
    return None


def _marcador(txt: str) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(txt or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _mid(code: str, f: date, home: str, away: str) -> str:
    crudo = f"{code}|{f.isoformat()}|{_norm(home)}|{_norm(away)}"
    return hashlib.sha1(crudo.encode()).hexdigest()[:20]


def _get(url: str, timeout: int = 40) -> tuple[str | None, str | None]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore"), None
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _fila(linea: str) -> list[str]:
    """CSV simple del espejo (sin comillas con comas dentro)."""
    return [c.strip() for c in linea.split(",")]


def _parse(txt: str) -> list[dict]:
    lineas = [x for x in txt.splitlines() if x.strip()]
    if len(lineas) < 2:
        return []
    cab = [c.strip().lower() for c in _fila(lineas[0])]
    try:
        i_fecha, i_local, i_ft, i_vis = (cab.index("date"), cab.index("team 1"),
                                         cab.index("ft"), cab.index("team 2"))
    except ValueError:
        return []
    i_ht = cab.index("ht") if "ht" in cab else None
    i_etapa = cab.index("round") if "round" in cab else None
    out = []
    for ln in lineas[1:]:
        c = _fila(ln)
        if len(c) <= max(i_fecha, i_local, i_ft, i_vis):
            continue
        f, ft = _fecha(c[i_fecha]), _marcador(c[i_ft])
        if f is None or ft is None:                 # partido sin jugar o fila rota: se ignora
            continue
        ht = _marcador(c[i_ht]) if i_ht is not None and i_ht < len(c) else None
        out.append({"fecha": f, "home": c[i_local].strip(), "away": c[i_vis].strip(),
                    "hg": ft[0], "ag": ft[1],
                    "hthg": ht[0] if ht else None, "htag": ht[1] if ht else None,
                    "etapa": c[i_etapa].strip() if i_etapa is not None and i_etapa < len(c) else None})
    return out


def _log(s, source, scope, url, status, n=0, err=None):
    s.add(SoccerSourceLog(source=source, scope=scope, url=url, status=status, records=n,
                          error=err, retrieved_at=utcnow()))


def ingest(codes: list[str] | None = None, desde: int = 2012, progress=None) -> dict:
    """Descarga e inserta el historico. Idempotente: no duplica partidos."""
    init_db()
    ahora = utcnow()
    resumen: dict[str, dict] = {}
    with session_scope() as s:
        existentes = set(s.execute(select(SoccerMatch.match_id)).scalars())
        for lg in LIGAS:
            if codes and lg.code not in [c.upper() for c in codes]:
                continue
            r = {"nuevos": 0, "temporadas_ok": 0, "temporadas_fallidas": 0, "total_fuente": 0}
            if not lg.historico:
                r["nota"] = "sin fuente historica gratuita verificada"
                resumen[lg.code] = r
                continue
            for temp in temporadas(desde):
                url = BASE.format(temp=temp, f=lg.historico)
                txt, err = _get(url)
                if txt is None:
                    _log(s, "footballcsv", f"{lg.code} {temp}", url, "error", 0, err)
                    r["temporadas_fallidas"] += 1
                    continue
                partidos = _parse(txt)
                _log(s, "footballcsv", f"{lg.code} {temp}", url, "ok", len(partidos))
                if not partidos:
                    continue
                r["temporadas_ok"] += 1
                r["total_fuente"] += len(partidos)
                for p in partidos:
                    mid = _mid(lg.code, p["fecha"], p["home"], p["away"])
                    if mid in existentes:
                        continue
                    existentes.add(mid)
                    s.add(SoccerMatch(
                        match_id=mid, league_code=lg.code, season=temp, stage=p["etapa"],
                        match_date=p["fecha"], kickoff_utc=None,
                        home_team=p["home"], away_team=p["away"],
                        home_goals=p["hg"], away_goals=p["ag"],
                        ht_home_goals=p["hthg"], ht_away_goals=p["htag"],
                        source="footballcsv", source_url=url, fetched_at=ahora))
                    r["nuevos"] += 1
                if progress:
                    progress(f"{lg.code} {temp}: {len(partidos)} partidos")
                s.flush()
            resumen[lg.code] = r
        _equipos(s, ahora)
        estados = _estado_ligas(s, ahora)
    return {"ligas": resumen, "estado": estados}


def _equipos(s, ahora) -> None:
    vistos = {(t.league_code, t.nombre) for t in s.execute(select(SoccerTeam)).scalars()}
    ultimos: dict[tuple[str, str], date] = {}
    for m in s.execute(select(SoccerMatch)).scalars():
        for nombre in (m.home_team, m.away_team):
            k = (m.league_code, nombre)
            if ultimos.get(k) is None or m.match_date > ultimos[k]:
                ultimos[k] = m.match_date
    for (code, nombre), ult in ultimos.items():
        if (code, nombre) in vistos:
            continue
        s.add(SoccerTeam(league_code=code, nombre=nombre, nombre_norm=_norm(nombre),
                         visto_por_ultima_vez=ult))
    s.flush()


def _estado_ligas(s, ahora) -> dict:
    """Recalcula el estado de cada liga a partir de lo que HAY, no de lo que se espera."""
    out = {}
    for lg in LIGAS:
        ms = s.execute(select(SoccerMatch).where(SoccerMatch.league_code == lg.code)).scalars().all()
        n = len(ms)
        fechas = sorted(m.match_date for m in ms) if ms else []
        estado = "ok" if n >= MIN_PARTIDOS_ENTRENAR else "insufficient_data"
        row = s.get(SoccerLeague, lg.code)
        if row is None:
            row = SoccerLeague(code=lg.code)
            s.add(row)
        row.nombre, row.pais, row.odds_key = lg.nombre, lg.pais, lg.odds_key
        row.historico_file = lg.historico
        row.partidos_historicos = n
        row.primera_fecha = fechas[0] if fechas else None
        row.ultima_fecha = fechas[-1] if fechas else None
        row.estado = estado
        row.actualizado_en = ahora
        out[lg.code] = {"partidos": n, "estado": estado,
                        "desde": fechas[0].isoformat() if fechas else None,
                        "hasta": fechas[-1].isoformat() if fechas else None}
    s.flush()
    return out
