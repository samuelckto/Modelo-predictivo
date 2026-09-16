"""Copas europeas. Hoy: Champions League.

Una copa no es una liga y el modulo lo trata distinto a proposito:

- Sus equipos NO son suyos: vienen de las ligas nacionales. El estado de un club
  (Elo, forma, goles, corners) es el mismo en su liga y aqui, porque el resolver
  de nombres y el recorrido de features usan un pool GLOBAL de clubes.
- Por eso los partidos de Champions son los unicos que CONECTAN los Elo de ligas
  distintas. Sin ellos, un 1700 de LaLiga y un 1700 de la Bundesliga no eran
  comparables: nunca se habian enfrentado.
- No hay fuente de corners para Champions, asi que ese mercado queda
  INSUFFICIENT DATA en esta competicion. Goles y BTTS si.

Fuente: openfootball/champions-league, formato de texto por jornadas, con hora de
saque y marcador de medio tiempo. Verificado: 2011-12 a 2025-26.
"""
from __future__ import annotations

import urllib.request

from sqlalchemy import select

from SOCCER.db import SoccerMatch, SoccerSourceLog, init_db, session_scope
from SOCCER.ingestion.historical import _mid
from SOCCER.ingestion.recent import _parse_txt
from shared.timeutil import utcnow

BASE = "https://raw.githubusercontent.com/openfootball/champions-league/master/{temp}/{f}.txt"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124"}

# competicion interna -> archivos del repositorio.
# 'cl' es la fase principal; 'clq' la previa. Se cargan las dos: los equipos de
# la previa tambien juegan y sus partidos son historia real del club.
COPAS = {"UCL": ("cl", "clq")}


def temporadas(desde: int = 2011, hasta: int | None = None) -> list[str]:
    fin = hasta if hasta is not None else utcnow().year
    return ["%d-%02d" % (y, (y + 1) % 100) for y in range(desde, fin + 1)]


def _get(url, timeout=40):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore"), None
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def ingest(codes: list[str] | None = None, desde: int = 2011, progress=None) -> dict:
    """Carga el historico de copa. Idempotente."""
    init_db()
    ahora = utcnow()
    resumen: dict[str, dict] = {}
    with session_scope() as s:
        existentes = set(s.execute(select(SoccerMatch.match_id)).scalars())
        for code, archivos in COPAS.items():
            if codes and code not in [c.upper() for c in codes]:
                continue
            r = {"nuevos": 0, "temporadas_ok": 0, "sin_datos": 0, "partidos_fuente": 0}
            for temp in temporadas(desde):
                encontrado = False
                for f in archivos:
                    url = BASE.format(temp=temp, f=f)
                    txt, err = _get(url)
                    if txt is None:
                        s.add(SoccerSourceLog(source="openfootball-cup", scope=f"{code} {temp} {f}",
                                              url=url, status="error", records=0, error=err,
                                              retrieved_at=ahora))
                        continue
                    ms = _parse_txt(txt, int(temp.split("-")[0]))
                    s.add(SoccerSourceLog(source="openfootball-cup", scope=f"{code} {temp} {f}",
                                          url=url, status="ok", records=len(ms),
                                          retrieved_at=ahora))
                    if not ms:
                        continue
                    encontrado = True
                    r["partidos_fuente"] += len(ms)
                    for p in ms:
                        if not p["home"] or not p["away"]:
                            continue
                        mid = _mid(code, p["fecha"], p["home"], p["away"])
                        if mid in existentes:
                            continue
                        existentes.add(mid)
                        s.add(SoccerMatch(
                            match_id=mid, league_code=code, season=temp,
                            stage=(p["etapa"] or ("previa" if f.endswith("q") else None)),
                            match_date=p["fecha"], kickoff_utc=p["kickoff"],
                            home_team=p["home"], away_team=p["away"],
                            home_goals=p["hg"], away_goals=p["ag"],
                            ht_home_goals=p["hthg"], ht_away_goals=p["htag"],
                            source="openfootball-cup", source_url=url, fetched_at=ahora))
                        r["nuevos"] += 1
                if encontrado:
                    r["temporadas_ok"] += 1
                    if progress:
                        progress(f"{code} {temp}: {r['partidos_fuente']} partidos acumulados")
                else:
                    r["sin_datos"] += 1
                s.flush()
            resumen[code] = r
    return resumen
