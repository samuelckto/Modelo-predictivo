"""Cerrar fixtures jugados con el marcador del historico GRATUITO.

Por que existe este modulo:

    El unico camino para cerrar un partido de futbol era `live.results()`, que
    pega a the-odds-api. Esa clave tiene 500 peticiones al mes y hoy esta a 0,
    asi que devuelve 401 en las 8 ligas. Resultado: 69 predicciones de partidos
    ya jugados seguian sin calificar, y en el dashboard salian eternamente
    pendientes, sin acierto ni fallo.

    Pero el marcador YA ESTA en el sistema: `recent.py` lo baja gratis de
    openfootball (GitHub, sin cuota) y lo guarda en la tabla de historico. Lo
    unico que faltaba era cruzar una cosa con la otra.

Como se cruza, y por que asi:

    Por FECHA + EQUIPOS RESUELTOS, nunca por parecido de texto. El resolvedor
    de nombres del proyecto ya mapea 'Man United' y 'Manchester United' al mismo
    equipo dentro de su liga; usarlo aqui evita el error clasico de emparejar
    'Manchester United' con 'Manchester City' por similitud.

    Si un fixture no encuentra su partido, se queda abierto. Cerrar un partido
    con el marcador de otro seria mucho peor que dejarlo pendiente: contaminaria
    el acierto historico con resultados falsos.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select

from SOCCER.db import SoccerFixture, SoccerMatch, init_db, session_scope
from SOCCER.ingestion.teams import Resolver

# Tolerancia de fecha. Un partido que empieza a las 19:00 UTC del dia 10 puede
# estar registrado como dia 10 u 11 segun la zona de la fuente. Un dia de margen
# cubre eso sin llegar a confundir dos jornadas distintas del mismo cruce.
DIAS_TOLERANCIA = 1


def _clave(res: Resolver, code: str, home: str, away: str):
    h, _ = res.resolver(code, home)
    a, _ = res.resolver(code, away)
    if not h or not a:
        return None
    return (code, h, a)


def cerrar(dias_atras: int = 30) -> dict:
    """Rellena marcador y `completed` en los fixtures que ya se jugaron.

    Devuelve cuantos se cerraron y cuantos siguen sin marcador, con el motivo.
    No inventa ningun resultado: si no hay partido que case, no toca el fixture.
    """
    init_db()
    res = Resolver()
    out = {"ok": True, "cerrados": 0, "sin_marcador": 0, "sin_resolver": 0,
           "revisados": 0, "por_liga": {}}

    with session_scope() as s:
        abiertos = s.execute(select(SoccerFixture).where(
            SoccerFixture.home_goals.is_(None))).scalars().all()
        if not abiertos:
            return out

        desde = min(f.commence_utc for f in abiertos).date() - timedelta(days=DIAS_TOLERANCIA)
        historico = s.execute(select(SoccerMatch).where(
            SoccerMatch.match_date >= desde,
            SoccerMatch.home_goals.isnot(None))).scalars().all()

        # Indice por (liga, local, visitante, fecha). El mismo cruce se repite
        # en temporadas distintas, asi que la fecha forma parte de la clave.
        indice: dict[tuple, SoccerMatch] = {}
        for m in historico:
            k = _clave(res, m.league_code, m.home_team, m.away_team)
            if k:
                indice[(*k, m.match_date)] = m

        for f in abiertos:
            out["revisados"] += 1
            k = _clave(res, f.league_code, f.home_team, f.away_team)
            if k is None:
                out["sin_resolver"] += 1
                continue
            dia = f.commence_utc.date()
            m = None
            for delta in range(-DIAS_TOLERANCIA, DIAS_TOLERANCIA + 1):
                m = indice.get((*k, dia + timedelta(days=delta)))
                if m is not None:
                    break
            if m is None:
                out["sin_marcador"] += 1
                continue
            f.home_goals = m.home_goals
            f.away_goals = m.away_goals
            f.completed = True
            out["cerrados"] += 1
            out["por_liga"][f.league_code] = out["por_liga"].get(f.league_code, 0) + 1
    return out
