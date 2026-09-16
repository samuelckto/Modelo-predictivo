"""Deduplicacion de partidos. Sin esto los numeros del backtest son mentira.

Tres fuentes cargan el mismo partido con nombres distintos:

    openfootball        'Liverpool FC'  vs 'AFC Bournemouth'
    football-datasets   'Liverpool'     vs 'Bournemouth'

El `match_id` se calcula con el nombre CRUDO, asi que el mismo partido entraba dos
veces. Se detecto porque el holdout mejoraba de golpe sin razon: 733 filas en una
temporada de 380 partidos, con el equipo actualizando su estado dos veces por
partido.

La union se hace por (liga, fecha, equipos canonicos). Se conserva el registro
mas completo y se rellenan sus huecos con el otro; el sobrante se borra. Si dos
registros dan marcadores DISTINTOS para el mismo partido no se fusionan: se marca
el conflicto y se deja el mas antiguo, porque inventar cual es el bueno seria peor.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select

from SOCCER.db import SoccerMatch, SoccerSourceLog, init_db, session_scope
from SOCCER.ingestion.teams import Resolver
from shared.timeutil import utcnow

CAMPOS_STATS = ("home_corners", "away_corners", "home_shots", "away_shots",
                "home_sot", "away_sot", "home_fouls", "away_fouls",
                "home_yellow", "away_yellow", "ht_home_goals", "ht_away_goals",
                "kickoff_utc", "stage")


def _riqueza(m: SoccerMatch) -> int:
    return sum(1 for c in CAMPOS_STATS if getattr(m, c) is not None)


def run(dry_run: bool = False) -> dict:
    init_db()
    with session_scope() as s:
        partidos = s.execute(select(SoccerMatch).order_by(
            SoccerMatch.match_date, SoccerMatch.match_id)).scalars().all()
        res = Resolver()
        clave: dict[tuple, list[SoccerMatch]] = {}
        for m in partidos:
            h, _ = res.resolver(m.league_code, m.home_team)
            a, _ = res.resolver(m.league_code, m.away_team)
            clave.setdefault((m.league_code, m.match_date, h, a), []).append(m)

        # segunda pasada: fechas desplazadas un dia entre fuentes
        por_equipos: dict[tuple, list[tuple]] = {}
        for k in clave:
            por_equipos.setdefault((k[0], k[2], k[3]), []).append(k)
        for _, ks in por_equipos.items():
            ks.sort(key=lambda x: x[1])
            for i in range(len(ks) - 1):
                a, b = ks[i], ks[i + 1]
                if b[1] - a[1] == timedelta(days=1) and clave.get(a) and clave.get(b):
                    ga, gb = clave[a], clave[b]
                    if {(x.home_goals, x.away_goals) for x in ga} == {
                            (x.home_goals, x.away_goals) for x in gb}:
                        clave[a] = ga + gb
                        clave[b] = []

        fusionados = borrados = conflictos = 0
        detalle_conflictos = []
        for k, grupo in clave.items():
            if len(grupo) < 2:
                continue
            marcadores = {(m.home_goals, m.away_goals) for m in grupo}
            if len(marcadores) > 1:
                conflictos += 1
                detalle_conflictos.append({"liga": k[0], "fecha": str(k[1]),
                                           "local": k[2], "visitante": k[3],
                                           "marcadores": sorted(marcadores)})
                continue
            grupo = sorted(grupo, key=lambda m: (-_riqueza(m), m.match_id))
            principal, resto = grupo[0], grupo[1:]
            for otro in resto:
                for c in CAMPOS_STATS:
                    if getattr(principal, c) is None and getattr(otro, c) is not None:
                        setattr(principal, c, getattr(otro, c))
                if principal.stats_source is None and otro.stats_source:
                    principal.stats_source = otro.stats_source
                if not dry_run:
                    s.delete(otro)
                borrados += 1
            fusionados += 1
        if not dry_run:
            s.add(SoccerSourceLog(source="dedupe", scope="matches", url=None, status="ok",
                                  records=borrados, retrieved_at=utcnow(),
                                  error=(f"{conflictos} conflictos de marcador" if conflictos
                                         else None)))
            s.flush()
        total = len(partidos)
    return {"partidos_antes": total, "grupos_fusionados": fusionados,
            "registros_borrados": borrados, "partidos_despues": total - borrados,
            "conflictos_de_marcador": conflictos,
            "ejemplos_conflicto": detalle_conflictos[:8], "dry_run": dry_run}
