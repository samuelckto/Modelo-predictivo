"""Ciclos de operacion de SOCCER.

    train()   historico + estadisticas + dedupe + features + auditoria + modelos
    cycle()   calendario + cuotas + resultados + features + predicciones actuales
    score()   califica lo ya jugado

`cycle` esta escrito para que ninguna fuente caida detenga el resto: cada paso
devuelve su propio estado y el ciclo continua. Lo que no se pudo actualizar se
reporta, no se disimula.
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
from sqlalchemy import select

from SOCCER.db import SoccerFixture, SoccerMatch, SoccerPrediction, init_db, session_scope
from SOCCER.ingestion import cups, dedupe, historical, live, matchstats, recent
from SOCCER.ingestion.teams import Resolver
from SOCCER.markets import predict as P
from SOCCER.markets.gating import desde_walkforward, estado_por_liga
from shared.paths import SOCCER_OUT_DIR
from shared.timeutil import utcnow


def train(desde: int = 2012, progress=None) -> dict:
    """Todo lo que se hace una vez (o cuando se quiere reentrenar)."""
    p = progress or (lambda *_: None)
    out = {}
    p("1/8 historico (espejo footballcsv)")
    out["historico"] = historical.ingest(desde=desde)["estado"]
    p("2/8 temporadas recientes (openfootball)")
    out["recientes"] = recent.ingest(desde=2024)
    p("3/8 copas europeas (Champions League)")
    out["copas"] = cups.ingest(desde=2011)
    p("4/8 estadisticas de partido y corners (football-datasets)")
    out["estadisticas"] = matchstats.ingest(desde=desde)
    # Intento oportunista: si football-data.co.uk vuelve, Liga MX gana sus corners
    # y el gating la activa sola en el siguiente refresco.
    out["corners_ligamx"] = matchstats.liga_mx_corners(progress=p)
    p("5/8 deduplicacion")
    out["dedupe"] = dedupe.run()
    # El estado de cada liga se recalcula DESPUES de cargarlo todo: si no, una
    # competicion cargada al final aparece con cero partidos en Data Health.
    with session_scope() as s:
        out["estado_ligas"] = historical._estado_ligas(s, utcnow())
    p("6/8 features as-of")
    from SOCCER.features.build import build                    # noqa: PLC0415
    out["features"] = build()
    p("7/8 auditoria anti-leakage")
    from SOCCER.evaluation.leakage import audit                # noqa: PLC0415
    aud = audit()
    out["leakage"] = {"ok": aud["ok"],
                      "pruebas": {k: v["ok"] for k, v in aud["pruebas"].items()}}
    if not aud["ok"]:
        return {**out, "abortado": "la auditoria anti-leakage fallo: no se entrena"}
    p("8/8 entrenamiento final")
    from SOCCER.markets.train import entrenar                  # noqa: PLC0415
    out["modelos"] = entrenar(progress=p)
    return out


def refrescar_gating() -> dict:
    """Recalcula el gating con lo que hay ahora en la base."""
    g = desde_walkforward(hay_cuotas_historicas=False)
    with session_scope() as s:
        el = historical._estado_ligas(s, utcnow())
    por_liga = estado_por_liga(g, matchstats.cobertura(), el)
    SOCCER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (SOCCER_OUT_DIR / "gating.json").write_text(
        json.dumps({"global": g, "por_liga": por_liga}, indent=1, default=str), encoding="utf-8")
    return {"mercados": {k: v["estado"] for k, v in g["mercados"].items()},
            "por_liga": {k: {m: x["estado"] for m, x in v.items()} for k, v in por_liga.items()}}


def cycle(dias: int = 8, progress=None) -> dict:
    """Ciclo diario. Datos ACTUALES -> predicciones ACTUALES."""
    init_db()
    p = progress or (lambda *_: None)
    out = {"iniciado_en": utcnow().isoformat()}
    p("calendario")
    out["fixtures"] = live.fixtures(dias=dias)
    p("cuotas")
    out["odds"] = live.odds()
    p("resultados")
    out["resultados"] = live.results()
    p("incorporando resultados nuevos al historico")
    out["historico_nuevo"] = absorber_resultados()
    if out["historico_nuevo"]["nuevos"]:
        p("reconstruyendo features con los resultados nuevos")
        from SOCCER.features.build import build                # noqa: PLC0415
        out["features"] = build()
        out["gating"] = refrescar_gating()
    p("predicciones")
    out["predicciones"] = P.generar(dias=dias)
    p("calificacion")
    out["score"] = score()
    out["terminado_en"] = utcnow().isoformat()
    return out


def absorber_resultados() -> dict:
    """Un fixture terminado pasa al historico propio. Asi crece Saudi sin tocar codigo."""
    init_db()
    ahora = utcnow()
    nuevos = 0
    with session_scope() as s:
        existentes = set(s.execute(select(SoccerMatch.match_id)).scalars())
        res = Resolver()
        for f in s.execute(select(SoccerFixture).where(
                SoccerFixture.completed.is_(True))).scalars():
            if f.home_goals is None or f.away_goals is None:
                continue
            mid = historical._mid(f.league_code, f.commence_utc.date(), f.home_team, f.away_team)
            if mid in existentes:
                continue
            temporada = f"{f.commence_utc.year}-{(f.commence_utc.year + 1) % 100:02d}"
            s.add(SoccerMatch(match_id=mid, league_code=f.league_code, season=temporada,
                              match_date=f.commence_utc.date(), kickoff_utc=f.commence_utc,
                              home_team=f.home_team, away_team=f.away_team,
                              home_goals=f.home_goals, away_goals=f.away_goals,
                              source="live", source_url=None, fetched_at=ahora))
            existentes.add(mid)
            nuevos += 1
        s.flush()
    if nuevos:
        dedupe.run()
    return {"nuevos": nuevos}


def score() -> dict:
    """Califica las predicciones cuyos partidos ya terminaron."""
    init_db()
    ahora = utcnow()
    n = 0
    with session_scope() as s:
        fixtures = {f.event_id: f for f in s.execute(select(SoccerFixture).where(
            SoccerFixture.completed.is_(True))).scalars()}
        pendientes = s.execute(select(SoccerPrediction).where(
            SoccerPrediction.result.is_(None),
            SoccerPrediction.record_status == "active")).scalars().all()
        for pr in pendientes:
            f = fixtures.get(pr.event_id)
            if f is None or f.home_goals is None or f.away_goals is None:
                continue
            if pr.status in ("insufficient_data", "blocked"):
                continue                       # no se publico probabilidad: no se califica
            if pr.market == "double_chance":
                # OJO: este mercado publica DOS tipos de apuesta distintos y
                # calificarlos igual seria regalarle aciertos al sistema.
                #   'X gana'        -> solo acierta si X gana. Un empate es fallo.
                #   'X o empate'    -> acierta si X gana o empata.
                # Antes solo existia el segundo caso, asi que bastaba mirar de
                # quien era la seleccion; ahora hay que mirar tambien el tipo.
                eligio_local = pr.selection.startswith(pr.home_team)
                solo_ganador = pr.selection.endswith(" gana")
                if solo_ganador:
                    acerto = (f.home_goals > f.away_goals if eligio_local
                              else f.away_goals > f.home_goals)
                else:
                    acerto = (f.home_goals >= f.away_goals if eligio_local
                              else f.away_goals >= f.home_goals)
                pr.actual_value = float(f.home_goals - f.away_goals)
                pr.result = "win" if acerto else "loss"
                pr.correct = bool(acerto)
            elif pr.market == "btts":
                real = 1.0 if (f.home_goals > 0 and f.away_goals > 0) else 0.0
                acerto = (real == 1.0) == (pr.selection == "SI")
                pr.actual_value = real
                pr.result = "win" if acerto else "loss"
                pr.correct = bool(acerto)
            elif pr.market == "total_goals":
                real = float(f.home_goals + f.away_goals)
                pr.actual_value = real
                if pr.line is not None and real == pr.line:
                    pr.result, pr.correct = "push", None
                else:
                    over = real > (pr.line or 0)
                    acerto = over == pr.selection.startswith("OVER")
                    pr.result = "win" if acerto else "loss"
                    pr.correct = bool(acerto)
            else:                              # corners: sin marcador de corners en vivo
                m = s.execute(select(SoccerMatch).where(
                    SoccerMatch.league_code == pr.league_code,
                    SoccerMatch.match_date == pr.match_date,
                    SoccerMatch.home_corners.isnot(None))).scalars().all()
                r = next((x for x in m if x.home_team == pr.home_team
                          or x.away_team == pr.away_team), None)
                if r is None:
                    continue                   # queda pendiente, no se cierra a ojo
                real = float(r.home_corners + r.away_corners)
                pr.actual_value = real
                if pr.line is not None and real == pr.line:
                    pr.result, pr.correct = "push", None
                else:
                    over = real > (pr.line or 0)
                    acerto = over == pr.selection.startswith("OVER")
                    pr.result = "win" if acerto else "loss"
                    pr.correct = bool(acerto)
            pr.scored_at = ahora
            n += 1
        s.flush()
    return {"calificadas": n}
