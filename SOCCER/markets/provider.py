"""Adaptador de SOCCER al dashboard. Solo LEE: no entrena ni predice.

Devuelve las mismas claves que los demas deportes para que la tarjeta unificada
funcione sin tocar la logica de NFL, MLB, NBA ni TENIS.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from sqlalchemy import select

from SOCCER.db import (SoccerFixture, SoccerLeague, SoccerMatch, SoccerOdds, SoccerPrediction,
                       SoccerSourceLog, init_db, session_scope)
from SOCCER.ingestion.matchstats import cobertura
from SOCCER.leagues import LIGAS, POR_CODE
from SOCCER.markets.gating import LABEL, ORDEN
from shared.paths import SOCCER_OUT_DIR
from shared.timeutil import utcnow

ETIQUETA_ESTADO = {"pick": "PICK", "projection": "PROYECCIÓN", "no_pick": "NO PICK",
                   "insufficient_data": "INSUFFICIENT DATA", "blocked": "BLOQUEADO"}
DIAS_STALE = 45


def _card(p: SoccerPrediction) -> dict:
    ex = json.loads(p.explanation or "{}")
    dc = json.loads(p.data_completeness or "{}")
    liga = POR_CODE.get(p.league_code)
    return {
        "sport": "SOCCER", "game_id": p.event_id, "prediction_id": p.prediction_id,
        "game_date": p.match_date.isoformat(), "start_utc": (p.kickoff_utc.isoformat()
                                                             if p.kickoff_utc else None),
        "status": "final" if p.result else "programado",
        "home": p.home_team, "away": p.away_team,
        "home_name": p.home_team, "away_name": p.away_team,
        "venue": f"{liga.nombre if liga else p.league_code} · {liga.pais if liga else ''}",
        "market": LABEL.get(p.market, p.market), "selection": p.selection, "line": p.line,
        "model_probability": p.probability, "market_probability": p.market_probability,
        "ensemble_probability": p.probability, "elo_probability": None,
        "confidence": None, "model_market_gap": p.gap_pp,
        "model_version": p.model_version, "version": p.version,
        "prediction_timestamp": p.prediction_timestamp.isoformat(),
        "status_note": ex.get("motivo_estado"),
        "result": p.result, "correct": p.correct,
        "data_completeness": {"complete": dc.get("cuotas", False) is not None, "missing": [],
                              "notes": []},
        "extra": {"league": p.league_code,
                  "league_name": liga.nombre if liga else p.league_code,
                  "market_key": p.market, "mode": p.status,
                  "estado": ETIQUETA_ESTADO.get(p.status, p.status),
                  "publish_pick": p.status == "pick",
                  "expected_value": p.expected_goals,
                  "lambda_home": p.lambda_home, "lambda_away": p.lambda_away,
                  "line": p.line, "lineas": ex.get("lineas", {}),
                  "odds_price": p.odds_price, "bookmaker": p.bookmaker,
                  "odds_timestamp": p.odds_timestamp.isoformat() if p.odds_timestamp else None,
                  "actual_value": p.actual_value,
                  "factors": ex.get("factores", []),
                  "market_caveat": ex.get("aviso_mercado"),
                  "feature_cutoff": p.feature_cutoff.isoformat(),
                  "data": dc,
                  "risk_meaning": "probabilidad calibrada del lado mostrado"},
    }


def games(date_from: str, date_to: str, limit: int = 800) -> list[dict]:
    init_db()
    a = date.fromisoformat(str(date_from)[:10])
    b = date.fromisoformat(str(date_to)[:10])
    with session_scope() as s:
        filas = s.execute(select(SoccerPrediction).where(
            SoccerPrediction.match_date >= a, SoccerPrediction.match_date <= b,
            SoccerPrediction.record_status == "active").limit(limit)).scalars().all()
        return [_card(p) for p in filas]


def top_picks(date_from: str, date_to: str) -> list[dict]:
    """SOLO estado PICK. Las proyecciones no entran aqui nunca."""
    return [c for c in games(date_from, date_to) if c["extra"]["mode"] == "pick"]


def markets_status() -> dict:
    """Que publica cada mercado y por que, con la evidencia detras."""
    p = SOCCER_OUT_DIR / "gating.json"
    if not p.exists():
        return {m: {"enabled": False, "reason": "sin gating: ejecuta soccer-train"} for m in ORDEN}
    g = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for m in ORDEN:
        v = g.get("global", {}).get("mercados", {}).get(m, {})
        out[m] = {"enabled": v.get("publica_probabilidad", False),
                  "publish_pick": v.get("permite_pick", False),
                  "mode": v.get("estado", "insufficient_data"),
                  "label": LABEL[m], "reason": v.get("motivo"),
                  "log_loss": v.get("log_loss"), "baseline": v.get("baseline"),
                  "mejora": v.get("mejora"), "accuracy": v.get("accuracy"),
                  "accuracy_baseline": v.get("accuracy_baseline"), "n": v.get("n")}
    out["_por_liga"] = g.get("por_liga", {})
    return out


def health() -> dict:
    """Data Health: frescura real por liga. No esconde lo viejo."""
    init_db()
    ahora = utcnow()
    cob = cobertura()
    with session_scope() as s:
        ligas = {l.code: l for l in s.execute(select(SoccerLeague)).scalars()}
        fixtures = s.execute(select(SoccerFixture)).scalars().all()
        odds = s.execute(select(SoccerOdds)).scalars().all()
        logs = s.execute(select(SoccerSourceLog).order_by(
            SoccerSourceLog.retrieved_at.desc()).limit(60)).scalars().all()
        preds = s.execute(select(SoccerPrediction).where(
            SoccerPrediction.record_status == "active")).scalars().all()

        por_liga = {}
        for lg in LIGAS:
            row = ligas.get(lg.code)
            ult = row.ultima_fecha if row else None
            edad = (ahora.date() - ult).days if ult else None
            fx = [f for f in fixtures if f.league_code == lg.code
                  and f.commence_utc >= ahora - timedelta(hours=2)]
            con_cuotas = {o.event_id for o in odds if o.league_code == lg.code}
            c = cob.get(lg.code, {})
            if not row or row.partidos_historicos < 1200:
                estado = "INSUFFICIENT DATA"
            elif edad is not None and edad > DIAS_STALE:
                estado = "STALE"
            else:
                estado = "OK"
            por_liga[lg.code] = {
                "liga": lg.nombre, "pais": lg.pais, "estado": estado,
                "partidos_historicos": row.partidos_historicos if row else 0,
                "ultima_fecha_historica": ult.isoformat() if ult else None,
                "antiguedad_dias": edad,
                "corners_partidos": c.get("con_corners", 0),
                "corners_cobertura": c.get("cobertura", 0.0),
                "corners_hasta": c.get("hasta"),
                "fixtures_proximos": len(fx),
                "fixtures_sin_cuotas": sum(1 for f in fx if f.event_id not in con_cuotas),
                "odds_key": lg.odds_key,
            }
        ult_odds = max((o.fetched_at for o in odds), default=None)
        ult_api = max((l.retrieved_at for l in logs if l.source == "the_odds_api"), default=None)
        errores = [{"fuente": l.source, "ambito": l.scope, "error": l.error,
                    "cuando": l.retrieved_at.isoformat()}
                   for l in logs if l.status == "error"][:10]
    return {
        "generado_en": ahora.isoformat(),
        "ligas": por_liga,
        "fixtures_totales": len([f for f in fixtures if f.commence_utc >= ahora]),
        "snapshots_de_cuotas": len(odds),
        "ultimo_snapshot": ult_odds.isoformat() if ult_odds else None,
        "edad_snapshot_horas": (round((ahora - ult_odds).total_seconds() / 3600, 1)
                                if ult_odds else None),
        "ultima_consulta_api": ult_api.isoformat() if ult_api else None,
        "predicciones_activas": len(preds),
        "por_estado": {e: sum(1 for p in preds if p.status == e)
                       for e in {p.status for p in preds}},
        "errores_recientes": errores,
        "avisos": _avisos(por_liga, ult_odds, ahora),
    }


def _avisos(por_liga: dict, ult_odds, ahora) -> list[str]:
    out = []
    for code, v in por_liga.items():
        if v["estado"] == "INSUFFICIENT DATA":
            out.append(f"{v['liga']}: histórico insuficiente ({v['partidos_historicos']} partidos), "
                       f"no se generan picks")
        elif v["estado"] == "STALE":
            out.append(f"{v['liga']}: el histórico termina el {v['ultima_fecha_historica']} "
                       f"({v['antiguedad_dias']} días): la forma de los equipos está vieja")
        if v["corners_partidos"] == 0:
            out.append(f"{v['liga']}: sin fuente de córners, ese mercado no se publica")
    if ult_odds is None:
        out.append("todavía no hay ningún snapshot de cuotas guardado")
    elif (ahora - ult_odds).total_seconds() > 6 * 3600:
        out.append(f"las cuotas tienen más de 6 horas (último snapshot {ult_odds.isoformat()})")
    return out
