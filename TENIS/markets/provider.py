"""Tarjetas de tenis para el dashboard."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import func, select

from TENIS.markets.db import (TenisMatch, TenisModelVersion, TenisOdds, TenisPrediction, TenisResult,
                              TenisSchedule, TenisSourceLog, init_db, session_scope)
from TENIS.markets.gating import decide
from shared.schema import empty_card

LABEL = {"winner": "Ganador", "total_games": "Total juegos", "handicap_games": "Hándicap juegos"}
_GATE = None


def gate():
    global _GATE
    if _GATE is None:
        _GATE = decide()
    return _GATE


def _card(p: TenisPrediction) -> dict:
    g = (gate().get(p.tour) or {}).get(p.market_type, {})
    side1 = (p.final_probability or 0.5) >= 0.5
    side = lambda v: None if v is None else (v if side1 else 1 - v)
    ex = p.explanation or {}
    is_pick = p.status_market == "pick"
    c = empty_card(
        sport="TENIS", game_id=p.event_id, game_date=str(p.match_date), start_utc=p.start_utc,
        status="scheduled" if p.result is None else "final",
        home=p.p1_name, away=p.p2_name, home_name=p.p1_name, away_name=p.p2_name,
        venue=f"{p.tourney_name or ''} · {p.surface or ''} · al mejor de {p.best_of}".strip(" ·"),
        market=LABEL.get(p.market_type, p.market_type),
        selection=(p.selection if p.status_market != "no_pick" else f"NO PICK ({p.selection})"),
        line=p.line, model_probability=side(p.model_probability), market_probability=side(p.market_probability),
        ensemble_probability=p.pick_probability, elo_probability=None, confidence=p.confidence,
        upset_risk=round((1 - (p.pick_probability or 0.5)) * 100, 2),
        upset_label=("LOW RISK" if (p.pick_probability or 0) >= 0.65 else
                     "MEDIUM RISK" if (p.pick_probability or 0) >= 0.57 else "HIGH RISK"),
        upset_explanation={"summary": ex.get("summary"), "flags": ex.get("flags", []),
                           "drivers": [{"text": f"{f['label']}: {f['value']}", **f} for f in ex.get("factors", [])],
                           "verdict": (None if is_pick else
                                       "Proyeccion informativa: sin historico de cuotas para demostrar ventaja."
                                       if p.status_market == "projection" else "NO PICK: demasiado parejo.")},
        model_market_gap=(None if p.gap_pp is None else p.gap_pp / 100),
        directional_disagreement=(None if p.market_probability is None else
                                  bool((p.model_probability >= .5) != (p.market_probability >= .5))),
        agreement_bucket=None, model_dispersion=None, market_signal=None,
        blend_policy={"modelo": 1.0, "mercado": 0.0, "nota": "mercado solo como referencia"},
        sources=["sackmann-archive (ATP/WTA)"] + (["the_odds_api"] if p.market_probability is not None
                                                  or p.line_source == "the_odds_api" else []),
        model_version=p.model_version, prediction_timestamp=p.prediction_timestamp, prediction_id=p.id,
        version=p.version,
        status_note=(None if is_pick else "NO PICK: por debajo del umbral validado" if p.status_market == "no_pick"
                     else "PROYECCION, no pick: sin historico de cuotas"),
        result=p.result, correct=p.correct)
    dc = p.data_completeness or {}
    c["data_completeness"] = {"complete": bool(dc.get("complete", True)), "missing": dc.get("missing", []), "notes": []}
    c["extra"] = {"tour": p.tour, "market_key": p.market_type, "publish_pick": is_pick, "mode": p.status_market,
                  "expected_value": p.expected_value, "line": p.line, "line_source": p.line_source,
                  "p_serve_1": p.p_serve_1, "p_serve_2": p.p_serve_2, "engine": p.engine,
                  "odds_1": p.odds_1, "odds_2": p.odds_2, "gap_pp": p.gap_pp, "surface": p.surface,
                  "tourney": p.tourney_name, "best_of": p.best_of, "actual_value": p.actual_value,
                  "factors": ex.get("factors", []), "market_caveat": g.get("reason"),
                  "threshold": g.get("threshold"),
                  "risk_meaning": "probabilidad calibrada de que el lado mostrado falle"}
    return c


def games(date_from: str, date_to: str, limit: int = 600) -> list[dict]:
    init_db()
    with session_scope() as s:
        rows = s.execute(select(TenisPrediction).where(
            TenisPrediction.match_date >= date.fromisoformat(date_from),
            TenisPrediction.match_date <= date.fromisoformat(date_to),
            TenisPrediction.status != "superseded")
            .order_by(TenisPrediction.start_utc.asc()).limit(limit)).scalars().all()
        return [_card(p) for p in rows]


def markets_status() -> dict:
    return gate()


def health() -> dict:
    init_db()
    with session_scope() as s:
        out = {"circuitos": {}}
        for tour in ("ATP", "WTA"):
            n = s.scalar(select(func.count()).select_from(TenisMatch).where(TenisMatch.tour == tour)) or 0
            last = s.scalar(select(func.max(TenisMatch.match_date)).where(TenisMatch.tour == tour))
            mv = s.execute(select(TenisModelVersion).where(TenisModelVersion.tour == tour,
                                                           TenisModelVersion.market == "winner",
                                                           TenisModelVersion.is_production.is_(True))).scalars().first()
            preds = s.execute(select(TenisPrediction).where(TenisPrediction.tour == tour,
                                                            TenisPrediction.status != "superseded")).scalars().all()
            g = gate()[tour]
            out["circuitos"][tour] = {
                "partidos": n, "ultimo_partido": str(last) if last else None,
                "modelo": mv.name if mv else None, "entrenado": str(mv.created_at) if mv else None,
                "predicciones_activas": len(preds), "con_resultado": sum(1 for p in preds if p.result is not None),
                "mercados": {m: {"estado": ("OK" if g[m]["enabled"] else "BLOCKED"), "modo": g[m]["mode"]}
                             for m in ("winner", "total_games", "handicap_games")},
                "estado": ("OK" if (mv and n > 10000) else "WARNING" if mv else "BLOCKED")}
        res_n = s.scalar(select(func.count()).select_from(TenisResult)) or 0
        jugados = s.execute(select(TenisPrediction).where(
            TenisPrediction.status != "superseded", TenisPrediction.result.is_(None))).scalars().all()
        from datetime import datetime as _dt
        pend = [p for p in jugados if p.start_utc and p.start_utc < _dt.utcnow()]
        out["resultados"] = {
            "fuente": "the_odds_api (/scores): ganador y sets",
            "partidos_con_resultado": res_n,
            "predicciones_jugadas_sin_calificar": len(pend),
            "por_mercado_sin_calificar": {m: sum(1 for p in pend if p.market_type == m)
                                          for m in ("winner", "total_games", "handicap_games")},
            "limitacion": ("no hay fuente gratuita con marcador POR JUEGOS: total de juegos y handicap "
                           "no se pueden calificar y quedan pendientes; el ganador si se califica")}
        sched = s.scalar(select(func.count()).select_from(TenisSchedule)) or 0
        odds = s.scalar(select(func.count()).select_from(TenisOdds)) or 0
        last_fetch = s.scalar(select(func.max(TenisSourceLog.retrieved_at)))
        errors = s.scalar(select(func.count()).select_from(TenisSourceLog).where(TenisSourceLog.status == "error")) or 0
        last_match = s.scalar(select(func.max(TenisMatch.match_date)))
        stale = (date.today() - last_match).days if last_match else None
        out.update({"calendario_eventos": sched, "cuotas_snapshots": odds,
                    "ultima_descarga": str(last_fetch) if last_fetch else None, "errores_descarga": errors,
                    "archivo_congelado_en": str(last_match) if last_match else None,
                    "antiguedad_archivo_dias": stale,
                    "cuotas_historicas": "ninguna: solo snapshots propios desde la activacion",
                    "leakage": "features as-of por dia; tests automaticos",
                    "estado_global": ("WARNING" if (errors > 0 or (stale or 0) > 45) else "OK")})
        return out
