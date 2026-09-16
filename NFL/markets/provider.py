"""Tarjetas de total y spread NFL para el dashboard (misma forma que las demas)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from NFL.adapter import TEAM_NAMES
from NFL.markets.db import NflMarketPrediction, NflModelVersion, NflOdds, init_db, session_scope
from NFL.markets.gating import decide
from shared.schema import empty_card

LABEL = {"total": "Total", "spread": "Spread"}
_GATE = None


def gate() -> dict:
    global _GATE
    if _GATE is None:
        _GATE = decide()
    return _GATE


def _card(p: NflMarketPrediction) -> dict:
    g = gate().get(p.market_type, {})
    over_side = (p.final_probability or 0.5) >= 0.5
    side = lambda v: None if v is None else (v if over_side else 1 - v)
    ex = p.explanation or {}
    risk = round((1 - (p.pick_probability or 0.5)) * 100, 2)
    c = empty_card(
        sport="NFL", game_id=p.game_id, game_date=str(p.kickoff_utc)[:10], start_utc=p.kickoff_utc,
        status="scheduled" if p.result is None else "final",
        home=p.home_team, away=p.away_team,
        home_name=TEAM_NAMES.get(p.home_team, p.home_team), away_name=TEAM_NAMES.get(p.away_team, p.away_team),
        market=LABEL[p.market_type], selection=p.selection if p.pick == "pick" else p.selection,
        line=p.line,
        model_probability=side(p.model_probability), market_probability=side(p.market_probability),
        ensemble_probability=p.pick_probability, elo_probability=None,
        confidence=p.confidence, upset_risk=risk,
        upset_label=("LOW RISK" if p.pick_probability and p.pick_probability >= 0.62 else
                     "MEDIUM RISK" if p.pick_probability and p.pick_probability >= 0.545 else "HIGH RISK"),
        upset_explanation={"summary": ex.get("summary"), "flags": ex.get("flags", []),
                           "drivers": [{"text": f"{f['label']}: {f['points']:+.1f} pts", **f}
                                       for f in ex.get("factors", [])],
                           "verdict": ("Proyeccion informativa: sin ventaja demostrada frente a la linea."
                                       if p.pick != "pick" else None)},
        model_market_gap=(None if p.gap_pp is None else p.gap_pp / 100),
        directional_disagreement=(None if p.market_probability is None else
                                  bool((p.model_probability >= 0.5) != (p.market_probability >= 0.5))),
        agreement_bucket=None, model_dispersion=None,
        market_signal=None, blend_policy={"modelo": 1.0, "mercado": 0.0, "nota": "mercado solo como referencia"},
        sources=["nfl-prediction-app (features, solo lectura)"] +
                (["the_odds_api"] if p.line_source == "the_odds_api" else
                 ["nflverse (linea del calendario)"] if p.line_source.startswith("nflverse") else []),
        model_version=p.model_version, prediction_timestamp=p.prediction_timestamp,
        prediction_id=p.id, version=p.version,
        status_note=(None if p.pick == "pick" else
                     "PROYECCION, no pick: sin ventaja demostrada frente a la linea (walk-forward 2020-2025)"),
        result=p.result, correct=p.correct)
    dc = p.data_completeness or {}
    c["data_completeness"] = {"complete": bool(dc.get("complete", True)), "missing": dc.get("missing", []),
                              "notes": []}
    c["extra"] = {"season": p.season, "week": p.week, "market_key": p.market_type,
                  "publish_pick": p.pick == "pick", "mode": p.pick,
                  "expected_value": p.expected_value, "expected_median": p.expected_median,
                  "expected_std": p.expected_std, "line": p.line, "line_source": p.line_source,
                  "line_timestamp": p.line_timestamp, "p_raw": p.p_over_raw,
                  "gap_pp": p.gap_pp, "cutoff_timestamp": p.cutoff_timestamp,
                  "actual_value": p.actual_value, "factors": ex.get("factors", []),
                  "market_caveat": g.get("reason"),
                  "risk_meaning": "probabilidad (calibrada) de que el lado mostrado falle"}
    return c


def games(date_from: str, date_to: str, limit: int = 400) -> list[dict]:
    init_db()
    with session_scope() as s:
        rows = s.execute(select(NflMarketPrediction).where(
            NflMarketPrediction.kickoff_utc >= datetime.fromisoformat(f"{date_from} 00:00:00"),
            NflMarketPrediction.kickoff_utc <= datetime.fromisoformat(f"{date_to} 23:59:59"),
            NflMarketPrediction.status != "superseded")
            .order_by(NflMarketPrediction.kickoff_utc.asc()).limit(limit)).scalars().all()
        return [_card(p) for p in rows]


def markets_status() -> dict:
    out = {}
    for m, g in gate().items():
        out[m] = {"enabled": g["enabled"], "publish_pick": g["publish_pick"], "mode": g["mode"],
                  "reason": g["reason"], "algorithm": g.get("algorithm"),
                  "n_features": g.get("n_features"), "families": g.get("families"),
                  "regression": g.get("regression"), "vs_line": g.get("vs_line"),
                  "calibration": g.get("calibration"), "key_numbers": g.get("key_numbers"),
                  "por_temporada": g.get("por_temporada"), "umbrales": g.get("umbrales"),
                  "families_dropped": g.get("families_dropped"), "distribution": g.get("distribution")}
    return out


def health() -> dict:
    """Indicadores para Data health por mercado NFL propio."""
    init_db()
    with session_scope() as s:
        out = {}
        for m in ("total", "spread"):
            preds = s.execute(select(NflMarketPrediction).where(
                NflMarketPrediction.market_type == m, NflMarketPrediction.status != "superseded")).scalars().all()
            mv = s.execute(select(NflModelVersion).where(NflModelVersion.market == m,
                                                          NflModelVersion.is_production.is_(True))).scalars().first()
            odds_n = s.execute(select(NflOdds).where(NflOdds.market == m)).scalars().all()
            last = max((p.prediction_timestamp for p in preds), default=None)
            g = gate()[m]
            estado = ("verde" if mv and preds and g["enabled"] else "amarillo" if mv else "rojo")
            out[m] = {"estado": estado, "modo": g["mode"], "modelo": mv.name if mv else None,
                      "entrenado": str(mv.created_at) if mv else None,
                      "features": len(mv.features) if mv else None,
                      "predicciones_activas": len(preds), "ultima_prediccion": str(last) if last else None,
                      "cuotas_snapshots": len(odds_n),
                      "mercado_disponible": any(p.market_probability is not None for p in preds),
                      "con_resultado": sum(1 for p in preds if p.result is not None)}
        return out
