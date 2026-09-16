"""Tarjetas NBA para el dashboard (misma forma que NFL/MLB)."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import func, select

from NBA.markets.db import (NbaGame, NbaModelVersion, NbaOdds, NbaPrediction, NbaSourceLog,
                            NbaTeamGameLog, init_db, session_scope)
from NBA.markets.gating import decide
from shared.schema import empty_card

LABEL = {"moneyline": "Moneyline", "spread": "Spread", "total": "Total"}
NAMES = {"ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets", "CHI": "Bulls", "CLE": "Cavaliers",
         "DAL": "Mavericks", "DEN": "Nuggets", "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
         "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat", "MIL": "Bucks", "MIN": "Timberwolves",
         "NOP": "Pelicans", "NYK": "Knicks", "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
         "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors", "UTA": "Jazz", "WAS": "Wizards"}
_GATE = None


def gate():
    global _GATE
    if _GATE is None:
        _GATE = decide()
    return _GATE


def _card(p: NbaPrediction) -> dict:
    g = gate().get(p.market_type, {})
    home_side = (p.final_probability or 0.5) >= 0.5
    side = lambda v: None if v is None else (v if home_side else 1 - v)
    ex = p.explanation or {}
    is_pick = p.status_market == "pick"
    c = empty_card(
        sport="NBA", game_id=p.game_id, game_date=str(p.game_date), start_utc=p.start_utc or datetime.combine(p.game_date, datetime.min.time()),
        status="scheduled" if p.result is None else "final",
        home=p.home_abbr, away=p.away_abbr, home_name=NAMES.get(p.home_abbr, p.home_abbr), away_name=NAMES.get(p.away_abbr, p.away_abbr),
        market=LABEL[p.market_type], selection=(p.selection if p.status_market != "no_pick" else f"NO PICK ({p.selection})"),
        line=p.line, model_probability=side(p.model_probability), market_probability=side(p.market_probability),
        ensemble_probability=p.pick_probability, elo_probability=None,
        confidence=p.confidence, upset_risk=round((1 - (p.pick_probability or 0.5)) * 100, 2),
        upset_label=("LOW RISK" if (p.pick_probability or 0) >= 0.62 else "MEDIUM RISK" if (p.pick_probability or 0) >= 0.545 else "HIGH RISK"),
        upset_explanation={"summary": ex.get("summary"), "flags": ex.get("flags", []),
                           "drivers": [{"text": f"{f['label']}", **f} for f in ex.get("factors", [])],
                           "verdict": (None if is_pick else "Proyeccion informativa: sin ventaja demostrada frente al mercado."
                                       if p.status_market == "projection" else "NO PICK: demasiado cerca del 50 %.")},
        model_market_gap=(None if p.gap_pp is None else p.gap_pp / 100),
        directional_disagreement=(None if p.market_probability is None else bool((p.model_probability >= 0.5) != (p.market_probability >= 0.5))),
        agreement_bucket=None, model_dispersion=None, market_signal=None,
        blend_policy={"modelo": 1.0, "mercado": 0.0, "nota": "mercado solo como referencia"},
        sources=["pbpstats (play-by-play oficial)"] + (["the_odds_api"] if p.line_source == "the_odds_api" or p.market_probability is not None else []),
        model_version=p.model_version, prediction_timestamp=p.prediction_timestamp, prediction_id=p.id, version=p.version,
        status_note=(None if is_pick else "NO PICK: por debajo del umbral validado (55 %)" if p.status_market == "no_pick"
                     else "PROYECCION, no pick: sin historico de cuotas para demostrar ventaja"),
        result=p.result, correct=p.correct)
    dc = p.data_completeness or {}
    c["data_completeness"] = {"complete": bool(dc.get("complete", True)), "missing": dc.get("missing", []), "notes": []}
    c["extra"] = {"season": p.season, "market_key": p.market_type, "publish_pick": is_pick, "mode": p.status_market,
                  "expected_value": p.expected_value, "expected_std": p.expected_std, "line": p.line,
                  "line_source": p.line_source, "line_timestamp": p.line_timestamp, "p_raw": p.p_raw,
                  "odds_home": p.odds_home, "odds_away": p.odds_away, "gap_pp": p.gap_pp,
                  "cutoff_timestamp": p.cutoff_timestamp, "actual_value": p.actual_value,
                  "factors": ex.get("factors", []), "market_caveat": g.get("reason"),
                  "threshold": g.get("threshold"), "risk_meaning": "probabilidad calibrada de que el lado mostrado falle"}
    return c


def games(date_from: str, date_to: str, limit: int = 600) -> list[dict]:
    init_db()
    with session_scope() as s:
        rows = s.execute(select(NbaPrediction).where(
            NbaPrediction.game_date >= date.fromisoformat(date_from), NbaPrediction.game_date <= date.fromisoformat(date_to),
            NbaPrediction.status != "superseded").order_by(NbaPrediction.start_utc.asc(), NbaPrediction.game_id).limit(limit)).scalars().all()
        return [_card(p) for p in rows]


def markets_status() -> dict:
    return {m: {k: v for k, v in g.items()} for m, g in gate().items()}


def health() -> dict:
    init_db()
    with session_scope() as s:
        n_games = s.scalar(select(func.count()).select_from(NbaGame)) or 0
        n_final = s.scalar(select(func.count()).select_from(NbaGame).where(NbaGame.status == "final")) or 0
        last_game = s.scalar(select(func.max(NbaGame.game_date)).where(NbaGame.status == "final"))
        n_logs = s.scalar(select(func.count()).select_from(NbaTeamGameLog)) or 0
        missing_logs = 2 * n_final - n_logs
        last_fetch = s.scalar(select(func.max(NbaSourceLog.retrieved_at)))
        errors = s.scalar(select(func.count()).select_from(NbaSourceLog).where(NbaSourceLog.status == "error")) or 0
        odds_n = s.scalar(select(func.count()).select_from(NbaOdds)) or 0
        odds_last = s.scalar(select(func.max(NbaOdds.available_at)))
        odds_games = s.scalar(select(func.count(func.distinct(NbaOdds.game_id)))) or 0
        out = {"partidos": n_games, "partidos_terminados": n_final, "ultimo_partido": str(last_game) if last_game else None,
               "logs_faltantes": int(max(missing_logs, 0)), "ultima_descarga": str(last_fetch) if last_fetch else None,
               "errores_descarga": errors, "cuotas_snapshots": odds_n, "cuotas_partidos": odds_games,
               "cuotas_ultima": str(odds_last) if odds_last else None,
               "cuotas_historicas": "ninguna: solo snapshots propios desde la activacion",
               "lesiones_timestamp": "sin historico con timestamp (excluidas del modelo)",
               "quintetos_timestamp": "sin historico con timestamp (excluidos del modelo)",
               "leakage": "features as-of por dia; tests automaticos", "mercados": {}}
        for m in ("moneyline", "spread", "total"):
            mv = s.execute(select(NbaModelVersion).where(NbaModelVersion.market == m, NbaModelVersion.is_production.is_(True))).scalars().first()
            preds = s.execute(select(NbaPrediction).where(NbaPrediction.market_type == m, NbaPrediction.status != "superseded")).scalars().all()
            g = gate()[m]
            estado = "OK" if (mv and g["enabled"]) else "WARNING" if mv else "BLOCKED"
            if g["mode"] == "blocked":
                estado = "BLOCKED"
            out["mercados"][m] = {"estado": estado, "modo": g["mode"], "modelo": mv.name if mv else None,
                                  "entrenado": str(mv.created_at) if mv else None, "features": len(mv.features) if mv else None,
                                  "predicciones_activas": len(preds), "con_resultado": sum(1 for p in preds if p.result is not None),
                                  "ultima_prediccion": str(max((p.prediction_timestamp for p in preds), default=None))}
        stale = last_game is not None and (date.today() - last_game).days > 3 and 10 <= date.today().month or (date.today().month <= 6 and last_game is not None and (date.today() - last_game).days > 3)
        out["estado_global"] = "WARNING" if (errors > 0 or missing_logs > 0 or stale) else "OK"
        return out
