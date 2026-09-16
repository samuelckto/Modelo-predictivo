"""Ciclo NBA: partidos y logs nuevos (pbpstats) -> features -> cuotas/calendario ->
predicciones -> calificacion. Todo dentro del arbol NBA."""
from __future__ import annotations

import time
from datetime import date

from NBA.markets.db import NbaPipelineRun, init_db, session_scope
from NBA.markets.features import build
from NBA.markets.ingest import run as ingest_run
from NBA.markets.odds import ingest as ingest_odds, mark_closing
from NBA.markets.predict import predict_upcoming, score
from shared.timeutil import json_safe, utcnow


def current_season(today: date | None = None) -> int:
    d = today or date.today()
    return d.year + 1 if d.month >= 9 else d.year


def run(days_ahead: int = 3, with_odds: bool = True, budget_s: float = 120.0, reason: str = "nba-cycle") -> dict:
    init_db()
    t0 = time.time(); out = {"started_at": utcnow()}
    with session_scope() as s:
        run_ = NbaPipelineRun(kind="nba-cycle", started_at=utcnow(), status="running"); s.add(run_); s.flush()
    out["ingesta"] = ingest_run([current_season()], budget_s=budget_s, progress=lambda m: None)
    with session_scope() as s:
        if with_odds:
            out["cuotas"] = ingest_odds(s)
            out["cierre_marcado"] = mark_closing(s)
    X = build(save=True)
    out["features"] = list(X.shape)
    with session_scope() as s:
        out["predicciones"] = predict_upcoming(s, days_ahead, reason)
        out["calificadas"] = score(s, X)
        run_ = s.query(NbaPipelineRun).order_by(NbaPipelineRun.id.desc()).first()
        run_.finished_at = utcnow(); run_.status = "ok"; run_.summary = json_safe(out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def refresh_results() -> dict:
    """Solo califica con las features actuales (rapido). Lo usa el auto-scoring."""
    init_db()
    with session_scope() as s:
        return score(s)
