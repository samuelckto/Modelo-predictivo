"""Ciclo de tenis: partidos nuevos del archivo -> features -> calendario y cuotas
-> predicciones -> calificacion."""
from __future__ import annotations

import time
from datetime import date

from TENIS.markets.db import TenisPipelineRun, init_db, session_scope
from TENIS.markets.features import build
from TENIS.markets.ingest import run as ingest_run
from TENIS.markets.odds import ingest as ingest_odds, mark_closing
from TENIS.markets.predict import predict_upcoming, score
from shared.timeutil import json_safe, utcnow


def run(days_ahead: int = 3, with_odds: bool = True, rebuild_features: bool = True,
        budget_s: float = 60.0) -> dict:
    init_db()
    t0 = time.time(); out = {"started_at": utcnow()}
    with session_scope() as s:
        r = TenisPipelineRun(kind="tenis-cycle", started_at=utcnow(), status="running"); s.add(r); s.flush()
    out["archivo"] = ingest_run([date.today().year], budget_s=budget_s, progress=lambda m: None)
    with session_scope() as s:
        if with_odds:
            out["calendario_y_cuotas"] = ingest_odds(s)
            out["cierre_marcado"] = mark_closing(s)
    X = build(save=True) if rebuild_features else None
    if X is not None:
        out["features"] = list(X.shape)
    with session_scope() as s:
        out["predicciones"] = predict_upcoming(s, days_ahead, "tenis-cycle", X=X)
        out["calificadas"] = score(s, X)
        r = s.query(TenisPipelineRun).order_by(TenisPipelineRun.id.desc()).first()
        r.finished_at = utcnow(); r.status = "ok"; r.summary = json_safe(out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def refresh_results() -> dict:
    init_db()
    with session_scope() as s:
        return score(s)
