"""Ciclo de los mercados NFL propios: cuotas -> predicciones -> calificacion."""
from __future__ import annotations

from NFL.markets.data import load
from NFL.markets.db import NflPipelineRun, init_db, session_scope
from NFL.markets.odds import ingest, mark_closing
from NFL.markets.predict import predict_upcoming, score, upcoming
from shared.timeutil import json_safe, utcnow


def _games_for_odds(X):
    now = utcnow()
    sub = X[(X.kickoff_utc > now - __import__("pandas").Timedelta(days=2)) &
            (X.kickoff_utc < now + __import__("pandas").Timedelta(days=12))]
    return [{"game_id": r.game_id, "home_team": r.home_team, "away_team": r.away_team,
             "kickoff_utc": r.kickoff_utc.to_pydatetime()} for r in sub.itertuples()]


def run(days_ahead: int = 10, with_odds: bool = True, reason: str = "nfl-cycle") -> dict:
    init_db()
    X = load()
    out = {"started_at": utcnow()}
    with session_scope() as s:
        run_ = NflPipelineRun(kind="nfl-markets", started_at=utcnow(), status="running")
        s.add(run_); s.flush()
        if with_odds:
            out["cuotas"] = ingest(s, _games_for_odds(X))
            out["cierre_marcado"] = mark_closing(s, _games_for_odds(X))
        out["predicciones"] = predict_upcoming(s, days_ahead, reason)
        out["calificadas"] = score(s, X)
        run_.finished_at = utcnow(); run_.status = "ok"; run_.summary = json_safe(out)
    out["finished_at"] = utcnow()
    return out


def refresh_results() -> dict:
    """Solo califica (rapido). Lo usa el auto-scoring del servidor."""
    init_db()
    out = {}
    try:
        from NFL.espn_scorer import score_nfl
        out["espn"] = score_nfl()
    except Exception as e:
        out["espn_error"] = str(e)
    with session_scope() as s:
        out["markets"] = score(s)
    return out
