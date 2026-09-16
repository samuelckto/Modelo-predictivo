"""Adaptador del sistema NFL existente. SOLO LECTURA.

El motor NFL (nfl-prediction-app) no se mueve ni se modifica. Este adaptador abre
su SQLite en modo `ro` a nivel de driver, de manera que cualquier intento de
escritura falla con un error de sqlite, no con una convencion nuestra.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from shared.paths import NFL_HOME
from shared.schema import empty_card, risk_label

TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
    "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
    "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}


def db_path() -> Path:
    return NFL_HOME / "database" / "nflpred.sqlite3"


def available() -> tuple[bool, str]:
    p = db_path()
    if not p.exists():
        return False, f"no se encontro la base NFL en {p} (define NFL_HOME en .env)"
    return True, str(p)


def _conn() -> sqlite3.Connection:
    """Conexion de solo lectura reforzada por el propio sqlite (mode=ro)."""
    ok, msg = available()
    if not ok:
        raise FileNotFoundError(msg)
    c = sqlite3.connect(f"file:{db_path()}?mode=ro&immutable=0", uri=True, timeout=15)
    c.row_factory = sqlite3.Row
    return c


def _j(v, default=None):
    if v in (None, ""):
        return default
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return default


def games(date_from: str, date_to: str, limit: int = 400) -> list[dict]:
    """Predicciones activas de NFL cuyo kickoff cae en [date_from, date_to] (fechas UTC)."""
    ok, _ = available()
    if not ok:
        return []
    q = """
      select p.*, g.kickoff_utc as g_kick, g.gameday, g.stadium, g.roof, g.home_score, g.away_score
      from predictions p left join games g on g.game_id = p.game_id
      where p.is_active = 1
        and coalesce(p.kickoff_utc, g.kickoff_utc) >= :a
        and coalesce(p.kickoff_utc, g.kickoff_utc) <= :b
      order by coalesce(p.kickoff_utc, g.kickoff_utc) asc
      limit :lim
    """
    with _conn() as c:
        rows = c.execute(q, {"a": f"{date_from} 00:00:00", "b": f"{date_to} 23:59:59",
                             "lim": limit}).fetchall()
    return [_row_to_card(r) for r in rows]


def _row_to_card(r: sqlite3.Row) -> dict:
    d = dict(r)
    kick = d.get("kickoff_utc") or d.get("g_kick")
    pick = d.get("pick")
    home = d.get("home_team")
    is_home_pick = pick == home
    pol = _j(d.get("blend_policy"), {}) or {}
    miss = []
    if d.get("home_prob_market") is None:
        miss.append("cuotas de mercado no disponibles para este partido")
    card = empty_card(
        sport="NFL",
        game_id=d.get("game_id"),
        game_date=(kick or "")[:10] or d.get("gameday"),
        start_utc=kick,
        status=("final" if d.get("home_score") is not None else "scheduled"),
        home=home, away=d.get("away_team"),
        home_name=TEAM_NAMES.get(home, home), away_name=TEAM_NAMES.get(d.get("away_team")),
        venue=d.get("stadium"),
        market="moneyline", selection=pick, line=None,
        model_probability=_side(d.get("home_prob_ensemble"), is_home_pick),
        market_probability=_side(d.get("home_prob_market"), is_home_pick),
        ensemble_probability=d.get("pick_prob"),
        elo_probability=_side(d.get("home_prob_elo"), is_home_pick),
        confidence=d.get("confidence"),
        upset_risk=d.get("upset_risk"),
        upset_label=d.get("upset_label") or risk_label(d.get("upset_risk")),
        upset_explanation=_j(d.get("upset_explanation")),
        model_market_gap=d.get("market_model_gap"),
        directional_disagreement=(None if d.get("directional_disagreement") is None
                                  else bool(d.get("directional_disagreement"))),
        agreement_bucket=d.get("agreement_bucket"),
        model_dispersion=d.get("model_dispersion"),
        market_signal=d.get("market_signal"),
        blend_policy=pol,
        sources=_j(d.get("sources"), []) or [],
        model_version=d.get("model_version_label"),
        prediction_timestamp=d.get("predicted_at"),
        prediction_id=d.get("id"),
        version=1,
        status_note=d.get("status_note"),
        result=("win" if d.get("correct") == 1 else "loss" if d.get("correct") == 0 else None),
        correct=(None if d.get("correct") is None else bool(d.get("correct"))),
    )
    card["data_completeness"] = {"complete": not miss, "missing": miss, "notes": []}
    card["extra"] = {"season": d.get("season"), "week": d.get("week"),
                     "fantasy_points": d.get("points"),
                     "n_games_week": d.get("n_games_week"),
                     "roof": d.get("roof"),
                     "home_qb": d.get("home_qb_name"), "away_qb": d.get("away_qb_name"),
                     "factors": _j(d.get("factors"), []),
                     "data_cutoff_utc": d.get("data_cutoff_utc")}
    return card


def _side(home_prob, is_home_pick):
    if home_prob is None or is_home_pick is None:
        return None
    return float(home_prob) if is_home_pick else 1.0 - float(home_prob)


def performance() -> dict:
    """Desempeno historico real leido de la base NFL. Sin numeros inventados."""
    ok, msg = available()
    if not ok:
        return {"available": False, "reason": msg}
    with _conn() as c:
        row = c.execute("""
          select count(*) n, sum(case when correct=1 then 1 else 0 end) hits,
                 sum(coalesce(points_earned,0)) pts, sum(coalesce(points,0)) pts_max
          from predictions where is_active=1 and correct is not null
        """).fetchone()
        bt = c.execute("select id,label,created_at,summary from backtest_runs "
                       "order by id desc limit 1").fetchone()
        mv = c.execute("select name,version,created_at,metrics from model_versions "
                       "where is_production=1 order by id desc limit 1").fetchone()
        src = c.execute("select source, max(fetched_at) last, status from data_sources_log "
                        "group by source order by last desc").fetchall()
    n = row["n"] or 0
    return {
        "available": True,
        "graded_predictions": n,
        "accuracy": (row["hits"] / n) if n else None,
        "fantasy_points": row["pts"], "fantasy_points_max": row["pts_max"],
        "fantasy_points_pct": (row["pts"] / row["pts_max"]) if row["pts_max"] else None,
        "latest_backtest": ({"id": bt["id"], "label": bt["label"], "created_at": bt["created_at"],
                             "summary": _j(bt["summary"], {})} if bt else None),
        "production_model": ({"name": mv["name"], "version": mv["version"],
                              "created_at": mv["created_at"],
                              "metrics": _j(mv["metrics"], {})} if mv else None),
        "sources": [{"source": s["source"], "last_fetch": s["last"], "status": s["status"]}
                    for s in src],
    }


def read_only_check() -> str:
    """Demuestra que la conexion no puede escribir. Lo usan los tests de aislamiento."""
    try:
        with _conn() as c:
            c.execute("create table _spc_probe (x int)")
        return "ERROR: la conexion permitio escribir"
    except sqlite3.OperationalError as e:
        return f"ok: sqlite rechazo la escritura ({e})"
