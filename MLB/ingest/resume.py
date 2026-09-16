"""Ingesta reanudable con limite de tiempo.

    python -m MLB.ingest.resume --minutes 9

Se puede llamar tantas veces como haga falta: cada ejecucion continua donde
quedo la anterior. Es idempotente (nunca duplica filas) y todo queda en
`data_source_logs`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import func, select                                   # noqa: E402

from MLB.database.models import (BatterGameLog, Game, PitcherGameLog,
                                 StatcastAgg)                          # noqa: E402
from MLB.database.session import init_db, session_scope                # noqa: E402
from MLB.ingest.boxscores import ingest_boxscores, rebuild_bullpen_usage   # noqa: E402
from MLB.ingest.statcast import ingest_parallel, pending_chunks        # noqa: E402
from MLB.ingest.statsapi import FINAL_STATES, ingest_schedule, ingest_teams  # noqa: E402
from shared.timeutil import utcnow                                     # noqa: E402

LOG = Path(__file__).resolve().parents[2] / "logs" / "mlb_ingest.log"


def say(m):
    line = f"{utcnow()} {m}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def missing_boxscores(session, season):
    got = set(session.execute(select(PitcherGameLog.game_id.distinct())
                              .where(PitcherGameLog.season == season)).scalars().all())
    todo = session.execute(select(Game.id).where(
        Game.season == season, Game.status.in_(FINAL_STATES))).scalars().all()
    return [g for g in todo if g not in got]


def progress_report(seasons):
    init_db()
    with session_scope() as s:
        rows = []
        for y in seasons:
            g = s.scalar(select(func.count()).select_from(Game).where(Game.season == y)) or 0
            gf = s.scalar(select(func.count()).select_from(Game).where(
                Game.season == y, Game.status.in_(FINAL_STATES))) or 0
            bx = len(set(s.execute(select(PitcherGameLog.game_id.distinct())
                                   .where(PitcherGameLog.season == y)).scalars().all()))
            sc = s.scalar(select(func.count()).select_from(StatcastAgg)
                          .where(StatcastAgg.season == y)) or 0
            pend = len(pending_chunks(s, y))
            rows.append({"season": y, "games": g, "final": gf, "boxscores": bx,
                         "statcast_rows": sc, "statcast_chunks_pending": pend})
        return rows


def run(seasons, minutes: float, workers: int, sc_workers: int, step_days: int) -> dict:
    init_db()
    deadline = time.time() + minutes * 60
    out = {"seasons": {}, "stopped_by_time": False}
    for season in seasons:
        r = out["seasons"].setdefault(season, {})
        with session_scope() as s:
            n = s.scalar(select(func.count()).select_from(Game).where(Game.season == season)) or 0
            # una temporada completa ronda los 2.400 partidos; por debajo de 1.500
            # el calendario esta incompleto y se vuelve a pedir (es idempotente).
            if n < 1500:
                say(f"[{season}] equipos y calendario…")
                ingest_teams(s, season)
                r["schedule"] = ingest_schedule(s, season)
                say(f"[{season}] {r['schedule']}")
        while True:
            if time.time() > deadline:
                out["stopped_by_time"] = True
                return out
            with session_scope() as s:
                todo = missing_boxscores(s, season)
            if not todo:
                break
            batch = todo[:400]
            with session_scope() as s:
                got = ingest_boxscores(s, batch, workers=workers)
            r["boxscores"] = r.get("boxscores", 0) + got["games_ok"]
            say(f"[{season}] boxscores +{got['games_ok']} (faltan {len(todo)-len(batch)})")
        with session_scope() as s:
            if missing_boxscores(s, season) == []:
                r["bullpen"] = rebuild_bullpen_usage(s, season)
        while True:
            if time.time() > deadline:
                out["stopped_by_time"] = True
                return out
            with session_scope() as s:
                pend = pending_chunks(s, season, step_days=step_days)
            if not pend:
                break
            with session_scope() as s:
                got = ingest_parallel(s, season, pend[:sc_workers * 3], workers=sc_workers,
                                      progress=say)
            r["statcast_rows"] = r.get("statcast_rows", 0) + got["rows"]
            say(f"[{season}] statcast +{got['rows']} filas (bloques restantes {len(pend)-len(pend[:sc_workers*3])})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--minutes", type=float, default=9.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sc-workers", type=int, default=6)
    ap.add_argument("--step-days", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(json.dumps(progress_report(a.seasons), indent=1))
    else:
        print(json.dumps(run(a.seasons, a.minutes, a.workers, a.sc_workers, a.step_days),
                         indent=1, default=str))
