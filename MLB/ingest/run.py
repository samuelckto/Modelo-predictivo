"""Orquestador de la ingesta historica MLB.

    python -m MLB.ingest.run --seasons 2021 2022 2023 2024 2025

Cada paso escribe en `data_source_logs` y en `pipeline_runs`. Si un paso falla,
se registra y se continua con el siguiente: nunca se rellena con datos falsos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import select                                        # noqa: E402

from MLB.database.models import Game, PipelineRun                    # noqa: E402
from MLB.database.session import init_db, session_scope              # noqa: E402
from MLB.ingest.boxscores import ingest_boxscores, rebuild_bullpen_usage  # noqa: E402
from MLB.ingest.statcast import ingest_season as statcast_season     # noqa: E402
from MLB.ingest.statsapi import (FINAL_STATES, ingest_schedule, ingest_teams)  # noqa: E402
from shared.timeutil import utcnow                                   # noqa: E402

LOG = Path(__file__).resolve().parents[2] / "logs" / "mlb_ingest.log"


def say(msg: str) -> None:
    line = f"{utcnow()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(seasons: list[int], skip_statcast: bool = False, workers: int = 8) -> dict:
    init_db()
    out = {}
    for season in seasons:
        t0 = time.time()
        with session_scope() as s:
            pr = PipelineRun(kind="ingest_history", trigger="cli",
                             detail={"season": season}, started_at=utcnow())
            s.add(pr); s.flush(); run_id = pr.id
        try:
            with session_scope() as s:
                say(f"[{season}] equipos…")
                r_teams = ingest_teams(s, season)
                say(f"[{season}] calendario…")
                r_sched = ingest_schedule(s, season)
                say(f"[{season}] calendario: {r_sched}")

            with session_scope() as s:
                pks = list(s.execute(select(Game.id).where(
                    Game.season == season, Game.status.in_(FINAL_STATES))).scalars().all())
            say(f"[{season}] boxscores de {len(pks)} partidos…")
            r_box = {"games_ok": 0}
            for i in range(0, len(pks), 400):                 # commits parciales
                with session_scope() as s:
                    part = ingest_boxscores(s, pks[i:i + 400], workers=workers,
                                            progress=lambda m: say(f"[{season}] {m}"))
                r_box["games_ok"] += part["games_ok"]
                say(f"[{season}] boxscores {min(i+400,len(pks))}/{len(pks)} {part}")

            with session_scope() as s:
                r_bp = rebuild_bullpen_usage(s, season)
                say(f"[{season}] bullpen: {r_bp}")

            r_sc = {"status": "skipped"}
            if not skip_statcast:
                say(f"[{season}] statcast…")
                with session_scope() as s:
                    r_sc = statcast_season(s, season, progress=say)
                say(f"[{season}] statcast: {r_sc['pitches']} lanzamientos, {r_sc['rows']} filas")

            res = {"teams": r_teams, "schedule": r_sched, "boxscores": r_box,
                   "bullpen": r_bp, "statcast": r_sc,
                   "minutes": round((time.time() - t0) / 60, 1)}
            with session_scope() as s:
                p = s.get(PipelineRun, run_id)
                p.status, p.finished_at, p.detail = "ok", utcnow(), res
            out[season] = res
            say(f"[{season}] LISTO en {res['minutes']} min")
        except Exception as e:                                  # noqa: BLE001
            with session_scope() as s:
                p = s.get(PipelineRun, run_id)
                p.status, p.finished_at, p.error = "failed", utcnow(), f"{type(e).__name__}: {e}"
            out[season] = {"status": "failed", "error": str(e)}
            say(f"[{season}] FALLO: {type(e).__name__}: {e}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--skip-statcast", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    say(f"=== ingesta MLB {a.seasons} ===")
    print(json.dumps(run(a.seasons, a.skip_statcast, a.workers), indent=1, default=str))
