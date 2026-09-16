"""Rellena Statcast comparando los dias con partido contra los dias ya agregados.

Mas robusto que fiarse del registro de bloques: mira el dato real que falta.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import select                                    # noqa: E402

from MLB.database.models import Game, StatcastAgg               # noqa: E402
from MLB.database.session import init_db, session_scope         # noqa: E402
from MLB.ingest.resume import say                               # noqa: E402
from MLB.ingest.statcast import ingest_parallel                 # noqa: E402
from MLB.ingest.statsapi import FINAL_STATES                    # noqa: E402


def missing_ranges(session, season: int, step: int = 8) -> list[tuple[str, str]]:
    days = set(session.execute(select(Game.game_date.distinct()).where(
        Game.season == season, Game.status.in_(FINAL_STATES))).scalars().all())
    have = set(session.execute(select(StatcastAgg.game_date.distinct()).where(
        StatcastAgg.season == season)).scalars().all())
    miss = sorted(d for d in days if d and d not in have)
    out, cur = [], []
    for d in miss:
        if not cur:
            cur = [d]
            continue
        gap = (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(cur[-1], "%Y-%m-%d")).days
        if gap <= step and len(cur) < step:
            cur.append(d)
        else:
            out.append((cur[0], cur[-1])); cur = [d]
    if cur:
        out.append((cur[0], cur[-1]))
    return out


def run(seasons, minutes, workers, step):
    init_db()
    deadline = time.time() + minutes * 60
    res = {}
    for y in seasons:
        with session_scope() as s:
            miss = missing_ranges(s, y, step)
        res[y] = {"missing_ranges": len(miss)}
        while miss and time.time() < deadline:
            take = miss[:workers]  # lotes cortos: cada uno hace commit y sobrevive al corte
            with session_scope() as s:
                got = ingest_parallel(s, y, take, workers=workers, progress=say)
            res[y]["rows"] = res[y].get("rows", 0) + got["rows"]
            with session_scope() as s:
                miss = missing_ranges(s, y, step)
            say(f"[{y}] faltan {len(miss)} rangos de statcast")
        res[y]["still_missing"] = len(miss)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--minutes", type=float, default=2.4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--step", type=int, default=8)
    a = ap.parse_args()
    print(json.dumps(run(a.seasons, a.minutes, a.workers, a.step), indent=1))
