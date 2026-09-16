"""Backtest walk-forward por partes reanudables.

    python -m MLB.backtests.run_wf --minutes 2.4      # avanza lo que pueda
    python -m MLB.backtests.run_wf --merge            # une las partes y guarda el run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from MLB.backtests.runner import BacktestConfig, _summarise, run  # noqa: E402
from MLB.database.models import BacktestRun                   # noqa: E402
from MLB.database.session import init_db, session_scope       # noqa: E402
from shared.paths import MLB_BACKTEST_DIR, MLB_PROCESSED_DIR  # noqa: E402
from shared.timeutil import utcnow                            # noqa: E402

PARTS = MLB_BACKTEST_DIR / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
FEATS = MLB_PROCESSED_DIR / "features.parquet"
MARKETS = ("moneyline", "f5_moneyline", "run_line", "total")


def say(m):
    print(f"{utcnow()} {m}", flush=True)


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def todo(first, last):
    X = pd.read_parquet(FEATS)
    seasons = sorted(X.season.unique())
    out = []
    for T in [s for s in seasons if first <= s <= last]:
        if len([s for s in seasons if s < T]) < 2:
            continue
        for m in MARKETS:
            if not (PARTS / f"{T}_{m}.json").exists():
                out.append((T, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=2.4)
    ap.add_argument("--first", type=int, default=2023)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.status:
        print(json.dumps(json_safe({"pendientes": todo(a.first, a.last),
                          "hechas": sorted(p.stem for p in PARTS.glob("*.json"))}), indent=1))
        return

    if a.merge:
        rows, elo = [], {}
        for p in sorted(PARTS.glob("*.json")):
            d = json.loads(p.read_text())
            rows.append(d["row"])
            elo[str(d["row"]["season"])] = d["elo"]
        per = {}
        for r in rows:
            per.setdefault(str(r["season"]), {})[r["market"]] = r
        summary = {"config": {"first_test": a.first, "last_test": a.last},
                   "elo": elo, "per_season": per, "summary": _summarise(rows),
                   "created_at": str(utcnow()),
                   "market_comparison": "no disponible: sin historico gratuito de cuotas MLB"}
        init_db()
        with session_scope() as s:
            br = BacktestRun(label="walkforward_v1", config=json_safe(summary["config"]),
                             summary=json_safe(summary))
            s.add(br); s.flush(); rid = br.id
        (MLB_BACKTEST_DIR / "walkforward_v1.json").write_text(
            json.dumps(json_safe(summary), indent=1))
        print(json.dumps({"backtest_run_id": rid,
                          "partes": len(rows),
                          "summary": json_safe(summary["summary"])}, indent=1))
        return

    X = pd.read_parquet(FEATS)
    deadline = time.time() + a.minutes * 60
    for T, m in todo(a.first, a.last):
        if time.time() > deadline:
            say("tiempo agotado; vuelve a ejecutar para continuar")
            return
        say(f"→ {T} / {m}")
        cfg = BacktestConfig(first_test=T, last_test=T, markets=(m,))
        res = run(X, cfg, progress=say)
        if not res["rows"]:
            say(f"  sin datos suficientes para {T}/{m}")
            (PARTS / f"{T}_{m}.json").write_text(json.dumps(
                {"row": {"season": T, "market": m, "n": 0, "skipped": True}, "elo": {}}))
            continue
        (PARTS / f"{T}_{m}.json").write_text(json.dumps(json_safe(
            {"row": res["rows"][0], "elo": res["elo"].get(T, {})}), indent=1))
        say(f"  guardado {T}/{m}")
    say("todas las partes completas; ejecuta --merge")


if __name__ == "__main__":
    main()
