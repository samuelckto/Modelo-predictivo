"""Ciclo diario MLB: actualizar datos -> reconstruir features -> predecir -> puntuar.

Cada paso queda en `pipeline_runs`. Si un paso falla se registra y se continua,
marcando lo que no se pudo obtener. Nunca se rellena con datos inventados.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import select                                     # noqa: E402

from MLB.database.models import Game, PipelineRun, Prediction     # noqa: E402
from MLB.database.session import init_db, session_scope           # noqa: E402
from MLB.ingest.boxscores import ingest_boxscores, rebuild_bullpen_usage  # noqa: E402
from MLB.ingest.odds import ingest as ingest_odds                 # noqa: E402
from MLB.ingest.statcast import ingest_parallel                   # noqa: E402
from MLB.ingest.statsapi import (FINAL_STATES, ingest_injuries, ingest_lineups,
                                 ingest_players, ingest_schedule)                 # noqa: E402
from shared.timeutil import json_safe, utcnow                     # noqa: E402


def _season(today: date) -> int:
    return today.year


def run(days_ahead: int = 7, days_back: int = 3, skip_statcast: bool = False,
        progress=print) -> dict:
    init_db()
    today = date.today()
    season = _season(today)
    a, b = str(today - timedelta(days=days_back)), str(today + timedelta(days=days_ahead))
    res = {"date": str(today), "season": season, "window": [a, b], "steps": {}}
    with session_scope() as s:
        pr = PipelineRun(kind="daily", trigger="cli", started_at=utcnow(),
                         detail={"window": [a, b]})
        s.add(pr); s.flush(); rid = pr.id

    def step(name, fn):
        try:
            with session_scope() as s:
                res["steps"][name] = fn(s)
        except Exception as e:                                     # noqa: BLE001
            res["steps"][name] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                                  "traceback": traceback.format_exc()[-400:]}
        progress(f"  {name}: {res['steps'][name].get('status')}")

    step("calendario", lambda s: ingest_schedule(s, season, a, b))
    step("jugadores", lambda s: ingest_players(s, season))
    step("lesiones", lambda s: ingest_injuries(s, season))
    step("cuotas", lambda s: ingest_odds(s))

    def _odds_meta(s):
        from MLB.ingest.odds import mark_closing, snapshot_stats
        mc = mark_closing(s)
        return {"status": "ok", **mc, "historico": snapshot_stats(s)}
    step("mercado_historico", _odds_meta)

    def _finished(s):
        pks = list(s.execute(select(Game.id).where(
            Game.season == season, Game.game_date >= a, Game.game_date <= b,
            Game.status.in_(FINAL_STATES))).scalars().all())
        if not pks:
            return {"status": "ok", "games": 0, "note": "sin partidos terminados en la ventana"}
        r = ingest_boxscores(s, pks, workers=4)
        r["status"] = "ok"
        return r
    step("resultados", _finished)
    step("bullpen", lambda s: rebuild_bullpen_usage(s, season))

    def _lineups(s):
        pks = list(s.execute(select(Game.id).where(
            Game.game_date >= str(today), Game.game_date <= str(today + timedelta(days=1)),
            ~Game.status.in_(FINAL_STATES))).scalars().all())
        if not pks:
            return {"status": "ok", "games": 0}
        r = ingest_lineups(s, pks[:40])
        r["status"] = "ok"
        return r
    step("alineaciones", _lineups)

    if not skip_statcast:
        def _sc(s):
            from MLB.ingest.fill_statcast import missing_ranges
            miss = missing_ranges(s, season, 3)[-4:]
            if not miss:
                return {"status": "ok", "ranges": 0}
            r = ingest_parallel(s, season, miss, workers=2)
            r["status"] = "ok"
            return r
        step("statcast", _sc)

    try:
        from MLB.features.builder import build
        X = build(sorted({season - 5, season - 4, season - 3, season - 2, season - 1, season}))
        res["steps"]["features"] = {"status": "ok", "rows": int(len(X)),
                                    "cols": int(X.shape[1])}
        try:
            from MLB.engine import detail
            detail.invalidate()
        except Exception:                                       # noqa: BLE001
            pass
    except Exception as e:                                          # noqa: BLE001
        res["steps"]["features"] = {"status": "error", "error": str(e)}
    progress(f"  features: {res['steps']['features'].get('status')}")

    def _pred(s):
        from MLB.engine.predict import predict_range
        r = predict_range(s, str(today), b, reason="ciclo diario")
        return r
    # el calendario trae los abridores anunciados: si fallo al principio (DNS),
    # se vuelve a intentar aqui, ya con la red menos cargada
    if res["steps"].get("calendario", {}).get("status") != "ok":
        progress("  calendario fallo al inicio: reintentando…")
        step("calendario_reintento", lambda s: ingest_schedule(s, season, a, b))
    step("predicciones", _pred)
    step("resultados_predicciones", lambda s: score_predictions(s))

    with session_scope() as s:
        p = s.get(PipelineRun, rid)
        p.status = "ok" if all(v.get("status") != "error" for v in res["steps"].values()) \
            else "partial"
        p.finished_at, p.detail = utcnow(), json_safe(res)
    res["pipeline_run_id"] = rid
    return res


def score_predictions(session) -> dict:
    """Marca acierto/fallo de las predicciones cuyos partidos ya terminaron."""
    preds = session.execute(select(Prediction).where(
        Prediction.result.is_(None), Prediction.status != "superseded")).scalars().all()
    if not preds:
        return {"status": "ok", "scored": 0}
    gids = {p.game_id for p in preds}
    games = {g.id: g for g in session.execute(
        select(Game).where(Game.id.in_(gids))).scalars().all()}
    n = 0
    for p in preds:
        g = games.get(p.game_id)
        if not g or g.status not in FINAL_STATES or g.home_score is None:
            continue
        home_pick = (p.selection or "").startswith(g.home_abbr or "\x00")
        if p.market == "moneyline":
            hw = g.home_score > g.away_score
            ok = hw if home_pick else not hw
        elif p.market == "f5_moneyline":
            if g.home_score_f5 is None or g.away_score_f5 is None:
                continue
            if g.home_score_f5 == g.away_score_f5:
                p.result, p.correct = "push", None
                n += 1
                continue
            hw = g.home_score_f5 > g.away_score_f5
            ok = hw if home_pick else not hw
        elif p.market == "run_line":
            m = g.home_score - g.away_score
            ok = (m >= 2) if home_pick else (m <= 1)
        elif p.market == "total":
            if p.line is None:
                continue
            tot = g.home_score + g.away_score
            if tot == p.line:
                p.result, p.correct = "push", None
                n += 1
                continue
            over_pick = (p.selection or "").lower().startswith("over")
            ok = (tot > p.line) if over_pick else (tot < p.line)
        else:
            continue
        p.result = "win" if ok else "loss"
        p.correct = bool(ok)
        n += 1
    session.flush()
    return {"status": "ok", "scored": n}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-ahead", type=int, default=7)
    ap.add_argument("--days-back", type=int, default=3)
    ap.add_argument("--skip-statcast", action="store_true")
    a = ap.parse_args()
    print(json.dumps(json_safe(run(a.days_ahead, a.days_back, a.skip_statcast)),
                     indent=1, default=str))


def refresh_results(days_back: int = 1) -> dict:
    """Baja marcadores finales recientes y califica predicciones. Rapido; sin
    features ni reentrenamiento. Lo usan `spc.py score` y el auto-scoring del servidor."""
    from datetime import date, timedelta
    from MLB.database.session import init_db, session_scope
    from MLB.ingest.statsapi import ingest_schedule
    init_db()
    hoy = date.today()
    x, y = str(hoy - timedelta(days=days_back)), str(hoy)
    with session_scope() as s:
        r = ingest_schedule(s, hoy.year, x, y)
        sc = score_predictions(s)
    try:
        from MLB.engine import detail
        detail.invalidate()
    except Exception:
        pass
    return {"desde": x, "hasta": y, "partidos": r.get("games"), "status": r.get("status"),
            "calificadas": sc.get("scored", 0)}
