"""Metricas avanzadas de contacto desde Baseball Savant (Statcast).

Se descarga por bloques de dias y se AGREGA al vuelo por jugador y fecha; el CSV
crudo no se guarda (serian varios GB) pero si su checksum, el rango y el numero
de filas, para que la descarga sea auditable.

`available_at` = dia siguiente a las 06:00 UTC. Es deliberadamente conservador:
garantiza que las metricas de un dia jamas se usen para predecir un partido de
ese mismo dia. Queda declarado aqui y en el README.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from MLB.database.models import StatcastAgg
from MLB.ingest.http import get_csv, log_fetch

SOURCE = "baseball_savant"

# Baseball Savant devuelve como maximo 25.000 filas por consulta CSV. Si un
# bloque llega justo a ese numero el dato esta TRUNCADO y seria falso guardarlo:
# se parte el rango en dos y se vuelve a pedir. Verificado el 2026-09-07:
# 2 dias -> 5.997 filas, 4 dias -> 15.140, 8 dias -> 25.000 (tope exacto).
SAVANT_ROW_CAP = 25000
BASE = "https://baseballsavant.mlb.com/statcast_search/csv"

NEEDED = ["game_date", "game_pk", "pitcher", "batter", "events", "description",
          "release_speed", "release_spin_rate", "launch_speed", "launch_angle",
          "launch_speed_angle", "estimated_ba_using_speedangle",
          "estimated_woba_using_speedangle", "estimated_slg_using_speedangle",
          "pitch_name", "stand", "p_throws", "inning", "type"]

SWING_MISS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED = {"called_strike"}


def url_for(a: str, b: str, season: int) -> str:
    return (f"{BASE}?all=true&hfSea={season}%7C&player_type=pitcher"
            f"&game_date_gt={a}&game_date_lt={b}&type=details"
            f"&min_pitches=0&min_results=0&group_by=name&sort_col=pitches"
            f"&player_event_sort=api_p_release_speed&sort_order=desc")


def _agg(df: pd.DataFrame, key: str, scope: str) -> pd.DataFrame:
    d = df.copy()
    d["is_swstr"] = d.description.isin(SWING_MISS)
    d["is_csw"] = d.description.isin(SWING_MISS | CALLED)
    d["is_bb"] = d.launch_speed.notna()
    d["hard"] = d.launch_speed >= 95
    d["barrel"] = d.launch_speed_angle == 6
    g = d.groupby([key, "game_date"], as_index=False).agg(
        pitches=("description", "size"),
        batted_balls=("is_bb", "sum"),
        hard_hit=("hard", "sum"),
        barrels=("barrel", "sum"),
        xba=("estimated_ba_using_speedangle", "mean"),
        xslg=("estimated_slg_using_speedangle", "mean"),
        xwoba=("estimated_woba_using_speedangle", "mean"),
        exit_velocity=("launch_speed", "mean"),
        launch_angle=("launch_angle", "mean"),
        swstr=("is_swstr", "mean"),
        csw=("is_csw", "mean"),
        velocity=("release_speed", "mean"),
        spin=("release_spin_rate", "mean"))
    g = g.rename(columns={key: "player_id"})
    g["scope"] = scope
    return g


def ingest_chunk(session, a: str, b: str, season: int) -> dict:
    df, f = get_csv(url_for(a, b, season), "MLB", SOURCE, "statcast", timeout=300)
    f.notes = (f.notes or "") + f" rango {a}..{b}"
    log_fetch(session, f)
    if df is None or len(df) == 0:
        return {"status": f.status, "rows": 0, "range": f"{a}..{b}", "error": f.error}
    miss = [c for c in NEEDED if c not in df.columns]
    if miss:                                   # la fuente cambio de formato: se registra
        f.status, f.error = "error", f"faltan columnas: {miss}"
        log_fetch(session, f)
        return {"status": "schema_changed", "missing": miss, "range": f"{a}..{b}"}
    df = df[NEEDED]
    out = pd.concat([_agg(df, "pitcher", "pitcher"), _agg(df, "batter", "batter")],
                    ignore_index=True)
    existing = {(s, p, d) for s, p, d in session.execute(
        select(StatcastAgg.scope, StatcastAgg.player_id, StatcastAgg.game_date)
        .where(StatcastAgg.game_date >= a, StatcastAgg.game_date <= b)).all()}
    n = 0
    for r in out.itertuples():
        key = (r.scope, int(r.player_id), str(r.game_date))
        if key in existing:
            continue
        avail = datetime.strptime(str(r.game_date), "%Y-%m-%d") + timedelta(days=1, hours=6)
        session.add(StatcastAgg(
            scope=r.scope, player_id=int(r.player_id), game_date=str(r.game_date),
            season=season, pitches=int(r.pitches), batted_balls=int(r.batted_balls),
            hard_hit=int(r.hard_hit), barrels=int(r.barrels),
            xba=_f(r.xba), xslg=_f(r.xslg), xwoba=_f(r.xwoba),
            exit_velocity=_f(r.exit_velocity), launch_angle=_f(r.launch_angle),
            swstr=_f(r.swstr), csw=_f(r.csw), velocity=_f(r.velocity), spin=_f(r.spin),
            available_at=avail, source=SOURCE))
        n += 1
    session.flush()
    return {"status": "ok", "rows_raw": int(len(df)), "rows_agg": n, "range": f"{a}..{b}"}


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def pending_chunks(session, season: int, step_days: int = 4, start: str | None = None,
                   end: str | None = None) -> list[tuple[str, str]]:
    """Bloques que aun no se han descargado con exito (permite reanudar)."""
    from MLB.database.models import DataSourceLog
    done = {(r or "").split("rango ")[-1] for r in session.execute(
        select(DataSourceLog.notes).where(DataSourceLog.source == SOURCE,
                                          DataSourceLog.status.in_(("ok", "empty")))).scalars()}
    a = datetime.strptime(start or f"{season}-03-01", "%Y-%m-%d").date()
    z = datetime.strptime(end or f"{season}-11-15", "%Y-%m-%d").date()
    out = []
    while a <= z:
        b = min(a + timedelta(days=step_days - 1), z)
        if f"{a}..{b}" not in done:
            out.append((str(a), str(b)))
        a = b + timedelta(days=1)
    return out


def ingest_parallel(session, season: int, chunks: list[tuple[str, str]], workers: int = 6,
                    progress=None) -> dict:
    """Descarga varios bloques a la vez y los guarda en serie (SQLite: un escritor)."""
    from concurrent.futures import ThreadPoolExecutor
    raw = agg = 0
    errs = []
    for i in range(0, len(chunks), workers):
        batch = chunks[i:i + workers]
        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            got = list(ex.map(lambda ab: (ab, *get_csv(url_for(ab[0], ab[1], season),
                                                       "MLB", SOURCE, "statcast", timeout=300)),
                              batch))
        requeue = []
        for (a, b), df, f in got:
            f.notes = (f.notes or "") + f" rango {a}..{b}"
            if df is not None and len(df) >= SAVANT_ROW_CAP:
                f.status = "error"
                f.error = (f"respuesta truncada en {SAVANT_ROW_CAP} filas; "
                           f"el rango {a}..{b} se parte en dos y se reintenta")
                log_fetch(session, f)
                requeue += _split(a, b)
                continue
            log_fetch(session, f)
            if df is None or len(df) == 0:
                if f.status == "error":
                    errs.append({"range": f"{a}..{b}", "error": f.error})
                continue
            n = _store(session, df, season, a, b)
            raw += len(df); agg += n
        if requeue:
            chunks = chunks[:i + workers] + requeue + chunks[i + workers:]
        session.commit()
        if progress:
            progress(f"statcast {season}: {min(i+workers,len(chunks))}/{len(chunks)} bloques, "
                     f"{raw} lanzamientos, {agg} filas")
    return {"status": "ok" if not errs else "partial", "season": season,
            "chunks": len(chunks), "pitches": raw, "rows": agg,
            "errors": errs[:5], "n_errors": len(errs)}


def _split(a: str, b: str) -> list[tuple[str, str]]:
    """Parte un rango de fechas en dos mitades. Si ya es de un solo dia, no se
    puede partir mas: se devuelve igual y quedara registrado como error."""
    d0 = datetime.strptime(a, "%Y-%m-%d").date()
    d1 = datetime.strptime(b, "%Y-%m-%d").date()
    if d0 >= d1:
        return []
    mid = d0 + timedelta(days=(d1 - d0).days // 2)
    return [(str(d0), str(mid)), (str(mid + timedelta(days=1)), str(d1))]


def _store(session, df: pd.DataFrame, season: int, a: str, b: str) -> int:
    miss = [c for c in NEEDED if c not in df.columns]
    if miss:
        return 0
    df = df[NEEDED]
    out = pd.concat([_agg(df, "pitcher", "pitcher"), _agg(df, "batter", "batter")],
                    ignore_index=True)
    existing = {(s, p, d) for s, p, d in session.execute(
        select(StatcastAgg.scope, StatcastAgg.player_id, StatcastAgg.game_date)
        .where(StatcastAgg.game_date >= a, StatcastAgg.game_date <= b)).all()}
    n = 0
    for r in out.itertuples():
        if (r.scope, int(r.player_id), str(r.game_date)) in existing:
            continue
        avail = datetime.strptime(str(r.game_date), "%Y-%m-%d") + timedelta(days=1, hours=6)
        session.add(StatcastAgg(
            scope=r.scope, player_id=int(r.player_id), game_date=str(r.game_date),
            season=season, pitches=int(r.pitches), batted_balls=int(r.batted_balls),
            hard_hit=int(r.hard_hit), barrels=int(r.barrels),
            xba=_f(r.xba), xslg=_f(r.xslg), xwoba=_f(r.xwoba),
            exit_velocity=_f(r.exit_velocity), launch_angle=_f(r.launch_angle),
            swstr=_f(r.swstr), csw=_f(r.csw), velocity=_f(r.velocity), spin=_f(r.spin),
            available_at=avail, source=SOURCE))
        n += 1
    session.flush()
    return n


def ingest_season(session, season: int, step_days: int = 4, start: str | None = None,
                  end: str | None = None, progress=None) -> dict:
    a = datetime.strptime(start or f"{season}-03-01", "%Y-%m-%d").date()
    z = datetime.strptime(end or f"{season}-11-15", "%Y-%m-%d").date()
    chunks, raw, agg, errs = 0, 0, 0, []
    while a <= z:
        b = min(a + timedelta(days=step_days - 1), z)
        r = ingest_chunk(session, str(a), str(b), season)
        chunks += 1
        raw += r.get("rows_raw", 0)
        agg += r.get("rows_agg", 0)
        if r["status"] not in ("ok", "empty"):
            errs.append(r)
        if progress:
            progress(f"statcast {season} {a}..{b}: {r.get('rows_raw',0)} lanz., "
                     f"{r.get('rows_agg',0)} filas ({r['status']})")
        a = b + timedelta(days=1)
    return {"status": "ok" if not errs else "partial", "season": season, "chunks": chunks,
            "pitches": raw, "rows": agg, "errors": errs[:5], "n_errors": len(errs)}
