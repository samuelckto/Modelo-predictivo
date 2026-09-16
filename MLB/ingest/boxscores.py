"""Lineas reales de cada partido terminado (pitcheo y bateo) desde el boxscore.

Por que el boxscore y no Statcast: los outs, entradas lanzadas y carreras
limpias del boxscore son el dato oficial. Statcast se usa solo para las
metricas avanzadas de contacto (xwOBA, barrels, velocidad, spin).

`available_at`: la API no publica la hora exacta de finalizacion, asi que se usa
`start_utc + DURACION_TIPICA`. Es una aproximacion CONSERVADORA y esta declarada
aqui y en el README: sirve para que estas lineas nunca se usen en una prediccion
del propio partido, que es lo que importa para el anti-leakage.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from sqlalchemy import select

from MLB.database.models import BatterGameLog, BullpenUsage, Game, Lineup, PitcherGameLog
from MLB.ingest.http import get_json, log_fetch

BASE = "https://statsapi.mlb.com/api/v1"
SOURCE = "mlb_stats_api"
DURACION_TIPICA = timedelta(hours=3, minutes=15)


def _ip_to_outs(ip) -> int | None:
    """'6.2' -> 20 outs. None si el dato no viene."""
    if ip in (None, ""):
        return None
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) * 3 + int(frac or 0)
    except ValueError:
        return None


def _fetch(pk: int):
    url = f"{BASE}/game/{pk}/boxscore"
    return pk, *get_json(url, "MLB", SOURCE, "boxscore", retries=2, timeout=60)


def ingest_boxscores(session, game_pks: list[int], workers: int = 8,
                     progress=None) -> dict:
    games = {g.id: g for g in session.execute(
        select(Game).where(Game.id.in_(game_pks))).scalars().all()}
    ok = err = n_pit = n_bat = n_lin = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pk, data, f in ex.map(_fetch, game_pks):
            log_fetch(session, f)
            done += 1
            if progress and done % 200 == 0:
                progress(f"boxscores {done}/{len(game_pks)}")
            if not data:
                err += 1
                continue
            ok += 1
            g = games.get(pk)
            avail = (g.start_utc + DURACION_TIPICA) if (g and g.start_utc) else None
            for side, is_home in (("home", True), ("away", False)):
                t = (data.get("teams") or {}).get(side) or {}
                team_id = (t.get("team") or {}).get("id")
                players = t.get("players") or {}
                order = [int(x) for x in (t.get("battingOrder") or [])]
                if order:
                    n_lin += _lineup(session, pk, team_id, is_home, order,
                                     g.start_utc if g else None)
                for _, p in players.items():
                    pid = (p.get("person") or {}).get("id")
                    st = p.get("stats") or {}
                    pit, bat = st.get("pitching") or {}, st.get("batting") or {}
                    if pit.get("inningsPitched") is not None:
                        n_pit += _pitcher(session, pk, g, pid, team_id, pit, p, avail)
                    if bat.get("plateAppearances") or bat.get("atBats"):
                        n_bat += _batter(session, pk, g, pid, team_id, bat, avail)
            if done % 300 == 0:
                session.flush()
    session.flush()
    return {"status": "ok", "games_ok": ok, "games_error": err,
            "pitcher_lines": n_pit, "batter_lines": n_bat, "lineups": n_lin}


def _lineup(session, pk, team_id, is_home, order, start_utc) -> int:
    exists = session.execute(
        select(Lineup.id).where(Lineup.game_id == pk, Lineup.team_id == team_id)
        .limit(1)).scalars().first()
    if exists:
        return 0
    # available_at = hora de inicio: garantiza que NO se use para predecir este partido.
    session.add(Lineup(game_id=pk, team_id=team_id, is_home=is_home, state="confirmed",
                       batting_order=order, available_at=start_utc, source=SOURCE))
    return 1


def _pitcher(session, pk, g, pid, team_id, s, p, avail) -> int:
    if session.execute(select(PitcherGameLog.id).where(
            PitcherGameLog.game_id == pk, PitcherGameLog.player_id == pid)).scalars().first():
        return 0
    starter = bool((p.get("gameStatus") or {}).get("isSubstitute") is False and
                   (p.get("stats", {}).get("pitching", {}).get("gamesStarted") == 1))
    session.add(PitcherGameLog(
        game_id=pk, game_date=(g.game_date if g else None), season=(g.season if g else None),
        player_id=pid, team_id=team_id, is_starter=starter,
        outs=_ip_to_outs(s.get("inningsPitched")), pitches=s.get("numberOfPitches"),
        batters_faced=s.get("battersFaced"), hits=s.get("hits"), runs=s.get("runs"),
        earned_runs=s.get("earnedRuns"), walks=s.get("baseOnBalls"),
        strikeouts=s.get("strikeOuts"), home_runs=s.get("homeRuns"),
        ground_outs=s.get("groundOuts"), air_outs=s.get("airOuts"),
        available_at=avail, source=SOURCE))
    return 1


def _batter(session, pk, g, pid, team_id, s, avail) -> int:
    if session.execute(select(BatterGameLog.id).where(
            BatterGameLog.game_id == pk, BatterGameLog.player_id == pid)).scalars().first():
        return 0
    session.add(BatterGameLog(
        game_id=pk, game_date=(g.game_date if g else None), season=(g.season if g else None),
        player_id=pid, team_id=team_id,
        plate_appearances=s.get("plateAppearances"), at_bats=s.get("atBats"),
        hits=s.get("hits"), doubles=s.get("doubles"), triples=s.get("triples"),
        home_runs=s.get("homeRuns"), walks=s.get("baseOnBalls"),
        strikeouts=s.get("strikeOuts"), rbi=s.get("rbi"),
        total_bases=s.get("totalBases"), available_at=avail, source=SOURCE))
    return 1


# ------------------------------------------------------------- bullpen usage
def rebuild_bullpen_usage(session, season: int) -> dict:
    """Deriva el uso del bullpen de las lineas ya guardadas. No pide nada extra."""
    import pandas as pd
    rows = session.execute(select(
        PitcherGameLog.team_id, PitcherGameLog.game_date, PitcherGameLog.season,
        PitcherGameLog.player_id, PitcherGameLog.is_starter, PitcherGameLog.outs,
        PitcherGameLog.pitches, PitcherGameLog.available_at)
        .where(PitcherGameLog.season == season)).all()
    if not rows:
        return {"status": "empty", "season": season}
    df = pd.DataFrame(rows, columns=["team_id", "game_date", "season", "player_id",
                                     "is_starter", "outs", "pitches", "available_at"])
    rel = df[~df.is_starter.fillna(False)]
    agg = (rel.groupby(["team_id", "game_date", "season"], as_index=False)
              .agg(relievers_used=("player_id", "nunique"),
                   bullpen_outs=("outs", "sum"), bullpen_pitches=("pitches", "sum"),
                   available_at=("available_at", "max")))
    # back to back: relevistas que lanzaron tambien el dia anterior
    rel = rel.sort_values("game_date")
    prev = rel.assign(next_date=rel.game_date)
    b2b = {}
    for tid, grp in rel.groupby("team_id"):
        by_date = grp.groupby("game_date").player_id.apply(set)
        dates = list(by_date.index)
        for i, d in enumerate(dates):
            if i == 0:
                b2b[(tid, d)] = 0
                continue
            b2b[(tid, d)] = len(by_date[d] & by_date[dates[i - 1]])
    agg["back_to_back"] = [b2b.get((r.team_id, r.game_date), 0) for r in agg.itertuples()]
    existing = {(t, d) for t, d in session.execute(
        select(BullpenUsage.team_id, BullpenUsage.game_date)
        .where(BullpenUsage.season == season)).all()}
    n = 0
    for r in agg.itertuples():
        if (r.team_id, r.game_date) in existing:
            continue
        session.add(BullpenUsage(team_id=int(r.team_id), game_date=r.game_date,
                                 season=int(season), relievers_used=int(r.relievers_used),
                                 bullpen_outs=int(r.bullpen_outs or 0),
                                 bullpen_pitches=int(r.bullpen_pitches or 0),
                                 back_to_back=int(r.back_to_back),
                                 available_at=r.available_at, source="derived_from_boxscores"))
        n += 1
    session.flush()
    return {"status": "ok", "season": season, "rows": n}
