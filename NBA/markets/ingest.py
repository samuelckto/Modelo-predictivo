"""Ingesta NBA desde pbpstats.com (partidos y logs por equipo). Reanudable.

available_at de cada log = 23:59:59 UTC del dia del partido (fin del dia):
como pbpstats no publica la hora de tipoff, un partido NUNCA puede usar datos de
otro partido del mismo dia. Conservador a proposito.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, datetime, time as dtime, timedelta

from sqlalchemy import select

from NBA.markets.db import NbaGame, NbaSourceLog, NbaTeam, NbaTeamGameLog, init_db, session_scope
from shared.timeutil import utcnow

BASE = "https://api.pbpstats.com"
SOURCE = "pbpstats"
H = {"User-Agent": "Mozilla/5.0 (SportsPredictionCenter)", "Accept": "application/json"}
SEASON_TYPES = ("Regular Season", "Playoffs")


def season_label(season: int) -> str:
    """2016 -> '2015-16'."""
    return f"{season-1}-{str(season)[-2:]}"


def _get(url: str, retries: int = 4, timeout: int = 90):
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8")), None
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (i + 1))
    return None, last


def _log(s, domain, url, status, records=0, error=None):
    s.add(NbaSourceLog(source=SOURCE, domain=domain, url=url, status=status, records=records,
                       error=error, retrieved_at=utcnow()))


def ingest_games(s, season: int) -> dict:
    n = 0
    for st in SEASON_TYPES:
        url = f"{BASE}/get-games/nba?Season={season_label(season)}&SeasonType={st.replace(' ', '%20')}"
        data, err = _get(url)
        if data is None:
            _log(s, "games", url, "error", 0, err)
            continue
        rows = data.get("results", [])
        for r in rows:
            gid = str(r["GameId"])
            g = s.get(NbaGame, gid)
            if g is None:
                g = NbaGame(game_id=gid); s.add(g)
            g.season, g.season_type = season, st
            g.game_date = date.fromisoformat(r["Date"])
            g.home_team_id, g.away_team_id = str(r["HomeTeamId"]), str(r["AwayTeamId"])
            g.home_abbr, g.away_abbr = r.get("HomeTeamAbbreviation"), r.get("AwayTeamAbbreviation")
            g.home_points, g.away_points = r.get("HomePoints"), r.get("AwayPoints")
            g.home_poss, g.away_poss = r.get("HomePossessions"), r.get("AwayPossessions")
            g.status = "final" if r.get("HomePoints") is not None else "scheduled"
            g.ingested_at = utcnow()
            for tid, ab in ((g.home_team_id, g.home_abbr), (g.away_team_id, g.away_abbr)):
                if s.get(NbaTeam, tid) is None:
                    s.add(NbaTeam(team_id=tid, abbr=ab, name=ab))
            n += 1
        _log(s, "games", url, "ok", len(rows))
    s.flush()
    return {"season": season, "games": n}


def pending_logs(s, season: int) -> list[tuple[str, str]]:
    """(team_id, season_type) que aun no tienen logs completos."""
    out = []
    for st in SEASON_TYPES:
        games = s.execute(select(NbaGame).where(NbaGame.season == season, NbaGame.season_type == st,
                                                NbaGame.status == "final")).scalars().all()
        if not games:
            continue
        teams = {g.home_team_id for g in games} | {g.away_team_id for g in games}
        have = {}
        for tid, cnt in s.execute(
                select(NbaTeamGameLog.team_id, __import__("sqlalchemy").func.count())
                .where(NbaTeamGameLog.season == season,
                       NbaTeamGameLog.game_id.in_([g.game_id for g in games]))
                .group_by(NbaTeamGameLog.team_id)).all():
            have[tid] = cnt
        for tid in sorted(teams):
            expected = sum(1 for g in games if tid in (g.home_team_id, g.away_team_id))
            if have.get(tid, 0) < expected:
                out.append((tid, st))
    return out


def _store_logs(s, season: int, team_id: str, season_type: str, rows: list) -> int:
    games = {g.game_id: g for g in s.execute(select(NbaGame).where(
        NbaGame.season == season, NbaGame.season_type == season_type)).scalars().all()}
    have = {r.game_id for r in s.execute(select(NbaTeamGameLog).where(
        NbaTeamGameLog.team_id == team_id, NbaTeamGameLog.season == season)).scalars().all()}
    n = 0
    for r in rows:
        gid = str(r.get("GameId"))
        g = games.get(gid)
        if g is None or gid in have:
            continue
        gd = date.fromisoformat(r["Date"])
        s.add(NbaTeamGameLog(
            game_id=gid, team_id=team_id,
            opp_id=(g.away_team_id if g.home_team_id == team_id else g.home_team_id),
            season=season, game_date=gd, is_home=(g.home_team_id == team_id),
            stats={k: v for k, v in r.items() if k not in ("GameId", "Date", "Opponent")},
            available_at=datetime.combine(gd, dtime(23, 59, 59))))
        n += 1
    s.flush()
    return n


def ingest_team_logs(s, season: int, team_id: str, season_type: str) -> dict:
    url = (f"{BASE}/get-game-logs/nba?Season={season_label(season)}"
           f"&SeasonType={season_type.replace(' ', '%20')}&EntityId={team_id}&EntityType=Team")
    data, err = _get(url)
    if data is None:
        _log(s, "team_logs", url, "error", 0, err)
        return {"team": team_id, "status": "error", "error": err}
    rows = data.get("multi_row_table_data", [])
    games = {g.game_id: g for g in s.execute(select(NbaGame).where(
        NbaGame.season == season, NbaGame.season_type == season_type)).scalars().all()}
    have = {r.game_id for r in s.execute(select(NbaTeamGameLog).where(
        NbaTeamGameLog.team_id == team_id, NbaTeamGameLog.season == season)).scalars().all()}
    n = 0
    for r in rows:
        gid = str(r.get("GameId"))
        g = games.get(gid)
        if g is None or gid in have:
            continue
        gd = date.fromisoformat(r["Date"])
        s.add(NbaTeamGameLog(
            game_id=gid, team_id=team_id,
            opp_id=(g.away_team_id if g.home_team_id == team_id else g.home_team_id),
            season=season, game_date=gd, is_home=(g.home_team_id == team_id),
            stats={k: v for k, v in r.items() if k not in ("GameId", "Date", "Opponent")},
            available_at=datetime.combine(gd, dtime(23, 59, 59))))
        n += 1
    _log(s, "team_logs", url, "ok", n)
    s.flush()
    return {"team": team_id, "season_type": season_type, "rows": n}


def run(seasons: list[int], budget_s: float = 150.0, progress=print) -> dict:
    """Ingesta reanudable con presupuesto de tiempo (los procesos largos mueren)."""
    init_db()
    t0 = time.time()
    done, left = [], []
    with session_scope() as s:
        for season in seasons:
            has = s.execute(select(NbaGame).where(NbaGame.season == season).limit(1)).scalars().first()
            if has is None:
                progress(f"partidos {season_label(season)}…")
                done.append(ingest_games(s, season)); s.commit()
            pend = pending_logs(s, season)
            # descargas en paralelo (3 hilos), escritura secuencial
            from concurrent.futures import ThreadPoolExecutor
            i = 0
            while i < len(pend):
                if time.time() - t0 > budget_s:
                    left.extend((season, tid, st) for tid, st in pend[i:])
                    break
                chunk = pend[i:i + 3]; i += 3
                urls = [(f"{BASE}/get-game-logs/nba?Season={season_label(season)}"
                         f"&SeasonType={st.replace(' ', '%20')}&EntityId={tid}&EntityType=Team")
                        for tid, st in chunk]
                with ThreadPoolExecutor(3) as ex:
                    res = list(ex.map(_get, urls))
                for (tid, st), url, (data, err) in zip(chunk, urls, res):
                    if data is None:
                        _log(s, "team_logs", url, "error", 0, err); continue
                    r = _store_logs(s, season, tid, st, data.get("multi_row_table_data", []))
                    _log(s, "team_logs", url, "ok", r)
                    progress(f"  {season_label(season)} {st} {tid}: {r}")
                s.commit()
    return {"done": done, "pending": len(left), "elapsed_s": round(time.time() - t0, 1)}
