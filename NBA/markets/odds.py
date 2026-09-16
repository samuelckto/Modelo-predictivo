"""Cuotas y calendario NBA en vivo (The Odds API, basketball_nba). Aislado de NFL/MLB.

Los eventos con cuotas sirven ademas de CALENDARIO de partidos futuros (tipoff en
`commence_time`): pbpstats solo publica partidos ya jugados.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select

from NBA.markets.db import NbaGame, NbaOdds, NbaSourceLog, NbaTeam
from shared.odds import american_to_prob, two_way
from shared.timeutil import utcnow

BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"
MARKET_MAP = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
NICK = {"hawks": "ATL", "celtics": "BOS", "nets": "BKN", "hornets": "CHA", "bulls": "CHI", "cavaliers": "CLE",
        "mavericks": "DAL", "nuggets": "DEN", "pistons": "DET", "warriors": "GSW", "rockets": "HOU",
        "pacers": "IND", "clippers": "LAC", "lakers": "LAL", "grizzlies": "MEM", "heat": "MIA", "bucks": "MIL",
        "timberwolves": "MIN", "pelicans": "NOP", "knicks": "NYK", "thunder": "OKC", "magic": "ORL",
        "76ers": "PHI", "suns": "PHX", "trail blazers": "POR", "kings": "SAC", "spurs": "SAS", "raptors": "TOR",
        "jazz": "UTA", "wizards": "WAS"}


def abbr_from_name(name):
    n = (name or "").lower()
    for k, v in NICK.items():
        if n.endswith(k):
            return v
    return None


def api_key():
    return os.getenv("THE_ODDS_API_KEY") or None


def _fetch(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "SPC"}), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _team_by_abbr(session):
    return {t.abbr: t for t in session.execute(select(NbaTeam)).scalars() if t.abbr}


def _schedule_game(session, home, away, comm, teams) -> NbaGame | None:
    """Crea o encuentra el partido futuro (id propio 'SCH_...' hasta que pbpstats lo publique)."""
    lo, hi = comm - timedelta(hours=30), comm + timedelta(hours=30)
    g = session.execute(select(NbaGame).where(NbaGame.home_abbr == home, NbaGame.away_abbr == away,
                                              NbaGame.game_date >= lo.date(), NbaGame.game_date <= hi.date())
                        ).scalars().first()
    if g is not None:
        if g.start_utc is None:
            g.start_utc = comm
        return g
    th, ta = teams.get(home), teams.get(away)
    if th is None or ta is None:
        return None
    season = comm.year + 1 if comm.month >= 9 else comm.year
    g = NbaGame(game_id=f"SCH_{comm:%Y%m%d}_{away}_{home}", season=season, season_type="Regular Season",
                game_date=comm.date(), start_utc=comm, home_team_id=th.team_id, away_team_id=ta.team_id,
                home_abbr=home, away_abbr=away, status="scheduled", source="the_odds_api", ingested_at=utcnow())
    session.add(g); session.flush()
    return g


def ingest(session, regions="us") -> dict:
    key = api_key()
    if not key:
        session.add(NbaSourceLog(source="the_odds_api", domain="odds", url=f"{BASE}/sports/{SPORT_KEY}/odds",
                                 status="skipped", records=0, error="THE_ODDS_API_KEY no configurada", retrieved_at=utcnow()))
        return {"status": "skipped", "reason": "THE_ODDS_API_KEY no configurada", "rows": 0}
    url = f"{BASE}/sports/{SPORT_KEY}/odds?regions={regions}&markets=h2h,spreads,totals&oddsFormat=american&apiKey={key}"
    try:
        data = _fetch(url)
    except Exception as e:                                  # noqa: BLE001
        session.add(NbaSourceLog(source="the_odds_api", domain="odds", url=url.split("&apiKey")[0], status="error",
                                 records=0, error=f"{type(e).__name__}: {e}", retrieved_at=utcnow()))
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "rows": 0}
    teams = _team_by_abbr(session)
    now, n, unmatched, sched = utcnow(), 0, 0, 0
    for ev in data:
        home, away = abbr_from_name(ev.get("home_team")), abbr_from_name(ev.get("away_team"))
        try:
            comm = datetime.strptime(ev["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
        except (KeyError, ValueError):
            unmatched += 1; continue
        g = _schedule_game(session, home, away, comm, teams) if home and away else None
        if g is None:
            unmatched += 1; continue
        sched += 1
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                market = MARKET_MAP.get(mk.get("key"))
                if not market:
                    continue
                for o in mk.get("outcomes", []):
                    name = o.get("name")
                    sel = ("home" if name == ev.get("home_team") else "away" if name == ev.get("away_team")
                           else str(name).lower())
                    price, line = o.get("price"), o.get("point")
                    prev = session.execute(select(NbaOdds).where(
                        NbaOdds.game_id == g.game_id, NbaOdds.market == market, NbaOdds.bookmaker == bk.get("key"),
                        NbaOdds.selection == sel).order_by(NbaOdds.available_at.desc()).limit(1)).scalars().first()
                    if prev is not None and prev.price_american == price and prev.line == line:
                        continue
                    session.add(NbaOdds(game_id=g.game_id, bookmaker=bk.get("key"), market=market, selection=sel,
                                        line=line, price_american=price, implied_prob=american_to_prob(price),
                                        is_opening=prev is None, available_at=now))
                    n += 1
    session.add(NbaSourceLog(source="the_odds_api", domain="odds", url=url.split("&apiKey")[0], status="ok",
                             records=n, retrieved_at=now))
    session.flush()
    return {"status": "ok", "rows": n, "events": len(data), "partidos": sched, "unmatched": unmatched}


def consensus(session, game_id, market, cutoff) -> dict | None:
    """Sin vig, cuotas anteriores a `cutoff`. p_home = lado local / over; line = spread del
    local (negativo = local favorito) o total."""
    rows = session.execute(select(NbaOdds).where(NbaOdds.game_id == game_id, NbaOdds.market == market,
                                                 NbaOdds.available_at < cutoff)).scalars().all()
    if not rows:
        return None
    last = {}
    for r in sorted(rows, key=lambda x: x.available_at):
        last[(r.bookmaker, r.selection)] = r
    by_book = {}
    for (bk, sel), r in last.items():
        by_book.setdefault(bk, {})[sel] = r
    probs, lines, prices = [], [], []
    for sides in by_book.values():
        a = sides.get("home") or sides.get("over"); b = sides.get("away") or sides.get("under")
        if not a or not b or a.implied_prob is None or b.implied_prob is None:
            continue
        p, _ = two_way(a.implied_prob, b.implied_prob)
        if p is None:
            continue
        probs.append(p); prices.append((a.price_american, b.price_american))
        if a.line is not None:
            lines.append(a.line)
    if not probs:
        return None
    opening = min(rows, key=lambda r: r.available_at)
    return {"p_home": float(np.mean(probs)), "n_books": len(probs), "line": float(np.median(lines)) if lines else None,
            "odds_home": float(np.median([p[0] for p in prices])), "odds_away": float(np.median([p[1] for p in prices])),
            "opening_line": opening.line, "opening_at": opening.available_at,
            "available_at": max(r.available_at for r in rows)}


def mark_closing(session, max_hours_before=3.0) -> int:
    now = utcnow(); n = 0
    games = session.execute(select(NbaGame).where(NbaGame.start_utc.isnot(None), NbaGame.start_utc <= now,
                                                  NbaGame.start_utc >= now - timedelta(days=3))).scalars().all()
    for g in games:
        rows = session.execute(select(NbaOdds).where(NbaOdds.game_id == g.game_id, NbaOdds.available_at < g.start_utc,
                                                     NbaOdds.available_at >= g.start_utc - timedelta(hours=max_hours_before))
                               ).scalars().all()
        if not rows:
            continue
        lastt = max(r.available_at for r in rows)
        for r in rows:
            if r.available_at == lastt and not r.is_closing:
                r.is_closing = True; n += 1
    session.flush()
    return n


def movement(session, game_id, market) -> dict | None:
    rows = session.execute(select(NbaOdds).where(NbaOdds.game_id == game_id, NbaOdds.market == market,
                                                 NbaOdds.selection.in_(("home", "over")))
                           .order_by(NbaOdds.available_at)).scalars().all()
    rows = [r for r in rows if r.line is not None]
    if len(rows) < 2:
        return None
    return {"apertura": rows[0].line, "actual": rows[-1].line, "movimiento": round(rows[-1].line - rows[0].line, 2),
            "desde": rows[0].available_at, "hasta": rows[-1].available_at}
