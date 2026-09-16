"""Cuotas NFL (The Odds API) para los mercados propios de SPC: moneyline, spread y total.

Cada descarga es un snapshot con timestamp. Se guarda apertura (primer snapshot
visto), actual y cierre (ultimo snapshot dentro de las 3 h previas al kickoff).
Nunca se usa el cierre para una prediccion anterior a el.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select

from NFL.adapter import TEAM_NAMES
from NFL.markets.db import NflOdds
from shared.odds import american_to_prob, two_way
from shared.timeutil import utcnow

BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"
MARKET_MAP = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
NICK_TO_ABBR = {v.lower(): k for k, v in TEAM_NAMES.items()}
NICK_TO_ABBR.update({"commanders": "WAS", "football team": "WAS", "redskins": "WAS", "rams": "LA",
                     "raiders": "LV", "chargers": "LAC"})


def abbr_from_name(name: str | None) -> str | None:
    if not name:
        return None
    n = name.lower()
    for nick, ab in NICK_TO_ABBR.items():
        if n.endswith(nick):
            return ab
    return None


def api_key() -> str | None:
    return os.getenv("THE_ODDS_API_KEY") or None


def _fetch(url: str, timeout: int = 60):
    import json
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ingest(session, games: list[dict], regions: str = "us") -> dict:
    """`games`: filas del motor NFL (game_id, home_team, away_team, kickoff_utc) — solo lectura."""
    key = api_key()
    if not key:
        return {"status": "skipped", "reason": "THE_ODDS_API_KEY no configurada", "rows": 0}
    url = (f"{BASE}/sports/{SPORT_KEY}/odds?regions={regions}&markets=h2h,spreads,totals"
           f"&oddsFormat=american&apiKey={key}")
    try:
        data = _fetch(url)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "rows": 0}
    now, n, unmatched = utcnow(), 0, 0
    for ev in data:
        home, away = abbr_from_name(ev.get("home_team")), abbr_from_name(ev.get("away_team"))
        try:
            comm = datetime.strptime(ev["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
        except (KeyError, ValueError):
            unmatched += 1
            continue
        g = next((g for g in games if g["home_team"] == home and g["away_team"] == away and
                  abs((g["kickoff_utc"] - comm).total_seconds()) < 36 * 3600), None)
        if not g:
            unmatched += 1
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                market = MARKET_MAP.get(mk.get("key"))
                if not market:
                    continue
                for o in mk.get("outcomes", []):
                    name = o.get("name")
                    sel = ("home" if name == ev.get("home_team") else
                           "away" if name == ev.get("away_team") else str(name).lower())
                    price, line = o.get("price"), o.get("point")
                    prev = session.execute(
                        select(NflOdds).where(NflOdds.game_id == g["game_id"], NflOdds.market == market,
                                              NflOdds.bookmaker == bk.get("key"), NflOdds.selection == sel)
                        .order_by(NflOdds.available_at.desc()).limit(1)).scalars().first()
                    if prev is not None and prev.price_american == price and prev.line == line:
                        continue
                    session.add(NflOdds(game_id=g["game_id"], bookmaker=bk.get("key"), market=market,
                                        selection=sel, line=line, price_american=price,
                                        implied_prob=american_to_prob(price), is_opening=prev is None,
                                        available_at=now))
                    n += 1
    session.flush()
    return {"status": "ok", "rows": n, "events": len(data), "unmatched": unmatched}


def consensus(session, game_id: str, market: str, cutoff) -> dict | None:
    """Consenso sin vig con cuotas ANTERIORES a `cutoff`. Devuelve p del lado local/over,
    la linea mediana (spread del local; total) y timestamps de apertura/actual."""
    rows = session.execute(select(NflOdds).where(NflOdds.game_id == game_id, NflOdds.market == market,
                                                 NflOdds.available_at < cutoff)).scalars().all()
    if not rows:
        return None
    last = {}
    for r in sorted(rows, key=lambda x: x.available_at):
        last[(r.bookmaker, r.selection)] = r
    by_book = {}
    for (bk, sel), r in last.items():
        by_book.setdefault(bk, {})[sel] = r
    probs, lines = [], []
    for bk, sides in by_book.items():
        a = sides.get("home") or sides.get("over")
        b = sides.get("away") or sides.get("under")
        if not a or not b or a.implied_prob is None or b.implied_prob is None:
            continue
        p, _ = two_way(a.implied_prob, b.implied_prob)
        if p is None:
            continue
        probs.append(p)
        if a.line is not None:
            lines.append(a.line)
    if not probs:
        return None
    opening = min(rows, key=lambda r: r.available_at)
    return {"p_home": float(np.mean(probs)), "n_books": len(probs),
            "line": float(np.median(lines)) if lines else None,
            "opening_line": opening.line, "opening_at": opening.available_at,
            "available_at": max(r.available_at for r in rows)}


def mark_closing(session, games: list[dict], max_hours_before: float = 3.0) -> int:
    n = 0
    for g in games:
        ko = g["kickoff_utc"]
        if ko > utcnow():
            continue
        rows = session.execute(select(NflOdds).where(
            NflOdds.game_id == g["game_id"], NflOdds.available_at < ko,
            NflOdds.available_at >= ko - timedelta(hours=max_hours_before))).scalars().all()
        if not rows:
            continue
        lastt = max(r.available_at for r in rows)
        for r in rows:
            if r.available_at == lastt and not r.is_closing:
                r.is_closing = True; n += 1
    session.flush()
    return n
