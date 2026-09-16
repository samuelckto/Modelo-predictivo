"""Calendario y cuotas de tenis (The Odds API). Es tambien la fuente de partidos
futuros: el archivo historico esta congelado."""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select

from TENIS.markets.db import TenisOdds, TenisPlayer, TenisSchedule, TenisSourceLog
from shared.odds import american_to_prob, two_way
from shared.timeutil import utcnow

BASE = "https://api.the-odds-api.com/v4"
MARKET_MAP = {"h2h": "winner", "totals": "total_games", "spreads": "handicap_games"}
SURFACE_HINTS = {"clay": "Clay", "roland": "Clay", "monte": "Clay", "madrid": "Clay", "rome": "Clay",
                 "wimbledon": "Grass", "queen": "Grass", "halle": "Grass", "eastbourne": "Grass",
                 "s-hertogenbosch": "Grass", "newport": "Grass"}


def api_key():
    return os.getenv("THE_ODDS_API_KEY") or None


def _fetch(url, timeout=45):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "SPC"}), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def tennis_sports(key: str) -> list[dict]:
    try:
        return [s for s in _fetch(f"{BASE}/sports/?apiKey={key}")
                if str(s.get("key", "")).startswith("tennis_")]
    except Exception:                                   # noqa: BLE001
        return []


def tour_of(sport_key: str) -> str | None:
    k = sport_key.lower()
    if "_atp" in k:
        return "ATP"
    if "_wta" in k:
        return "WTA"
    return None


def surface_of(title: str) -> str | None:
    t = (title or "").lower()
    for k, v in SURFACE_HINTS.items():
        if k in t:
            return v
    return "Hard"                                        # la mayoria del calendario


def best_of(sport_key: str, tour: str) -> int:
    k = (sport_key or "").lower()
    slam = any(x in k for x in ("aus_open", "french_open", "wimbledon", "us_open"))
    return 5 if (slam and tour == "ATP") else 3


def match_player(session, tour: str, name: str) -> tuple[str | None, str]:
    """Empareja 'Carlos Alcaraz' con el id del historico. Devuelve (id, nombre canonico)."""
    if not name:
        return None, name
    n = " ".join(name.replace(".", " ").split()).lower()
    cands = session.execute(select(TenisPlayer).where(TenisPlayer.tour == tour)).scalars().all()
    exact = [p for p in cands if (p.name or "").lower() == n]
    if exact:
        return exact[0].player_id, exact[0].name
    last = n.split()[-1]
    same_last = [p for p in cands if (p.name or "").lower().endswith(" " + last)]
    if len(same_last) == 1:
        return same_last[0].player_id, same_last[0].name
    ini = n[0]
    narrow = [p for p in same_last if (p.name or "").lower().startswith(ini)]
    if len(narrow) == 1:
        return narrow[0].player_id, narrow[0].name
    return None, name


def ingest(session, regions="us") -> dict:
    key = api_key()
    if not key:
        session.add(TenisSourceLog(source="the_odds_api", domain="odds", url=f"{BASE}/sports", status="skipped",
                                   records=0, error="THE_ODDS_API_KEY no configurada", retrieved_at=utcnow()))
        return {"status": "skipped", "reason": "THE_ODDS_API_KEY no configurada", "rows": 0}
    sports = tennis_sports(key)
    if not sports:
        return {"status": "empty", "reason": "The Odds API no lista torneos de tenis ahora mismo", "rows": 0}
    now, rows, eventos, sin_match = utcnow(), 0, 0, 0
    for sp in sports:
        sk = sp["key"]; tour = tour_of(sk)
        if tour is None:
            continue
        url = f"{BASE}/sports/{sk}/odds?regions={regions}&markets=h2h,totals,spreads&oddsFormat=american&apiKey={key}"
        try:
            data = _fetch(url, 60)
        except Exception as e:                          # noqa: BLE001
            session.add(TenisSourceLog(source="the_odds_api", domain="odds", url=url.split("&apiKey")[0],
                                       status="error", records=0, error=f"{type(e).__name__}: {e}",
                                       retrieved_at=utcnow()))
            continue
        for ev in data:
            eventos += 1
            try:
                comm = datetime.strptime(ev["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
            except (KeyError, ValueError):
                continue
            eid = ev.get("id") or f"{sk}_{comm:%Y%m%d%H%M}_{ev.get('home_team')}"
            p1n, p2n = ev.get("home_team"), ev.get("away_team")
            id1, nm1 = match_player(session, tour, p1n)
            id2, nm2 = match_player(session, tour, p2n)
            if id1 is None or id2 is None:
                sin_match += 1
            sc = session.get(TenisSchedule, eid)
            if sc is None:
                sc = TenisSchedule(event_id=eid); session.add(sc)
            sc.tour, sc.sport_key = tour, sk
            sc.tourney_name = sp.get("title"); sc.surface = surface_of(sp.get("title"))
            sc.start_utc = comm; sc.p1_name, sc.p2_name = nm1, nm2
            sc.p1_id, sc.p2_id = id1, id2; sc.best_of = best_of(sk, tour); sc.ingested_at = now
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    market = MARKET_MAP.get(mk.get("key"))
                    if not market:
                        continue
                    for o in mk.get("outcomes", []):
                        nm = o.get("name")
                        sel = ("p1" if nm == p1n else "p2" if nm == p2n else str(nm).lower())
                        price, line = o.get("price"), o.get("point")
                        prev = session.execute(select(TenisOdds).where(
                            TenisOdds.event_id == eid, TenisOdds.market == market,
                            TenisOdds.bookmaker == bk.get("key"), TenisOdds.selection == sel)
                            .order_by(TenisOdds.available_at.desc()).limit(1)).scalars().first()
                        if prev is not None and prev.price_american == price and prev.line == line:
                            continue
                        session.add(TenisOdds(event_id=eid, bookmaker=bk.get("key"), market=market, selection=sel,
                                              line=line, price_american=price, implied_prob=american_to_prob(price),
                                              is_opening=prev is None, available_at=now))
                        rows += 1
        session.add(TenisSourceLog(source="the_odds_api", domain=sk, url=url.split("&apiKey")[0], status="ok",
                                   records=rows, retrieved_at=now))
    session.flush()
    return {"status": "ok", "torneos": len(sports), "eventos": eventos, "rows": rows,
            "jugadores_sin_emparejar": sin_match}


def consensus(session, event_id, market, cutoff) -> dict | None:
    rows = session.execute(select(TenisOdds).where(TenisOdds.event_id == event_id, TenisOdds.market == market,
                                                   TenisOdds.available_at < cutoff)).scalars().all()
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
        a = sides.get("p1") or sides.get("over"); b = sides.get("p2") or sides.get("under")
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
    return {"p1": float(np.mean(probs)), "n_books": len(probs), "line": float(np.median(lines)) if lines else None,
            "odds_1": float(np.median([p[0] for p in prices])), "odds_2": float(np.median([p[1] for p in prices])),
            "opening_line": opening.line, "available_at": max(r.available_at for r in rows)}


def movement(session, event_id, market) -> dict | None:
    rows = session.execute(select(TenisOdds).where(TenisOdds.event_id == event_id, TenisOdds.market == market,
                                                   TenisOdds.selection.in_(("p1", "over")))
                           .order_by(TenisOdds.available_at)).scalars().all()
    if len(rows) < 2:
        return None
    a, b = rows[0], rows[-1]
    if market == "winner":
        if a.implied_prob is None or b.implied_prob is None:
            return None
        return {"apertura": round(a.implied_prob, 4), "actual": round(b.implied_prob, 4),
                "movimiento_pp": round((b.implied_prob - a.implied_prob) * 100, 2)}
    if a.line is None or b.line is None:
        return None
    return {"apertura": a.line, "actual": b.line, "movimiento": round(b.line - a.line, 2)}


def mark_closing(session, max_hours_before=3.0) -> int:
    now = utcnow(); n = 0
    scs = session.execute(select(TenisSchedule).where(TenisSchedule.start_utc <= now,
                                                      TenisSchedule.start_utc >= now - timedelta(days=3))).scalars().all()
    for sc in scs:
        rows = session.execute(select(TenisOdds).where(
            TenisOdds.event_id == sc.event_id, TenisOdds.available_at < sc.start_utc,
            TenisOdds.available_at >= sc.start_utc - timedelta(hours=max_hours_before))).scalars().all()
        if not rows:
            continue
        lastt = max(r.available_at for r in rows)
        for r in rows:
            if r.available_at == lastt and not r.is_closing:
                r.is_closing = True; n += 1
    session.flush()
    return n
