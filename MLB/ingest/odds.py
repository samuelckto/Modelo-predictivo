"""Cuotas MLB desde The Odds API.

Si no hay clave configurada NO se inventa nada: se registra `skipped` y las
predicciones salen sin mercado, marcadas como tales.

Se guarda cada snapshot como fila nueva (nunca se sobrescribe), con su
`available_at`, de modo que una prediccion antigua jamas puede evaluarse con
cuotas posteriores.
"""
from __future__ import annotations

import os
from datetime import timedelta

from sqlalchemy import select

from MLB.database.models import Game, Odds
from MLB.ingest.http import get_json, log_fetch
from shared.odds import american_to_prob
from shared.sources import Fetch
from shared.timeutil import utcnow

SOURCE = "the_odds_api"
BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
MARKET_MAP = {"h2h": "moneyline", "spreads": "run_line", "totals": "total"}


def api_key() -> str | None:
    k = os.getenv("THE_ODDS_API_KEY")
    return k.strip() if k and k.strip() else None


def _match(session, home_abbr, away_abbr, commence):
    """Empareja un evento de la API con un partido de la base por equipos y fecha."""
    lo, hi = commence - timedelta(hours=12), commence + timedelta(hours=12)
    return session.execute(
        select(Game).where(Game.home_abbr == home_abbr, Game.away_abbr == away_abbr,
                           Game.start_utc >= lo, Game.start_utc <= hi)
        .order_by(Game.start_utc)).scalars().first()


TEAM_BY_NAME: dict[str, str] = {}


def _load_names(session):
    global TEAM_BY_NAME
    if TEAM_BY_NAME:
        return
    from MLB.database.models import Team
    for t in session.execute(select(Team)).scalars():
        if t.name and t.abbr:
            TEAM_BY_NAME[t.name] = t.abbr


def snapshot_stats(session, game_ids: list[int] | None = None) -> dict:
    """Cuantos snapshots de mercado se han acumulado y desde cuando."""
    from sqlalchemy import func
    q = select(func.count(), func.count(func.distinct(Odds.game_id)),
               func.count(func.distinct(Odds.bookmaker)),
               func.min(Odds.available_at), func.max(Odds.available_at))
    if game_ids:
        q = q.where(Odds.game_id.in_(game_ids))
    n, games, books, a, b = session.execute(q).one()
    per_game = session.execute(
        select(Odds.game_id, func.count(func.distinct(Odds.available_at)))
        .group_by(Odds.game_id)).all()
    snaps = [c for _, c in per_game]
    return {"rows": n or 0, "games": games or 0, "bookmakers": books or 0,
            "first": a, "last": b,
            "snapshots_por_partido": {"min": min(snaps) if snaps else 0,
                                      "max": max(snaps) if snaps else 0,
                                      "media": round(sum(snaps) / len(snaps), 2) if snaps else 0},
            "listo_para_comparar_mercado": bool((games or 0) >= 300 and snaps and
                                                min(snaps) >= 1)}


def movements(session, game_id: int, market: str = "moneyline") -> list[dict]:
    """Movimiento del mercado para un partido: apertura, cambios y cierre."""
    rows = session.execute(
        select(Odds).where(Odds.game_id == game_id, Odds.market == market)
        .order_by(Odds.available_at)).scalars().all()
    out, last = [], {}
    for r in rows:
        key = (r.bookmaker, r.selection)
        if last.get(key) != r.price_american:
            out.append({"available_at": r.available_at, "bookmaker": r.bookmaker,
                        "selection": r.selection, "line": r.line,
                        "price": r.price_american,
                        "tipo": ("apertura" if key not in last else
                                 ("cierre" if r.is_closing else "movimiento"))})
            last[key] = r.price_american
    return out


def mark_closing(session, before_start_minutes: int = 15,
                 max_hours_before: float = 3.0) -> dict:
    """Marca como `is_closing` el ultimo snapshot anterior al inicio de cada partido.

    Solo se marca si ese snapshot es de las ultimas `max_hours_before` horas antes
    del partido. Un snapshot tomado dos dias antes NO es la linea de cierre, y
    llamarlo asi seria falsear el dato: con una sola toma diaria el sistema
    simplemente no tiene cierre y debe decirlo.

    IMPORTANTE: marcar el cierre NO lo hace utilizable por una prediccion anterior.
    El filtro `available_at < prediction_timestamp` sigue siendo el unico que decide.
    """
    from datetime import timedelta
    games = {g.id: g.start_utc for g in session.execute(
        select(Game).where(Game.id.in_(select(Odds.game_id.distinct())))).scalars()}
    n = 0
    for gid, start in games.items():
        if not start:
            continue
        limite = start - timedelta(minutes=before_start_minutes)
        minimo = start - timedelta(hours=max_hours_before)
        last = session.execute(
            select(Odds).where(Odds.game_id == gid, Odds.available_at <= limite,
                               Odds.available_at >= minimo)
            .order_by(Odds.available_at.desc()).limit(1)).scalars().first()
        if last is None:
            continue
        same = session.execute(select(Odds).where(
            Odds.game_id == gid, Odds.available_at == last.available_at)).scalars().all()
        for r in same:
            if not r.is_closing:
                r.is_closing = True
                n += 1
    session.flush()
    return {"status": "ok", "marcadas_como_cierre": n,
            "criterio": f"ultimo snapshot dentro de las {max_hours_before} h previas "
                        f"al partido; si no hay ninguno, ese partido queda sin cierre"}


def ingest(session, regions: str = "us", markets: str = "h2h,spreads,totals") -> dict:
    key = api_key()
    if not key:
        f = Fetch("MLB", SOURCE, "odds", f"{BASE}/sports/{SPORT_KEY}/odds", "skipped",
                  retrieved_at=utcnow(),
                  error="THE_ODDS_API_KEY no configurada: no se descargan cuotas")
        log_fetch(session, f)
        return {"status": "skipped", "reason": "sin clave de API configurada",
                "impact": "las predicciones saldran sin mercado y se marcaran asi"}
    _load_names(session)
    url = (f"{BASE}/sports/{SPORT_KEY}/odds?regions={regions}&markets={markets}"
           f"&oddsFormat=american&apiKey={key}")
    data, f = get_json(url, "MLB", SOURCE, "odds", timeout=60)
    f.url = f.url.split("&apiKey=")[0]              # nunca se guarda la clave
    log_fetch(session, f)
    if not data:
        return {"status": f.status, "error": f.error, "rows": 0}
    now, n, unmatched = utcnow(), 0, 0
    for ev in data:
        home = TEAM_BY_NAME.get(ev.get("home_team", ""))
        away = TEAM_BY_NAME.get(ev.get("away_team", ""))
        if not home or not away:
            unmatched += 1
            continue
        from datetime import datetime
        try:
            comm = datetime.strptime(ev["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
        except (KeyError, ValueError):
            unmatched += 1
            continue
        g = _match(session, home, away, comm)
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
                           "away" if name == ev.get("away_team") else
                           str(name).lower())
                    price = o.get("price")
                    # no se duplica si nada cambio desde el ultimo snapshot
                    prev = session.execute(
                        select(Odds).where(Odds.game_id == g.id, Odds.market == market,
                                           Odds.bookmaker == bk.get("key"),
                                           Odds.selection == sel)
                        .order_by(Odds.available_at.desc()).limit(1)).scalars().first()
                    if prev is not None and prev.price_american == price and \
                            prev.line == o.get("point"):
                        continue
                    session.add(Odds(
                        game_id=g.id, bookmaker=bk.get("key"), market=market, selection=sel,
                        line=o.get("point"), price_american=price,
                        implied_prob=american_to_prob(price),
                        is_closing=False, available_at=now, source=SOURCE))
                    n += 1
    session.flush()
    return {"status": "ok", "rows": n, "events": len(data), "unmatched": unmatched}


def consensus(session, game_id: int, market: str, cutoff) -> dict | None:
    """Consenso sin vig con las cuotas conocidas ANTES de `cutoff`.

    Devuelve None si no hay cuotas: el llamador debe marcar mercado no disponible.
    """
    from shared.odds import two_way
    rows = session.execute(
        select(Odds).where(Odds.game_id == game_id, Odds.market == market,
                           Odds.available_at < cutoff)).scalars().all()
    if not rows:
        return None
    last = {}
    for r in sorted(rows, key=lambda x: x.available_at):
        last[(r.bookmaker, r.selection, r.line)] = r
    by_book: dict[str, dict] = {}
    for (bk, sel, line), r in last.items():
        by_book.setdefault(bk, {})[sel] = r
    probs, lines = [], []
    for bk, sides in by_book.items():
        a = sides.get("home") or sides.get("over")
        b = sides.get("away") or sides.get("under")
        if not a or not b or a.implied_prob is None or b.implied_prob is None:
            continue
        p, ov = two_way(a.implied_prob, b.implied_prob)
        if p is None:
            continue
        probs.append(p)
        if a.line is not None:
            lines.append(a.line)
    if not probs:
        return None
    import numpy as np
    return {"p_home": float(np.mean(probs)), "n_books": len(probs),
            "line": float(np.median(lines)) if lines else None,
            "best": float(max(probs)), "worst": float(min(probs)),
            "available_at": max(r.available_at for r in rows)}
