"""Ingesta ATP/WTA desde el archivo espejo de los datasets de Jeff Sackmann.

Los repositorios originales (JeffSackmann/tennis_atp y tennis_wta) dejaron de ser
publicos; se usa el espejo `Aneeshers/tennis-sackmann-archive`, congelado el
2026-06-25 (ultimo partido 2026-05-25). Los partidos posteriores llegan por el
calendario de The Odds API y se resuelven con el resultado que el propio sistema
califica. Esta limitacion se reporta en Data Health y en el informe.
"""
from __future__ import annotations

import csv
import io
import time
import urllib.request
from datetime import date, datetime

from sqlalchemy import select

from TENIS.markets.db import (TenisMatch, TenisPlayer, TenisSourceLog, init_db, session_scope)
from shared.timeutil import utcnow

BASE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
SOURCE = "sackmann-archive"
H = {"User-Agent": "SportsPredictionCenter"}
TOURS = {"ATP": "atp", "WTA": "wta"}
SERVE_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]


def _get(url, retries=3, timeout=90):
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore"), None
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (i + 1))
    return None, last


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_score(score: str, retired: bool) -> tuple[int | None, int | None, int | None, int | None]:
    """Devuelve (juegos_ganador, juegos_perdedor, sets_ganador, sets_perdedor).
    Ignora tie-breaks entre parentesis. None si el marcador no es interpretable
    (walkover, 'Def.', abandonos sin sets completos)."""
    if not score or any(k in score.upper() for k in ("W/O", "WALKOVER", "DEF")):
        return None, None, None, None
    gw = gl = sw = sl = 0
    for chunk in score.split():
        c = chunk.split("(")[0]
        if "-" not in c:
            continue
        a, _, b = c.partition("-")
        a, b = _int(a), _int(b)
        if a is None or b is None:
            continue
        gw += a; gl += b
        if a > b:
            sw += 1
        elif b > a:
            sl += 1
    if gw + gl == 0:
        return None, None, None, None
    return gw, gl, sw, sl


def ingest_players(s, tour: str) -> dict:
    url = f"{BASE}/{TOURS[tour]}/{TOURS[tour]}_players.csv"
    txt, err = _get(url)
    if txt is None:
        s.add(TenisSourceLog(source=SOURCE, domain="players", url=url, status="error", records=0,
                             error=err, retrieved_at=utcnow()))
        return {"status": "error", "error": err}
    have = {p.player_id for p in s.execute(select(TenisPlayer).where(TenisPlayer.tour == tour)).scalars()}
    n = 0
    for r in csv.DictReader(io.StringIO(txt)):
        pid = f"{tour}_{r.get('player_id')}"
        if pid in have:
            continue
        s.add(TenisPlayer(player_id=pid, tour=tour,
                          name=f"{r.get('name_first', '')} {r.get('name_last', '')}".strip(),
                          hand=r.get("hand"), height=_int(r.get("height")), country=r.get("ioc"),
                          birth=r.get("dob")))
        n += 1
    s.add(TenisSourceLog(source=SOURCE, domain="players", url=url, status="ok", records=n, retrieved_at=utcnow()))
    s.flush()
    return {"tour": tour, "jugadores": n}


def ingest_season(s, tour: str, season: int) -> dict:
    url = f"{BASE}/{TOURS[tour]}/{TOURS[tour]}_matches_{season}.csv"
    txt, err = _get(url)
    if txt is None:
        s.add(TenisSourceLog(source=SOURCE, domain="matches", url=url, status="error", records=0,
                             error=err, retrieved_at=utcnow()))
        return {"tour": tour, "season": season, "status": "error", "error": err}
    have = {m.match_id for m in s.execute(select(TenisMatch).where(
        TenisMatch.tour == tour, TenisMatch.season == season)).scalars()}
    n = skipped = 0
    seen = set(have)
    for r in csv.DictReader(io.StringIO(txt)):
        # tourney_id + match_num se repite en eliminatorias por equipos (Copa Davis,
        # Billie Jean King Cup): se desambigua con los jugadores.
        mid = f"{tour}-{r.get('tourney_id')}-{r.get('match_num')}"
        if mid in seen:
            mid = f"{mid}-{r.get('winner_id')}v{r.get('loser_id')}"
        if mid in seen:
            skipped += 1; continue
        seen.add(mid)
        if mid in have:
            continue
        d = r.get("tourney_date") or ""
        if len(d) != 8:
            skipped += 1; continue
        score = (r.get("score") or "").strip()
        retired = "RET" in score.upper()
        gw, gl, sw, sl = parse_score(score, retired)
        serve = {}
        for side in ("w", "l"):
            for c in SERVE_COLS:
                serve[f"{side}_{c}"] = _int(r.get(f"{side}_{c}"))
        s.add(TenisMatch(
            match_id=mid, tour=tour, season=season, tourney_id=r.get("tourney_id"),
            tourney_name=r.get("tourney_name"), tourney_level=r.get("tourney_level"),
            surface=(r.get("surface") or None), draw_size=_int(r.get("draw_size")),
            match_date=date(int(d[:4]), int(d[4:6]), int(d[6:8])), round=r.get("round"),
            best_of=_int(r.get("best_of")), minutes=_int(r.get("minutes")), score=score, retirement=retired,
            winner_id=f"{tour}_{r.get('winner_id')}", loser_id=f"{tour}_{r.get('loser_id')}",
            winner_name=r.get("winner_name"), loser_name=r.get("loser_name"),
            winner_rank=_int(r.get("winner_rank")), loser_rank=_int(r.get("loser_rank")),
            winner_rank_points=_int(r.get("winner_rank_points")), loser_rank_points=_int(r.get("loser_rank_points")),
            winner_age=_float(r.get("winner_age")), loser_age=_float(r.get("loser_age")),
            winner_hand=r.get("winner_hand"), loser_hand=r.get("loser_hand"),
            winner_ht=_int(r.get("winner_ht")), loser_ht=_int(r.get("loser_ht")),
            games_winner=gw, games_loser=gl, games_total=(None if gw is None else gw + gl),
            sets_winner=sw, sets_loser=sl, serve=serve, ingested_at=utcnow()))
        n += 1
    s.add(TenisSourceLog(source=SOURCE, domain="matches", url=url, status="ok", records=n, retrieved_at=utcnow()))
    s.flush()
    return {"tour": tour, "season": season, "partidos": n, "descartados": skipped}


def run(seasons: list[int], tours=("ATP", "WTA"), budget_s: float = 150.0, progress=print) -> dict:
    init_db()
    t0 = time.time()
    done, pending = [], 0
    with session_scope() as s:
        for tour in tours:
            if not s.execute(select(TenisPlayer).where(TenisPlayer.tour == tour).limit(1)).scalars().first():
                progress(f"jugadores {tour}…")
                done.append(ingest_players(s, tour)); s.commit()
        for season in seasons:
            for tour in tours:
                if time.time() - t0 > budget_s:
                    pending += 1; continue
                has = s.execute(select(TenisMatch).where(TenisMatch.tour == tour,
                                                         TenisMatch.season == season).limit(1)).scalars().first()
                if has is not None:
                    continue
                r = ingest_season(s, tour, season); s.commit()
                progress(f"  {tour} {season}: {r.get('partidos', r.get('status'))}")
                done.append(r)
    return {"done": len(done), "pending": pending, "elapsed_s": round(time.time() - t0, 1)}
