"""Calificador automatico de resultados NFL usando ESPN (gratis, sin cuota).

Actualiza marcadores en NFL/database/nflpred.sqlite3 y en NFL/database/nfl_markets.sqlite3.
"""
from __future__ import annotations

import json
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path
from sqlalchemy import select

from NFL.adapter import db_path as nfl_db_path
from NFL.markets.db import NflMarketPrediction, session_scope

TEAM_MAP = {
    "WSH": "WAS", "LAR": "LA"
}

def fetch_espn_nfl_week(week: int = 1, season: int = 2026) -> list[dict]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?week={week}&year={season}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"[!] Error consultando ESPN NFL semana {week}: {e}")
        return []

    results = []
    for ev in d.get("events", []):
        st = ev.get("status", {}).get("type", {})
        if not st.get("completed"):
            continue
        comps = ev.get("competitions", [])[0].get("competitors", [])
        h = [c for c in comps if c.get("homeAway") == "home"][0]
        a = [c for c in comps if c.get("homeAway") == "away"][0]
        h_team = h.get("team", {}).get("abbreviation", "").upper()
        a_team = a.get("team", {}).get("abbreviation", "").upper()
        h_team = TEAM_MAP.get(h_team, h_team)
        a_team = TEAM_MAP.get(a_team, a_team)
        h_score = int(h.get("score", 0))
        a_score = int(a.get("score", 0))
        results.append({
            "season": season, "week": week,
            "home_team": h_team, "away_team": a_team,
            "home_score": h_score, "away_score": a_score,
            "winner": h_team if h_score > a_score else a_team if a_score > h_score else "TIE"
        })
    return results


def score_nfl(weeks: list[int] = [1, 2]) -> dict:
    """Descarga marcadores de ESPN y califica nflpred.sqlite3 y nfl_markets.sqlite3."""
    out = {"espn_games": 0, "nflpred_games": 0, "nflpred_preds": 0, "markets_scored": 0}
    all_results = []
    for w in weeks:
        res = fetch_espn_nfl_week(w)
        all_results.extend(res)
    out["espn_games"] = len(all_results)
    if not all_results:
        return out

    # 1. Actualizar nflpred.sqlite3
    dbp = nfl_db_path()
    if dbp.exists():
        try:
            with sqlite3.connect(str(dbp), timeout=15) as conn:
                cur = conn.cursor()
                for g in all_results:
                    # Actualizar games
                    cur.execute(
                        "UPDATE games SET home_score = ?, away_score = ? "
                        "WHERE season = ? AND week = ? AND home_team = ? AND away_team = ?",
                        (g["home_score"], g["away_score"], g["season"], g["week"], g["home_team"], g["away_team"])
                    )
                    if cur.rowcount > 0:
                        out["nflpred_games"] += cur.rowcount
                    
                    # Actualizar predictions (Moneyline)
                    winner = g["winner"]
                    if winner != "TIE":
                        # Acertaron
                        cur.execute(
                            "UPDATE predictions SET correct = 1, points_earned = points "
                            "WHERE season = ? AND week = ? AND home_team = ? AND away_team = ? AND pick = ?",
                            (g["season"], g["week"], g["home_team"], g["away_team"], winner)
                        )
                        out["nflpred_preds"] += cur.rowcount
                        # Fallaron
                        cur.execute(
                            "UPDATE predictions SET correct = 0, points_earned = 0 "
                            "WHERE season = ? AND week = ? AND home_team = ? AND away_team = ? AND pick != ?",
                            (g["season"], g["week"], g["home_team"], g["away_team"], winner)
                        )
                        out["nflpred_preds"] += cur.rowcount
                conn.commit()
        except Exception as e:
            out["nflpred_error"] = str(e)

    # 2. Actualizar nfl_markets.sqlite3
    try:
        with session_scope() as s:
            for g in all_results:
                preds = s.execute(select(NflMarketPrediction).where(
                    NflMarketPrediction.season == g["season"],
                    NflMarketPrediction.week == g["week"],
                    NflMarketPrediction.home_team == g["home_team"],
                    NflMarketPrediction.away_team == g["away_team"],
                    NflMarketPrediction.status != "superseded"
                )).scalars().all()

                for p in preds:
                    h_score = g["home_score"]
                    a_score = g["away_score"]
                    if p.market_type == "total":
                        actual = float(h_score + a_score)
                        p.actual_value = actual
                        if p.line is not None:
                            if actual == p.line:
                                p.result, p.correct = "push", None
                            else:
                                is_over = actual > p.line
                                pick_is_over = (p.selection or "").startswith("OVER")
                                p.correct = bool(is_over == pick_is_over)
                                p.result = "win" if p.correct else "loss"
                            out["markets_scored"] += 1
                    elif p.market_type == "spread":
                        actual = float(h_score - a_score)
                        p.actual_value = actual
                        if p.line is not None:
                            diff = actual + p.line
                            if diff == 0:
                                p.result, p.correct = "push", None
                            else:
                                home_covers = diff > 0
                                pick_is_home = (p.selection or "").startswith(g["home_team"])
                                p.correct = bool(home_covers == pick_is_home)
                                p.result = "win" if p.correct else "loss"
                            out["markets_scored"] += 1
            s.commit()
    except Exception as e:
        out["markets_error"] = str(e)

    return out


if __name__ == "__main__":
    print(score_nfl())
