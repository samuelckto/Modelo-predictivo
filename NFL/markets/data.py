"""Lectura (solo lectura) de los datos del motor NFL para los mercados propios."""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.paths import NFL_HOME

FEATURES_PARQUET = NFL_HOME / "data" / "processed" / "features.parquet"

# Columnas que NUNCA pueden ser feature: objetivos, resultados y mercado de cierre
# (las lineas de nflverse son de cierre y no tienen timestamp previo al kickoff).
TARGET_COLS = {"home_score", "away_score", "result", "total", "overtime", "home_win"}
MARKET_COLS = {"home_moneyline", "away_moneyline", "spread_line", "total_line",
               "market_home_prob", "market_spread_home", "market_source",
               "market_captured_at", "line_move", "n_odds_snapshots"}
CONTEXT = ["rest_diff", "home_short_rest", "away_short_rest", "is_dome", "is_grass",
           "temp_f", "wind_mph", "cold", "windy", "primetime", "is_thursday", "is_monday",
           "is_playoff", "div_game", "week_num", "home_rest", "away_rest"]
TEAM_METRICS = ["epa_off", "epa_def", "pass_epa_off", "rush_epa_off", "pass_epa_def",
                "rush_epa_def", "success_off", "success_def", "explosive_off", "explosive_def",
                "sack_rate_off", "sack_rate_def", "pressure_rate_def", "turnovers_off",
                "turnovers_def", "third_down_off", "third_down_def", "redzone_td_off",
                "redzone_td_def", "yards_off", "yards_def", "cpoe_off", "points_for",
                "points_against", "won", "point_diff", "turnover_margin"]
QB = ["qb_quality", "qb_experience", "qb_listed_out", "qb_questionable", "qb_change"]
INJ = ["inj_out_weighted", "inj_out_count", "inj_q_count", "inj_data_available"]


def load() -> pd.DataFrame:
    X = pd.read_parquet(FEATURES_PARQUET)
    X["kickoff_utc"] = pd.to_datetime(X["kickoff_utc"])
    X["margin"] = X["home_score"] - X["away_score"]
    X["total_pts"] = X["home_score"] + X["away_score"]
    # familias derivadas para totales: nivel conjunto de ambos ataques/defensas
    for m in TEAM_METRICS:
        for pre in ("", "l5_"):
            h, a = f"home_{pre}{m}", f"away_{pre}{m}"
            if h in X.columns and a in X.columns:
                X[f"sum_{pre}{m}"] = X[h] + X[a]
    for pre in ("", "l5_"):
        if f"home_{pre}points_for" in X.columns:
            X[f"pace_{pre}proxy"] = (X[f"home_{pre}points_for"] + X[f"home_{pre}points_against"] +
                                     X[f"away_{pre}points_for"] + X[f"away_{pre}points_against"]) / 2
    X["elo_sum"] = X["elo_home"] + X["elo_away"]
    return X


def families(X: pd.DataFrame) -> dict[str, list[str]]:
    """Familias de features (todas as-of, ninguna de mercado ni de resultado)."""
    cols = set(X.columns)
    fam = {
        "elo": [c for c in ("elo_diff", "elo_prob_home", "elo_home", "elo_away", "elo_sum") if c in cols],
        "diff_epa": [c for c in cols if c.startswith("diff_") and "epa" in c],
        "diff_eff": [c for c in cols if c.startswith("diff_") and any(
            k in c for k in ("success", "explosive", "third_down", "redzone", "yards", "cpoe"))],
        "diff_pressure": [c for c in cols if c.startswith("diff_") and any(
            k in c for k in ("sack", "pressure", "turnover"))],
        "diff_results": [c for c in cols if c.startswith("diff_") and any(
            k in c for k in ("points_for", "points_against", "won", "point_diff"))],
        "sum_epa": [c for c in cols if c.startswith("sum_") and "epa" in c],
        "sum_eff": [c for c in cols if c.startswith("sum_") and any(
            k in c for k in ("success", "explosive", "third_down", "redzone", "yards", "cpoe"))],
        "sum_pressure": [c for c in cols if c.startswith("sum_") and any(
            k in c for k in ("sack", "pressure", "turnover"))],
        "sum_scoring": [c for c in cols if c.startswith(("sum_points", "sum_l5_points", "pace_"))],
        "level_epa": [c for c in cols if c.startswith(("home_", "away_")) and "epa" in c and "src" not in c],
        "level_scoring": [c for c in cols if c.startswith(("home_", "away_")) and any(
            k in c for k in ("points_for", "points_against"))],
        "qb": [f"{s}_{q}" for s in ("home", "away") for q in QB if f"{s}_{q}" in cols] +
              [c for c in cols if c.startswith("diff_qb")],
        "injuries": [f"{s}_{q}" for s in ("home", "away") for q in INJ if f"{s}_{q}" in cols] +
                    [c for c in cols if c.startswith("diff_inj")],
        "context": [c for c in CONTEXT if c in cols],
        "h2h": [c for c in ("h2h_margin", "h2h_weight") if c in cols],
        "history": [c for c in ("home_games_played_season", "away_games_played_season",
                                "home_has_history", "away_has_history") if c in cols],
    }
    for k, v in fam.items():
        bad = (set(v) & (TARGET_COLS | MARKET_COLS))
        assert not bad, f"familia {k} contiene columnas prohibidas: {bad}"
        fam[k] = sorted(set(v))
    return fam


def numeric(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return X[cols].apply(pd.to_numeric, errors="coerce").astype(float)
