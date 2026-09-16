"""Features NBA as-of (point-in-time).

Regla: para el partido del dia D solo entran logs con available_at < D 00:00
(es decir, partidos de dias ANTERIORES). Sin hora de tipoff no se usa nada del
mismo dia. Cada feature se calcula con `shared.asof.rolling_asof`-style sumas
acumuladas por temporada y ventanas recientes (ultimos 10 partidos).

Familias: elo, strength (ratings of/def por posesion), pace, shooting, turnovers,
rebounding, free_throws, recent_form, rest, context, home_away.
Lesiones/quintetos: SIN historico con timestamp -> no son features (documentado).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from NBA.markets.db import NbaGame, NbaTeamGameLog, init_db, session_scope
from shared.paths import NBA_OUT_DIR

FEATURES_PARQUET = NBA_OUT_DIR / "features.parquet"
FEATURE_VERSION = "nba_features_v1"
LAST_N = 10
PRIOR_GAMES = 10.0          # encogimiento hacia la temporada anterior al inicio

# metricas por partido (derivadas de pbpstats) que se promedian as-of
def _metrics(st: dict, opp: dict, pts: int, opp_pts: int, poss: int, opp_poss: int) -> dict:
    g = lambda d, k: float(d.get(k) or 0.0)
    fga = g(st, "FG2A") + g(st, "FG3A")
    fgm = g(st, "FG2M") + g(st, "FG3M")
    fta = g(st, "FTA")
    ofga = g(opp, "FG2A") + g(opp, "FG3A")
    ofgm = g(opp, "FG2M") + g(opp, "FG3M")
    tov = g(st, "Turnovers"); otov = g(opp, "Turnovers")
    orb = g(st, "OffRebounds"); drb = g(st, "DefRebounds")
    oorb = g(opp, "OffRebounds"); odrb = g(opp, "DefRebounds")
    p = max(poss, 1); op = max(opp_poss, 1)
    return {
        "ortg": 100.0 * pts / p, "drtg": 100.0 * opp_pts / op,
        "pace": (p + op) / 2.0,
        "efg": (fgm + 0.5 * g(st, "FG3M")) / max(fga, 1),
        "ts": pts / max(2 * (fga + 0.44 * fta), 1),
        "fg3_rate": g(st, "FG3A") / max(fga, 1), "fg3_pct": g(st, "FG3M") / max(g(st, "FG3A"), 1),
        "ft_rate": fta / max(fga, 1),
        "tov_rate": tov / p, "opp_tov_rate": otov / op,
        "orb_pct": orb / max(orb + odrb, 1), "drb_pct": drb / max(drb + oorb, 1),
        "opp_efg": (ofgm + 0.5 * g(opp, "FG3M")) / max(ofga, 1),
        "rim_rate": g(st, "AtRimFGA") / max(fga, 1), "rim_acc": g(st, "AtRimFGM") / max(g(st, "AtRimFGA"), 1),
        "opp_rim_rate": g(opp, "AtRimFGA") / max(ofga, 1),
        "assists": g(st, "Assists"), "steals": g(st, "Steals"), "blocks": g(st, "Blocks"),
        "fouls": g(st, "Fouls"), "shot_quality": g(st, "ShotQualityAvg"),
        "pts": float(pts), "opp_pts": float(opp_pts), "margin": float(pts - opp_pts),
        "won": 1.0 if pts > opp_pts else 0.0,
    }


METRIC_KEYS = None


def _elo(games: pd.DataFrame, k: float = 20.0, home_adv: float = 70.0, regress: float = 0.25) -> pd.DataFrame:
    """Elo pre-partido. Parametros habituales; se recalculan en research."""
    r = {}
    last_season = {}
    out = []
    for g in games.itertuples():
        for t in (g.home_team_id, g.away_team_id):
            if t not in r:
                r[t] = 1500.0
            if last_season.get(t) is not None and last_season[t] != g.season:
                r[t] = 1500.0 + (r[t] - 1500.0) * (1 - regress)
            last_season[t] = g.season
        eh, ea = r[g.home_team_id] + home_adv, r[g.away_team_id]
        p = 1 / (1 + 10 ** ((ea - eh) / 400))
        out.append((g.game_id, r[g.home_team_id], r[g.away_team_id], p))
        if g.home_points is not None and not np.isnan(g.home_points):
            res = 1.0 if g.home_points > g.away_points else 0.0
            mov = abs(g.home_points - g.away_points)
            mult = np.log(mov + 1) * (2.2 / ((eh - ea) * 0.001 + 2.2)) if res == 1 else \
                np.log(mov + 1) * (2.2 / ((ea - eh) * 0.001 + 2.2))
            d = k * mult * (res - p)
            r[g.home_team_id] += d; r[g.away_team_id] -= d
    return pd.DataFrame(out, columns=["game_id", "elo_home", "elo_away", "elo_prob_home"])


def build(save: bool = True, progress=None) -> pd.DataFrame:
    init_db()
    with session_scope() as s:
        games = pd.DataFrame([{
            "game_id": g.game_id, "season": g.season, "season_type": g.season_type,
            "game_date": pd.Timestamp(g.game_date), "start_utc": g.start_utc,
            "home_team_id": g.home_team_id, "away_team_id": g.away_team_id,
            "home_abbr": g.home_abbr, "away_abbr": g.away_abbr,
            "home_points": g.home_points, "away_points": g.away_points,
            "home_poss": g.home_poss, "away_poss": g.away_poss}
            for g in s.execute(select(NbaGame)).scalars()])
        logs = {(l.game_id, l.team_id): l for l in s.execute(select(NbaTeamGameLog)).scalars()}
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    games["home_points"] = pd.to_numeric(games.home_points, errors="coerce")
    games["away_points"] = pd.to_numeric(games.away_points, errors="coerce")

    # --- metricas por equipo-partido --------------------------------------------
    rows = []
    for g in games.itertuples():
        lh, la = logs.get((g.game_id, g.home_team_id)), logs.get((g.game_id, g.away_team_id))
        if pd.isna(g.home_points) or lh is None or la is None:
            # partido futuro (o sin logs): fila sin metricas; recibe las features
            # as-of de los partidos anteriores y nunca aporta a los siguientes
            for tid, home in ((g.home_team_id, 1), (g.away_team_id, 0)):
                rows.append({"game_id": g.game_id, "team_id": tid, "season": g.season,
                             "game_date": g.game_date, "is_home": home, "available_at": pd.NaT})
            continue
        for tid, mine, other, pts, opts, poss, oposs, home in (
                (g.home_team_id, lh.stats, la.stats, g.home_points, g.away_points, g.home_poss, g.away_poss, 1),
                (g.away_team_id, la.stats, lh.stats, g.away_points, g.home_points, g.away_poss, g.home_poss, 0)):
            m = _metrics(mine, other, int(pts), int(opts), int(poss or 0), int(oposs or 0))
            m.update({"game_id": g.game_id, "team_id": tid, "season": g.season,
                      "game_date": g.game_date, "is_home": home,
                      "available_at": pd.Timestamp(mine and lh.available_at if tid == g.home_team_id else la.available_at)})
            rows.append(m)
    tg = pd.DataFrame(rows).sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    keys = [c for c in tg.columns if c not in ("game_id", "team_id", "season", "game_date", "is_home", "available_at")]

    # --- as-of por equipo: acumulado de temporada (desplazado) y ultimos N ----------
    feats = []
    for tid, d in tg.groupby("team_id", sort=False):
        d = d.sort_values(["game_date", "game_id"]).copy()
        f = pd.DataFrame(index=d.index)
        f["game_id"] = d.game_id.values
        f["team_id"] = tid
        prev_season_mean = {}
        for season, ds in d.groupby("season", sort=True):
            idx = ds.index
            f.loc[idx, "games_played"] = np.arange(len(ds))
            for k in keys:
                v = ds[k].values.astype(float)
                ok = ~np.isnan(v)
                n = np.concatenate([[0], np.cumsum(ok)[:-1]])          # partidos previos CON datos
                cum = np.concatenate([[0.0], np.nancumsum(v)[:-1]])
                std = np.where(n > 0, cum / np.maximum(n, 1), np.nan)
                prior = prev_season_mean.get(k)
                if prior is not None:
                    w = n / (n + PRIOR_GAMES)
                    std = np.where(n > 0, w * std + (1 - w) * prior, prior)
                f.loc[idx, f"s_{k}"] = std
                # ultimos N (solo temporada actual, desplazado)
                ser = pd.Series(v)
                f.loc[idx, f"l{LAST_N}_{k}"] = ser.shift(1).rolling(LAST_N, min_periods=3).mean().values
            for k in keys:
                prev_season_mean[k] = float(ds[k].mean())
        # descanso
        dates = ds_dates = d.game_date.values
        prev = pd.Series(d.game_date.values).shift(1)
        rest = (pd.Series(d.game_date.values) - prev).dt.days.values
        f["rest_days"] = rest
        f["b2b"] = (rest == 1).astype(float)
        f["games_last7"] = [int(((d.game_date.values < dt) & (d.game_date.values >= dt - np.timedelta64(7, "D"))).sum())
                            for dt in d.game_date.values]
        feats.append(f)
    F = pd.concat(feats)
    F["rest_days"] = F.rest_days.clip(upper=10)

    # --- unir a partidos ---------------------------------------------------------
    X = games.copy()
    Fh = F.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team_id": "home_team_id"})
    Fa = F.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team_id": "away_team_id"})
    X = X.merge(Fh, on=["game_id", "home_team_id"], how="left").merge(Fa, on=["game_id", "away_team_id"], how="left")
    elo = _elo(games)
    X = X.merge(elo, on="game_id", how="left")
    new = {"elo_diff": X.elo_home - X.elo_away, "elo_sum": X.elo_home + X.elo_away}
    for k in keys + ["games_played", "rest_days", "b2b", "games_last7"]:
        for pre in ("s_", f"l{LAST_N}_"):
            hc, ac = f"h_{pre}{k}", f"a_{pre}{k}"
            if hc in X.columns and ac in X.columns:
                new[f"d_{pre}{k}"] = X[hc] - X[ac]
                new[f"sum_{pre}{k}"] = X[hc] + X[ac]
    for k in ("rest_days", "b2b", "games_last7", "games_played"):
        if f"h_{k}" in X.columns:
            new[f"d_{k}"] = X[f"h_{k}"] - X[f"a_{k}"]
    new["is_playoff"] = (X.season_type == "Playoffs").astype(float)
    new["month"] = X.game_date.dt.month.astype(float)
    new["margin"] = X.home_points - X.away_points
    new["total_pts"] = X.home_points + X.away_points
    new["home_win"] = (new["margin"] > 0).astype(float).where(new["margin"].notna())
    new["cutoff_utc"] = X.game_date                          # 00:00 del dia del partido
    new["max_source_available_at"] = X.game_date - pd.Timedelta(seconds=1)
    X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
    if save:
        NBA_OUT_DIR.mkdir(parents=True, exist_ok=True)
        X.to_parquet(FEATURES_PARQUET, index=False)
    if progress:
        progress(f"features NBA: {X.shape}")
    return X


TARGET_COLS = {"home_points", "away_points", "home_poss", "away_poss", "margin", "total_pts", "home_win"}
ID_COLS = {"game_id", "season", "season_type", "game_date", "start_utc", "home_team_id", "away_team_id",
           "home_abbr", "away_abbr", "cutoff_utc", "max_source_available_at"}


def families(X: pd.DataFrame) -> dict[str, list[str]]:
    cols = set(X.columns) - TARGET_COLS - ID_COLS
    def pick(*subs, prefixes=("d_", "sum_")):
        return sorted(c for c in cols if c.startswith(prefixes) and any(s in c for s in subs))
    fam = {
        "elo": [c for c in ("elo_home", "elo_away", "elo_diff", "elo_prob_home", "elo_sum") if c in cols],
        "strength": pick("ortg", "drtg", "margin", "won", "_pts"),
        "pace": pick("pace"),
        "shooting": pick("efg", "ts", "fg3", "rim_", "shot_quality", "opp_efg", "opp_rim"),
        "turnovers": pick("tov"),
        "rebounding": pick("orb", "drb"),
        "free_throws": pick("ft_rate"),
        "defense_misc": pick("steals", "blocks", "fouls", "assists"),
        "rest": [c for c in cols if c.endswith(("rest_days", "b2b", "games_last7")) and c.startswith(("h_", "a_", "d_"))],
        "context": [c for c in ("is_playoff", "month", "h_games_played", "a_games_played", "d_games_played") if c in cols],
    }
    for k, v in fam.items():
        assert not (set(v) & TARGET_COLS), k
    return fam


def numeric(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return X[cols].apply(pd.to_numeric, errors="coerce").astype(float)
