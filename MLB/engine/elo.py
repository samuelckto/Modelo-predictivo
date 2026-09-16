"""Elo de beisbol. Baseline independiente del modelo de features.

Parametros (K, ventaja de local, regresion entre temporadas) NO se fijan a ojo:
`fit_params` los elige por validacion temporal sobre temporadas pasadas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared.calibration import log_loss


@dataclass
class EloConfig:
    k: float = 4.0
    home_adv: float = 24.0
    regress: float = 0.25          # hacia 1500 al cambiar de temporada
    mov: bool = True
    base: float = 1500.0
    fitted_on: tuple = ()
    rationale: str = ""


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def run(games: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    """Devuelve game_id, p_home_elo, elo_home_pre, elo_away_pre.

    Recorre en orden cronologico y solo usa partidos ANTERIORES: la calificacion
    previa a cada partido es, por construccion, informacion pasada.
    """
    g = games.sort_values("start_utc")
    r: dict[int, float] = {}
    last_season = None
    out = []
    for row in g.itertuples():
        if last_season is not None and row.season != last_season:
            for t in r:
                r[t] = cfg.base + (r[t] - cfg.base) * (1 - cfg.regress)
        last_season = row.season
        ra = r.get(row.home_team_id, cfg.base) + cfg.home_adv
        rb = r.get(row.away_team_id, cfg.base)
        p = _expected(ra, rb)
        out.append((row.game_id, p, ra - cfg.home_adv, rb))
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue
        y = 1.0 if row.home_score > row.away_score else 0.0
        mult = 1.0
        if cfg.mov:
            margin = abs(row.home_score - row.away_score)
            mult = np.log1p(margin)          # 1 carrera pesa menos que 6
        delta = cfg.k * mult * (y - p)
        r[row.home_team_id] = r.get(row.home_team_id, cfg.base) + delta
        r[row.away_team_id] = r.get(row.away_team_id, cfg.base) - delta
    return pd.DataFrame(out, columns=["game_id", "p_home_elo", "elo_home_pre", "elo_away_pre"])


GRID_K = [2.0, 3.0, 4.0, 6.0, 8.0]
GRID_HA = [12.0, 18.0, 24.0, 30.0]
GRID_REG = [0.0, 0.25, 0.5]


def fit_params(games: pd.DataFrame, fit_seasons: list[int], valid_season: int) -> EloConfig:
    """Rejilla evaluada SOLO en `valid_season`, que no se usa para nada mas."""
    sub = games[games.season.isin(list(fit_seasons) + [valid_season])]
    best, best_ll = None, np.inf
    tried = []
    for k in GRID_K:
        for ha in GRID_HA:
            for rg in GRID_REG:
                cfg = EloConfig(k=k, home_adv=ha, regress=rg)
                p = run(sub, cfg)
                m = sub[sub.season == valid_season][["game_id", "y_home_win"]].merge(
                    p, on="game_id").dropna(subset=["y_home_win"])
                ll = log_loss(m.y_home_win, m.p_home_elo)
                tried.append({"k": k, "home_adv": ha, "regress": rg,
                              "log_loss": round(float(ll), 5)})
                if ll < best_ll:
                    best, best_ll = cfg, ll
    best.fitted_on = tuple(fit_seasons)
    best.rationale = (f"rejilla de {len(tried)} combinaciones validada en {valid_season} "
                      f"(log loss {best_ll:.4f}); las temporadas de prueba nunca se tocaron")
    return best
