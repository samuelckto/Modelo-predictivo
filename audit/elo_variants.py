"""AUDITORIA DEL ELO (§9). Por que un Elo simple gana al machine learning.

Se prueban variantes SIN tocar la regla temporal: cada variante se ajusta con las
temporadas anteriores y se mide en las de prueba, que nunca se usan para elegir.

Variantes:
  base            : Elo con ventaja de local, MOV logaritmico y regresion
  sin_mov         : sin multiplicador por margen de carreras
  sin_regresion   : no se regresa a 1500 entre temporadas
  sin_ventaja     : sin ventaja de local
  k_alto / k_bajo : sensibilidad al ritmo de actualizacion
  decay           : la ventaja de local decae dentro de la temporada
  descanso        : ajuste por dias de descanso (dato del calendario, no del partido)
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402

from MLB.engine.elo import EloConfig, fit_params, run as run_elo   # noqa: E402
from shared.calibration import summary                   # noqa: E402
from shared.paths import MLB_PROCESSED_DIR               # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True, parents=True)
TESTS = [2023, 2024, 2025, 2026]


def elo_rest(games: pd.DataFrame, cfg: EloConfig, per_day: float) -> pd.DataFrame:
    """Elo con un ajuste por descanso. El descanso sale del calendario, que se
    publica con meses de antelacion: no es informacion del partido."""
    base = run_elo(games, cfg)
    g = games[["game_id", "h_rest_days", "a_rest_days"]].copy()
    m = base.merge(g, on="game_id", how="left")
    dr = (m.h_rest_days.fillna(1) - m.a_rest_days.fillna(1)).clip(-3, 3)
    z = np.log(m.p_home_elo / (1 - m.p_home_elo)) + per_day * dr
    m["p_home_elo"] = 1 / (1 + np.exp(-z))
    return m[["game_id", "p_home_elo"]]


def evaluate(X: pd.DataFrame) -> dict:
    res = {}
    for T in TESTS:
        seasons = sorted(int(s) for s in X.season.unique())
        past = [s for s in seasons if s < T]
        fitted = fit_params(X[X.season < T], past[:-1], past[-1])
        variants = {
            "base(ajustado)": fitted,
            "sin_mov": replace(fitted, mov=False),
            "sin_regresion": replace(fitted, regress=0.0),
            "sin_ventaja_local": replace(fitted, home_adv=0.0),
            "k_alto": replace(fitted, k=8.0),
            "k_bajo": replace(fitted, k=1.5),
            "k4_ha24_reg25(defecto)": EloConfig(k=4.0, home_adv=24.0, regress=0.25),
        }
        te_mask = (X.season == T) & X.y_home_win.notna()
        y = X.loc[te_mask, "y_home_win"]
        row = {}
        for name, cfg in variants.items():
            p = run_elo(X, cfg)
            m = X.loc[te_mask, ["game_id"]].merge(p, on="game_id")
            s = summary(y.values, m.p_home_elo.values)
            row[name] = {k: s[k] for k in ("accuracy", "log_loss", "brier", "ece")}
        for per_day in (0.02, 0.05):
            p = elo_rest(X, fitted, per_day)
            m = X.loc[te_mask, ["game_id"]].merge(p, on="game_id")
            s = summary(y.values, m.p_home_elo.values)
            row[f"descanso_{per_day}"] = {k: s[k] for k in
                                          ("accuracy", "log_loss", "brier", "ece")}
        row["_cfg_ajustado"] = {"k": fitted.k, "home_adv": fitted.home_adv,
                                "regress": fitted.regress}
        row["_n"] = int(te_mask.sum())
        res[T] = row
    names = [k for k in res[TESTS[0]] if not k.startswith("_")]
    tot = {}
    w = np.array([res[T]["_n"] for T in TESTS], dtype=float)
    for nme in names:
        tot[nme] = {k: float(np.average([res[T][nme][k] for T in TESTS], weights=w))
                    for k in ("accuracy", "log_loss", "brier", "ece")}
    return {"per_season": res, "total": tot, "n": int(w.sum())}


if __name__ == "__main__":
    X = pd.read_parquet(MLB_PROCESSED_DIR / "features.parquet")
    r = evaluate(X)
    (OUT / "elo_variants.json").write_text(json.dumps(r, indent=1, default=str))
    print(f"n total = {r['n']}\n")
    print(f"{'variante':<26}{'acc':>8}{'logloss':>10}{'brier':>9}{'ECE':>8}")
    for k, v in sorted(r["total"].items(), key=lambda kv: kv[1]["log_loss"]):
        print(f"{k:<26}{v['accuracy']*100:>7.2f}%{v['log_loss']:>10.4f}"
              f"{v['brier']:>9.4f}{v['ece']:>8.4f}")
    print("\nconfiguracion elegida por validacion temporal, por temporada:")
    for T in TESTS:
        print(f"  {T}: {r['per_season'][T]['_cfg_ajustado']}")
