"""Backtest WALK-FORWARD de MLB. Nunca se entrena y se prueba en el mismo periodo.

Esquema por temporada de prueba T:
    entrenamiento : todas las temporadas < T-1
    validacion    : T-1   (se usa para elegir Elo y la politica de combinacion)
    prueba        : T     (jamas se toca antes de medir)

Baselines obligatorios: siempre local, Elo, modelo puro, ensemble y aleatorio.
El mercado NO aparece en el backtest porque no existe historico gratuito de
cuotas MLB: se declara como no disponible en vez de simularlo.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from MLB.engine.elo import EloConfig, fit_params, run as run_elo
from MLB.engine.models import MARKETS, TARGET, dispersion, train
from shared.calibration import summary

TOTAL_LINES = [7.5, 8.5, 9.5]


@dataclass
class BacktestConfig:
    first_test: int = 2023
    last_test: int = 2026
    markets: tuple = ("moneyline", "f5_moneyline", "run_line", "total")
    seed: int = 7
    notes: str = ""


def _elo_frame(X: pd.DataFrame, test_season: int, progress=None) -> tuple[pd.DataFrame, EloConfig]:
    seasons = sorted(X.season.unique())
    past = [s for s in seasons if s < test_season]
    valid = past[-1]
    fit = [s for s in past if s < valid]
    cfg = fit_params(X[X.season < test_season], fit, valid) if len(fit) >= 1 else EloConfig()
    if progress:
        progress(f"  elo {test_season}: k={cfg.k} ha={cfg.home_adv} reg={cfg.regress}")
    return run_elo(X, cfg), cfg


def run(X: pd.DataFrame, cfg: BacktestConfig, progress=None) -> dict:
    X = X.sort_values("start_utc").reset_index(drop=True)
    seasons = sorted(X.season.unique())
    tests = [s for s in seasons if cfg.first_test <= s <= cfg.last_test]
    rows, per_season = [], {}
    elo_cfgs = {}

    for T in tests:
        train_seasons = [s for s in seasons if s < T]
        if len(train_seasons) < 2:
            continue
        if progress:
            progress(f"temporada de prueba {T} (entrena con {train_seasons})")
        elo, ecfg = _elo_frame(X, T, progress)
        elo_cfgs[T] = {**{k: v for k, v in asdict(ecfg).items() if k != "fitted_on"},
                       "fitted_on": list(ecfg.fitted_on)}
        Xe = X.merge(elo[["game_id", "p_home_elo"]], on="game_id", how="left")
        Xe["d_elo"] = Xe.p_home_elo - 0.5

        te = Xe[Xe.season == T]
        for market in cfg.markets:
            tgt = TARGET[market]
            tr_mask = Xe.season.isin(train_seasons) & Xe[tgt].notna()
            if tr_mask.sum() < 1000 or te[tgt].notna().sum() < 100:
                continue
            if progress:
                progress(f"  {market}: entrenando con {int(tr_mask.sum())} filas")
            mm = train(Xe, market, train_seasons, seed=cfg.seed)
            sub = te[te[tgt].notna()].copy()
            if market == "total":
                mu = mm.predict_total(sub)
                rec = {"season": T, "market": market, "n": int(len(sub)),
                       "rmse": float(np.sqrt(np.mean((sub[tgt] - mu) ** 2))),
                       "mae": float(np.mean(np.abs(sub[tgt] - mu))),
                       "mean_pred": float(np.mean(mu)), "mean_real": float(sub[tgt].mean()),
                       "baseline_rmse_media": float(np.sqrt(np.mean(
                           (sub[tgt] - Xe[Xe.season.isin(train_seasons)][tgt].mean()) ** 2))),
                       "lines": {}}
                for line in TOTAL_LINES:
                    p = mm.prob_over(sub, line)
                    y = (sub[tgt] > line).astype(float)
                    rec["lines"][str(line)] = summary(y, p)
                    rec["lines"][str(line)].pop("reliability", None)
                per_season.setdefault(T, {})[market] = rec
                rows.append(rec)
                continue

            y = sub[tgt].astype(int)
            p_model = mm.predict_proba(sub)
            members = mm.member_probs(sub)
            disp = dispersion(members)
            p_elo = sub.p_home_elo.values

            strategies = {
                "ensemble": p_model,
                "modelo_forest": members.get("forest", p_model),
                "elo": p_elo,
                "siempre_local": np.full(len(sub), float(y.mean() * 0 + 0.5) + 0.0001),
                "aleatorio": np.random.default_rng(cfg.seed).random(len(sub)),
            }
            rec = {"season": T, "market": market, "n": int(len(sub)),
                   "base_rate": float(y.mean()), "strategies": {},
                   "market_available": False,
                   "market_note": "sin historico gratuito de cuotas MLB: no se compara"}
            for name, p in strategies.items():
                s = summary(y, p)
                if name == "siempre_local":
                    s["accuracy"] = float(y.mean())     # acertar siempre al local
                s.pop("reliability", None)
                rec["strategies"][name] = s
            rec["ensemble_calibration"] = summary(y, p_model)["reliability"]
            rec["mean_dispersion"] = float(np.mean(disp)) if len(disp) else None
            rec["oof_train"] = mm.oof
            rec["weights"] = mm.weights
            per_season.setdefault(T, {})[market] = rec
            rows.append(rec)

    return {"config": asdict(cfg), "elo": elo_cfgs, "per_season": per_season,
            "rows": rows, "summary": _summarise(rows)}


def _summarise(rows: list) -> dict:
    out = {}
    for market in MARKETS:
        rs = [r for r in rows if r["market"] == market]
        if not rs:
            continue
        if market == "total":
            out[market] = {
                "seasons": [r["season"] for r in rs],
                "n": sum(r["n"] for r in rs),
                "rmse": float(np.mean([r["rmse"] for r in rs])),
                "mae": float(np.mean([r["mae"] for r in rs])),
                "baseline_rmse_media": float(np.mean([r["baseline_rmse_media"] for r in rs])),
                "lines": {ln: {
                    "n": sum(r["lines"][ln]["n"] for r in rs),
                    "accuracy": float(np.mean([r["lines"][ln]["accuracy"] for r in rs])),
                    "log_loss": float(np.mean([r["lines"][ln]["log_loss"] for r in rs])),
                    "brier": float(np.mean([r["lines"][ln]["brier"] for r in rs])),
                    "ece": float(np.mean([r["lines"][ln]["ece"] for r in rs])),
                } for ln in rs[0]["lines"]},
            }
            continue
        names = rs[0]["strategies"].keys()
        out[market] = {"seasons": [r["season"] for r in rs],
                       "n": sum(r["n"] for r in rs),
                       "base_rate": float(np.mean([r["base_rate"] for r in rs])),
                       "strategies": {}}
        for n in names:
            w = np.array([r["n"] for r in rs], dtype=float)
            g = lambda k: float(np.average([r["strategies"][n][k] for r in rs], weights=w))
            out[market]["strategies"][n] = {"accuracy": g("accuracy"), "log_loss": g("log_loss"),
                                            "brier": g("brier"), "ece": g("ece")}
    return out
