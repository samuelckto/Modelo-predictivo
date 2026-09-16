"""Genera las predicciones walk-forward PARTIDO A PARTIDO (base de §10, §11, §12).

Variante limpia (sin features del abridor), que es la que gatea el sistema.
Guarda tambien Elo, que no usa abridor en absoluto.
Reanudable: un parquet por (temporada, mercado).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402

from MLB.engine.models import TARGET, dispersion, train  # noqa: E402
from audit.models_compare import prep                    # noqa: E402
from shared.paths import MLB_PROCESSED_DIR               # noqa: E402

PRE = Path(__file__).resolve().parent / "preds"
PRE.mkdir(exist_ok=True, parents=True)
SEASONS = [2023, 2024, 2025, 2026]
MARKETS = ["moneyline", "f5_moneyline", "run_line"]


def one(T: int, market: str) -> pd.DataFrame:
    X = pd.read_parquet(MLB_PROCESSED_DIR / "features.parquet")
    Xe, cfg, train_seasons = prep(X, T)
    tgt = TARGET[market]
    mm = train(Xe, market, train_seasons, exclude_starter=True)
    te = Xe[(Xe.season == T) & Xe[tgt].notna()].copy()
    members = mm.member_probs(te)
    base = float(Xe[Xe.season.isin(train_seasons)][tgt].mean())   # tasa base PASADA
    return pd.DataFrame({
        "season": T, "market": market, "game_id": te.game_id.values,
        "game_date": te.game_date.values, "start_utc": te.start_utc.values,
        "home": te.home_abbr.values, "away": te.away_abbr.values,
        "y_home": te[tgt].astype(int).values,
        "p_home_model": mm.predict_proba(te),
        "p_home_forest": members.get("forest", np.nan),
        "p_home_elo": te.p_home_elo.values,
        "dispersion": dispersion(members),
        "base_rate_train": base,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=2.4)
    a = ap.parse_args()
    todo = [(T, m) for T in SEASONS for m in MARKETS
            if not (PRE / f"{T}_{m}.parquet").exists()]
    deadline = time.time() + a.minutes * 60
    for T, m in todo:
        if time.time() > deadline:
            print(f"tiempo agotado; quedan {len(todo)}", flush=True)
            return
        print(f"→ {T} {m}", flush=True)
        one(T, m).to_parquet(PRE / f"{T}_{m}.parquet", index=False)
    print("completo", flush=True)


if __name__ == "__main__":
    main()
