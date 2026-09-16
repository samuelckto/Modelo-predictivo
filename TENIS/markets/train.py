"""Entrena los modelos de produccion de tenis (uno por circuito).

TENIS_ATP_WINNER_v1 / TENIS_WTA_WINNER_v1: logistica sobre Elo, Elo de superficie
y ranking (el motor que gano en el walk-forward), con su calibrador. De la
probabilidad calibrada se derivan juegos y handicap con el modelo punto a punto,
asi que no hay un modelo separado para esos mercados: son el mismo objeto.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib                                            # noqa: E402
import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402
from sklearn.impute import SimpleImputer                 # noqa: E402
from sklearn.linear_model import LogisticRegression      # noqa: E402
from sklearn.pipeline import Pipeline                    # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402

from TENIS.markets.db import TenisModelVersion, init_db, session_scope  # noqa: E402
from TENIS.markets.features import FEATURES_PARQUET, FEATURE_VERSION, numeric  # noqa: E402
from TENIS.markets.gating import decide                  # noqa: E402
from shared.paths import TENIS_MODELS_DIR, TENIS_OUT_DIR, assert_writable  # noqa: E402
from shared.timeutil import json_safe, utcnow            # noqa: E402

VERSION = "v1"
COLS = ["d_elo", "d_elo_surface", "log_rank_diff", "d_pts", "best_of_5"]


def train_tour(tour: str, X: pd.DataFrame, gate: dict) -> dict:
    lines = json.loads((TENIS_OUT_DIR / "research_lines.json").read_text())["tours"][tour]
    T = X[(X.tour == tour) & X.p1_win.notna() & (~X.retirement.fillna(False))]
    cols = [c for c in COLS if c in T.columns]
    m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                  ("m", LogisticRegression(C=0.5, max_iter=2000))])
    m.fit(numeric(T, cols), T.p1_win.astype(int).values)
    art = {"tour": tour, "market": "winner", "model": m, "features": cols, "version": VERSION,
           "name": f"TENIS_{tour}_WINNER_{VERSION}", "engine": lines["motor_elegido"],
           "calibration": {"method": lines["calibracion_elegida"], "params": lines["calibrador"],
                           "fitted_on": "walk-forward OOS 2015-2019"},
           "avg_spw": lines["avg_spw"], "threshold": gate[tour]["winner"].get("threshold"),
           "n_train": int(len(T)), "train_seasons": sorted(int(s) for s in T.season.unique()),
           "feature_version": FEATURE_VERSION, "trained_at": utcnow()}
    TENIS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = assert_writable(TENIS_MODELS_DIR / f"{tour}_winner_{VERSION}.joblib", "TENIS")
    joblib.dump(art, path)
    with session_scope() as s:
        s.query(TenisModelVersion).filter(TenisModelVersion.tour == tour,
                                          TenisModelVersion.is_production.is_(True)).update({"is_production": False})
        for market in ("winner", "total_games", "handicap_games"):
            g = gate[tour][market]
            s.add(TenisModelVersion(
                name=(art["name"] if market == "winner" else f"TENIS_{tour}_{market.upper()}_{VERSION}"),
                market=market, tour=tour, version=VERSION,
                algorithm=("logistica Elo + calibracion" if market == "winner"
                           else "derivado del punto a punto sobre la probabilidad calibrada"),
                training_period=f"{art['train_seasons'][0]}-{art['train_seasons'][-1]}",
                validation_period="walk-forward: seleccion 2015-2019, holdout 2020-2026",
                features=cols, calibration=json_safe(art["calibration"]),
                metrics=json_safe({"gating": g}), artifact_path=str(path), is_production=True,
                notes=g["reason"], created_at=utcnow()))
    return {"name": art["name"], "n_train": art["n_train"], "features": len(cols),
            "modo": {k: gate[tour][k]["mode"] for k in ("winner", "total_games", "handicap_games")}}


def main():
    init_db()
    X = pd.read_parquet(FEATURES_PARQUET)
    gate = decide()
    out = {t: train_tour(t, X, gate) for t in ("ATP", "WTA")}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
