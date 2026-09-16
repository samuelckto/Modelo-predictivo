"""Entrena los modelos de produccion NFL_TOTAL_v1 y NFL_SPREAD_v1.

Usa la configuracion elegida en la investigacion walk-forward (algoritmo y
familias). Las metricas que se publican salen SOLO del walk-forward, nunca del
entrenamiento. Artefactos en NFL/models/ (arbol propio de SPC).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib                                      # noqa: E402
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from NFL.markets.data import load, numeric         # noqa: E402
from NFL.markets.db import NflModelVersion, init_db, session_scope  # noqa: E402
from NFL.markets.gating import OUT, decide         # noqa: E402
from NFL.markets.research import TARGETS, algos    # noqa: E402
from NFL.markets.research_lines import oof_residuals  # noqa: E402
from shared.paths import NFL_MARKETS_MODELS_DIR, assert_writable  # noqa: E402
from shared.timeutil import json_safe, utcnow      # noqa: E402

VERSION = "v1"
NAME = {"total": "NFL_TOTAL", "spread": "NFL_SPREAD"}


def train_market(market: str, X: pd.DataFrame, gate: dict) -> dict:
    research = json.loads((OUT / "research.json").read_text())["targets"][market]
    cols, algo = research["features"], research["algoritmo_elegido"]
    target = TARGETS[market]
    tr = X[X[target].notna()]
    make = algos()[algo]
    m = make(); m.fit(numeric(tr, cols), tr[target].values)
    res = oof_residuals(make, tr, cols, target)
    # calibrador de P(over|linea): logistica sobre logit(p_raw) con las predicciones
    # walk-forward FUERA de muestra (2020-2025). Como la evidencia dice que el
    # modelo no bate a la linea, este paso encoge las probabilidades hacia 50 %.
    lines = pd.read_parquet(OUT / f"lines_{market}.parquet")
    lines = lines[~lines.push]
    pcol = "p_emp" if gate[market]["distribution"] == "empirica" else "p_norm"
    z = np.log(np.clip(lines[pcol], 1e-4, 1 - 1e-4) / (1 - np.clip(lines[pcol], 1e-4, 1 - 1e-4)))
    # Sin intercepto: solo ENCOGE la confianza, nunca cambia el lado del modelo.
    # (El intercepto solo recogeria que los unders/underdogs cubren un poco mas a
    # menudo, y haria que un "over" proyectado se mostrara como "under".)
    cal = LogisticRegression(C=1e6, fit_intercept=False).fit(
        z.values.reshape(-1, 1), lines.over.astype(int).values)
    art = {"market": market, "algorithm": algo, "features": cols, "model": m,
           "residuals": res, "res_std": float(res.std()), "res_median": float(np.median(res)),
           "distribution": gate[market]["distribution"], "calibrator": cal,
           "calibrator_slope": float(cal.coef_[0][0]), "calibrator_intercept": 0.0,
           "train_seasons": sorted(int(s) for s in tr.season.unique()), "n_train": int(len(tr)),
           "version": VERSION, "name": f"{NAME[market]}_{VERSION}", "trained_at": utcnow()}
    NFL_MARKETS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = assert_writable(NFL_MARKETS_MODELS_DIR / f"{market}_{VERSION}.joblib", "NFL")
    joblib.dump(art, path)
    with session_scope() as s:
        s.query(NflModelVersion).filter(NflModelVersion.market == market,
                                        NflModelVersion.is_production.is_(True)) \
            .update({"is_production": False})
        s.add(NflModelVersion(
            name=art["name"], market=market, version=VERSION, algorithm=algo,
            train_seasons=f"{art['train_seasons'][0]}-{art['train_seasons'][-1]}",
            features=cols, metrics=json_safe({"gating": gate[market],
                                              "calibrator": {"slope": art["calibrator_slope"],
                                                             "intercept": art["calibrator_intercept"]}}),
            artifact_path=str(path), is_production=True,
            notes=("metricas = walk-forward 2020-2025 frente a linea de cierre; "
                   + gate[market]["reason"]), created_at=utcnow()))
    return {"name": art["name"], "algorithm": algo, "n_train": art["n_train"],
            "n_features": len(cols), "res_std": art["res_std"],
            "calibrator_slope": art["calibrator_slope"], "mode": gate[market]["mode"]}


def main():
    init_db()
    X = load()
    gate = decide()
    out = {m: train_market(m, X, gate) for m in ("total", "spread")}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
