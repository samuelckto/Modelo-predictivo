"""Investigacion walk-forward de totals y spread NFL. Genera NFL/markets/out/research.json.

Protocolo:
  * Walk-forward temporal: para la temporada de prueba T se entrena con 2015..T-1.
  * Seleccion de algoritmo y familias de features SOLO con las temporadas
    2020-2022 (bloque de seleccion). 2023-2025 es confirmacion: no se toca hasta
    que la configuracion esta cerrada.
  * Baselines: media historica de entrenamiento; linea de cierre del mercado
    (referencia fuerte, NO disponible antes del kickoff; nunca es feature).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from NFL.markets.data import families, load, numeric  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
SELECT_SEASONS = [2020, 2021, 2022]
CONFIRM_SEASONS = [2023, 2024, 2025]
TARGETS = {"total": "total_pts", "spread": "margin"}
LINE = {"total": "total_line", "spread": "spread_line"}
BASE_FAMILIES = {
    "total": ["sum_epa", "sum_eff", "sum_pressure", "sum_scoring", "level_epa", "level_scoring",
              "qb", "injuries", "context", "elo", "history"],
    "spread": ["elo", "diff_epa", "diff_eff", "diff_pressure", "diff_results", "qb", "injuries",
               "context", "h2h", "history"],
}


def algos():
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    out = {
        "ridge": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                   ("m", RidgeCV(alphas=np.logspace(-1, 3, 12)))]),
        "rf": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                                                            max_features=0.5, n_jobs=-1, random_state=7))]),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_depth=3,
                                                     min_samples_leaf=30, l2_regularization=1.0,
                                                     random_state=7),
    }
    try:
        import xgboost as xgb
        out["xgb"] = lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                              subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                                              reg_lambda=2.0, random_state=7, n_jobs=4, verbosity=0)
    except Exception:
        pass
    return out


def walk(X, target, cols, make, seasons):
    """Predicciones fuera de muestra por temporada. Devuelve DataFrame con y, pred, line."""
    rows = []
    for T in seasons:
        tr = X[(X.season < T) & X[target].notna()]
        te = X[(X.season == T) & X[target].notna()]
        if te.empty:
            continue
        if make is None:
            pred = np.full(len(te), tr[target].mean())
        else:
            m = make()
            m.fit(numeric(tr, cols), tr[target].values)
            pred = m.predict(numeric(te, cols))
        rows.append(pd.DataFrame({"game_id": te.game_id.values, "season": T, "y": te[target].values,
                                  "pred": pred, "line": te[LINE[{"total_pts": "total",
                                                                  "margin": "spread"}[target]]].values}))
    return pd.concat(rows, ignore_index=True)


def reg_metrics(d):
    e = d.y - d.pred
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "n": int(len(d))}


def line_metrics(d):
    e = d.y - d.line
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}


def main():
    X = load()
    fam = families(X)
    A = algos()
    report = {"protocol": __doc__, "algorithms": list(A), "targets": {}}
    for mk, target in TARGETS.items():
        rep = {"families_tested": BASE_FAMILIES[mk]}
        allcols = sorted({c for f in BASE_FAMILIES[mk] for c in fam[f]})
        # 1) algoritmo con todas las familias, bloque de seleccion
        sel = {}
        for name, make in A.items():
            d = walk(X, target, allcols, make, SELECT_SEASONS)
            sel[name] = reg_metrics(d)
        base = walk(X, target, allcols, None, SELECT_SEASONS)
        sel["media_historica"] = reg_metrics(base)
        sel["linea_cierre_mercado"] = line_metrics(base)
        best_algo = min((k for k in A), key=lambda k: sel[k]["mae"])
        rep["seleccion_algoritmo_2020_2022"] = sel
        rep["algoritmo_elegido"] = best_algo
        # 2) ablacion hacia atras por familias con el algoritmo elegido (bloque de seleccion)
        make = A[best_algo]
        current = list(BASE_FAMILIES[mk])
        cur_mae = sel[best_algo]["mae"]
        ablation = []
        improved = True
        while improved and len(current) > 1:
            improved = False
            trial = {}
            for f in current:
                cols = sorted({c for g in current if g != f for c in fam[g]})
                trial[f] = reg_metrics(walk(X, target, cols, make, SELECT_SEASONS))["mae"]
            f_drop = min(trial, key=trial.get)
            ablation.append({"quitando": f_drop, "mae_sin": round(trial[f_drop], 4),
                             "mae_con": round(cur_mae, 4)})
            if trial[f_drop] < cur_mae - 0.02:      # solo se quita si mejora de verdad
                current.remove(f_drop)
                cur_mae = trial[f_drop]
                improved = True
        rep["familias_finales"] = current
        rep["familias_descartadas"] = [f for f in BASE_FAMILIES[mk] if f not in current]
        rep["ablacion"] = ablation
        final_cols = sorted({c for g in current for c in fam[g]})
        rep["n_features"] = len(final_cols)
        rep["features"] = final_cols
        # 3) confirmacion 2023-2025 + serie completa 2020-2025 por temporada
        allseas = SELECT_SEASONS + CONFIRM_SEASONS
        d = walk(X, target, final_cols, make, allseas)
        b = walk(X, target, final_cols, None, allseas)
        d.to_parquet(OUT / f"preds_{mk}.parquet", index=False)
        per = {}
        for T in allseas:
            dd, bb = d[d.season == T], b[b.season == T]
            per[T] = {"modelo": reg_metrics(dd), "media": reg_metrics(bb), "mercado_cierre": line_metrics(dd)}
        rep["por_temporada"] = per
        rep["confirmacion_2023_2025"] = {
            "modelo": reg_metrics(d[d.season.isin(CONFIRM_SEASONS)]),
            "media": reg_metrics(b[b.season.isin(CONFIRM_SEASONS)]),
            "mercado_cierre": line_metrics(d[d.season.isin(CONFIRM_SEASONS)])}
        rep["total_2020_2025"] = {"modelo": reg_metrics(d), "media": reg_metrics(b),
                                  "mercado_cierre": line_metrics(d)}
        report["targets"][mk] = rep
        print(mk, best_algo, rep["familias_finales"], rep["confirmacion_2023_2025"])
    (OUT / "research.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
