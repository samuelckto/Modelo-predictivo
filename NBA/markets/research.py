"""Walk-forward NBA: moneyline (clasificacion), spread (margen) y total (puntos).

  * Temporadas de prueba 2019..2026 (2019 = 2018-19). Para T se entrena con < T.
  * SELECCION (algoritmo, familias, calibracion): 2019-2022. HOLDOUT: 2023-2026,
    que no se mira hasta cerrar la configuracion.
  * Sin lineas historicas: spread y total se evaluan como distribuciones
    (MAE/RMSE vs media historica y calibracion de P(margen > k), P(total > k)),
    no contra el mercado.
Salida: NBA/markets/out/research.json y preds_*.parquet (OOS, reproducibles).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from NBA.markets.features import FEATURES_PARQUET, families, numeric  # noqa: E402
from shared.calibration import summary  # noqa: E402
from shared.paths import NBA_OUT_DIR  # noqa: E402

warnings.filterwarnings("ignore")
SELECT = [2019, 2020, 2021, 2022]
HOLDOUT = [2023, 2024, 2025, 2026]
TARGET = {"moneyline": "home_win", "spread": "margin", "total": "total_pts"}
BASE_FAMILIES = {
    "moneyline": ["elo", "strength", "pace", "shooting", "turnovers", "rebounding", "free_throws",
                  "defense_misc", "rest", "context"],
    "spread": ["elo", "strength", "pace", "shooting", "turnovers", "rebounding", "free_throws",
               "defense_misc", "rest", "context"],
    "total": ["elo", "strength", "pace", "shooting", "turnovers", "rebounding", "free_throws",
              "defense_misc", "rest", "context"],
}
THRESH = {"spread": [-10, -5, 0, 5, 10], "total": [-15, -7.5, 0, 7.5, 15]}   # k relativo a la media


def clf_algos():
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    A = {
        "logistic": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                      ("m", LogisticRegression(C=0.05, max_iter=2000))]),
        "rf": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=25,
                                                             max_features=0.4, n_jobs=-1, random_state=7))]),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_depth=3,
                                                      min_samples_leaf=40, l2_regularization=1.0, random_state=7),
    }
    try:
        import xgboost as xgb
        A["xgb"] = lambda: xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                                             colsample_bytree=0.6, min_child_weight=10, reg_lambda=2.0,
                                             random_state=7, n_jobs=4, verbosity=0)
    except Exception:
        pass
    return A


def reg_algos():
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    A = {
        "ridge": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                   ("m", RidgeCV(alphas=np.logspace(-1, 3, 12)))]),
        "rf": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=25,
                                                            max_features=0.4, n_jobs=-1, random_state=7))]),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_depth=3,
                                                     min_samples_leaf=40, l2_regularization=1.0, random_state=7),
    }
    try:
        import xgboost as xgb
        A["xgb"] = lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                                            colsample_bytree=0.6, min_child_weight=10, reg_lambda=2.0,
                                            random_state=7, n_jobs=4, verbosity=0)
    except Exception:
        pass
    return A


def walk(X, market, cols, make, seasons):
    tgt = TARGET[market]
    rows = []
    for T in seasons:
        tr = X[(X.season < T) & X[tgt].notna()]
        te = X[(X.season == T) & X[tgt].notna()]
        if te.empty:
            continue
        if make is None:
            pred = np.full(len(te), tr[tgt].mean())
        else:
            m = make(); m.fit(numeric(tr, cols), tr[tgt].values)
            pred = m.predict_proba(numeric(te, cols))[:, 1] if market == "moneyline" else m.predict(numeric(te, cols))
        rows.append(pd.DataFrame({"game_id": te.game_id.values, "season": T, "y": te[tgt].values, "pred": pred,
                                  "elo_p": te.elo_prob_home.values, "date": te.game_date.values}))
    return pd.concat(rows, ignore_index=True)


def score_clf(d):
    s = summary(d.y.values.astype(float), np.clip(d.pred.values, 1e-4, 1 - 1e-4))
    from sklearn.metrics import roc_auc_score
    return {"n": int(len(d)), "accuracy": s["accuracy"], "log_loss": s["log_loss"], "brier": s["brier"],
            "ece": s["ece"], "auc": float(roc_auc_score(d.y, d.pred)) if d.y.nunique() > 1 else None}


def score_reg(d):
    e = d.y - d.pred
    return {"n": int(len(d)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}


PARTIAL = NBA_OUT_DIR / "research_partial.json"


def _load_partial():
    return json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {"targets": {}}


def _save_partial(d):
    PARTIAL.write_text(json.dumps(d, indent=1, default=str))


def main(markets=("moneyline", "spread", "total"), stage="all"):
    """Reanudable: cada etapa guarda en research_partial.json (los procesos largos mueren)."""
    X = pd.read_parquet(FEATURES_PARQUET)
    X = X[X.season_type == "Regular Season"]          # playoffs fuera: otro regimen (se documenta)
    fam = families(X)
    report = _load_partial()
    report.update({"protocol": __doc__, "select": SELECT, "holdout": HOLDOUT})
    for mk in markets:
        rep = report["targets"].setdefault(mk, {})
        A = clf_algos() if mk == "moneyline" else reg_algos()
        score = score_clf if mk == "moneyline" else score_reg
        key = "log_loss" if mk == "moneyline" else "mae"
        allcols = sorted({c for f in BASE_FAMILIES[mk] for c in fam[f]})
        if stage in ("all", "algos") and "algoritmo_elegido" not in rep:
            _stage_algos(X, mk, A, score, key, allcols, rep); _save_partial(report)
        if stage in ("all", "ablation") and "algoritmo_elegido" in rep and "familias_finales" not in rep:
            _stage_ablation(X, mk, A, score, key, fam, rep, report)
        if stage in ("all", "final") and "familias_finales" in rep and "holdout_2023_2026" not in rep:
            _stage_final(X, mk, A, score, fam, rep); _save_partial(report)
    if all("holdout_2023_2026" in report["targets"].get(m, {}) for m in ("moneyline", "spread", "total")):
        (NBA_OUT_DIR / "research.json").write_text(json.dumps(report, indent=1, default=str))
        print("research.json completo")


def _stage_algos(X, mk, A, score, key, allcols, rep):
        sel = rep.setdefault("algoritmos", {})
        for name, make in A.items():
            if name in sel:
                continue
            sel[name] = score(walk(X, mk, allcols, make, SELECT))
            _save_partial(_merge(mk, rep))
        base = walk(X, mk, allcols, None, SELECT)
        if mk == "moneyline":
            sel["tasa_local_historica"] = score(base)
            e = base.copy(); e["pred"] = e.elo_p
            sel["elo"] = score(e)
            sel["siempre_favorito_elo"] = {"accuracy": float(((base.elo_p >= 0.5) == (base.y == 1)).mean())}
        else:
            sel["media_historica"] = score(base)
        rep["algoritmo_elegido"] = min(A, key=lambda k: sel[k][key])


def _merge(mk, rep):
    d = _load_partial(); d["targets"][mk] = rep; return d


def _stage_ablation(X, mk, A, score, key, fam, rep, report):
        make = A[rep["algoritmo_elegido"]]
        st = rep.setdefault("_ablacion_estado", {"current": list(BASE_FAMILIES[mk]),
                                                  "cur": rep["algoritmos"][rep["algoritmo_elegido"]][key],
                                                  "trial": {}, "abl": []})
        tol = 0.0015 if mk == "moneyline" else 0.02
        while True:
            current = st["current"]
            for f in current:
                if f in st["trial"]:
                    continue
                cols = sorted({c for g in current if g != f for c in fam[g]})
                st["trial"][f] = score(walk(X, mk, cols, make, SELECT))[key]
                _save_partial(_merge(mk, rep))
            fd = min(st["trial"], key=st["trial"].get)
            st["abl"].append({"quitando": fd, "sin": round(st["trial"][fd], 5), "con": round(st["cur"], 5)})
            if st["trial"][fd] < st["cur"] - tol and len(current) > 1:
                current.remove(fd); st["cur"] = st["trial"][fd]; st["trial"] = {}
                _save_partial(_merge(mk, rep))
                continue
            break
        rep["familias_finales"] = st["current"]
        rep["familias_descartadas"] = [f for f in BASE_FAMILIES[mk] if f not in st["current"]]
        rep["ablacion"] = st["abl"]
        rep.pop("_ablacion_estado", None)
        _save_partial(_merge(mk, rep))


def _stage_final(X, mk, A, score, fam, rep):
        make = A[rep["algoritmo_elegido"]]
        current = rep["familias_finales"]
        cols = sorted({c for g in current for c in fam[g]})
        rep["features"] = cols; rep["n_features"] = len(cols)
        # serie completa, cacheada por temporada (reanudable)
        parts = []
        for T in SELECT + HOLDOUT:
            f = NBA_OUT_DIR / f"_cache_{mk}_{T}.parquet"
            if f.exists():
                parts.append(pd.read_parquet(f)); continue
            dT = walk(X, mk, cols, make, [T]); dT.to_parquet(f, index=False); parts.append(dT)
        d = pd.concat(parts, ignore_index=True)
        b = walk(X, mk, cols, None, SELECT + HOLDOUT)
        d.to_parquet(NBA_OUT_DIR / f"preds_{mk}.parquet", index=False)
        per = {}
        for T in SELECT + HOLDOUT:
            dd, bb = d[d.season == T], b[b.season == T]
            row = {"modelo": score(dd)}
            if mk == "moneyline":
                row["tasa_local"] = float(dd.y.mean())
                e = dd.copy(); e["pred"] = e.elo_p; row["elo"] = score(e)
                row["siempre_favorito_elo"] = float(((dd.elo_p >= 0.5) == (dd.y == 1)).mean())
            else:
                row["media"] = score(bb)
            per[T] = row
        rep["por_temporada"] = per
        rep["holdout_2023_2026"] = {"modelo": score(d[d.season.isin(HOLDOUT)]),
                                   **({"elo": score(d[d.season.isin(HOLDOUT)].assign(pred=lambda z: z.elo_p)),
                                       "tasa_local": float(d[d.season.isin(HOLDOUT)].y.mean())}
                                      if mk == "moneyline" else {"media": score(b[b.season.isin(HOLDOUT)])})}
        print(mk, rep["algoritmo_elegido"], current, rep["holdout_2023_2026"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--market", nargs="*", default=None)
    ap.add_argument("--stage", default="all"); a = ap.parse_args()
    main(tuple(a.market) if a.market else ("moneyline", "spread", "total"), a.stage)
