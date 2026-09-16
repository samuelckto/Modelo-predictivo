"""Walk-forward de tenis: ganador, total de juegos y handicap de juegos.

Dos motores comparados de forma independiente por circuito (ATP y WTA):

  A) PUNTO A PUNTO. Se estima la probabilidad de ganar el punto al saque de cada
     jugador con la formula clasica ajustada por rival y superficie
         spw_i = media_torneo + (spw_i - media) - (rpw_j - media)
     (variante 1: medias directas; variante 2: ridge sobre features as-of).
     De ahi, con `pointmodel`, salen P(ganar), la distribucion de juegos totales
     y la de margen de juegos: los tres mercados de la MISMA simulacion.

  B) ELO. Logistica sobre Elo global, Elo por superficie y ranking. Para totales
     y handicap se invierte la probabilidad a (p1, p2) de saque con
     `solve_serve_probs` y se usa la misma distribucion.

Baselines: mejor ranking, mejor Elo, mejor Elo de superficie, 50 %, media
historica de juegos.

Seleccion 2015-2019, holdout 2020-2026. Nunca se entrena con la temporada de
prueba. Salidas en TENIS/markets/out/.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from TENIS.markets.features import FEATURES_PARQUET, families, numeric  # noqa: E402
from TENIS.markets.pointmodel import match_distribution, solve_serve_probs  # noqa: E402
from shared.calibration import summary  # noqa: E402
from shared.paths import TENIS_OUT_DIR  # noqa: E402

warnings.filterwarnings("ignore")
SELECT = [2015, 2016, 2017, 2018, 2019]
HOLDOUT = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
TOURS = ("ATP", "WTA")
OUT = TENIS_OUT_DIR
_DIST_CACHE: dict[tuple[int, int, int], dict] = {}


def dist_for(p1: float, p2: float, best_of: int) -> dict:
    """Distribucion cacheada (redondeo a 0.002 en la prob. de punto al saque)."""
    k = (int(round(p1 * 500)), int(round(p2 * 500)), int(best_of))
    d = _DIST_CACHE.get(k)
    if d is None:
        d = match_distribution(k[0] / 500, k[1] / 500, best_of)
        _DIST_CACHE[k] = d
    return d


def serve_probs_classic(X: pd.DataFrame, tour_avg: float, window: str = "y1") -> tuple[np.ndarray, np.ndarray]:
    """spw ajustado por rival: propio + (1 - rpw del rival) - media. Usa la ventana
    de superficie si existe, si no la de 12 meses; NaN si falta."""
    def col(tag, w, m):
        c = f"{tag}_{w}_{m}"
        return X[c] if c in X.columns else pd.Series(np.nan, index=X.index)
    out = []
    for tag, opp in (("p1", "p2"), ("p2", "p1")):
        spw = col(tag, "srf", "spw").fillna(col(tag, "y1", "spw"))
        rpw_opp = col(opp, "srf", "rpw").fillna(col(opp, "y1", "rpw"))
        # rival fuerte al resto -> baja mi spw
        est = spw + (tour_avg - (1 - rpw_opp)) * 0.0 + ((1 - rpw_opp) - tour_avg)
        out.append(est.clip(0.35, 0.85).values)
    return out[0], out[1]


def apply_point_model(X: pd.DataFrame, p1: np.ndarray, p2: np.ndarray) -> pd.DataFrame:
    rows = []
    bo = X.best_of.fillna(3).astype(int).values
    for i in range(len(X)):
        a, b = p1[i], p2[i]
        if not np.isfinite(a) or not np.isfinite(b):
            rows.append((np.nan, np.nan, np.nan, None)); continue
        d = dist_for(float(a), float(b), int(bo[i]))
        rows.append((d["p_win"], d["exp_games"], d["exp_margin"], d))
    return pd.DataFrame(rows, columns=["p_win", "exp_games", "exp_margin", "dist"], index=X.index)


def elo_logit(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    cols = ["d_elo", "d_elo_surface", "log_rank_diff", "d_pts", "best_of_5"]
    cols = [c for c in cols if c in tr.columns]
    m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                  ("m", LogisticRegression(C=0.5, max_iter=2000))])
    m.fit(numeric(tr, cols), tr.p1_win.astype(int).values)
    return m.predict_proba(numeric(te, cols))[:, 1]


def spw_ridge(tr: pd.DataFrame, te: pd.DataFrame, fam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ridge que predice el spw OBSERVADO de cada jugador en el partido a partir de
    features as-of (propias y del rival). Se entrena con partidos anteriores."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    base = sorted({c for k in ("serve", "return", "surface_form", "form", "elo", "elo_surface",
                               "ranking", "context") for c in fam[k]})
    # objetivo: spw real del jugador 1 en ese partido (solo partidos con datos)
    y = tr["obs_spw_p1"]
    ok = y.notna()
    if ok.sum() < 500:
        return np.full(len(te), np.nan), np.full(len(te), np.nan)
    m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                  ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])
    m.fit(numeric(tr[ok], base), y[ok].values)
    p1 = m.predict(numeric(te, base))
    # simetria: para el jugador 2 se invierten las diferencias
    te2 = te.copy()
    for c in base:
        if c.startswith("d_"):
            te2[c] = -te2[c]
        elif c == "log_rank_diff":
            te2[c] = -te2[c]
        elif c == "elo_prob_p1":
            te2[c] = 1 - te2[c]
    p2 = m.predict(numeric(te2, base))
    return np.clip(p1, .35, .85), np.clip(p2, .35, .85)


def prep(X: pd.DataFrame) -> pd.DataFrame:
    """Anade el spw observado (solo para ENTRENAR el ridge; nunca es feature)."""
    X = X.copy()
    X["obs_spw_p1"] = np.nan
    return X


def evaluate(d: pd.DataFrame) -> dict:
    y = d.y.values.astype(float)
    p = np.clip(d.p.values, 1e-4, 1 - 1e-4)
    s = summary(y, p)
    from sklearn.metrics import roc_auc_score
    return {"n": int(len(d)), "accuracy": s["accuracy"], "log_loss": s["log_loss"], "brier": s["brier"],
            "ece": s["ece"], "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None}


def main(tours=TOURS, stage="all"):
    X = pd.read_parquet(FEATURES_PARQUET)
    X = X[(X.p1_win.notna()) & (~X.retirement.fillna(False))]
    fam = families(X)
    prev = json.loads((OUT / "research.json").read_text()) if (OUT / "research.json").exists() else {}
    report = {"protocol": __doc__, "select": SELECT, "holdout": HOLDOUT,
              "tours": prev.get("tours", {})}      # se fusiona: cada circuito se puede correr aparte
    for tour in tours:
        T = X[X.tour == tour]
        tour_avg = float(T[T.season.isin(SELECT)].p1_y1_spw.mean())
        rows = []
        for season in SELECT + HOLDOUT:
            tr = T[T.season < season]
            te = T[T.season == season]
            if len(tr) < 2000 or te.empty:
                continue
            p1c, p2c = serve_probs_classic(te, tour_avg)
            pm = apply_point_model(te, p1c, p2c)
            pe = elo_logit(tr, te)
            base_games = float(tr.games_total.mean())
            rows.append(pd.DataFrame({
                "match_id": te.match_id.values, "season": season, "tour": tour,
                "date": te.date.values, "best_of": te.best_of.fillna(3).values,
                "y": te.p1_win.values, "games": te.games_total.values, "margin": te.games_margin.values,
                "p_point": pm.p_win.values, "exp_games": pm.exp_games.values, "exp_margin": pm.exp_margin.values,
                "p_elo_model": pe, "elo_prob": te.elo_prob_p1.values,
                "elo_surface_prob": te.elo_surface_prob_p1.values,
                "rank_fav": (te.p1_rank < te.p2_rank).astype(float).values,
                "base_games": base_games}))
        d = pd.concat(rows, ignore_index=True)
        d.to_parquet(OUT / f"preds_{tour}.parquet", index=False)
        rep = {"tour": tour, "tour_avg_spw": tour_avg, "n": int(len(d))}
        # --- comparacion de motores en SELECCION ---------------------------------
        sel = d[d.season.isin(SELECT)]
        cand = {
            "punto_a_punto": sel.assign(p=sel.p_point).dropna(subset=["p"]),
            "elo_logistica": sel.assign(p=sel.p_elo_model),
            "elo_simple": sel.assign(p=sel.elo_prob),
            "elo_superficie": sel.assign(p=sel.elo_surface_prob),
        }
        rep["motores_seleccion"] = {k: evaluate(v) for k, v in cand.items()}
        rep["baselines_seleccion"] = {
            "favorito_ranking": float((sel.rank_fav == sel.y).mean()),
            "favorito_elo": float(((sel.elo_prob >= .5) == (sel.y == 1)).mean()),
            "favorito_elo_superficie": float(((sel.elo_surface_prob >= .5) == (sel.y == 1)).mean()),
            "moneda": 0.5,
        }
        best = min(rep["motores_seleccion"], key=lambda k: rep["motores_seleccion"][k]["log_loss"])
        rep["motor_elegido"] = best
        # --- por temporada -------------------------------------------------------
        col = {"punto_a_punto": "p_point", "elo_logistica": "p_elo_model",
               "elo_simple": "elo_prob", "elo_superficie": "elo_surface_prob"}[best]
        per = {}
        for s in SELECT + HOLDOUT:
            ds = d[d.season == s]
            if ds.empty:
                continue
            e = ds.assign(p=ds[col]).dropna(subset=["p"])
            g = ds.dropna(subset=["games", "exp_games"])
            per[s] = {"modelo": evaluate(e),
                      "favorito_elo": float(((ds.elo_prob >= .5) == (ds.y == 1)).mean()),
                      "favorito_ranking": float((ds.rank_fav == ds.y).mean()),
                      "juegos": ({"n": int(len(g)),
                                  "mae": float(np.mean(np.abs(g.games - g.exp_games))),
                                  "rmse": float(np.sqrt(np.mean((g.games - g.exp_games) ** 2))),
                                  "mae_media": float(np.mean(np.abs(g.games - g.base_games)))}
                                 if len(g) else None)}
        rep["por_temporada"] = per
        hold = d[d.season.isin(HOLDOUT)]
        eh = hold.assign(p=hold[col]).dropna(subset=["p"])
        gh = hold.dropna(subset=["games", "exp_games"])
        rep["holdout"] = {
            "modelo": evaluate(eh),
            "favorito_elo": float(((hold.elo_prob >= .5) == (hold.y == 1)).mean()),
            "favorito_ranking": float((hold.rank_fav == hold.y).mean()),
            "elo_simple": evaluate(hold.assign(p=hold.elo_prob)),
            "juegos": {"n": int(len(gh)), "mae": float(np.mean(np.abs(gh.games - gh.exp_games))),
                       "rmse": float(np.sqrt(np.mean((gh.games - gh.exp_games) ** 2))),
                       "mae_media": float(np.mean(np.abs(gh.games - gh.base_games))),
                       "rmse_media": float(np.sqrt(np.mean((gh.games - gh.base_games) ** 2)))},
            "margen": {"mae": float(np.mean(np.abs(gh.margin - gh.exp_margin))),
                       "mae_cero": float(np.mean(np.abs(gh.margin)))},
        }
        report["tours"][tour] = rep
        print(tour, best, {k: round(v["log_loss"], 4) for k, v in rep["motores_seleccion"].items()},
              "| holdout acc", round(rep["holdout"]["modelo"]["accuracy"], 4),
              "elo", round(rep["holdout"]["favorito_elo"], 4),
              "| juegos MAE", round(rep["holdout"]["juegos"]["mae"], 2),
              "vs media", round(rep["holdout"]["juegos"]["mae_media"], 2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "research.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tour", nargs="*", default=None)
    a = ap.parse_args()
    main(tuple(a.tour) if a.tour else TOURS)
