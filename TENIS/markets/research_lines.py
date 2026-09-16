"""Fase 2 de tenis: calibracion, eleccion final de motor, mercados de juegos y umbrales.

  * Se calibra CADA motor (ninguna / Platt / isotonica) ajustando con las
    temporadas anteriores al bloque de seleccion y midiendo en 2018-2019.
    El motor final se elige por log loss YA CALIBRADO (comparacion justa: el
    punto a punto sin calibrar esta sesgado por construccion).
  * Con la probabilidad de partido calibrada se derivan (p1, p2) de saque
    (`solve_serve_probs`) y de ahi la distribucion de juegos totales y de margen:
    los tres mercados salen coherentes del mismo objeto.
  * Totales y handicap se evaluan contra lineas sinteticas (proyeccion +- k):
    dice si P(over k) es fiable, NO si hay ventaja sobre las casas.
  * Umbral de NO PICK del ganador: por buckets, elegido en la temporada T-1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from TENIS.markets.pointmodel import solve_serve_probs  # noqa: E402
from TENIS.markets.research import HOLDOUT, SELECT, TOURS, dist_for  # noqa: E402
from shared.calibration import summary  # noqa: E402
from shared.paths import TENIS_OUT_DIR  # noqa: E402

OUT = TENIS_OUT_DIR
CAL_FIT = [2015, 2016, 2017]
CAL_VAL = [2018, 2019]
BUCKETS = ((0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.01))
OFFSETS = [-6, -3, -1.5, 0, 1.5, 3, 6]
ENGINES = {"punto_a_punto": "p_point", "elo_logistica": "p_elo_model",
           "elo_simple": "elo_prob", "elo_superficie": "elo_surface_prob"}
_SOLVE: dict[tuple[int, int, int], tuple[float, float]] = {}


def serve_from_prob(p_win: float, avg: float, best_of: int) -> tuple[float, float]:
    k = (int(round(p_win * 200)), int(round(avg * 200)), int(best_of))
    v = _SOLVE.get(k)
    if v is None:
        v = solve_serve_probs(k[0] / 200, k[1] / 200, best_of)
        _SOLVE[k] = v
    return v


def buckets(p_pick, hit):
    out = {}
    for lo, hi in BUCKETS:
        m = (p_pick >= lo) & (p_pick < hi)
        out[f"{int(lo*100)}-{int(hi*100) if hi < 1 else '100'}"] = {
            "n": int(m.sum()), "prob_media": float(p_pick[m].mean()) if m.sum() else None,
            "acierto_real": float(hit[m].mean()) if m.sum() else None}
    return out


def calibrate(fit_p, fit_y, method):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    z = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    if method == "ninguna":
        return lambda p: np.clip(p, 1e-4, 1 - 1e-4), None
    if method == "platt":
        m = LogisticRegression(C=1e6).fit(z(fit_p).reshape(-1, 1), fit_y.astype(int))
        return (lambda p: m.predict_proba(z(p).reshape(-1, 1))[:, 1],
                {"slope": float(m.coef_[0][0]), "intercept": float(m.intercept_[0])})
    m = IsotonicRegression(out_of_bounds="clip").fit(fit_p, fit_y)
    return (lambda p: np.clip(m.predict(p), 1e-4, 1 - 1e-4),
            {"x": m.X_thresholds_.tolist(), "y": m.y_thresholds_.tolist()})


def main(tours=TOURS):
    research = json.loads((OUT / "research.json").read_text())
    prev = json.loads((OUT / "research_lines.json").read_text()) if (OUT / "research_lines.json").exists() else {}
    report = {"protocol": __doc__, "tours": prev.get("tours", {})}
    for tour in tours:
        d = pd.read_parquet(OUT / f"preds_{tour}.parquet")
        avg = research["tours"][tour]["tour_avg_spw"]
        fit = d[d.season.isin(CAL_FIT)]
        val = d[d.season.isin(CAL_VAL)]
        rep = {"tour": tour, "avg_spw": avg, "calibracion": {}}
        best, best_ll = None, 9e9
        for eng, col in ENGINES.items():
            f = fit.dropna(subset=[col]); v = val.dropna(subset=[col])
            if len(f) < 1000 or len(v) < 500:
                continue
            rep["calibracion"][eng] = {}
            for meth in ("ninguna", "platt", "isotonica"):
                fn, _ = calibrate(f[col].values, f.y.values, meth)
                ll = summary(v.y.values, fn(v[col].values))["log_loss"]
                rep["calibracion"][eng][meth] = ll
                if ll < best_ll:
                    best, best_ll = (eng, meth), ll
        eng, meth = best
        rep["motor_elegido"], rep["calibracion_elegida"] = eng, meth
        col = ENGINES[eng]
        # calibrador final: ajustado con SELECCION completa, aplicado al HOLDOUT
        f = d[d.season.isin(SELECT)].dropna(subset=[col])
        fn, params = calibrate(f[col].values, f.y.values, meth)
        rep["calibrador"] = params
        d = d.dropna(subset=[col]).copy()
        d["p_cal"] = np.where(d.season.isin(HOLDOUT), fn(d[col].values), d[col].values)
        d["p_pick"] = np.maximum(d.p_cal, 1 - d.p_cal)
        d["hit"] = ((d.p_cal >= .5) == (d.y == 1)).astype(int)
        d["elo_hit"] = ((d.elo_prob >= .5) == (d.y == 1)).astype(int)
        hold = d[d.season.isin(HOLDOUT)]
        rep["holdout_calibrado"] = summary(hold.y.values, hold.p_cal.values)
        rep["buckets_holdout"] = buckets(hold.p_pick.values, hold.hit.values)
        # umbral por buckets elegido en T-1
        per, thr = {}, {}
        for T in SELECT + HOLDOUT:
            v = d[d.season == T - 1]
            b = None
            if len(v) > 500:
                zs = {}
                for lo, hi in BUCKETS:
                    s = v[(v.p_pick >= lo) & (v.p_pick < hi)]
                    if len(s) >= 100:
                        acc = s.hit.mean(); zz = (acc - 0.5) / np.sqrt(0.25 / len(s))
                        zs[lo] = {"n": int(len(s)), "acc": round(float(acc), 4), "z": round(float(zz), 2)}
                los = [lo for lo, _ in BUCKETS]
                for i, lo in enumerate(los):
                    hi_ = [x for x in los[i:] if x in zs]
                    if hi_ and all(zs[x]["z"] >= 1.96 for x in hi_):
                        b = lo; break
                thr[T] = {"umbral": b, "candidatos": {str(k): val for k, val in zs.items()}}
            else:
                thr[T] = {"umbral": None, "candidatos": {}}
            te = d[d.season == T]
            pk = te[te.p_pick >= b] if b is not None else te.iloc[0:0]
            per[T] = {"n": int(len(te)), "acc": float(te.hit.mean()), "acc_elo": float(te.elo_hit.mean()),
                      "umbral": b, "n_picks": int(len(pk)),
                      "acc_picks": float(pk.hit.mean()) if len(pk) else None,
                      "acc_elo_en_picks": float(pk.elo_hit.mean()) if len(pk) else None,
                      "z_vs_50": (float((pk.hit.mean() - .5) / np.sqrt(.25 / len(pk))) if len(pk) else None)}
        rep["por_temporada"], rep["umbrales"] = per, thr
        pk = pd.concat([d[(d.season == T) & (d.p_pick >= thr[T]["umbral"])] for T in HOLDOUT
                        if thr[T]["umbral"] is not None]) if any(thr[T]["umbral"] is not None for T in HOLDOUT) else d.iloc[0:0]
        rep["picks_holdout"] = {
            "n": int(len(pk)), "acc": float(pk.hit.mean()) if len(pk) else None,
            "acc_elo_mismos": float(pk.elo_hit.mean()) if len(pk) else None,
            "z_vs_50": (float((pk.hit.mean() - .5) / np.sqrt(.25 / len(pk))) if len(pk) else None),
            "z_vs_elo": (float((pk.hit.mean() - pk.elo_hit.mean()) / np.sqrt(pk.hit.var() / len(pk) + 1e-9))
                         if len(pk) else None)}
        # --- juegos: distribucion coherente desde la probabilidad calibrada ---------
        g = d.dropna(subset=["games", "margin"]).copy()
        exp_g, exp_m, rows = [], [], []
        for r in g.itertuples():
            p1, p2 = serve_from_prob(float(r.p_cal), avg, int(r.best_of))
            dd = dist_for(p1, p2, int(r.best_of))
            exp_g.append(dd["exp_games"]); exp_m.append(dd["exp_margin"])
            for off in OFFSETS:
                line_t = dd["exp_games"] + off
                line_h = dd["exp_margin"] + off
                rows.append((r.season, off, r.games, line_t, r.margin, line_h,
                             sum(v for k, v in dd["games_total"].items() if k > line_t),
                             sum(v for k, v in dd["games_margin"].items() if k + (-line_h) > 0)))
        g["exp_games_cal"] = exp_g; g["exp_margin_cal"] = exp_m
        L = pd.DataFrame(rows, columns=["season", "off", "games", "line_t", "margin", "line_h", "p_over", "p_cover"])
        L["over"] = (L.games > L.line_t).astype(float)
        L["cover"] = (L.margin > L.line_h).astype(float)
        L.to_parquet(OUT / f"lines_{tour}.parquet", index=False)
        hl = L[L.season.isin(HOLDOUT)]
        hg = g[g.season.isin(HOLDOUT)]
        base = float(d[d.season.isin(SELECT)].base_games.mean())
        rep["juegos"] = {
            "holdout": {"n": int(len(hg)),
                        "mae": float(np.mean(np.abs(hg.games - hg.exp_games_cal))),
                        "rmse": float(np.sqrt(np.mean((hg.games - hg.exp_games_cal) ** 2))),
                        "mae_media": float(np.mean(np.abs(hg.games - base))),
                        "rmse_media": float(np.sqrt(np.mean((hg.games - base) ** 2)))},
            "p_over_calibracion": summary(hl.over.values, np.clip(hl.p_over.values, 1e-4, 1 - 1e-4)),
            "buckets": buckets(np.maximum(hl.p_over, 1 - hl.p_over).values,
                               ((hl.p_over >= .5) == (hl.over == 1)).astype(int).values),
            "por_temporada": {int(s): {"n": int((hg.season == s).sum()),
                                       "mae": float(np.mean(np.abs(hg[hg.season == s].games - hg[hg.season == s].exp_games_cal))),
                                       "mae_media": float(np.mean(np.abs(hg[hg.season == s].games - base)))}
                              for s in sorted(hg.season.unique())},
        }
        rep["handicap"] = {
            "holdout": {"mae": float(np.mean(np.abs(hg.margin - hg.exp_margin_cal))),
                        "mae_cero": float(np.mean(np.abs(hg.margin)))},
            "p_cover_calibracion": summary(hl.cover.values, np.clip(hl.p_cover.values, 1e-4, 1 - 1e-4)),
            "buckets": buckets(np.maximum(hl.p_cover, 1 - hl.p_cover).values,
                               ((hl.p_cover >= .5) == (hl.cover == 1)).astype(int).values),
        }
        report["tours"][tour] = rep
        print(tour, eng, meth, "| acc holdout", round(rep["holdout_calibrado"]["accuracy"], 4),
              "ece", round(rep["holdout_calibrado"]["ece"], 4),
              "| picks", rep["picks_holdout"]["n"], rep["picks_holdout"]["acc"] and round(rep["picks_holdout"]["acc"], 4),
              "| juegos MAE", round(rep["juegos"]["holdout"]["mae"], 2), "vs", round(rep["juegos"]["holdout"]["mae_media"], 2))
    (OUT / "research_lines.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tour", nargs="*", default=None)
    a = ap.parse_args()
    main(tuple(a.tour) if a.tour else TOURS)
