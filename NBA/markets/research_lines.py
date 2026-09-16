"""Fase 2 NBA: calibracion, distribuciones y umbrales, con las predicciones OOS.

  * Moneyline: calibracion (ninguna / Platt / isotonica) elegida en 2019-2021 ->
    2022, buckets 50-55..70+, umbral de NO PICK elegido en T-1 (z>=1.64 sobre 50 %).
  * Spread y total: distribucion de residuos (normal vs empirica, elegida en
    2019-2022 por log loss de P(y > k)), calibracion de P(y > k) por buckets.
    Sin lineas historicas NO se mide acierto contra el mercado.
Salida: research_lines.json.
"""
from __future__ import annotations

import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from NBA.markets.features import FEATURES_PARQUET, numeric  # noqa: E402
from NBA.markets.research import HOLDOUT, SELECT, TARGET, reg_algos  # noqa: E402
from shared.calibration import summary  # noqa: E402
from shared.paths import NBA_OUT_DIR  # noqa: E402

BUCKETS = ((0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.01))
OFFSETS = {"spread": [-10, -5, -2.5, 0, 2.5, 5, 10], "total": [-15, -10, -5, 0, 5, 10, 15]}


def buckets(p_pick, hit):
    out = {}
    for lo, hi in BUCKETS:
        m = (p_pick >= lo) & (p_pick < hi)
        out[f"{int(lo*100)}-{int(hi*100) if hi < 1 else '100'}"] = {
            "n": int(m.sum()), "prob_media": float(p_pick[m].mean()) if m.sum() else None,
            "acierto_real": float(hit[m].mean()) if m.sum() else None}
    return out


def oof_residuals(make, tr, cols, target, k=5):
    from sklearn.model_selection import KFold
    oof = np.zeros(len(tr)); Xn, y = numeric(tr, cols), tr[target].values
    for a, b in KFold(k, shuffle=False).split(tr):
        m = make(); m.fit(Xn.iloc[a], y[a]); oof[b] = m.predict(Xn.iloc[b])
    return y - oof


def p_over(pred, line, res, normal):
    if normal:
        s = res.std()
        return np.array([0.5 * (1 - erf((l - p) / (s * sqrt(2)))) for p, l in zip(pred, line)])
    return np.array([float(np.mean((p + res) > l)) for p, l in zip(pred, line)])


def moneyline(report):
    d = pd.read_parquet(NBA_OUT_DIR / "preds_moneyline.parquet")
    d["pred"] = d.pred.clip(1e-4, 1 - 1e-4)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    z = lambda p: np.log(p / (1 - p))
    fit = d[d.season.isin([2019, 2020, 2021])]; val = d[d.season == 2022]
    platt = LogisticRegression(C=1e6).fit(z(fit.pred).values.reshape(-1, 1), fit.y.astype(int))
    iso = IsotonicRegression(out_of_bounds="clip").fit(fit.pred.values, fit.y.values)
    cands = {"ninguna": val.pred.values,
             "platt": platt.predict_proba(z(val.pred).values.reshape(-1, 1))[:, 1],
             "isotonica": np.clip(iso.predict(val.pred.values), 1e-4, 1 - 1e-4)}
    sel = {k: summary(val.y.values, v)["log_loss"] for k, v in cands.items()}
    method = min(sel, key=sel.get)
    rep = {"calibracion_candidatos_2022": sel, "calibracion_elegida": method}
    # calibrador final = ajustado con SELECT (2019-2022) y aplicado a HOLDOUT (nunca al reves)
    fit = d[d.season.isin(SELECT)]
    if method == "platt":
        cal = LogisticRegression(C=1e6).fit(z(fit.pred).values.reshape(-1, 1), fit.y.astype(int))
        apply = lambda p: cal.predict_proba(z(np.clip(p, 1e-4, 1 - 1e-4)).reshape(-1, 1))[:, 1]
        rep["calibrador"] = {"slope": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0])}
    elif method == "isotonica":
        cal = IsotonicRegression(out_of_bounds="clip").fit(fit.pred.values, fit.y.values)
        apply = lambda p: np.clip(cal.predict(p), 1e-4, 1 - 1e-4)
        rep["calibrador"] = {"isotonic_x": cal.X_thresholds_.tolist(), "isotonic_y": cal.y_thresholds_.tolist()}
    else:
        apply = lambda p: p; rep["calibrador"] = None
    d["p_cal"] = np.where(d.season.isin(HOLDOUT), apply(d.pred.values), d.pred.values)  # SELECT queda sin calibrar (honesto)
    d["p_pick"] = np.maximum(d.p_cal, 1 - d.p_cal)
    d["hit"] = ((d.p_cal >= 0.5) == (d.y == 1)).astype(int)
    d["fav_elo_hit"] = ((d.elo_p >= 0.5) == (d.y == 1)).astype(int)
    hold = d[d.season.isin(HOLDOUT)]
    rep["holdout_calibrado"] = summary(hold.y.values, hold.p_cal.values)
    rep["buckets_holdout"] = buckets(hold.p_pick.values, hold.hit.values)
    rep["buckets_select_sin_calibrar"] = buckets(d[d.season.isin(SELECT)].p_pick.values, d[d.season.isin(SELECT)].hit.values)
    # umbral de NO PICK elegido en T-1
    per, thr = {}, {}
    for T in SELECT + HOLDOUT:
        v = d[d.season == T - 1]
        best, cands_ = None, {}
        if len(v) > 300:
            # regla por BUCKET (no acumulada): un bucket entra si en T-1 acierta
            # significativamente mas del 50 % (z >= 1.96, n >= 100) y todos los
            # buckets superiores tambien. El umbral es el borde inferior del
            # primer bucket que cumple. Asi el 50-55 % (moneda al aire) queda fuera.
            zs = {}
            for lo, hi in BUCKETS:
                sub = v[(v.p_pick >= lo) & (v.p_pick < hi)]
                if len(sub) >= 100:
                    acc = sub.hit.mean(); zz = (acc - 0.5) / np.sqrt(0.25 / len(sub))
                    zs[lo] = {"n": int(len(sub)), "acc": round(float(acc), 4), "z": round(float(zz), 2)}
            cands_ = {str(k): val for k, val in zs.items()}
            los = [lo for lo, _ in BUCKETS]
            for i, lo in enumerate(los):
                higher = [x for x in los[i:] if x in zs]
                if higher and all(zs[x]["z"] >= 1.96 for x in higher):
                    best = lo; break
        thr[T] = {"umbral": best, "candidatos": cands_}
        te = d[d.season == T]
        picks = te[te.p_pick >= best] if best is not None else te.iloc[0:0]
        per[T] = {"n": int(len(te)), "acc_todos": float(te.hit.mean()), "acc_fav_elo": float(te.fav_elo_hit.mean()),
                  "tasa_local": float(te.y.mean()), "umbral": best, "n_picks": int(len(picks)),
                  "acc_picks": float(picks.hit.mean()) if len(picks) else None,
                  "acc_fav_elo_en_picks": float(picks.fav_elo_hit.mean()) if len(picks) else None,
                  "z_picks_vs_50": (float((picks.hit.mean() - 0.5) / np.sqrt(0.25 / len(picks))) if len(picks) else None)}
    rep["por_temporada"] = per; rep["umbrales"] = thr
    pk = pd.concat([d[(d.season == T) & (d.p_pick >= thr[T]["umbral"])] for T in HOLDOUT if thr[T]["umbral"] is not None]) \
        if any(thr[T]["umbral"] is not None for T in HOLDOUT) else d.iloc[0:0]
    rep["picks_holdout"] = {"n": int(len(pk)), "acc": float(pk.hit.mean()) if len(pk) else None,
                            "acc_fav_elo_mismos_partidos": float(pk.fav_elo_hit.mean()) if len(pk) else None,
                            "z_vs_50": (float((pk.hit.mean() - 0.5) / np.sqrt(0.25 / len(pk))) if len(pk) else None),
                            "z_vs_fav_elo": (float((pk.hit.mean() - pk.fav_elo_hit.mean()) /
                                                   np.sqrt(pk.hit.var() / len(pk) + 1e-9)) if len(pk) else None)}
    d.to_parquet(NBA_OUT_DIR / "lines_moneyline.parquet", index=False)
    report["targets"]["moneyline"] = rep
    print("moneyline", method, rep["holdout_calibrado"]["accuracy"], rep["picks_holdout"])


def regression(report, mk):
    X = pd.read_parquet(FEATURES_PARQUET); X = X[X.season_type == "Regular Season"]
    research = json.loads((NBA_OUT_DIR / "research.json").read_text())["targets"][mk]
    cols, target = research["features"], TARGET[mk]
    make = reg_algos()[research["algoritmo_elegido"]]
    d = pd.read_parquet(NBA_OUT_DIR / f"preds_{mk}.parquet")
    rows = []
    for T in SELECT + HOLDOUT:
        tr = X[(X.season < T) & X[target].notna()]
        res = oof_residuals(make, tr, cols, target)
        te = d[d.season == T]
        for off in OFFSETS[mk]:
            line = te.pred.values + off
            rows.append(pd.DataFrame({"season": T, "off": off, "y": te.y.values, "pred": te.pred.values, "line": line,
                                      "p_norm": p_over(te.pred.values, line, res, True),
                                      "p_emp": p_over(te.pred.values, line, res, False),
                                      "res_std": res.std(), "res_median": float(np.median(res))}))
    L = pd.concat(rows, ignore_index=True)
    L["over"] = (L.y > L.line).astype(float)
    L = L[L.y != L.line]
    s = L[L.season.isin(SELECT)]
    ll = {"normal": summary(s.over.values, s.p_norm.values)["log_loss"],
          "empirica": summary(s.over.values, s.p_emp.values)["log_loss"]}
    dist = min(ll, key=ll.get)
    pc = "p_norm" if dist == "normal" else "p_emp"
    h = L[L.season.isin(HOLDOUT)]
    hp = np.maximum(h[pc], 1 - h[pc]).values; hh = ((h[pc] >= 0.5) == (h.over == 1)).astype(int).values
    rep = {"distribucion_candidatas_select": ll, "distribucion_elegida": dist,
           "residuo_std": float(L.res_std.mean()), "residuo_mediana": float(L.res_median.mean()),
           "holdout_p_over_k": summary(h.over.values, h[pc].values),
           "buckets_holdout": buckets(hp, hh),
           "por_offset_holdout": {str(o): {"n": int((h.off == o).sum()),
                                           "log_loss": summary(h[h.off == o].over.values, h[h.off == o][pc].values)["log_loss"],
                                           "acc": float((((h[h.off == o][pc] >= 0.5) == (h[h.off == o].over == 1))).mean())}
                                  for o in OFFSETS[mk]},
           "nota": ("Sin lineas historicas del mercado: la calibracion se mide sobre lineas sinteticas "
                    "(proyeccion + k). Dice si P(over k) es fiable en terminos absolutos, NO si hay ventaja "
                    "sobre las casas de apuestas.")}
    L.to_parquet(NBA_OUT_DIR / f"lines_{mk}.parquet", index=False)
    report["targets"][mk] = rep
    print(mk, dist, rep["holdout_p_over_k"]["ece"], rep["buckets_holdout"])


def main():
    report = {"protocol": __doc__, "targets": {}}
    moneyline(report)
    for mk in ("spread", "total"):
        regression(report, mk)
    (NBA_OUT_DIR / "research_lines.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
