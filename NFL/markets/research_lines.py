"""Fase 2: del modelo de puntos a probabilidades frente a la linea de cierre.

Para cada temporada de prueba T:
  * modelo ridge entrenado con < T (mismas features que research.py);
  * residuos FUERA de muestra (KFold sobre el entrenamiento) -> distribucion
    empirica; tambien una normal con la misma desviacion, para comparar;
  * P(over linea) = P(pred + residuo > linea); P(home cubre) = P(pred + residuo > linea).
Baselines: siempre over / siempre under / siempre favorito / siempre underdog / 50 %.
Umbral de NO PICK: se elige en la temporada T-1 (validacion) y se aplica en T.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from NFL.markets.data import families, load, numeric  # noqa: E402
from NFL.markets.research import LINE, TARGETS, algos  # noqa: E402
from shared.calibration import summary  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
CANDIDATE_THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]
KEY_SPREAD = [1, 3, 4, 6, 7, 10, 14]
KEY_TOTAL = [37, 41, 43, 44, 47, 51, 54]


def oof_residuals(make, tr, cols, target, k=5):
    from sklearn.model_selection import KFold
    oof = np.zeros(len(tr))
    Xn, y = numeric(tr, cols), tr[target].values
    for a, b in KFold(k, shuffle=False).split(tr):
        m = make(); m.fit(Xn.iloc[a], y[a]); oof[b] = m.predict(Xn.iloc[b])
    return y - oof


def p_over(pred, line, res, normal=False):
    if normal:
        from math import erf, sqrt
        s = res.std()
        return np.array([0.5 * (1 - erf((l - p) / (s * sqrt(2)))) for p, l in zip(pred, line)])
    return np.array([float(np.mean((p + res) > l)) for p, l in zip(pred, line)])


def main():
    X = load(); fam = families(X)
    research = json.loads((OUT / "research.json").read_text())
    A = algos()
    report = {"protocol": __doc__, "targets": {}}
    for mk, target in TARGETS.items():
        r = research["targets"][mk]
        cols, make = r["features"], A[r["algoritmo_elegido"]]
        rows = []
        for T in SEASONS:
            tr = X[(X.season < T) & X[target].notna()]
            te = X[(X.season == T) & X[target].notna() & X[LINE[mk]].notna()]
            m = make(); m.fit(numeric(tr, cols), tr[target].values)
            pred = m.predict(numeric(te, cols))
            res = oof_residuals(make, tr, cols, target)
            line = te[LINE[mk]].values.astype(float)
            pe = p_over(pred, line, res)
            pn = p_over(pred, line, res, normal=True)
            y = te[target].values.astype(float)
            rows.append(pd.DataFrame({
                "game_id": te.game_id.values, "season": T, "pred": pred, "line": line, "y": y,
                "p_emp": pe, "p_norm": pn, "push": y == line, "over": y > line,
                "home_fav": line > 0 if mk == "spread" else np.nan,
                "res_std": res.std(), "res_median": float(np.median(res))}))
        d = pd.concat(rows, ignore_index=True)
        d.to_parquet(OUT / f"lines_{mk}.parquet", index=False)
        dn = d[~d.push].copy()
        rep = {"algoritmo": r["algoritmo_elegido"], "n": int(len(d)), "pushes": int(d.push.sum())}
        # --- distribucion: empirica vs normal ---------------------------------
        y01 = dn.over.astype(float).values
        rep["distribucion"] = {
            "empirica": summary(y01, dn.p_emp.values),
            "normal": summary(y01, dn.p_norm.values),
            "residuo_std_medio": float(d.res_std.mean()),
            "residuo_mediana_media": float(d.res_median.mean()),
        }
        # key numbers: masa real en cada numero vs la que da una normal
        val = (X[X.season >= 2015][target].dropna()).values
        keys = KEY_SPREAD if mk == "spread" else KEY_TOTAL
        from math import erf, sqrt
        s = val.std(); mu = val.mean()
        def normal_mass(k):
            z1, z2 = (k - 0.5 - mu) / s, (k + 0.5 - mu) / s
            return 0.5 * (erf(z2 / sqrt(2)) - erf(z1 / sqrt(2)))
        km = {}
        for k in keys:
            if mk == "spread":
                real = float(np.mean(np.abs(val) == k)); nm = normal_mass(k) + normal_mass(-k)
            else:
                real = float(np.mean(val == k)); nm = normal_mass(k)
            km[k] = {"real": round(real, 4), "normal": round(nm, 4), "ratio": round(real / nm, 2) if nm else None}
        rep["key_numbers"] = km
        # --- seleccion de distribucion (mejor log loss en 2020-2022) --------------
        s3 = dn[dn.season <= 2022]
        use = "p_emp" if summary(s3.over.astype(float).values, s3.p_emp.values)["log_loss"] <= \
            summary(s3.over.astype(float).values, s3.p_norm.values)["log_loss"] else "p_norm"
        rep["distribucion_elegida"] = "empirica" if use == "p_emp" else "normal"
        dn["p"] = dn[use]
        dn["p_pick"] = np.maximum(dn.p, 1 - dn.p)
        dn["pick_over"] = dn.p >= 0.5
        dn["hit"] = (dn.pick_over == dn.over)
        # --- por temporada, con umbral elegido en T-1 -----------------------------
        per, thr_hist = {}, {}
        for T in SEASONS:
            val_ = dn[dn.season == T - 1]
            best_thr, best_val = None, None
            if len(val_) > 100:
                cands = {}
                for t in CANDIDATE_THRESHOLDS:
                    sub = val_[val_.p_pick >= t]
                    if len(sub) >= 60:
                        acc = sub.hit.mean()
                        z = (acc - 0.5) / np.sqrt(0.25 / len(sub))
                        cands[t] = {"n": int(len(sub)), "acc": round(float(acc), 4), "z": round(float(z), 2)}
                # umbral: el menor que alcanza z >= 1.64 (p<0.05 unilateral); si ninguno, sin pick
                ok = [t for t, c in cands.items() if c["z"] >= 1.64]
                best_thr = min(ok) if ok else None
                best_val = cands
            thr_hist[T] = {"umbral_elegido_en": T - 1, "umbral": best_thr, "candidatos": best_val}
            te = dn[dn.season == T]
            y_ = te.over.astype(float).values
            base = {
                "siempre_over" if mk == "total" else "siempre_local": float(te.over.mean()),
                "siempre_under" if mk == "total" else "siempre_visitante": float(1 - te.over.mean()),
            }
            if mk == "spread":
                fav_cover = np.where(te.home_fav, te.over, ~te.over)
                base["siempre_favorito"] = float(np.mean(fav_cover))
                base["siempre_underdog"] = float(1 - np.mean(fav_cover))
            picks = te[te.p_pick >= best_thr] if best_thr is not None else te.iloc[0:0]
            per[T] = {
                "n": int(len(te)), "acc_todos": float(te.hit.mean()),
                "metricas_prob": summary(y_, te.p.values),
                "baselines": base,
                "umbral": best_thr, "n_picks": int(len(picks)),
                "acc_picks": float(picks.hit.mean()) if len(picks) else None,
                "z_picks": (float((picks.hit.mean() - 0.5) / np.sqrt(0.25 / len(picks)))
                            if len(picks) else None),
            }
        rep["por_temporada"] = per
        rep["umbrales"] = thr_hist
        # --- agregado 2020-2025 y calibracion por buckets --------------------------
        allp = dn
        rep["agregado_2020_2025"] = {"n": int(len(allp)), "acc_todos": float(allp.hit.mean()),
                                     "metricas_prob": summary(allp.over.astype(float).values, allp.p.values)}
        buckets = {}
        for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.01)):
            sub = allp[(allp.p_pick >= lo) & (allp.p_pick < hi)]
            buckets[f"{int(lo*100)}-{int(hi*100) if hi < 1 else '100'}"] = {
                "n": int(len(sub)), "prob_media": float(sub.p_pick.mean()) if len(sub) else None,
                "acierto_real": float(sub.hit.mean()) if len(sub) else None}
        rep["calibracion_buckets"] = buckets
        # picks agregados con umbral por temporada
        pk = []
        for T in SEASONS:
            t = thr_hist[T]["umbral"]
            if t is not None:
                pk.append(dn[(dn.season == T) & (dn.p_pick >= t)])
        pk = pd.concat(pk) if pk else dn.iloc[0:0]
        rep["picks_agregado"] = {"n": int(len(pk)), "acc": float(pk.hit.mean()) if len(pk) else None,
                                 "z": (float((pk.hit.mean() - 0.5) / np.sqrt(0.25 / len(pk))) if len(pk) else None)}
        report["targets"][mk] = rep
        print(mk, "dist:", rep["distribucion_elegida"], "acc_todos", round(rep["agregado_2020_2025"]["acc_todos"], 4),
              "picks", rep["picks_agregado"])
        for T, v in per.items():
            print("  ", T, "acc", round(v["acc_todos"], 3), "umbral", v["umbral"], "picks", v["n_picks"], v["acc_picks"])
    (OUT / "research_lines.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
