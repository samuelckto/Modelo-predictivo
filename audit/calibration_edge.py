"""AUDITORIA §10 (calibracion por bucket), §11 (¿sirve el edge?) y §12 (riesgo).

Todo se calcula sobre las predicciones walk-forward partido a partido de la
variante LIMPIA (sin features del abridor).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from shared.calibration import summary               # noqa: E402

PRE = Path(__file__).resolve().parent / "preds"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True, parents=True)
BUCKETS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
           (0.70, 0.75), (0.75, 0.80), (0.80, 1.01)]


def load(market: str) -> pd.DataFrame:
    fs = sorted(PRE.glob(f"*_{market}.parquet"))
    fs = [f for f in fs if f.stem.split("_", 1)[1] == market]
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def to_pick(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pasa de 'probabilidad del local' a 'probabilidad de la seleccion'."""
    p = d[col].values
    home = p >= 0.5
    out = d.copy()
    out["pick_home"] = home
    out["p_pick"] = np.where(home, p, 1 - p)
    out["hit"] = np.where(home, d.y_home == 1, d.y_home == 0).astype(int)
    b = d.base_rate_train.values
    out["base_pick"] = np.where(home, b, 1 - b)
    out["edge"] = out.p_pick - out.base_pick
    return out


def buckets(d: pd.DataFrame) -> list[dict]:
    rows = []
    for lo, hi in BUCKETS:
        m = (d.p_pick >= lo) & (d.p_pick < hi)
        n = int(m.sum())
        rows.append({"bucket": f"{lo*100:.0f}-{hi*100:.0f}%", "n": n,
                     "predicho": round(float(d.p_pick[m].mean()), 4) if n else None,
                     "real": round(float(d.hit[m].mean()), 4) if n else None,
                     "dif_pp": (round((d.hit[m].mean() - d.p_pick[m].mean()) * 100, 2)
                                if n else None)})
    return rows


def edge_test(d: pd.DataFrame, k: int = 4) -> dict:
    """§11: ¿el edge ordena mejor que la probabilidad bruta o la confianza?

    Se parte cada criterio en cuartiles y se mide el acierto real por cuartil.
    Un buen criterio deberia dar cuartiles crecientes y separar mas.
    """
    res = {}
    d = d.copy()
    d["confianza"] = (d.p_pick - 0.5) * 2
    for name, col in (("edge_sobre_base", "edge"), ("probabilidad_bruta", "p_pick"),
                      ("confianza", "confianza")):
        try:
            q = pd.qcut(d[col], k, labels=False, duplicates="drop")
        except ValueError:
            continue
        g = d.assign(q=q).groupby("q").agg(n=("hit", "size"), acierto=("hit", "mean"),
                                           medio=(col, "mean"))
        res[name] = {"cuartiles": [{"q": int(i), "n": int(r.n),
                                    "acierto": round(float(r.acierto), 4),
                                    "valor_medio": round(float(r.medio), 4)}
                                   for i, r in g.iterrows()],
                     "separacion_pp": round(float(g.acierto.iloc[-1] - g.acierto.iloc[0]) * 100, 2),
                     "monotono": bool(g.acierto.is_monotonic_increasing)}
    return res


def top_n_test(d: pd.DataFrame, ns=(20, 50, 100, 200)) -> dict:
    """¿Los N mejores por edge aciertan mas que los N mejores por probabilidad?"""
    out = {}
    for n in ns:
        by_edge = d.nlargest(n, "edge").hit.mean()
        by_prob = d.nlargest(n, "p_pick").hit.mean()
        out[str(n)] = {"por_edge": round(float(by_edge), 4),
                       "por_probabilidad": round(float(by_prob), 4),
                       "dif_pp": round(float(by_edge - by_prob) * 100, 2)}
    return out


def risk_calibration(d: pd.DataFrame) -> dict:
    """§12: el riesgo publicado es 100-p. ¿Coincide con el fallo real?"""
    d = d.copy()
    d["riesgo"] = (1 - d.p_pick) * 100
    d["fallo"] = 1 - d.hit
    rows = []
    for lo, hi in ((0, 30), (30, 35), (35, 40), (40, 45), (45, 51)):
        m = (d.riesgo >= lo) & (d.riesgo < hi)
        n = int(m.sum())
        rows.append({"riesgo": f"{lo}-{hi}%", "n": n,
                     "riesgo_declarado": round(float(d.riesgo[m].mean()), 2) if n else None,
                     "fallo_real": round(float(d.fallo[m].mean() * 100), 2) if n else None,
                     "dif_pp": (round(float(d.fallo[m].mean() * 100 - d.riesgo[m].mean()), 2)
                                if n else None)})
    valid = [r for r in rows if r["n"] > 100]
    mono = all(valid[i]["fallo_real"] <= valid[i + 1]["fallo_real"] + 1e-9
               for i in range(len(valid) - 1)) if len(valid) > 1 else None
    err = float(np.mean([abs(r["dif_pp"]) for r in valid])) if valid else None
    return {"buckets": rows, "monotono": mono, "error_medio_pp": round(err, 2) if err else None,
            "veredicto": ("calibrado y monotono: el riesgo publicado significa lo que dice"
                          if mono and err is not None and err < 3 else
                          "revisar: el riesgo declarado no coincide con el fallo real")}


if __name__ == "__main__":
    res = {}
    for market in ("moneyline", "f5_moneyline", "run_line"):
        raw = load(market)
        if raw.empty:
            continue
        col = "p_home_elo" if market == "moneyline" else "p_home_model"
        d = to_pick(raw, col)
        res[market] = {"modelo_usado": col, "n": int(len(d)),
                       "acierto_global": round(float(d.hit.mean()), 4),
                       "calibracion_buckets": buckets(d),
                       "edge_vs_alternativas": edge_test(d),
                       "top_n": top_n_test(d),
                       "riesgo": risk_calibration(d),
                       "resumen_metricas": {k: v for k, v in
                                            summary(d.hit, d.p_pick).items()
                                            if k != "reliability"}}
        if market == "moneyline":
            dm = to_pick(raw, "p_home_model")
            res[market]["comparacion_modelo_ml"] = {
                "acierto": round(float(dm.hit.mean()), 4),
                "calibracion_buckets": buckets(dm)}
    (OUT / "calibration_edge.json").write_text(json.dumps(res, indent=1, default=str))
    for m, r in res.items():
        print(f"\n=== {m.upper()} ({r['modelo_usado']}, n={r['n']}, "
              f"acierto {r['acierto_global']*100:.2f}%)")
        print(f"{'bucket':<12}{'n':>7}{'predicho':>11}{'real':>9}{'dif':>9}")
        for b in r["calibracion_buckets"]:
            if not b["n"]:
                continue
            print(f"{b['bucket']:<12}{b['n']:>7}{b['predicho']*100:>10.1f}%"
                  f"{b['real']*100:>8.1f}%{b['dif_pp']:>+8.1f}pp")
