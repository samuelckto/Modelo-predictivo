"""Upset risk / CASINAZO para MLB.

NO significa "el underdog va a ganar". Significa: esta prediccion tiene mas
incertidumbre de lo normal y merece precaucion.

Los componentes son magnitudes medibles, no opiniones. Los pesos NO se inventan:
`fit_weights` los ajusta con una regresion logistica que predice el ERROR del
modelo en datos pasados, y `validate` comprueba a posteriori que a mayor riesgo
declarado, menor acierto real. Si esa relacion no aparece, el riesgo no sirve y
hay que decirlo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("closeness", "dispersion", "elo_gap", "missing_data", "lineup_unconfirmed",
              "starter_unknown", "market_gap")


def components(p_final, dispersion=0.0, p_elo=None, missing=0, lineup_confirmed=True,
               starter_known=True, p_market=None) -> dict:
    c = {
        # 1 cuando la probabilidad esta pegada al 50 %
        "closeness": float(1.0 - abs(float(p_final) - 0.5) * 2),
        # desacuerdo entre los algoritmos del ensemble (0.05 ya es mucho)
        "dispersion": float(min(dispersion / 0.05, 1.0)),
        "elo_gap": float(min(abs(float(p_final) - float(p_elo)) / 0.15, 1.0))
                   if p_elo is not None else 0.0,
        "missing_data": float(min(missing / 4.0, 1.0)),
        "lineup_unconfirmed": 0.0 if lineup_confirmed else 1.0,
        "starter_unknown": 0.0 if starter_known else 1.0,
        "market_gap": float(min(abs(float(p_final) - float(p_market)) / 0.15, 1.0))
                      if p_market is not None else 0.0,
    }
    return c


DEFAULT_WEIGHTS = {"closeness": 1.0, "dispersion": 0.5, "elo_gap": 0.5, "missing_data": 0.6,
                   "lineup_unconfirmed": 0.4, "starter_unknown": 0.6, "market_gap": 0.5}


def score(c: dict, weights: dict | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    keys = [k for k in COMPONENTS if k in c]
    tot = sum(abs(w.get(k, 0.0)) for k in keys) or 1.0
    raw = sum(w.get(k, 0.0) * c[k] for k in keys) / tot
    return float(np.clip(raw, 0, 1) * 100)


def label(r) -> str:
    return "HIGH RISK" if r >= 65 else "MEDIUM RISK" if r >= 40 else "LOW RISK"


def fit_weights(df: pd.DataFrame) -> dict:
    """Ajusta pesos prediciendo el error del modelo. `df` necesita las columnas de
    COMPONENTS presentes mas `wrong` (1 si el pick fallo)."""
    from sklearn.linear_model import LogisticRegression
    cols = [c for c in COMPONENTS if c in df.columns and df[c].std() > 1e-9]
    d = df.dropna(subset=cols + ["wrong"])
    if len(d) < 500 or not cols:
        return dict(DEFAULT_WEIGHTS)
    lr = LogisticRegression(max_iter=1000).fit(d[cols], d.wrong.astype(int))
    w = {c: float(v) for c, v in zip(cols, lr.coef_[0])}
    for k in COMPONENTS:                       # los ausentes no aportan
        w.setdefault(k, 0.0)
    return w


def validate(risk: np.ndarray, wrong: np.ndarray, bins=(0, 40, 65, 101)) -> dict:
    """A mayor riesgo declarado deberia haber mas fallos. Si no ocurre, se dice."""
    r = np.asarray(risk, float)
    w = np.asarray(wrong, float)
    out = {"buckets": [], "monotonic": None, "n": int(len(r))}
    prev = -1.0
    mono = True
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() == 0:
            out["buckets"].append({"range": f"{lo}-{hi-1}", "n": 0, "error_rate": None})
            continue
        er = float(w[m].mean())
        out["buckets"].append({"range": f"{lo}-{hi-1}", "n": int(m.sum()),
                               "error_rate": round(er, 4),
                               "accuracy": round(1 - er, 4)})
        if er < prev:
            mono = False
        prev = er
    out["monotonic"] = mono
    valid = [b for b in out["buckets"] if b["n"] > 50 and b["error_rate"] is not None]
    if len(valid) >= 2:
        out["spread_pp"] = round((valid[-1]["error_rate"] - valid[0]["error_rate"]) * 100, 2)
        out["useful"] = bool(mono and out["spread_pp"] > 1.0)
    else:
        out["useful"] = None
    return out
