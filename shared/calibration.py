"""Metricas de calibracion. Se usan igual en NFL y en MLB."""
from __future__ import annotations

import numpy as np


def _clean(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~(np.isnan(y) | np.isnan(p))
    return y[m], np.clip(p[m], 1e-6, 1 - 1e-6)


def log_loss(y, p) -> float | None:
    y, p = _clean(y, p)
    if len(y) == 0:
        return None
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float | None:
    y, p = _clean(y, p)
    return float(np.mean((p - y) ** 2)) if len(y) else None


def accuracy(y, p, thr: float = 0.5) -> float | None:
    y, p = _clean(y, p)
    return float(np.mean((p >= thr) == (y == 1))) if len(y) else None


def ece(y, p, bins: int = 10) -> float | None:
    """Expected Calibration Error con bins de igual anchura."""
    y, p = _clean(y, p)
    if len(y) == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def reliability(y, p, bins: int = 10) -> list[dict]:
    """Curva de calibracion: por bucket, cuantas veces ocurrio realmente."""
    y, p = _clean(y, p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append({
            "bucket": f"{edges[b]:.0%}-{edges[b+1]:.0%}",
            "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "n": int(m.sum()),
            "predicted": float(p[m].mean()) if m.sum() else None,
            "observed": float(y[m].mean()) if m.sum() else None,
        })
    return out


def summary(y, p, bins: int = 10) -> dict:
    y2, p2 = _clean(y, p)
    return {
        "n": int(len(y2)),
        "accuracy": accuracy(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p, bins),
        "mean_predicted": float(p2.mean()) if len(p2) else None,
        "base_rate": float(y2.mean()) if len(y2) else None,
        "reliability": reliability(y, p, bins),
    }
