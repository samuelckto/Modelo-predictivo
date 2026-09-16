"""Conversion de cuotas a probabilidad y eliminacion del vig.

Nada aqui inventa una cuota. Si falta un lado del mercado, se devuelve None y el
llamador debe marcar el dato como `unavailable`.
"""
from __future__ import annotations

import math

VIG_METHODS = ("proportional", "shin", "log")


def american_to_prob(ml) -> float | None:
    if ml is None:
        return None
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if math.isnan(ml) or ml == 0:
        return None
    return (-ml / (-ml + 100.0)) if ml < 0 else (100.0 / (ml + 100.0))


def decimal_to_prob(d) -> float | None:
    try:
        d = float(d)
    except (TypeError, ValueError):
        return None
    if math.isnan(d) or d <= 1.0:
        return None
    return 1.0 / d


def prob_to_american(p) -> float | None:
    if p is None or not (0 < p < 1):
        return None
    return round(-100 * p / (1 - p), 0) if p >= 0.5 else round(100 * (1 - p) / p, 0)


def overround(probs) -> float | None:
    vals = [p for p in probs if p is not None]
    if len(vals) < 2:
        return None
    return sum(vals) - 1.0


def devig(probs, method: str = "proportional") -> list | None:
    """Normaliza un mercado de N vias. Devuelve la lista de probabilidades sin vig.

    proportional: p_i / sum(p)          (metodo por defecto, transparente)
    log:          normaliza en espacio logaritmico (power method aproximado)
    shin:         modelo de Shin para insider trading, resuelto numericamente
    """
    vals = [p for p in probs if p is not None]
    if len(vals) != len(probs) or not vals or sum(vals) <= 0:
        return None
    s = sum(vals)
    if method == "proportional":
        return [p / s for p in vals]
    if method == "log":
        k, lo, hi = 1.0, 0.5, 3.0
        for _ in range(60):
            k = (lo + hi) / 2
            if sum(p ** k for p in vals) > 1:
                lo = k
            else:
                hi = k
        t = sum(p ** k for p in vals)
        return [p ** k / t for p in vals]
    if method == "shin":
        if len(vals) != 2:
            return devig(vals, "proportional")
        lo, hi = 0.0, 0.4
        for _ in range(80):
            z = (lo + hi) / 2
            tot = sum(_shin(p, z, s) for p in vals)
            if tot > 1:
                lo = z
            else:
                hi = z
        z = (lo + hi) / 2
        out = [_shin(p, z, s) for p in vals]
        t = sum(out)
        return [o / t for o in out]
    raise ValueError(f"metodo de devig desconocido: {method}")


def _shin(p: float, z: float, s: float) -> float:
    inner = z * z + 4 * (1 - z) * (p * p) / s
    return (math.sqrt(max(inner, 0.0)) - z) / (2 * (1 - z)) if z < 1 else p / s


def two_way(p_a, p_b, method: str = "proportional"):
    """Devuelve (prob_a_sin_vig, overround) o (None, None) si falta un lado."""
    if p_a is None or p_b is None:
        return None, None
    ov = overround([p_a, p_b])
    d = devig([p_a, p_b], method)
    return (d[0] if d else None), ov


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
