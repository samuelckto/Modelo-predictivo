"""Distribucion conjunta de goles y los tres mercados derivados de ella.

La idea central del modulo: NO son tres modelos sueltos. Se estima una sola
distribucion P(goles_local = i, goles_visitante = j) y de ella salen 1X2, BTTS y
Over/Under, que asi quedan matematicamente coherentes entre si (por ejemplo,
P(BTTS) nunca puede contradecir a P(Over 0.5)).

Familias implementadas, todas sobre las mismas lambdas:

  poisson            independiente. La referencia clasica.
  dixon_coles        Poisson con la correccion tau para marcadores bajos, que es
                     justo donde el Poisson puro falla (0-0, 1-0, 0-1, 1-1).
  negative_binomial  permite sobredispersion (varianza > media).
  bivariate_poisson  covarianza positiva explicita mediante un termino comun.

Ninguna se elige por elegancia: la eligen los resultados fuera de muestra en
`SOCCER/evaluation/walkforward.py`.
"""
from __future__ import annotations

import math

import numpy as np

MAX_GOLES = 10                      # cubre >99.9% de la masa en futbol


# --------------------------------------------------------------------------- #
# Distribuciones marginales
# --------------------------------------------------------------------------- #
def _poisson_pmf(lam: float, n: int = MAX_GOLES) -> np.ndarray:
    k = np.arange(n + 1)
    with np.errstate(over="ignore"):
        p = np.exp(-lam + k * np.log(max(lam, 1e-9)) - np.array([math.lgamma(i + 1) for i in k]))
    return p / p.sum()


def _nbinom_pmf(lam: float, phi: float, n: int = MAX_GOLES) -> np.ndarray:
    """Binomial negativa parametrizada por media `lam` y dispersion `phi`.

    var = lam + lam^2/phi. Con phi -> infinito recupera Poisson.
    """
    lam = max(lam, 1e-9)
    phi = max(phi, 1e-6)
    k = np.arange(n + 1)
    logp = (np.array([math.lgamma(i + phi) - math.lgamma(phi) - math.lgamma(i + 1) for i in k])
            + phi * math.log(phi / (phi + lam)) + k * math.log(lam / (phi + lam)))
    p = np.exp(logp)
    return p / p.sum()


# --------------------------------------------------------------------------- #
# Distribuciones conjuntas
# --------------------------------------------------------------------------- #
def _tau_dixon_coles(i: int, j: int, lh: float, la: float, rho: float) -> float:
    """Correccion de Dixon-Coles para los cuatro marcadores bajos."""
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 0 and j == 1:
        return 1.0 + lh * rho
    if i == 1 and j == 0:
        return 1.0 + la * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def joint(lambda_home: float, lambda_away: float, familia: str = "dixon_coles",
          rho: float = -0.03, phi: float = 8.0, n: int = MAX_GOLES) -> np.ndarray:
    """Matriz P(i,j) normalizada. `i` = goles del local, `j` = del visitante."""
    lh = float(max(lambda_home, 1e-6))
    la = float(max(lambda_away, 1e-6))

    if familia == "poisson":
        m = np.outer(_poisson_pmf(lh, n), _poisson_pmf(la, n))

    elif familia == "dixon_coles":
        m = np.outer(_poisson_pmf(lh, n), _poisson_pmf(la, n))
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= _tau_dixon_coles(i, j, lh, la, rho)
        m = np.clip(m, 1e-15, None)

    elif familia == "negative_binomial":
        m = np.outer(_nbinom_pmf(lh, phi, n), _nbinom_pmf(la, phi, n))

    elif familia == "bivariate_poisson":
        # X = X1 + X3, Y = X2 + X3 con X3 ~ Poisson(l3): induce covarianza l3 >= 0.
        l3 = min(rho if rho > 0 else 0.08, lh, la) * 1.0
        l1, l2 = max(lh - l3, 1e-6), max(la - l3, 1e-6)
        p1, p2, p3 = _poisson_pmf(l1, n), _poisson_pmf(l2, n), _poisson_pmf(l3, n)
        m = np.zeros((n + 1, n + 1))
        for k in range(n + 1):
            if p3[k] < 1e-12:
                continue
            a = p1[: n + 1 - k]
            b = p2[: n + 1 - k]
            m[k:, k:] += p3[k] * np.outer(a, b)
    else:
        raise ValueError(f"familia desconocida: {familia}")

    return m / m.sum()


# --------------------------------------------------------------------------- #
# Mercados derivados de la conjunta
# --------------------------------------------------------------------------- #
def probs_1x2(m: np.ndarray) -> tuple[float, float, float]:
    n = m.shape[0]
    idx = np.arange(n)
    local = float(m[idx[:, None] > idx[None, :]].sum())
    empate = float(np.trace(m))
    visita = float(m[idx[:, None] < idx[None, :]].sum())
    t = local + empate + visita
    return local / t, empate / t, visita / t


def prob_btts(m: np.ndarray) -> float:
    """Ambos marcan = 1 - P(local 0) - P(visita 0) + P(0-0)."""
    return float(1.0 - m[0, :].sum() - m[:, 0].sum() + m[0, 0])


def dist_total(m: np.ndarray) -> np.ndarray:
    """Distribucion del total de goles del partido."""
    n = m.shape[0]
    out = np.zeros(2 * n - 1)
    for i in range(n):
        out[i:i + n] += m[i, :]
    return out


def prob_over(m: np.ndarray, linea: float) -> tuple[float, float, float]:
    """Devuelve (P(over), P(under), P(push)). Push solo existe en linea entera."""
    d = dist_total(m)
    tot = np.arange(len(d))
    over = float(d[tot > linea].sum())
    under = float(d[tot < linea].sum())
    push = float(d[tot == linea].sum())          # 0 si la linea es .5
    return over, under, push


def esperados(m: np.ndarray) -> dict:
    n = m.shape[0]
    k = np.arange(n)
    eh = float((m.sum(axis=1) * k).sum())
    ea = float((m.sum(axis=0) * k).sum())
    return {"home": eh, "away": ea, "total": eh + ea}


# --------------------------------------------------------------------------- #
# Version vectorizada: mismos numeros, miles de partidos a la vez
# --------------------------------------------------------------------------- #
def _pmf_batch(lam: np.ndarray, n: int = MAX_GOLES) -> np.ndarray:
    """(N, n+1) con la Poisson de cada lambda."""
    k = np.arange(n + 1)
    lg = np.array([math.lgamma(i + 1) for i in k])
    lam = np.clip(lam, 1e-9, None)[:, None]
    p = np.exp(-lam + k * np.log(lam) - lg)
    return p / p.sum(axis=1, keepdims=True)


def _nbinom_batch(lam: np.ndarray, phi: float, n: int = MAX_GOLES) -> np.ndarray:
    k = np.arange(n + 1)
    lam = np.clip(lam, 1e-9, None)[:, None]
    phi = max(phi, 1e-6)
    coef = np.array([math.lgamma(i + phi) - math.lgamma(phi) - math.lgamma(i + 1) for i in k])
    logp = coef + phi * np.log(phi / (phi + lam)) + k * np.log(lam / (phi + lam))
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def mercados_batch(lh: np.ndarray, la: np.ndarray, familia: str = "dixon_coles",
                   rho: float = -0.03, phi: float = 8.0,
                   lineas: tuple[float, ...] = (2.5,), n: int = MAX_GOLES) -> dict:
    """1X2, BTTS, totales y goles esperados para N partidos de una vez.

    Da exactamente lo mismo que `mercados()` partido a partido; existe porque el
    walk-forward evalua cientos de miles de combinaciones y el bucle no acaba.
    """
    lh = np.asarray(lh, dtype=float)
    la = np.asarray(la, dtype=float)
    N = len(lh)

    if familia in ("poisson", "dixon_coles"):
        A, B = _pmf_batch(lh, n), _pmf_batch(la, n)
    elif familia == "negative_binomial":
        A, B = _nbinom_batch(lh, phi, n), _nbinom_batch(la, phi, n)
    elif familia == "bivariate_poisson":
        l3 = np.minimum(np.minimum(lh, la), max(rho, 0.0) if rho > 0 else 0.08)
        A = _pmf_batch(np.maximum(lh - l3, 1e-6), n)
        B = _pmf_batch(np.maximum(la - l3, 1e-6), n)
        C = _pmf_batch(l3, n)
    else:
        raise ValueError(f"familia desconocida: {familia}")

    # matriz conjunta por partido (N, n+1, n+1)
    if familia == "bivariate_poisson":
        M = np.zeros((N, n + 1, n + 1))
        for k in range(n + 1):
            if C[:, k].max() < 1e-12:
                continue
            bloque = np.einsum("ni,nj->nij", A[:, : n + 1 - k], B[:, : n + 1 - k])
            M[:, k:, k:] += C[:, k][:, None, None] * bloque
    else:
        M = np.einsum("ni,nj->nij", A, B)
        if familia == "dixon_coles":
            M[:, 0, 0] *= 1.0 - lh * la * rho
            M[:, 0, 1] *= 1.0 + lh * rho
            M[:, 1, 0] *= 1.0 + la * rho
            M[:, 1, 1] *= 1.0 - rho
            M = np.clip(M, 1e-15, None)
    M /= M.sum(axis=(1, 2), keepdims=True)

    idx = np.arange(n + 1)
    tri_local = idx[:, None] > idx[None, :]
    tri_visita = idx[:, None] < idx[None, :]
    p_home = M[:, tri_local].sum(axis=1)
    p_away = M[:, tri_visita].sum(axis=1)
    p_draw = np.einsum("nii->n", M)
    btts = 1.0 - M[:, 0, :].sum(axis=1) - M[:, :, 0].sum(axis=1) + M[:, 0, 0]

    # distribucion del total por partido
    D = np.zeros((N, 2 * n + 1))
    for i in range(n + 1):
        D[:, i:i + n + 1] += M[:, i, :]
    tot = np.arange(2 * n + 1)
    totales = {}
    for ln in lineas:
        totales[f"{ln:g}"] = {
            "over": D[:, tot > ln].sum(axis=1),
            "under": D[:, tot < ln].sum(axis=1),
            "push": D[:, tot == ln].sum(axis=1),
        }
    eh = (M.sum(axis=2) * idx).sum(axis=1)
    ea = (M.sum(axis=1) * idx).sum(axis=1)
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "btts_yes": btts, "totales": totales,
            "eg_home": eh, "eg_away": ea, "eg_total": eh + ea, "matriz": M}


def mercados(lambda_home: float, lambda_away: float, familia: str = "dixon_coles",
             rho: float = -0.03, phi: float = 8.0,
             lineas: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5)) -> dict:
    """Todo lo que publica el sistema para un partido, desde UNA sola distribucion."""
    m = joint(lambda_home, lambda_away, familia, rho, phi)
    ph, pd_, pa = probs_1x2(m)
    tot = {}
    for ln in lineas:
        o, u, p = prob_over(m, ln)
        tot[f"{ln:g}"] = {"over": o, "under": u, "push": p}
    e = esperados(m)
    return {"familia": familia, "lambda_home": float(lambda_home), "lambda_away": float(lambda_away),
            "p_home": ph, "p_draw": pd_, "p_away": pa,
            "btts_yes": prob_btts(m), "btts_no": 1.0 - prob_btts(m),
            "totales": tot, "expected_goals": e, "matriz": m}
