"""Mercado de CORNERS: over / under del total del partido.

Sustituye al 1X2 como mercado publicado. Dos caminos que se comparan de verdad:

  suma      se estiman los corners de cada equipo por separado y se convoluciona
            la distribucion del total. Respeta la asimetria local/visitante.
  total     se modela directamente el total del partido.

Y dos familias de conteo sobre cada camino: Poisson y binomial negativa. Los
corners suelen estar sobredispersos (varianza > media), asi que la binomial
negativa tiene una razon fisica para estar aqui, pero solo entra si gana OOS.

Las lineas tipicas del mercado son 8.5, 9.5, 10.5, 11.5 y 12.5. Las enteras
(9, 10, 11) generan PUSH y estan contempladas.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from SOCCER.features.build import ID_COLS, TARGET_COLS, numeric

MAX_CORNERS = 30                    # cubre practicamente toda la masa
CORNERS_MIN, CORNERS_MAX = 1.0, 12.0
LINEAS = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)


def columnas(X) -> list[str]:
    return [c for c in X.columns if c not in ID_COLS + TARGET_COLS]


def _pmf_poisson(lam: np.ndarray, n: int = MAX_CORNERS) -> np.ndarray:
    k = np.arange(n + 1)
    lg = np.array([math.lgamma(i + 1) for i in k])
    lam = np.clip(np.asarray(lam, float), 1e-9, None)[:, None]
    p = np.exp(-lam + k * np.log(lam) - lg)
    return p / p.sum(axis=1, keepdims=True)


def _pmf_nbinom(lam: np.ndarray, phi: float, n: int = MAX_CORNERS) -> np.ndarray:
    k = np.arange(n + 1)
    lam = np.clip(np.asarray(lam, float), 1e-9, None)[:, None]
    phi = max(float(phi), 1e-6)
    coef = np.array([math.lgamma(i + phi) - math.lgamma(phi) - math.lgamma(i + 1) for i in k])
    logp = coef + phi * np.log(phi / (phi + lam)) + k * np.log(lam / (phi + lam))
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def dist_total(lh: np.ndarray, la: np.ndarray | None, familia: str = "nbinom",
               phi: float = 8.0, camino: str = "suma") -> np.ndarray:
    """Distribucion del TOTAL de corners por partido. Devuelve (N, 2*MAX+1)."""
    pmf = _pmf_poisson if familia == "poisson" else (lambda x, n=MAX_CORNERS: _pmf_nbinom(x, phi, n))
    if camino == "total" or la is None:
        D = pmf(lh)
        out = np.zeros((D.shape[0], 2 * MAX_CORNERS + 1))
        out[:, : D.shape[1]] = D
        return out
    A, B = pmf(lh), pmf(la)
    N, n = A.shape[0], A.shape[1]
    out = np.zeros((N, 2 * n - 1))
    for i in range(n):                       # convolucion explicita: n es pequeno
        out[:, i:i + n] += A[:, i][:, None] * B
    return out / out.sum(axis=1, keepdims=True)


def over_under(D: np.ndarray, linea: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tot = np.arange(D.shape[1])
    return (D[:, tot > linea].sum(axis=1),
            D[:, tot < linea].sum(axis=1),
            D[:, tot == linea].sum(axis=1))


def esperado(D: np.ndarray) -> np.ndarray:
    return (D * np.arange(D.shape[1])).sum(axis=1)


class EstimadorCorners:
    """Estima corners esperados. `objetivo` = 'equipos' (dos lambdas) o 'total'."""

    def __init__(self, algoritmo: str = "hgb", objetivo: str = "equipos",
                 cols: list[str] | None = None) -> None:
        self.algoritmo = algoritmo
        self.objetivo = objetivo
        self.cols = cols
        self.mh = self.ma = self.mt = None
        self.phi = 8.0

    def _nuevo(self):
        if self.algoritmo == "glm":
            return PoissonRegressor(alpha=1e-3, max_iter=800)
        return HistGradientBoostingRegressor(
            loss="poisson", max_iter=300, learning_rate=0.06, max_depth=5,
            min_samples_leaf=60, l2_regularization=1.0, random_state=7)

    def fit(self, X, y_home=None, y_away=None, y_total=None):
        self.cols = self.cols or columnas(X)
        M = numeric(X, self.cols)
        if self.objetivo == "total":
            self.mt = self._nuevo().fit(M, np.asarray(y_total, float))
            r = np.asarray(y_total, float)
        else:
            self.mh = self._nuevo().fit(M, np.asarray(y_home, float))
            self.ma = self._nuevo().fit(M, np.asarray(y_away, float))
            r = np.concatenate([np.asarray(y_home, float), np.asarray(y_away, float)])
        media, var = float(r.mean()), float(r.var())
        # phi de la binomial negativa por momentos, estimado SOLO en train
        self.phi = max(media ** 2 / max(var - media, 1e-6), 0.5) if var > media else 200.0
        return self

    def predict(self, X):
        M = numeric(X, self.cols)
        if self.objetivo == "total":
            lt = np.clip(self.mt.predict(M), 2 * CORNERS_MIN, 2 * CORNERS_MAX)
            return lt, None
        lh = np.clip(self.mh.predict(M), CORNERS_MIN, CORNERS_MAX)
        la = np.clip(self.ma.predict(M), CORNERS_MIN, CORNERS_MAX)
        return lh, la


def mercados(lh, la, familia: str = "nbinom", phi: float = 8.0, camino: str = "suma",
             lineas=LINEAS) -> dict:
    """Salida publicable: over/under por linea y corners esperados."""
    lh = np.atleast_1d(np.asarray(lh, float))
    la = np.atleast_1d(np.asarray(la, float)) if la is not None else None
    D = dist_total(lh, la, familia, phi, camino)
    out = {"familia": familia, "camino": camino,
           "lambda_home": lh, "lambda_away": la,
           "expected_corners": esperado(D), "totales": {}}
    for ln in lineas:
        o, u, p = over_under(D, ln)
        out["totales"][f"{ln:g}"] = {"over": o, "under": u, "push": p}
    return out
