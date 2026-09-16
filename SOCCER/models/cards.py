"""Mercado de TARJETAS AMARILLAS. Version SOCCER_CARDS_v1.

Que hay realmente detras, medido sobre los 25 237 partidos con tarjetas de la
base (2012-13 a 2025-26, cinco ligas europeas):

    por equipo y partido    media 2.0314   varianza 1.8938   -> INFRAdisperso
    total del partido       media 4.0627   varianza 4.5442   -> SOBREdisperso

Los dos numeros juntos dicen algo concreto: si las tarjetas de los dos equipos
fueran independientes, la varianza del total seria la suma de las varianzas
(3.79) y se observa 4.54. Sobra varianza, o sea que las tarjetas de un equipo y
las del rival estan POSITIVAMENTE correlacionadas. Tiene sentido futbolistico:
un partido aspero reparte tarjetas a los dos lados, no a uno.

Por eso se comparan dos caminos y no se da por bueno el obvio:

  suma    dos lambdas por equipo y se convoluciona. Respeta la asimetria
          local/visitante pero SUPONE independencia, que acabamos de ver que
          no se cumple, asi que subestimara la cola alta.
  total   se modela el total directamente. Pierde el desglose por equipo pero
          absorbe la correlacion sin tener que modelarla.

Y dos familias de conteo sobre cada camino, Poisson y binomial negativa. El
ganador se decide fuera de muestra, nunca por lo que "deberia" ganar.

LO QUE FALTA, Y HAY QUE DECIRLO: el arbitro. Es el factor mas fuerte conocido
en este mercado y la base NO guarda quien pito cada partido, asi que no entra
como feature. Cualquier ventaja que reclamara este modelo sin arbitro seria
sospechosa. Se publica como PROJECTION.

Cobertura: solo EPL, LALIGA, SERIEA, BUNDES y LIGUE1 traen amarillas. Champions,
Liga MX y Saudi quedan INSUFFICIENT DATA en este mercado.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from SOCCER.features.build import ID_COLS, TARGET_COLS, numeric

VERSION = "SOCCER_CARDS_v1"
MAX_TARJETAS = 20                   # el maximo observado por equipo es muy inferior
TARJETAS_MIN, TARJETAS_MAX = 0.3, 6.0
# Lineas habituales del mercado de amarillas totales. Las enteras dan PUSH.
LINEAS = (2.5, 3.5, 4.5, 5.5, 6.5)
LIGAS_CON_DATOS = ("EPL", "LALIGA", "SERIEA", "BUNDES", "LIGUE1")


def columnas(X) -> list[str]:
    return [c for c in X.columns if c not in ID_COLS + TARGET_COLS]


def _pmf_poisson(lam: np.ndarray, n: int = MAX_TARJETAS) -> np.ndarray:
    k = np.arange(n + 1)
    lg = np.array([math.lgamma(i + 1) for i in k])
    lam = np.clip(np.asarray(lam, float), 1e-9, None)[:, None]
    p = np.exp(-lam + k * np.log(lam) - lg)
    return p / p.sum(axis=1, keepdims=True)


def _pmf_nbinom(lam: np.ndarray, phi: float, n: int = MAX_TARJETAS) -> np.ndarray:
    k = np.arange(n + 1)
    lam = np.clip(np.asarray(lam, float), 1e-9, None)[:, None]
    phi = max(float(phi), 1e-6)
    coef = np.array([math.lgamma(i + phi) - math.lgamma(phi) - math.lgamma(i + 1) for i in k])
    logp = coef + phi * np.log(phi / (phi + lam)) + k * np.log(lam / (phi + lam))
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def dist_total(lh: np.ndarray, la: np.ndarray | None, familia: str = "nbinom",
               phi: float = 20.0, camino: str = "total") -> np.ndarray:
    """Distribucion del TOTAL de amarillas del partido."""
    def pmf(x, n=MAX_TARJETAS):
        return _pmf_poisson(x, n) if familia == "poisson" else _pmf_nbinom(x, phi, n)
    if camino == "total" or la is None:
        D = pmf(np.asarray(lh, float), 2 * MAX_TARJETAS)
        return D / D.sum(axis=1, keepdims=True)
    A, B = pmf(lh), pmf(la)
    n = A.shape[1]
    out = np.zeros((A.shape[0], 2 * n - 1))
    for i in range(n):
        out[:, i:i + n] += A[:, i][:, None] * B
    return out / out.sum(axis=1, keepdims=True)


def over_under(D: np.ndarray, linea: float):
    tot = np.arange(D.shape[1])
    return (D[:, tot > linea].sum(axis=1),
            D[:, tot < linea].sum(axis=1),
            D[:, tot == linea].sum(axis=1))


def esperado(D: np.ndarray) -> np.ndarray:
    return (D * np.arange(D.shape[1])).sum(axis=1)


def al_menos_una(lam, familia: str = "nbinom", phi: float = 20.0) -> np.ndarray:
    """P(un equipo recibe al menos una amarilla) = 1 - P(0)."""
    D = _pmf_poisson(lam) if familia == "poisson" else _pmf_nbinom(lam, phi)
    return 1.0 - D[:, 0]


def ambos_reciben(lh, la, familia: str = "nbinom", phi: float = 20.0) -> np.ndarray:
    """P(los DOS equipos reciben amarilla).

    Bajo independencia es el producto. Sabemos que hay correlacion positiva,
    asi que este numero es una COTA INFERIOR: en la realidad ocurre algo mas a
    menudo. Se devuelve tal cual y se declara el sesgo, en lugar de corregirlo
    con un factor inventado.
    """
    return al_menos_una(lh, familia, phi) * al_menos_una(la, familia, phi)


class EstimadorTarjetas:
    """Estima amarillas esperadas. `objetivo` = 'equipos' (dos lambdas) o 'total'."""

    def __init__(self, algoritmo: str = "hgb", objetivo: str = "total",
                 cols: list[str] | None = None) -> None:
        self.algoritmo = algoritmo
        self.objetivo = objetivo
        self.cols = cols
        self.mh = self.ma = self.mt = None
        self.phi = 20.0

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
        # phi por momentos, SOLO en train. Si los datos estan infradispersos
        # (var < media, que es el caso por equipo) la binomial negativa no puede
        # representarlos: phi se va a un valor alto y la NB colapsa a Poisson,
        # que es el comportamiento correcto y no un error.
        self.phi = max(media ** 2 / max(var - media, 1e-6), 0.5) if var > media else 500.0
        return self

    def predict(self, X):
        M = numeric(X, self.cols)
        if self.objetivo == "total":
            lt = np.clip(self.mt.predict(M), 2 * TARJETAS_MIN, 2 * TARJETAS_MAX)
            return lt, None
        lh = np.clip(self.mh.predict(M), TARJETAS_MIN, TARJETAS_MAX)
        la = np.clip(self.ma.predict(M), TARJETAS_MIN, TARJETAS_MAX)
        return lh, la


def mercados(lh, la, familia: str = "nbinom", phi: float = 20.0, camino: str = "total",
             lineas=LINEAS) -> dict:
    """Salida publicable: over/under por linea, esperadas y 'ambos reciben'."""
    lh = np.atleast_1d(np.asarray(lh, float))
    la = np.atleast_1d(np.asarray(la, float)) if la is not None else None
    D = dist_total(lh, la, familia, phi, camino)
    out = {"version": VERSION, "familia": familia, "camino": camino,
           "lambda_home": lh, "lambda_away": la,
           "expected_cards": esperado(D), "totales": {}, "distribucion": D}
    for ln in lineas:
        o, u, p = over_under(D, ln)
        out["totales"][f"{ln:g}"] = {"over": o, "under": u, "push": p}
    if la is not None:
        out["home_al_menos_1"] = al_menos_una(lh, familia, phi)
        out["away_al_menos_1"] = al_menos_una(la, familia, phi)
        out["ambos_reciben"] = ambos_reciben(lh, la, familia, phi)
        out["ambos_reciben_nota"] = (
            "calculado bajo independencia entre los dos equipos. Los datos muestran "
            "correlacion positiva (varianza del total 4.54 frente a 3.79 esperada), "
            "asi que este valor es una cota INFERIOR de la probabilidad real.")
    return out
