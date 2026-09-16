"""Estimacion de lambda_home y lambda_away (goles esperados de cada equipo).

Tres estimadores que se comparan fuera de muestra; ninguno se da por bueno:

  ratings      fuerza ofensiva/defensiva clasica sobre las medias as-of.
               Es el baseline honesto: no aprende nada, solo multiplica.
  poisson_glm  regresion de Poisson sobre las features (lineal, interpretable).
  hgb_poisson  gradient boosting con perdida de Poisson (no lineal).

Los tres entrenan DOS modelos, uno por equipo, sobre exactamente las mismas
features as-of. El modelo del local ve la fila tal cual; el del visitante ve la
misma fila (no se invierte: la asimetria local/visitante es informacion real).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from SOCCER.features.build import ID_COLS, TARGET_COLS, numeric

MEDIA_LIGA_MIN, MEDIA_LIGA_MAX = 0.15, 5.0     # recorte de seguridad de lambda


def columnas(X: pd.DataFrame) -> list[str]:
    """Features utilizables: todo menos identificadores y objetivos."""
    return [c for c in X.columns if c not in ID_COLS + TARGET_COLS]


class Ratings:
    """lambda = fuerza_ataque * fuerza_defensa_rival * media_de_la_liga.

    Las fuerzas salen de las medias ya calculadas as-of, asi que este estimador
    no mira nada que las features no vieran.
    """

    nombre = "ratings"

    def __init__(self) -> None:
        self.media_local: dict[str, float] = {}
        self.media_visita: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y_home, y_away) -> "Ratings":
        d = X.assign(_h=np.asarray(y_home), _a=np.asarray(y_away))
        self.media_local = d.groupby("league_code")["_h"].mean().to_dict()
        self.media_visita = d.groupby("league_code")["_a"].mean().to_dict()
        self._gl = float(d["_h"].mean())
        self._gv = float(d["_a"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ml = X["league_code"].map(self.media_local).fillna(self._gl).to_numpy(float)
        mv = X["league_code"].map(self.media_visita).fillna(self._gv).to_numpy(float)
        # ataque del local relativo a la media, por la debilidad defensiva del rival
        at_h = X["home_gf_casa"].to_numpy(float) / np.clip(ml, 1e-6, None)
        df_a = X["away_gc_fuera"].to_numpy(float) / np.clip(mv, 1e-6, None)
        at_a = X["away_gf_fuera"].to_numpy(float) / np.clip(mv, 1e-6, None)
        df_h = X["home_gc_casa"].to_numpy(float) / np.clip(ml, 1e-6, None)
        lh = np.clip(at_h * df_a * ml, MEDIA_LIGA_MIN, MEDIA_LIGA_MAX)
        la = np.clip(at_a * df_h * mv, MEDIA_LIGA_MIN, MEDIA_LIGA_MAX)
        return lh, la


class _Regresion:
    """Base para los estimadores que entrenan un regresor por equipo."""

    nombre = "regresion"

    def __init__(self, cols: list[str] | None = None) -> None:
        self.cols = cols
        self.mh = None
        self.ma = None

    def _nuevo(self):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y_home, y_away):
        self.cols = self.cols or columnas(X)
        M = numeric(X, self.cols)
        self.mh = self._nuevo().fit(M, np.asarray(y_home))
        self.ma = self._nuevo().fit(M, np.asarray(y_away))
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        M = numeric(X, self.cols)
        lh = np.clip(self.mh.predict(M), MEDIA_LIGA_MIN, MEDIA_LIGA_MAX)
        la = np.clip(self.ma.predict(M), MEDIA_LIGA_MIN, MEDIA_LIGA_MAX)
        return lh, la


class PoissonGLM(_Regresion):
    nombre = "poisson_glm"

    def _nuevo(self):
        return PoissonRegressor(alpha=1e-3, max_iter=800)


class HGBPoisson(_Regresion):
    nombre = "hgb_poisson"

    def _nuevo(self):
        return HistGradientBoostingRegressor(
            loss="poisson", max_iter=300, learning_rate=0.06, max_depth=5,
            min_samples_leaf=60, l2_regularization=1.0, random_state=7)


ESTIMADORES = {"ratings": Ratings, "poisson_glm": PoissonGLM, "hgb_poisson": HGBPoisson}


def ajustar_rho(X: pd.DataFrame, lh: np.ndarray, la: np.ndarray,
                rejilla=np.arange(-0.20, 0.061, 0.01)) -> float:
    """Elige rho de Dixon-Coles por maxima verosimilitud SOLO en el bloque de train.

    Vectorizado: la correccion tau solo toca cuatro celdas, asi que la
    verosimilitud se evalua sin construir la matriz conjunta de cada partido.
    """
    from SOCCER.models.goals import MAX_GOLES, _pmf_batch      # noqa: PLC0415

    hg = np.clip(X["home_goals"].to_numpy(int), 0, MAX_GOLES)
    ag = np.clip(X["away_goals"].to_numpy(int), 0, MAX_GOLES)
    A, B = _pmf_batch(lh), _pmf_batch(la)
    n = np.arange(len(hg))
    base = A[n, hg] * B[n, ag]                    # densidad Poisson del marcador real
    bajo = (hg <= 1) & (ag <= 1)
    mejor, mejor_ll = -0.03, -np.inf
    for r in rejilla:
        tau = np.ones(len(hg))
        m00 = bajo & (hg == 0) & (ag == 0)
        m01 = bajo & (hg == 0) & (ag == 1)
        m10 = bajo & (hg == 1) & (ag == 0)
        m11 = bajo & (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lh[m00] * la[m00] * r
        tau[m01] = 1.0 + lh[m01] * r
        tau[m10] = 1.0 + la[m10] * r
        tau[m11] = 1.0 - r
        if (tau <= 0).any():                      # rho invalido: produce probabilidad negativa
            continue
        ll = float(np.log(np.clip(base * tau, 1e-15, None)).sum())
        if ll > mejor_ll:
            mejor, mejor_ll = float(r), ll
    return mejor
