"""Distribucion de carreras de MLB: partido completo y primeras 5 entradas.

Version MLB_RUNS_DIST_v1.

Para que existe. El motor de MLB sabia decir "over 8.5 al 57 %", pero no sabia
decir "P(el equipo anota 0) = 6 %". Un mercado como "¿habra carreras en las
primeras 5?" o "¿anota Boston 4 carreras?" necesita la DISTRIBUCION completa,
no una probabilidad binaria contra una linea concreta.

Lo que dicen los datos reales (14 521 partidos, 2021-2026):

    total del partido      media 8.904   varianza 20.151   var/media = 2.26
    total primeras 5       media 5.019   varianza 11.149   var/media = 2.22

Las dos SOBREdispersas con creces. Una Poisson impone varianza = media y aqui
se quedaria corta por mas del doble: subestimaria las colas, o sea los partidos
de 15 carreras y los de 2. Por eso la binomial negativa entra con motivo, no
por adorno. Aun asi compite: las dos familias se miden fuera de muestra y gana
la que gane.

REGLA QUE NO SE ROMPE: F5 NO se deriva del partido completo. Tiene su propio
objetivo (`home_f5`, `away_f5`, marcadores reales de 14 529 partidos) y su
propio ajuste. Repartir el total de 9 entradas en proporcion 5/9 seria inventar
un dato que si existe medido.

Lo que se supone y hay que decirlo: al convolucionar las dos distribuciones de
equipo para obtener el total se supone INDEPENDENCIA entre lo que anota cada
uno. No es exacta (comparten parque, clima y dia), asi que `camino='total'`
existe justamente para no depender de ese supuesto y se compara contra el otro.

Ninguna feature de este modulo viene de una casa de apuestas. El constructor de
features de MLB no lee la tabla de cuotas.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

VERSION = "MLB_RUNS_DIST_v1"

MAX_CARRERAS = 25                # por equipo; cubre practicamente toda la masa
CARRERAS_MIN, CARRERAS_MAX = 0.5, 12.0
TRAMOS = ("completo", "f5")
# Columnas de marcador reales que hay en features.parquet.
OBJETIVO = {"completo": ("home_score", "away_score", "total_runs"),
            "f5": ("home_f5", "away_f5", "total_runs_f5")}
LINEAS = {"completo": (6.5, 7.5, 8.5, 9.5, 10.5, 11.5),
          "f5": (3.5, 4.5, 5.5, 6.5)}
# Columnas que NUNCA pueden ser feature: son el resultado o lo identifican.
PROHIBIDAS = ("home_score", "away_score", "home_f5", "away_f5", "total_runs",
              "total_runs_f5", "home_margin", "y_home_win", "y_home_rl", "y_home_f5",
              "game_id", "season", "game_date", "start_utc", "game_type", "status",
              "home_team_id", "away_team_id", "home_abbr", "away_abbr", "venue_id",
              "dh", "feature_version")


def columnas(X) -> list[str]:
    """Features utilizables. Se excluye todo lo que sea resultado o identificador."""
    return [c for c in X.columns if c not in PROHIBIDAS
            and X[c].dtype.kind in "ifb"]


def _pmf_poisson(lam: np.ndarray, n: int = MAX_CARRERAS) -> np.ndarray:
    k = np.arange(n + 1)
    lg = np.array([math.lgamma(i + 1) for i in k])
    lam = np.clip(np.asarray(lam, float), 1e-9, None)[:, None]
    p = np.exp(-lam + k * np.log(lam) - lg)
    return p / p.sum(axis=1, keepdims=True)


def _pmf_nbinom(lam: np.ndarray, phi: float, n: int = MAX_CARRERAS) -> np.ndarray:
    k = np.arange(n + 1)
    lam = np.clip(np.asarray(lam, float), 1e-9, None)[:, None]
    phi = max(float(phi), 1e-6)
    coef = np.array([math.lgamma(i + phi) - math.lgamma(phi) - math.lgamma(i + 1) for i in k])
    logp = coef + phi * np.log(phi / (phi + lam)) + k * np.log(lam / (phi + lam))
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def pmf_equipo(lam, familia: str = "nbinom", phi: float = 4.0,
               n: int = MAX_CARRERAS) -> np.ndarray:
    """P(el equipo anota k carreras), k = 0..n."""
    lam = np.atleast_1d(np.asarray(lam, float))
    return _pmf_poisson(lam, n) if familia == "poisson" else _pmf_nbinom(lam, phi, n)


def dist_total(lh, la, familia: str = "nbinom", phi: float = 4.0,
               camino: str = "suma") -> np.ndarray:
    """Distribucion del TOTAL de carreras del partido."""
    if camino == "total" or la is None:
        D = pmf_equipo(lh, familia, phi, 2 * MAX_CARRERAS)
        return D / D.sum(axis=1, keepdims=True)
    A, B = pmf_equipo(lh, familia, phi), pmf_equipo(la, familia, phi)
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


def al_menos(pmf: np.ndarray, k: int) -> np.ndarray:
    """P(el equipo anota k carreras o mas)."""
    return pmf[:, k:].sum(axis=1)


class EstimadorCarreras:
    """Carreras esperadas por equipo (o total directo) en un tramo del partido."""

    def __init__(self, tramo: str = "completo", objetivo: str = "equipos",
                 cols: list[str] | None = None) -> None:
        if tramo not in TRAMOS:
            raise ValueError(f"tramo desconocido: {tramo}")
        self.tramo = tramo
        self.objetivo = objetivo
        self.cols = cols
        self.mh = self.ma = self.mt = None
        self.phi = 4.0
        self.media = self.varianza = None

    def _nuevo(self):
        return HistGradientBoostingRegressor(
            loss="poisson", max_iter=300, learning_rate=0.06, max_depth=5,
            min_samples_leaf=60, l2_regularization=1.0, random_state=7)

    def fit(self, X, y_home=None, y_away=None, y_total=None):
        self.cols = self.cols or columnas(X)
        M = X.reindex(columns=self.cols).apply(
            lambda c: c.astype(float)).fillna(0.0)
        if self.objetivo == "total":
            self.mt = self._nuevo().fit(M, np.asarray(y_total, float))
            r = np.asarray(y_total, float)
        else:
            self.mh = self._nuevo().fit(M, np.asarray(y_home, float))
            self.ma = self._nuevo().fit(M, np.asarray(y_away, float))
            r = np.concatenate([np.asarray(y_home, float), np.asarray(y_away, float)])
        self.media, self.varianza = float(r.mean()), float(r.var())
        # phi por momentos, SOLO con datos de entrenamiento.
        self.phi = (max(self.media ** 2 / max(self.varianza - self.media, 1e-6), 0.5)
                    if self.varianza > self.media else 200.0)
        return self

    def predict(self, X):
        M = X.reindex(columns=self.cols).apply(lambda c: c.astype(float)).fillna(0.0)
        tope = CARRERAS_MAX if self.tramo == "completo" else CARRERAS_MAX / 2
        if self.objetivo == "total":
            return np.clip(self.mt.predict(M), 2 * CARRERAS_MIN, 2 * tope), None
        return (np.clip(self.mh.predict(M), CARRERAS_MIN, tope),
                np.clip(self.ma.predict(M), CARRERAS_MIN, tope))


def mercados(lh, la, tramo: str = "completo", familia: str = "nbinom", phi: float = 4.0,
             camino: str = "suma", lineas=None) -> dict:
    """Todo lo derivable de la distribucion, listo para publicar o para el chat."""
    lh = np.atleast_1d(np.asarray(lh, float))
    la = np.atleast_1d(np.asarray(la, float)) if la is not None else None
    D = dist_total(lh, la, familia, phi, camino)
    out = {"version": VERSION, "tramo": tramo, "familia": familia, "camino": camino,
           "lambda_home": lh, "lambda_away": la,
           "expected_runs": esperado(D), "game_total": {}, "distribucion_total": D}
    for ln in (lineas or LINEAS[tramo]):
        o, u, p = over_under(D, ln)
        out["game_total"][f"{ln:g}"] = {"over": o, "under": u, "push": p}
    if la is not None:
        ph, pa = pmf_equipo(lh, familia, phi), pmf_equipo(la, familia, phi)
        out["pmf_home"], out["pmf_away"] = ph, pa
        out["home_expected"], out["away_expected"] = lh, la
        out["team_total"] = {
            "home": {f"{k}+": al_menos(ph, k) for k in (1, 2, 3, 4, 5)},
            "away": {f"{k}+": al_menos(pa, k) for k in (1, 2, 3, 4, 5)},
        }
        out["home_cero"], out["away_cero"] = ph[:, 0], pa[:, 0]
        out["ambos_anotan"] = al_menos(ph, 1) * al_menos(pa, 1)
        out["alguna_carrera"] = 1.0 - (ph[:, 0] * pa[:, 0])
        out["nota_independencia"] = (
            "el total y 'ambos anotan' se calculan convolucionando las dos "
            "distribuciones de equipo bajo INDEPENDENCIA. Comparten parque, clima "
            "y dia, asi que la independencia es una aproximacion, no un hecho.")
    return out
