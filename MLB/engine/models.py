"""Ensemble MLB por mercado.

Cuatro algoritmos (logistica, random forest, XGBoost, LightGBM), cada uno
calibrado por separado con `CalibratedClassifierCV`, y combinados en espacio
logit con pesos inversos a su log loss fuera de muestra (OOF). Encima, un
calibrador final para que la probabilidad publicada este calibrada.

Ningun peso se elige a mano: salen del rendimiento OOF del propio entrenamiento.

Mercados:
  * moneyline      -> clasificacion (gana el local)
  * f5_moneyline   -> clasificacion (local por delante tras 5 entradas)
  * run_line       -> clasificacion (local gana por 2 o mas)
  * total          -> regresion de carreras + distribucion empirica del residuo,
                      que permite calcular P(over) para CUALQUIER linea sin
                      suponer una forma parametrica inventada.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.calibration import log_loss
from shared.odds import logit, sigmoid

warnings.filterwarnings("ignore")

MARKETS = ("moneyline", "f5_moneyline", "run_line", "total")
TARGET = {"moneyline": "y_home_win", "f5_moneyline": "y_home_f5",
          "run_line": "y_home_rl", "total": "total_runs"}

EXCLUDE_PREFIX = ("y_", "home_score", "away_score", "home_f5", "away_f5", "total_runs",
                  "home_margin", "game_id", "venue_id", "home_team_id", "away_team_id",
                  "h_sp_player_id", "a_sp_player_id", "d_sp_player_id", "season")
# d_season / h_season / a_season se colaban por el bucle de diferenciales.
# d_season es constante 0 (ambos equipos juegan la misma temporada): no aporta
# nada y no debe estar. Detectado en la auditoria §3.
EXCLUDE_EXACT = {"game_number", "d_season", "h_season", "a_season"}


CONTEXT = ("park_runs_factor", "is_doubleheader", "month", "h_rest_days", "a_rest_days")


def feature_columns(X: pd.DataFrame, mode: str = "diff",
                    exclude_starter: bool = False) -> list[str]:
    """`diff` (por defecto): diferenciales local-visitante + contexto.

    Se prefieren los diferenciales porque el resultado depende del enfrentamiento,
    no del nivel absoluto, y porque reduce a un tercio el numero de columnas
    (menos sobreajuste y entrenamiento mucho mas rapido). `all` conserva tambien
    los valores absolutos de cada equipo.
    """
    cols = []
    for c in X.columns:
        if c in EXCLUDE_EXACT or c.startswith(EXCLUDE_PREFIX):
            continue
        if not pd.api.types.is_numeric_dtype(X[c]):
            continue
        if X[c].notna().mean() < 0.5:      # menos de la mitad con dato: se descarta
            continue
        if mode == "diff" and not (c.startswith("d_") or c in CONTEXT):
            continue
        if exclude_starter and "_sp_" in c:
            continue
        cols.append(c)
    return sorted(cols)


def _native_ok(build, name: str) -> bool:
    """Comprueba que una libreria nativa sobrevive EL CAMINO REAL de entrenamiento.

    No basta con un `fit` simple: LightGBM y XGBoost pueden funcionar sueltos y
    reventar con `OSError: access violation` dentro del pipeline + calibrador +
    joblib, que es como los usa este sistema. Verificado en Windows 11 / Python
    3.12 con lightgbm 4.7.0: el ajuste directo funciona y el mismo modelo dentro
    de `CalibratedClassifierCV` sobre un DataFrame troceado falla.

    Por eso la prueba reproduce en miniatura exactamente esa llamada. Si falla,
    el algoritmo se descarta, se registra el motivo en UNAVAILABLE y el ensemble
    sigue con los demas: nunca se cae el sistema por una libreria rota.
    """
    try:
        import numpy as _np
        import pandas as _pd
        from sklearn.calibration import CalibratedClassifierCV as _Cal
        rng = _np.random.default_rng(0)
        d = _pd.DataFrame(rng.normal(size=(300, 6)),
                          columns=[f"f{i}" for i in range(6)])
        d.iloc[5, 2] = _np.nan
        y = (d.f0 > 0).astype(int).values
        idx = _np.arange(0, 300, 2)              # indice no contiguo, como en el real
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", build())])
        cal = _Cal(pipe, method="sigmoid", cv=3)
        cal.fit(d.iloc[idx], y[idx])
        cal.predict_proba(d.iloc[:20])
        return True
    except BaseException as e:                                # noqa: BLE001
        UNAVAILABLE[name] = f"{type(e).__name__}: {str(e)[:200]}"
        return False


UNAVAILABLE: dict[str, str] = {}     # algoritmos descartados por libreria rota


def _algos(seed: int = 7) -> dict:
    a = {
        "logistic": Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler()),
                              ("m", LogisticRegression(C=0.3, max_iter=1000))]),
        "forest": Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("m", RandomForestClassifier(
                                n_estimators=250, min_samples_leaf=25, max_features="sqrt",
                                n_jobs=-1, random_state=seed))]),
    }
    try:
        from xgboost import XGBClassifier

        def _xgb():
            return XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.6, reg_lambda=2.0, eval_metric="logloss",
                n_jobs=-1, random_state=seed)
        if _native_ok(_xgb, "xgboost"):
            a["xgboost"] = Pipeline([("imp", SimpleImputer(strategy="median")),
                                     ("m", _xgb())])
    except ImportError as e:
        UNAVAILABLE["xgboost"] = f"no instalado: {e}"
    try:
        from lightgbm import LGBMClassifier

        def _lgbm():
            return LGBMClassifier(
                n_estimators=300, num_leaves=15, learning_rate=0.05,
                min_child_samples=40, subsample=0.8, colsample_bytree=0.6,
                reg_lambda=2.0, n_jobs=-1, random_state=seed, verbose=-1)
        if _native_ok(_lgbm, "lightgbm"):
            a["lightgbm"] = Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("m", _lgbm())])
    except ImportError as e:
        UNAVAILABLE["lightgbm"] = f"no instalado: {e}"
    return a


@dataclass
class MarketModel:
    market: str
    features: list[str] = field(default_factory=list)
    members: dict = field(default_factory=dict)      # nombre -> modelo calibrado
    weights: dict = field(default_factory=dict)
    oof: dict = field(default_factory=dict)
    final_calibrator: object = None
    residuals: np.ndarray | None = None              # solo para 'total'
    train_seasons: list = field(default_factory=list)
    n_train: int = 0
    degraded: dict = field(default_factory=dict)   # algoritmos descartados y por que

    # -------------------------------------------------------------- predecir
    def _member_or_none(self, name, m, X):
        """Devuelve la probabilidad de un algoritmo, o None si su artefacto no es
        utilizable en esta maquina.

        LightGBM y XGBoost guardan un puntero nativo al serializar. Si el modelo
        se entreno con otra version o en otro sistema, la libreria puede fallar
        con OSError (violacion de acceso) en vez de con un error de Python. En ese
        caso NO se cae el sistema: se descarta ese algoritmo, se renormalizan los
        pesos y se registra el aviso en `self.degraded` para que se vea.
        """
        try:
            return np.clip(m.predict_proba(X[self.features])[:, 1], 1e-4, 1 - 1e-4)
        except (OSError, AttributeError, ValueError) as e:
            self.degraded = getattr(self, "degraded", {})
            self.degraded[name] = (
                f"{type(e).__name__}: {e}. El artefacto no es compatible con las "
                f"librerias de esta maquina. Reentrena con "
                f"`python -m MLB.engine.train_production --markets {self.market}`.")
            return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = np.zeros(len(X))
        usable = {}
        for name, m in self.members.items():
            p = self._member_or_none(name, m, X)
            if p is not None:
                usable[name] = p
        if not usable:
            raise RuntimeError(
                f"ningun algoritmo de '{self.market}' pudo predecir en esta maquina; "
                f"reentrena con `python -m MLB.engine.train_production "
                f"--markets {self.market}`")
        tot = sum(self.weights[n] for n in usable) or 1.0
        for name, p in usable.items():
            z += (self.weights[name] / tot) * np.log(p / (1 - p))
        p = 1 / (1 + np.exp(-z))
        if self.final_calibrator is not None:
            p = self.final_calibrator.predict_proba(
                np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                       (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1))[:, 1]
        return np.clip(p, 1e-4, 1 - 1e-4)

    def member_probs(self, X: pd.DataFrame) -> dict:
        out = {}
        for n, m in self.members.items():
            p = self._member_or_none(n, m, X)
            if p is not None:
                out[n] = p
        return out

    def predict_total(self, X: pd.DataFrame) -> np.ndarray:
        return self.members["reg"].predict(X[self.features])

    def prob_over(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """P(total > linea) con la distribucion EMPIRICA de los residuos de
        entrenamiento. No se supone normalidad ni Poisson: se usa lo observado."""
        mu = self.predict_total(X)
        res = self.residuals
        return np.array([float(np.mean((m + res) > line)) for m in mu])


def train(X: pd.DataFrame, market: str, train_seasons: list[int], seed: int = 7,
          progress=None, exclude_starter: bool = False) -> MarketModel:
    tgt = TARGET[market]
    tr = X[X.season.isin(train_seasons) & X[tgt].notna()].copy()
    feats = feature_columns(X, exclude_starter=exclude_starter)
    mm = MarketModel(market=market, features=feats, train_seasons=list(train_seasons),
                     n_train=int(len(tr)))
    if len(tr) < 500:
        raise ValueError(f"datos insuficientes para {market}: {len(tr)} filas")

    if market == "total":
        reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("m", RandomForestRegressor(n_estimators=400, min_samples_leaf=25,
                                                    max_features="sqrt", n_jobs=-1,
                                                    random_state=seed))])
        # residuos fuera de muestra, no del ajuste
        from sklearn.model_selection import KFold
        oof = np.zeros(len(tr))
        for a, b in KFold(4, shuffle=False).split(tr):
            reg.fit(tr.iloc[a][feats], tr.iloc[a][tgt])
            oof[b] = reg.predict(tr.iloc[b][feats])
        reg.fit(tr[feats], tr[tgt])
        mm.members = {"reg": reg}
        mm.residuals = (tr[tgt].values - oof)
        mm.oof = {"rmse": float(np.sqrt(np.mean((tr[tgt].values - oof) ** 2))),
                  "mae": float(np.mean(np.abs(tr[tgt].values - oof))),
                  "mean_total": float(tr[tgt].mean())}
        return mm

    y = tr[tgt].astype(int).values
    algos = _algos(seed)
    if UNAVAILABLE:
        mm.degraded = dict(UNAVAILABLE)
        if progress:
            for k, v in UNAVAILABLE.items():
                progress(f"AVISO: '{k}' no se usa en esta maquina -> {v[:120]}")
    if not algos:
        raise RuntimeError("ninguna libreria de modelado funciona en esta maquina")
    skf = StratifiedKFold(3, shuffle=False)
    oof_p = {}
    for name, algo in algos.items():
        if progress:
            progress(f"{market}: {name}")
        p = np.zeros(len(tr))
        for a, b in skf.split(tr[feats], y):
            cal = CalibratedClassifierCV(algo, method="sigmoid", cv=3)
            cal.fit(tr.iloc[a][feats], y[a])
            p[b] = cal.predict_proba(tr.iloc[b][feats])[:, 1]
        oof_p[name] = p
        ll = log_loss(y, p)
        mm.oof[name] = {"log_loss": round(float(ll), 5),
                        "accuracy": float(np.mean((p >= .5) == (y == 1)))}
        cal = CalibratedClassifierCV(algo, method="sigmoid", cv=3)
        cal.fit(tr[feats], y)
        mm.members[name] = cal

    # pesos inversos al exceso de log loss sobre el mejor
    lls = {n: mm.oof[n]["log_loss"] for n in mm.members}
    best = min(lls.values())
    raw = {n: np.exp(-(lls[n] - best) / 0.01) for n in lls}
    tot = sum(raw.values())
    mm.weights = {n: float(v / tot) for n, v in raw.items()}

    zz = sum(mm.weights[n] * np.log(np.clip(oof_p[n], 1e-4, 1 - 1e-4) /
                                    (1 - np.clip(oof_p[n], 1e-4, 1 - 1e-4)))
             for n in mm.members)
    ens = 1 / (1 + np.exp(-zz))
    mm.oof["ensemble"] = {"log_loss": round(float(log_loss(y, ens)), 5),
                          "accuracy": float(np.mean((ens >= .5) == (y == 1)))}
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.log(np.clip(ens, 1e-6, 1 - 1e-6) / (1 - np.clip(ens, 1e-6, 1 - 1e-6))).reshape(-1, 1), y)
    mm.final_calibrator = lr
    return mm


def dispersion(members: dict) -> np.ndarray:
    """Desviacion tipica entre algoritmos: mide desacuerdo interno del modelo."""
    if not members:
        return np.array([])
    return np.std(np.column_stack(list(members.values())), axis=1)
