"""Validacion walk-forward. Aqui se decide todo, y se decide con numeros.

Estructura temporal, nunca aleatoria:

    entrenar con todo lo anterior a la temporada T  ->  evaluar la temporada T

Y dos bloques separados que no se mezclan:

    SELECCION  temporadas de seleccion: aqui se compara y se elige.
    HOLDOUT    temporadas finales: se miran UNA vez, al terminar. No se usan
               para elegir nada. Si el holdout sale peor que la seleccion, se
               reporta tal cual; no se vuelve atras a retocar.

Se comparan a la vez:
  - 3 estimadores de lambda  x  4 familias de distribucion conjunta
  - modelos independientes de BTTS y de total (para ver si derivar es peor)
  - baselines: frecuencia historica, media de goles, Elo

Metricas: log loss, Brier, MAE de goles, calibracion y accuracy (secundaria).
"""
from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from SOCCER.features.build import FEATURES_PARQUET, numeric
from SOCCER.models.goals import mercados_batch
from SOCCER.models.lambdas import ESTIMADORES, ajustar_rho, columnas
from shared.paths import SOCCER_OUT_DIR, assert_writable

FAMILIAS = ("poisson", "dixon_coles", "negative_binomial", "bivariate_poisson")
LINEA_PRINCIPAL = 2.5


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
def log_loss_multi(y: np.ndarray, P: np.ndarray) -> float:
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def brier_multi(y: np.ndarray, P: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(np.mean(((P - Y) ** 2).sum(axis=1)))


def log_loss_bin(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_bin(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calibracion(y: np.ndarray, p: np.ndarray, bordes=(0.5, 0.55, 0.6, 0.65, 0.7, 1.01)) -> list:
    """Buckets sobre el lado mostrado (probabilidad >= 0.5)."""
    lado = np.where(p >= 0.5, p, 1 - p)
    acierto = np.where(p >= 0.5, y, 1 - y)
    out = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (lado >= a) & (lado < b)
        if m.sum() == 0:
            continue
        out.append({"bucket": f"{int(a*100)}-{int(min(b,1)*100)}", "n": int(m.sum()),
                    "anunciada": round(float(lado[m].mean()), 4),
                    "real": round(float(acierto[m].mean()), 4)})
    return out


def ece(buckets: list) -> float | None:
    n = sum(b["n"] for b in buckets)
    if not n:
        return None
    return round(sum(b["n"] * abs(b["anunciada"] - b["real"]) for b in buckets) / n, 4)


# --------------------------------------------------------------------------- #
# Particion temporal
# --------------------------------------------------------------------------- #
def temporadas_ordenadas(X: pd.DataFrame) -> list[str]:
    return sorted(X["season"].dropna().unique().tolist())


def particion(X: pd.DataFrame, n_holdout: int = 2, min_train: int = 4) -> dict:
    ts = temporadas_ordenadas(X)
    holdout = ts[-n_holdout:]
    seleccion = ts[min_train:-n_holdout]
    return {"todas": ts, "seleccion": seleccion, "holdout": holdout,
            "train_inicial": ts[:min_train]}


# --------------------------------------------------------------------------- #
# Evaluacion de un bloque de temporadas
# --------------------------------------------------------------------------- #
def _predecir_temporada(X: pd.DataFrame, temp: str, est_nombre: str,
                        entrenar_indep: bool = True) -> dict:
    """Entrena con todo lo anterior a `temp` y predice `temp`. Sin mirar el futuro."""
    ts = temporadas_ordenadas(X)
    prev = [t for t in ts if t < temp]
    tr = X[X["season"].isin(prev)]
    te = X[X["season"] == temp]
    if len(tr) < 500 or te.empty:
        return {}

    est = ESTIMADORES[est_nombre]().fit(tr, tr["home_goals"], tr["away_goals"])
    lh_tr, la_tr = est.predict(tr)
    rho = ajustar_rho(tr, lh_tr, la_tr)
    lh, la = est.predict(te)

    # dispersion de la binomial negativa, estimada SOLO en train
    g = np.concatenate([tr["home_goals"].to_numpy(float), tr["away_goals"].to_numpy(float)])
    media, var = float(g.mean()), float(g.var())
    phi = max(media ** 2 / max(var - media, 1e-6), 0.5) if var > media else 50.0

    salida = {"temporada": temp, "n_train": int(len(tr)), "n_test": int(len(te)),
              "rho": rho, "phi": round(phi, 3), "familias": {}}

    y_1x2 = te["resultado"].to_numpy(int)
    y_btts = te["btts"].to_numpy(int)
    y_tot = te["total_goals"].to_numpy(int)
    y_over = (y_tot > LINEA_PRINCIPAL).astype(int)
    # 1X = el local no pierde. Se calcula una sola vez y NO toca a y_1x2.
    y_1x = (te["resultado"] != 2).astype(int).to_numpy()

    for fam in FAMILIAS:
        r = mercados_batch(lh, la, fam, rho=rho, phi=phi, lineas=(LINEA_PRINCIPAL,))
        P = np.column_stack([r["p_home"], r["p_draw"], r["p_away"]])
        p_btts = r["btts_yes"]
        p_over = r["totales"][f"{LINEA_PRINCIPAL:g}"]["over"]
        eg = np.column_stack([r["eg_home"], r["eg_away"]])
        # doble oportunidad: 1X = local o empate, X2 = visitante o empate
        p_1x = P[:, 0] + P[:, 1]
        cal_dc = calibracion(y_1x, p_1x)
        cal_b = calibracion(y_btts, p_btts)
        cal_o = calibracion(y_over, p_over)
        salida["familias"][fam] = {
            "1x2": {"log_loss": round(log_loss_multi(y_1x2, P), 5),
                    "brier": round(brier_multi(y_1x2, P), 5),
                    "accuracy": round(float((P.argmax(axis=1) == y_1x2).mean()), 4)},
            "btts": {"log_loss": round(log_loss_bin(y_btts, p_btts), 5),
                     "brier": round(brier_bin(y_btts, p_btts), 5),
                     "accuracy": round(float(((p_btts >= .5).astype(int) == y_btts).mean()), 4),
                     "ece": ece(cal_b), "calibracion": cal_b},
            "total_2.5": {"log_loss": round(log_loss_bin(y_over, p_over), 5),
                          "brier": round(brier_bin(y_over, p_over), 5),
                          "accuracy": round(float(((p_over >= .5).astype(int) == y_over).mean()), 4),
                          "ece": ece(cal_o), "calibracion": cal_o},
            "double_chance": {"log_loss": round(log_loss_bin(y_1x, p_1x), 5),
                              "brier": round(brier_bin(y_1x, p_1x), 5),
                              "accuracy": round(float(((p_1x >= .5).astype(int) == y_1x).mean()), 4),
                              "ece": ece(cal_dc), "calibracion": cal_dc},
            "goles": {"mae_home": round(float(np.abs(eg[:, 0] - te["home_goals"]).mean()), 4),
                      "mae_away": round(float(np.abs(eg[:, 1] - te["away_goals"]).mean()), 4),
                      "mae_total": round(float(np.abs(eg.sum(axis=1) - y_tot).mean()), 4),
                      "rmse_total": round(float(np.sqrt(((eg.sum(axis=1) - y_tot) ** 2).mean())), 4),
                      "media_predicha": round(float(eg.sum(axis=1).mean()), 4),
                      "media_real": round(float(y_tot.mean()), 4)},
        }

    # --- modelos INDEPENDIENTES: puede que derivar sea peor -------------------
    if entrenar_indep:
        cols = columnas(tr)
        Mtr, Mte = numeric(tr, cols), numeric(te, cols)
        indep = {}
        for nombre, y_tr, y_te in (("btts", tr["btts"].to_numpy(int), y_btts),
                                   ("total_2.5", (tr["total_goals"] > LINEA_PRINCIPAL).astype(int)
                                    .to_numpy(), y_over)):
            clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_depth=5, min_samples_leaf=60,
                l2_regularization=1.0, random_state=7).fit(Mtr, y_tr)
            p = clf.predict_proba(Mte)[:, 1]
            cal = calibracion(y_te, p)
            indep[nombre] = {"log_loss": round(log_loss_bin(y_te, p), 5),
                             "brier": round(brier_bin(y_te, p), 5),
                             "accuracy": round(float(((p >= .5).astype(int) == y_te).mean()), 4),
                             "ece": ece(cal), "calibracion": cal}
        clf3 = LogisticRegression(max_iter=1500, multi_class="multinomial", C=0.5)
        clf3.fit(Mtr, tr["resultado"].to_numpy(int))
        P3 = clf3.predict_proba(Mte)
        indep["1x2_multinomial"] = {"log_loss": round(log_loss_multi(y_1x2, P3), 5),
                                    "brier": round(brier_multi(y_1x2, P3), 5),
                                    "accuracy": round(float((P3.argmax(axis=1) == y_1x2).mean()), 4)}
        salida["independientes"] = indep

    # --- baselines -----------------------------------------------------------
    base_1x2 = np.tile([(tr["resultado"] == k).mean() for k in (0, 1, 2)], (len(te), 1))
    p_1x_base = np.full(len(te), float((tr["resultado"] != 2).mean()))
    y_1x_te = (te["resultado"] != 2).astype(int).to_numpy()
    p_btts_base = np.full(len(te), float(tr["btts"].mean()))
    p_over_base = np.full(len(te), float((tr["total_goals"] > LINEA_PRINCIPAL).mean()))
    media_goles = float(tr["total_goals"].mean())
    p_elo = te["elo_p_home"].to_numpy(float)
    P_elo = np.column_stack([p_elo * 0.75, np.full(len(te), 0.25), (1 - p_elo) * 0.75])
    P_elo = P_elo / P_elo.sum(axis=1, keepdims=True)
    salida["baselines"] = {
        "1x2_frecuencia": {"log_loss": round(log_loss_multi(y_1x2, base_1x2), 5),
                           "brier": round(brier_multi(y_1x2, base_1x2), 5),
                           "accuracy": round(float((base_1x2.argmax(axis=1) == y_1x2).mean()), 4)},
        "1x2_elo": {"log_loss": round(log_loss_multi(y_1x2, P_elo), 5),
                    "brier": round(brier_multi(y_1x2, P_elo), 5),
                    "accuracy": round(float((P_elo.argmax(axis=1) == y_1x2).mean()), 4)},
        "btts_frecuencia": {"log_loss": round(log_loss_bin(y_btts, p_btts_base), 5),
                            "brier": round(brier_bin(y_btts, p_btts_base), 5),
                            "accuracy": round(float(((p_btts_base >= .5).astype(int) == y_btts).mean()), 4)},
        "total_frecuencia": {"log_loss": round(log_loss_bin(y_over, p_over_base), 5),
                             "brier": round(brier_bin(y_over, p_over_base), 5),
                             "accuracy": round(float(((p_over_base >= .5).astype(int) == y_over).mean()), 4)},
        "double_chance_frecuencia": {
            "log_loss": round(log_loss_bin(y_1x_te, p_1x_base), 5),
            "brier": round(brier_bin(y_1x_te, p_1x_base), 5),
            "accuracy": round(float(((p_1x_base >= .5).astype(int) == y_1x_te).mean()), 4),
            "p": round(float(p_1x_base[0]), 4)},
        "goles_media": {"mae_total": round(float(np.abs(media_goles - y_tot).mean()), 4),
                        "media_predicha": round(media_goles, 4)},
    }
    return salida


def ejecutar(X: pd.DataFrame | None = None, estimadores=("ratings", "poisson_glm", "hgb_poisson"),
             n_holdout: int = 2, progress=None) -> dict:
    """Walk-forward completo. Devuelve seleccion y holdout por separado."""
    if X is None:
        X = pd.read_parquet(FEATURES_PARQUET)
    X = X.dropna(subset=["home_goals", "away_goals", "season"])
    part = particion(X, n_holdout=n_holdout)
    resultados = {"particion": part, "seleccion": {}, "holdout": {},
                  "ejecutado_en": datetime.utcnow().isoformat()}
    for bloque in ("seleccion", "holdout"):
        for est in estimadores:
            for temp in part[bloque]:
                r = _predecir_temporada(X, temp, est)
                if not r:
                    continue
                resultados[bloque].setdefault(est, []).append(r)
                if progress:
                    dc = r["familias"]["dixon_coles"]
                    progress(f"{bloque} {est} {temp}: 1X2 ll={dc['1x2']['log_loss']} "
                             f"BTTS ll={dc['btts']['log_loss']} O2.5 ll={dc['total_2.5']['log_loss']}")
    SOCCER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert_writable(SOCCER_OUT_DIR, "SOCCER")
    (SOCCER_OUT_DIR / "walkforward.json").write_text(
        json.dumps(resultados, indent=1, default=str), encoding="utf-8")
    return resultados


def resumen(res: dict, bloque: str = "seleccion") -> pd.DataFrame:
    """Promedio ponderado por n_test de cada combinacion estimador x familia."""
    filas = []
    for est, temporadas in res.get(bloque, {}).items():
        for fam in FAMILIAS:
            n = sum(t["n_test"] for t in temporadas)
            if not n:
                continue
            def w(clave, sub):
                return sum(t["familias"][fam][clave][sub] * t["n_test"] for t in temporadas) / n
            filas.append({
                "estimador": est, "familia": fam, "n": n,
                "1x2_ll": round(w("1x2", "log_loss"), 5),
                "1x2_brier": round(w("1x2", "brier"), 5),
                "1x2_acc": round(w("1x2", "accuracy"), 4),
                "btts_ll": round(w("btts", "log_loss"), 5),
                "btts_acc": round(w("btts", "accuracy"), 4),
                "o25_ll": round(w("total_2.5", "log_loss"), 5),
                "o25_acc": round(w("total_2.5", "accuracy"), 4),
                "mae_total": round(w("goles", "mae_total"), 4),
            })
    return pd.DataFrame(filas).sort_values("1x2_ll")
