"""Walk-forward del mercado de TARJETAS. Mismas reglas que goles y corners.

Compara 2 caminos (dos lambdas por equipo vs total directo) x 2 familias
(Poisson, binomial negativa) x 2 algoritmos (GLM, gradient boosting), contra
baselines honestos: siempre over, siempre under y la frecuencia historica.

Solo se evaluan partidos que TIENEN amarillas reales. Las competiciones sin la
fuente (Champions, Liga MX, Saudi) no aparecen aqui y su mercado de tarjetas
queda INSUFFICIENT DATA.

La linea principal es 4.5. No es una eleccion estetica: la media real del total
es 4.06, asi que 4.5 es la linea que parte la distribucion mas cerca del 50/50
y por tanto la mas exigente para el modelo. Elegir una linea muy alta o muy
baja haria subir el acierto sin que el modelo aporte nada, porque bastaria con
apostar siempre al mismo lado.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from SOCCER.evaluation.walkforward import (brier_bin, calibracion, ece, log_loss_bin,
                                           temporadas_ordenadas)
from SOCCER.features.build import FEATURES_PARQUET
from SOCCER.models.cards import (EstimadorTarjetas, LINEAS, dist_total, esperado,
                                 over_under)
from shared.paths import SOCCER_OUT_DIR, assert_writable

LINEA_PRINCIPAL = 4.5
# El GLM se probo y se retiro: no converge en 800 iteraciones sobre estas 144
# features sin escalar (ConvergenceWarning en las 8 temporadas de seleccion). Un
# modelo que no converge no compite en igualdad, asi que no se puntua. Queda
# anotado aqui para que la retirada sea explicita y no un olvido.
CONFIGS = [("hgb", "equipos", "poisson"), ("hgb", "equipos", "nbinom"),
           ("hgb", "total", "nbinom"), ("hgb", "total", "poisson")]


def con_tarjetas(X: pd.DataFrame) -> pd.DataFrame:
    return X[X["total_yellow"].notna()].copy()


def _temporada(X: pd.DataFrame, temp: str) -> dict:
    ts = temporadas_ordenadas(X)
    tr = X[X["season"].isin([t for t in ts if t < temp])]
    te = X[X["season"] == temp]
    if len(tr) < 800 or te.empty:
        return {}
    y_tot = te["total_yellow"].to_numpy(float)
    y_over = (y_tot > LINEA_PRINCIPAL).astype(int)
    out = {"temporada": temp, "n_train": int(len(tr)), "n_test": int(len(te)),
           "media_real": round(float(y_tot.mean()), 3),
           "varianza_real": round(float(y_tot.var()), 3), "configs": {}}

    for algo, objetivo, familia in CONFIGS:
        est = EstimadorTarjetas(algo, objetivo).fit(
            tr, tr["home_yellow"], tr["away_yellow"], tr["total_yellow"])
        lh, la = est.predict(te)
        D = dist_total(lh, la, familia, est.phi,
                       "total" if objetivo == "total" else "suma")
        p_over, _, _ = over_under(D, LINEA_PRINCIPAL)
        eg = esperado(D)
        cal = calibracion(y_over, p_over)
        clave = f"{algo}_{objetivo}_{familia}"
        out["configs"][clave] = {
            "log_loss": round(log_loss_bin(y_over, p_over), 5),
            "brier": round(brier_bin(y_over, p_over), 5),
            "accuracy": round(float(((p_over >= .5).astype(int) == y_over).mean()), 4),
            "ece": ece(cal), "calibracion": cal,
            "mae": round(float(np.abs(eg - y_tot).mean()), 4),
            "rmse": round(float(np.sqrt(((eg - y_tot) ** 2).mean())), 4),
            "media_predicha": round(float(eg.mean()), 3),
            "phi": round(float(est.phi), 3),
            "por_linea": {f"{ln:g}": round(log_loss_bin((y_tot > ln).astype(int),
                                                        over_under(D, ln)[0]), 5)
                          for ln in LINEAS},
        }

    base_freq = float((tr["total_yellow"] > LINEA_PRINCIPAL).mean())
    p_base = np.full(len(te), base_freq)
    media_tr = float(tr["total_yellow"].mean())
    out["baselines"] = {
        "frecuencia_historica": {
            "log_loss": round(log_loss_bin(y_over, p_base), 5),
            "brier": round(brier_bin(y_over, p_base), 5),
            "accuracy": round(float(((p_base >= .5).astype(int) == y_over).mean()), 4),
            "p": round(base_freq, 4)},
        "siempre_over": {"accuracy": round(float(y_over.mean()), 4)},
        "siempre_under": {"accuracy": round(float(1 - y_over.mean()), 4)},
        "media_historica": {"mae": round(float(np.abs(media_tr - y_tot).mean()), 4),
                            "media": round(media_tr, 3)},
    }
    return out


def ejecutar(X: pd.DataFrame | None = None, n_holdout: int = 2, progress=None) -> dict:
    if X is None:
        X = pd.read_parquet(FEATURES_PARQUET)
    X = con_tarjetas(X)
    ts = temporadas_ordenadas(X)
    holdout, seleccion = ts[-n_holdout:], ts[4:-n_holdout]
    res = {"particion": {"todas": ts, "seleccion": seleccion, "holdout": holdout},
           "partidos_con_tarjetas": int(len(X)), "seleccion": {}, "holdout": {},
           "linea_principal": LINEA_PRINCIPAL,
           "ejecutado_en": datetime.utcnow().isoformat()}
    for bloque, temps in (("seleccion", seleccion), ("holdout", holdout)):
        for t in temps:
            r = _temporada(X, t)
            if not r:
                continue
            res[bloque][t] = r
            if progress:
                mejor = min(r["configs"].items(), key=lambda kv: kv[1]["log_loss"])
                progress(f"{bloque} {t}: n={r['n_test']} mejor={mejor[0]} "
                         f"ll={mejor[1]['log_loss']} "
                         f"base={r['baselines']['frecuencia_historica']['log_loss']}")
    SOCCER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert_writable(SOCCER_OUT_DIR, "SOCCER")
    (SOCCER_OUT_DIR / "walkforward_cards.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


def resumen(res: dict, bloque: str = "seleccion") -> pd.DataFrame:
    temps = res.get(bloque, {})
    n = sum(t["n_test"] for t in temps.values())
    if not n:
        return pd.DataFrame()
    filas = []
    claves = next(iter(temps.values()))["configs"].keys()
    for c in claves:
        def w(k, c=c):
            return sum(t["configs"][c][k] * t["n_test"] for t in temps.values()) / n
        filas.append({"config": c, "n": n, "log_loss": round(w("log_loss"), 5),
                      "brier": round(w("brier"), 5), "accuracy": round(w("accuracy"), 4),
                      "mae": round(w("mae"), 4)})
    b = sum(t["baselines"]["frecuencia_historica"]["log_loss"] * t["n_test"]
            for t in temps.values()) / n
    filas.append({"config": "BASELINE frecuencia", "n": n, "log_loss": round(b, 5),
                  "brier": None,
                  "accuracy": round(sum(t["baselines"]["frecuencia_historica"]["accuracy"]
                                        * t["n_test"] for t in temps.values()) / n, 4),
                  "mae": round(sum(t["baselines"]["media_historica"]["mae"] * t["n_test"]
                                   for t in temps.values()) / n, 4)})
    return pd.DataFrame(filas).sort_values("log_loss")
