"""Walk-forward de la distribucion de carreras de MLB (completo y F5).

Se entrena con las temporadas anteriores y se evalua en la siguiente, nunca al
azar. Se comparan 2 caminos (dos lambdas por equipo vs total directo) x 2
familias (Poisson, binomial negativa) contra baselines honestos.

Los baselines importan mas que el modelo. Si el modelo no le gana a "la
frecuencia historica de esta linea", entonces no aporta informacion y el
mercado debe quedarse en NO PICK por mucho que las probabilidades se vean
bonitas.

Se mide con log loss sobre la linea principal (la mas cercana a la mediana, que
es la mas dificil) y ademas con el LOG LOSS DEL CONTEO COMPLETO: cuanta
probabilidad asigno el modelo al numero exacto de carreras que ocurrio. Esa
segunda metrica es la que importa para el chat, porque el chat no pregunta solo
por una linea: pregunta P(0), P(1), P(2+)...
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from MLB.engine.distributions import (EstimadorCarreras, LINEAS, OBJETIVO, dist_total,
                                      esperado, over_under, pmf_equipo)
from shared.paths import MLB_BACKTEST_DIR, MLB_PROCESSED_DIR, assert_writable

FEATURES = MLB_PROCESSED_DIR / "features.parquet"
LINEA_PRINCIPAL = {"completo": 8.5, "f5": 4.5}
CONFIGS = [("equipos", "nbinom"), ("equipos", "poisson"),
           ("total", "nbinom"), ("total", "poisson")]


def _ll_bin(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ll_conteo(D: np.ndarray, y: np.ndarray) -> float:
    """Log loss del conteo exacto: -log P(k real). Mide la distribucion entera."""
    k = np.clip(np.asarray(y, int), 0, D.shape[1] - 1)
    p = np.clip(D[np.arange(len(k)), k], 1e-9, 1.0)
    return float(-np.mean(np.log(p)))


def _temporada(X: pd.DataFrame, temp: int, tramo: str) -> dict:
    """Una temporada de test.

    Tres bloques temporales, no dos, porque ademas del modelo se evalua un
    CALIBRADOR y ajustarlo con los mismos datos con que se mide seria trampa:

        temporadas < temp-1   ajuste del modelo
        temporada   temp-1    ajuste del calibrador (el modelo no la vio)
        temporada   temp      test (ni el modelo ni el calibrador la vieron)

    El calibrador se prueba porque el sintoma observado es exactamente el suyo:
    el modelo ACIERTA mas que el baseline pero tiene PEOR log loss, que es lo
    que ocurre cuando el lado elegido es bueno y la confianza esta inflada.
    """
    ch, ca, ct = OBJETIVO[tramo]
    D_ = X[X[ct].notna()]
    tr = D_[D_["season"] < temp]
    te = D_[D_["season"] == temp]
    tr_fit = D_[D_["season"] < temp - 1]
    tr_cal = D_[D_["season"] == temp - 1]
    if len(tr) < 1500 or te.empty:
        return {}
    linea = LINEA_PRINCIPAL[tramo]
    y_tot = te[ct].to_numpy(float)
    y_over = (y_tot > linea).astype(int)
    out = {"temporada": int(temp), "tramo": tramo, "n_train": int(len(tr)),
           "n_test": int(len(te)), "media_real": round(float(y_tot.mean()), 3),
           "varianza_real": round(float(y_tot.var()), 3), "configs": {}}

    for objetivo, familia in CONFIGS:
        est = EstimadorCarreras(tramo, objetivo).fit(tr, tr[ch], tr[ca], tr[ct])
        lh, la = est.predict(te)
        Dd = dist_total(lh, la, familia, est.phi,
                        "total" if objetivo == "total" else "suma")
        p_over, _, _ = over_under(Dd, linea)
        eg = esperado(Dd)
        out["configs"][f"{objetivo}_{familia}"] = {
            "log_loss_linea": round(_ll_bin(y_over, p_over), 5),
            "log_loss_conteo": round(_ll_conteo(Dd, y_tot), 5),
            "accuracy": round(float(((p_over >= .5).astype(int) == y_over).mean()), 4),
            "mae": round(float(np.abs(eg - y_tot).mean()), 4),
            "rmse": round(float(np.sqrt(((eg - y_tot) ** 2).mean())), 4),
            "media_predicha": round(float(eg.mean()), 3),
            "phi": round(float(est.phi), 3),
            "por_linea": {f"{ln:g}": round(_ll_bin((y_tot > ln).astype(int),
                                                   over_under(Dd, ln)[0]), 5)
                          for ln in LINEAS[tramo]},
        }
        if objetivo == "equipos":
            # P(al menos una carrera) en el tramo: la pregunta directa del chat.
            ph, pa = pmf_equipo(lh, familia, est.phi), pmf_equipo(la, familia, est.phi)
            p_alguna = 1.0 - ph[:, 0] * pa[:, 0]
            y_alguna = (y_tot > 0).astype(int)
            out["configs"][f"{objetivo}_{familia}"]["alguna_carrera"] = {
                "log_loss": round(_ll_bin(y_alguna, p_alguna), 5),
                "p_media": round(float(p_alguna.mean()), 4),
                "frecuencia_real": round(float(y_alguna.mean()), 4)}

    # --- el mismo modelo, pero con la confianza calibrada -------------------
    if len(tr_fit) >= 1500 and len(tr_cal) >= 500:
        from shared.calibrators import apply, seleccionar
        for objetivo, familia in CONFIGS:
            est = EstimadorCarreras(tramo, objetivo).fit(
                tr_fit, tr_fit[ch], tr_fit[ca], tr_fit[ct])
            lc, lac = est.predict(tr_cal)
            Dc = dist_total(lc, lac, familia, est.phi,
                            "total" if objetivo == "total" else "suma")
            p_cal_in, _, _ = over_under(Dc, linea)
            y_cal = (tr_cal[ct].to_numpy(float) > linea).astype(int)
            sel = seleccionar(p_cal_in, y_cal,
                              orden=tr_cal["game_date"].astype(str).to_numpy())
            lt, lat = est.predict(te)
            Dt = dist_total(lt, lat, familia, est.phi,
                            "total" if objetivo == "total" else "suma")
            p_raw, _, _ = over_under(Dt, linea)
            p_fin = apply(sel["calibrador"], p_raw)
            out["configs"][f"{objetivo}_{familia}_CAL"] = {
                "log_loss_linea": round(_ll_bin(y_over, p_fin), 5),
                "log_loss_linea_sin_calibrar": round(_ll_bin(y_over, p_raw), 5),
                "log_loss_conteo": None,     # el calibrador actua sobre la linea
                "accuracy": round(float(((p_fin >= .5).astype(int) == y_over).mean()), 4),
                "mae": None, "rmse": None,
                "media_predicha": round(float(p_fin.mean()), 4),
                "phi": round(float(est.phi), 3),
                "calibrador": sel["elegido"],
                "calibrador_motivo": sel["motivo"],
                "n_ajuste_calibrador": int(len(tr_cal)),
                "por_linea": {},
            }

    # --- baselines honestos ------------------------------------------------
    base_freq = float((tr[ct] > linea).mean())
    p_base = np.full(len(te), base_freq)
    media_tr = float(tr[ct].mean())
    # Distribucion empirica del train: el baseline correcto para el conteo.
    cuenta = np.bincount(tr[ct].to_numpy(int), minlength=2 * 25 + 1).astype(float)
    emp = cuenta / cuenta.sum()
    D_emp = np.tile(emp, (len(te), 1))
    out["baselines"] = {
        "frecuencia_historica": {
            "log_loss_linea": round(_ll_bin(y_over, p_base), 5),
            "accuracy": round(float(((p_base >= .5).astype(int) == y_over).mean()), 4),
            "p": round(base_freq, 4)},
        "distribucion_empirica": {"log_loss_conteo": round(_ll_conteo(D_emp, y_tot), 5)},
        "siempre_over": {"accuracy": round(float(y_over.mean()), 4)},
        "siempre_under": {"accuracy": round(float(1 - y_over.mean()), 4)},
        "media_historica": {"mae": round(float(np.abs(media_tr - y_tot).mean()), 4),
                            "media": round(media_tr, 3)},
    }
    return out


def ejecutar(X: pd.DataFrame | None = None, n_holdout: int = 2, progress=None) -> dict:
    if X is None:
        X = pd.read_parquet(FEATURES)
    temps = sorted(int(t) for t in X["season"].dropna().unique())
    holdout, seleccion = temps[-n_holdout:], temps[1:-n_holdout]
    res = {"version": "MLB_RUNS_DIST_v1",
           "particion": {"todas": temps, "seleccion": seleccion, "holdout": holdout},
           "ejecutado_en": datetime.utcnow().isoformat()}
    for tramo in ("completo", "f5"):
        res[tramo] = {"seleccion": {}, "holdout": {},
                      "linea_principal": LINEA_PRINCIPAL[tramo]}
        for bloque, ts in (("seleccion", seleccion), ("holdout", holdout)):
            for t in ts:
                r = _temporada(X, t, tramo)
                if not r:
                    continue
                res[tramo][bloque][str(t)] = r
                if progress:
                    mejor = min(r["configs"].items(),
                                key=lambda kv: kv[1]["log_loss_linea"])
                    progress(f"{tramo} {bloque} {t}: n={r['n_test']} mejor={mejor[0]} "
                             f"ll={mejor[1]['log_loss_linea']} "
                             f"base={r['baselines']['frecuencia_historica']['log_loss_linea']}")
    MLB_BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    assert_writable(MLB_BACKTEST_DIR, "MLB")
    (MLB_BACKTEST_DIR / "walkforward_runs.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


def resumen(res: dict, tramo: str = "completo", bloque: str = "seleccion") -> pd.DataFrame:
    temps = res.get(tramo, {}).get(bloque, {})
    n = sum(t["n_test"] for t in temps.values())
    if not n:
        return pd.DataFrame()
    filas = []
    # Las variantes calibradas no existen en la primera temporada evaluada (no
    # hay bloque previo con el que ajustar el calibrador), asi que cada config
    # se promedia SOBRE LAS TEMPORADAS EN QUE EXISTE. Promediar con ceros o con
    # el total de partidos falsearia la comparacion.
    claves = sorted({c for t in temps.values() for c in t["configs"]})
    for c in claves:
        presentes = [t for t in temps.values() if c in t["configs"]]

        def w(k, c=c, presentes=presentes):
            vals = [(t["configs"][c].get(k), t["n_test"]) for t in presentes]
            vals = [(v, m) for v, m in vals if v is not None]
            return (sum(v * m for v, m in vals) / sum(m for _, m in vals)
                    if vals else None)

        def r(k, d=5, c=c):
            v = w(k, c)
            return None if v is None else round(v, d)

        filas.append({"config": c, "n": sum(t["n_test"] for t in presentes),
                      "ll_linea": r("log_loss_linea"), "ll_conteo": r("log_loss_conteo"),
                      "accuracy": r("accuracy", 4), "mae": r("mae", 4)})
    filas.append({
        "config": "BASELINE frecuencia/empirica", "n": n,
        "ll_linea": round(sum(t["baselines"]["frecuencia_historica"]["log_loss_linea"]
                              * t["n_test"] for t in temps.values()) / n, 5),
        "ll_conteo": round(sum(t["baselines"]["distribucion_empirica"]["log_loss_conteo"]
                               * t["n_test"] for t in temps.values()) / n, 5),
        "accuracy": round(sum(t["baselines"]["frecuencia_historica"]["accuracy"]
                              * t["n_test"] for t in temps.values()) / n, 4),
        "mae": round(sum(t["baselines"]["media_historica"]["mae"] * t["n_test"]
                         for t in temps.values()) / n, 4)})
    return pd.DataFrame(filas).sort_values("ll_linea")
