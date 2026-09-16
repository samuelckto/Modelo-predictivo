"""Calibradores serializables, elegidos por validacion temporal.

Version: CALIBRATION_ENGINE_v1

Un calibrador NO cambia lo que el modelo cree que va a pasar. Solo corrige
cuanto se cree a si mismo: si el modelo dice 70 % en mil partidos y solo acierta
620, el 70 % esta mal aunque el lado elegido sea el correcto.

Reglas que impone este modulo, y que existen porque son faciles de romper sin
darse cuenta:

1. TRES bloques temporales, nunca uno.

       ajuste          seleccion        holdout
    [============][=============][=============]
     se ajustan     se elige el     no se toca
     los tres       ganador         hasta el final

   El bloque de ajuste entrena cada calibrador. El de seleccion decide cual
   gana. El holdout es el unico numero que se puede publicar, porque es el
   unico que no participo en ninguna decision. Ajustar y elegir sobre el mismo
   bloque garantiza que SIEMPRE gane un calibrador, aunque no sirva de nada.

2. Se corta por TIEMPO, jamas al azar. Un split aleatorio deja partidos del
   futuro en el bloque de ajuste.

3. Si ninguno mejora al crudo por encima de MIN_GANANCIA, gana el crudo. Elegir
   el mejor de tres cuando los tres empeoran no es calibrar, es sobreajustar.

4. Todo se serializa a JSON plano (nada de pickle). El calibrador viaja dentro
   de la fila de la version del modelo y se puede leer a mano.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = "CALIBRATION_ENGINE_v1"
METODOS = ("ninguna", "platt", "isotonica", "beta")

# Ganancia minima de log loss en el bloque de SELECCION para preferir un
# calibrador al crudo. Por debajo de esto la diferencia es ruido muestral.
MIN_GANANCIA = 0.0005
# Por debajo de esto no hay muestra suficiente para calibrar nada.
MIN_MUESTRA = 200


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _ll(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --------------------------------------------------------------------------
# Ajuste de cada familia. Todas devuelven un dict serializable o None.
# --------------------------------------------------------------------------

def fit_platt(p, y) -> dict | None:
    """Regresion logistica sobre el logit. Dos parametros: pendiente y sesgo.

    Es la unica familia que puede corregir un sesgo sistematico (el modelo
    siempre alto) ademas de un exceso de confianza.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    z = _logit(p).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    return {"metodo": "platt", "a": float(m.coef_[0][0]), "b": float(m.intercept_[0])}


def fit_isotonic(p, y) -> dict | None:
    """Monotona por tramos. No supone ninguna forma, pero necesita mas datos y
    se pega a la muestra: con pocos casos memoriza en lugar de calibrar."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return None
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    xs = np.unique(np.round(np.linspace(p.min(), p.max(), 100), 6))
    return {"metodo": "isotonica",
            "x": [float(v) for v in xs],
            "y": [float(v) for v in ir.predict(xs)]}


def fit_beta(p, y) -> dict | None:
    """Calibracion beta (Kull et al.): logistica sobre [log p, -log(1-p)].

    A diferencia de Platt puede mover los dos extremos de forma independiente,
    que es justo lo que suele hacer falta cuando el modelo se pasa de confiado
    solo en un lado.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    p = _clip(p)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(X, y)
    return {"metodo": "beta", "a": float(m.coef_[0][0]), "b": float(m.coef_[0][1]),
            "c": float(m.intercept_[0])}


# --------------------------------------------------------------------------
# Aplicacion
# --------------------------------------------------------------------------

def apply(cal: dict | None, p):
    """Aplica un calibrador serializado. `None` o 'ninguna' devuelve p intacto."""
    escalar = np.isscalar(p)
    arr = _clip(np.atleast_1d(p))
    if not cal or cal.get("metodo") in (None, "ninguna"):
        out = arr
    elif cal["metodo"] == "platt":
        out = 1.0 / (1.0 + np.exp(-(cal["a"] * _logit(arr) + cal["b"])))
    elif cal["metodo"] == "isotonica":
        out = np.interp(arr, cal["x"], cal["y"])
    elif cal["metodo"] == "beta":
        z = cal["a"] * np.log(arr) + cal["b"] * (-np.log(1 - arr)) + cal["c"]
        out = 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError(f"calibrador desconocido: {cal.get('metodo')}")
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return float(out[0]) if escalar else out


# --------------------------------------------------------------------------
# Seleccion por validacion temporal
# --------------------------------------------------------------------------

def seleccionar(p, y, orden=None, frac_ajuste: float = 0.5,
                frac_seleccion: float = 0.25) -> dict:
    """Elige calibrador con tres bloques temporales disjuntos.

    `orden` es el vector por el que se ordena el tiempo (fecha, timestamp o
    indice ya ordenado). Si es None se asume que p e y YA vienen en orden
    cronologico; se avisa en el resultado porque es un supuesto fuerte.

    Devuelve siempre un dict con `calibrador` (aplicable con apply) y el
    detalle completo de por que gano, incluido el caso de que gane el crudo.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(p) | np.isnan(y))
    p, y = p[m], y[m]
    if orden is not None:
        orden = np.asarray(orden)[m]
        idx = np.argsort(orden, kind="stable")
        p, y = p[idx], y[idx]

    n = len(y)
    out = {"version": VERSION, "n": int(n), "orden_explicito": orden is not None,
           "calibrador": {"metodo": "ninguna"}, "elegido": "ninguna",
           "candidatos": [], "holdout": None, "motivo": None}

    if n < MIN_MUESTRA:
        out["motivo"] = (f"muestra insuficiente para calibrar: {n} casos, "
                         f"se requieren {MIN_MUESTRA}. Se publica la probabilidad cruda.")
        out["estado"] = "insufficient_data"
        return out
    if len(np.unique(y)) < 2:
        out["motivo"] = "el objetivo no tiene las dos clases; no se puede calibrar."
        out["estado"] = "insufficient_data"
        return out

    a = int(n * frac_ajuste)
    b = a + int(n * frac_seleccion)
    p_aj, y_aj = p[:a], y[:a]
    p_se, y_se = p[a:b], y[a:b]
    p_ho, y_ho = p[b:], y[b:]
    if min(len(y_aj), len(y_se), len(y_ho)) < 30 or len(np.unique(y_aj)) < 2:
        out["motivo"] = (f"los bloques quedan demasiado cortos "
                         f"({len(y_aj)}/{len(y_se)}/{len(y_ho)}); no se calibra.")
        out["estado"] = "insufficient_data"
        return out

    ajustados = {"ninguna": {"metodo": "ninguna"}}
    for nombre, fn in (("platt", fit_platt), ("isotonica", fit_isotonic), ("beta", fit_beta)):
        c = fn(p_aj, y_aj)
        if c:
            ajustados[nombre] = c

    ll_crudo_sel = _ll(y_se, p_se)
    for nombre, cal in ajustados.items():
        ll = _ll(y_se, apply(cal, p_se))
        out["candidatos"].append({
            "metodo": nombre, "log_loss_seleccion": round(ll, 6),
            "ganancia_vs_crudo": round(ll_crudo_sel - ll, 6),
        })
    out["candidatos"].sort(key=lambda d: d["log_loss_seleccion"])

    mejor = out["candidatos"][0]
    if mejor["metodo"] == "ninguna" or mejor["ganancia_vs_crudo"] < MIN_GANANCIA:
        elegido, cal = "ninguna", {"metodo": "ninguna"}
        out["motivo"] = (f"ningun calibrador mejora al crudo por encima de "
                         f"{MIN_GANANCIA} de log loss en el bloque de seleccion "
                         f"(mejor: {mejor['metodo']}, {mejor['ganancia_vs_crudo']:+.6f}). "
                         f"Se mantiene la probabilidad cruda.")
    else:
        elegido, cal = mejor["metodo"], ajustados[mejor["metodo"]]
        out["motivo"] = (f"{elegido} gana en el bloque de seleccion con "
                         f"{mejor['ganancia_vs_crudo']:+.6f} de log loss frente al crudo.")

    # El holdout NO participo en ajustar ni en elegir: es el unico numero publicable.
    from shared.calibration import brier, ece
    p_cal_ho = apply(cal, p_ho)
    out["calibrador"] = cal
    out["elegido"] = elegido
    out["estado"] = "ok"
    out["bloques"] = {"ajuste": int(len(y_aj)), "seleccion": int(len(y_se)),
                      "holdout": int(len(y_ho))}
    out["holdout"] = {
        "n": int(len(y_ho)),
        "crudo": {"log_loss": round(_ll(y_ho, p_ho), 6),
                  "brier": round(brier(y_ho, p_ho), 6),
                  "ece": round(ece(y_ho, p_ho), 6)},
        "calibrado": {"log_loss": round(_ll(y_ho, p_cal_ho), 6),
                      "brier": round(brier(y_ho, p_cal_ho), 6),
                      "ece": round(ece(y_ho, p_cal_ho), 6)},
    }
    out["holdout"]["mejora_log_loss"] = round(
        out["holdout"]["crudo"]["log_loss"] - out["holdout"]["calibrado"]["log_loss"], 6)
    out["holdout"]["mejora_ece"] = round(
        out["holdout"]["crudo"]["ece"] - out["holdout"]["calibrado"]["ece"], 6)
    return out


def comparar(y, p_raw, p_cal, bins: int = 10) -> dict:
    """Tabla cruda vs calibrada lista para el dashboard."""
    from shared.calibration import summary
    return {"crudo": summary(y, p_raw, bins), "calibrado": summary(y, p_cal, bins)}
