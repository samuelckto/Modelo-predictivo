"""Que se publica y como. PICK / PROJECTION / NO PICK / INSUFFICIENT DATA / BLOCKED.

El estado NO se decide a mano ni con un umbral inventado: sale de comparar el
walk-forward contra el baseline del propio mercado, liga por liga.

Reglas, en orden:

  INSUFFICIENT DATA  la liga no tiene datos para ese mercado (Saudi en todo;
                     Liga MX en corners, porque no hay fuente de corners).
  BLOCKED            hay datos, pero el modelo NO supera al baseline OOS.
                     No se publica probabilidad: publicarla seria fingir señal.
  PROJECTION         supera al baseline, pero sin historico de cuotas no se puede
                     demostrar ventaja frente al mercado. Se muestra, no se apuesta.
  PICK               ademas de superar al baseline, el bucket de confianza esta
                     validado: en el holdout, ese rango acerto de forma
                     significativa (z >= 1.96 contra el baseline del bucket).
  NO PICK            el mercado permite picks, pero ESTE partido no llega al
                     umbral validado.

Mientras no haya historico de cuotas propias, ningun mercado puede pasar de
PROJECTION. Esta escrito aqui a proposito, no es un olvido.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

from shared.paths import SOCCER_OUT_DIR

MERCADOS = ("double_chance", "btts", "total_goals")
LABEL = {"double_chance": "Doble oportunidad", "btts": "Ambos marcan",
         "total_goals": "Total de goles", "corners": "Córners"}
ORDEN = ("double_chance", "total_goals", "btts")

# Un mercado necesita al menos esta ventaja en log loss sobre su baseline para
# considerarse util. Es pequena a proposito: la exigencia real la pone el
# significado estadistico, no el tamano del numero.
MEJORA_MINIMA = 0.002
MIN_PARTIDOS = 1200


def _z(aciertos: int, n: int, p0: float) -> float:
    if n <= 0 or not (0 < p0 < 1):
        return 0.0
    se = math.sqrt(p0 * (1 - p0) / n)
    return (aciertos / n - p0) / se if se > 0 else 0.0


def evaluar_mercado(nombre: str, ll_modelo: float | None, ll_base: float | None,
                    acc_modelo: float | None, acc_base: float | None, n: int,
                    hay_cuotas_historicas: bool = False) -> dict:
    """Decide el estado de un mercado a partir de sus numeros OOS."""
    if ll_modelo is None or n < MIN_PARTIDOS:
        return {"estado": "insufficient_data", "motivo":
                f"sin evaluacion suficiente ({n} partidos, minimo {MIN_PARTIDOS})",
                "publica_probabilidad": False}
    mejora = (ll_base - ll_modelo) if ll_base is not None else None
    if mejora is None or mejora < MEJORA_MINIMA:
        return {"estado": "blocked", "publica_probabilidad": False,
                "log_loss": ll_modelo, "baseline": ll_base, "mejora": mejora,
                "motivo": ("no supera al baseline en las pruebas fuera de muestra "
                           f"({ll_modelo:.5f} vs {ll_base:.5f})" if ll_base else
                           "sin baseline comparable")}
    z = _z(round((acc_modelo or 0) * n), n, acc_base or 0.5)
    base = {"log_loss": ll_modelo, "baseline": ll_base, "mejora": round(mejora, 5),
            "accuracy": acc_modelo, "accuracy_baseline": acc_base, "z": round(z, 2),
            "n": n, "publica_probabilidad": True}
    if not hay_cuotas_historicas:
        return {**base, "estado": "projection", "permite_pick": False,
                "motivo": ("supera al baseline, pero sin historico de cuotas no se puede "
                           "demostrar ventaja frente al mercado: se publica como proyeccion")}
    if z >= 1.96:
        return {**base, "estado": "pick_habilitado", "permite_pick": True,
                "motivo": f"supera al baseline con significancia (z={z:.2f})"}
    return {**base, "estado": "projection", "permite_pick": False,
            "motivo": f"mejora presente pero sin significancia (z={z:.2f} < 1.96)"}


def desde_walkforward(ruta_goles=None, ruta_corners=None,
                      hay_cuotas_historicas: bool = False) -> dict:
    """Construye el gating leyendo los JSON de walk-forward ya generados."""
    ruta_goles = ruta_goles or (SOCCER_OUT_DIR / "walkforward.json")
    ruta_corners = ruta_corners or (SOCCER_OUT_DIR / "walkforward_corners.json")
    out = {"generado_en": datetime.utcnow().isoformat(), "mercados": {},
           "hay_cuotas_historicas": hay_cuotas_historicas}

    if ruta_corners.exists():
        d = json.loads(ruta_corners.read_text(encoding="utf-8"))
        hold = d.get("holdout", {})
        n = sum(t["n_test"] for t in hold.values())
        if n:
            def w(f):
                return sum(f(t) * t["n_test"] for t in hold.values()) / n
            claves = list(next(iter(hold.values()))["configs"])
            mejor = min(claves, key=lambda c: w(lambda t, c=c: t["configs"][c]["log_loss"]))
            out["mercados"]["corners"] = {
                **evaluar_mercado("corners",
                                  w(lambda t: t["configs"][mejor]["log_loss"]),
                                  w(lambda t: t["baselines"]["frecuencia_historica"]["log_loss"]),
                                  w(lambda t: t["configs"][mejor]["accuracy"]),
                                  w(lambda t: t["baselines"]["frecuencia_historica"]["accuracy"]),
                                  n, hay_cuotas_historicas),
                "config": mejor, "bloque": "holdout"}

    if ruta_goles.exists():
        d = json.loads(ruta_goles.read_text(encoding="utf-8"))
        for bloque in ("holdout", "seleccion"):
            datos = d.get(bloque) or {}
            temporadas = [t for lista in datos.values() for t in lista] if isinstance(
                datos, dict) and datos and isinstance(next(iter(datos.values())), list) else []
            if not temporadas:
                continue
            n = sum(t["n_test"] for t in temporadas)
            def w(f, temporadas=temporadas, n=n):
                return sum(f(t) * t["n_test"] for t in temporadas) / n
            fam = "dixon_coles"
            out["mercados"]["double_chance"] = {
                **evaluar_mercado("double_chance",
                                  w(lambda t: t["familias"][fam]["double_chance"]["log_loss"]),
                                  w(lambda t: t["baselines"]["double_chance_frecuencia"]["log_loss"]),
                                  w(lambda t: t["familias"][fam]["double_chance"]["accuracy"]),
                                  w(lambda t: t["baselines"]["double_chance_frecuencia"]["accuracy"]),
                                  n, hay_cuotas_historicas), "bloque": bloque}
            out["mercados"]["btts"] = {
                **evaluar_mercado("btts",
                                  w(lambda t: t["familias"][fam]["btts"]["log_loss"]),
                                  w(lambda t: t["baselines"]["btts_frecuencia"]["log_loss"]),
                                  w(lambda t: t["familias"][fam]["btts"]["accuracy"]),
                                  w(lambda t: t["baselines"]["btts_frecuencia"]["accuracy"]),
                                  n, hay_cuotas_historicas), "bloque": bloque}
            out["mercados"]["total_goals"] = {
                **evaluar_mercado("total_goals",
                                  w(lambda t: t["familias"][fam]["total_2.5"]["log_loss"]),
                                  w(lambda t: t["baselines"]["total_frecuencia"]["log_loss"]),
                                  w(lambda t: t["familias"][fam]["total_2.5"]["accuracy"]),
                                  w(lambda t: t["baselines"]["total_frecuencia"]["accuracy"]),
                                  n, hay_cuotas_historicas), "bloque": bloque}
            break
    return out


def estado_por_liga(gating: dict, cobertura_corners: dict, estado_ligas: dict) -> dict:
    """Cruza el gating global con lo que cada liga tiene de verdad."""
    out = {}
    for code, info in estado_ligas.items():
        cob = (cobertura_corners.get(code) or {})
        fila = {}
        for m in MERCADOS:
            g = dict(gating.get("mercados", {}).get(m, {"estado": "insufficient_data",
                                                        "motivo": "sin evaluacion"}))
            # Dos situaciones MUY distintas que antes se confundian:
            #
            #   sin fuente     -> INSUFFICIENT DATA. No hay dato, no hay numero.
            #   poca historia  -> NO PICK. El modelo SI calcula una probabilidad
            #                     con lo que hay; simplemente no se apuesta.
            #
            # Tratar la segunda como la primera borraba de la pantalla ligas que
            # si tenian prediccion, como la Saudi.
            if m == "corners" and cob.get("con_corners", 0) < MIN_PARTIDOS:
                g = {"estado": "insufficient_data",
                     "motivo": (f"sin fuente de corners para esta competicion "
                                f"({cob.get('con_corners', 0)} partidos con corners)"),
                     "publica_probabilidad": False}
            elif info.get("partidos", 0) < MIN_PARTIDOS:
                g = {**g, "estado": "no_pick", "permite_pick": False,
                     "publica_probabilidad": True,
                     "motivo": (f"la competicion solo tiene {info.get('partidos', 0)} partidos "
                                f"historicos: se publica la probabilidad, pero no como pick")}
            fila[m] = g
        out[code] = fila
    return out
