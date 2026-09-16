"""Gating de los mercados secundarios. Version CHAT_GATING_v1.

Decide si un mercado secundario puede publicar probabilidad, leyendo los
walk-forward que hay en disco. No hay ningun estado escrito a mano: si se
reentrena y los numeros cambian, el estado cambia solo.

La regla, y es la misma para todos:

    el modelo tiene que ganarle a su BASELINE en el bloque de SELECCION.

El bloque de seleccion, no el holdout. Elegir mirando el holdout es exactamente
el error que el holdout existe para impedir: se acaba escogiendo la variante
que tuvo suerte en la ultima temporada. El holdout se reporta, no decide.

MARGEN_MINIMO existe porque una diferencia en el cuarto decimal de log loss
sobre 7 000 partidos no distingue un modelo mejor de un modelo con suerte. Por
debajo de ese margen se considera empate, y en un empate gana el baseline: es
mas simple y no hay que mantenerlo.
"""
from __future__ import annotations

import json

from shared.paths import MLB_BACKTEST_DIR, SOCCER_OUT_DIR

VERSION = "CHAT_GATING_v1"
MARGEN_MINIMO = 0.002        # log loss. Por debajo de esto es empate.


def _leer(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (ValueError, OSError):
        return None


def _pond(temps: dict, extraer) -> float | None:
    """Media ponderada por numero de partidos de test."""
    vals = []
    for t in (temps or {}).values():
        v, n = extraer(t), t.get("n_test")
        if v is not None and n:
            vals.append((v, n))
    return sum(v * n for v, n in vals) / sum(n for _, n in vals) if vals else None


def _decidir(nombre, seleccion, holdout, mejor_config) -> dict:
    """Compara modelo y baseline en seleccion; reporta el holdout."""
    if seleccion is None or seleccion["baseline"] is None or seleccion["modelo"] is None:
        return {"estado": "insufficient_data", "version": VERSION, "mercado": nombre,
                "motivo": "no hay walk-forward en disco para este mercado."}
    ganancia = seleccion["baseline"] - seleccion["modelo"]
    ok = ganancia > MARGEN_MINIMO
    r = {
        "version": VERSION, "mercado": nombre,
        "estado": "projection" if ok else "no_pick",
        "permite_publicar_probabilidad": bool(ok),
        "config_elegida": mejor_config,
        "seleccion": {"log_loss_modelo": round(seleccion["modelo"], 5),
                      "log_loss_baseline": round(seleccion["baseline"], 5),
                      "ganancia": round(ganancia, 5),
                      "margen_exigido": MARGEN_MINIMO},
        "holdout": ({"log_loss_modelo": round(holdout["modelo"], 5),
                     "log_loss_baseline": round(holdout["baseline"], 5),
                     "ganancia": round(holdout["baseline"] - holdout["modelo"], 5)}
                    if holdout and holdout["modelo"] is not None else None),
    }
    if ok:
        r["motivo"] = (f"el modelo le gana al baseline en el bloque de seleccion por "
                       f"{ganancia:.5f} de log loss. Se publica como PROJECTION: sin "
                       f"cuota historica no se puede demostrar ventaja economica.")
    else:
        r["motivo"] = (f"el modelo NO le gana al baseline en el bloque de seleccion "
                       f"(modelo {seleccion['modelo']:.5f} frente a baseline "
                       f"{seleccion['baseline']:.5f}, diferencia {ganancia:+.5f}). "
                       f"No se publica ninguna probabilidad de este mercado.")
    return r


def tarjetas() -> dict:
    wf = _leer(SOCCER_OUT_DIR / "walkforward_cards.json")
    if wf is None:
        return {"estado": "insufficient_data", "version": VERSION, "mercado": "soccer.cards",
                "motivo": "no existe walkforward_cards.json; ejecuta `spc soccer-cards`."}
    from SOCCER.markets.train import CONFIG_TARJETAS
    cfg = f"{CONFIG_TARJETAS['algoritmo']}_{CONFIG_TARJETAS['objetivo']}_{CONFIG_TARJETAS['familia']}"

    def bloque(b):
        t = wf.get(b) or {}
        return {"modelo": _pond(t, lambda x: (x["configs"].get(cfg) or {}).get("log_loss")),
                "baseline": _pond(t, lambda x: x["baselines"]["frecuencia_historica"]["log_loss"])}
    return _decidir("soccer.cards", bloque("seleccion"), bloque("holdout"), cfg)


def carreras(tramo: str = "completo") -> dict:
    wf = _leer(MLB_BACKTEST_DIR / "walkforward_runs.json")
    mid = "mlb.f5_total" if tramo == "f5" else "mlb.runs_dist"
    if wf is None or tramo not in wf:
        return {"estado": "insufficient_data", "version": VERSION, "mercado": mid,
                "motivo": "no existe walkforward_runs.json; ejecuta `spc mlb-runs`."}

    def bloque(b, cfg):
        t = (wf[tramo].get(b) or {})
        return {"modelo": _pond(t, lambda x: (x["configs"].get(cfg) or {}).get("log_loss_linea")),
                "baseline": _pond(
                    t, lambda x: x["baselines"]["frecuencia_historica"]["log_loss_linea"])}

    # La configuracion se elige en SELECCION, entre todas las probadas.
    sel = wf[tramo].get("seleccion") or {}
    cfgs = sorted({c for t in sel.values() for c in t["configs"]})
    puntuadas = [(bloque("seleccion", c)["modelo"], c) for c in cfgs]
    puntuadas = [(v, c) for v, c in puntuadas if v is not None]
    if not puntuadas:
        return {"estado": "insufficient_data", "version": VERSION, "mercado": mid,
                "motivo": "el walk-forward no tiene configuraciones evaluables."}
    mejor = min(puntuadas)[1]
    return _decidir(mid, bloque("seleccion", mejor), bloque("holdout", mejor), mejor)


def todos() -> dict:
    return {"version": VERSION,
            "soccer.cards": tarjetas(),
            "mlb.runs_dist": carreras("completo"),
            "mlb.f5_total": carreras("f5")}


def aplicar_al_registro() -> dict:
    """Escribe el estado medido en el registro de mercados."""
    from CHAT import registry
    g = todos()
    cambios = {}
    for mid, r in g.items():
        if mid == "version":
            continue
        d = registry.get(mid)
        if d and d["estado"] != r["estado"]:
            cambios[mid] = {"antes": d["estado"], "ahora": r["estado"],
                            "motivo": r["motivo"]}
        if d:
            d["estado"] = r["estado"]
            d["evidencia"] = {**(d.get("evidencia") or {}), "gating": r}
    return {"version": VERSION, "cambios": cambios, "gating": g}
