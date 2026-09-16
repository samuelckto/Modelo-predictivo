"""Motor de discrepancia modelo vs mercado. Version DISCREPANCY_ENGINE_v1.

La regla que este modulo hace cumplir, y la razon de que exista:

    QUE EL MODELO DE MAS PROBABILIDAD QUE EL MERCADO NO ES VALOR.

Es una hipotesis. Se convierte en valor solo cuando existe un historico que
demuestre que apostar esas discrepancias gano dinero despues del margen de la
casa. Sin ese historico el estado correcto es UNVALIDATED, nunca VALUE.

El orden de decision no se puede invertir:

    datos -> features pregame -> modelo -> prob cruda -> calibracion
          -> prob final del modelo -> comparacion con mercado
          -> backtest de ESA discrepancia -> decision

El mercado entra en el penultimo paso. Si entrara antes, el modelo terminaria
copiando al casino y la "discrepancia" mediria cero por construccion.

Sobre los umbrales. Los tramos de 0-3 / 3-5 / 5-8 / 8-10 / 10-15 / 15+ puntos
porcentuales son los pedidos en la especificacion y se usan tal cual. Las
etiquetas ALIGNED / MINOR / STRONG no son un umbral inventado: se anclan al
vigorish real del mercado. Si la discrepancia es menor que el margen que cobra
la casa, no se esta discrepando con el mercado, se esta discrepando con su
comision. Cuando no se conoce el overround se usa VIG_TIPICO como cota
conservadora y se deja constancia en `vig_conocido`.
"""
from __future__ import annotations

VERSION = "DISCREPANCY_ENGINE_v1"

# Tramos de |gap| en puntos porcentuales (especificacion del usuario).
BUCKETS = ((0, 3), (3, 5), (5, 8), (8, 10), (10, 15), (15, 100))
BUCKET_LABELS = ["0-3 pp", "3-5 pp", "5-8 pp", "8-10 pp", "10-15 pp", "15+ pp"]

# Overround tipico de un mercado a dos vias cuando no se pudo medir.
VIG_TIPICO = 0.045
# A partir de aqui la discrepancia deja de explicarse por el margen.
STRONG_PP = 5.0

# Muestra minima para que un tramo pueda opinar sobre rentabilidad. Por debajo
# de esto el ROI observado no distingue ventaja de ruido.
MIN_MUESTRA_VALIDACION = 100

ESTADOS = ("ALIGNED", "MINOR DISAGREEMENT", "STRONG MODEL-MARKET DISAGREEMENT",
           "VALIDATED VALUE", "UNVALIDATED", "NO PICK", "NO MARKET")


def bucket(gap_pp) -> str | None:
    """Tramo de |gap| en puntos porcentuales."""
    if gap_pp is None:
        return None
    g = abs(float(gap_pp))
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
        if lo <= g < hi:
            return lab
    return BUCKET_LABELS[-1]


def evaluar(model_raw=None, model_cal=None, market=None, odds_decimal=None,
            overround=None, market_source=None, fetched_at=None,
            event_time=None, model_version=None, evidencia=None,
            favorito_mercado=None, lado_modelo=None) -> dict:
    """Compara modelo y mercado y devuelve el registro completo.

    `evidencia` es el resultado de `backtest_buckets` para este deporte/mercado.
    Si no se pasa, o el tramo no tiene muestra, el estado NUNCA puede ser
    VALIDATED VALUE.
    """
    p_raw = None if model_raw is None else float(model_raw)
    p_cal = float(model_cal) if model_cal is not None else p_raw
    p_mkt = None if market is None else float(market)

    r = {
        "version": VERSION,
        "MODEL_PROBABILITY_RAW": p_raw,
        "MODEL_PROBABILITY_CALIBRATED": p_cal,
        "MARKET_IMPLIED_PROBABILITY": p_mkt,
        "MARKET_ODDS": odds_decimal,
        "MODEL_MARKET_GAP": None,
        "gap_pp": None,
        "gap_relativo": None,
        "bucket": None,
        "direccion": None,
        "market_source": market_source,
        "fetched_at": str(fetched_at) if fetched_at else None,
        "event_time": str(event_time) if event_time else None,
        "model_version": model_version,
        "overround": overround,
        "vig_conocido": overround is not None,
        "contradice_favorito": None,
        "FINAL_STATUS": "NO MARKET",
        "explicacion": None,
        "evidencia": None,
    }

    if p_cal is None:
        r["FINAL_STATUS"] = "NO PICK"
        r["explicacion"] = "el modelo no produjo probabilidad para este mercado."
        return r
    if p_mkt is None:
        r["explicacion"] = ("no hay cuota registrada para este mercado, asi que no "
                            "existe probabilidad implicita con la que comparar. La "
                            "proyeccion del modelo se publica sola.")
        return r

    gap = p_cal - p_mkt
    r["MODEL_MARKET_GAP"] = round(gap, 6)
    r["gap_pp"] = round(gap * 100, 4)
    r["gap_relativo"] = round(gap / p_mkt, 6) if p_mkt > 0 else None
    r["bucket"] = bucket(r["gap_pp"])
    r["direccion"] = ("MODEL > MARKET" if gap > 0 else
                      "MODEL < MARKET" if gap < 0 else "IGUAL")
    if favorito_mercado is not None and lado_modelo is not None:
        r["contradice_favorito"] = bool(str(favorito_mercado) != str(lado_modelo))

    vig = float(overround) if overround is not None else VIG_TIPICO
    # La mitad del overround es lo que el margen desplaza cada lado del mercado.
    umbral_ruido_pp = max(vig * 100 / 2.0, 1.0)
    g = abs(r["gap_pp"])

    if g <= umbral_ruido_pp:
        r["FINAL_STATUS"] = "ALIGNED"
        r["explicacion"] = (
            f"modelo {p_cal:.1%} vs mercado {p_mkt:.1%}: {r['gap_pp']:+.1f} pp. "
            f"La diferencia no supera la mitad del margen de la casa "
            f"({umbral_ruido_pp:.1f} pp), asi que no es un desacuerdo real.")
        return r

    if g < STRONG_PP:
        r["FINAL_STATUS"] = "MINOR DISAGREEMENT"
    else:
        r["FINAL_STATUS"] = "STRONG MODEL-MARKET DISAGREEMENT"

    ev = _evidencia_del_bucket(evidencia, r["bucket"])
    r["evidencia"] = ev
    base = (f"modelo calibrado {p_cal:.1%} vs mercado {p_mkt:.1%} "
            f"({r['gap_pp']:+.1f} pp, tramo {r['bucket']}).")
    if ev is None:
        r["FINAL_STATUS"] = "UNVALIDATED"
        r["explicacion"] = (
            base + " No existe historico de discrepancias de este tipo, asi que no se "
                   "puede saber si esta diferencia se traduce en ventaja. Se publica "
                   "como desacuerdo informativo, NO como valor.")
    elif ev["n"] < MIN_MUESTRA_VALIDACION:
        r["FINAL_STATUS"] = "UNVALIDATED"
        r["explicacion"] = (
            base + f" El historico de este tramo tiene solo {ev['n']} casos "
                   f"(hacen falta {MIN_MUESTRA_VALIDACION}). Sin muestra no hay veredicto: "
                   f"desacuerdo informativo, NO valor.")
    elif ev.get("roi_ic95") and ev["roi_ic95"][0] > 0:
        r["FINAL_STATUS"] = "VALIDATED VALUE"
        r["explicacion"] = (
            base + f" Historicamente este tramo dio ROI {ev['roi']:+.1%} en {ev['n']} casos "
                   f"con intervalo 95 % {ev['roi_ic95'][0]:+.1%} a {ev['roi_ic95'][1]:+.1%}: "
                   f"el limite inferior esta por encima de cero.")
    else:
        r["FINAL_STATUS"] = "UNVALIDATED"
        ic = ev.get("roi_ic95")
        detalle = (f"intervalo 95 % {ic[0]:+.1%} a {ic[1]:+.1%}, el cero esta dentro"
                   if ic else "sin intervalo calculable")
        r["explicacion"] = (
            base + f" Historicamente este tramo dio ROI {ev['roi']:+.1%} en {ev['n']} casos "
                   f"({detalle}). No se demuestra ventaja rentable.")
    return r


def _evidencia_del_bucket(evidencia, buck):
    if not evidencia or not buck:
        return None
    filas = evidencia.get("buckets") if isinstance(evidencia, dict) else evidencia
    if not filas:
        return None
    for f in filas:
        if f.get("bucket") == buck:
            return f if f.get("n") else None
    return None


def backtest_buckets(registros) -> dict:
    """Backtest historico de discrepancias por tramo de gap.

    Cada registro necesita `gap_pp`, `result` y una cuota. Los que no traen
    cuota se cuentan aparte: sin precio no hay ROI que medir.

    Devuelve tambien el analisis que pidio la especificacion: que pasa cuando
    el modelo contradice al favorito del mercado, comparando seguir al modelo
    con seguir al mercado sobre EXACTAMENTE los mismos partidos.
    """
    from shared.calibration import brier, log_loss
    from shared.roi import liquidar, roi_ic95

    por_bucket = {b: [] for b in BUCKET_LABELS}
    contradicen, sin_cuota, total = [], 0, 0
    for reg in registros or []:
        total += 1
        b = bucket(reg.get("gap_pp"))
        if b is None or not reg.get("result"):
            continue
        if reg.get("odds_decimal") is None and reg.get("odds_american") is None:
            sin_cuota += 1
        por_bucket[b].append(reg)
        if reg.get("contradice_favorito"):
            contradicen.append(reg)

    filas = []
    for b in BUCKET_LABELS:
        regs = por_bucket[b]
        caja = liquidar(regs)
        y = [1.0 if r["result"] == "win" else 0.0 for r in regs
             if r["result"] in ("win", "loss") and r.get("probability") is not None]
        p = [float(r["probability"]) for r in regs
             if r["result"] in ("win", "loss") and r.get("probability") is not None]
        filas.append({
            "bucket": b,
            "n": caja["n_contabilizadas"],
            "n_con_resultado": len(regs),
            "sin_cuota": caja["sin_cuota"],
            "wins": caja["wins"], "losses": caja["losses"],
            "pushes": caja["pushes"], "voids": caja["voids"],
            "win_rate": caja["win_rate"],
            "win_rate_ic95": caja["win_rate_ic95"],
            "cuota_media": caja["cuota_media"],
            "implied_media": caja["implied_media"],
            "prob_modelo_media": caja["prob_modelo_media"],
            "profit_unidades": caja["profit_unidades"],
            "roi": caja["roi"],
            "yield": caja["yield_"],
            "roi_ic95": roi_ic95(caja),
            "log_loss": log_loss(y, p) if len(y) >= 2 else None,
            "brier": brier(y, p) if len(y) >= 2 else None,
            "suficiente": caja["n_contabilizadas"] >= MIN_MUESTRA_VALIDACION,
        })

    return {"version": VERSION, "n_registros": total, "sin_cuota": sin_cuota,
            "buckets": filas,
            "modelo_vs_favorito": _modelo_vs_mercado(contradicen),
            "estado": ("ok" if any(f["suficiente"] for f in filas)
                       else "insufficient_data"),
            "nota": ("Ningun tramo alcanza la muestra minima "
                     f"({MIN_MUESTRA_VALIDACION}): no se puede validar ninguna "
                     "discrepancia como valor."
                     if not any(f["suficiente"] for f in filas) else None)}


def _modelo_vs_mercado(regs) -> dict:
    """¿Que pasa cuando el modelo contradice al favorito del mercado?

    Se comparan las dos estrategias sobre los MISMOS partidos: seguir al modelo
    o seguir al favorito del mercado. Cualquier otra comparacion mezcla
    muestras distintas y no dice nada.
    """
    from shared.roi import liquidar, roi_ic95
    if not regs:
        return {"n": 0, "estado": "insufficient_data",
                "nota": "no hay ningun caso registrado en que el modelo contradiga "
                        "al favorito del mercado."}
    modelo = liquidar(regs)
    # Seguir al mercado sobre los mismos partidos: el resultado se invierte solo
    # en mercados a dos vias, y solo si se registro la cuota del otro lado.
    mercado = liquidar([
        {"result": ("loss" if r["result"] == "win" else
                    "win" if r["result"] == "loss" else r["result"]),
         "odds_decimal": r.get("odds_decimal_contrario"),
         "stake": r.get("stake", 1.0)}
        for r in regs if r.get("odds_decimal_contrario") is not None
    ])
    return {
        "n": modelo["n_contabilizadas"],
        "gano_el_modelo": modelo["wins"],
        "gano_el_mercado": modelo["losses"],
        "push_void": modelo["pushes"] + modelo["voids"],
        "roi_siguiendo_al_modelo": modelo["roi"],
        "roi_ic95_modelo": roi_ic95(modelo),
        "roi_siguiendo_al_mercado": mercado["roi"],
        "n_mercado": mercado["n_contabilizadas"],
        "estado": "ok" if modelo["n_contabilizadas"] >= MIN_MUESTRA_VALIDACION
                  else "insufficient_data",
        "nota": (None if modelo["n_contabilizadas"] >= MIN_MUESTRA_VALIDACION else
                 f"solo {modelo['n_contabilizadas']} casos contabilizados; no permite "
                 f"concluir nada sobre quien acierta mas."),
    }


def veredicto(bt: dict) -> dict:
    """Resumen final: que tramos tienen evidencia de valor y cuales no."""
    con, sin = [], []
    for f in (bt or {}).get("buckets", []):
        if not f["suficiente"]:
            sin.append({"bucket": f["bucket"], "n": f["n"],
                        "motivo": "muestra insuficiente"})
        elif f.get("roi_ic95") and f["roi_ic95"][0] > 0:
            con.append({"bucket": f["bucket"], "n": f["n"], "roi": f["roi"],
                        "ic95": f["roi_ic95"]})
        else:
            sin.append({"bucket": f["bucket"], "n": f["n"], "roi": f["roi"],
                        "motivo": "el intervalo del ROI incluye el cero"})
    return {"con_evidencia_de_valor": con, "sin_evidencia_de_valor": sin,
            "conclusion": ("NO SE HA DEMOSTRADO VENTAJA RENTABLE en ningun tramo."
                           if not con else
                           f"{len(con)} tramo(s) con ROI significativamente > 0.")}
