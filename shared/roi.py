"""Contabilidad de apuestas: unidades, ROI y yield. Version VALUE_TRACKER_v1.

Este modulo existe para separar dos cosas que se confunden todo el tiempo:

    RENDIMIENTO DEL MODELO    log loss, Brier, ECE, acierto.
                              Mide si las probabilidades son correctas.

    RENDIMIENTO APOSTANDO     ROI, yield, unidades.
                              Mide si con esas probabilidades se gana dinero.

No son lo mismo y ninguno implica el otro. Un modelo perfectamente calibrado
pierde dinero si siempre apuesta a cuotas peores que su probabilidad, porque el
margen de la casa (vigorish) se cobra en cada apuesta. Y un modelo mal
calibrado puede ganar por suerte en una muestra corta.

Por eso aqui NUNCA se declara ventaja. Se calculan numeros y se acompanan de
su intervalo de confianza y del tamano de muestra. La decision de si eso es
ventaja la toma `shared.discrepancy.veredicto`, que exige evidencia.

Sin cuota registrada NO hay ROI. Una apuesta sin precio no se puede contabilizar
y se cuenta aparte como `sin_cuota`; jamas se le asigna una cuota tipica.
"""
from __future__ import annotations

import math

VERSION = "VALUE_TRACKER_v1"

# Resultados contables. `void` es apuesta anulada (partido suspendido): devuelve
# el stake y NO cuenta como apuesta hecha. `push` es empate contra la linea:
# tambien devuelve el stake, pero SI ocurrio y cuenta en la muestra.
RESULTADOS = ("win", "loss", "push", "void")


def to_decimal(american=None, decimal=None) -> float | None:
    """Normaliza a cuota decimal. Devuelve None si no hay precio utilizable."""
    if decimal is not None:
        try:
            d = float(decimal)
        except (TypeError, ValueError):
            return None
        return d if d > 1.0 else None
    if american is None:
        return None
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0 or math.isnan(a):
        return None
    # Una cuota americana no existe entre -100 y +100: ese rango no representa
    # ningun precio. Aparecen valores asi en los datos (p.ej. -9.0 en un
    # handicap de tenis) y convertirlos daria una cuota decimal de 12.1, que
    # inventaria una ganancia enorme donde solo hay un dato roto.
    if abs(a) < 100:
        return None
    return 1.0 + (100.0 / abs(a) if a < 0 else a / 100.0)


def to_american(decimal) -> float | None:
    d = to_decimal(decimal=decimal)
    if d is None:
        return None
    return round((d - 1) * 100, 0) if d >= 2.0 else round(-100 / (d - 1), 0)


def profit(resultado: str, decimal=None, american=None, stake: float = 1.0):
    """Ganancia neta en unidades de una apuesta liquidada.

    win  -> stake * (cuota - 1)     (la ganancia, no el retorno bruto)
    loss -> -stake
    push -> 0                        (devuelve el stake)
    void -> 0                        (devuelve el stake, apuesta anulada)

    Devuelve None si el resultado es win/loss y no hay cuota: sin precio no se
    puede saber cuanto se gano.
    """
    r = (resultado or "").lower()
    if r not in RESULTADOS:
        return None
    if r in ("push", "void"):
        return 0.0
    d = to_decimal(american=american, decimal=decimal)
    if d is None:
        return None
    return round(stake * (d - 1.0), 6) if r == "win" else round(-float(stake), 6)


def _wilson(exitos: int, n: int, z: float = 1.96):
    """Intervalo de Wilson. Con muestras pequenas el intervalo normal da limites
    fuera de [0,1] y hace parecer significativo lo que no lo es."""
    if n <= 0:
        return None, None
    ph = exitos / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, c - h), 6), round(min(1.0, c + h), 6)


def liquidar(apuestas, stake: float = 1.0) -> dict:
    """Contabiliza una lista de apuestas.

    Cada apuesta es un dict con al menos `result`, y con `odds_decimal` o
    `odds_american`. Opcional: `stake`, `implied` (prob implicita del mercado),
    `probability` (prob del modelo).

    El denominador del ROI es el stake de las apuestas LIQUIDADAS CON PRECIO.
    Las anuladas (void) no cuentan como apuesta; los push si (ocurrieron), con
    profit 0. Se declaran los dos criterios por separado para que nadie tenga
    que adivinar cual se uso.
    """
    tot = {"version": VERSION, "n_total": 0, "n_contabilizadas": 0,
           "wins": 0, "losses": 0, "pushes": 0, "voids": 0, "sin_cuota": 0,
           "stake_total": 0.0, "profit_unidades": 0.0,
           "roi": None, "yield_": None, "cuota_media": None,
           "implied_media": None, "prob_modelo_media": None,
           "win_rate": None, "win_rate_ic95": None,
           "curva": [], "advertencia": None}

    cuotas, implied, probs, acumulado = [], [], [], 0.0
    for a in apuestas:
        tot["n_total"] += 1
        r = (a.get("result") or "").lower()
        if r not in RESULTADOS:
            continue
        s = float(a.get("stake") or stake)
        d = to_decimal(american=a.get("odds_american"), decimal=a.get("odds_decimal"))
        if r == "void":
            tot["voids"] += 1
            continue
        if r in ("win", "loss") and d is None:
            # Ocurrio y se sabe si acerto, pero sin precio no entra en la caja.
            tot["sin_cuota"] += 1
            continue
        g = profit(r, decimal=d, stake=s)
        if g is None:
            tot["sin_cuota"] += 1
            continue
        tot["n_contabilizadas"] += 1
        tot["stake_total"] += s
        tot["profit_unidades"] += g
        acumulado += g
        tot["wins"] += r == "win"
        tot["losses"] += r == "loss"
        tot["pushes"] += r == "push"
        if d:
            cuotas.append(d)
        if a.get("implied") is not None:
            implied.append(float(a["implied"]))
        if a.get("probability") is not None:
            probs.append(float(a["probability"]))
        tot["curva"].append({"n": tot["n_contabilizadas"],
                             "profit_acumulado": round(acumulado, 6),
                             "result": r})

    tot["stake_total"] = round(tot["stake_total"], 6)
    tot["profit_unidades"] = round(tot["profit_unidades"], 6)
    if tot["stake_total"] > 0:
        # ROI y yield se calculan igual sobre el volumen apostado. Se devuelven
        # los dos nombres porque en la literatura se usan indistintamente y
        # esconderlo detras de uno solo genera dudas de que se esta midiendo.
        r = tot["profit_unidades"] / tot["stake_total"]
        tot["roi"] = round(r, 6)
        tot["yield_"] = round(r, 6)
    decididas = tot["wins"] + tot["losses"]
    if decididas:
        tot["win_rate"] = round(tot["wins"] / decididas, 6)
        tot["win_rate_ic95"] = _wilson(tot["wins"], decididas)
    if cuotas:
        tot["cuota_media"] = round(sum(cuotas) / len(cuotas), 4)
    if implied:
        tot["implied_media"] = round(sum(implied) / len(implied), 6)
    if probs:
        tot["prob_modelo_media"] = round(sum(probs) / len(probs), 6)

    if tot["n_contabilizadas"] == 0:
        tot["advertencia"] = ("no hay ninguna apuesta liquidada con cuota registrada: "
                              "no se puede calcular ROI ni yield.")
    elif tot["n_contabilizadas"] < 100:
        tot["advertencia"] = (
            f"muestra de {tot['n_contabilizadas']} apuestas. Con menos de 100 el ROI "
            f"esta dominado por el azar y NO demuestra ventaja en ningun sentido.")
    return tot


def roi_ic95(tot: dict) -> tuple | None:
    """Intervalo de confianza del ROI por bootstrap normal sobre la curva.

    Sirve para lo unico importante: ver si el cero esta dentro. Si lo esta, el
    ROI observado es compatible con no tener ninguna ventaja.
    """
    curva = tot.get("curva") or []
    n = len(curva)
    if n < 2 or not tot.get("stake_total"):
        return None
    ganancias, prev = [], 0.0
    for c in curva:
        ganancias.append(c["profit_acumulado"] - prev)
        prev = c["profit_acumulado"]
    media = sum(ganancias) / n
    var = sum((g - media) ** 2 for g in ganancias) / (n - 1)
    se = math.sqrt(var / n)
    stake_medio = tot["stake_total"] / n
    if stake_medio <= 0:
        return None
    return (round((media - 1.96 * se) / stake_medio, 6),
            round((media + 1.96 * se) / stake_medio, 6))
