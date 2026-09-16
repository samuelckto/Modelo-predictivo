"""Incertidumbre de una prediccion MLB. Sustituye al "Upset Risk" compuesto.

POR QUE ESTE MODULO EXISTE
--------------------------
Se construyo primero un Upset Risk al estilo del sistema NFL: un indice 0-100 que
mezclaba cercania al 50 %, dispersion entre algoritmos, distancia al Elo, datos
faltantes y desacuerdo con el mercado. Se valido fuera de muestra (entrenamiento
2021-2024, prueba 2025-2026, n=4.631) y FALLO:

    tercil de riesgo    n      tasa de error
    bajo              1544        43.65 %
    medio             1543        44.72 %
    alto              1544        44.43 %

Es decir: declarar "riesgo alto" no identificaba predicciones mas equivocadas.
Correlacion riesgo-error: 0.008 (practicamente cero).

En cambio, la propia probabilidad SI separa:

    tercil de probabilidad   prob. media   tasa de error
    parejo (~51.9 %)                        46.18 %
    medio  (~55.9 %)                        46.34 %
    claro  (~62.7 %)                        40.28 %

CONCLUSION APLICADA: no se publica un indice compuesto inventado. Se publica
(1) la probabilidad de error que el propio modelo estima, que esta calibrada
(ECE 0.02 en el backtest), y (2) los avisos de incertidumbre como HECHOS
separados, sin puntuarlos ni mezclarlos en un numero.
"""
from __future__ import annotations

VALIDATION = {
    "composite_index": {"validated": False, "n": 4631, "test_seasons": [2025, 2026],
                        "error_by_tercile": [0.4365, 0.4472, 0.4443],
                        "correlation_with_error": 0.008,
                        "decision": "descartado: no separa los fallos"},
    "probability": {"validated": True, "n": 4631,
                    "error_by_tercile": [0.4618, 0.4634, 0.4028],
                    "decision": "se publica: la probabilidad calibrada si informa"},
}


def uncertainty(p_selection: float) -> float:
    """Probabilidad de fallar, 0-100. Es 1 - p, nada mas: honesto y calibrado."""
    return float(round((1.0 - float(p_selection)) * 100, 2))


def band(p_selection: float) -> str:
    """Bandas fijadas por la distribucion real de MLB, no por gusto.

    En MLB casi todo cae entre 50 % y 76 %; usar los cortes de la NFL (65/40)
    dejaria todo en la misma banda. Los cortes son los terciles observados.
    """
    p = float(p_selection)
    if p >= 0.62:
        return "LOW RISK"
    if p >= 0.545:
        return "MEDIUM RISK"
    return "HIGH RISK"


def flags(dispersion: float = 0.0, p_elo: float | None = None, p_selection: float = 0.5,
          missing: list | None = None, lineup_confirmed: bool = True,
          starter_known: bool = True, p_market: float | None = None) -> list[dict]:
    """Avisos factuales. Cada uno dice que se observo, sin puntuarlo."""
    out = []
    missing = missing or []
    if p_selection < 0.545:
        out.append({"code": "COIN_FLIP", "level": "warn",
                    "text": f"partido muy parejo: el pick solo tiene {p_selection*100:.0f}%, casi un volado"})
    if dispersion and dispersion > 0.04:
        out.append({"code": "MODEL_DISAGREEMENT", "level": "warn",
                    "text": "los distintos algoritmos no se ponen de acuerdo"})
    if p_elo is not None and abs(p_selection - p_elo) > 0.10:
        out.append({"code": "ELO_GAP", "level": "info",
                    "text": f"el rating de equipos da {p_elo*100:.0f}% y el modelo {p_selection*100:.0f}%: no coinciden"})
    if p_market is not None and abs(p_selection - p_market) > 0.06:
        code = "MODEL_OVER_MARKET" if p_selection > p_market else "MARKET_OVER_MODEL"
        out.append({"code": code, "level": "info",
                    "text": f"el modelo dice {p_selection*100:.0f}% y las casas de apuestas {p_market*100:.0f}%"})
    if not starter_known:
        out.append({"code": "PITCHER_UNKNOWN", "level": "warn",
                    "text": "todavia no se anuncio quien abre el partido"})
    if not lineup_confirmed:
        out.append({"code": "LINEUP_UNCONFIRMED", "level": "warn",
                    "text": "la alineacion aun no esta confirmada: la prediccion puede cambiar"})
    for m in missing:
        if m not in [f["text"] for f in out]:
            out.append({"code": "DATA_INCOMPLETE", "level": "warn", "text": m})
    return out
