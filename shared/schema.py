"""Formato unico de tarjeta de partido. NFL y MLB deben producir exactamente esto.

Cualquier campo que el motor no pueda calcular con datos reales va en None y se
declara en `data_completeness.missing`. Nunca se rellena con un valor inventado.
"""
from __future__ import annotations

CARD_FIELDS = (
    "sport", "game_id", "game_date", "start_utc", "status",
    "home", "away", "home_name", "away_name", "venue",
    "market", "selection", "line",
    "model_probability", "market_probability", "ensemble_probability", "elo_probability",
    "confidence", "upset_risk", "upset_label", "upset_explanation",
    "model_market_gap", "directional_disagreement", "agreement_bucket",
    "model_dispersion", "market_signal", "blend_policy",
    "data_completeness", "sources", "model_version", "prediction_timestamp",
    "prediction_id", "version", "status_note", "result", "correct", "extra",
)


def empty_card(**kw) -> dict:
    c = {k: None for k in CARD_FIELDS}
    c["extra"] = {}
    c["sources"] = []
    c["data_completeness"] = {"complete": True, "missing": [], "notes": []}
    c.update(kw)
    return c


def norm_ts(v) -> str:
    """Normaliza un timestamp a 'YYYY-MM-DD HH:MM:SS' para poder ORDENAR mezclando
    deportes. MLB los guarda como '2026-09-13T16:10:00' y el motor NFL como
    '2026-09-13 17:00:00.000000'; ordenar esas cadenas tal cual pone la 'T'
    (0x54) despues del espacio (0x20) y desordena el calendario unificado.
    Bug detectado por la prueba `test_el_orden_del_calendario_es_cronologico`.
    """
    if v is None:
        return ""
    s = str(v).replace("T", " ")
    if "." in s:
        s = s.split(".")[0]
    return s[:19]


def risk_label(risk) -> str | None:
    if risk is None:
        return None
    r = float(risk)
    if r >= 65:
        return "HIGH RISK"
    if r >= 40:
        return "MEDIUM RISK"
    return "LOW RISK"


def validate_card(c: dict) -> list[str]:
    """Devuelve la lista de problemas. Se usa en los tests del dashboard."""
    problems = []
    for k in ("sport", "game_id", "start_utc", "home", "away"):
        if c.get(k) in (None, ""):
            problems.append(f"falta {k}")
    for k in ("model_probability", "market_probability", "ensemble_probability"):
        v = c.get(k)
        if v is not None and not (0.0 <= float(v) <= 1.0):
            problems.append(f"{k} fuera de [0,1]: {v}")
    from shared.paths import SPORTS
    if c.get("sport") not in SPORTS:
        problems.append(f"deporte invalido: {c.get('sport')}")
    return problems
