"""Que mercados publica el sistema y cuales NO. Version posterior a la auditoria.

CAMBIO IMPORTANTE (auditoria §3-§7): la decision ya NO se toma con el backtest
que usaba las features del abridor.

La auditoria demostro con una prueba de truncamiento partido a partido que las
116 features del abridor dependen de saber QUIEN lanzo realmente, dato que en un
partido historico no existe antes de jugarlo (las otras 243 features salieron
perfectamente limpias: 0 diferencias). Ademas el sistema NO tiene evidencia sobre
la fiabilidad del abridor anunciado, porque los anuncios historicos se ingirieron
despues de los partidos (0 anuncios con timestamp anterior).

Por eso se gatea con la COTA LIMPIA: el backtest sin ninguna feature del abridor.
Es conservador a proposito. Cuando el sistema acumule anuncios reales anteriores
a los partidos podra medirse la fiabilidad y revisarse esta decision.

Fuente de los numeros: audit/parts/*_sin_abridor.json, producidos por
`python -m audit.models_compare`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from shared.paths import MLB_BACKTEST_DIR, ROOT

PARTS = ROOT / "audit" / "parts"
VARIANT_GATE = "sin_abridor"       # la cota que se puede demostrar
MIN_LOGLOSS_GAIN = 0.002
Z_SIGNIF = 1.645                   # una cola, 95 %
MAX_ECE = 0.05


def _naive_ll(p: float) -> float:
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def _load(market: str, variant: str) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(PARTS.glob(f"[0-9][0-9][0-9][0-9]_{market}_{variant}.json"))]


def _agg(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    base = float(np.average([r["base_rate"] for r in rows],
                            weights=[r["n"] for r in rows]))
    out = {"n": n, "base_rate": base, "seasons": [r["season"] for r in rows],
           "strategies": {}}
    for k in rows[0]["strategies"]:
        w = np.array([r["n"] for r in rows], dtype=float)
        g = lambda f: float(np.average([r["strategies"][k][f] for r in rows], weights=w))
        out["strategies"][k] = {"accuracy": g("accuracy"), "log_loss": g("log_loss"),
                                "brier": g("brier"), "ece": g("ece")}
    return out


def _z(acc: float, major: float, n: int) -> float:
    se = math.sqrt(major * (1 - major) / n)
    return (acc - major) / se if se else 0.0


def decide() -> dict:
    out = {}
    for market in ("moneyline", "f5_moneyline", "run_line"):
        clean = _agg(_load(market, VARIANT_GATE))
        dirty = _agg(_load(market, "con_abridor"))
        if clean is None:
            out[market] = {"enabled": False,
                           "reason": "sin backtest limpio ejecutado: no se publica"}
            continue
        base, n = clean["base_rate"], clean["n"]
        major = max(base, 1 - base)
        st = clean["strategies"]
        # se excluyen las lineas base como "mejor estrategia"
        cands = {k: v for k, v in st.items()
                 if k not in ("siempre_local", "siempre_visitante", "aleatorio")}
        best_name = max(cands, key=lambda k: cands[k]["accuracy"])
        best = cands[best_name]
        z = _z(best["accuracy"], major, n)
        ll_gain = _naive_ll(base) - best["log_loss"]
        sig = z > Z_SIGNIF
        enabled = ll_gain > MIN_LOGLOSS_GAIN
        leak = None
        if dirty:
            d_best = dirty["strategies"].get(best_name, {}).get("accuracy")
            if d_best is not None:
                leak = round((d_best - best["accuracy"]) * 100, 2)
        out[market] = {
            "enabled": bool(enabled), "publish_pick": bool(enabled and sig),
            "n": n, "base_rate": base, "majority_baseline": major,
            "best_strategy": best_name, "best_accuracy": best["accuracy"],
            "edge_vs_majority_pp": round((best["accuracy"] - major) * 100, 2),
            "z": round(z, 2), "significant_95": bool(sig),
            "logloss_gain_vs_base_rate": round(ll_gain, 4), "ece": best["ece"],
            "variant_used": VARIANT_GATE,
            "leakage_inflation_pp": leak,
            "all_strategies": {k: round(v["accuracy"], 4) for k, v in st.items()},
            "reason": _reason(market, enabled, sig, ll_gain, z, best_name, leak),
        }
    out["total"] = _total()
    return out


def _reason(market, enabled, sig, ll_gain, z, best, leak) -> str:
    extra = ("" if leak is None else
             f" (el backtest con features del abridor daba {leak:+.2f} pp mas: contaminado)")
    if not enabled:
        return (f"no supera el umbral: mejora de log loss {ll_gain:+.4f}. No se publica.{extra}")
    if sig:
        return (f"habilitado como PICK con '{best}': ventaja sobre la clase mayoritaria "
                f"significativa (z={z:.2f}) y log loss {ll_gain:+.4f}.{extra}")
    return (f"habilitado SOLO como probabilidad calibrada: el log loss mejora "
            f"{ll_gain:+.4f}, pero la ventaja de acierto NO es significativa "
            f"(z={z:.2f}). No debe leerse como 'el sistema acierta mas'.{extra}")


def _total() -> dict:
    f = MLB_BACKTEST_DIR / "walkforward_v1.json"
    if not f.exists():
        return {"enabled": False, "reason": "sin backtest de totales"}
    t = json.loads(f.read_text())["summary"].get("total")
    if not t:
        return {"enabled": False, "reason": "sin backtest de totales"}
    gain = t["baseline_rmse_media"] - t["rmse"]
    # Se publica SIEMPRE como probabilidad informativa (Over/Under frente a la
    # linea del mercado), nunca como pick: el backtest no demostro que el
    # modelo proyecte carreras mejor que la media historica.
    return {"enabled": True, "rmse": t["rmse"],
            "baseline_rmse": t["baseline_rmse_media"], "rmse_gain": round(gain, 4),
            "publish_pick": gain > 0.15,
            "reason": ("mejora el RMSE frente a predecir la media" if gain > 0.15 else
                       f"probabilidad informativa: el modelo no proyecta carreras mejor "
                       f"que la media historica ({gain:+.3f} carreras de RMSE). "
                       f"No se publica como pick.")}


def enabled_markets(d: dict | None = None) -> list[str]:
    d = d or decide()
    return [k for k, v in d.items() if isinstance(v, dict) and v.get("enabled")]


def production_model(market: str, d: dict | None = None) -> str | None:
    """Que algoritmo usar en produccion segun la evidencia limpia."""
    d = d or decide()
    v = d.get(market, {})
    return v.get("best_strategy") if v.get("enabled") else None
