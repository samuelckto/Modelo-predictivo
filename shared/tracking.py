"""Seguimiento de favoritos >60 % y backtest de discrepancias.

Version FAVORITES_60_TRACKER_v1.

Que mide y que NO mide.

  MIDE     de todas las veces que el sistema dijo "esto pasa el 72 %",
           ¿cuantas veces paso? y ¿cuanto habria dejado apostarlo?

  NO MIDE  si conviene apostar. Eso necesita muestra, cuota registrada y un
           intervalo de confianza que no incluya el cero. Cuando falta algo de
           eso, este modulo lo dice y se calla.

Se agrupa por deporte, liga, mercado, temporada, modelo y version, y NUNCA se
mezclan mercados sin etiquetarlos: el 65 % de un moneyline y el 65 % de un
over de carreras no son la misma apuesta y promediarlos no significa nada.

La probabilidad que se sigue es la CALIBRADA del modelo, no la publicada tras
mezclar con el mercado. Seguir la publicada mediria una mezcla de dos fuentes
y no diria nada sobre el modelo.
"""
from __future__ import annotations

VERSION = "FAVORITES_60_TRACKER_v1"

UMBRAL = 0.60
BUCKETS = ((0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.01))
BUCKET_LABELS = ["60-65", "65-70", "70-75", "75-80", "80+"]
MIN_MUESTRA = 100


def bucket(p) -> str | None:
    if p is None:
        return None
    p = float(p)
    if p < UMBRAL:
        return None
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
        if lo <= p < hi:
            return lab
    return BUCKET_LABELS[-1]


def _metricas(regs: list[dict]) -> dict:
    from shared.calibration import brier, ece, log_loss
    from shared.roi import liquidar, roi_ic95
    caja = liquidar(regs)
    y = [1.0 if r["result"] == "win" else 0.0 for r in regs if r.get("result") in ("win", "loss")]
    p = [float(r["probability"]) for r in regs
         if r.get("result") in ("win", "loss") and r.get("probability") is not None]
    n = min(len(y), len(p))
    return {
        "n": len(regs),
        "n_con_resultado": caja["wins"] + caja["losses"] + caja["pushes"],
        "wins": caja["wins"], "losses": caja["losses"],
        "pushes": caja["pushes"], "voids": caja["voids"],
        "sin_cuota": caja["sin_cuota"],
        "win_rate": caja["win_rate"], "win_rate_ic95": caja["win_rate_ic95"],
        "prob_media_anunciada": caja["prob_modelo_media"],
        "implied_media": caja["implied_media"],
        "cuota_media": caja["cuota_media"],
        "n_apostables": caja["n_contabilizadas"],
        "profit_unidades": caja["profit_unidades"],
        "roi": caja["roi"], "yield": caja["yield_"], "roi_ic95": roi_ic95(caja),
        "log_loss": log_loss(y[:n], p[:n]) if n >= 2 else None,
        "brier": brier(y[:n], p[:n]) if n >= 2 else None,
        "ece": ece(y[:n], p[:n]) if n >= 20 else None,
        "muestra_suficiente": caja["n_contabilizadas"] >= MIN_MUESTRA,
    }


def _norm(rs: list[dict]) -> list[dict]:
    """Del registro del ledger al registro contable."""
    out = []
    for r in rs:
        p = r.get("model_probability_calibrated")
        if p is None:
            continue
        # La probabilidad del LADO PUBLICADO. Si el modelo dice 35 % para el
        # lado que eligio, el favorito es el otro lado y este registro no entra.
        out.append({
            "sport": r.get("sport"), "league": r.get("league"), "market": r.get("market"),
            "season": r.get("season"), "model_version": r.get("model_version"),
            "probability": float(p), "implied": r.get("market_probability"),
            "gap_pp": r.get("gap_pp"), "result": r.get("result"),
            "odds_decimal": r.get("odds_decimal"), "odds_american": r.get("odds_american"),
            "event_time": r.get("event_time"), "status": r.get("status"),
        })
    return out


def favoritos(sport: str | None = None, solo_calificadas: bool = True) -> dict:
    """Tracking de todo pick con probabilidad calibrada > 60 %."""
    from shared import ledger
    rs = _norm(ledger.records(sport=sport, solo_activas=False,
                              solo_calificadas=solo_calificadas))
    altos = [r for r in rs if r["probability"] > UMBRAL]

    por_bucket, por_grupo = {}, {}
    for r in altos:
        por_bucket.setdefault(bucket(r["probability"]), []).append(r)
        clave = (r["sport"], r["league"], r["market"], r["season"], r["model_version"])
        por_grupo.setdefault(clave, []).append(r)

    res = {
        "version": VERSION, "umbral": UMBRAL,
        "n_predicciones_totales": len(rs),
        "n_por_encima_del_umbral": len(altos),
        "global": _metricas(altos) if altos else None,
        "buckets": [{"bucket": b, **_metricas(por_bucket.get(b, []))}
                    for b in BUCKET_LABELS],
        "grupos": [{"sport": k[0], "league": k[1], "market": k[2], "season": k[3],
                    "model_version": k[4], **_metricas(v)}
                   for k, v in sorted(por_grupo.items(), key=lambda kv: str(kv[0]))],
    }
    apostables = sum(b["n_apostables"] for b in res["buckets"])
    res["estado"] = "ok" if apostables >= MIN_MUESTRA else "insufficient_data"
    if res["estado"] != "ok":
        res["nota"] = (
            f"{apostables} picks liquidados con cuota registrada de los {len(altos)} "
            f"que superaron el 60 %. Hacen falta {MIN_MUESTRA} para que el ROI "
            f"signifique algo. Los aciertos SI se pueden leer; el dinero todavia no.")
    return res


def discrepancias(sport: str | None = None) -> dict:
    """Backtest historico de discrepancias sobre predicciones ya calificadas."""
    from shared import discrepancy, ledger
    rs = ledger.records(sport=sport, solo_activas=False, solo_calificadas=True)
    regs = []
    for r in rs:
        if r.get("gap_pp") is None:
            continue
        regs.append({
            "sport": r["sport"], "market": r["market"], "gap_pp": r["gap_pp"],
            "probability": r.get("model_probability_calibrated"),
            "implied": r.get("market_probability"),
            "result": r.get("result"),
            "odds_decimal": r.get("odds_decimal"),
            "odds_american": r.get("odds_american"),
            "contradice_favorito": (r.get("gap_pp") or 0) > 0
                                   and (r.get("market_probability") or 1) < 0.5,
        })
    bt = discrepancy.backtest_buckets(regs)
    bt["por_deporte"] = {}
    for s in sorted({r["sport"] for r in regs}):
        bt["por_deporte"][s] = discrepancy.backtest_buckets(
            [r for r in regs if r["sport"] == s])
    bt["veredicto"] = discrepancy.veredicto(bt)
    return bt


def resumen() -> dict:
    """Vista unica para el dashboard: cobertura, favoritos y discrepancias."""
    from shared import ledger
    return {"cobertura": ledger.cobertura(),
            "favoritos": favoritos(),
            "discrepancias": discrepancias()}
