"""Estado de cada mercado de tenis por circuito, segun la evidencia walk-forward.

  PICK        ganador: los picks del umbral (elegido en T-1) baten al 50 % con
              z >= 1.96 en el holdout, la calibracion es buena (ECE < 0.05) y
              ademas superan al favorito Elo (z >= 1.64).
  PROJECTION  juegos/handicap: la proyeccion mejora al baseline (media historica
              o margen cero) pero no hay lineas historicas para probar ventaja.
  BLOCKED     no mejora al baseline.
"""
from __future__ import annotations

import json

from shared.paths import TENIS_OUT_DIR

OUT = TENIS_OUT_DIR
TOURS = ("ATP", "WTA")
MARKETS = ("winner", "total_games", "handicap_games")


def _blocked(reason):
    return {m: {"enabled": False, "publish_pick": False, "mode": "blocked", "reason": reason} for m in MARKETS}


def decide() -> dict:
    f1, f2 = OUT / "research.json", OUT / "research_lines.json"
    if not f1.exists() or not f2.exists():
        return {t: _blocked("sin investigacion walk-forward: python spc.py tenis-train") for t in TOURS}
    r1, r2 = json.loads(f1.read_text())["tours"], json.loads(f2.read_text())["tours"]
    out = {}
    for tour in TOURS:
        if tour not in r2:
            out[tour] = _blocked(f"{tour} sin investigacion"); continue
        a, b = r1.get(tour, {}), r2[tour]
        res = {}
        # --- ganador ---------------------------------------------------------------
        pk, h = b["picks_holdout"], b["holdout_calibrado"]
        ok = ((pk["n"] or 0) >= 500 and (pk["z_vs_50"] or 0) >= 1.96 and h["ece"] < 0.05
              and (pk["z_vs_elo"] or -9) >= 1.64)
        thr = sorted({v["umbral"] for v in b["por_temporada"].values() if v["umbral"] is not None})
        res["winner"] = {
            "enabled": True, "publish_pick": ok, "mode": "pick" if ok else "projection",
            "engine": b["motor_elegido"], "calibration_method": b["calibracion_elegida"],
            "threshold": thr[-1] if thr else None, "thresholds_used": thr,
            "holdout": h, "picks": pk, "buckets": b["buckets_holdout"],
            "por_temporada": b["por_temporada"], "umbrales": b["umbrales"],
            "baselines": a.get("baselines_seleccion"),
            "engines_compared": b["calibracion"],
            "reason": (f"picks del holdout (umbral {thr} elegido cada ano en T-1): n={pk['n']}, acierto "
                       f"{pk['acc']*100:.1f}% frente a 50 % (z={pk['z_vs_50']:.1f}) y frente al favorito Elo en "
                       f"los mismos partidos ({pk['acc_elo_mismos']*100:.1f}%, z={pk['z_vs_elo']:.2f}); "
                       f"calibracion ECE {h['ece']:.3f}. NO esta medido contra las cuotas.")}
        # --- total de juegos --------------------------------------------------------
        j = b["juegos"]; hj = j["holdout"]
        mejora = hj["mae"] < hj["mae_media"] - 0.10
        cal_ok = j["p_over_calibracion"]["ece"] < 0.05
        mode = "projection" if mejora else "blocked"
        res["total_games"] = {
            "enabled": mode != "blocked", "publish_pick": False, "mode": mode,
            "holdout": hj, "calibracion": j["p_over_calibracion"], "buckets": j["buckets"],
            "por_temporada": j["por_temporada"],
            "reason": (f"juegos proyectados: MAE {hj['mae']:.2f} frente a {hj['mae_media']:.2f} de la media "
                       f"historica (holdout, n={hj['n']}). "
                       + ("Calibracion de P(over) aceptable" if cal_ok else
                          f"P(over) MAL calibrada (ECE {j['p_over_calibracion']['ece']:.3f}): se muestra la "
                          f"proyeccion de juegos, la probabilidad es orientativa")
                       + ". Sin lineas historicas del mercado: nunca es pick." if mejora else
                       f"BLOQUEADO: no mejora a la media historica ({hj['mae']:.2f} vs {hj['mae_media']:.2f}).")}
        # --- handicap ---------------------------------------------------------------
        hh = b["handicap"]["holdout"]
        mejora_h = hh["mae"] < hh["mae_cero"] - 0.10
        cal_h = b["handicap"]["p_cover_calibracion"]["ece"] < 0.05
        mode_h = "projection" if mejora_h else "blocked"
        res["handicap_games"] = {
            "enabled": mode_h != "blocked", "publish_pick": False, "mode": mode_h,
            "holdout": hh, "calibracion": b["handicap"]["p_cover_calibracion"], "buckets": b["handicap"]["buckets"],
            "reason": (f"margen de juegos proyectado: MAE {hh['mae']:.2f} frente a {hh['mae_cero']:.2f} de "
                       f"suponer empate; P(cubre) con ECE {b['handicap']['p_cover_calibracion']['ece']:.3f}"
                       + (" (calibrada)" if cal_h else " (orientativa)")
                       + ". Sin lineas historicas: nunca es pick." if mejora_h else
                       f"BLOQUEADO: no mejora al margen cero.")}
        out[tour] = res
    return out


def market_state(tour: str, market: str, d: dict | None = None) -> dict:
    d = d or decide()
    return (d.get(tour) or {}).get(market, {"enabled": False, "mode": "blocked", "publish_pick": False,
                                            "reason": "circuito o mercado desconocido"})
