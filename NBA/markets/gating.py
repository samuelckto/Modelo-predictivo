"""Estado de cada mercado NBA por evidencia walk-forward (research*.json).

  PICK        moneyline si sus picks (umbral elegido en T-1) baten al 50 % con
              z >= 1.96 en el holdout y la calibracion es aceptable (ECE < 0.05).
  PROJECTION  spread/total: la proyeccion mejora la media historica y la
              distribucion esta calibrada, pero NO hay lineas historicas para
              demostrar ventaja frente al mercado.
  BLOCKED     no mejora la media historica o la distribucion no esta calibrada.
"""
from __future__ import annotations

import json

from shared.paths import NBA_OUT_DIR

OUT = NBA_OUT_DIR


def decide() -> dict:
    f1, f2 = OUT / "research.json", OUT / "research_lines.json"
    if not f1.exists() or not f2.exists():
        return {m: {"enabled": False, "publish_pick": False, "mode": "blocked",
                    "reason": "sin investigacion walk-forward: python spc.py nba-train"}
                for m in ("moneyline", "spread", "total")}
    r1, r2 = json.loads(f1.read_text()), json.loads(f2.read_text())
    out = {}
    # --- moneyline ---------------------------------------------------------------
    a, b = r1["targets"]["moneyline"], r2["targets"]["moneyline"]
    pk, h = b["picks_holdout"], b["holdout_calibrado"]
    ok = (pk["n"] or 0) >= 300 and (pk["z_vs_50"] or 0) >= 1.96 and h["ece"] < 0.05
    thr = sorted({v["umbral"] for v in b["por_temporada"].values() if v["umbral"] is not None})
    reason = (f"picks del holdout 2023-2026 (umbral {thr} elegido cada ano en T-1): n={pk['n']}, acierto "
              f"{pk['acc']*100:.1f}% frente a 50 % (z={pk['z_vs_50']:.1f}); tasa local {a['holdout_2023_2026']['tasa_local']*100:.1f}%; "
              f"calibracion ECE {h['ece']:.3f}. OJO: iguala al favorito Elo en esos mismos partidos "
              f"({pk['acc_fav_elo_mismos_partidos']*100:.1f}%, z={pk['z_vs_fav_elo']:.2f}) y NO esta medido contra el mercado.")
    out["moneyline"] = {"enabled": True, "publish_pick": ok, "mode": "pick" if ok else "projection",
                        "algorithm": a["algoritmo_elegido"], "n_features": a["n_features"],
                        "families": a["familias_finales"], "families_dropped": a["familias_descartadas"],
                        "threshold": thr[-1] if thr else None, "calibration_method": b["calibracion_elegida"],
                        "holdout": a["holdout_2023_2026"], "holdout_calibrado": h, "picks": pk,
                        "buckets": b["buckets_holdout"], "por_temporada": _ml_seasons(a, b),
                        "umbrales": b["umbrales"], "reason": reason}
    # --- spread / total --------------------------------------------------------------
    for m in ("spread", "total"):
        a, b = r1["targets"][m], r2["targets"][m]
        reg = a["holdout_2023_2026"]
        proj_ok = reg["modelo"]["mae"] < reg["media"]["mae"] - 0.25
        cal_ok = b["holdout_p_over_k"]["ece"] < 0.05
        mode = "projection" if (proj_ok and cal_ok) else "blocked"
        reason = (f"proyeccion: MAE {reg['modelo']['mae']:.2f} frente a {reg['media']['mae']:.2f} de la media "
                  f"historica (holdout 2023-2026, n={reg['modelo']['n']}); distribucion {b['distribucion_elegida']} "
                  f"(sigma {b['residuo_std']:.1f}) calibrada para P(y > k): ECE {b['holdout_p_over_k']['ece']:.3f}. "
                  f"SIN lineas historicas del mercado: no se puede demostrar ventaja contra las casas -> "
                  f"{'PROYECCION, nunca pick' if mode == 'projection' else 'BLOQUEADO'}.")
        out[m] = {"enabled": mode != "blocked", "publish_pick": False, "mode": mode,
                  "algorithm": a["algoritmo_elegido"], "n_features": a["n_features"],
                  "families": a["familias_finales"], "families_dropped": a["familias_descartadas"],
                  "distribution": b["distribucion_elegida"], "res_std": b["residuo_std"],
                  "holdout": reg, "holdout_p_over_k": b["holdout_p_over_k"], "buckets": b["buckets_holdout"],
                  "por_offset": b["por_offset_holdout"],
                  "por_temporada": {T: v for T, v in a["por_temporada"].items()}, "reason": reason}
    return out


def _ml_seasons(a, b):
    out = {}
    for T, v in a["por_temporada"].items():
        w = b["por_temporada"].get(T, {})
        out[T] = {"n": v["modelo"]["n"], "accuracy": v["modelo"]["accuracy"], "log_loss": v["modelo"]["log_loss"],
                  "brier": v["modelo"]["brier"], "ece": v["modelo"]["ece"], "auc": v["modelo"]["auc"],
                  "elo_accuracy": v["elo"]["accuracy"], "elo_log_loss": v["elo"]["log_loss"],
                  "tasa_local": v["tasa_local"], "siempre_favorito_elo": v["siempre_favorito_elo"],
                  "umbral": w.get("umbral"), "n_picks": w.get("n_picks"), "acc_picks": w.get("acc_picks"),
                  "acc_fav_elo_en_picks": w.get("acc_fav_elo_en_picks")}
    return out
