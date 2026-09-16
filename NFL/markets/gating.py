"""Decision por evidencia para total y spread NFL. Lee NFL/markets/out/*.json
(walk-forward 2020-2025 frente a la linea de cierre). Nunca se inventa nada:
si no hay evidencia, el mercado queda como PROYECCION (nunca PICK)."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
Z_PICK = 1.96          # p < 0.05 bilateral sobre los picks del umbral elegido en T-1
LL_COIN = 0.693147     # log loss de decir siempre 50 %


def decide() -> dict:
    f1, f2 = OUT / "research.json", OUT / "research_lines.json"
    if not f1.exists() or not f2.exists():
        return {m: {"enabled": False, "publish_pick": False, "mode": "bloqueado",
                    "reason": "sin investigacion walk-forward: ejecuta research.py y research_lines.py"}
                for m in ("total", "spread")}
    r1, r2 = json.loads(f1.read_text()), json.loads(f2.read_text())
    out = {}
    for m in ("total", "spread"):
        a, b = r1["targets"][m], r2["targets"][m]
        reg = a["total_2020_2025"]
        proj_ok = reg["modelo"]["mae"] < reg["media"]["mae"] - 0.1
        picks = b["picks_agregado"]
        ll = b["agregado_2020_2025"]["metricas_prob"]["log_loss"]
        pick_ok = (picks["n"] or 0) >= 100 and (picks["z"] or 0) >= Z_PICK and ll < LL_COIN
        mode = "pick" if pick_ok else ("proyeccion" if proj_ok else "bloqueado")
        reasons = []
        if proj_ok:
            reasons.append(f"la proyeccion de puntos si aporta: MAE {reg['modelo']['mae']:.2f} "
                           f"frente a {reg['media']['mae']:.2f} de la media historica "
                           f"(mercado de cierre {reg['mercado_cierre']['mae']:.2f})")
        else:
            reasons.append("la proyeccion no mejora a la media historica")
        if not pick_ok:
            reasons.append(f"frente a la linea de cierre NO hay ventaja: acierto "
                           f"{b['agregado_2020_2025']['acc_todos']*100:.1f}% en {b['agregado_2020_2025']['n']} "
                           f"partidos, log loss {ll:.4f} (peor que 50 % = {LL_COIN:.4f}); picks con umbral "
                           f"elegido en T-1: n={picks['n']}, acierto "
                           f"{(picks['acc'] or 0)*100:.1f}%, z={picks['z'] if picks['z'] is not None else 'n/a'}")
        out[m] = {"enabled": mode != "bloqueado", "publish_pick": pick_ok, "mode": mode,
                  "algorithm": a["algoritmo_elegido"], "n_features": a["n_features"],
                  "families": a["familias_finales"], "families_dropped": a["familias_descartadas"],
                  "distribution": b["distribucion_elegida"],
                  "regression": reg, "vs_line": b["agregado_2020_2025"],
                  "picks": picks, "calibration": b["calibracion_buckets"],
                  "key_numbers": b["key_numbers"], "reason": "; ".join(reasons),
                  "por_temporada": {T: {"regresion": a["por_temporada"][T],
                                        "vs_linea": {k: v for k, v in b["por_temporada"][T].items()
                                                     if k != "metricas_prob"},
                                        "log_loss": b["por_temporada"][T]["metricas_prob"]["log_loss"],
                                        "brier": b["por_temporada"][T]["metricas_prob"]["brier"]}
                                    for T in a["por_temporada"]},
                  "umbrales": b["umbrales"]}
    return out
