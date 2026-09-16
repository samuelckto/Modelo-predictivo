"""Valida el Upset Risk con datos fuera de muestra.

Entrena moneyline con 2021-2024, predice 2025 y 2026, y comprueba si a mayor
riesgo declarado hay realmente mas fallos. Si la relacion no aparece, el riesgo
no sirve y hay que decirlo en el informe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from MLB.engine import upset                                # noqa: E402
from MLB.engine.elo import fit_params, run as run_elo       # noqa: E402
from MLB.engine.models import dispersion, train             # noqa: E402
from shared.paths import MLB_BACKTEST_DIR, MLB_PROCESSED_DIR  # noqa: E402
from shared.timeutil import json_safe                       # noqa: E402


def main():
    X = pd.read_parquet(MLB_PROCESSED_DIR / "features.parquet")
    cfg = fit_params(X[X.season <= 2024], [2021, 2022], 2023)
    elo = run_elo(X, cfg)
    Xe = X.merge(elo[["game_id", "p_home_elo"]], on="game_id", how="left")
    Xe["d_elo"] = Xe.p_home_elo - 0.5
    mm = train(Xe, "moneyline", [2021, 2022, 2023, 2024])
    te = Xe[(Xe.season >= 2025) & Xe.y_home_win.notna()].copy()
    p = mm.predict_proba(te)
    disp = dispersion(mm.member_probs(te))
    pick_home = p >= 0.5
    p_sel = np.where(pick_home, p, 1 - p)
    elo_sel = np.where(pick_home, te.p_home_elo, 1 - te.p_home_elo)
    wrong = (pick_home != (te.y_home_win == 1)).astype(int).values

    rows = []
    for i in range(len(te)):
        r = te.iloc[i]
        missing = int(pd.isna(r.get("h_sp_era_s"))) + int(pd.isna(r.get("a_sp_era_s")))
        c = upset.components(p_sel[i], dispersion=float(disp[i]), p_elo=float(elo_sel[i]),
                             missing=missing, lineup_confirmed=True,
                             starter_known=(missing == 0), p_market=None)
        c["wrong"] = wrong[i]
        c["risk_default"] = upset.score(c)
        rows.append(c)
    d = pd.DataFrame(rows)

    out = {"n": int(len(d)), "seasons_test": [2025, 2026],
           "pesos_por_defecto": upset.validate(d.risk_default.values, d.wrong.values)}

    # pesos ajustados a datos ANTERIORES (2021-2024) y evaluados en 2025-2026
    tr = Xe[(Xe.season <= 2024) & Xe.y_home_win.notna()].copy()
    ptr = mm.predict_proba(tr)
    dtr = dispersion(mm.member_probs(tr))
    ph = ptr >= 0.5
    ptr_sel = np.where(ph, ptr, 1 - ptr)
    elo_tr = np.where(ph, tr.p_home_elo, 1 - tr.p_home_elo)
    wtr = (ph != (tr.y_home_win == 1)).astype(int).values
    rtr = []
    for i in range(len(tr)):
        r = tr.iloc[i]
        miss = int(pd.isna(r.get("h_sp_era_s"))) + int(pd.isna(r.get("a_sp_era_s")))
        c = upset.components(ptr_sel[i], dispersion=float(dtr[i]), p_elo=float(elo_tr[i]),
                             missing=miss, lineup_confirmed=True, starter_known=(miss == 0))
        c["wrong"] = wtr[i]
        rtr.append(c)
    dtr_df = pd.DataFrame(rtr)
    w = upset.fit_weights(dtr_df)
    d["risk_fitted"] = [upset.score(row, w) for row in d[list(upset.COMPONENTS)].to_dict("records")]
    out["pesos_ajustados"] = {"weights": {k: round(v, 4) for k, v in w.items()},
                              "ajustados_con": "2021-2024 (en muestra de entrenamiento)",
                              "evaluados_en": "2025-2026",
                              **upset.validate(d.risk_fitted.values, d.wrong.values)}
    out["correlacion_riesgo_error"] = {
        "default": round(float(np.corrcoef(d.risk_default, d.wrong)[0, 1]), 4),
        "fitted": round(float(np.corrcoef(d.risk_fitted, d.wrong)[0, 1]), 4)}
    (MLB_BACKTEST_DIR / "upset_validation.json").write_text(json.dumps(json_safe(out), indent=1))
    print(json.dumps(json_safe(out), indent=1))


if __name__ == "__main__":
    main()
