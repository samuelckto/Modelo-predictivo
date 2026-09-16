"""Predicciones NFL de total y spread (mercados propios de SPC), versionadas.

Para cada partido futuro del motor NFL (leido, nunca escrito):
  * proyeccion de puntos/margen con el modelo de produccion;
  * linea: consenso de cuotas propias anteriores al momento de la prediccion;
    si no hay, la linea publicada por nflverse en el calendario (fuente marcada);
    si tampoco, una linea de referencia (proyeccion al .5) marcada como tal;
  * P(over)/P(local cubre) cruda (distribucion de residuos) y CALIBRADA con el
    walk-forward. La decision pick/proyeccion/no_pick la dicta gating.decide().
"""
from __future__ import annotations

from math import erf, sqrt

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from NFL.markets.data import load, numeric
from NFL.markets.db import NflMarketPrediction, NflModelVersion
from NFL.markets.gating import decide
from NFL.markets.odds import consensus
from shared.paths import NFL_MARKETS_MODELS_DIR
from shared.timeutil import json_safe, utcnow

SRC_TS = ["home_src_stats_ts", "away_src_stats_ts", "home_src_depth_ts", "away_src_depth_ts",
          "home_src_inj_ts", "away_src_inj_ts"]
LABEL = {
    "sum_epa_off": "eficiencia ofensiva conjunta (EPA)", "sum_epa_def": "eficiencia defensiva conjunta (EPA)",
    "sum_l5_epa_off": "ataque reciente de ambos (EPA, ultimos 5)", "sum_l5_epa_def": "defensa reciente de ambos",
    "sum_points_for": "puntos que anotan ambos", "sum_points_against": "puntos que permiten ambos",
    "pace_proxy": "ritmo de anotacion de ambos", "pace_l5_proxy": "ritmo reciente de ambos",
    "sum_redzone_td_off": "eficiencia en zona roja", "sum_third_down_off": "conversion de tercer down",
    "sum_explosive_off": "jugadas explosivas", "sum_success_off": "tasa de exito ofensiva",
    "wind_mph": "viento", "cold": "frio", "is_dome": "estadio techado", "windy": "viento fuerte",
    "temp_f": "temperatura", "primetime": "horario estelar", "is_playoff": "playoffs",
    "elo_diff": "diferencia de rating Elo", "elo_prob_home": "rating Elo (prob. local)",
    "diff_epa_off": "diferencia de eficiencia ofensiva (EPA)", "diff_epa_def": "diferencia de eficiencia defensiva",
    "diff_l5_epa_off": "ataque reciente (EPA, ultimos 5)", "diff_l5_epa_def": "defensa reciente",
    "diff_point_diff": "diferencia de margen de puntos", "diff_points_for": "diferencia de puntos anotados",
    "diff_won": "diferencia de victorias", "diff_success_off": "diferencia de tasa de exito",
    "diff_explosive_off": "diferencia en jugadas explosivas", "diff_cpoe_off": "precision del QB (CPOE)",
    "home_qb_quality": "calidad del QB local", "away_qb_quality": "calidad del QB visitante",
    "home_qb_listed_out": "QB local fuera", "away_qb_listed_out": "QB visitante fuera",
    "home_inj_out_weighted": "lesionados del local", "away_inj_out_weighted": "lesionados del visitante",
    "rest_diff": "diferencia de descanso", "div_game": "partido divisional", "week_num": "semana",
}


def _p_over(pred: float, line: float, art: dict) -> float:
    res = art["residuals"]
    if art["distribution"] == "normal":
        s = art["res_std"]
        return float(0.5 * (1 - erf((line - pred) / (s * sqrt(2)))))
    return float(np.mean((pred + res) > line))


def _calibrate(p: float, art: dict) -> float:
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    return float(art["calibrator"].predict_proba(np.array([[z]]))[0, 1])


def _contributions(art: dict, x: pd.DataFrame, top: int = 5) -> list[dict]:
    """Aportes del ridge: coeficiente (estandarizado) x valor estandarizado."""
    m = art["model"]
    imp, sc, ridge = m.named_steps["imp"], m.named_steps["sc"], m.named_steps["m"]
    z = sc.transform(imp.transform(x[art["features"]]))[0]
    contrib = ridge.coef_ * z
    order = np.argsort(-np.abs(contrib))[:top]
    return [{"feature": art["features"][i], "label": LABEL.get(art["features"][i], art["features"][i]),
             "points": round(float(contrib[i]), 2)} for i in order if abs(contrib[i]) >= 0.3]


def _ref_line(v: float) -> float:
    return float(np.floor(v) + 0.5)


def confidence_label(p_pick: float, mode: str) -> str:
    if mode != "pick":
        return "BAJA"
    return "ALTA" if p_pick >= 0.62 else "MEDIA" if p_pick >= 0.56 else "BAJA"


def upcoming(X: pd.DataFrame, days_ahead: int = 10) -> pd.DataFrame:
    now = utcnow()
    return X[(X.home_score.isna()) & (X.kickoff_utc > now - pd.Timedelta(hours=4)) &
             (X.kickoff_utc <= now + pd.Timedelta(days=days_ahead))].copy()


def predict_upcoming(session, days_ahead: int = 10, reason: str = "manual") -> dict:
    X = load()
    sub = upcoming(X, days_ahead)
    if sub.empty:
        return {"status": "empty", "games": 0, "predictions": 0}
    gate = decide()
    now = utcnow()
    mv = {m.market: m for m in session.execute(select(NflModelVersion).where(
        NflModelVersion.is_production.is_(True))).scalars()}
    made = revised = 0
    skipped = []
    for market in ("total", "spread"):
        g = gate[market]
        if not g["enabled"]:
            skipped.append({"market": market, "reason": g["reason"]})
            continue
        path = NFL_MARKETS_MODELS_DIR / f"{market}_v1.joblib"
        if not path.exists():
            skipped.append({"market": market, "reason": "sin modelo entrenado (spc.py nfl-train)"})
            continue
        art = joblib.load(path)
        preds = art["model"].predict(numeric(sub, art["features"]))
        for i, row in enumerate(sub.itertuples()):
            mu = float(preds[i])
            median = mu + art["res_median"]
            cons = consensus(session, row.game_id, market, now)
            if cons and cons.get("line") is not None:
                line, src, lts = cons["line"], "the_odds_api", cons["available_at"]
            else:
                nv = row.total_line if market == "total" else row.spread_line
                if nv is not None and not pd.isna(nv):
                    line, src, lts = float(nv), "nflverse (linea publicada en el calendario)", None
                else:
                    line, src, lts = _ref_line(median), "referencia (sin mercado)", None
            p_raw = _p_over(mu, line, art)
            p_cal = _calibrate(p_raw, art)
            p_mkt = cons["p_home"] if cons else None
            p_final = p_cal
            over_side = p_final >= 0.5
            if market == "total":
                sel = f"{'OVER' if over_side else 'UNDER'} {line:g}"
            else:
                # linea = spread del local con signo nflverse (positivo = local favorito)
                sel = (f"{row.home_team} {-line:+g}" if over_side else f"{row.away_team} {line:+g}")
            p_pick = p_final if over_side else 1 - p_final
            mode = g["mode"]
            pick = "pick" if (mode == "pick") else "proyeccion"
            x1 = sub.iloc[[i]]
            contrib = _contributions(art, x1)
            missing = []
            for c in SRC_TS:
                v = getattr(row, c, None)
                if v is None or pd.isna(v):
                    missing.append(c.replace("_src_", " ").replace("_ts", ""))
            if getattr(row, "home_qb_listed_out", 0) == 1 or getattr(row, "away_qb_listed_out", 0) == 1:
                missing.append("QB titular en duda")
            cutoff = max([pd.Timestamp(getattr(row, c)) for c in SRC_TS
                          if getattr(row, c, None) is not None and not pd.isna(getattr(row, c))],
                         default=None)
            expl = {
                "summary": _summary(market, row, mu, median, art["res_std"], line, src, sel, p_pick, p_raw, mode),
                "factors": contrib, "mode": mode, "gate_reason": g["reason"],
                "flags": _flags(p_pick, p_mkt, over_side, missing, src),
            }
            stored, was_rev = _store(session, row, market, art, mu, median, line, src, lts, p_raw, p_cal,
                                     p_mkt, p_final, sel, pick, p_pick, confidence_label(p_pick, mode),
                                     expl, missing, cutoff, now, mv.get(market))
            made += stored
            revised += was_rev
    session.flush()
    return {"status": "ok", "games": int(len(sub)), "predictions": made, "revisions": revised,
            "skipped": skipped}


def _summary(market, row, mu, median, sd, line, src, sel, p_pick, p_raw, mode):
    h, a = row.home_team, row.away_team
    if market == "total":
        s = (f"El modelo proyecta {mu:.1f} puntos en total (mediana {median:.1f}, desviacion "
             f"tipica {sd:.1f}) frente a una linea de {line:g} ({src}). ")
    else:
        fav = h if mu > 0 else a
        s = (f"El modelo proyecta un margen de {abs(mu):.1f} puntos a favor de {fav} "
             f"(desviacion tipica {sd:.1f}) frente a una linea de {h} {-line:+g} ({src}). ")
    s += f"Lado mas probable: {sel} con {p_pick*100:.0f}% (sin calibrar seria {max(p_raw, 1-p_raw)*100:.0f}%). "
    if mode != "pick":
        s += ("Esto es una PROYECCION, no un pick: en las pruebas 2020-2025 el modelo no supero a la "
              "linea de cierre, y una probabilidad alta sin calibrar no acertaba mas que una baja.")
    return s


def _flags(p_pick, p_mkt, over_side, missing, src):
    out = []
    if p_pick < 0.53:
        out.append({"code": "NO_PICK", "text": "modelo y linea practicamente coinciden: no hay pick"})
    if p_mkt is not None:
        pm = p_mkt if over_side else 1 - p_mkt
        if abs(p_pick - pm) >= 0.06:
            out.append({"code": "MODEL_MARKET_DISAGREEMENT",
                        "text": f"desacuerdo modelo/mercado: modelo {p_pick*100:.0f}% frente a {pm*100:.0f}% "
                                f"del mercado sin vig ({(p_pick-pm)*100:+.1f} pp). No validado como valor."})
    if "referencia" in src:
        out.append({"code": "NO_MARKET", "text": "sin linea de mercado: se usa una linea de referencia"})
    for m in missing:
        out.append({"code": "DATA_INCOMPLETE", "text": m})
    return out


def _store(session, row, market, art, mu, median, line, src, lts, p_raw, p_cal, p_mkt, p_final,
           sel, pick, p_pick, conf, expl, missing, cutoff, now, mv):
    prev = session.execute(select(NflMarketPrediction).where(
        NflMarketPrediction.game_id == row.game_id, NflMarketPrediction.market_type == market,
        NflMarketPrediction.status != "superseded").order_by(NflMarketPrediction.version.desc())
    ).scalars().first()
    if prev is not None:
        same = (prev.line == line and abs((prev.final_probability or 0) - p_final) < 0.005 and
                abs((prev.expected_value or 0) - mu) < 0.05)
        if same:
            return 0, 0
        prev.status = "superseded"
    p = NflMarketPrediction(
        game_id=row.game_id, season=int(row.season), week=int(row.week),
        kickoff_utc=pd.Timestamp(row.kickoff_utc).to_pydatetime(),
        home_team=row.home_team, away_team=row.away_team, market_type=market,
        model_version=art["name"], prediction_timestamp=now,
        cutoff_timestamp=cutoff.to_pydatetime() if cutoff is not None else None,
        expected_value=mu, expected_median=median, expected_std=art["res_std"],
        line=line, line_source=src, line_timestamp=lts,
        p_over_raw=p_raw, model_probability=p_cal, market_probability=p_mkt, final_probability=p_final,
        selection=sel, pick=pick, pick_probability=p_pick, confidence=conf,
        gap_pp=(None if p_mkt is None else round((p_final - p_mkt) * 100, 2)),
        explanation=json_safe(expl),
        data_completeness=json_safe({"complete": not missing, "missing": missing}),
        status="published", version=(prev.version + 1) if prev else 1,
        parent_id=prev.id if prev else None)
    session.add(p)
    return 1, (1 if prev else 0)


def score(session, X: pd.DataFrame | None = None) -> dict:
    """Califica predicciones cuyos partidos ya tienen marcador (leido del motor NFL)."""
    X = X if X is not None else load()
    fin = X[X.home_score.notna()].set_index("game_id")
    preds = session.execute(select(NflMarketPrediction).where(
        NflMarketPrediction.result.is_(None), NflMarketPrediction.status != "superseded")).scalars().all()
    n = 0
    for p in preds:
        if p.game_id not in fin.index or p.line is None:
            continue
        g = fin.loc[p.game_id]
        actual = float(g.home_score + g.away_score) if p.market_type == "total" else float(g.home_score - g.away_score)
        p.actual_value = actual
        if actual == p.line:
            p.result, p.correct = "push", None
        else:
            over = actual > p.line
            picked_over = (p.final_probability or 0.5) >= 0.5
            p.correct = bool(over == picked_over)
            p.result = "win" if p.correct else "loss"
        n += 1
    session.flush()
    return {"scored": n}
