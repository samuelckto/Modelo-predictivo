"""Predicciones NBA versionadas: moneyline (pick/no_pick), spread y total (projection).

Corte as-of: 00:00 del dia del partido (las features solo usan dias anteriores).
La linea viene de las cuotas propias anteriores al momento de la prediccion; si
no hay, se usa una linea de referencia (proyeccion al .5) y se marca.
"""
from __future__ import annotations

from math import erf, sqrt

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from NBA.markets.db import NbaModelVersion, NbaPrediction
from NBA.markets.features import FEATURES_PARQUET, numeric
from NBA.markets.gating import decide
from NBA.markets.odds import consensus, movement
from shared.paths import NBA_MODELS_DIR
from shared.timeutil import json_safe, utcnow

LABEL = {
    "elo_diff": "diferencia de rating Elo", "elo_prob_home": "rating Elo", "elo_sum": "nivel Elo de ambos",
    "d_s_ortg": "eficiencia ofensiva (temporada)", "d_s_drtg": "eficiencia defensiva (temporada)",
    "d_l10_ortg": "ataque reciente (10 partidos)", "d_l10_drtg": "defensa reciente (10 partidos)",
    "d_s_margin": "margen medio de puntos", "d_l10_margin": "margen reciente", "d_s_won": "porcentaje de victorias",
    "d_s_pace": "diferencia de ritmo", "sum_s_pace": "ritmo de ambos", "sum_l10_pace": "ritmo reciente de ambos",
    "d_s_efg": "eficiencia de tiro (eFG%)", "d_s_ts": "eficiencia de tiro real (TS%)", "d_l10_efg": "tiro reciente",
    "d_s_fg3_rate": "volumen de triples", "d_s_fg3_pct": "acierto en triples", "d_s_rim_rate": "ataques al aro",
    "d_s_tov_rate": "perdidas de balon", "d_s_opp_tov_rate": "perdidas forzadas", "d_s_orb_pct": "rebote ofensivo",
    "d_s_drb_pct": "rebote defensivo", "d_s_ft_rate": "tiros libres", "d_s_opp_efg": "tiro permitido al rival",
    "d_rest_days": "diferencia de descanso", "h_b2b": "local en back-to-back", "a_b2b": "visitante en back-to-back",
    "d_games_last7": "carga de partidos en 7 dias", "sum_s_ortg": "ataque de ambos", "sum_s_drtg": "defensa de ambos",
    "sum_s_pts": "puntos que anotan ambos", "sum_s_opp_pts": "puntos que permiten ambos",
    "sum_l10_pts": "anotacion reciente de ambos", "sum_s_tov_rate": "perdidas de ambos", "month": "mes de temporada",
}
SIDE = {"moneyline": ("local", "visitante"), "spread": ("local cubre", "visitante cubre"), "total": ("over", "under")}


def _p_over(pred, line, art):
    if art["distribution"] == "normal":
        return float(0.5 * (1 - erf((line - pred) / (art["res_std"] * sqrt(2)))))
    return float(np.mean((pred + art["residuals"]) > line))


def _calibrate_ml(p, art):
    cal = art.get("calibration") or {}
    if cal.get("method") == "platt" and cal.get("params"):
        z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        return float(1 / (1 + np.exp(-(cal["params"]["slope"] * z + cal["params"]["intercept"]))))
    if cal.get("method") == "isotonica" and cal.get("params"):
        return float(np.clip(np.interp(p, cal["params"]["isotonic_x"], cal["params"]["isotonic_y"]), 1e-4, 1 - 1e-4))
    return float(p)


def _ref(v):
    return float(np.floor(v) + 0.5)


def _factors(art, x, top=5):
    m = art["model"]
    try:                                            # ridge: coeficiente x valor estandarizado
        imp, sc, reg = m.named_steps["imp"], m.named_steps["sc"], m.named_steps["m"]
        z = sc.transform(imp.transform(x[art["features"]]))[0]
        contrib = reg.coef_ * z
        unit = "pts"
    except Exception:                               # random forest: importancia x desviacion del valor
        try:
            rf = m.named_steps["m"]; imp = m.named_steps["imp"]
            xv = imp.transform(x[art["features"]])[0]
            med = imp.statistics_
            contrib = rf.feature_importances_ * np.sign(xv - med) * np.minimum(np.abs(xv - med), 1e9)
            unit = ""
        except Exception:
            return []
    order = np.argsort(-np.abs(contrib))[:top]
    out = []
    for i in order:
        f = art["features"][i]
        if abs(contrib[i]) < 1e-6:
            continue
        out.append({"feature": f, "label": LABEL.get(f, f), "value": round(float(contrib[i]), 3), "unit": unit})
    return out


def upcoming(X, days_ahead=3):
    now = utcnow()
    return X[X.home_points.isna() & (X.game_date >= pd.Timestamp(now.date()) - pd.Timedelta(days=1)) &
             (X.game_date <= pd.Timestamp(now.date()) + pd.Timedelta(days=days_ahead))].copy()


def predict_upcoming(session, days_ahead=3, reason="manual", X=None) -> dict:
    X = X if X is not None else pd.read_parquet(FEATURES_PARQUET)
    sub = upcoming(X, days_ahead)
    if sub.empty:
        return {"status": "empty", "games": 0, "predictions": 0, "note": "sin partidos NBA programados con features"}
    gate = decide(); now = utcnow()
    made = revised = 0; skipped = []
    mv = {m.market: m for m in session.execute(select(NbaModelVersion).where(NbaModelVersion.is_production.is_(True))).scalars()}
    for market in ("moneyline", "spread", "total"):
        g = gate[market]
        path = NBA_MODELS_DIR / f"{market}_v1.joblib"
        if not g["enabled"] or not path.exists():
            skipped.append({"market": market, "reason": g["reason"] if not g["enabled"] else "sin modelo (nba-train)"})
            continue
        art = joblib.load(path)
        Xn = numeric(sub, art["features"])
        raw = art["model"].predict_proba(Xn)[:, 1] if market == "moneyline" else art["model"].predict(Xn)
        for i, row in enumerate(sub.itertuples()):
            cons = consensus(session, row.game_id, market, now)
            p_mkt = cons["p_home"] if cons else None
            line = None; src = None; lts = None; expected = None
            if market == "moneyline":
                p_raw = float(raw[i]); p_cal = _calibrate_ml(p_raw, art)
                home_side = p_cal >= 0.5
                sel = row.home_abbr if home_side else row.away_abbr
                p_pick = p_cal if home_side else 1 - p_cal
                thr = art.get("threshold") or g.get("threshold")
                status_m = ("pick" if (g["publish_pick"] and thr is not None and p_pick >= thr)
                            else "no_pick" if g["publish_pick"] else "projection")
            else:
                expected = float(raw[i])
                if cons and cons.get("line") is not None:
                    line, src, lts = cons["line"], "the_odds_api", cons["available_at"]
                else:
                    line = _ref(-expected) if market == "spread" else _ref(expected)
                    src = "referencia (sin mercado)"
                # spread: linea = spread del local (negativo = favorito). Local cubre si margen > -linea
                thr_line = -line if market == "spread" else line
                p_raw = _p_over(expected, thr_line, art); p_cal = p_raw
                home_side = p_cal >= 0.5
                if market == "spread":
                    sel = f"{row.home_abbr} {line:+g}" if home_side else f"{row.away_abbr} {-line:+g}"
                else:
                    sel = f"{'OVER' if home_side else 'UNDER'} {line:g}"
                p_pick = p_cal if home_side else 1 - p_cal
                status_m = "projection"
            x1 = sub.iloc[[i]]
            expl = {"summary": _summary(market, row, expected, art, line, src, sel, p_pick, status_m, g),
                    "factors": _factors(art, x1), "mode": status_m, "gate_reason": g["reason"],
                    "flags": _flags(market, p_pick, p_mkt, home_side, src, session, row.game_id, status_m)}
            missing = []
            if pd.isna(getattr(row, "h_l10_pace", np.nan)) or pd.isna(getattr(row, "a_l10_pace", np.nan)):
                missing.append("menos de 3 partidos recientes de algun equipo")
            if getattr(row, "start_utc", None) is None or pd.isna(getattr(row, "start_utc", None)):
                missing.append("hora de tipoff desconocida")
            stored, rev = _store(session, row, market, art, expected, line, src, lts, cons, p_raw, p_cal, p_mkt,
                                 sel, status_m, p_pick, expl, missing, now, mv.get(market))
            made += stored; revised += rev
    session.flush()
    return {"status": "ok", "games": int(len(sub)), "predictions": made, "revisions": revised, "skipped": skipped}


def _summary(market, row, expected, art, line, src, sel, p_pick, status_m, g):
    h, a = row.home_abbr, row.away_abbr
    if market == "moneyline":
        s = f"{sel} gana con {p_pick*100:.0f}% (probabilidad calibrada). "
        s += ("Es PICK: en 2023-2026 los picks con esta confianza acertaron el 69 %." if status_m == "pick" else
              "NO PICK: por debajo del umbral del 55 % que la validacion exige." if status_m == "no_pick" else
              "Proyeccion informativa.")
        return s
    if market == "spread":
        fav = h if expected > 0 else a
        s = (f"Margen proyectado: {fav} por {abs(expected):.1f} (desviacion {art['res_std']:.1f}). Linea {h} {line:+g} "
             f"({src}). Lado mas probable: {sel} con {p_pick*100:.0f}%. ")
    else:
        s = (f"Total proyectado: {expected:.1f} puntos (desviacion {art['res_std']:.1f}). Linea {line:g} ({src}). "
             f"Lado mas probable: {sel} con {p_pick*100:.0f}%. ")
    s += ("PROYECCION, no pick: la probabilidad esta calibrada frente a lineas neutras, pero no hay historico "
          "de cuotas para demostrar ventaja contra las casas de apuestas.")
    return s


def _flags(market, p_pick, p_mkt, home_side, src, session, game_id, status_m):
    out = []
    if status_m == "no_pick" or (market != "moneyline" and p_pick < 0.53):
        out.append({"code": "NO_PICK", "text": "demasiado cerca del 50 %: no hay pick"})
    if p_mkt is not None:
        pm = p_mkt if home_side else 1 - p_mkt
        if abs(p_pick - pm) >= 0.06:
            out.append({"code": "MODEL_MARKET_DISAGREEMENT",
                        "text": f"Existe desacuerdo significativo entre modelo y mercado: modelo {p_pick*100:.0f}% "
                                f"frente a {pm*100:.0f}% sin vig ({(p_pick-pm)*100:+.1f} pp). No validado como valor."})
    if src and "referencia" in src:
        out.append({"code": "NO_MARKET", "text": "sin cuotas: linea de referencia"})
    mv = movement(session, game_id, market) if market != "moneyline" else None
    if mv and abs(mv["movimiento"]) >= (1.5 if market == "spread" else 2.5):
        out.append({"code": "LINE_MOVEMENT", "text": f"la linea se movio {mv['movimiento']:+g} desde la apertura ({mv['apertura']:g} -> {mv['actual']:g})"})
    return out


def _store(session, row, market, art, expected, line, src, lts, cons, p_raw, p_cal, p_mkt, sel, status_m, p_pick,
           expl, missing, now, mv):
    prev = session.execute(select(NbaPrediction).where(
        NbaPrediction.game_id == row.game_id, NbaPrediction.market_type == market,
        NbaPrediction.status != "superseded").order_by(NbaPrediction.version.desc())).scalars().first()
    if prev is not None:
        same = (prev.line == line and abs((prev.final_probability or 0) - p_cal) < 0.005 and
                prev.selection == sel and prev.status_market == status_m)
        if same:
            return 0, 0
        prev.status = "superseded"
    conf = "ALTA" if p_pick >= 0.62 else "MEDIA" if p_pick >= 0.545 else "BAJA"
    p = NbaPrediction(
        game_id=row.game_id, season=int(row.season), game_date=row.game_date.date(),
        start_utc=(None if getattr(row, "start_utc", None) is None or pd.isna(row.start_utc) else pd.Timestamp(row.start_utc).to_pydatetime()),
        home_abbr=row.home_abbr, away_abbr=row.away_abbr, market_type=market,
        model_version=art["name"], calibration_version=(art.get("calibration") or {}).get("method"),
        prediction_timestamp=now, cutoff_timestamp=pd.Timestamp(row.game_date).to_pydatetime(),
        expected_value=expected, expected_std=art.get("res_std"),
        line=line, line_source=src, line_timestamp=lts,
        odds_home=(cons or {}).get("odds_home"), odds_away=(cons or {}).get("odds_away"),
        p_raw=p_raw, model_probability=p_cal, market_probability=p_mkt, final_probability=p_cal,
        selection=sel, pick_probability=p_pick, status_market=status_m, confidence=conf,
        gap_pp=(None if p_mkt is None else round((p_cal - p_mkt) * 100, 2)),
        explanation=json_safe(expl), data_completeness=json_safe({"complete": not missing, "missing": missing}),
        status="published", version=(prev.version + 1) if prev else 1, parent_id=prev.id if prev else None)
    session.add(p)
    return 1, (1 if prev else 0)


def score(session, X=None) -> dict:
    X = X if X is not None else pd.read_parquet(FEATURES_PARQUET)
    fin = X[X.home_points.notna()].set_index("game_id")
    preds = session.execute(select(NbaPrediction).where(NbaPrediction.result.is_(None),
                                                        NbaPrediction.status != "superseded")).scalars().all()
    n = 0
    for p in preds:
        if p.game_id not in fin.index:
            continue
        g = fin.loc[p.game_id]
        hp, ap = float(g.home_points), float(g.away_points)
        if p.market_type == "moneyline":
            actual = hp - ap
            p.actual_value = actual
            home_won = actual > 0
            p.correct = bool(home_won == ((p.final_probability or .5) >= 0.5)); p.result = "win" if p.correct else "loss"
        elif p.market_type == "spread":
            if p.line is None:
                continue
            actual = hp - ap; p.actual_value = actual
            cover_val = actual + p.line                 # local cubre si margen + linea > 0
            if cover_val == 0:
                p.result, p.correct = "push", None
            else:
                home_cover = cover_val > 0
                p.correct = bool(home_cover == ((p.final_probability or .5) >= 0.5)); p.result = "win" if p.correct else "loss"
        else:
            if p.line is None:
                continue
            actual = hp + ap; p.actual_value = actual
            if actual == p.line:
                p.result, p.correct = "push", None
            else:
                over = actual > p.line
                p.correct = bool(over == ((p.final_probability or .5) >= 0.5)); p.result = "win" if p.correct else "loss"
        n += 1
    session.flush()
    return {"scored": n}
