"""Predicciones de tenis versionadas: ganador (pick), total de juegos y handicap
(proyeccion). Los tres salen de la MISMA distribucion punto a punto.

Los partidos futuros vienen del calendario de The Odds API. Las features del
jugador se calculan como en el historico, con corte 00:00 del dia del partido.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from TENIS.markets.db import TenisPrediction, TenisSchedule, init_db
from TENIS.markets.features import FEATURES_PARQUET, numeric
from TENIS.markets.gating import decide
from TENIS.markets.odds import consensus, movement
from TENIS.markets.pointmodel import match_distribution, solve_serve_probs
from shared.paths import TENIS_MODELS_DIR
from shared.timeutil import json_safe, utcnow

LABEL = {"winner": "Ganador", "total_games": "Total de juegos", "handicap_games": "Hándicap de juegos"}
_D: dict = {}


def dist_for(p1, p2, bo):
    k = (int(round(p1 * 500)), int(round(p2 * 500)), int(bo))
    if k not in _D:
        _D[k] = match_distribution(k[0] / 500, k[1] / 500, int(bo))
    return _D[k]


def _ref_line(v, half=True):
    return float(np.floor(v) + 0.5) if half else float(round(v))


def player_state_features(X: pd.DataFrame, tour: str, pid: str, cutoff: pd.Timestamp, surface: str) -> dict:
    """Ultimas features conocidas del jugador (del historico), y su antiguedad."""
    h = X[((X.p1_id == pid) | (X.p2_id == pid)) & (X.tour == tour) & (X.date < cutoff)]
    if h.empty:
        return {}
    last = h.iloc[-1]
    tag = "p1" if last.p1_id == pid else "p2"
    out = {k[3:]: last[k] for k in X.columns if k.startswith(tag + "_") and k not in (f"{tag}_id", f"{tag}_name")}
    out["_last_match"] = last.date
    out["_days_since"] = (cutoff - last.date).days
    return out


def build_row(X: pd.DataFrame, sc: TenisSchedule, cutoff: pd.Timestamp) -> dict | None:
    f1 = player_state_features(X, sc.tour, sc.p1_id, cutoff, sc.surface)
    f2 = player_state_features(X, sc.tour, sc.p2_id, cutoff, sc.surface)
    if not f1 or not f2:
        return None
    row = {"best_of_5": 1.0 if (sc.best_of or 3) == 5 else 0.0}
    row["d_elo"] = (f1.get("elo") or 1500) - (f2.get("elo") or 1500)
    row["d_elo_surface"] = (f1.get("elo_surface") or 1500) - (f2.get("elo_surface") or 1500)
    r1, r2 = f1.get("rank"), f2.get("rank")
    row["log_rank_diff"] = (np.log(max(r1, 1)) - np.log(max(r2, 1))) if (r1 and r2) else 0.0
    row["d_pts"] = ((f1.get("pts") or 0) - (f2.get("pts") or 0))
    row["_f1"], row["_f2"] = f1, f2
    return row


def predict_upcoming(session, days_ahead: int = 3, reason: str = "manual", X: pd.DataFrame | None = None,
                     now: datetime | None = None) -> dict:
    X = X if X is not None else pd.read_parquet(FEATURES_PARQUET)
    now = now or utcnow()
    gate = decide()
    scs = session.execute(select(TenisSchedule).where(
        TenisSchedule.start_utc >= now - timedelta(hours=6),
        TenisSchedule.start_utc <= now + timedelta(days=days_ahead))).scalars().all()
    if not scs:
        return {"status": "empty", "partidos": 0, "predictions": 0,
                "note": "sin partidos de tenis en el calendario (The Odds API)"}
    made = revised = 0
    skipped = []
    arts = {}
    for t in ("ATP", "WTA"):
        p = TENIS_MODELS_DIR / f"{t}_winner_v1.joblib"
        if p.exists():
            arts[t] = joblib.load(p)
    for sc in scs:
        art = arts.get(sc.tour)
        if art is None:
            skipped.append({"event": sc.event_id, "reason": f"sin modelo {sc.tour} (tenis-train)"}); continue
        cutoff = pd.Timestamp(sc.start_utc.date())
        row = build_row(X, sc, cutoff)
        if row is None:
            skipped.append({"event": sc.event_id, "reason": "jugador sin historico suficiente"}); continue
        f1, f2 = row.pop("_f1"), row.pop("_f2")
        xr = pd.DataFrame([row])
        p_raw = float(art["model"].predict_proba(numeric(xr, art["features"]))[0, 1])
        p_cal = _calibrate(p_raw, art)
        p1s, p2s = solve_serve_probs(p_cal, art["avg_spw"], sc.best_of or 3)
        d = dist_for(p1s, p2s, sc.best_of or 3)
        missing = []
        for tag, f in (("p1", f1), ("p2", f2)):
            if f.get("_days_since", 999) > 120:
                missing.append(f"{sc.p1_name if tag == 'p1' else sc.p2_name}: sin partidos en {int(f['_days_since'])} dias")
        dias = (pd.Timestamp(sc.start_utc).normalize() - pd.Timestamp(X.date.max())).days
        if dias > 30:
            missing.append(f"el archivo historico termina el {pd.Timestamp(X.date.max()).date()} "
                           f"({dias} dias antes de este partido): forma y ranking pueden estar viejos")
        for market in ("winner", "total_games", "handicap_games"):
            g = gate.get(sc.tour, {}).get(market, {})
            if not g.get("enabled"):
                continue
            cons = consensus(session, sc.event_id, market, now)
            if market == "winner":
                p_mkt = cons["p1"] if cons else None
                p1_side = p_cal >= 0.5
                sel = sc.p1_name if p1_side else sc.p2_name
                p_pick = p_cal if p1_side else 1 - p_cal
                thr = art.get("threshold") or g.get("threshold")
                status = ("pick" if (g["publish_pick"] and thr is not None and p_pick >= thr)
                          else "no_pick" if g["publish_pick"] else "projection")
                line = src = lts = None; exp = None; praw_m = p_raw
            else:
                exp = d["exp_games"] if market == "total_games" else d["exp_margin"]
                if cons and cons.get("line") is not None:
                    line, src, lts = cons["line"], "the_odds_api", cons["available_at"]
                else:
                    line, src, lts = _ref_line(exp if market == "total_games" else -exp), "referencia (sin mercado)", None
                if market == "total_games":
                    praw_m = sum(v for k, v in d["games_total"].items() if k > line)
                else:
                    praw_m = sum(v for k, v in d["games_margin"].items() if k + line > 0)
                p_mkt = cons["p1"] if cons else None
                p1_side = praw_m >= 0.5
                p_pick = praw_m if p1_side else 1 - praw_m
                if market == "total_games":
                    sel = f"{'OVER' if p1_side else 'UNDER'} {line:g}"
                else:
                    sel = (f"{sc.p1_name} {line:+g}" if p1_side else f"{sc.p2_name} {-line:+g}")
                status = "projection"
                p_cal_m = praw_m
            expl = {"summary": _summary(market, sc, d, p_cal, p_pick, sel, line, src, exp, status, g),
                    "factors": _factors(row, f1, f2, sc, p1s, p2s), "mode": status, "gate_reason": g["reason"],
                    "flags": _flags(market, p_pick, p_mkt, p1_side, src, session, sc.event_id, status, missing)}
            stored, rev = _store(session, sc, market, art, (p_cal if market == "winner" else praw_m),
                                 (p_raw if market == "winner" else praw_m), p_mkt, sel, status, p_pick,
                                 exp, d, line, src, lts, cons, expl, missing, cutoff, now, p1s, p2s)
            made += stored; revised += rev
    session.flush()
    return {"status": "ok", "partidos": len(scs), "predictions": made, "revisions": revised, "skipped": skipped}


def _calibrate(p, art):
    cal = art.get("calibration") or {}
    par = cal.get("params")
    if cal.get("method") == "platt" and par:
        z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        return float(1 / (1 + np.exp(-(par["slope"] * z + par["intercept"]))))
    if cal.get("method") == "isotonica" and par:
        return float(np.clip(np.interp(p, par["x"], par["y"]), 1e-4, 1 - 1e-4))
    return float(p)


def _summary(market, sc, d, p_cal, p_pick, sel, line, src, exp, status, g):
    if market == "winner":
        s = f"{sel} gana con {p_pick*100:.0f}% (probabilidad calibrada). "
        s += ("Es PICK: en el holdout 2020-2026 los picks con esta confianza acertaron cerca del 68-69 %, "
              "por encima del favorito por Elo." if status == "pick" else
              "NO PICK: por debajo del umbral que la validacion exige." if status == "no_pick" else "Proyeccion.")
        return s
    if market == "total_games":
        return (f"Juegos proyectados: {exp:.1f} (partido al mejor de {sc.best_of}). Linea {line:g} ({src}). "
                f"Lado mas probable: {sel} con {p_pick*100:.0f}%. PROYECCION, no pick: sin historico de cuotas.")
    return (f"Margen de juegos proyectado: {sc.p1_name} {exp:+.1f}. Linea {sc.p1_name} {line:+g} ({src}). "
            f"Lado mas probable: {sel} con {p_pick*100:.0f}%. PROYECCION, no pick.")


def _factors(row, f1, f2, sc, p1s, p2s):
    out = [{"label": "diferencia de rating Elo", "value": round(float(row["d_elo"]), 1)},
           {"label": f"Elo en {sc.surface.lower() if sc.surface else 'esta superficie'}",
            "value": round(float(row["d_elo_surface"]), 1)},
           {"label": "puntos ganados al saque proyectados",
            "value": f"{sc.p1_name} {p1s*100:.1f}% vs {sc.p2_name} {p2s*100:.1f}%"}]
    for tag, f, nm in (("p1", f1, sc.p1_name), ("p2", f2, sc.p2_name)):
        if f.get("srf_spw") is not None and f.get("srf_spw") == f.get("srf_spw"):
            out.append({"label": f"{nm}: saque en {sc.surface}", "value": f"{float(f['srf_spw'])*100:.1f}%"})
        if f.get("y1_win") is not None:
            out.append({"label": f"{nm}: victorias 12 meses", "value": f"{float(f['y1_win'])*100:.0f}%"})
        if f.get("rest_days") is not None and f.get("rest_days") == f.get("rest_days"):
            out.append({"label": f"{nm}: dias de descanso", "value": int(f["rest_days"])})
    return out


def _flags(market, p_pick, p_mkt, side1, src, session, event_id, status, missing):
    out = []
    if status == "no_pick" or (market != "winner" and p_pick < 0.53):
        out.append({"code": "NO_PICK", "text": "demasiado cerca del 50 %: no hay pick"})
    if p_mkt is not None:
        pm = p_mkt if side1 else 1 - p_mkt
        if abs(p_pick - pm) >= 0.06:
            out.append({"code": "MODEL_MARKET_DISAGREEMENT",
                        "text": f"Existe desacuerdo significativo entre modelo y mercado: modelo {p_pick*100:.0f}% "
                                f"frente a {pm*100:.0f}% sin vig ({(p_pick-pm)*100:+.1f} pp). No validado como valor."})
    if src and "referencia" in str(src):
        out.append({"code": "NO_MARKET", "text": "sin cuotas: linea de referencia"})
    mv = movement(session, event_id, market)
    if mv:
        big = abs(mv.get("movimiento_pp", 0)) >= 5 if market == "winner" else abs(mv.get("movimiento", 0)) >= 1.5
        if big:
            out.append({"code": "LINE_MOVEMENT", "text": f"la linea se movio desde la apertura: {mv}"})
    for m in missing:
        out.append({"code": "DATA_INCOMPLETE", "text": m})
    return out


def _store(session, sc, market, art, p_final, p_raw, p_mkt, sel, status, p_pick, exp, d, line, src, lts,
           cons, expl, missing, cutoff, now, p1s, p2s):
    prev = session.execute(select(TenisPrediction).where(
        TenisPrediction.event_id == sc.event_id, TenisPrediction.market_type == market,
        TenisPrediction.status != "superseded").order_by(TenisPrediction.version.desc())).scalars().first()
    if prev is not None:
        same = (prev.line == line and abs((prev.final_probability or 0) - p_final) < 0.005
                and prev.selection == sel and prev.status_market == status)
        if same:
            return 0, 0
        prev.status = "superseded"
    conf = "ALTA" if p_pick >= 0.65 else "MEDIA" if p_pick >= 0.57 else "BAJA"
    p = TenisPrediction(
        event_id=sc.event_id, tour=sc.tour, season=sc.start_utc.year, match_date=sc.start_utc.date(),
        start_utc=sc.start_utc, tourney_name=sc.tourney_name, surface=sc.surface, best_of=sc.best_of,
        p1_name=sc.p1_name, p2_name=sc.p2_name, market_type=market, model_version=art["name"],
        engine=art.get("engine"), calibration_version=(art.get("calibration") or {}).get("method"),
        prediction_timestamp=now, cutoff_timestamp=cutoff.to_pydatetime(),
        expected_value=exp, expected_std=None, p_serve_1=p1s, p_serve_2=p2s,
        line=line, line_source=src, line_timestamp=lts,
        odds_1=(cons or {}).get("odds_1"), odds_2=(cons or {}).get("odds_2"),
        p_raw=p_raw, model_probability=p_final, market_probability=p_mkt, final_probability=p_final,
        selection=sel, pick_probability=p_pick, status_market=status, confidence=conf,
        gap_pp=(None if p_mkt is None else round((p_final - p_mkt) * 100, 2)),
        explanation=json_safe(expl),
        data_completeness=json_safe({"complete": not missing, "missing": missing}),
        status="published", version=(prev.version + 1) if prev else 1, parent_id=prev.id if prev else None)
    session.add(p)
    return 1, (1 if prev else 0)


def _grade(p: TenisPrediction, p1_won: bool, games_total, games_margin) -> bool:
    """Aplica la regla del mercado. Devuelve True si quedo calificada."""
    if p.market_type == "winner":
        p.actual_value = 1.0 if p1_won else 0.0
        p.correct = bool(p1_won == ((p.final_probability or .5) >= .5))
        p.result = "win" if p.correct else "loss"
        return True
    if p.line is None:
        return False
    if p.market_type == "total_games":
        if games_total is None:
            return False
        p.actual_value = float(games_total)
        if float(games_total) == p.line:
            p.result, p.correct = "push", None
        else:
            over = float(games_total) > p.line
            p.correct = bool(over == ((p.final_probability or .5) >= .5))
            p.result = "win" if p.correct else "loss"
        return True
    if games_margin is None:
        return False
    p.actual_value = float(games_margin)
    cover = float(games_margin) + p.line
    if cover == 0:
        p.result, p.correct = "push", None
    else:
        p.correct = bool((cover > 0) == ((p.final_probability or .5) >= .5))
        p.result = "win" if p.correct else "loss"
    return True


def score(session, X: pd.DataFrame | None = None) -> dict:
    """Califica con los resultados EN VIVO del calendario y, si no hay, con el
    archivo historico (que esta congelado: sin la fuente en vivo casi nada se
    podria calificar)."""
    from TENIS.markets.db import TenisResult
    from TENIS.markets.results import fetch as fetch_results
    fuentes = {}
    try:
        fuentes = fetch_results(session)
    except Exception as e:                                   # noqa: BLE001
        fuentes = {"error": f"{type(e).__name__}: {e}"}
    res = {r.event_id: r for r in session.execute(select(TenisResult)).scalars()}
    preds = session.execute(select(TenisPrediction).where(TenisPrediction.result.is_(None),
                                                          TenisPrediction.status != "superseded")).scalars().all()
    n = 0
    for p in preds:
        r = res.get(p.event_id)
        if r is not None and r.winner_name:
            p1_won = (r.winner_name == p.p1_name)
            if _grade(p, p1_won, r.games_total, r.games_margin):
                n += 1
            continue
        X = X if X is not None else pd.read_parquet(FEATURES_PARQUET)
        m = X[(X.tour == p.tour) & (X.date >= pd.Timestamp(p.match_date) - pd.Timedelta(days=2)) &
              (X.date <= pd.Timestamp(p.match_date) + pd.Timedelta(days=2)) &
              (((X.p1_name == p.p1_name) & (X.p2_name == p.p2_name)) |
               ((X.p1_name == p.p2_name) & (X.p2_name == p.p1_name)))]
        if m.empty:
            continue
        r = m.iloc[0]
        p1_first = (r.p1_name == p.p1_name)
        if p.market_type == "winner":
            p1_won = bool(r.p1_win == 1) if p1_first else bool(r.p1_win == 0)
            p.actual_value = 1.0 if p1_won else 0.0
            p.correct = bool(p1_won == ((p.final_probability or .5) >= .5))
            p.result = "win" if p.correct else "loss"
        elif p.market_type == "total_games":
            if r.games_total is None or p.line is None or pd.isna(r.games_total):
                continue
            p.actual_value = float(r.games_total)
            if float(r.games_total) == p.line:
                p.result, p.correct = "push", None
            else:
                over = float(r.games_total) > p.line
                p.correct = bool(over == ((p.final_probability or .5) >= .5))
                p.result = "win" if p.correct else "loss"
        else:
            if r.games_margin is None or p.line is None or pd.isna(r.games_margin):
                continue
            marg = float(r.games_margin) if p1_first else -float(r.games_margin)
            p.actual_value = marg
            cover = marg + p.line
            if cover == 0:
                p.result, p.correct = "push", None
            else:
                p.correct = bool((cover > 0) == ((p.final_probability or .5) >= .5))
                p.result = "win" if p.correct else "loss"
        p.match_id = r.match_id
        n += 1
    session.flush()
    return {"scored": n, "fuentes": fuentes}
