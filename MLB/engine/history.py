"""Historico de aciertos REALES del sistema (no del backtest).

Solo cuenta predicciones ya evaluadas contra el resultado del partido. Mientras
la muestra sea pequena lo dice explicitamente: con 20 partidos no se puede
concluir nada, y el sistema no va a fingir lo contrario.
"""
from __future__ import annotations

import math

import numpy as np
from sqlalchemy import select

from MLB.database.models import Prediction
from MLB.database.session import session_scope
from shared.calibration import summary

from MLB.engine.markets import LABEL as MARKET_LABEL, HIDDEN as HIDDEN_MARKETS  # noqa: E402
# n minimo para que la muestra empiece a decir algo (error tipico < 5 pp)
MIN_N_UTIL = 100


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Intervalo de confianza del 95 % para una proporcion. Con muestras pequenas
    es enorme, y esa es justamente la informacion que hay que enseñar."""
    if n == 0:
        return None
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def nfl_history() -> dict:
    """Historico real del motor NFL, leido en solo lectura. Si no hay partidos
    evaluados lo dice; no se rellena con el backtest, que es otra cosa."""
    try:
        import NFL.adapter as nfl
    except Exception as e:                                   # noqa: BLE001
        return {"disponible": False, "motivo": f"no se pudo leer el motor NFL: {e}"}
    ok, msg = nfl.available()
    if not ok:
        return {"disponible": False, "motivo": msg}
    import sqlite3
    with nfl._conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "select game_id, season, week, home_team, away_team, pick, pick_prob, "
            "correct, points, points_earned, kickoff_utc, model_version_label "
            "from predictions where is_active=1 and correct is not null "
            "order by kickoff_utc desc").fetchall()
    preds = [{
        "prediction_id": None, "game_id": r["game_id"], "fecha": (r["kickoff_utc"] or "")[:10],
        "inicio": r["kickoff_utc"], "home": r["home_team"], "away": r["away_team"],
        "market": "moneyline", "market_label": "Moneyline", "selection": r["pick"],
        "line": None, "probabilidad": r["pick_prob"], "modelo": None, "mercado": None,
        "riesgo": None, "resultado": "win" if r["correct"] else "loss",
        "acierto": bool(r["correct"]), "modelo_version": r["model_version_label"],
        "version": 1, "puntos": r["points_earned"], "puntos_max": r["points"],
    } for r in rows]
    n = len(preds)
    hits = sum(1 for p in preds if p["acierto"])
    ci = _wilson(hits, n)
    met = {}
    if n:
        y = np.array([1.0 if p["acierto"] else 0.0 for p in preds])
        pr = np.array([p["probabilidad"] for p in preds], dtype=float)
        met = summary(y, pr)
    pts = sum(p["puntos"] or 0 for p in preds)
    mx = sum(p["puntos_max"] or 0 for p in preds)
    out = {
        "disponible": True,
        "total": {"n": n, "aciertos": hits, "fallos": n - hits,
                  "accuracy": (hits / n) if n else None,
                  "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                  "muestra_util": n >= MIN_N_UTIL},
        "por_mercado": {"moneyline": {
            "market_label": "Moneyline", "n": n, "aciertos": hits, "fallos": n - hits,
            "accuracy": (hits / n) if n else None,
            "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "prob_media": (float(np.mean([p["probabilidad"] for p in preds]))
                           if n else None),
            "log_loss": met.get("log_loss"), "brier": met.get("brier"),
            "ece": met.get("ece"), "muestra_util": n >= MIN_N_UTIL,
            "fantasy_puntos": pts, "fantasy_max": mx,
            "fantasy_pct": (pts / mx) if mx else None,
            "aviso": (None if n >= MIN_N_UTIL else
                      (f"solo {n} predicciones evaluadas: muestra insuficiente"
                       if n else "la temporada NFL acaba de empezar: todavia no hay "
                                 "ningun partido evaluado")),
        }} if True else {},
        "predicciones": preds,
        "nota": ("Predicciones REALES del motor NFL ya jugadas. El backtest historico "
                 "(2020-2025) esta en la pestana Performance."),
    }
    _add_nfl_markets(out)
    return out


def _add_nfl_markets(out: dict) -> None:
    """Anade total/spread NFL (mercados propios de SPC) al historico real."""
    try:
        from sqlalchemy import select as _sel
        from NFL.markets.db import NflMarketPrediction, init_db, session_scope as nss
        init_db()
        with nss() as s:
            rows = s.execute(_sel(NflMarketPrediction).where(
                NflMarketPrediction.result.isnot(None), NflMarketPrediction.status != "superseded")
                .order_by(NflMarketPrediction.kickoff_utc.desc())).scalars().all()
    except Exception as e:                                   # noqa: BLE001
        out["nota_mercados"] = f"total/spread no disponibles: {e}"
        return
    label = {"total": "Total", "spread": "Spread"}
    for m in ("total", "spread"):
        sub = [p for p in rows if p.market_type == m and p.result != "push"]
        n = len(sub); hits = sum(1 for p in sub if p.correct)
        ci = _wilson(hits, n)
        met = {}
        if n:
            met = summary(np.array([1.0 if p.correct else 0.0 for p in sub]),
                          np.array([p.pick_probability for p in sub], dtype=float))
        out["por_mercado"][m] = {
            "market_label": label[m], "n": n, "aciertos": hits, "fallos": n - hits,
            "accuracy": (hits / n) if n else None,
            "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "prob_media": float(np.mean([p.pick_probability for p in sub])) if n else None,
            "log_loss": met.get("log_loss"), "brier": met.get("brier"), "ece": met.get("ece"),
            "muestra_util": n >= MIN_N_UTIL, "modo": "proyeccion",
            "pushes": sum(1 for p in rows if p.market_type == m and p.result == "push"),
            "aviso": (None if n >= MIN_N_UTIL else
                      (f"solo {n} proyecciones evaluadas" if n else "todavia sin partidos evaluados")),
        }
        for p in [p for p in rows if p.market_type == m]:
            out["predicciones"].append({
                "prediction_id": p.id, "game_id": p.game_id, "fecha": str(p.kickoff_utc)[:10],
                "inicio": p.kickoff_utc, "home": p.home_team, "away": p.away_team,
                "market": m, "market_label": label[m], "selection": p.selection, "line": p.line,
                "probabilidad": p.pick_probability, "modelo": p.model_probability,
                "mercado": p.market_probability, "riesgo": round((1 - (p.pick_probability or .5)) * 100, 1),
                "resultado": p.result, "acierto": bool(p.correct) if p.correct is not None else None,
                "modelo_version": p.model_version, "version": p.version,
                "proyeccion": p.expected_value, "real": p.actual_value})
    tot = [p for p in rows if p.result != "push"]
    ml = out["total"]
    n2 = ml["n"] + len(tot); h2 = ml["aciertos"] + sum(1 for p in tot if p.correct)
    ci = _wilson(h2, n2)
    out["total"] = {"n": n2, "aciertos": h2, "fallos": n2 - h2, "accuracy": (h2 / n2) if n2 else None,
                    "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                    "muestra_util": n2 >= MIN_N_UTIL}


def history(date_from: str | None = None, date_to: str | None = None) -> dict:
    with session_scope() as s:
        q = select(Prediction).where(Prediction.correct.isnot(None),
                                     Prediction.status != "superseded",
                                     Prediction.market.notin_(list(HIDDEN_MARKETS)))
        if date_from:
            q = q.where(Prediction.game_date >= date_from)
        if date_to:
            q = q.where(Prediction.game_date <= date_to)
        rows = s.execute(q.order_by(Prediction.game_start_utc.desc())).scalars().all()
        preds = [{
            "prediction_id": p.id, "game_id": p.game_id, "fecha": p.game_date,
            "inicio": p.game_start_utc, "home": p.home_abbr, "away": p.away_abbr,
            "market": p.market, "market_label": MARKET_LABEL.get(p.market, p.market),
            "selection": p.selection, "line": p.line,
            "probabilidad": p.ensemble_probability, "modelo": p.model_probability,
            "mercado": p.market_probability, "riesgo": p.upset_risk,
            "resultado": p.result, "acierto": bool(p.correct),
            "modelo_version": p.model_version, "version": p.version,
        } for p in rows]

    from MLB.engine.markets import ORDER as MARKET_ORDER
    por_mercado = {}
    for m in MARKET_ORDER + sorted({p["market"] for p in preds} - set(MARKET_ORDER)):
        sub = [p for p in preds if p["market"] == m]
        y = np.array([1.0 if p["acierto"] else 0.0 for p in sub])
        pr = np.array([p["probabilidad"] for p in sub], dtype=float)
        n, hits = len(sub), int(y.sum())
        ci = _wilson(hits, n)
        met = summary(y, pr) if n else {}
        por_mercado[m] = {
            "market_label": MARKET_LABEL.get(m, m), "n": n, "aciertos": hits,
            "fallos": n - hits, "accuracy": (hits / n) if n else None,
            "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "prob_media": float(pr.mean()) if n else None,
            "log_loss": met.get("log_loss"), "brier": met.get("brier"),
            "ece": met.get("ece"),
            "muestra_util": n >= MIN_N_UTIL,
            "aviso": (None if n >= MIN_N_UTIL else
                      (f"solo {n} predicciones evaluadas: con esta muestra el intervalo "
                       f"de confianza es demasiado ancho para concluir nada") if n else
                      "todavia sin partidos evaluados en este mercado"),
        }

    y = np.array([1.0 if p["acierto"] else 0.0 for p in preds])
    n, hits = len(preds), int(y.sum())
    ci = _wilson(hits, n)
    return {
        "disponible": True,
        "total": {"n": n, "aciertos": hits, "fallos": n - hits,
                  "accuracy": (hits / n) if n else None,
                  "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                  "muestra_util": n >= MIN_N_UTIL},
        "por_mercado": por_mercado,
        "predicciones": preds,
        "nota": ("Estas cifras son de predicciones REALES ya jugadas, no del backtest. "
                 "El backtest esta en la pestana Performance. Hasta acumular unos "
                 "cientos de partidos, la diferencia entre estas cifras y las del "
                 "backtest no significa que el sistema haya mejorado ni empeorado."),
    }


def nba_history() -> dict:
    """Historico real NBA (mercados propios de SPC)."""
    try:
        from sqlalchemy import select as _sel
        from NBA.markets.db import NbaPrediction, init_db, session_scope as nss
        init_db()
        with nss() as s:
            rows = s.execute(_sel(NbaPrediction).where(NbaPrediction.result.isnot(None), NbaPrediction.status != "superseded")
                             .order_by(NbaPrediction.game_date.desc())).scalars().all()
    except Exception as e:                                   # noqa: BLE001
        return {"disponible": False, "motivo": f"NBA no disponible: {e}"}
    label = {"moneyline": "Moneyline", "spread": "Spread", "total": "Total"}
    out = {"disponible": True, "por_mercado": {}, "predicciones": [],
           "nota": "Predicciones REALES NBA ya jugadas. Solo moneyline es pick; spread/total son proyecciones."}
    tot_n = tot_h = 0
    for m in ("moneyline", "spread", "total"):
        sub = [p for p in rows if p.market_type == m and p.result != "push"]
        picks = [p for p in sub if p.status_market == "pick"] if m == "moneyline" else sub
        n = len(sub); hits = sum(1 for p in sub if p.correct)
        ci = _wilson(hits, n)
        met = summary(np.array([1.0 if p.correct else 0.0 for p in sub]),
                      np.array([p.pick_probability for p in sub], dtype=float)) if n else {}
        out["por_mercado"][m] = {
            "market_label": label[m], "n": n, "aciertos": hits, "fallos": n - hits,
            "accuracy": (hits / n) if n else None, "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "prob_media": float(np.mean([p.pick_probability for p in sub])) if n else None,
            "log_loss": met.get("log_loss"), "brier": met.get("brier"), "ece": met.get("ece"),
            "muestra_util": n >= MIN_N_UTIL, "modo": "pick" if m == "moneyline" else "proyeccion",
            "pushes": sum(1 for p in rows if p.market_type == m and p.result == "push"),
            "solo_picks": ({"n": len(picks), "aciertos": sum(1 for p in picks if p.correct)} if m == "moneyline" else None),
            "aviso": (None if n >= MIN_N_UTIL else (f"solo {n} evaluadas" if n else "todavia sin partidos evaluados"))}
        if m == "moneyline":
            tot_n += len(picks); tot_h += sum(1 for p in picks if p.correct)
        for p in [p for p in rows if p.market_type == m]:
            out["predicciones"].append({
                "prediction_id": p.id, "game_id": p.game_id, "fecha": str(p.game_date), "inicio": p.start_utc,
                "home": p.home_abbr, "away": p.away_abbr, "market": m, "market_label": label[m],
                "selection": p.selection, "line": p.line, "probabilidad": p.pick_probability,
                "modelo": p.model_probability, "mercado": p.market_probability,
                "riesgo": round((1 - (p.pick_probability or .5)) * 100, 1), "resultado": p.result,
                "acierto": bool(p.correct) if p.correct is not None else None, "modelo_version": p.model_version,
                "version": p.version, "estado": p.status_market})
    ci = _wilson(tot_h, tot_n)
    out["total"] = {"n": tot_n, "aciertos": tot_h, "fallos": tot_n - tot_h, "accuracy": (tot_h / tot_n) if tot_n else None,
                    "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None, "muestra_util": tot_n >= MIN_N_UTIL,
                    "nota": "solo picks de moneyline cuentan en el total; las proyecciones se listan aparte"}
    return out


def tenis_history() -> dict:
    """Historico real de tenis (mercados propios de SPC)."""
    try:
        from sqlalchemy import select as _sel
        from TENIS.markets.db import TenisPrediction, init_db, session_scope as tss
        init_db()
        with tss() as s:
            rows = s.execute(_sel(TenisPrediction).where(TenisPrediction.result.isnot(None),
                                                         TenisPrediction.status != "superseded")
                             .order_by(TenisPrediction.match_date.desc())).scalars().all()
    except Exception as e:                                   # noqa: BLE001
        return {"disponible": False, "motivo": f"TENIS no disponible: {e}"}
    label = {"winner": "Ganador", "total_games": "Total juegos", "handicap_games": "Handicap juegos"}
    out = {"disponible": True, "por_mercado": {}, "predicciones": [],
           "nota": "Predicciones REALES de tenis ya jugadas. Solo el ganador es pick; juegos y handicap son proyecciones."}
    tot_n = tot_h = 0
    for m in ("winner", "total_games", "handicap_games"):
        sub = [p for p in rows if p.market_type == m and p.result != "push"]
        picks = [p for p in sub if p.status_market == "pick"] if m == "winner" else sub
        n = len(sub); hits = sum(1 for p in sub if p.correct)
        ci = _wilson(hits, n)
        met = summary(np.array([1.0 if p.correct else 0.0 for p in sub]),
                      np.array([p.pick_probability for p in sub], dtype=float)) if n else {}
        out["por_mercado"][m] = {
            "market_label": label[m], "n": n, "aciertos": hits, "fallos": n - hits,
            "accuracy": (hits / n) if n else None, "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "prob_media": float(np.mean([p.pick_probability for p in sub])) if n else None,
            "log_loss": met.get("log_loss"), "brier": met.get("brier"), "ece": met.get("ece"),
            "muestra_util": n >= MIN_N_UTIL, "modo": "pick" if m == "winner" else "proyeccion",
            "pushes": sum(1 for p in rows if p.market_type == m and p.result == "push"),
            "aviso": (None if n >= MIN_N_UTIL else (f"solo {n} evaluadas" if n else "todavia sin partidos evaluados"))}
        if m == "winner":
            tot_n += len(picks); tot_h += sum(1 for p in picks if p.correct)
        for p in [p for p in rows if p.market_type == m]:
            out["predicciones"].append({
                "prediction_id": p.id, "game_id": p.event_id, "fecha": str(p.match_date), "inicio": p.start_utc,
                "home": p.p1_name, "away": p.p2_name, "market": m, "market_label": label[m],
                "selection": p.selection, "line": p.line, "probabilidad": p.pick_probability,
                "modelo": p.model_probability, "mercado": p.market_probability,
                "riesgo": round((1 - (p.pick_probability or .5)) * 100, 1), "resultado": p.result,
                "acierto": bool(p.correct) if p.correct is not None else None,
                "modelo_version": p.model_version, "version": p.version, "estado": p.status_market,
                "circuito": p.tour})
    ci = _wilson(tot_h, tot_n)
    out["total"] = {"n": tot_n, "aciertos": tot_h, "fallos": tot_n - tot_h,
                    "accuracy": (tot_h / tot_n) if tot_n else None,
                    "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None, "muestra_util": tot_n >= MIN_N_UTIL,
                    "nota": "solo los picks del ganador cuentan en el total"}
    return out


def soccer_history() -> dict:
    """Historico real de futbol (mercados propios de SPC)."""
    try:
        from sqlalchemy import select as _sel
        from SOCCER.db import SoccerPrediction, init_db, session_scope as sss
        init_db()
        with sss() as s:
            rows = s.execute(_sel(SoccerPrediction).where(
                SoccerPrediction.result.isnot(None),
                SoccerPrediction.record_status == "active")
                .order_by(SoccerPrediction.match_date.desc())).scalars().all()
    except Exception as e:                                   # noqa: BLE001
        return {"disponible": False, "motivo": f"SOCCER no disponible: {e}"}

    label = {
        "double_chance": "Doble oportunidad",
        "total_goals": "Total goles",
        "btts": "Ambos marcan",
        "corners": "Córners"
    }
    out = {"disponible": True, "por_mercado": {}, "predicciones": [],
           "nota": "Predicciones REALES de fútbol ya jugadas."}
    tot_n = tot_h = 0

    for m in ("double_chance", "total_goals", "btts"):
        sub = [p for p in rows if p.market == m and p.result != "push"]
        n = len(sub); hits = sum(1 for p in sub if p.correct)
        ci = _wilson(hits, n)
        met = summary(np.array([1.0 if p.correct else 0.0 for p in sub]),
                      np.array([p.probability for p in sub], dtype=float)) if n else {}
        out["por_mercado"][m] = {
            "market_label": label.get(m, m), "n": n, "aciertos": hits, "fallos": n - hits,
            "accuracy": (hits / n) if n else None, "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "prob_media": float(np.mean([p.probability for p in sub])) if n else None,
            "log_loss": met.get("log_loss"), "brier": met.get("brier"), "ece": met.get("ece"),
            "muestra_util": n >= MIN_N_UTIL, "modo": "proyeccion",
            "pushes": sum(1 for p in rows if p.market == m and p.result == "push"),
            "aviso": (None if n >= MIN_N_UTIL else (f"solo {n} evaluadas" if n else "todavia sin partidos evaluados"))
        }
        tot_n += n; tot_h += hits
        for p in [p for p in rows if p.market == m]:
            out["predicciones"].append({
                "prediction_id": p.id, "game_id": p.event_id, "fecha": str(p.match_date),
                "inicio": str(p.kickoff_utc) if p.kickoff_utc else None,
                "home": p.home_team, "away": p.away_team, "market": m, "market_label": label.get(m, m),
                "selection": p.selection, "line": p.line, "probabilidad": p.probability,
                "modelo": p.probability, "mercado": p.market_probability,
                "riesgo": round((1 - (p.probability or .5)) * 100, 1), "resultado": p.result,
                "acierto": bool(p.correct) if p.correct is not None else None,
                "modelo_version": p.model_version, "version": p.version,
                "liga": p.league_code
            })

    ci = _wilson(tot_h, tot_n)
    out["total"] = {
        "n": tot_n, "aciertos": tot_h, "fallos": tot_n - tot_h,
        "accuracy": (tot_h / tot_n) if tot_n else None,
        "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "muestra_util": tot_n >= MIN_N_UTIL
    }
    return out


def combined(sport: str = "all", date_from: str | None = None,
             date_to: str | None = None) -> dict:
    """Historico por deporte. Cada deporte responde por si mismo; si uno no tiene
    nada evaluado lo dice en su propio bloque en vez de desaparecer."""
    out = {"sport": sport.upper(), "deportes": {}}
    if sport.upper() in ("ALL", "MLB"):
        out["deportes"]["MLB"] = history(date_from, date_to)
    if sport.upper() in ("ALL", "NFL"):
        out["deportes"]["NFL"] = nfl_history()
    if sport.upper() in ("ALL", "NBA"):
        out["deportes"]["NBA"] = nba_history()
    if sport.upper() in ("ALL", "TENIS"):
        out["deportes"]["TENIS"] = tenis_history()
    if sport.upper() in ("ALL", "SOCCER", "FUTBOL", "FÚTBOL"):
        out["deportes"]["SOCCER"] = soccer_history()
    tot_n = tot_h = 0
    for d in out["deportes"].values():
        if d.get("disponible") is False:
            continue
        tot_n += d["total"]["n"]
        tot_h += d["total"]["aciertos"]
    ci = _wilson(tot_h, tot_n)
    out["total"] = {"n": tot_n, "aciertos": tot_h, "fallos": tot_n - tot_h,
                    "accuracy": (tot_h / tot_n) if tot_n else None,
                    "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                    "muestra_util": tot_n >= MIN_N_UTIL}
    return out
