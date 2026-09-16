"""Calificacion de combinadas y su historico.

Cada pata se resuelve leyendo el resultado que ya guardo su propio deporte (solo
lectura). Una combinada gana si TODAS sus patas ganan; si alguna es push, esa
pata se anula (como en las casas) y la combinada se juzga con las restantes.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import select

from PARLAY.engine.db import Parlay, ParlayLeg, init_db, session_scope
from shared.timeutil import utcnow


def _leg_result(sport: str, game_id: str, prediction_id, market_key: str, selection: str):
    """(result, correct) de la pata desde la base del deporte. None si sigue pendiente."""
    try:
        if sport == "MLB":
            from MLB.database.models import Prediction
            from MLB.database.session import session_scope as ss
            with ss() as s:
                q = select(Prediction).where(Prediction.game_id == int(game_id),
                                             Prediction.market == market_key,
                                             Prediction.selection == selection)
                p = s.execute(q.order_by(Prediction.version.desc())).scalars().first()
                return (p.result, p.correct) if p and p.result else (None, None)
        if sport == "NBA":
            from NBA.markets.db import NbaPrediction, session_scope as ss
            with ss() as s:
                p = s.execute(select(NbaPrediction).where(NbaPrediction.game_id == str(game_id),
                                                          NbaPrediction.market_type == market_key)
                              .order_by(NbaPrediction.version.desc())).scalars().first()
                return (p.result, p.correct) if p and p.result else (None, None)
        if sport == "TENIS":
            from TENIS.markets.db import TenisPrediction, session_scope as ss
            with ss() as s:
                p = s.execute(select(TenisPrediction).where(TenisPrediction.event_id == str(game_id),
                                                            TenisPrediction.market_type == market_key)
                              .order_by(TenisPrediction.version.desc())).scalars().first()
                return (p.result, p.correct) if p and p.result else (None, None)
        if sport == "NFL":
            if market_key in ("total", "spread"):
                from NFL.markets.db import NflMarketPrediction, session_scope as ss
                with ss() as s:
                    p = s.execute(select(NflMarketPrediction).where(
                        NflMarketPrediction.game_id == str(game_id),
                        NflMarketPrediction.market_type == market_key)
                        .order_by(NflMarketPrediction.version.desc())).scalars().first()
                    return (p.result, p.correct) if p and p.result else (None, None)
            import NFL.adapter as nfl                       # moneyline: motor original, solo lectura
            with nfl._conn() as c:
                r = c.execute("select correct from predictions where game_id=? and is_active=1",
                              (str(game_id),)).fetchone()
            if r is None or r["correct"] is None:
                return (None, None)
            ok = bool(r["correct"])
            return ("win" if ok else "loss", ok)
    except Exception:                                       # noqa: BLE001
        return (None, None)
    return (None, None)


def score(session=None) -> dict:
    """Califica las combinadas abiertas cuyas patas ya tienen resultado."""
    init_db()
    cerradas = actualizadas = 0
    with session_scope() as s:
        abiertas = s.execute(select(Parlay).where(Parlay.status == "open")).scalars().all()
        for p in abiertas:
            legs = s.execute(select(ParlayLeg).where(ParlayLeg.parlay_id == p.id)).scalars().all()
            for l in legs:
                if l.result:
                    continue
                res, ok = _leg_result(l.sport, l.game_id, l.prediction_id, l.market_key, l.selection)
                if res:
                    l.result, l.correct = res, ok
                    actualizadas += 1
            pendientes = [l for l in legs if not l.result]
            perdidas = [l for l in legs if l.result == "loss"]
            validas = [l for l in legs if l.result in ("win", "loss")]
            p.legs_won = sum(1 for l in legs if l.result == "win")
            if perdidas:                                    # una sola pata perdida cierra la combinada
                p.status, p.result, p.graded_at = "graded", "loss", utcnow(); cerradas += 1
            elif not pendientes:
                if not validas:                             # todo push: se anula
                    p.status, p.result, p.graded_at = "graded", "push", utcnow()
                else:
                    p.status, p.result, p.graded_at = "graded", "win", utcnow()
                cerradas += 1
    return {"patas_resueltas": actualizadas, "combinadas_cerradas": cerradas}


def _wilson(h, n):
    if not n:
        return None
    z = 1.96
    p = h / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def history(limit: int = 400) -> dict:
    """Historico de combinadas ya calificadas + las abiertas."""
    init_db()
    with session_scope() as s:
        parlays = s.execute(select(Parlay).order_by(Parlay.created_at.desc()).limit(limit)).scalars().all()
        legs = {}
        for l in s.execute(select(ParlayLeg)).scalars().all():
            legs.setdefault(l.parlay_id, []).append(l)
    def dump(p):
        return {"id": p.id, "size": p.size, "sports": p.sports, "mixed": bool(p.mixed),
                "fecha": str(p.match_date) if p.match_date else None,
                "creada": str(p.created_at)[:19], "probabilidad": p.probability,
                "cuota_justa": p.fair_odds, "prob_mercado": p.market_probability,
                "cuota_mercado": p.market_odds, "estado": p.status, "resultado": p.result,
                "patas_ganadas": p.legs_won, "patas": p.legs_total,
                "legs": [{"sport": l.sport, "teams": l.teams, "market": l.market, "selection": l.selection,
                          "probabilidad": l.probability, "resultado": l.result,
                          "inicio": str(l.start_utc) if l.start_utc else None}
                         for l in sorted(legs.get(p.id, []), key=lambda x: (x.start_utc or ""))]}
    cerradas = [p for p in parlays if p.status == "graded" and p.result in ("win", "loss")]
    por_tam = {}
    for size in (2, 3):
        sub = [p for p in cerradas if p.size == size]
        n = len(sub); h = sum(1 for p in sub if p.result == "win")
        ci = _wilson(h, n)
        prom = float(np.mean([p.probability for p in sub])) if n else None
        por_tam[size] = {"n": n, "aciertos": h, "fallos": n - h, "accuracy": (h / n) if n else None,
                         "prob_media_anunciada": prom,
                         "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None}
    n = len(cerradas); h = sum(1 for p in cerradas if p.result == "win")
    ci = _wilson(h, n)
    return {"total": {"n": n, "aciertos": h, "fallos": n - h, "accuracy": (h / n) if n else None,
                      "ic95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                      "prob_media_anunciada": float(np.mean([p.probability for p in cerradas])) if n else None},
            "por_tamano": por_tam,
            "abiertas": [dump(p) for p in parlays if p.status == "open"],
            "calificadas": [dump(p) for p in cerradas],
            "nota": ("Una combinada gana solo si TODAS sus patas ganan. La probabilidad anunciada es el "
                     "producto de las probabilidades calibradas de cada pata (patas de partidos y equipos "
                     "distintos). Si la muestra es pequena, comparar el acierto real con la probabilidad "
                     "media anunciada no significa nada todavia.")}
