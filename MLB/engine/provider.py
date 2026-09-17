"""Lectura de predicciones MLB en el formato unico de tarjeta.

En la Fase 1 la base MLB existe pero esta vacia: este modulo devuelve una lista
vacia y un estado que dice exactamente eso. No fabrica partidos ni numeros.
"""
from __future__ import annotations

from sqlalchemy import func, select

from MLB.database.models import DataSourceLog, Game, ModelVersion, Prediction, Team
from MLB.database.session import init_db, session_scope
from shared.schema import empty_card

_GATE = None


def _gate() -> dict:
    global _GATE
    if _GATE is None:
        from MLB.engine.gating import decide
        _GATE = decide()
    return _GATE


from MLB.engine.markets import LABEL as MARKET_LABEL, HIDDEN as HIDDEN_MARKETS  # noqa: E402
_NAMES: dict[str, str] = {}


def team_names() -> dict:
    global _NAMES
    if not _NAMES:
        init_db()
        with session_scope() as s:
            _NAMES = {t.abbr: t.name for t in s.execute(select(Team)).scalars() if t.abbr}
    return _NAMES


def _card(p: Prediction, g: Game | None) -> dict:
    c = empty_card(
        sport="MLB", game_id=p.game_id, game_date=p.game_date, start_utc=p.game_start_utc,
        status=(g.status if g else None),
        home=p.home_abbr, away=p.away_abbr,
        home_name=team_names().get(p.home_abbr or ""),
        away_name=team_names().get(p.away_abbr or ""),
        venue=(g.venue_name if g else None),
        market=MARKET_LABEL.get(p.market, p.market), selection=p.selection, line=p.line,
        model_probability=p.model_probability, market_probability=p.market_probability,
        ensemble_probability=p.ensemble_probability, elo_probability=p.elo_probability,
        confidence=p.confidence, upset_risk=p.upset_risk,
        upset_label=p.upset_label,
        upset_explanation=p.upset_explanation,
        model_market_gap=p.model_market_gap,
        directional_disagreement=p.directional_disagreement,
        agreement_bucket=p.agreement_bucket, model_dispersion=p.model_dispersion,
        market_signal=p.market_signal, blend_policy=p.blend_policy,
        sources=p.sources or [], model_version=p.model_version,
        prediction_timestamp=p.prediction_timestamp, prediction_id=p.id, version=p.version,
        status_note=p.status_note, result=p.result, correct=p.correct,
    )
    c["data_completeness"] = p.data_completeness or {"complete": True, "missing": [], "notes": []}
    g8 = _gate().get(p.market, {})
    base = g8.get("base_rate")
    if base is not None and p.ensemble_probability is not None:
        # tasa base DE LA SELECCION: si el pick es el visitante, es 1 - base
        home_side = (p.selection or "").startswith(p.home_abbr or "\x00")
        base_sel = base if home_side else 1 - base
        c["extra_edge"] = float(p.ensemble_probability) - base_sel
    c["extra"] = {"season": p.season, "status": p.status, "market_key": p.market,
                  "base_rate_selection": (base if base is None else
                                          (base if (p.selection or "").startswith(p.home_abbr or "\x00")
                                           else 1 - base)),
                  "edge_vs_base": c.get("extra_edge"),
                  "publish_pick": g8.get("publish_pick"),
                  "market_caveat": (None if g8.get("publish_pick") else
                                    "Over/Under solo informativo: en las pruebas el modelo no "
                                    "proyecto carreras mejor que la media historica"
                                    if p.market == "total" else
                                    "Solo probabilidad, no pick: el modelo acierta un poco mas "
                                    "que la media, pero no lo suficiente para asegurarlo"),
                  "total": (p.upset_explanation or {}).get("total"),
                  "flags": (p.upset_explanation or {}).get("flags", []),
                  "risk_meaning": "probabilidad estimada de que el pick falle"}
    # Factor contextual: si hay ajuste de clima o lineup, se anade a la tarjeta
    try:
        from shared.contextual import get_adjustments
        adj = get_adjustments(p.game_id, p.market)
        if adj:
            c["extra"]["contextual_adjustments"] = adj
    except Exception:
        pass
    return c


def games(date_from: str, date_to: str, limit: int = 600) -> list[dict]:
    init_db()
    with session_scope() as s:
        rows = s.execute(
            select(Prediction, Game).outerjoin(Game, Game.id == Prediction.game_id)
            .where(Prediction.game_date >= date_from, Prediction.game_date <= date_to,
                   Prediction.status != "superseded",
                   Prediction.market.notin_(list(HIDDEN_MARKETS)))
            .order_by(Prediction.game_start_utc.asc()).limit(limit)).all()
        return [_card(p, g) for p, g in rows]


def status() -> dict:
    """Estado honesto del motor MLB: que hay y que falta."""
    init_db()
    with session_scope() as s:
        n_games = s.scalar(select(func.count()).select_from(Game)) or 0
        n_preds = s.scalar(select(func.count()).select_from(Prediction)) or 0
        n_models = s.scalar(select(func.count()).select_from(ModelVersion)) or 0
        last_fetch = s.scalar(select(func.max(DataSourceLog.retrieved_at)))
        srcs = s.execute(
            select(DataSourceLog.source, func.max(DataSourceLog.retrieved_at),
                   func.count()).group_by(DataSourceLog.source)).all()
    ready = n_games > 0 and n_models > 0
    missing = []
    if n_games == 0:
        missing.append("no se ha ingerido ningun partido (MLB Stats API)")
    if n_models == 0:
        missing.append("no existe ningun modelo MLB entrenado ni validado")
    if n_preds == 0:
        missing.append("no hay predicciones MLB generadas")
    return {"sport": "MLB", "ready": ready, "games": n_games, "predictions": n_preds,
            "models": n_models, "last_fetch": last_fetch, "missing": missing,
            "sources": [{"source": a, "last_fetch": b, "calls": c} for a, b, c in srcs],
            "phase": "Fase 1: esquema creado, ingesta pendiente"}


def markets_status() -> dict:
    """Que mercados publica el sistema y por que, con los numeros del backtest."""
    from MLB.engine.gating import decide
    return decide()


def alerts(date_from: str, date_to: str) -> list[dict]:
    """Alertas derivadas de datos reales, no de reglas inventadas.

    Se agrupan POR PARTIDO: si al mismo partido le faltan las cuotas, esa alerta
    aparece una vez, no una por cada mercado. Antes se repetia tres veces y la
    lista era ilegible.
    """
    porjuego: dict = {}
    for c in games(date_from, date_to):
        for f in (c["extra"] or {}).get("flags", []):
            k = (c["game_id"], f["code"], f["text"])
            if k in porjuego:
                porjuego[k]["mercados"].append(c["market"])
                continue
            porjuego[k] = {
                "sport": "MLB", "game_id": c["game_id"],
                "game": f"{c['away']} en {c['home']}", "date": c["game_date"],
                "hora": c.get("start_utc"), "mercados": [c["market"]],
                "code": f["code"], "level": f["level"], "text": f["text"]}
    out = list(porjuego.values())
    for a in out:
        a["market"] = ", ".join(dict.fromkeys(a.pop("mercados")))
        a["selection"] = ""
    return out


def performance() -> dict:
    init_db()
    with session_scope() as s:
        n = s.scalar(select(func.count()).select_from(Prediction)
                     .where(Prediction.correct.isnot(None))) or 0
        hits = s.scalar(select(func.count()).select_from(Prediction)
                        .where(Prediction.correct.is_(True))) or 0
    import json as _json

    from shared.paths import MLB_BACKTEST_DIR
    bt = None
    f = MLB_BACKTEST_DIR / "walkforward_v1.json"
    if f.exists():
        d = _json.loads(f.read_text())
        strategies = {}
        for mkt, m in d["summary"].items():
            if mkt == "total" or "strategies" not in m:
                continue
            for st, v in m["strategies"].items():
                strategies[f"{mkt} · {st}"] = v
        bt = {"id": None, "label": "walkforward_v1",
              "created_at": d.get("created_at"),
              "summary": {"strategies": strategies},
              "market_comparison": d.get("market_comparison")}
    prod = None
    with session_scope() as s:
        m = s.execute(select(ModelVersion).where(ModelVersion.is_production.is_(True))
                      .order_by(ModelVersion.id.desc())).scalars().first()
        if m:
            prod = {"name": m.name, "version": m.version, "created_at": m.created_at,
                    "metrics": {}}
    base = {"latest_backtest": bt, "production_model": prod,
            "note": ("Las cifras del backtest son walk-forward: cada temporada de prueba "
                     "nunca se uso para entrenar ni para elegir nada.")}
    if n == 0:
        return {**base, "available": bool(bt),
                "reason": "todavia no hay predicciones MLB evaluadas en vivo",
                "graded_predictions": 0, "accuracy": None}
    return {**base, "available": True, "graded_predictions": n, "accuracy": hits / n}
