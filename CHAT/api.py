"""API interna del chat. Version CHAT_API_v1.

El chat NO consulta las bases de datos. Pasa por aqui, y aqui se decide que se
puede responder y que no. Esa frontera es deliberada: si el chat pudiera hacer
consultas libres, tarde o temprano leeria una columna de resultado y contestaria
con informacion del futuro sin que nadie se diera cuenta.

Funciones:

    get_prediction(game_id, market)              la prediccion publicada
    get_probability(game_id, market, line)       probabilidad de una linea
    get_distribution(game_id, market)            la distribucion completa
    get_market_comparison(game_id, market, line) modelo vs mercado
    get_explanation(game_id, market)             por que ese numero
    get_secondary_markets(game_id)               que mas se puede preguntar

Todas devuelven SIEMPRE un dict con `disponible`. Cuando es False trae `motivo`
en castellano llano. Ninguna devuelve un numero inventado para rellenar: si no
hay modelo validado, la respuesta correcta es decirlo.
"""
from __future__ import annotations

from functools import lru_cache

from CHAT import registry

VERSION = "CHAT_API_v1"


def _no(motivo, **kw):
    return {"disponible": False, "motivo": motivo, "version": VERSION, **kw}


def _sport_de(market_id: str) -> str:
    return market_id.split(".", 1)[0].upper()


# ---------------------------------------------------------------------------
# Predicciones publicadas
# ---------------------------------------------------------------------------

# Cache corto de las predicciones publicadas. Un resumen de partido consulta
# hasta cinco mercados seguidos y cada consulta releia la tabla entera. TTL bajo
# a proposito: en cuanto un ciclo publica predicciones nuevas se ven al momento.
_TTL_CARDS = 20
_cache_cards: dict = {}


def _cards_publicadas(sport=None):
    import time

    from shared import ledger
    clave = (sport or "all")
    hit = _cache_cards.get(clave)
    if hit and (time.time() - hit[0]) < _TTL_CARDS:
        return hit[1]
    rs = ledger.records(sport=sport, solo_activas=True)
    _cache_cards[clave] = (time.time(), rs)
    return rs


def get_prediction(game_id: str, market: str) -> dict:
    """La prediccion publicada para un partido y un mercado."""
    reg = registry.get(market)
    if reg is None:
        return _no(f"no existe el mercado '{market}' en el registro.",
                   mercados_validos=[d["market_id"] for d in registry.listar()])
    if reg["estado"] == "insufficient_data":
        return _no(f"{reg['etiqueta']}: no hay fuente de datos para este mercado.",
                   market=reg)
    clave = market.split(".", 1)[1]
    for r in _cards_publicadas(reg["sport"]):
        if str(r.get("event_id")) != str(game_id):
            continue
        if str(r.get("market")) not in (clave, market):
            continue
        return {
            "disponible": True, "version": VERSION, "market_id": market,
            "sport": reg["sport"], "etiqueta": reg["etiqueta"],
            "event_id": r["event_id"], "event_time": r["event_time"],
            "selection": r["selection"], "line": r["line"],
            "MODEL_PROBABILITY_RAW": r["model_probability_raw"],
            "MODEL_PROBABILITY_CALIBRATED": r["model_probability_calibrated"],
            "published_probability": r["published_probability"],
            "MARKET_IMPLIED_PROBABILITY": r["market_probability"],
            "MARKET_ODDS": r["odds_decimal"],
            "gap_pp": r["gap_pp"],
            "status": r["status"], "model_version": r["model_version"],
            "calibration_version": r["calibration_version"],
            "prediction_timestamp": r["prediction_timestamp"],
            "registro": reg,
        }
    return _no(f"no hay prediccion publicada de {reg['etiqueta']} para el partido "
               f"{game_id}.", market=reg)


# ---------------------------------------------------------------------------
# Distribuciones (mercados de conteo)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _artefacto(ruta: str):
    import joblib
    from pathlib import Path
    p = Path(ruta)
    return joblib.load(p) if p.exists() else None


def _dist_soccer_cards(event_id: str) -> dict:
    """Distribucion de tarjetas de un fixture de SOCCER."""
    import pandas as pd
    from sqlalchemy import select

    from SOCCER.db import SoccerFixture, session_scope
    from SOCCER.features.build import estado_actual, fila_fixture
    from SOCCER.models import cards
    from shared.paths import SOCCER_MODELS_DIR

    from CHAT import gating
    g = gating.tarjetas()
    if not g.get("permite_publicar_probabilidad"):
        return _no(f"No hay modelo validado de tarjetas. {g['motivo']}",
                   gating=g, market_id="soccer.cards", status="NO PICK")

    art = _artefacto(str(SOCCER_MODELS_DIR / "cards_v1.joblib"))
    if art is None:
        return _no("el modelo de tarjetas no esta entrenado todavia "
                   "(falta cards_v1.joblib). Ejecuta `spc soccer-train`.", gating=g)
    with session_scope() as s:
        fx = s.execute(select(SoccerFixture).where(
            SoccerFixture.event_id == str(event_id))).scalars().first()
        if fx is None:
            return _no(f"no encuentro el partido {event_id} en el calendario de futbol.")
        code, home, away = fx.league_code, fx.home_team, fx.away_team
        fecha, ko = fx.commence_utc.date(), fx.commence_utc
    if code not in cards.LIGAS_CON_DATOS:
        return _no(f"no hay fuente de tarjetas para {code}. Solo tienen amarillas "
                   f"{', '.join(cards.LIGAS_CON_DATOS)}.", liga=code)

    estados, ventaja, res, _ = estado_actual()
    fila = fila_fixture(estados, ventaja, res, code, home, away, fecha, ko)
    X = pd.DataFrame([fila])
    est = art["estimador"]
    lh, la = est.predict(X)
    mk = cards.mercados(lh, la, art["familia"], art["phi"], art["camino"])
    hist = int(fila.get("tarjetas_historial") or 0)
    return {
        "disponible": True, "version": VERSION, "market_id": "soccer.cards",
        "event_id": str(event_id), "partido": f"{home} vs {away}", "liga": code,
        "model_version": "SOCCER_CARDS_v1",
        "expected_cards": round(float(mk["expected_cards"][0]), 3),
        "lambda_home": round(float(lh[0]), 3), "lambda_away": round(float(la[0]), 3),
        "lineas": {k: {kk: round(float(vv[0]), 4) for kk, vv in v.items()}
                   for k, v in mk["totales"].items()},
        "home_al_menos_1": round(float(mk["home_al_menos_1"][0]), 4),
        "away_al_menos_1": round(float(mk["away_al_menos_1"][0]), 4),
        "ambos_reciben": round(float(mk["ambos_reciben"][0]), 4),
        "ambos_reciben_nota": mk["ambos_reciben_nota"],
        "partidos_de_respaldo": hist,
        "advertencia": ("no hay arbitro en los datos, y es el factor conocido mas "
                        "fuerte de este mercado. Trata el numero como orientativo."),
        "status": ("projection" if hist >= 20 else "no_pick"),
        "evidencia": {"holdout_log_loss": (g.get("holdout") or {}).get("log_loss_modelo"),
                      "baseline": (g.get("holdout") or {}).get("log_loss_baseline")},
        "gating": g,
    }


def _dist_mlb_runs(game_id, tramo: str = "completo") -> dict:
    """Distribucion de carreras de un partido de MLB (completo o F5)."""
    import pandas as pd

    from CHAT import gating
    from MLB.engine import distributions as D
    from shared.paths import MLB_MODELS_DIR, MLB_PROCESSED_DIR

    # El gating manda. Este mercado esta medido y NO le gana a su baseline, asi
    # que no publica ninguna probabilidad. Es la regla que pidio el usuario para
    # F5: si no hay modelo validado, se dice, no se estima de otra forma.
    g = gating.carreras(tramo)
    if not g.get("permite_publicar_probabilidad"):
        que = ("las primeras 5 entradas" if tramo == "f5" else "las carreras del partido")
        return _no(
            f"No existe actualmente un modelo validado de {que}. {g['motivo']} "
            f"No voy a darte un porcentaje: repartir el total entre entradas o "
            f"usar una media serian numeros inventados.",
            gating=g, market_id="mlb.f5_total" if tramo == "f5" else "mlb.runs_dist",
            status="NO PICK")

    art = _artefacto(str(MLB_MODELS_DIR / f"runs_{tramo}_v1.joblib"))
    if art is None:
        return _no(f"el gating permite este mercado pero falta el artefacto "
                   f"runs_{tramo}_v1.joblib. Ejecuta el entrenamiento.", gating=g)
    f = MLB_PROCESSED_DIR / "features.parquet"
    if not f.exists():
        return _no("no hay features de MLB construidas.")
    X = pd.read_parquet(f)
    fila = X[X["game_id"].astype(str) == str(game_id)]
    if fila.empty:
        return _no(f"no encuentro el partido {game_id} en las features de MLB.")
    est = art["estimador"]
    lh, la = est.predict(fila)
    mk = D.mercados(lh, la, tramo, art["familia"], art["phi"], art["camino"])
    r = fila.iloc[0]
    out = {
        "disponible": True, "version": VERSION,
        "market_id": "mlb.f5_total" if tramo == "f5" else "mlb.runs_dist",
        "game_id": str(game_id), "tramo": tramo,
        "partido": f"{r.get('away_abbr')} @ {r.get('home_abbr')}",
        "model_version": art.get("model_version", "MLB_RUNS_DIST_v1"),
        "expected_runs": round(float(mk["expected_runs"][0]), 3),
        "home_expected": round(float(lh[0]), 3),
        "away_expected": round(float(la[0]), 3),
        "lineas": {k: {kk: round(float(vv[0]), 4) for kk, vv in v.items()}
                   for k, v in mk["game_total"].items()},
        "team_total": {lado: {k: round(float(v[0]), 4) for k, v in d.items()}
                       for lado, d in mk["team_total"].items()},
        "home_cero": round(float(mk["home_cero"][0]), 4),
        "away_cero": round(float(mk["away_cero"][0]), 4),
        "ambos_anotan": round(float(mk["ambos_anotan"][0]), 4),
        "alguna_carrera": round(float(mk["alguna_carrera"][0]), 4),
        "pmf_home": [round(float(v), 5) for v in mk["pmf_home"][0][:13]],
        "pmf_away": [round(float(v), 5) for v in mk["pmf_away"][0][:13]],
        "nota_independencia": mk["nota_independencia"],
        "evidencia": art.get("evidencia"),
        "status": art.get("status", "projection"),
    }
    return out


DISTRIBUCIONES = {
    "soccer.cards": lambda gid: _dist_soccer_cards(gid),
    "mlb.runs_dist": lambda gid: _dist_mlb_runs(gid, "completo"),
    "mlb.f5_total": lambda gid: _dist_mlb_runs(gid, "f5"),
}


def get_distribution(game_id: str, market: str) -> dict:
    reg = registry.get(market)
    if reg is None:
        return _no(f"no existe el mercado '{market}'.")
    if reg["distribution_type"] == "binaria":
        return _no(f"{reg['etiqueta']} es un mercado binario: no tiene distribucion "
                   f"de conteo, solo una probabilidad.", market=reg)
    fn = DISTRIBUCIONES.get(market)
    if fn is None:
        return _no(f"no tengo un motor de distribucion conectado para "
                   f"{reg['etiqueta']} todavia.", market=reg)
    try:
        return fn(game_id)
    except Exception as e:                                    # noqa: BLE001
        return _no(f"no pude calcular la distribucion: {type(e).__name__}: {e}")


def get_probability(game_id: str, market: str, line=None, lado: str = "over") -> dict:
    """Probabilidad de una linea concreta. Usa la distribucion si el mercado la tiene."""
    reg = registry.get(market)
    if reg is None:
        return _no(f"no existe el mercado '{market}'.")
    if reg["distribution_type"] != "binaria" and line is not None:
        d = get_distribution(game_id, market)
        if not d.get("disponible"):
            return d
        clave = f"{float(line):g}"
        if clave not in d.get("lineas", {}):
            disp = sorted(d.get("lineas", {}), key=float)
            return _no(f"no tengo calculada la linea {clave}. Disponibles: "
                       f"{', '.join(disp)}.", lineas_disponibles=disp)
        v = d["lineas"][clave]
        return {"disponible": True, "version": VERSION, "market_id": market,
                "line": float(line), "lado": lado,
                "probabilidad": v.get(lado), "detalle": v,
                "model_version": d.get("model_version"),
                "status": d.get("status"), "advertencia": d.get("advertencia")}
    p = get_prediction(game_id, market)
    if not p.get("disponible"):
        return p
    return {"disponible": True, "version": VERSION, "market_id": market,
            "line": p.get("line"), "selection": p.get("selection"),
            "probabilidad": p.get("MODEL_PROBABILITY_CALIBRATED"),
            "status": p.get("status"), "model_version": p.get("model_version")}


# ---------------------------------------------------------------------------
# Modelo vs mercado
# ---------------------------------------------------------------------------

def get_market_comparison(game_id: str, market: str, line=None) -> dict:
    from shared import discrepancy, tracking
    p = get_prediction(game_id, market)
    if not p.get("disponible"):
        return p
    sport = p["sport"]
    ev = None
    try:
        bt = tracking.discrepancias(sport)
        ev = bt
    except Exception:                                          # noqa: BLE001
        ev = None
    r = discrepancy.evaluar(
        model_raw=p["MODEL_PROBABILITY_RAW"],
        model_cal=p["MODEL_PROBABILITY_CALIBRATED"],
        market=p["MARKET_IMPLIED_PROBABILITY"],
        odds_decimal=p["MARKET_ODDS"], market_source=None,
        event_time=p["event_time"], model_version=p["model_version"],
        evidencia=ev)
    r["disponible"] = True
    r["market_id"] = market
    r["etiqueta"] = p["etiqueta"]
    r["selection"] = p["selection"]
    return r


# ---------------------------------------------------------------------------
# Explicabilidad
# ---------------------------------------------------------------------------

def get_explanation(game_id: str, market: str) -> dict:
    """Por que ese numero, con los factores REALES guardados con la prediccion.

    Si la prediccion no guardo su explicacion, se dice. Redactar una explicacion
    plausible a posteriori seria inventarla: sonaria bien y no tendria ninguna
    relacion con lo que el modelo uso.
    """
    import json

    reg = registry.get(market)
    if reg is None:
        return _no(f"no existe el mercado '{market}'.")
    bruto = _explicacion_guardada(reg["sport"], game_id, market.split(".", 1)[1])
    if bruto is None:
        return _no("esta prediccion no guardo su explicacion, y no voy a redactar "
                   "una a posteriori: sonaria convincente sin corresponder a lo "
                   "que el modelo uso de verdad.")
    try:
        factores = json.loads(bruto) if isinstance(bruto, str) else bruto
    except (ValueError, TypeError):
        factores = {"texto": str(bruto)}
    return {"disponible": True, "version": VERSION, "market_id": market,
            "factores": factores, "fuente": "explicacion guardada con la prediccion"}


def _explicacion_guardada(sport, game_id, market):
    import sqlite3

    from shared.ledger import DBS
    q = {
        "SOCCER": ("select explanation from predictions where event_id=? and market=? "
                   "and record_status='active' limit 1"),
        "NFL": ("select explanation from nfl_market_predictions where game_id=? "
                "and market_type=? and status='published' limit 1"),
        "NBA": ("select explanation from nba_predictions where game_id=? "
                "and market_type=? and status='published' limit 1"),
        "TENIS": ("select explanation from tenis_predictions where event_id=? "
                  "and market_type=? and status='published' limit 1"),
        "MLB": ("select upset_explanation from predictions where game_id=? "
                "and market=? and status!='superseded' limit 1"),
    }.get(sport)
    if not q:
        return None
    p = DBS.get(sport)
    if not p or not p.exists():
        return None
    c = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    try:
        row = c.execute(q, (str(game_id), market)).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        c.close()
    return row[0] if row and row[0] else None


# ---------------------------------------------------------------------------
# Mercados secundarios
# ---------------------------------------------------------------------------

def get_secondary_markets(game_id: str, sport: str | None = None) -> dict:
    """Que mas se puede preguntar de este partido. SOLO por chat, nunca en Top Picks."""
    if sport is None:
        for r in _cards_publicadas():
            if str(r.get("event_id")) == str(game_id):
                sport = r["sport"]
                break
    if sport is None:
        return _no(f"no se de que deporte es el partido {game_id}.")
    out = []
    for d in registry.secundarios(sport):
        est = get_distribution(game_id, d["market_id"])
        out.append({"market_id": d["market_id"], "etiqueta": d["etiqueta"],
                    "pregunta_tipo": d["pregunta_tipo"],
                    "disponible": bool(est.get("disponible")),
                    "motivo": est.get("motivo"),
                    "limitaciones": d["limitaciones"]})
    return {"disponible": True, "version": VERSION, "sport": sport,
            "game_id": str(game_id), "mercados": out,
            "nota": "estos mercados no aparecen en Top Picks; solo se consultan aqui."}


def catalogo(sport: str | None = None) -> dict:
    """Todo lo que el chat puede contestar, con su estado MEDIDO.

    El estado no viene del registro sino del gating, que lo recalcula leyendo
    los walk-forward en disco. Asi, si un modelo deja de ganarle a su baseline,
    el catalogo lo refleja sin que nadie tenga que acordarse de editarlo.
    """
    from CHAT import gating
    registry.refrescar_desde_gating()
    g = gating.aplicar_al_registro()
    return {"version": VERSION,
            "primarios": registry.listar(sport, "primary"),
            "secundarios": registry.listar(sport, "secondary"),
            "gating_secundarios": g["gating"]}
