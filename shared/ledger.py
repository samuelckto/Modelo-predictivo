"""Lectura unificada de TODAS las predicciones del centro, en solo lectura.

Cada deporte guarda sus predicciones en su propia base con su propio esquema.
Este modulo las traduce a un registro comun para poder medir calibracion,
discrepancia y ROI con el mismo codigo en los cinco. No escribe nunca: abre
cada SQLite con `mode=ro` en la URI, asi que el aislamiento entre deportes no
depende de la buena voluntad del que llame.

La distincion mas importante que hace este modulo:

    model_probability      lo que cree el MODELO, sin mercado dentro.
    ensemble/final         lo que se publica DESPUES de mezclar con el mercado.

Son columnas distintas y aqui no se confunden. La discrepancia modelo-mercado
se calcula siempre contra la primera. Si se usara la segunda, el mercado
estaria en los dos lados de la resta y el gap se encogeria solo, que es
exactamente la forma silenciosa de que el sistema termine siguiendo al casino.

Las cuotas de MLB y NFL no viven en la fila de la prediccion sino en su tabla
de cuotas. Se unen con corte point-in-time: solo cuotas con
`available_at <= prediction_timestamp`. Una cuota posterior a la prediccion
seria informacion del futuro.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from shared.paths import (MLB_DB_DIR, NBA_DB_DIR, NFL_MARKETS_DB_DIR, SOCCER_DB_DIR,
                          TENIS_DB_DIR)

VERSION = "LEDGER_v1"

DBS = {
    "MLB": MLB_DB_DIR / "mlb.sqlite3",
    "NFL": NFL_MARKETS_DB_DIR / "nfl_markets.sqlite3",
    "NBA": NBA_DB_DIR / "nba_markets.sqlite3",
    "TENIS": TENIS_DB_DIR / "tenis_markets.sqlite3",
    "SOCCER": SOCCER_DB_DIR / "soccer_markets.sqlite3",
}

CAMPOS = (
    "sport", "league", "season", "event_id", "event_time", "market", "selection", "line",
    "model_probability_raw", "model_probability_calibrated", "published_probability",
    "market_probability", "gap_pp", "odds_decimal", "odds_american",
    "market_source", "odds_fetched_at", "status", "model_version",
    "calibration_version", "prediction_timestamp", "result", "correct",
)


def _conn(path: Path):
    if not Path(path).exists():
        return None
    try:
        return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None


# Errores de consulta de la ultima lectura. NO se tragan en silencio: una
# consulta rota devuelve cero filas y eso es indistinguible de "no hay datos".
# Asi fue como `nfl_odds` parecio vacia durante un rato: la consulta pedia una
# columna `price_decimal` que en esa tabla no existe.
ERRORES: list[str] = []


def _rows(c, sql, args=()):
    try:
        cur = c.execute(sql, args)
    except sqlite3.Error as e:
        ERRORES.append(f"{type(e).__name__}: {e} | {sql[:90]}")
        return []
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _dec(american=None, decimal=None):
    from shared.roi import to_american, to_decimal
    d = to_decimal(american=american, decimal=decimal)
    return d, (to_american(d) if d else None)


def _base(**kw):
    r = {k: None for k in CAMPOS}
    r.update(kw)
    if r.get("market_probability") is not None and r.get("model_probability_calibrated") is not None:
        r["gap_pp"] = round((r["model_probability_calibrated"] - r["market_probability"]) * 100, 4)
    return r


# --------------------------------------------------------------------------

def _misma_linea(o, p) -> bool:
    """¿La cuota es de la MISMA linea que la prediccion?

    Es la comprobacion que faltaba y que producia los desacuerdos mas grandes
    del panel. Caso real, partido 824063 (KC local, AZ visitante): la prediccion
    era 'AZ +1.5' pero el mercado solo tenia AZ -1.5 y KC +1.5. Comparando sin
    mirar la linea salia una diferencia de +30 pp contra un precio que NO existe.

    Un mercado sin linea (moneyline) no tiene nada que comparar y pasa siempre.
    """
    lp, lo = p.get("line"), o.get("line")
    if lp is None and lo is None:
        return True
    if lp is None or lo is None:
        return False
    return abs(float(lp) - float(lo)) < 1e-6


def _lado_mlb(p) -> str | None:
    """Traduce la seleccion de MLB al lado que usa la tabla de cuotas."""
    sel = str(p.get("selection") or "")
    s = sel.lower()
    if p.get("market") == "total":
        return "over" if "over" in s else "under" if "under" in s else None
    h, a = str(p.get("home_abbr") or ""), str(p.get("away_abbr") or "")
    # Se compara por token exacto: 'CHC' no debe casar dentro de otra cadena.
    tokens = set(sel.replace("(", " ").replace(")", " ").split())
    if h and h in tokens:
        return "home"
    if a and a in tokens:
        return "away"
    return None


def _mlb(solo_activas=True):
    c = _conn(DBS["MLB"])
    if not c:
        return []
    filtro = "where status != 'superseded'" if solo_activas else ""
    preds = _rows(c, f"select * from predictions {filtro}")
    # Cuotas point-in-time: la mejor disponible ANTES de emitir la prediccion.
    odds = _rows(c, "select game_id, market, selection, line, price_decimal, "
                    "price_american, available_at, source, bookmaker from odds")
    idx = {}
    for o in odds:
        idx.setdefault((o["game_id"], o["market"]), []).append(o)
    out = []
    for p in preds:
        # La tabla de cuotas etiqueta los lados home/away/over/under; la prediccion
        # escribe el nombre del equipo ('BOS') o 'Over 8.5'. Sin esta traduccion el
        # cruce daba 0 filas y el ROI parecia imposible cuando solo estaba mal unido.
        lado = _lado_mlb(p)
        cands = [o for o in idx.get((p["game_id"], p["market"]), [])
                 if lado and str(o.get("selection") or "").lower() == lado
                 and _misma_linea(o, p)
                 and (o.get("available_at") or "") <= (p.get("prediction_timestamp") or "9999")]
        o = max(cands, key=lambda x: x.get("available_at") or "") if cands else None
        d, a = _dec(american=(o or {}).get("price_american"),
                    decimal=(o or {}).get("price_decimal"))
        # La probabilidad de mercado se acepta SOLO si existe una cuota de la
        # misma linea y el mismo lado. El motor de MLB guarda su propio consenso
        # sin filtrar por linea, y cuando la linea publicada no coincide con la
        # de la prediccion ese numero no corresponde a ninguna apuesta real.
        mp = p.get("market_probability") if o is not None else None
        out.append(_base(
            sport="MLB", league="MLB", season=p.get("season"), event_id=str(p.get("game_id")),
            event_time=p.get("game_start_utc") or p.get("game_date"),
            market=p.get("market"), selection=p.get("selection"), line=p.get("line"),
            model_probability_raw=p.get("model_probability"),
            model_probability_calibrated=p.get("model_probability"),
            published_probability=p.get("ensemble_probability"),
            market_probability=mp,
            odds_decimal=d, odds_american=a,
            market_source=(o or {}).get("source"), odds_fetched_at=(o or {}).get("available_at"),
            status=p.get("status"), model_version=p.get("model_version"),
            prediction_timestamp=p.get("prediction_timestamp"),
            result=p.get("result"), correct=p.get("correct")))
    c.close()
    return out


def _nfl(solo_activas=True):
    c = _conn(DBS["NFL"])
    if not c:
        return []
    filtro = "where status = 'published'" if solo_activas else ""
    preds = _rows(c, f"select * from nfl_market_predictions {filtro}")
    # nfl_odds solo guarda precio americano (no hay columna price_decimal).
    odds = _rows(c, "select game_id, market, selection, line, "
                    "price_american, available_at, source from nfl_odds")
    idx = {}
    for o in odds:
        idx.setdefault((o["game_id"], o["market"]), []).append(o)
    out = []
    for p in preds:
        m = p.get("market_type")
        lado = _lado_mlb({"market": "total" if m == "total" else m,
                          "selection": p.get("selection"),
                          "home_abbr": p.get("home_team"), "away_abbr": p.get("away_team")})
        cands = [o for o in idx.get((p["game_id"], m), [])
                 if lado and str(o.get("selection") or "").lower() == lado
                 and _misma_linea(o, p)
                 and (o.get("available_at") or "") <= (p.get("prediction_timestamp") or "9999")]
        o = max(cands, key=lambda x: x.get("available_at") or "") if cands else None
        d, a = _dec(american=(o or {}).get("price_american"),
                    decimal=(o or {}).get("price_decimal"))
        out.append(_base(
            sport="NFL", league="NFL", season=p.get("season"), event_id=str(p.get("game_id")),
            event_time=p.get("kickoff_utc"), market=m, selection=p.get("selection"),
            line=p.get("line"),
            model_probability_raw=p.get("p_over_raw"),
            model_probability_calibrated=p.get("model_probability"),
            published_probability=p.get("final_probability"),
            market_probability=(p.get("market_probability") if o is not None else None),
            odds_decimal=d, odds_american=a,
            market_source=(o or {}).get("source"), odds_fetched_at=(o or {}).get("available_at"),
            status=p.get("pick"), model_version=p.get("model_version"),
            prediction_timestamp=p.get("prediction_timestamp"),
            result=p.get("result"), correct=p.get("correct")))
    c.close()
    return out


def _dos_vias(sport, db_key, tabla, id_col, liga_col, fecha_col, odds_a, odds_b,
              solo_activas=True):
    """NBA y TENIS comparten esquema: dos precios en la propia fila."""
    c = _conn(DBS[db_key])
    if not c:
        return []
    filtro = "where status = 'published'" if solo_activas else ""
    out = []
    for p in _rows(c, f"select * from {tabla} {filtro}"):
        sel = str(p.get("selection") or "")
        # El precio que corresponde al lado elegido; si no se distingue, ninguno.
        pa, pb = p.get(odds_a), p.get(odds_b)
        precio = None
        if sport == "NBA":
            h, a = str(p.get("home_abbr") or ""), str(p.get("away_abbr") or "")
            precio = pa if (h and h in sel) else pb if (a and a in sel) else None
        else:
            p1, p2 = str(p.get("p1_name") or ""), str(p.get("p2_name") or "")
            precio = pa if (p1 and p1 in sel) else pb if (p2 and p2 in sel) else None
        d, am = _dec(american=precio)
        out.append(_base(
            sport=sport, league=(p.get(liga_col) or sport), season=p.get("season"),
            event_id=str(p.get(id_col)), event_time=p.get("start_utc") or p.get(fecha_col),
            market=p.get("market_type"), selection=sel, line=p.get("line"),
            model_probability_raw=p.get("p_raw"),
            model_probability_calibrated=p.get("model_probability"),
            published_probability=p.get("final_probability"),
            market_probability=p.get("market_probability"),
            odds_decimal=d, odds_american=am,
            market_source=p.get("line_source"), odds_fetched_at=p.get("line_timestamp"),
            status=p.get("status_market"), model_version=p.get("model_version"),
            calibration_version=p.get("calibration_version"),
            prediction_timestamp=p.get("prediction_timestamp"),
            result=p.get("result"), correct=p.get("correct")))
    c.close()
    return out


def _soccer(solo_activas=True):
    c = _conn(DBS["SOCCER"])
    if not c:
        return []
    filtro = "where record_status = 'active'" if solo_activas else ""
    out = []
    for p in _rows(c, f"select * from predictions {filtro}"):
        d, a = _dec(decimal=p.get("odds_price"))
        out.append(_base(
            sport="SOCCER", league=p.get("league_code"), season=None,
            event_id=p.get("event_id"), event_time=p.get("kickoff_utc") or p.get("match_date"),
            market=p.get("market"), selection=p.get("selection"), line=p.get("line"),
            model_probability_raw=p.get("probability"),
            model_probability_calibrated=p.get("probability"),
            published_probability=p.get("probability"),
            market_probability=p.get("market_probability"),
            odds_decimal=d, odds_american=a,
            market_source=p.get("bookmaker"), odds_fetched_at=p.get("odds_timestamp"),
            status=p.get("status"), model_version=p.get("model_version"),
            calibration_version=p.get("calibration_version"),
            prediction_timestamp=p.get("prediction_timestamp"),
            result=p.get("result"), correct=p.get("correct")))
    c.close()
    return out


LECTORES = {
    "MLB": _mlb,
    "NFL": _nfl,
    "NBA": lambda a=True: _dos_vias("NBA", "NBA", "nba_predictions", "game_id", None,
                                    "game_date", "odds_home", "odds_away", a),
    "TENIS": lambda a=True: _dos_vias("TENIS", "TENIS", "tenis_predictions", "event_id",
                                      "tour", "match_date", "odds_1", "odds_2", a),
    "SOCCER": _soccer,
}


def records(sport: str | None = None, market: str | None = None,
            solo_activas: bool = True, solo_calificadas: bool = False) -> list[dict]:
    """Todas las predicciones normalizadas. `solo_calificadas` deja solo las que
    ya tienen resultado (las unicas que sirven para medir nada)."""
    deportes = [sport.upper()] if sport and sport.upper() in LECTORES else list(LECTORES)
    out = []
    for s in deportes:
        try:
            out.extend(LECTORES[s](solo_activas))
        except Exception as e:                                   # noqa: BLE001
            out.append(_base(sport=s, market="__error__", status=f"{type(e).__name__}: {e}"))
    out = [r for r in out if r.get("market") != "__error__"]
    if market:
        out = [r for r in out if str(r.get("market")) == market]
    if solo_calificadas:
        out = [r for r in out if r.get("result")]
    return out


def cobertura() -> dict:
    """Que hay realmente disponible, por deporte. Alimenta Data Health.

    Deliberadamente cuenta por separado 'tiene resultado' y 'tiene cuota': son
    los dos requisitos del ROI y casi siempre falta el segundo.
    """
    ERRORES.clear()
    res = {}
    for s in LECTORES:
        rs = records(s, solo_activas=False)
        cal = [r for r in rs if r.get("result")]
        res[s] = {
            "predicciones": len(rs),
            "calificadas": len(cal),
            "con_mercado": sum(r.get("market_probability") is not None for r in rs),
            "con_cuota": sum(r.get("odds_decimal") is not None for r in rs),
            "calificadas_con_cuota": sum(
                r.get("odds_decimal") is not None for r in cal),
            "mercados": sorted({str(r.get("market")) for r in rs if r.get("market")}),
        }
    return {"deportes": res, "errores_de_consulta": list(ERRORES)}
