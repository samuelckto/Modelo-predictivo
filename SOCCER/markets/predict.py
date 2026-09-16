"""Predicciones para partidos PROXIMOS. Esto es lo que hace que el modulo no sea
solamente un backtester.

Flujo:
    1. estado de cada equipo despues de todo el historico  (features as-of)
    2. fila de features para cada fixture del calendario en vivo
    3. modelos entrenados -> distribucion de corners y de goles
    4. estado del mercado (PICK / PROJECTION / NO PICK / INSUFFICIENT DATA / BLOCKED)
    5. se guarda con timestamp real, version de modelo y corte de features

Nada de esto usa informacion posterior a `prediction_timestamp`. El corte de las
features es el estado del historico en el momento de predecir, y queda escrito en
`feature_cutoff` para poder auditarlo despues.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from SOCCER.db import (SoccerFixture, SoccerOdds, SoccerPrediction, init_db, session_scope)
from SOCCER.features.build import estado_actual, fila_fixture
from SOCCER.markets.gating import LABEL, MERCADOS, ORDEN
from SOCCER.models.corners import LINEAS as LINEAS_CORNERS, dist_total, esperado, over_under
from SOCCER.models.goals import mercados_batch
from SOCCER.markets.train import (VERSION_BTTS, VERSION_CORNERS, VERSION_DC, VERSION_TOTAL)
from shared.paths import SOCCER_MODELS_DIR, SOCCER_OUT_DIR
from shared.timeutil import utcnow

LINEAS_GOLES = (0.5, 1.5, 2.5, 3.5, 4.5)
LINEA_GOLES_PRINCIPAL = 2.5
LINEA_CORNERS_PRINCIPAL = 9.5
MIN_PARTIDOS_EQUIPO = 20            # historia minima del equipo para publicar algo

# Umbral para dejar de cubrirse con el empate. Por encima de esto el modelo ve
# un favorito claro y la doble oportunidad solo regala cuota: se paga mucho menos
# por cubrir un empate que el modelo considera poco probable.
UMBRAL_GANADOR = 0.62
# Por debajo de esta ventaja entre los dos equipos, el partido esta demasiado
# igualado para senalar un ganador y la cobertura del empate SI tiene sentido.
MARGEN_IGUALADO = 0.10


def _eleccion_1x2(home: str, away: str, ph: float, pd_: float, pa: float):
    """Que se publica en el mercado de doble oportunidad.

    Tres situaciones distintas que antes se trataban igual:

      1. Un equipo pasa del 62 % de ganar SOLO  -> se publica el ganador. Cubrir
         el empate ahi baja mucho la cuota a cambio de un riesgo que el modelo
         considera pequeno.
      2. Partido igualado (menos de 10 pp entre los dos) -> doble oportunidad
         del lado con ligera ventaja: es justo donde cubrir el empate paga.
      3. El resto -> doble oportunidad del lado mas probable.

    Y se arregla un error real de calculo. Antes, cuando el visitante era
    favorito, se hacia `1 - P(1X)`, que NO es X2:

        1 - P(1X) = 1 - (ph + pd) = pa          <- solo gana el visitante
        X2        = pd + pa                     <- gana o empata el visitante

    Con eso, 18 de 90 predicciones activas publicaban hasta 25 puntos MENOS de
    los que le correspondian a la seleccion que anunciaban (p.ej. 'AS Roma o
    empate' al 50.7 % cuando la cifra real era 75.4 %).
    """
    p_1x, p_x2 = ph + pd_, pd_ + pa
    if ph >= UMBRAL_GANADOR:
        return f"{home} gana", ph, "ganador"
    if pa >= UMBRAL_GANADOR:
        return f"{away} gana", pa, "ganador"
    # Cada lado con su probabilidad CORRECTA, y gana el mas probable.
    if p_1x >= p_x2:
        return f"{home} o empate", p_1x, ("doble_igualado"
                                          if abs(ph - pa) < MARGEN_IGUALADO else "doble")
    return f"{away} o empate", p_x2, ("doble_igualado"
                                      if abs(ph - pa) < MARGEN_IGUALADO else "doble")


def _pid(event_id: str, market: str, ts: datetime) -> str:
    return hashlib.sha1(f"{event_id}|{market}|{ts.isoformat()}".encode()).hexdigest()[:20]


def _gating() -> dict:
    p = SOCCER_OUT_DIR / "gating.json"
    if not p.exists():
        return {"global": {"mercados": {}}, "por_liga": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _lado_mercado(market: str, seleccion: str, home: str, away: str):
    """Que fila de cuotas corresponde a la seleccion que se va a publicar.

    Antes esto no existia: se cogia SIEMPRE el lado OVER/YES. Con eso, un pick
    de 'UNDER 2.5' se comparaba contra la probabilidad del OVER y la diferencia
    salia del reves. Caso real: UNDER 2.5 con el modelo al 59.2 % aparecia como
    +27.9 pp de ventaja sobre un mercado del 31.3 %; pero ese 31.3 % era la
    probabilidad del OVER, asi que el mercado daba ~68.7 % al UNDER y el modelo
    iba 9.5 pp POR DEBAJO. Un desacuerdo a favor convertido en uno en contra.

    Devuelve un predicado sobre el nombre de la seleccion en la tabla de cuotas,
    o None si ese mercado no tiene precio directo (la doble oportunidad no se
    cotiza en h2h: 'gana o empata' no es ninguna de las tres vias).
    """
    sl = str(seleccion).lower()
    if market == "totals":
        quiere = "under" if sl.startswith("under") else "over"
        return lambda n: str(n).lower().startswith(quiere)
    if market == "btts":
        quiere = ("no",) if sl == "NO".lower() else ("yes", "si")
        return lambda n: str(n).lower().startswith(quiere)
    if market == "h2h":
        # Solo el ganador directo tiene precio propio. 'X o empate' no.
        if not sl.endswith(" gana"):
            return None
        equipo = home if seleccion.startswith(home) else away
        eq = equipo.lower()
        return lambda n: str(n).lower() == eq
    return None


def _mercado_actual(s, event_id: str, market: str, linea: float | None,
                    seleccion: str = "", home: str = "", away: str = ""):
    """Consenso de cuotas mas reciente PARA EL LADO QUE SE PUBLICA.

    Devuelve (prob_sin_vig, precio, casa, ts). La probabilidad y el precio son
    siempre del mismo lado que `seleccion`; comparar lados distintos es peor que
    no comparar, porque produce una diferencia con signo equivocado.
    """
    filas = s.execute(select(SoccerOdds).where(
        SoccerOdds.event_id == event_id, SoccerOdds.market == market)).scalars().all()
    if not filas:
        return None, None, None, None
    ultimo = max(f.fetched_at for f in filas)
    filas = [f for f in filas if f.fetched_at == ultimo]
    if linea is not None:
        filas = [f for f in filas if f.line is not None and abs(f.line - linea) < 1e-6]
    casa = _lado_mercado(market, seleccion, home, away)
    if casa is None:
        return None, None, None, ultimo
    lado = [f for f in filas if casa(f.selection)]
    if not lado:
        return None, None, None, ultimo
    probs = [f.novig for f in lado if f.novig is not None]
    p = float(np.mean(probs)) if probs else None
    mejor = max(lado, key=lambda f: f.price)
    return p, mejor.price, mejor.bookmaker, ultimo


def _estado(gating: dict, code: str, market: str) -> dict:
    liga = (gating.get("por_liga") or {}).get(code, {})
    return liga.get(market) or (gating.get("global", {}).get("mercados", {}).get(market)
                                or {"estado": "insufficient_data", "motivo": "sin evaluacion"})


def generar(dias: int = 8, progress=None) -> dict:
    """Predice todos los fixtures proximos. Idempotente: versiona, no sobrescribe."""
    init_db()
    ahora = utcnow()
    gating = _gating()
    art_g = SOCCER_MODELS_DIR / "goles_v1.joblib"
    art_c = SOCCER_MODELS_DIR / "corners_v1.joblib"
    if not art_g.exists():
        return {"ok": False, "motivo": "faltan modelos: ejecuta soccer-train", "predicciones": 0}
    goles = joblib.load(art_g)
    corners = joblib.load(art_c) if art_c.exists() else None

    if progress:
        progress("reconstruyendo el estado de los equipos desde el historico")
    estados, ventaja, resolver, ultima = estado_actual()
    cutoff = ahora

    with session_scope() as s:
        fixtures = s.execute(select(SoccerFixture).where(
            SoccerFixture.commence_utc >= ahora - timedelta(hours=2),
            SoccerFixture.commence_utc <= ahora + timedelta(days=dias))).scalars().all()
        if not fixtures:
            return {"ok": True, "predicciones": 0, "fixtures": 0,
                    "motivo": "no hay partidos proximos en el calendario",
                    "consultado_en": ahora.isoformat()}

        filas = [fila_fixture(estados, ventaja, resolver, f.league_code,
                              f.home_team, f.away_team, f.commence_utc.date(), f.commence_utc)
                 for f in fixtures]
        X = pd.DataFrame(filas)

        lh, la = goles["estimador"].predict(X)
        mg = mercados_batch(lh, la, goles["familia"], rho=goles["rho"], phi=goles["phi"],
                            lineas=LINEAS_GOLES)
        if corners is not None:
            clh, cla = corners["estimador"].predict(X)
            Dc = dist_total(clh, cla, corners["familia"], corners["phi"], "suma")
            eg_c = esperado(Dc)
        else:
            clh = cla = eg_c = None

        creadas = 0
        resumen: dict[str, int] = {}
        for i, f in enumerate(fixtures):
            fila = filas[i]
            poca_historia = min(fila["_partidos_home"], fila["_partidos_away"]) < MIN_PARTIDOS_EQUIPO
            for market in ORDEN:
                g = _estado(gating, f.league_code, market)
                estado = g.get("estado", "insufficient_data")
                if estado == "pick_habilitado":
                    estado = "pick"
                if poca_historia and estado not in ("insufficient_data", "blocked"):
                    estado = "no_pick"

                if market == "double_chance":
                    # Sale de la MISMA distribucion conjunta que el resto: 1X es
                    # P(gana local) + P(empate), X2 es P(empate) + P(gana visitante).
                    # No necesita fuente extra, asi que funciona en TODAS las
                    # competiciones, tambien en Champions y Saudi.
                    ph = float(mg["p_home"][i])
                    pd_ = float(mg["p_draw"][i])
                    pa = float(mg["p_away"][i])
                    p_1x, p_x2 = ph + pd_, pd_ + pa
                    linea = None
                    eg = float(mg["eg_total"][i])
                    lam_h, lam_a = float(lh[i]), float(la[i])
                    todas_lineas = {"1X": p_1x, "X2": p_x2, "12": ph + pa,
                                    "local": ph, "empate": pd_, "visitante": pa}
                    sel, prob, apuesta = _eleccion_1x2(f.home_team, f.away_team,
                                                       ph, pd_, pa)
                    p_over = None                     # ya esta decidido aqui
                elif market == "corners":
                    if corners is None or fila["corners_historial"] < MIN_PARTIDOS_EQUIPO:
                        estado = "insufficient_data"
                        g = {**g, "motivo": "sin historial de corners suficiente para estos equipos"}
                    linea = LINEA_CORNERS_PRINCIPAL
                    if estado == "insufficient_data":
                        p_over = eg = None
                    else:
                        p_over = float(over_under(Dc[i:i + 1], linea)[0][0])
                        eg = float(eg_c[i])
                    lam_h, lam_a = (float(clh[i]), float(cla[i])) if clh is not None else (None, None)
                    todas_lineas = ({f"{ln:g}": float(over_under(Dc[i:i + 1], ln)[0][0])
                                     for ln in LINEAS_CORNERS} if p_over is not None else {})
                elif market == "total_goals":
                    linea = LINEA_GOLES_PRINCIPAL
                    p_over = float(mg["totales"][f"{linea:g}"]["over"][i])
                    eg = float(mg["eg_total"][i])
                    lam_h, lam_a = float(lh[i]), float(la[i])
                    todas_lineas = {f"{ln:g}": float(mg["totales"][f"{ln:g}"]["over"][i])
                                    for ln in LINEAS_GOLES}
                else:                                   # btts
                    linea = None
                    p_over = float(mg["btts_yes"][i])
                    eg = float(mg["eg_total"][i])
                    lam_h, lam_a = float(lh[i]), float(la[i])
                    todas_lineas = {}

                if market == "double_chance":
                    pass                      # sel y prob ya se decidieron arriba
                elif p_over is None:
                    prob, sel = 0.0, "SIN DATOS"
                else:
                    lado_over = p_over >= 0.5
                    prob = p_over if lado_over else 1 - p_over
                    if market == "btts":
                        sel = "SI" if lado_over else "NO"
                    else:
                        sel = f"{'OVER' if lado_over else 'UNDER'} {linea:g}"

                mkey = {"corners": "totals", "total_goals": "totals", "btts": "btts",
                        "double_chance": "h2h"}[market]
                p_mkt, precio, casa, ots = _mercado_actual(
                    s, f.event_id, mkey, linea, sel, f.home_team, f.away_team)
                gap = round((prob - (p_mkt if p_mkt is not None else prob)) * 100, 2) if p_mkt else None

                prev = s.execute(select(SoccerPrediction).where(
                    SoccerPrediction.event_id == f.event_id,
                    SoccerPrediction.market == market,
                    SoccerPrediction.record_status == "active")).scalars().all()
                version = 1
                for p in prev:
                    p.record_status = "superseded"
                    version = max(version, p.version + 1)

                s.add(SoccerPrediction(
                    prediction_id=_pid(f.event_id, market, ahora),
                    event_id=f.event_id, league_code=f.league_code,
                    match_date=f.commence_utc.date(), kickoff_utc=f.commence_utc,
                    home_team=f.home_team, away_team=f.away_team,
                    market=market, selection=sel, line=linea,
                    probability=prob, lambda_home=lam_h, lambda_away=lam_a,
                    expected_goals=eg, market_probability=p_mkt, gap_pp=gap,
                    odds_price=precio, bookmaker=casa, odds_timestamp=ots,
                    status=estado,
                    model_version={"corners": VERSION_CORNERS, "btts": VERSION_BTTS,
                                   "total_goals": VERSION_TOTAL,
                                   "double_chance": VERSION_DC}[market],
                    calibration_version=None, feature_cutoff=cutoff,
                    prediction_timestamp=ahora, version=version, record_status="active",
                    explanation=json.dumps({
                        "mercado": LABEL[market], "motivo_estado": g.get("motivo"),
                        "lineas": todas_lineas,
                        # Que TIPO de apuesta es la seleccion. Sin esto, al
                        # calificar no se puede distinguir 'gana' de 'gana o
                        # empata' y un empate contaria como acierto del ganador.
                        "apuesta": (apuesta if market == "double_chance" else None),
                        "factores": _factores(market, fila, lam_h, lam_a, eg),
                        "aviso_mercado": ("diferencia frente al mercado: INFORMATIVA, no validada"
                                          if gap is not None else None)}, default=str),
                    features_snapshot=json.dumps(
                        {k: v for k, v in fila.items() if not k.startswith("_")}, default=str),
                    data_completeness=json.dumps({
                        "partidos_home": fila["_partidos_home"],
                        "partidos_away": fila["_partidos_away"],
                        "historial_corners": fila["corners_historial"],
                        "nombre_home_resuelto": fila["_similitud_home"],
                        "nombre_away_resuelto": fila["_similitud_away"],
                        "ultimo_partido_historico": str(ultima.get(f.league_code)),
                        "cuotas": bool(p_mkt)}, default=str)))
                creadas += 1
                resumen[estado] = resumen.get(estado, 0) + 1
        s.flush()
    return {"ok": True, "fixtures": len(fixtures), "predicciones": creadas,
            "por_estado": resumen, "prediction_timestamp": ahora.isoformat(),
            "feature_cutoff": cutoff.isoformat()}


def _factores(market: str, fila: dict, lam_h, lam_a, eg) -> list[dict]:
    """Solo features que el modelo usa de verdad. No se inventan explicaciones."""
    if market == "double_chance":
        return [{"label": "goles esperados del local", "value": round(lam_h, 2) if lam_h else None},
                {"label": "goles esperados del visitante", "value": round(lam_a, 2) if lam_a else None},
                {"label": "diferencia de Elo", "value": round(fila["elo_diff"], 1)},
                {"label": "ventaja local de la competición (puntos Elo)",
                 "value": round(fila["ventaja_local_liga"], 1)},
                {"label": "puntos por partido del local", "value": round(fila["home_ppp"], 2)},
                {"label": "puntos por partido del visitante", "value": round(fila["away_ppp"], 2)}]
    if market == "corners":
        return [{"label": "córners esperados del partido", "value": round(eg, 2) if eg else None},
                {"label": "córners a favor del local en casa (media)",
                 "value": round(fila["home_cf_casa"], 2)},
                {"label": "córners en contra del visitante fuera (media)",
                 "value": round(fila["away_cc_fuera"], 2)},
                {"label": "media de córners de los dos en sus últimos 5",
                 "value": round(fila["corners_medio_5"], 2)},
                {"label": "partidos con córners en el historial",
                 "value": int(fila["corners_historial"])}]
    base = [{"label": "goles esperados del local", "value": round(lam_h, 2) if lam_h else None},
            {"label": "goles esperados del visitante", "value": round(lam_a, 2) if lam_a else None},
            {"label": "diferencia de Elo", "value": round(fila["elo_diff"], 1)}]
    if market == "btts":
        return base + [
            {"label": "frecuencia histórica de ambos marcan", "value": round(fila["btts_medio"], 3)},
            {"label": "porterías a cero del local", "value": round(fila["home_cs_rate"], 3)},
            {"label": "partidos sin marcar del visitante", "value": round(fila["away_fts_rate"], 3)}]
    return base + [
        {"label": "total esperado", "value": round(eg, 2) if eg else None},
        {"label": "frecuencia histórica de over 2.5", "value": round(fila["over25_medio"], 3)}]
