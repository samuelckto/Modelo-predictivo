"""Registro de mercados. Version CHAT_MARKETS_v1.

Un sitio unico donde cada mercado declara que es, que sabe hacer y con que
respaldo. El chat NO conoce mercados: consulta este registro. Anadir tarjetas,
corners, tiros a puerta o props de jugador es anadir una entrada aqui, no tocar
el chat.

Los dos ejes que decide este archivo:

  VISIBILIDAD   primary    aparece en Top Picks y en el reporte principal.
                secondary  SOLO se puede consultar por chat, bajo demanda.

  ESTADO        ok                 hay modelo validado y datos.
                projection         hay modelo, pero sin cuota historica no se
                                   puede demostrar ventaja: se publica la
                                   probabilidad y nada mas.
                no_pick            el modelo no le gana a su baseline.
                insufficient_data  no hay fuente.
                blocked            hay fuente pero algo impide usarla.

Ser `secondary` NO rebaja el liston. Un mercado secundario pasa por los mismos
requisitos que uno principal: datos, features pregame, modelo, validacion
temporal y calibracion cuando toque. Lo unico que cambia es donde se ve.
"""
from __future__ import annotations

VERSION = "CHAT_MARKETS_v1"

VISIBILIDAD = ("primary", "secondary")
ESTADOS = ("ok", "projection", "no_pick", "insufficient_data", "blocked")
DISTRIBUCIONES = ("binaria", "conteo", "conteo_bivariado", "ninguna")


def m(market_id, sport, etiqueta, pregunta, **kw) -> dict:
    """Una entrada del registro. Todo campo no declarado queda explicitamente en None."""
    d = {
        "market_id": market_id,
        "sport": sport,
        "league": kw.get("league"),
        "etiqueta": etiqueta,
        "pregunta_tipo": pregunta,
        "visibilidad": kw.get("visibilidad", "primary"),
        "estado": kw.get("estado", "insufficient_data"),
        "model_version": kw.get("model_version"),
        "distribution_type": kw.get("distribution_type", "binaria"),
        "supported_lines": kw.get("supported_lines"),
        "probability_method": kw.get("probability_method"),
        "calibration_status": kw.get("calibration_status", "sin_calibrar"),
        "data_requirements": kw.get("data_requirements", []),
        "ligas_con_datos": kw.get("ligas_con_datos"),
        "evidencia": kw.get("evidencia"),
        "limitaciones": kw.get("limitaciones", []),
        "sinonimos": kw.get("sinonimos", []),
    }
    assert d["visibilidad"] in VISIBILIDAD, d["visibilidad"]
    assert d["estado"] in ESTADOS, d["estado"]
    assert d["distribution_type"] in DISTRIBUCIONES, d["distribution_type"]
    return d


# ---------------------------------------------------------------------------
# MLB
# ---------------------------------------------------------------------------
_MLB = [
    m("mlb.moneyline", "MLB", "Gana el partido", "¿quien gana?",
      estado="projection", model_version="elo_v2", distribution_type="binaria",
      probability_method="Elo + mezcla con mercado",
      data_requirements=["resultado historico", "alineaciones", "abridores"],
      sinonimos=["quien gana", "ganador", "moneyline", "linea de dinero"]),
    m("mlb.total", "MLB", "Over/Under carreras", "¿cuantas carreras se esperan?",
      estado="projection", model_version="total_v2", distribution_type="conteo",
      supported_lines=[6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
      probability_method="RandomForest + residuos empiricos",
      data_requirements=["marcadores", "abridor", "bullpen", "parque", "clima"],
      sinonimos=["carreras", "total", "over under", "cuantas carreras"]),
    m("mlb.run_line", "MLB", "Run line", "¿gana por mas de una carrera?",
      estado="projection", model_version="run_line_v2", distribution_type="binaria",
      sinonimos=["run line", "handicap", "-1.5"]),
    m("mlb.runs_dist", "MLB", "Distribucion de carreras", "¿que probabilidad hay de N carreras?",
      visibilidad="secondary", distribution_type="conteo",
      model_version="MLB_RUNS_DIST_v1",
      probability_method="lambdas por equipo (gradient boosting Poisson) + conteo",
      data_requirements=["marcadores por equipo 2021-2026"],
      sinonimos=["distribucion", "team total", "anota", "carreras del equipo",
                 "ambos anotan", "al menos una carrera"]),
    m("mlb.f5_total", "MLB", "Primeras 5 entradas", "¿habra carreras en las primeras 5?",
      visibilidad="secondary", distribution_type="conteo",
      model_version="MLB_RUNS_DIST_v1",
      supported_lines=[3.5, 4.5, 5.5, 6.5],
      probability_method="modelo F5 PROPIO (marcadores reales de 5 entradas)",
      data_requirements=["home_score_f5 y away_score_f5"],
      limitaciones=["no se deriva del partido completo: tiene objetivo propio"],
      sinonimos=["primeras 5", "f5", "primeras cinco", "cinco entradas"]),
]

# ---------------------------------------------------------------------------
# SOCCER
# ---------------------------------------------------------------------------
_SOCCER = [
    m("soccer.double_chance", "SOCCER", "Gana o empata", "¿gana o empata?",
      estado="projection", model_version="SOCCER_DC_v1", distribution_type="conteo_bivariado",
      probability_method="lambdas gradient boosting + Dixon-Coles",
      evidencia={"holdout_log_loss": 0.57024, "baseline": 0.62715,
                 "accuracy": 0.701, "baseline_accuracy": 0.681, "ece": 0.011},
      sinonimos=["gana o empata", "doble oportunidad", "1x", "x2"]),
    m("soccer.total_goals", "SOCCER", "Total de goles", "¿cuantos goles se esperan?",
      estado="projection", model_version="SOCCER_TOTAL_v1", distribution_type="conteo_bivariado",
      supported_lines=[1.5, 2.5, 3.5, 4.5],
      evidencia={"holdout_log_loss": 0.67745, "baseline": 0.69074,
                 "accuracy": 0.572, "baseline_accuracy": 0.540},
      sinonimos=["goles", "over 2.5", "total de goles", "cuantos goles"]),
    m("soccer.btts", "SOCCER", "Ambos marcan", "¿marcan los dos equipos?",
      estado="projection", model_version="SOCCER_BTTS_v1", distribution_type="conteo_bivariado",
      evidencia={"holdout_log_loss": 0.68531, "baseline": 0.68975,
                 "accuracy": 0.539, "baseline_accuracy": 0.544},
      limitaciones=["la mejora sobre el baseline es minima y el acierto es PEOR "
                    "que el baseline: este mercado casi no aporta"],
      sinonimos=["ambos marcan", "btts", "los dos marcan"]),
    m("soccer.cards", "SOCCER", "Tarjetas amarillas", "¿cuantas tarjetas se esperan?",
      visibilidad="secondary", model_version="SOCCER_CARDS_v1", distribution_type="conteo",
      supported_lines=[2.5, 3.5, 4.5, 5.5, 6.5],
      probability_method="lambdas por equipo (gradient boosting Poisson)",
      data_requirements=["home_yellow y away_yellow"],
      ligas_con_datos=["EPL", "LALIGA", "SERIEA", "BUNDES", "LIGUE1"],
      limitaciones=["NO hay arbitro en la base y es el factor conocido mas fuerte",
                    "Champions, Liga MX y Saudi no tienen fuente de tarjetas"],
      sinonimos=["tarjetas", "amarillas", "cartones", "ambos reciben tarjeta"]),
    m("soccer.corners", "SOCCER", "Corners", "¿cuantos corners se esperan?",
      visibilidad="secondary", model_version="SOCCER_CORNERS_v1", distribution_type="conteo",
      supported_lines=[7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
      ligas_con_datos=["EPL", "LALIGA", "SERIEA", "BUNDES", "LIGUE1"],
      evidencia={"holdout_log_loss": 0.68476, "baseline": 0.69560,
                 "accuracy": 0.546, "baseline_accuracy": 0.493},
      sinonimos=["corners", "esquinas", "tiros de esquina"]),
]

# ---------------------------------------------------------------------------
# NFL / NBA / TENIS
# ---------------------------------------------------------------------------
_OTROS = [
    m("nfl.total", "NFL", "Over/Under puntos", "¿cuantos puntos se esperan?",
      estado="projection", model_version="total_v1", distribution_type="binaria",
      sinonimos=["puntos", "total", "over under"]),
    m("nfl.spread", "NFL", "Spread", "¿cubre el spread?",
      estado="projection", model_version="spread_v1", distribution_type="binaria",
      sinonimos=["spread", "handicap", "cubre"]),
    m("nba.moneyline", "NBA", "Gana el partido", "¿quien gana?",
      estado="projection", distribution_type="binaria", sinonimos=["quien gana", "ganador"]),
    m("nba.total", "NBA", "Over/Under puntos", "¿cuantos puntos se esperan?",
      estado="projection", distribution_type="binaria", sinonimos=["puntos", "total"]),
    m("nba.spread", "NBA", "Spread", "¿cubre el spread?",
      estado="projection", distribution_type="binaria", sinonimos=["spread", "handicap"]),
    m("tenis.winner", "TENIS", "Gana el partido", "¿quien gana?",
      estado="projection", distribution_type="binaria",
      sinonimos=["quien gana", "ganador"]),
    m("tenis.total_games", "TENIS", "Total de juegos", "¿cuantos juegos se esperan?",
      estado="projection", distribution_type="binaria", sinonimos=["juegos", "games", "total"]),
    m("tenis.handicap_games", "TENIS", "Handicap de juegos", "¿cubre el handicap?",
      estado="projection", distribution_type="binaria", sinonimos=["handicap", "cubre"]),
]

MERCADOS = {d["market_id"]: d for d in (_MLB + _SOCCER + _OTROS)}


def listar(sport: str | None = None, visibilidad: str | None = None) -> list[dict]:
    out = list(MERCADOS.values())
    if sport:
        out = [d for d in out if d["sport"] == sport.upper()]
    if visibilidad:
        out = [d for d in out if d["visibilidad"] == visibilidad]
    return out


def get(market_id: str) -> dict | None:
    return MERCADOS.get(market_id)


def secundarios(sport: str | None = None) -> list[dict]:
    """Mercados que NO aparecen en Top Picks y solo se consultan por chat."""
    return listar(sport, "secondary")


def buscar(texto: str, sport: str | None = None) -> list[dict]:
    """Mercados cuyo id, etiqueta o sinonimos casan con el texto del usuario.

    Comparacion por SUBCADENA sobre sinonimos declarados, no por parecido de
    cadenas. Un parecido difuso confunde 'tarjetas' con 'tiros' y termina
    respondiendo con el modelo equivocado, que es peor que no responder.
    """
    t = (texto or "").lower()
    hits = []
    for d in listar(sport):
        campos = [d["market_id"], d["etiqueta"].lower(), *[s.lower() for s in d["sinonimos"]]]
        peso = sum(len(c) for c in campos if c and c in t)
        if peso:
            hits.append((peso, d))
    return [d for _, d in sorted(hits, key=lambda kv: -kv[0])]


def refrescar_desde_gating() -> dict:
    """Actualiza el estado de los mercados de SOCCER con el gating real en disco.

    El registro trae valores por defecto; la verdad esta en gating.json, que se
    recalcula con los datos actuales. Si el archivo no existe se dice, no se
    supone que todo esta bien.
    """
    import json

    from shared.paths import SOCCER_OUT_DIR
    f = SOCCER_OUT_DIR / "gating.json"
    if not f.exists():
        return {"ok": False, "motivo": f"no existe {f}: el estado mostrado es el "
                                       f"declarado en el registro, no el medido."}
    g = json.loads(f.read_text(encoding="utf-8"))
    cambios = {}
    for mid, d in MERCADOS.items():
        if d["sport"] != "SOCCER":
            continue
        clave = mid.split(".", 1)[1]
        info = (g.get("global", {}).get("mercados", {}) or {}).get(clave)
        if info and info.get("estado") in ESTADOS and info["estado"] != d["estado"]:
            cambios[mid] = {"antes": d["estado"], "ahora": info["estado"]}
            d["estado"] = info["estado"]
    return {"ok": True, "cambios": cambios, "archivo": str(f)}
