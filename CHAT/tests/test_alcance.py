"""Alcance: periodo y equipo.

Los dos fallos reales que protege este archivo, ambos de Levante vs Barcelona:

    "¿cuantos goles le va a anotar el barcelona?"
        -> el chat lo trataba como «¿y el otro lado?» y contestaba con una
           frase sobre restar probabilidades. El usuario no pidio restar nada.

    "¿cuantos goles se esperan antes de la primera mitad?"
        -> el chat devolvia OVER 2.5 al 56.4 %, que es el TOTAL DEL PARTIDO
           COMPLETO. No es una aproximacion de la primera mitad: es otro
           numero, y el usuario no tenia forma de notar el cambio de pregunta.
"""
from __future__ import annotations

from CHAT import alcance, followup, registry

PARTIDO = {"game_id": "x1", "sport": "SOCCER", "home": "Levante",
           "away": "Barcelona", "nombre": "Levante vs Barcelona"}
TOTAL = registry.get("soccer.total_goals")
CARDS = registry.get("soccer.cards")


# --------------------------------------------------------------------------
# Periodo
# --------------------------------------------------------------------------

def test_detecta_la_primera_mitad():
    for t in ["goles antes de la primera mitad", "en el primer tiempo",
              "cuantos goles en la primera parte", "al descanso"]:
        assert alcance.periodo(t) == "primera_mitad", t


def test_detecta_la_segunda_mitad():
    assert alcance.periodo("goles del segundo tiempo") == "segunda_mitad"


def test_una_pregunta_normal_no_tiene_periodo():
    assert alcance.periodo("cuantos goles se esperan?") is None


def test_la_primera_mitad_no_se_contesta_con_el_partido_completo():
    av = alcance.revisar("cuantos goles se esperan antes de la primera mitad",
                         TOTAL, PARTIDO)
    assert av is not None and av["tipo"] == "no_cubierto"
    assert "primera mitad" in av["motivo"]
    assert "PARTIDO COMPLETO" in av["motivo"]


def test_en_mlb_las_primeras_5_si_existen_y_se_redirige():
    """Negarse estaria mal: ese mercado SI esta publicado."""
    av = alcance.revisar("cuantas carreras en las primeras 5",
                         registry.get("mlb.total"), None)
    assert av is not None and av["tipo"] == "redirigir"
    assert av["market_id"] == "mlb.f5_total"


def test_el_mercado_f5_no_se_queja_de_su_propio_periodo():
    assert alcance.revisar("carreras en las primeras 5",
                           registry.get("mlb.f5_total"), None) is None


# --------------------------------------------------------------------------
# Equipo
# --------------------------------------------------------------------------

def test_detecta_que_la_pregunta_es_de_un_equipo():
    assert alcance.equipo_objetivo(
        "cuantos goles le va a anotar el barcelona", PARTIDO) == "Barcelona"
    assert alcance.equipo_objetivo(
        "cuantas tarjetas del levante", PARTIDO) == "Levante"


def test_nombrar_a_los_dos_equipos_es_una_pregunta_del_partido():
    """«goles en el Levante vs Barcelona» NO es una pregunta por equipo."""
    assert alcance.equipo_objetivo(
        "cuantos goles en el levante vs barcelona", PARTIDO) is None


def test_nombrar_un_equipo_sin_acotar_no_basta():
    assert alcance.equipo_objetivo("gana el barcelona", PARTIDO) is None


def test_el_total_no_se_reparte_a_ojo_entre_los_dos_equipos():
    av = alcance.revisar("cuantos goles le va a anotar el barcelona",
                         TOTAL, PARTIDO)
    assert av is not None and av["tipo"] == "sin_desglose"
    assert av["equipo"] == "Barcelona"
    assert "inventarlo" in av["motivo"]


def test_las_tarjetas_si_tienen_desglose_por_equipo():
    """Ese modelo SI publica reparto local/visitante: no hay que negarse."""
    assert alcance.revisar("cuantas tarjetas del levante", CARDS, PARTIDO) is None


# --------------------------------------------------------------------------
# La repregunta ya no secuestra una pregunta nueva
# --------------------------------------------------------------------------

CTX = {"market_id": "soccer.total_goals", "etiqueta": "Total de goles",
       "sport": "SOCCER", "selection": "OVER 2.5", "p_cal": 0.564,
       "p_market": 0.729, "gap_pp": -16.4, "partido": PARTIDO}


def test_una_pregunta_nueva_no_se_trata_como_repregunta():
    assert followup.responder(
        "cuantos goles le va a anotar el barcelona", CTX) is None


def test_la_repregunta_de_verdad_sigue_funcionando():
    r = followup.responder("entonces solo tiene el 56%?", CTX)
    assert r is not None and r["ok"] is True


def test_una_mencion_suelta_sigue_pidiendo_el_otro_lado():
    r = followup.responder("y el levante?", CTX)
    assert r is not None
