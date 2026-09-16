"""Tests de la conversacion: repreguntas sobre la respuesta anterior.

El caso que origino este modulo, tal cual ocurrio:

    Caballo Chat:  Elena Rybakina · 29.2%
    Usuario:       entonces elena solo tiene el 29% de ganar?
    Caballo Chat:  no reconozco ni el mercado ni el tipo de pregunta.

El dato estaba en pantalla. Faltaba memoria.
"""
from __future__ import annotations

import pytest

from CHAT import answer, followup

CTX = {
    "market_id": "tenis.winner", "etiqueta": "Gana el partido", "sport": "TENIS",
    "game_id": "abc", "selection": "Elena Rybakina",
    "p_cal": 0.292, "p_raw": 0.315, "p_market": 0.286, "gap_pp": 0.5,
    "status": "pick", "model_version": "TENIS_WTA_WINNER_v1",
    "calibracion": "isotonica",
    "partido": {"sport": "TENIS", "game_id": "abc", "home": "Qinwen Zheng",
                "away": "Elena Rybakina", "nombre": "Qinwen Zheng vs Elena Rybakina"},
    "evidencia": {"holdout_log_loss": 0.61, "baseline": 0.66,
                  "accuracy": 0.66, "baseline_accuracy": 0.63},
}


def test_el_caso_que_fallaba():
    r = answer.responder("entonces elena solo tiene el 29% de ganar?", contexto=CTX)
    assert r["ok"] is True
    assert r["tipo"] == "seguimiento"
    assert r["status"] != "BLOCKED"
    assert "29.2%" in r["respuesta"]
    assert "Sí, eso es" in r["respuesta"]
    # y da el otro lado, que es lo que uno quiere saber a continuacion
    assert "Qinwen Zheng" in r["respuesta"]
    assert "70.8%" in r["respuesta"]


def test_corrige_un_porcentaje_equivocado():
    r = followup.responder("entonces tiene el 50%?", CTX)
    assert "No exactamente" in r["respuesta"]
    assert "29.2%" in r["respuesta"]


def test_el_otro_lado_por_nombre():
    r = followup.responder("¿y Qinwen?", CTX)
    assert r["ok"] is True
    assert "70.8%" in r["respuesta"]
    assert r["numeros"]["lado"] == "Qinwen Zheng"


def test_el_otro_lado_generico():
    r = followup.responder("¿y el otro?", CTX)
    assert "70.8%" in r["respuesta"]


def test_no_complementa_mercados_que_no_son_de_dos_vias():
    """En 'gana o empata' el complemento NO es la probabilidad del rival."""
    ctx = {**CTX, "market_id": "soccer.double_chance",
           "etiqueta": "Gana o empata", "sport": "SOCCER",
           "selection": "Union Berlin o empate", "p_cal": 0.769,
           "partido": {"home": "Union Berlin", "away": "FC Schalke 04",
                       "nombre": "Union Berlin vs FC Schalke 04"}}
    r = followup.responder("¿y el otro?", ctx)
    assert r["ok"] is False
    assert "no es un mercado de dos vías" in r["respuesta"]
    assert "76.9" not in r["respuesta"] or "23" not in r["respuesta"]


def test_pedir_consejo_no_produce_una_recomendacion():
    """La honestidad sobre el ROI se mantiene aunque la pregunta sea coloquial."""
    r = followup.responder("¿conviene apostarle?", CTX)
    assert r["ok"] is True
    assert "no te lo puedo decir" in r["respuesta"]
    assert "11 apuestas" in r["respuesta"]
    # pero sí dice lo que SÍ está medido
    assert "0.61" in r["respuesta"]


def test_un_mercado_sin_evidencia_registrada_no_revienta():
    """`evidencia` existe con valor None en tenis: el default de .get no aplica."""
    ctx = {**CTX, "evidencia": None}
    r = followup.responder("¿conviene apostarle?", ctx)
    assert r["ok"] is True
    assert "no te lo puedo decir" in r["respuesta"]


def test_es_mucho_o_poco_no_es_pedir_consejo():
    """«¿es mucho?» pregunta qué significa el número, no si apostarlo.
    Soltar el aviso del ROI ahí convierte una pregunta legítima en un sermón."""
    r = followup.responder("¿es poco?", CTX)
    assert r["ok"] is True
    assert "29.2%" in r["respuesta"]
    assert "de cada 100" in r["respuesta"]
    assert "no te lo puedo decir" not in r["respuesta"]


def test_consejo_y_magnitud_se_distinguen():
    assert followup.pide_magnitud("¿es mucho?")
    assert not followup.pide_consejo("¿es mucho?")
    assert followup.pide_consejo("¿conviene apostarle?")
    assert not followup.pide_magnitud("¿conviene apostarle?")


def test_sin_contexto_no_hay_seguimiento():
    assert followup.responder("entonces solo tiene el 29%?", {}) is None
    assert followup.responder("entonces solo tiene el 29%?", None) is None


def test_una_pregunta_normal_no_se_traga_como_seguimiento():
    """Una pregunta de mercado con contexto debe seguir su ruta normal."""
    assert followup.responder("¿cuantas tarjetas se esperan?", CTX) is None
    assert followup.responder("¿cuantos goles?", CTX) is None


def test_extrae_el_porcentaje_citado():
    assert followup.porcentaje_mencionado("solo el 29%") == pytest.approx(0.29)
    assert followup.porcentaje_mencionado("el 62.5 %") == pytest.approx(0.625)
    assert followup.porcentaje_mencionado("sin numero") is None


def test_identifica_al_participante_sin_confundirlo():
    p = CTX["partido"]
    assert followup.participante("y elena?", p)["lado"] == "away"
    assert followup.participante("y qinwen?", p)["lado"] == "home"
    # un nombre que no es de nadie NO elige al mas parecido
    assert followup.participante("y serena?", p) is None
    # nombrar a los dos tampoco desempata solo
    assert followup.participante("qinwen contra elena", p) is None


@pytest.mark.parametrize("q", [
    "entonces?", "osea que si?", "en serio?", "confirma eso",
    "asi que tiene 29%", "solo tiene eso?",
])
def test_frases_de_seguimiento(q):
    assert followup.es_seguimiento(q)


def test_una_respuesta_normal_deja_contexto_para_la_siguiente():
    """Sin esto la conversación se corta después de cada respuesta."""
    import inspect
    src = inspect.getsource(answer)
    assert '"contexto": _contexto(' in src
    assert "def _contexto(" in src
