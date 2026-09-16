"""Tests del conocimiento del SISTEMA (calendario, cobertura, modelos, ROI).

El bug que motivo este modulo: con un partido de tenis seleccionado,
"¿que partidos hay de beisbol hoy?" devolvia el resumen del tenis. La pregunta
no era del partido, era del calendario, y no habia a donde enviarla.
"""
from __future__ import annotations

import pytest

from CHAT import answer, knowledge


@pytest.mark.parametrize("q,tipo", [
    ("¿que partidos hay de beisbol hoy?", "calendario"),
    ("que juegos hay hoy", "calendario"),
    ("dame el calendario de futbol", "calendario"),
    ("¿que se juega mañana?", "calendario"),
    ("¿que ligas tienes?", "cobertura"),
    ("¿de que hay datos?", "cobertura"),
    ("¿como va el sistema?", "rendimiento"),
    ("¿cuanto acierta?", "rendimiento"),
    ("¿vas ganando?", "rendimiento"),
    ("¿que tan bueno es el modelo de tarjetas?", "modelos"),
    ("¿cual es el mejor mercado?", "modelos"),
    ("¿que recomiendas hoy?", "picks"),
])
def test_clasifica_las_preguntas_del_sistema(q, tipo):
    assert knowledge.clasificar(q) == tipo


@pytest.mark.parametrize("q", [
    "¿cuantas tarjetas se esperan?", "¿quien gana?", "hola",
    "¿que probabilidad hay de over 2.5?", "entonces solo tiene 29%?",
])
def test_no_secuestra_las_preguntas_de_partido(q):
    """Una pregunta sobre el partido NO debe caer aqui."""
    assert knowledge.clasificar(q) is None


def test_detecta_el_deporte_en_la_pregunta():
    assert knowledge._deporte(knowledge._norm("partidos de beisbol")) == "MLB"
    assert knowledge._deporte(knowledge._norm("juegos de futbol")) == "SOCCER"
    assert knowledge._deporte(knowledge._norm("partidos de tenis")) == "TENIS"
    assert knowledge._deporte(knowledge._norm("que hay hoy")) is None


def test_el_calendario_gana_al_partido_seleccionado():
    """El bug exacto: con tenis seleccionado, preguntar por béisbol."""
    r = answer.responder("¿que partidos hay de beisbol hoy?",
                         game_id="cualquiera", sport="TENIS")
    assert r["intencion"] == "calendario"
    assert r["tipo"] == "calendario"
    # no debe haberse ido por la rama del resumen del partido seleccionado
    assert r.get("mercados") is None or r["tipo"] != "resumen"


def test_el_rendimiento_no_maquilla_el_veredicto():
    r = knowledge.responder("¿como va el sistema?")
    assert r["tipo"] == "rendimiento"
    assert "NO se ha demostrado ventaja rentable" in r["respuesta"]
    assert "no te voy a maquillar" in r["respuesta"]


def test_los_picks_avisan_de_como_leerlos():
    r = knowledge.responder("¿que recomiendas hoy?")
    assert r["tipo"] == "picks"
    if r["ok"]:
        assert "no de las más rentables" in r["respuesta"]


def test_los_modelos_reportan_contra_su_baseline():
    r = knowledge.responder("¿cual es el mejor mercado?")
    assert r["tipo"] == "modelos"
    assert "baseline" in r["respuesta"]
    assert "PROJECTION" in r["respuesta"]


def test_un_fallo_al_consultar_no_tumba_la_respuesta(monkeypatch):
    def revienta(_):
        raise RuntimeError("base caida")
    monkeypatch.setitem(knowledge.DESPACHO, "cobertura", revienta)
    r = knowledge.responder("¿que ligas tienes?")
    assert r["ok"] is False
    assert r["status"] == "ERROR"
    assert "base caida" in r["respuesta"]
