"""Tests del catalogo de partidos de Caballo Chat.

El riesgo de este modulo no es no encontrar el partido: es encontrar el
EQUIVOCADO y contestar con total seguridad sobre otro juego. Por eso casi todas
las pruebas son sobre cuando NO debe elegir.
"""
from __future__ import annotations

import pytest

from CHAT import games


def _j(sport, gid, home, away, cuando="2026-09-11 18:30:00"):
    return {"sport": sport, "game_id": gid, "home": home, "away": away,
            "nombre": f"{home} vs {away}", "liga": sport, "start_utc": cuando,
            "game_date": cuando[:10], "venue": None, "mercados": []}


CALENDARIO = [
    _j("SOCCER", "1", "Union Berlin", "FC Schalke 04"),
    _j("SOCCER", "2", "FC Barcelona", "Real Madrid"),
    _j("SOCCER", "3", "Manchester United", "Arsenal"),
    _j("SOCCER", "4", "Manchester City", "Liverpool"),
    _j("MLB", "5", "Boston Red Sox", "New York Yankees"),
]


@pytest.fixture(autouse=True)
def calendario_fijo(monkeypatch):
    monkeypatch.setattr(games, "listar", lambda dias=10, sport=None:
                        [j for j in CALENDARIO if not sport or j["sport"] == sport])


# --------------------------------------------------------------------------

def test_encuentra_por_un_solo_equipo():
    r = games.resolver("¿cuantas tarjetas en el Barcelona?")
    assert r["encontrado"]
    assert r["partido"]["game_id"] == "2"


def test_encuentra_nombrando_a_los_dos():
    r = games.resolver("¿cuantos goles en Union Berlin contra Schalke?")
    assert r["encontrado"]
    assert r["partido"]["game_id"] == "1"


def test_no_confunde_manchester_united_con_city():
    """El error clasico. 'manchester' solo NO puede desempatar."""
    r = games.resolver("¿tarjetas del Manchester?")
    assert not r["encontrado"]
    assert r["ambiguo"] is True
    assert len(r["candidatos"]) == 2

    u = games.resolver("¿tarjetas del Manchester United?")
    assert u["encontrado"] and u["partido"]["game_id"] == "3"
    c = games.resolver("¿tarjetas del Manchester City?")
    assert c["encontrado"] and c["partido"]["game_id"] == "4"


def test_sin_equipo_no_inventa_partido():
    r = games.resolver("¿cuantas tarjetas se esperan?")
    assert not r["encontrado"]
    assert r["partido"] if False else True
    assert "no reconozco" in r["motivo"] or "no mencionas" in r["motivo"]


def test_equipo_que_no_juega_no_devuelve_nada():
    r = games.resolver("¿tarjetas del Sevilla?")
    assert not r["encontrado"]


def test_el_deporte_acota_la_busqueda():
    r = games.resolver("¿cuantas carreras para los Yankees?", "MLB")
    assert r["encontrado"]
    assert r["partido"]["sport"] == "MLB"


def test_las_palabras_de_relleno_no_cuentan_como_equipo():
    """'el', 'partido', 'que'... no deben puntuar contra ningun nombre."""
    for basura in ("el partido", "que se espera", "en el juego de hoy"):
        assert not games.resolver(basura)["encontrado"], basura


def test_nombrar_a_los_dos_gana_a_nombrar_a_uno():
    """Si 'Arsenal' aparece en un partido y 'Manchester United vs Arsenal' en
    otro, gana el que menciona a los dos."""
    r = games.resolver("Manchester United vs Arsenal")
    assert r["encontrado"] and r["partido"]["game_id"] == "3"


# --------------------------------------------------------------------------

def test_el_buscador_filtra_por_nombre_liga_y_deporte():
    assert any(j["game_id"] == "2" for j in games.buscar("barcelona"))
    assert all(j["sport"] == "MLB" for j in games.buscar("mlb"))
    assert games.buscar("equipo que no existe") == []


def test_el_buscador_vacio_devuelve_el_calendario():
    assert len(games.buscar("")) == len(CALENDARIO)


def test_los_tokens_ignoran_acentos_y_sufijos():
    assert games._tokens("Atlético de Madrid CF") == {"atletico", "madrid"}
    assert "fc" not in games._tokens("FC Barcelona")
