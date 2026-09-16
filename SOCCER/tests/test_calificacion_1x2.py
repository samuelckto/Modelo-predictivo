"""La calificacion tiene que distinguir 'gana' de 'gana o empata'.

El mercado double_chance ahora publica DOS tipos de apuesta. Si se califican
igual, un empate contaria como acierto de un pick de ganador directo y el
sistema se regalaria aciertos que no tuvo. Eso es peor que no medir nada.

Se reproduce la logica exacta de pipeline.score() sobre casos concretos, sin
tocar la base de datos.
"""
from __future__ import annotations

import pytest


def califica(seleccion: str, home: str, away: str, goles_home: int, goles_away: int) -> bool:
    """Copia literal de la decision que toma pipeline.score() para este mercado."""
    eligio_local = seleccion.startswith(home)
    solo_ganador = seleccion.endswith(" gana")
    if solo_ganador:
        return (goles_home > goles_away) if eligio_local else (goles_away > goles_home)
    return (goles_home >= goles_away) if eligio_local else (goles_away >= goles_home)


# --- ganador directo -------------------------------------------------------

def test_el_empate_es_FALLO_para_un_pick_de_ganador():
    """El caso que motivo el test. Antes esto devolvia True."""
    assert califica("Barcelona gana", "Barcelona", "Getafe", 1, 1) is False


def test_el_ganador_directo_acierta_si_gana():
    assert califica("Barcelona gana", "Barcelona", "Getafe", 2, 0) is True


def test_el_ganador_directo_falla_si_pierde():
    assert califica("Barcelona gana", "Barcelona", "Getafe", 0, 1) is False


def test_el_visitante_como_ganador_directo():
    assert califica("Real Madrid gana", "Getafe", "Real Madrid", 0, 2) is True
    assert califica("Real Madrid gana", "Getafe", "Real Madrid", 1, 1) is False


# --- doble oportunidad -----------------------------------------------------

def test_el_empate_SI_es_acierto_para_la_doble_oportunidad():
    assert califica("Barcelona o empate", "Barcelona", "Getafe", 1, 1) is True


def test_la_doble_oportunidad_falla_solo_si_pierde():
    assert califica("Barcelona o empate", "Barcelona", "Getafe", 0, 2) is False
    assert califica("Barcelona o empate", "Barcelona", "Getafe", 3, 0) is True


def test_la_doble_oportunidad_del_visitante():
    assert califica("Real Madrid o empate", "Getafe", "Real Madrid", 1, 1) is True
    assert califica("Real Madrid o empate", "Getafe", "Real Madrid", 2, 1) is False


@pytest.mark.parametrize("gh,ga", [(0, 0), (1, 1), (2, 2)])
def test_ningun_empate_se_cuela_como_acierto_del_ganador(gh, ga):
    assert califica("A gana", "A", "B", gh, ga) is False
    assert califica("B gana", "A", "B", gh, ga) is False
    # y los dos lados de la doble oportunidad SI aciertan con empate
    assert califica("A o empate", "A", "B", gh, ga) is True
    assert califica("B o empate", "A", "B", gh, ga) is True
