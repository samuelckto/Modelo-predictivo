"""Tests del lector de marcadores set a set (el que permite calificar juegos).

El HTML es una copia reducida de una pagina real de TennisExplorer, con los
casos que importan: set con tiebreak ('68' = 6 juegos perdiendo el TB 6-8),
partido a 3 sets y partido sin marcador (abandono), que debe ignorarse.
"""
from __future__ import annotations

from TENIS.markets.results import _juegos, _te_apellido, _te_partidos

HTML = """
<table><tbody>
<tr class="one"><td>22:10</td><td><a>Rybakina E. (2)</a></td><td>2</td>
    <td>6</td><td>6</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>1.70</td><td>2.10</td></tr>
<tr class="two"><td><a>Osaka N. (13)</a></td><td>0</td>
    <td>1</td><td>4</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>
<tr class="one"><td>19:10</td><td><a>Romero Gormaz L. (4)</a></td><td>2</td>
    <td>68</td><td>7</td><td>7</td><td>&nbsp;</td><td>&nbsp;</td><td>1.52</td><td>2.40</td></tr>
<tr class="two"><td><a>Cengiz B.</a></td><td>1</td>
    <td>7</td><td>5</td><td>66</td><td>&nbsp;</td><td>&nbsp;</td></tr>
<tr class="one"><td>12:00</td><td><a>Nadie A.</a></td><td>&nbsp;</td>
    <td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>1.10</td><td>6.0</td></tr>
<tr class="two"><td><a>Otro B.</a></td><td>&nbsp;</td>
    <td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>
</tbody></table>
"""


def test_juegos_cuenta_bien_los_tiebreaks():
    # '68' es 6 juegos (perdio el tiebreak 6-8), no 68 juegos.
    assert _juegos("6") == 6
    assert _juegos("68") == 6
    assert _juegos("7") == 7
    assert _juegos("10") == 10          # super tiebreak
    assert _juegos("&nbsp;") is None
    assert _juegos("") is None


def test_apellido_con_iniciales_y_apellido_compuesto():
    assert _te_apellido("Rybakina E. (2)") == "rybakina"
    assert _te_apellido("Romero Gormaz L. (4)") == "gormaz"
    assert _te_apellido("Van De Zandschulp B.") == "zandschulp"


def test_partidos_terminados_con_juegos_reales():
    ms = _te_partidos(HTML)
    # el partido sin marcador no se inventa: queda fuera
    assert len(ms) == 2

    a = ms[0]
    assert (a["sets1"], a["sets2"]) == (2, 0)
    assert (a["juegos1"], a["juegos2"]) == (12, 5)      # 6-1 6-4
    assert a["n_sets"] == 2

    b = ms[1]
    assert (b["sets1"], b["sets2"]) == (2, 1)
    assert (b["juegos1"], b["juegos2"]) == (20, 18)     # 6-7 7-5 7-6
    assert b["juegos1"] + b["juegos2"] == 38            # total de juegos calificable
    assert b["juegos1"] - b["juegos2"] == 2             # margen para el handicap
