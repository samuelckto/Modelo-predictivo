"""Valor estimado y desacuerdo con el mercado. Version VALOR_v1.

Dos preguntas distintas que la gente mezcla, y que aqui se responden por
separado porque tienen respuestas distintas:

  1. ¿El modelo y la casa apuntan a LADOS OPUESTOS?
     Es un hecho observable: el modelo da favorito a uno y la casa al otro.
     No hace falta demostrar nada para afirmarlo.

  2. ¿Eso da dinero?
     Eso NO se sabe. Con el historico de cuotas que hay hoy (11 apuestas
     liquidadas con precio registrado) no se puede demostrar que ir contra la
     casa gane, ni que pierda.

Por eso este modulo calcula y etiqueta, pero NUNCA llama 'valor validado' a
nada. El EV que devuelve es una ESTIMACION que depende por completo de que la
probabilidad del modelo este bien calibrada; si no lo esta, el EV es un numero
bonito y falso.
"""
from __future__ import annotations

VERSION = "VALOR_v1"

# El margen de la casa. Un mercado de dos vias con vigorish tipico suma ~105 %
# de probabilidad implicita: ese 5 % es su comision. La mitad (2.5 pp) es lo que
# se le puede atribuir a UNA seleccion, asi que una diferencia menor que eso no
# es desacuerdo: es el margen.
VIG_TIPICO_PP = 2.5

# Tramos de desacuerdo. Anclados al margen real, no inventados.
LEVE = VIG_TIPICO_PP          # por debajo: estan de acuerdo
MODERADO = 5.0
FUERTE = 10.0


def decimal_desde_americana(a) -> float | None:
    """Cuota americana -> decimal. None si el dato no es una cuota real.

    Una cuota americana no existe entre -100 y +100. Aparecen valores asi en los
    datos y convertirlos inventaria ganancias enormes donde solo hay un dato roto.
    """
    if a is None:
        return None
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    if abs(a) < 100:
        return None
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def _decimal(cuota) -> float | None:
    """Acepta decimal (>= 1.01) o americana y devuelve decimal."""
    if cuota is None:
        return None
    try:
        c = float(cuota)
    except (TypeError, ValueError):
        return None
    if 1.01 <= c <= 50:
        return c
    return decimal_desde_americana(c)


def evaluar(p_modelo, p_mercado=None, cuota=None) -> dict:
    """Todo lo que se puede decir de una seleccion frente al mercado.

    Devuelve siempre las mismas claves, con None donde no hay dato. Nunca
    inventa la cuota ni la probabilidad del mercado: sin ellas, simplemente no
    hay comparacion que hacer y se dice.
    """
    out = {
        "version": VERSION,
        "gap_pp": None,             # modelo - mercado, en puntos porcentuales
        "tramo": None,              # de acuerdo / leve / moderado / fuerte
        "contra_mercado": False,    # ¿lados opuestos?
        "cuota_decimal": None,
        "ev": None,                 # ganancia esperada por unidad apostada
        "ev_pct": None,
        "validado": False,          # SIEMPRE False: no hay muestra para validar
        "aviso": None,
    }
    if p_modelo is None:
        return out

    dec = _decimal(cuota)
    out["cuota_decimal"] = dec

    if p_mercado is not None:
        gap = (float(p_modelo) - float(p_mercado)) * 100
        out["gap_pp"] = round(gap, 2)
        a = abs(gap)
        out["tramo"] = ("de acuerdo" if a < LEVE else "leve" if a < MODERADO
                        else "moderado" if a < FUERTE else "fuerte")
        # Lados opuestos: el modelo cree que la seleccion ocurre y la casa cree
        # que no (o al reves). El 50 % es la frontera en un mercado de dos vias.
        out["contra_mercado"] = ((float(p_modelo) - 0.5) * (float(p_mercado) - 0.5)) < 0

    if dec is not None:
        ev = float(p_modelo) * dec - 1
        out["ev"] = round(ev, 4)
        out["ev_pct"] = round(ev * 100, 2)
        out["aviso"] = (
            "EV estimado: sale de multiplicar la probabilidad del modelo por la "
            "cuota. Solo vale si esa probabilidad esta bien calibrada, y eso no "
            "esta demostrado con el historico de cuotas que hay hoy.")
    elif p_mercado is not None:
        out["aviso"] = ("Sin cuota registrada no se puede calcular valor: la "
                        "diferencia con el mercado es informativa, nada mas.")
    return out


def ordenar_por_valor(items, clave=lambda x: x) -> list:
    """Ordena de mayor a menor EV. Los que no tienen cuota van al final.

    No se les asigna EV 0 ni se les estima uno: 'no se sabe' y 'da cero' son
    cosas distintas y mezclarlas colocaria un desconocido por delante de una
    perdida conocida.
    """
    con = [x for x in items if (clave(x) or {}).get("ev") is not None]
    sin = [x for x in items if (clave(x) or {}).get("ev") is None]
    con.sort(key=lambda x: clave(x)["ev"], reverse=True)
    return con + sin
