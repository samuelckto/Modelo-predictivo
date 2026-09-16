"""Alcance de la pregunta: QUE PERIODO y DE QUIEN. Version CHAT_ALCANCE_v1.

Los dos casos reales que obligaron a escribir esto, ambos del mismo partido
(Levante vs Barcelona):

    Usuario:  ¿cuantos goles le va a anotar el barcelona?
    Chat:     [contesta con el TOTAL del partido, 2.5 goles]      <-- mal

    Usuario:  ¿cuantos goles se esperan antes de la primera mitad?
    Chat:     TOTAL DE GOLES · OVER 2.5 · 56.4%                    <-- mal

En los dos casos el mercado que se consulto (`soccer.total_goals`) es correcto
por palabras y equivocado por ALCANCE:

  - el total son los goles de LOS DOS equipos, no los del Barcelona;
  - el total es del PARTIDO COMPLETO, no de la primera mitad.

Contestar el numero del partido completo a una pregunta de primera mitad no es
una aproximacion: es otro numero. Y es peor que negarse, porque el usuario no
tiene forma de notar el cambio de pregunta.

Aqui NO se calcula nada nuevo. Solo se comprueba si lo que el usuario pidio
cabe en lo que el mercado publica, y si no cabe se dice, ofreciendo lo que si
existe.
"""
from __future__ import annotations

import re
import unicodedata

VERSION = "CHAT_ALCANCE_v1"

# --------------------------------------------------------------------------
# Periodos
# --------------------------------------------------------------------------
PERIODOS = [
    ("primera_mitad", [
        r"\bprimer(?:a)? (?:tiempo|mitad|parte)\b", r"\b1(?:er|ra|°|º)? tiempo\b",
        r"\bprimera mitad\b", r"\bantes del descanso\b", r"\bal descanso\b",
        r"\bmedio tiempo\b", r"\bprimeros 45\b", r"\bprimer half\b", r"\b1t\b",
    ]),
    ("segunda_mitad", [
        r"\bsegund(?:o|a) (?:tiempo|mitad|parte)\b", r"\b2(?:do|da|°|º)? tiempo\b",
        r"\bsegunda mitad\b", r"\bdespues del descanso\b", r"\bultimos 45\b",
        r"\b2t\b",
    ]),
    ("primeros_5", [
        r"\bprimeras? (?:5|cinco)\b", r"\b(?:5|cinco) entradas\b", r"\bf5\b",
        r"\bprimeros? (?:5|cinco) innings\b",
    ]),
]

# Que periodo cubre cada mercado. Lo que no aparece cubre el PARTIDO COMPLETO.
COBERTURA = {"mlb.f5_total": "primeros_5"}

# Si el usuario pide un periodo y existe un mercado que SI lo cubre, no hay que
# negarse: hay que ir a ese mercado.
REDIRECCION = {("MLB", "primeros_5"): "mlb.f5_total"}

# Mercados que SI saben contestar por equipo (publican reparto local/visitante).
POR_EQUIPO_SOPORTADO = {"soccer.cards", "mlb.runs_dist"}

# Frases que hacen la pregunta sobre UN equipo, no sobre el partido.
ALCANCE_EQUIPO = [
    r"\ble va a (?:anotar|meter|marcar|hacer)\b", r"\bva a (?:anotar|meter|marcar)\b",
    r"\bcuant[oa]s? \w+ (?:anota|mete|marca|hace)\b",
    r"\banota el\b", r"\bmete el\b", r"\bmarca el\b",
    r"\bde parte del\b", r"\bpor parte del\b",
]

RUIDO = {"el", "la", "los", "las", "de", "del", "y", "que", "un", "una", "al",
         "a", "en", "con", "por", "para", "es", "su", "sus", "va", "ir", "le"}


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _toks(s) -> set[str]:
    return {t for t in _norm(s).split() if t not in RUIDO and len(t) > 2}


def periodo(texto: str) -> str | None:
    """Que periodo pide la pregunta. None = partido completo."""
    n = _norm(texto)
    for nombre, pats in PERIODOS:
        if any(re.search(p, n) for p in pats):
            return nombre
    return None


def equipo_objetivo(texto: str, partido: dict | None) -> str | None:
    """¿La pregunta es sobre UN equipo concreto del partido?

    Exige DOS cosas a la vez, y las dos importan:

      1. una frase que acote a un equipo ('le va a anotar el X', 'goles del X')
      2. que el nombre case con EXACTAMENTE uno de los dos participantes

    Con solo la (2) se rompia lo obvio: «¿cuantos goles en el Barcelona vs
    Levante?» nombra a los dos y es una pregunta del partido, no de un equipo.
    """
    if not partido:
        return None
    home, own = partido.get("home") or "", partido.get("away") or ""
    th, ta = _toks(home), _toks(own)
    tt = _toks(texto)
    ch, ca = th & tt, ta & tt
    if bool(ch) == bool(ca):        # los dos o ninguno -> es del partido
        return None
    nombre = home if ch else own
    n = _norm(texto)
    if any(re.search(p, n) for p in ALCANCE_EQUIPO):
        return nombre
    # 'goles del Barcelona', 'tarjetas de Levante': la preposicion justo antes
    # del nombre tambien acota. Sin ella, 'Barcelona gana' no es una pregunta
    # de conteo por equipo.
    tokens = (ch or ca)
    if any(re.search(rf"\bde(?:l)?\s+(?:\w+\s+)?{re.escape(t)}\b", n) for t in tokens):
        return nombre
    return None


# --------------------------------------------------------------------------

def _nombre_periodo(p: str) -> str:
    return {"primera_mitad": "la primera mitad", "segunda_mitad": "la segunda mitad",
            "primeros_5": "las primeras 5 entradas"}.get(p, p)


def revisar(texto: str, reg: dict, partido: dict | None = None) -> dict | None:
    """¿Cabe la pregunta en lo que publica este mercado?

    Devuelve None si cabe. Si no cabe devuelve el motivo y, cuando existe, el
    mercado al que habria que ir en su lugar.
    """
    if not reg:
        return None
    mid, sport = reg["market_id"], reg["sport"]

    per = periodo(texto)
    if per and COBERTURA.get(mid) != per:
        destino = REDIRECCION.get((sport, per))
        if destino and destino != mid:
            return {"tipo": "redirigir", "market_id": destino, "periodo": per}
        return {
            "tipo": "no_cubierto", "periodo": per, "market_id": mid,
            "motivo": (
                f"No tengo mercado de {_nombre_periodo(per)}. "
                f"«{reg['etiqueta']}» es del PARTIDO COMPLETO, y darte ese número "
                f"como si fuera el de {_nombre_periodo(per)} sería contestarte otra "
                f"pregunta sin avisarte.\n\n"
                f"Si quieres el del partido entero, pídemelo así y te lo doy."),
        }

    eq = equipo_objetivo(texto, partido)
    if eq and mid not in POR_EQUIPO_SOPORTADO:
        return {
            "tipo": "sin_desglose", "equipo": eq, "market_id": mid,
            "motivo": (
                f"«{reg['etiqueta']}» cuenta los de LOS DOS equipos juntos, así que "
                f"no puedo sacar de ahí los de {eq} solo: no hay modelo por equipo "
                f"publicado para este mercado, y repartir el total a ojo sería "
                f"inventarlo.\n\n"
                f"Lo que sí tengo del partido completo te lo doy si me lo pides sin "
                f"acotar a un equipo."),
        }
    return None
