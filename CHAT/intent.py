"""Deteccion de intencion en lenguaje natural. Version CHAT_INTENT_v1.

Reglas explicitas, no un modelo de lenguaje. Es una decision, no una limitacion:
un clasificador entrenado acertaria mas veces en la superficie pero fallaria en
silencio, y aqui un fallo silencioso significa contestar con el modelo
equivocado. Con reglas, cuando no se entiende algo se sabe que no se entendio y
se puede decir.

Lo que se extrae de una frase:

    intencion   probabilidad / distribucion / comparacion / explicacion /
                esperado / catalogo / desconocida
    sport       si se puede deducir
    market_id   consultando el registro, nunca adivinando
    linea       el numero que acompana a over/under o mas de/menos de
    lado        over o under
    equipo      nombre suelto que pueda ser un equipo o jugador

Si no se identifica el mercado, se devuelve `desconocida` con los candidatos.
Preferimos "no tengo un modelo validado para eso" a un porcentaje inventado.
"""
from __future__ import annotations

import re
import unicodedata

from CHAT import registry

VERSION = "CHAT_INTENT_v1"

INTENCIONES = ("probabilidad", "distribucion", "esperado", "comparacion",
               "explicacion", "catalogo", "saludo", "resumen", "desconocida")

# Saludos y cortesia. Existen como intencion propia porque contestar
# "no tengo un modelo validado para eso, estado BLOCKED" a un "hola" no es
# rigor: es una interfaz mal hecha. La regla de no inventar aplica a las
# PROBABILIDADES, no a la conversacion.
SALUDOS = [r"^\s*(hola|holi|buenas|buenos dias|buenas tardes|buenas noches|hey|"
           r"que tal|que onda|qué onda|saludos|hi|hello|ey)\b",
           r"^\s*(gracias|muchas gracias|ok|vale|perfecto|genial|listo|adios|"
           r"hasta luego|bye|chao)\s*[.!]*\s*$"]

# "¿que tienes de este partido?" -> resumen de TODO lo que hay del partido.
RESUMEN = [r"\bresumen\b", r"\bque tienes\b", r"\bqu[eé] tienes\b",
           r"\bque hay\b", r"\bqu[eé] hay\b", r"\bdime todo\b", r"\btodo\b",
           r"\banaliza\b", r"\banalisis\b", r"\ban[aá]lisis\b",
           r"\bque sabes de\b", r"\bqu[eé] sabes de\b", r"\bcomo ves\b",
           r"\bc[oó]mo ves\b", r"\bque opinas\b", r"\bqu[eé] opinas\b",
           r"\beste partido\b", r"\bdel partido\b"]

DEPORTES = {
    "MLB": ["mlb", "beisbol", "baseball", "carreras", "entradas", "innings",
            "pitcher", "abridor", "bateo"],
    "SOCCER": ["futbol", "soccer", "goles", "gol", "tarjetas", "amarillas",
               "corners", "esquinas", "liga mx", "premier", "laliga", "champions",
               "serie a", "bundesliga", "ligue 1"],
    "NFL": ["nfl", "futbol americano", "touchdown", "yardas"],
    "NBA": ["nba", "basquet", "basket", "baloncesto", "canastas"],
    "TENIS": ["tenis", "tennis", "sets", "juegos", "atp", "wta"],
}

# Frases que piden explicitamente CADA intencion. El orden importa: se evalua
# de la mas especifica a la mas general.
PATRONES = [
    ("explicacion", [r"\bpor qu[eé]\b", r"\bpor que\b", r"\bc[oó]mo llegas\b",
                     r"\bexpl[ií]ca", r"\ben qu[eé] te basas\b", r"\bde d[oó]nde sale\b"]),
    ("comparacion", [r"\bcasa de apuestas\b", r"\bcasa\b", r"\bmercado\b",
                     r"\bmomio\b", r"\bcuota\b", r"\bfavorito\b", r"\bimpl[ií]cita\b",
                     r"\bdiferencia con\b", r"\bvs el mercado\b"]),
    # OJO con "que sabes": "¿que sabes hacer?" es el catalogo, pero
    # "¿que sabes de este partido?" es un resumen. Sin el "hacer" la frase se
    # comia las preguntas abiertas sobre un partido y devolvia la lista de
    # mercados en vez de los numeros que el usuario pedia.
    ("catalogo", [r"\bqu[eé] puedes\b", r"\bqu[eé] sabes hacer\b",
                  r"\bqu[eé] mercados\b", r"\bqu[eé] m[aá]s puedes\b",
                  r"\bopciones\b", r"\bayuda\b"]),
    ("distribucion", [r"\bdistribuci[oó]n\b", r"\bprobabilidad de que .* anote\b",
                      r"\bal menos\b", r"\bexactamente\b", r"\bambos\b",
                      r"\blos dos\b", r"\bcada equipo\b"]),
    ("esperado", [r"\bcu[aá]nt[oa]s? .* se esperan\b", r"\bcu[aá]nt[oa]s\b",
                  r"\besperad[oa]s?\b", r"\bpromedio\b", r"\bmedia\b"]),
    ("probabilidad", [r"\bprobabilidad\b", r"\bqu[eé] tanto\b", r"\bchance\b",
                      r"\bhabr[aá]\b", r"\bva a haber\b", r"\bover\b", r"\bunder\b",
                      r"\bm[aá]s de\b", r"\bmenos de\b", r"\bqui[eé]n gana\b",
                      r"\bgana\b", r"\bcubre\b"]),
]

LADO = [("over", [r"\bover\b", r"\bm[aá]s de\b", r"\bencima\b", r"\barriba de\b"]),
        ("under", [r"\bunder\b", r"\bmenos de\b", r"\bdebajo\b", r"\bpor debajo\b"])]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t.lower()).strip()


def _deporte(t: str) -> str | None:
    puntos = {}
    for s, palabras in DEPORTES.items():
        p = sum(len(w) for w in palabras if w in t)
        if p:
            puntos[s] = p
    return max(puntos, key=puntos.get) if puntos else None


def _linea(t: str):
    """El numero de una linea: 'over 2.5', 'mas de 8.5', 'over 4 tarjetas'."""
    m = re.search(r"(?:over|under|mas de|menos de|linea|encima de|debajo de)\s+"
                  r"(\d+(?:[.,]\d+)?)", t)
    if m:
        return float(m.group(1).replace(",", "."))
    # 'over 2.5 goles' ya cubierto; ahora un decimal suelto tipico de linea
    m = re.search(r"\b(\d+\.5)\b", t)
    return float(m.group(1)) if m else None


def _cantidad(t: str):
    """'¿que probabilidad hay de que Boston anote 4 carreras?' -> 4."""
    m = re.search(r"anot[ea]n?\s+(?:al menos\s+)?(\d+)", t)
    return int(m.group(1)) if m else None


def detectar(texto: str, sport_por_defecto: str | None = None) -> dict:
    """Interpreta la pregunta. Nunca adivina un mercado que el registro no tenga."""
    crudo = str(texto or "")
    t = _norm(crudo)
    out = {"version": VERSION, "texto": crudo, "texto_normalizado": t,
           "intencion": "desconocida", "sport": None, "market_id": None,
           "linea": _linea(t), "lado": "over", "cantidad": _cantidad(t),
           "candidatos": [], "motivo": None, "mercado_desconocido": False}
    if not t:
        out["motivo"] = "la pregunta esta vacia."
        return out

    for lado, pats in LADO:
        if any(re.search(p, t) for p in pats):
            out["lado"] = lado
            break

    out["sport"] = _deporte(t) or sport_por_defecto

    # El saludo se comprueba PRIMERO y corta: "hola, que tal" no debe caer en
    # ningun patron de mercado por casualidad.
    if any(re.search(p, t) for p in SALUDOS):
        out["intencion"] = "saludo"
        return out

    for nombre, pats in PATRONES:
        if any(re.search(p, t) for p in pats):
            out["intencion"] = nombre
            break

    cands = registry.buscar(t, out["sport"])
    if not cands and out["sport"]:
        cands = registry.buscar(t)                 # el deporte puede estar mal deducido
    out["candidatos"] = [d["market_id"] for d in cands[:5]]
    if cands:
        out["market_id"] = cands[0]["market_id"]
        out["sport"] = out["sport"] or cands[0]["sport"]
        if out["intencion"] == "desconocida":
            # Se reconoce el mercado pero no el verbo: preguntar por un mercado
            # sin mas suele significar "dame la probabilidad".
            out["intencion"] = ("distribucion"
                                if cands[0]["distribution_type"] != "binaria"
                                and out["linea"] is None and out["cantidad"] is not None
                                else "probabilidad")
    elif out["intencion"] == "catalogo":
        # Preguntar "¿que puedes contestar?" no menciona ningun mercado, y eso
        # es correcto: no falta nada. Marcarlo como no entendido contradecia la
        # propia respuesta, que si era la buena.
        pass
    elif out["intencion"] == "desconocida" and any(re.search(p, t) for p in RESUMEN):
        # Pregunta abierta sobre un partido: "¿que tienes de esto?". No falta
        # nada, se quiere TODO lo que haya.
        #
        # Solo se aplica si NO se entendio ya otra cosa. "¿que dice el mercado
        # de este partido?" contiene "este partido", pero su intencion es
        # comparar con el mercado y pisarla la perdia. Cuando hay partido
        # seleccionado el resumen igual acaba saliendo, porque una intencion
        # clara sin mercado concreto cae en la misma rama.
        out["intencion"] = "resumen"
    elif out["intencion"] != "desconocida":
        # Se entendio QUE se pregunta pero no DE QUE mercado. La intencion se
        # CONSERVA: perderla tambien impedia distinguir "explicame esto" de
        # "no entiendo nada", y las dos merecen respuestas distintas.
        out["mercado_desconocido"] = True
        out["motivo"] = ("entiendo QUE me preguntas pero no DE QUE mercado. "
                         "No hay ninguno en el registro que case con esas palabras.")
    else:
        out["mercado_desconocido"] = True
        out["motivo"] = "no reconozco ni el mercado ni el tipo de pregunta."
    return out
