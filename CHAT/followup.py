"""Repreguntas y conversacion. Version CHAT_FOLLOWUP_v1.

El problema que resuelve, con el caso real que lo motivo:

    Caballo Chat:  Seleccion: Elena Rybakina
                   Probabilidad del modelo (calibrada): 29.2%
    Usuario:       entonces elena solo tiene el 29% de ganar?
    Caballo Chat:  no reconozco ni el mercado ni el tipo de pregunta.  <-- mal

El numero estaba en pantalla. Lo que faltaba no era el dato: era MEMORIA. Un
chat que no recuerda lo que acaba de decir no es un chat, es un formulario.

Aqui no se inventa nada nuevo. Todo lo que se contesta sale de:

  - el CONTEXTO de la respuesta anterior (que mercado, que partido, que cifras)
  - el complemento aritmetico cuando el mercado es de dos vias
  - los participantes reales del partido

Si una repregunta necesita un numero que no esta en ninguno de esos tres
sitios, se dice. La conversacion se vuelve natural; la disciplina con los
numeros no se toca.
"""
from __future__ import annotations

import re
import unicodedata

VERSION = "CHAT_FOLLOWUP_v1"

# Mercados de DOS VIAS excluyentes: el complemento (1 - p) es el otro lado y se
# puede afirmar sin calcular nada nuevo. Fuera de esta lista NO se complementa:
# en 'gana o empata' el complemento no es "pierde el otro", y en un over/under
# con linea entera existe el push.
DOS_VIAS = {"tenis.winner", "mlb.moneyline", "nba.moneyline"}

# Repreguntas sobre la respuesta anterior.
SEGUIMIENTO = [
    r"^\s*(entonces|osea|o sea|asi que|as[ií] que|y entonces)\b",
    r"\bsolo tiene\b", r"\bnada mas tiene\b", r"\bnada m[aá]s tiene\b",
    r"\bes correcto\b", r"\bes as[ií]\b", r"\bes eso\b", r"\bseguro\b",
    r"\bde verdad\b", r"\ben serio\b", r"\bconfirma\b",
]

# Dos preguntas que NO son la misma y no merecen la misma respuesta:
#
#   MAGNITUD  "¿es mucho 24%?"      -> ¿que significa ese numero?
#   CONSEJO   "¿conviene apostarle?" -> ¿me va a hacer ganar dinero?
#
# La segunda exige el aviso sobre el ROI; la primera no, y soltarselo igual
# convierte una pregunta legitima en un sermon.
MAGNITUD = [r"\bes mucho\b", r"\bes poco\b", r"\bes alto\b", r"\bes bajo\b",
            r"\bque tan\b", r"\bqu[eé] tan\b", r"\bes normal\b", r"\bes raro\b"]
CONSEJO = [r"\bes bueno\b", r"\bes buena\b", r"\bes mala\b", r"\bvale la pena\b",
           r"\bconviene\b", r"\bapost", r"\ble entro\b", r"\brecomiendas\b",
           r"\bconf[ií]o\b", r"\bme fio\b", r"\bme f[ií]o\b", r"\bjuego\b",
           r"\barriesgo\b"]
INTERPRETACION = MAGNITUD + CONSEJO

# Una PREGUNTA NUEVA no es una repregunta, aunque nombre a un equipo del que ya
# estabamos hablando. Caso real:
#
#   Chat:     TOTAL DE GOLES · OVER 2.5 · 56.4%
#   Usuario:  ¿cuantos goles le va a anotar el barcelona?
#   Chat:     "De Barcelona no puedo darte el numero restando..."   <-- absurdo
#
# El usuario no pidio restar nada: hizo una pregunta nueva. Bastaba con que
# nombrara a uno de los dos equipos para que esta capa la secuestrara y la
# tratara como «¿y el otro lado?». Si la frase trae su propio interrogativo,
# esto no es cosa nuestra.
NUEVA_PREGUNTA = [
    r"\bcu[aá]nt[oa]s?\b", r"\bqu[eé] probabilidad\b", r"\bcu[aá]l es\b",
    r"\bqui[eé]n gana\b", r"\bexpl[ií]ca", r"\bpor qu[eé]\b", r"\bpor que\b",
    r"\bqu[eé] dice\b", r"\bqu[eé] tienes\b", r"\bdame\b", r"\bme das\b",
    r"\bhabr[aá]\b", r"\bva a haber\b", r"\bse esperan\b", r"\banot",
    r"\bmercado\b", r"\bcuota\b", r"\bmomio\b", r"\btarjetas\b", r"\bcorners\b",
]

# "¿y el otro?", "¿y Qinwen?"
OTRO_LADO = [r"\by el otro\b", r"\bel otro\b", r"\bla otra\b", r"\bel rival\b",
             r"\bel contrario\b", r"\bel oponente\b", r"\bqui[eé]n m[aá]s\b"]

RUIDO = {"el", "la", "los", "las", "de", "del", "y", "que", "tiene", "solo",
         "nada", "mas", "más", "entonces", "osea", "sea", "por", "para", "con",
         "un", "una", "es", "esta", "este", "su", "sus", "al", "a", "en", "ganar",
         "gana", "ganara", "chance", "probabilidad", "porcentaje", "asi", "así"}


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _toks(s) -> set[str]:
    return {t for t in _norm(s).split() if t not in RUIDO and len(t) > 2}


def es_seguimiento(t: str) -> bool:
    n = _norm(t)
    return any(re.search(p, n) for p in SEGUIMIENTO)


def pide_interpretacion(t: str) -> bool:
    n = _norm(t)
    return any(re.search(p, n) for p in INTERPRETACION)


def pide_consejo(t: str) -> bool:
    n = _norm(t)
    return any(re.search(p, n) for p in CONSEJO)


def pide_magnitud(t: str) -> bool:
    n = _norm(t)
    return any(re.search(p, n) for p in MAGNITUD)


def pide_el_otro_lado(t: str) -> bool:
    n = _norm(t)
    return any(re.search(p, n) for p in OTRO_LADO)


def es_pregunta_nueva(t: str) -> bool:
    """¿La frase trae su propio interrogativo? Entonces no es una repregunta."""
    n = _norm(t)
    return any(re.search(p, n) for p in NUEVA_PREGUNTA)


def porcentaje_mencionado(t: str):
    """El numero que cita el usuario: 'solo tiene el 29%' -> 0.29."""
    m = re.search(r"(\d{1,3}(?:[.,]\d+)?)\s*%", str(t))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v / 100 if v > 1 else v


def participante(texto: str, partido: dict | None):
    """¿A quien nombra el usuario, al local o al visitante?

    Compara por TOKENS del nombre. 'elena' encaja con 'Elena Rybakina' pero no
    con 'Qinwen Zheng', y si no encaja con ninguno se devuelve None en vez de
    elegir el mas parecido: contestar del jugador equivocado es peor que
    preguntar de cual se habla.
    """
    if not partido:
        return None
    tt = _toks(texto)
    if not tt:
        return None
    home, away = partido.get("home") or "", partido.get("away") or ""
    ch, ca = _toks(home) & tt, _toks(away) & tt
    if ch and not ca:
        return {"lado": "home", "nombre": home}
    if ca and not ch:
        return {"lado": "away", "nombre": away}
    return None


def _pct(p):
    return None if p is None else f"{float(p):.1%}"


def _complemento(ctx: dict):
    """El otro lado, SOLO si el mercado es de dos vias excluyentes."""
    if ctx.get("market_id") not in DOS_VIAS:
        return None
    p = ctx.get("p_cal")
    return None if p is None else 1.0 - float(p)


def _otro_nombre(ctx: dict):
    par = ctx.get("partido") or {}
    sel = _norm(ctx.get("selection"))
    home, away = par.get("home"), par.get("away")
    if not home or not away:
        return None
    if _toks(sel) & _toks(home):
        return away
    if _toks(sel) & _toks(away):
        return home
    return None


# --------------------------------------------------------------------------

def responder(texto: str, ctx: dict) -> dict | None:
    """Contesta una repregunta usando el contexto. None si no aplica.

    `ctx` es lo que el chat dijo la vez anterior: mercado, partido, seleccion y
    cifras. Sin contexto no hay repregunta posible y se devuelve None para que
    el flujo normal siga su curso.
    """
    if not ctx or not ctx.get("market_id"):
        return None
    n = _norm(texto)
    seguimiento = es_seguimiento(texto)
    interpretar = pide_interpretacion(texto)
    otro = pide_el_otro_lado(texto)
    quien = participante(texto, ctx.get("partido"))
    citado = porcentaje_mencionado(texto)

    nueva = es_pregunta_nueva(texto)
    # Una pregunta nueva solo sigue aqui si ademas cita una cifra nuestra o
    # empieza como repregunta ("entonces..."): eso si es hablar de lo anterior.
    if nueva and not (seguimiento or citado is not None):
        return None

    # Nombrar a un participante NO es, por si solo, una repregunta. Solo cuenta
    # cuando la frase es una mencion suelta ("¿y Qinwen?"), que es como se pide
    # el otro lado sin decir "el otro".
    mencion_suelta = (quien is not None and not nueva
                      and len(_norm(texto).split()) <= 4)

    if not (seguimiento or interpretar or otro or mencion_suelta
            or citado is not None):
        return None

    base = {"version": VERSION, "ok": True, "tipo": "seguimiento",
            "pregunta": texto, "intencion": "seguimiento",
            "market_id": ctx.get("market_id"), "etiqueta": ctx.get("etiqueta"),
            "sport": ctx.get("sport"), "game_id": ctx.get("game_id"),
            "partido": ctx.get("partido"), "model_version": ctx.get("model_version"),
            "calibracion": ctx.get("calibracion"), "status": ctx.get("status"),
            "contexto": ctx}

    p_cal, p_mkt = ctx.get("p_cal"), ctx.get("p_market")
    sel = ctx.get("selection")
    comp = _complemento(ctx)
    nombre_otro = _otro_nombre(ctx)

    # --- "¿y el otro?" / "¿y Qinwen?" -----------------------------------
    if otro or (quien and sel and not (_toks(sel) & _toks(quien["nombre"]))):
        objetivo = quien["nombre"] if quien else nombre_otro
        if comp is None:
            base["respuesta"] = (
                f"De {objetivo or 'el otro lado'} no puedo darte el número restando: "
                f"{ctx.get('etiqueta')} no es un mercado de dos vías excluyentes, así "
                f"que 1 menos la probabilidad no es la del rival. Pregúntame por ese "
                f"lado y lo busco.")
            base["ok"] = False
            base["status"] = "NO MARKET"
            return base
        base["respuesta"] = (
            f"{objetivo}: {_pct(comp)} según el modelo.\n\n"
            f"Es el complemento directo: {sel} tiene {_pct(p_cal)} y el mercado es "
            f"de dos vías, así que entre los dos suman 100%.")
        base["numeros"] = {"lado": objetivo, "modelo": round(comp, 4)}
        return base

    # --- "entonces solo tiene el 29%?" ----------------------------------
    if citado is not None and p_cal is not None:
        cerca = abs(citado - float(p_cal)) < 0.015
        lineas = []
        if cerca:
            lineas.append(f"Sí, eso es. {sel}: {_pct(p_cal)} según el modelo.")
        else:
            lineas.append(f"No exactamente. El modelo da {_pct(p_cal)} a {sel}, "
                          f"no {citado:.1%}.")
        if comp is not None and nombre_otro:
            lineas.append(f"Dicho al revés: {nombre_otro} tiene {_pct(comp)}.")
        if p_mkt is not None:
            lineas.append(f"El mercado le da {_pct(p_mkt)}, así que el modelo y la "
                          f"casa están de acuerdo ({ctx.get('gap_pp', 0):+.1f} pp de "
                          f"diferencia).")
        base["respuesta"] = "\n\n".join(lineas)
        return base

    # --- "¿es mucho?", "¿conviene?" -------------------------------------
    if interpretar or seguimiento:
        lineas = []
        if p_cal is not None:
            lineas.append(f"{sel}: {_pct(p_cal)} según el modelo"
                          + (f", {_pct(p_mkt)} según el mercado." if p_mkt is not None
                             else ", sin cuota con la que compararlo."))
        if comp is not None and nombre_otro:
            lineas.append(f"El otro lado, {nombre_otro}, queda en {_pct(comp)}.")

        # "¿es mucho?" pregunta que SIGNIFICA el numero, no si apostarlo.
        if pide_magnitud(texto) and not pide_consejo(texto) and p_cal is not None:
            p = float(p_cal)
            veces = round(p * 100)
            lineas.append(
                f"Para leerlo: de cada 100 partidos en los que el modelo dice "
                f"~{p:.0%}, espera que ocurra unas {veces} veces. "
                + ("Es un lado claramente perdedor según el modelo."
                   if p < 0.35 else
                   "Es prácticamente un volado." if p < 0.55 else
                   "Es el lado favorito, sin ser una barbaridad." if p < 0.75 else
                   "Es un favorito muy marcado."))
            if p_mkt is not None:
                d = (p - float(p_mkt)) * 100
                lineas.append(
                    f"Y la casa piensa casi lo mismo: {_pct(p_mkt)} frente a "
                    f"{_pct(p_cal)} del modelo ({d:+.1f} pp)."
                    if abs(d) < 2.5 else
                    f"La casa lo ve distinto: {_pct(p_mkt)} frente a {_pct(p_cal)} "
                    f"del modelo ({d:+.1f} pp de diferencia).")

        # La parte honesta: que significa ese numero y que NO significa.
        if pide_consejo(texto):
            lineas.append(
                "Sobre si conviene apostarlo: no te lo puedo decir. El sistema no "
                "tiene histórico de cuotas suficiente para demostrar que ninguna de "
                "sus probabilidades gane dinero después del margen de la casa. "
                "Hoy solo hay 11 apuestas liquidadas con cuota registrada en todo "
                "el histórico; hacen falta unas 100 por tramo para que el ROI "
                "signifique algo. Lo que sí te puedo decir es qué tan buenas son "
                "las probabilidades, y eso está medido.")
            # OJO: `.get("evidencia", {})` NO basta. La clave existe con valor
            # None en los mercados sin evidencia registrada (tenis), y el default
            # solo se aplica cuando la clave falta. Hay que colapsar el None.
            e = ctx.get("evidencia") or {}
            if e.get("holdout_log_loss") is not None:
                lineas.append(
                    f"Fuera de muestra este modelo saca {e['holdout_log_loss']} de "
                    f"log loss frente a {e['baseline']} de su baseline"
                    + (f", con {e['accuracy']:.1%} de acierto frente a "
                       f"{e['baseline_accuracy']:.1%}." if e.get("accuracy") else "."))
        base["respuesta"] = "\n\n".join(lineas) if lineas else (
            "No tengo más detalle guardado de esa respuesta. Vuelve a preguntarme "
            "por el mercado y te lo saco otra vez.")
        base["aviso"] = ("Una probabilidad del modelo por encima de la del mercado "
                         "NO es valor mientras no haya histórico que lo demuestre.")
        return base

    return None
