"""Respuestas estructuradas del chat. Version CHAT_ANSWER_v1.

Junta deteccion de intencion + API interna y devuelve una respuesta con la
misma forma siempre, para que la interfaz no tenga que adivinar nada.

Las dos reglas que este archivo hace cumplir, y que son la razon de que exista
en vez de dejar que el chat lea la base:

1. NUNCA se contesta "quien gana" mirando quien es favorito en la casa. Se
   contesta con el modelo, y DESPUES se ensena la comparacion. Si el modelo y
   el mercado discrepan, se muestra la discrepancia; no se ajusta el modelo
   hacia el mercado ni se presenta el favorito del casino como respuesta.

2. Sin modelo validado no hay porcentaje. Ni aproximado, ni "mas o menos", ni
   derivado de una media. La respuesta correcta es decir que no lo hay.
"""
from __future__ import annotations

from CHAT import alcance, api, followup, games, intent, knowledge, llm, registry

VERSION = "CHAT_ANSWER_v1"
NOMBRE = "Caballo Chat"


def _contexto(reg, p, partido, det) -> dict:
    """Lo que hay que recordar para poder contestar la SIGUIENTE pregunta.

    Sin esto, "entonces solo tiene el 29%?" era imposible de contestar aunque el
    29 % estuviera en pantalla: el chat no guardaba de que estaba hablando.
    """
    return {
        "market_id": reg["market_id"], "etiqueta": reg["etiqueta"],
        "sport": reg["sport"], "game_id": str(p.get("event_id") or ""),
        "partido": partido, "selection": p.get("selection"),
        "p_cal": p.get("MODEL_PROBABILITY_CALIBRATED"),
        "p_raw": p.get("MODEL_PROBABILITY_RAW"),
        "p_market": p.get("MARKET_IMPLIED_PROBABILITY"),
        "gap_pp": p.get("gap_pp"), "status": p.get("status"),
        "model_version": p.get("model_version"),
        "calibracion": p.get("calibration_version") or "sin_calibrar",
        "evidencia": reg.get("evidencia"), "line": p.get("line"),
    }


def _sin_modelo(texto, det, extra=None):
    return {"version": VERSION, "ok": False, "tipo": "sin_modelo",
            "pregunta": texto, "intencion": det.get("intencion"),
            "respuesta": ("No tengo un modelo validado para eso todavia. "
                          "Prefiero decirtelo a darte un numero inventado."),
            "detalle": extra, "candidatos": det.get("candidatos"),
            "status": "BLOCKED"}


def _pct(p):
    return None if p is None else f"{float(p):.1%}"


def _partido_de(game_id, sport):
    if not game_id:
        return None
    for j in games.listar(10, sport):
        if j["game_id"] == str(game_id):
            return j
    return None


def _saludo(texto, det, game_id, sport) -> dict:
    """Contestar a un saludo. No hace falta ningun modelo para esto."""
    p = _partido_de(game_id, sport)
    if p:
        r = _resumen(texto, det, p, game_id, p["sport"])
        cabeza = (f"Hola. Tienes seleccionado {p['nombre']}. Esto es lo que tengo:"
                  if r["ok"] else
                  f"Hola. Tienes seleccionado {p['nombre']}, pero")
        r["respuesta"] = f"{cabeza}\n\n{r['respuesta']}"
        r["tipo"] = "saludo"
        return r
    juegos = games.listar(10)
    return {
        "version": VERSION, "ok": True, "tipo": "saludo", "pregunta": texto,
        "intencion": "saludo", "status": None,
        "respuesta": ("Hola. Soy Caballo Chat: consulto los modelos del centro y te "
                      "doy sus números, no los del casino.\n\n"
                      "Elige un partido arriba o nómbralo en la pregunta "
                      "(«¿tarjetas del Barcelona?») y te digo todo lo que tengo. "
                      "Si algo no tiene modelo validado, te lo digo en vez de "
                      "inventarme un porcentaje."),
        "partidos": juegos[:8],
    }


def _resumen(texto, det, partido, game_id, sport) -> dict:
    """TODO lo que hay de un partido, mercado por mercado.

    Recorre el registro entero de ese deporte en vez de esperar a que el usuario
    acierte con el nombre del mercado. Los que no tienen dato aparecen igual,
    con su motivo: saber que NO se puede contestar tambien es informacion.
    """
    lineas, disponibles, bloqueados = [], [], []
    for reg in registry.listar(sport):
        mid = reg["market_id"]
        if mid in api.DISTRIBUCIONES:
            d = api.get_distribution(game_id, mid)
            if not d.get("disponible"):
                bloqueados.append((reg, d.get("motivo")))
                continue
            esperado = d.get("expected_cards", d.get("expected_runs"))
            bloque = [f"▸ {reg['etiqueta'].upper()}"]
            if esperado is not None:
                bloque.append(f"    esperadas: {esperado}")
            for k in sorted(d.get("lineas", {}), key=float)[:4]:
                v = d["lineas"][k]
                bloque.append(f"    over {k}: {_pct(v['over'])}   "
                              f"under {k}: {_pct(v['under'])}")
            if d.get("ambos_reciben") is not None:
                bloque.append(f"    ambos reciben tarjeta: {_pct(d['ambos_reciben'])}")
            if d.get("alguna_carrera") is not None:
                bloque.append(f"    al menos 1 carrera: {_pct(d['alguna_carrera'])}")
            lineas.append("\n".join(bloque))
            disponibles.append({"market_id": mid, "etiqueta": reg["etiqueta"],
                                "model_version": d.get("model_version"),
                                "status": d.get("status"), "solo_chat": True})
            continue

        p = api.get_prediction(game_id, mid)
        if not p.get("disponible"):
            bloqueados.append((reg, p.get("motivo")))
            continue
        cal = p["MODEL_PROBABILITY_CALIBRATED"]
        fila = [f"▸ {reg['etiqueta'].upper()}",
                f"    {p['selection']}: {_pct(cal)}"]
        if p.get("MARKET_IMPLIED_PROBABILITY") is not None:
            fila.append(f"    mercado: {_pct(p['MARKET_IMPLIED_PROBABILITY'])}"
                        f"   diferencia: {p['gap_pp']:+.1f} pp")
        else:
            fila.append("    mercado: sin cuota registrada")
        lineas.append("\n".join(fila))
        disponibles.append({"market_id": mid, "etiqueta": reg["etiqueta"],
                            "model_version": p.get("model_version"),
                            "status": p.get("status"), "solo_chat": False})

    if not lineas:
        motivos = " · ".join(f"{r['etiqueta']}: {m}" for r, m in bloqueados[:3])
        return {"version": VERSION, "ok": False, "tipo": "resumen_vacio",
                "pregunta": texto, "intencion": det["intencion"],
                "partido": partido, "sport": sport, "game_id": str(game_id),
                "status": "INSUFFICIENT DATA",
                "respuesta": (f"No tengo ninguna predicción publicada de este partido "
                              f"todavía."),
                "motivo": motivos or None}

    cuerpo = "\n\n".join(lineas)
    cola = ""
    if bloqueados:
        cola = ("\n\nSin dato en este partido: "
                + ", ".join(r["etiqueta"] for r, _ in bloqueados))
    return {
        "version": VERSION, "ok": True, "tipo": "resumen", "pregunta": texto,
        "intencion": det["intencion"], "partido": partido, "sport": sport,
        "game_id": str(game_id), "status": None,
        "respuesta": cuerpo + cola,
        "mercados": disponibles,
        "no_disponibles": [{"etiqueta": r["etiqueta"], "motivo": m}
                           for r, m in bloqueados],
        "aviso": ("Todas son proyecciones del modelo. Sin histórico de cuotas no se "
                  "puede demostrar ventaja sobre la casa en ningún mercado."),
    }


def responder(texto: str, game_id: str | None = None, sport: str | None = None,
              contexto: dict | None = None, historial: list | None = None) -> dict:
    """Punto de entrada unico del chat.

    `contexto` es lo que se contesto la vez anterior. Se comprueba ANTES que
    nada porque una repregunta ("entonces solo tiene el 29%?") no menciona
    ningun mercado y el detector de intencion la daria por incomprensible,
    teniendo el dato a un paso.
    """
    # 1) Preguntas sobre el SISTEMA, no sobre un partido. Van PRIMERO porque
    # tener un partido seleccionado no debe secuestrarlas: "¿que partidos hay de
    # beisbol hoy?" devolvia el resumen del tenis que estaba elegido.
    sis = knowledge.responder(texto)
    if sis is not None:
        return sis

    # 2) Repreguntas sobre la respuesta anterior.
    seg = followup.responder(texto, contexto or {})
    if seg is not None:
        return seg

    det = intent.detectar(texto, sport)

    if det["intencion"] == "saludo":
        return _saludo(texto, det, game_id, sport)

    # Pregunta abierta sobre un partido concreto: se contesta con TODO lo que
    # haya, en vez de exigir que se nombre un mercado.
    if det["intencion"] == "resumen" or (det.get("mercado_desconocido") and game_id):
        p = _partido_de(game_id, sport)
        if p is None and det["intencion"] == "resumen":
            r = games.resolver(texto, det["sport"] or sport)
            if r.get("encontrado"):
                p = r["partido"]
                game_id = p["game_id"]
        if p is not None:
            return _resumen(texto, det, p, game_id, p["sport"])

    # Antes de rendirse: si hay un LLM configurado, que lo intente. Recibe los
    # datos YA CALCULADOS y solo los redacta; no computa ninguna probabilidad.
    if det.get("mercado_desconocido") and det["intencion"] != "catalogo":
        r = llm.responder(texto, game_id, sport, contexto, historial)
        if r is not None:
            # Si el LLM FALLA hay que decirlo. Antes se caia al mensaje generico
            # de "no reconozco la pregunta", que es mentira: la pregunta se
            # entendio, lo que fallo fue el modelo. El usuario se quedaba
            # creyendo que el chat no sabia, cuando el problema era otro.
            return r

    if det["intencion"] == "catalogo" or det.get("mercado_desconocido"):
        cat = api.catalogo(det["sport"] or sport)
        es_catalogo = det["intencion"] == "catalogo"
        # Dos situaciones distintas que no se pueden contestar igual:
        #   no se ENTIENDE la pregunta   -> no tengo modelo para eso
        #   se entiende pero falta el MERCADO -> dime de que mercado
        # Decir "no tengo modelo" cuando si lo hay y solo falta precisar es
        # rechazar una pregunta que en realidad si se puede contestar.
        falta_mercado = det["intencion"] != "desconocida" and not det["market_id"]
        cabeza = ("Esto es lo que puedo contestar hoy. Los secundarios "
                  "solo existen aqui: no salen en Top Picks.") if es_catalogo else (
            "Entiendo lo que quieres saber, pero no de que mercado. Dime uno de estos:"
            if falta_mercado else
            "No tengo un modelo validado para eso, y prefiero decirtelo a darte "
            "un numero inventado. Esto si te lo puedo contestar:")
        estado_llm = llm.disponible()
        return {
            "version": VERSION, "ok": es_catalogo,
            "tipo": ("catalogo" if es_catalogo else
                     "falta_mercado" if falta_mercado else "sin_modelo"),
            "pregunta": texto, "intencion": det["intencion"],
            "motivo": det.get("motivo"),
            "status": None if es_catalogo else ("NO MARKET" if falta_mercado else "BLOCKED"),
            "respuesta": cabeza,
            # Si no hay LLM, se dice: es la diferencia entre "no puedo" y
            # "no sé", y el usuario merece saber cual de las dos es.
            "llm": estado_llm,
            "sugerencia_llm": (None if estado_llm["ok"] else
                               "Para que pueda contestar preguntas libres como una "
                               "IA, conéctame un motor de lenguaje. Hay opciones "
                               "gratis que corren en tu propia PC (Ollama, "
                               "LM Studio). Los números seguirían saliendo de los "
                               "modelos; el LLM solo los redactaría."),
            "primarios": [{"market_id": d["market_id"], "sport": d["sport"],
                           "etiqueta": d["etiqueta"], "pregunta": d["pregunta_tipo"],
                           "estado": d["estado"]} for d in cat["primarios"]],
            "secundarios": [{"market_id": d["market_id"], "sport": d["sport"],
                             "etiqueta": d["etiqueta"], "pregunta": d["pregunta_tipo"],
                             "estado": d["estado"],
                             "limitaciones": d["limitaciones"]} for d in cat["secundarios"]],
        }

    mid = det["market_id"]
    reg = registry.get(mid) if mid else None
    if reg is None:
        return _sin_modelo(texto, det)

    partido = None
    if game_id is None:
        # Nadie deberia saberse un event_id de 32 caracteres. Si el usuario
        # nombro un equipo, se busca el partido; si no, se le ofrece la lista
        # en vez de pedirle un identificador que no puede conocer.
        r = games.resolver(texto, reg["sport"])
        if r.get("encontrado"):
            partido = r["partido"]
            game_id = partido["game_id"]
        else:
            return {"version": VERSION, "ok": False, "tipo": "falta_partido",
                    "pregunta": texto, "intencion": det["intencion"], "market_id": mid,
                    "sport": reg["sport"], "etiqueta": reg["etiqueta"],
                    "respuesta": (f"Puedo contestarte {reg['etiqueta'].lower()}, pero "
                                  f"necesito saber de que partido. Elige uno:"),
                    "motivo": r.get("motivo"), "ambiguo": r.get("ambiguo", False),
                    "partidos": r.get("candidatos") or games.listar(10, reg["sport"])[:12],
                    "market": reg}
    if partido is None:
        partido = next((j for j in games.listar(10, reg["sport"])
                        if j["game_id"] == str(game_id)), None)

    def _con_partido(r):
        """Adjunta el partido a la respuesta y a su contexto, en un solo sitio."""
        r["partido"] = partido
        if isinstance(r.get("contexto"), dict):
            r["contexto"]["partido"] = partido
            r["contexto"]["game_id"] = str(game_id)
        return r

    # El mercado casa por PALABRAS, pero puede no casar por ALCANCE: los goles
    # del partido completo no son los de la primera mitad, y el total no son
    # los goles de un equipo. Contestar con el numero del alcance equivocado es
    # responder otra pregunta sin avisar, asi que se comprueba antes.
    av = alcance.revisar(texto, reg, partido)
    if av is not None:
        if av["tipo"] == "redirigir":
            reg = registry.get(av["market_id"]) or reg
        else:
            return _con_partido({
                **_cabecera(reg, det, game_id), "ok": False,
                "tipo": "fuera_de_alcance", "respuesta": av["motivo"],
                "status": "NO MARKET", "alcance": av,
                "limitaciones": reg.get("limitaciones")})

    if det["intencion"] == "explicacion":
        return _con_partido(_explicar(texto, det, reg, game_id))
    if det["intencion"] == "comparacion":
        return _con_partido(_comparar(texto, det, reg, game_id))
    # El enrutado NO depende de `distribution_type`: ese campo describe como
    # calcula el modelo por dentro (los goles usan una Poisson bivariada), no
    # como se consulta. Lo que decide es si hay un motor de distribucion
    # conectado para ese mercado. Sin esta distincion, "¿cuantos goles?" se iba
    # al motor de conteo, no encontraba ninguno y contestaba "no disponible"
    # cuando la prediccion estaba publicada y a mano.
    if reg["market_id"] in api.DISTRIBUCIONES:
        return _con_partido(_conteo(texto, det, reg, game_id))
    return _con_partido(_binario(texto, det, reg, game_id))


# ---------------------------------------------------------------------------

def _cabecera(reg, det, game_id):
    return {"version": VERSION, "pregunta": det["texto"], "intencion": det["intencion"],
            "market_id": reg["market_id"], "sport": reg["sport"],
            "etiqueta": reg["etiqueta"], "game_id": str(game_id)}


def _binario(texto, det, reg, game_id) -> dict:
    p = api.get_prediction(game_id, reg["market_id"])
    if not p.get("disponible"):
        return {**_cabecera(reg, det, game_id), "ok": False, "tipo": "no_disponible",
                "respuesta": p["motivo"], "status": "INSUFFICIENT DATA"}
    comp = api.get_market_comparison(game_id, reg["market_id"])
    lineas = [f"{reg['etiqueta'].upper()}", ""]
    lineas.append(f"Seleccion: {p['selection']}")
    lineas.append(f"Probabilidad del modelo (calibrada): {_pct(p['MODEL_PROBABILITY_CALIBRATED'])}")
    if p.get("MODEL_PROBABILITY_RAW") is not None and \
            p["MODEL_PROBABILITY_RAW"] != p["MODEL_PROBABILITY_CALIBRATED"]:
        lineas.append(f"Probabilidad cruda: {_pct(p['MODEL_PROBABILITY_RAW'])}")
    if p.get("MARKET_IMPLIED_PROBABILITY") is not None:
        lineas.append(f"Implicita del mercado: {_pct(p['MARKET_IMPLIED_PROBABILITY'])}")
        lineas.append(f"Diferencia: {p['gap_pp']:+.1f} pp")
    else:
        lineas.append("Mercado: sin cuota registrada para este partido.")
    return {**_cabecera(reg, det, game_id), "ok": True, "tipo": "probabilidad",
            "respuesta": "\n".join(lineas),
            "prediccion": p, "comparacion": comp,
            "model_version": p["model_version"],
            "calibracion": p.get("calibration_version") or "sin_calibrar",
            "status": p.get("status"),
            "evidencia": reg.get("evidencia"),
            "limitaciones": reg.get("limitaciones"),
            "aviso": (comp.get("explicacion") if comp.get("disponible") else None),
            # Memoria para la siguiente pregunta del usuario.
            "contexto": _contexto(reg, p, None, det)}


# Cuando un mercado no puede publicar, cual es lo mas parecido que SI existe.
HERMANOS = {
    "mlb.f5_total": ("mlb.total", "del partido completo sí tengo total de carreras"),
    "mlb.runs_dist": ("mlb.total", "sí tengo el over/under de carreras del partido"),
    "soccer.cards": ("soccer.total_goals", "de este partido sí tengo goles"),
    "soccer.corners": ("soccer.total_goals", "de este partido sí tengo goles"),
}


def _alternativa(reg, game_id):
    """Lo más cercano que sí está publicado, si existe."""
    par = HERMANOS.get(reg["market_id"])
    if not par:
        return None
    mid, frase = par
    otro = registry.get(mid)
    p = api.get_prediction(game_id, mid)
    if not (otro and p.get("disponible")):
        return None
    return {
        "market_id": mid,
        "texto": (f"Pero {frase}: {p['selection']} al "
                  f"{_pct(p['MODEL_PROBABILITY_CALIBRATED'])}"
                  + (f" (mercado {_pct(p['MARKET_IMPLIED_PROBABILITY'])})."
                     if p.get("MARKET_IMPLIED_PROBABILITY") is not None else ".")),
        "contexto": _contexto(otro, p, None, None),
    }


def _conteo(texto, det, reg, game_id) -> dict:
    d = api.get_distribution(game_id, reg["market_id"])
    if not d.get("disponible"):
        # Negarse esta bien; dejar al usuario sin nada teniendo algo parecido
        # publicado, no. Si hay un mercado hermano disponible, se ofrece.
        alt = _alternativa(reg, game_id)
        return {**_cabecera(reg, det, game_id), "ok": False, "tipo": "no_disponible",
                "respuesta": d["motivo"] + (f"\n\n{alt['texto']}" if alt else ""),
                "status": "INSUFFICIENT DATA",
                "alternativa": alt["market_id"] if alt else None,
                "limitaciones": reg.get("limitaciones"),
                "contexto": alt.get("contexto") if alt else None}

    unidad = {"soccer.cards": "tarjetas", "mlb.runs_dist": "carreras",
              "mlb.f5_total": "carreras en las primeras 5",
              "soccer.corners": "corners"}.get(reg["market_id"], "unidades")
    esperado = d.get("expected_cards", d.get("expected_runs"))
    lineas = [reg["etiqueta"].upper(), ""]
    if esperado is not None:
        lineas.append(f"{unidad.capitalize()} esperadas: {esperado}")

    ln = det.get("linea")
    if ln is not None:
        clave = f"{float(ln):g}"
        v = (d.get("lineas") or {}).get(clave)
        if v is None:
            disp = ", ".join(sorted(d.get("lineas", {}), key=float))
            lineas.append(f"No tengo calculada la linea {clave}. Tengo: {disp}.")
        else:
            lineas.append(f"Over {clave}: {_pct(v['over'])}")
            lineas.append(f"Under {clave}: {_pct(v['under'])}")
            if v.get("push"):
                lineas.append(f"Push (exacto {clave}): {_pct(v['push'])}")
    else:
        for k in sorted(d.get("lineas", {}), key=float):
            v = d["lineas"][k]
            lineas.append(f"  Over {k}: {_pct(v['over'])}   Under {k}: {_pct(v['under'])}")

    # Preguntas de equipo que solo la distribucion puede contestar.
    if d.get("ambos_reciben") is not None:
        lineas.append(f"Ambos reciben tarjeta: {_pct(d['ambos_reciben'])}")
        lineas.append(f"  {d.get('partido','local')}: local {_pct(d['home_al_menos_1'])} "
                      f"· visitante {_pct(d['away_al_menos_1'])}")
    if d.get("alguna_carrera") is not None:
        lineas.append(f"Al menos 1 carrera: {_pct(d['alguna_carrera'])}")
        lineas.append(f"0 carreras: {_pct(1 - d['alguna_carrera'])}")
        lineas.append(f"Ambos anotan: {_pct(d['ambos_anotan'])}")
    k = det.get("cantidad")
    if k is not None and d.get("team_total"):
        lineas.append(f"Anotar {k}+ carreras: local "
                      f"{_pct(d['team_total']['home'].get(f'{k}+'))} · visitante "
                      f"{_pct(d['team_total']['away'].get(f'{k}+'))}")

    return {**_cabecera(reg, det, game_id), "ok": True, "tipo": "distribucion",
            "respuesta": "\n".join(lineas), "distribucion": d,
            "model_version": d.get("model_version"),
            "calibracion": "sin_calibrar",
            "status": d.get("status", "projection").upper(),
            "confianza": _confianza(d),
            "limitaciones": reg.get("limitaciones"),
            "aviso": d.get("advertencia") or d.get("nota_independencia"),
            "evidencia": d.get("evidencia") or reg.get("evidencia")}


def _confianza(d) -> str:
    n = d.get("partidos_de_respaldo")
    if n is None:
        return "moderada"
    if n >= 60:
        return "moderada"
    if n >= 20:
        return "baja"
    return "muy baja"


def _comparar(texto, det, reg, game_id) -> dict:
    c = api.get_market_comparison(game_id, reg["market_id"])
    if not c.get("disponible"):
        return {**_cabecera(reg, det, game_id), "ok": False, "tipo": "no_disponible",
                "respuesta": c.get("motivo", "no puedo comparar con el mercado."),
                "status": "NO MARKET"}
    lineas = [f"MODELO VS MERCADO — {reg['etiqueta']}", ""]
    lineas.append(f"Modelo calibrado: {_pct(c['MODEL_PROBABILITY_CALIBRATED'])}")
    if c.get("MARKET_IMPLIED_PROBABILITY") is None:
        lineas.append("Mercado: sin cuota registrada.")
    else:
        lineas.append(f"Mercado implicito: {_pct(c['MARKET_IMPLIED_PROBABILITY'])}")
        lineas.append(f"Diferencia: {c['gap_pp']:+.1f} pp ({c['direccion']})")
        lineas.append(f"Tramo: {c['bucket']}")
    lineas.append("")
    lineas.append(c["explicacion"] or "")
    return {**_cabecera(reg, det, game_id), "ok": True, "tipo": "comparacion",
            "respuesta": "\n".join(lineas), "comparacion": c,
            "status": c["FINAL_STATUS"],
            "advertencia": ("Una diferencia a favor del modelo NO es valor mientras "
                            "no exista historico que lo demuestre.")}


def _explicar(texto, det, reg, game_id) -> dict:
    e = api.get_explanation(game_id, reg["market_id"])
    p = api.get_prediction(game_id, reg["market_id"])
    if not e.get("disponible"):
        return {**_cabecera(reg, det, game_id), "ok": False, "tipo": "sin_explicacion",
                "respuesta": e["motivo"],
                "prediccion": p if p.get("disponible") else None,
                "status": "NO EXPLANATION"}
    return {**_cabecera(reg, det, game_id), "ok": True, "tipo": "explicacion",
            "respuesta": "Estos son los factores que uso el modelo, tal como se "
                         "guardaron al emitir la prediccion.",
            "factores": e["factores"], "fuente": e["fuente"],
            "prediccion": p if p.get("disponible") else None,
            "model_version": (p or {}).get("model_version")}
