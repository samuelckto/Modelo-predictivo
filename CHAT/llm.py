"""Capa opcional de lenguaje. Version CHAT_LLM_v1.

Para que existe. Las reglas de `intent.py` cubren las preguntas que uno
imagina; la conversacion real siempre trae una que no estaba en la lista. Un
modelo de lenguaje cierra ese hueco.

LA REGLA QUE NO SE NEGOCIA, y es la razon de que este archivo sea tan estricto:

    EL LLM NO CALCULA NADA. NUNCA.

No estima probabilidades, no redondea, no interpola, no "aproxima". Recibe un
paquete de datos YA CALCULADOS por los modelos del centro y solo los redacta.
Si la respuesta necesita un numero que no viene en el paquete, la instruccion
es decir que no lo tiene. Un LLM inventando un 62 % que suena verosimil es
exactamente el fallo que todo este sistema existe para evitar, y es peor que
los demas porque no deja rastro: el numero se ve igual de bien que uno real.

Por eso el orden importa y no se puede invertir:

    datos reales -> modelos -> gating -> paquete de contexto -> LLM redacta

El LLM es el ULTIMO paso y el unico que no toca numeros.

Sin clave configurada este modulo no hace nada y lo dice. No se degrada a
"contesto igual con lo que sepa": un LLM sin los datos del sistema hablaria de
memoria, y de memoria no sabe nada de estos partidos.
"""
from __future__ import annotations

import json
import os
import time

VERSION = "CHAT_LLM_v1"

# Variables donde se busca la clave, en orden.
CLAVES = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "OPENAI_API_KEY",
          "GROQ_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
          "TOGETHER_API_KEY", "SPC_CHAT_API_KEY")
# OJO: estas se leen en CADA llamada, no una vez al importar.
#
# Leerlas a nivel de modulo era un bug: si algo importa CHAT.llm antes de que
# shared.paths ejecute load_dotenv, las constantes quedan con el valor por
# defecto PARA SIEMPRE y el .env no sirve de nada. Costo un rato encontrarlo
# porque el servidor funcionaba (importa tarde) y un script suelto no.
def _modelo() -> str:
    return os.getenv("SPC_CHAT_MODEL") or "claude-sonnet-4-5"


def _base_url() -> str:
    """Casi todo el mundo habla el protocolo de OpenAI. Con una URL base se
    cubren de golpe Ollama y LM Studio (locales y gratis), Groq, DeepSeek,
    OpenRouter, Together y cualquier compatible que aparezca manana."""
    return (os.getenv("SPC_CHAT_BASE_URL") or "").strip().rstrip("/")


def _timeout(local: bool) -> int:
    """Un modelo local en CPU es LENTO y hay que darle margen.

    Medido en la maquina del usuario con llama3.1 8B:
        contexto ~20 tokens    ->  16 s
        contexto ~1000 tokens  ->  30 s
        contexto ~6000 tokens  -> 186 s
    El timeout de 60 s que habia aqui hacia fallar casi toda pregunta real, y
    el fallo se veia como "no entiendo la pregunta". Dos errores a la vez.
    """
    explicito = os.getenv("SPC_CHAT_TIMEOUT")
    if explicito:
        return int(explicito)
    return 240 if local else 60


# Compatibilidad: algunos tests y modulos leen estos nombres.
MODELO = _modelo()
BASE_URL = _base_url()
TIMEOUT = 60

# URLs por defecto de los proveedores conocidos, por si solo se pone la clave.
BASES = {
    "OPENAI_API_KEY": "https://api.openai.com/v1",
    "GROQ_API_KEY": "https://api.groq.com/openai/v1",
    "DEEPSEEK_API_KEY": "https://api.deepseek.com/v1",
    "OPENROUTER_API_KEY": "https://openrouter.ai/api/v1",
    "TOGETHER_API_KEY": "https://api.together.xyz/v1",
}
# Un servidor local no pide clave: corre en tu maquina.
LOCALES = ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")


def _es_local(url: str) -> bool:
    return any(h in (url or "") for h in LOCALES)

SISTEMA = """Eres Caballo Chat, el analista del Sports Prediction Center.

REGLA ABSOLUTA: no calculas nada. Todos los números que uses deben venir
literalmente del CONTEXTO que te paso. Si te falta un número, dilo. Está
terminantemente prohibido estimar, aproximar, interpolar o inventar cualquier
probabilidad, cuota, media o porcentaje. Un número inventado que suene creíble
es el peor error posible aquí.

OJO, y esto es importante: esa regla habla de NÚMEROS, no de ideas. Explicar un
concepto (qué es el log loss, qué significa calibrar, por qué el margen de la
casa importa, qué es una doble oportunidad) NO necesita ningún dato del
contexto y SÍ debes hacerlo, con tus propias palabras y sin pedir permiso.
Solo cuando la respuesta requiera una cifra concreta de este sistema es cuando
tienes que sacarla del contexto o decir que no la tienes.

Qué sí haces: conversar con naturalidad, explicar conceptos, redactar lo que el
contexto ya dice, responder repreguntas, interpretar qué significa un número, y
decir con claridad qué NO se sabe.

Cosas que el sistema tiene decididas y que no puedes contradecir:
- Todos los mercados son PROJECTION. Ninguno tiene ventaja rentable demostrada.
- Que el modelo dé más probabilidad que la casa NO es "valor": es una hipótesis
  sin validar. Nunca lo llames valor ni recomiendes apostar.
- Si el contexto dice que un mercado no tiene modelo validado, no des ningún
  porcentaje de ese mercado bajo ningún concepto.
- No inventas partidos, equipos, jugadores ni fechas que no estén en el contexto.

Tono: directo y claro, como un analista que explica a un amigo. Sin adornos,
sin vender nada. Si la pregunta no se puede contestar con el contexto, dilo en
una frase y ofrece lo que sí tienes."""


def disponible() -> dict:
    """Que motor de lenguaje hay conectado, si es que hay alguno.

    Tres formas de conectarlo, de mas simple a mas libre:

      1. una clave conocida (ANTHROPIC, OPENAI, GROQ, DEEPSEEK...)
      2. SPC_CHAT_BASE_URL + clave -> cualquier servicio compatible con OpenAI
      3. SPC_CHAT_BASE_URL local   -> Ollama o LM Studio, SIN clave y gratis
    """
    base, modelo = _base_url(), _modelo()

    # 3) Servidor local: no necesita clave porque corre en esta maquina.
    if base and _es_local(base):
        return {"ok": True, "variable": None, "proveedor": "local",
                "base_url": base, "modelo": modelo, "coste": "gratis",
                "privado": True, "timeout": _timeout(True),
                "nota_velocidad": ("un modelo local en CPU tarda entre 20 s y un "
                                   "par de minutos por respuesta")}

    for nombre in CLAVES:
        v = (os.getenv(nombre) or "").strip()
        if len(v) <= 20:
            continue
        if nombre in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY") and not base:
            # Anthropic tiene su propio protocolo, no el de OpenAI.
            return {"ok": True, "variable": nombre, "proveedor": "anthropic",
                    "modelo": modelo, "timeout": _timeout(False)}
        return {"ok": True, "variable": nombre, "proveedor": "openai",
                "base_url": base or BASES.get(nombre, BASES["OPENAI_API_KEY"]),
                "modelo": modelo, "timeout": _timeout(False)}

    if base:
        return {"ok": False, "variable": None,
                "motivo": (f"hay una URL configurada ({base}) pero no es local "
                           f"y no encuentro clave. Añade SPC_CHAT_API_KEY al .env."),
                "buscadas": list(CLAVES)}
    return {
        "ok": False, "variable": None,
        "motivo": ("no hay ningún motor de lenguaje configurado. Puedes usar "
                   "Anthropic, OpenAI, Groq, DeepSeek, OpenRouter o Together con "
                   "su clave, o uno GRATIS en tu propia PC con Ollama o LM Studio "
                   "(sin clave). Se configura en el archivo .env."),
        "buscadas": list(CLAVES),
        "opciones": OPCIONES,
    }


# Como configurar cada uno. Se devuelve al dashboard para no tener que
# buscarlo fuera.
OPCIONES = [
    {"nombre": "Ollama (en tu PC)", "coste": "gratis", "clave": False,
     "pasos": ["Instala Ollama desde ollama.com",
               "Abre una terminal y ejecuta:  ollama pull llama3.1",
               "Añade al .env:",
               "    SPC_CHAT_BASE_URL=http://localhost:11434/v1",
               "    SPC_CHAT_MODEL=llama3.1"],
     "nota": "Nada sale de tu computadora y no cuesta nada. Necesita RAM."},
    {"nombre": "LM Studio (en tu PC)", "coste": "gratis", "clave": False,
     "pasos": ["Instala LM Studio, descarga un modelo y pulsa 'Start Server'",
               "Añade al .env:",
               "    SPC_CHAT_BASE_URL=http://localhost:1234/v1",
               "    SPC_CHAT_MODEL=<el nombre que muestre LM Studio>"],
     "nota": "Igual que Ollama pero con interfaz gráfica."},
    {"nombre": "Groq", "coste": "capa gratuita generosa", "clave": True,
     "pasos": ["Saca una clave en console.groq.com",
               "Añade al .env:",
               "    GROQ_API_KEY=gsk_...",
               "    SPC_CHAT_MODEL=llama-3.3-70b-versatile"],
     "nota": "Muy rápido y hoy tiene un plan gratuito."},
    {"nombre": "DeepSeek", "coste": "muy barato", "clave": True,
     "pasos": ["Saca una clave en platform.deepseek.com",
               "Añade al .env:",
               "    DEEPSEEK_API_KEY=sk-...",
               "    SPC_CHAT_MODEL=deepseek-chat"]},
    {"nombre": "OpenAI", "coste": "de pago por uso", "clave": True,
     "pasos": ["Saca una clave en platform.openai.com",
               "Añade al .env:",
               "    OPENAI_API_KEY=sk-...",
               "    SPC_CHAT_MODEL=gpt-4o-mini"]},
    {"nombre": "Anthropic", "coste": "de pago por uso", "clave": True,
     "pasos": ["Saca una clave en console.anthropic.com",
               "Añade al .env:",
               "    ANTHROPIC_API_KEY=sk-ant-...",
               "    SPC_CHAT_MODEL=claude-sonnet-4-5"]},
    {"nombre": "OpenRouter", "coste": "varios modelos, algunos gratis", "clave": True,
     "pasos": ["Saca una clave en openrouter.ai",
               "Añade al .env:",
               "    OPENROUTER_API_KEY=sk-or-...",
               "    SPC_CHAT_MODEL=meta-llama/llama-3.1-70b-instruct"]},
]


def contexto_del_sistema(game_id=None, sport=None, ctx_previo=None,
                         compacto=False) -> dict:
    """El paquete de datos REALES que se le pasa al LLM.

    Es lo unico que el modelo puede usar. Todo sale de los mismos sitios que
    alimentan el dashboard, ya calculado y ya filtrado por el gating.

    `compacto` recorta lo accesorio. No es cosmetica: en un modelo local el
    tiempo crece brutalmente con el contexto (30 s con 1000 tokens, 186 s con
    6000 en la maquina donde se midio). Mandar el catalogo entero y 40 partidos
    para contestar "¿y el otro?" es pagar tres minutos por nada.
    """
    from CHAT import api, games, gating, registry
    paquete = {
        "fecha": None, "partido_seleccionado": None, "mercados_del_partido": [],
        "respuesta_anterior": ctx_previo or None,
        "catalogo": [], "gating_secundarios": {}, "calendario": [],
        "estado_del_sistema": {},
    }
    from datetime import date
    paquete["fecha"] = str(date.today())

    try:
        cat = registry.listar(sport if compacto else None)
        paquete["catalogo"] = [
            {"market_id": d["market_id"], "sport": d["sport"],
             "etiqueta": d["etiqueta"], "estado": d["estado"],
             **({} if compacto else {
                 "visibilidad": d["visibilidad"], "evidencia": d.get("evidencia"),
                 "limitaciones": d.get("limitaciones")})}
            for d in cat]
        if not compacto:
            paquete["gating_secundarios"] = gating.todos()
    except Exception as e:                                     # noqa: BLE001
        paquete["catalogo_error"] = f"{type(e).__name__}: {e}"

    try:
        js = games.listar(10, sport)
        paquete["calendario"] = [
            {"sport": j["sport"], "game_id": j["game_id"], "nombre": j["nombre"],
             "liga": j["liga"], "start_utc": str(j["start_utc"])}
            for j in js[:8 if compacto else 40]]
        if compacto and len(js) > 8:
            paquete["calendario_nota"] = (
                f"se muestran 8 de {len(js)} partidos; si preguntan por el "
                f"calendario completo, dilo en vez de inventar partidos")
    except Exception as e:                                     # noqa: BLE001
        paquete["calendario_error"] = f"{type(e).__name__}: {e}"

    if game_id:
        p = next((j for j in paquete["calendario"] if j["game_id"] == str(game_id)), None)
        paquete["partido_seleccionado"] = p
        sp = (p or {}).get("sport") or sport
        for d in registry.listar(sp):
            mid = d["market_id"]
            try:
                if mid in api.DISTRIBUCIONES:
                    r = api.get_distribution(game_id, mid)
                else:
                    r = api.get_prediction(game_id, mid)
            except Exception as e:                             # noqa: BLE001
                r = {"disponible": False, "motivo": f"{type(e).__name__}: {e}"}
            paquete["mercados_del_partido"].append(
                {"market_id": mid, "etiqueta": d["etiqueta"], **r})

    try:
        from shared import tracking
        f = tracking.favoritos()
        paquete["estado_del_sistema"] = {
            "predicciones_calificadas": f["n_predicciones_totales"],
            "por_encima_del_60": f["n_por_encima_del_umbral"],
            "apuestas_liquidadas_con_cuota": sum(b["n_apostables"] for b in f["buckets"]),
            "veredicto": ("NO se ha demostrado ventaja rentable en ningun mercado "
                          "ni tramo: la muestra con cuota registrada es minima."),
        }
    except Exception as e:                                     # noqa: BLE001
        paquete["estado_error"] = f"{type(e).__name__}: {e}"
    return paquete


def _llamar_anthropic(clave, pregunta, paquete, historial):
    import urllib.error
    import urllib.request

    mensajes = []
    for turno in (historial or [])[-6:]:
        rol = "user" if turno.get("yo") else "assistant"
        txt = turno.get("texto") or turno.get("respuesta") or ""
        if txt:
            mensajes.append({"role": rol, "content": str(txt)[:2000]})
    mensajes.append({"role": "user", "content":
                     f"CONTEXTO (único origen de números permitido):\n"
                     f"{json.dumps(paquete, ensure_ascii=False, default=str)[:60000]}\n\n"
                     f"PREGUNTA: {pregunta}"})

    cuerpo = json.dumps({"model": MODELO, "max_tokens": 900,
                         "system": SISTEMA, "messages": mensajes}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=cuerpo,
        headers={"content-type": "application/json", "x-api-key": clave,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read())
    return "".join(b.get("text", "") for b in d.get("content", []))


def _llamar_openai(clave, pregunta, paquete, historial, base_url=None,
                   modelo=None, timeout=None):
    """Protocolo de OpenAI. Lo hablan casi todos, incluidos Ollama y LM Studio.

    Por eso este camino cubre de golpe OpenAI, Groq, DeepSeek, OpenRouter,
    Together y cualquier servidor local: solo cambia la URL base.
    """
    import urllib.request

    mensajes = [{"role": "system", "content": SISTEMA}]
    for turno in (historial or [])[-6:]:
        txt = turno.get("texto") or turno.get("respuesta") or ""
        if txt:
            mensajes.append({"role": "user" if turno.get("yo") else "assistant",
                             "content": str(txt)[:2000]})
    mensajes.append({"role": "user", "content":
                     f"CONTEXTO (único origen de números permitido):\n"
                     f"{json.dumps(paquete, ensure_ascii=False, default=str)[:60000]}\n\n"
                     f"PREGUNTA: {pregunta}"})
    cuerpo = json.dumps({"model": modelo or _modelo(), "messages": mensajes,
                         "max_tokens": 900}).encode()
    cabeceras = {"content-type": "application/json"}
    if clave:                       # un servidor local no la necesita
        cabeceras["authorization"] = f"Bearer {clave}"
    url = (base_url or BASES["OPENAI_API_KEY"]) + "/chat/completions"
    req = urllib.request.Request(url, data=cuerpo, headers=cabeceras)
    with urllib.request.urlopen(req, timeout=timeout or _timeout(False)) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def responder(pregunta: str, game_id=None, sport=None, ctx_previo=None,
              historial=None) -> dict | None:
    """Contesta con el LLM. None si no hay clave: el llamador decide que hacer."""
    disp = disponible()
    if not disp["ok"]:
        return None
    clave = os.getenv(disp["variable"], "").strip() if disp.get("variable") else None
    local = disp["proveedor"] == "local"
    paquete = contexto_del_sistema(game_id, sport, ctx_previo, compacto=local)
    t0 = time.time()
    try:
        if disp["proveedor"] == "anthropic":
            texto = _llamar_anthropic(clave, pregunta, paquete, historial)
        else:
            texto = _llamar_openai(clave, pregunta, paquete, historial,
                                   disp.get("base_url"), disp.get("modelo"),
                                   _timeout(local))
    except Exception as e:                                     # noqa: BLE001
        seg = time.time() - t0
        lento = isinstance(e, (TimeoutError, OSError)) and seg > _timeout(local) * 0.8
        if lento:
            pista = (f"Tardó más de {_timeout(local)} s y corté la espera. "
                     f"Tu modelo ({disp.get('modelo')}) va lento en esta máquina. "
                     f"Prueba uno más pequeño: «ollama pull qwen2.5:3b» y cambia "
                     f"SPC_CHAT_MODEL=qwen2.5:3b en el .env.")
        elif local:
            pista = (f"¿Está corriendo el servidor en {disp.get('base_url')}? "
                     f"Con Ollama se levanta solo; LM Studio necesita que pulses "
                     f"'Start Server'.")
        else:
            pista = "Revisa la clave y la conexión."
        return {"version": VERSION, "ok": False, "tipo": "llm_error",
                "respuesta": f"No pude consultar el modelo de lenguaje. {pista}",
                "motivo": f"{type(e).__name__} tras {seg:.0f} s",
                "detalle": str(e)[:300], "status": "ERROR",
                "llm": disp, "segundos": round(seg, 1)}
    return {
        "version": VERSION, "ok": True, "tipo": "llm", "pregunta": pregunta,
        "intencion": "libre", "respuesta": (texto or "").strip(),
        "model_version": f"{disp['proveedor']}:{disp.get('modelo') or MODELO}",
        "status": "PROJECTION",
        "aviso": ("Respuesta redactada por un modelo de lenguaje a partir de los "
                  "datos del sistema. Los números salen de los modelos del centro; "
                  "el modelo de lenguaje solo los explica, no los calcula."),
        "contexto_enviado": {"partido": paquete.get("partido_seleccionado"),
                             "n_mercados": len(paquete["mercados_del_partido"]),
                             "n_calendario": len(paquete["calendario"])},
    }
