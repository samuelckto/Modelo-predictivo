"""Lo que Caballo Chat sabe del SISTEMA, no de un partido. CHAT_KNOWLEDGE_v1.

Hasta ahora el chat solo sabia contestar sobre el partido seleccionado. Por eso
"¿que partidos hay de beisbol hoy?" devolvia el resumen del tenis que estaba
elegido: la pregunta no era del partido, era del CALENDARIO, y no habia a donde
enviarla.

Aqui viven las preguntas sobre el sistema mismo:

    calendario     ¿que partidos hay hoy / manana / de beisbol?
    cobertura      ¿que ligas tienes? ¿de que hay datos?
    rendimiento    ¿como va el sistema? ¿cuanto acierta?
    modelos        ¿que tan bueno es el modelo de tarjetas?
    picks          ¿que recomiendas hoy?

Todo sale de los mismos sitios que alimentan el dashboard. Ningun numero se
escribe a mano aqui: si el dato cambia en la base, cambia en la respuesta.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta

VERSION = "CHAT_KNOWLEDGE_v1"

DEPORTES = {
    "MLB": ["beisbol", "baseball", "mlb", "carreras", "entradas"],
    "SOCCER": ["futbol", "soccer", "goles", "liga mx", "premier", "laliga",
               "champions", "serie a", "bundesliga", "ligue 1", "arabia"],
    "NFL": ["nfl", "futbol americano", "americano"],
    "NBA": ["nba", "basquet", "basket", "baloncesto"],
    "TENIS": ["tenis", "tennis", "atp", "wta"],
}

CALENDARIO = [r"\bque partidos\b", r"\bqu[eé] partidos\b", r"\bque juegos\b",
              r"\bqu[eé] juegos\b", r"\bhay hoy\b", r"\bjuegan hoy\b",
              r"\bcalendario\b", r"\bagenda\b", r"\bcartelera\b",
              r"\bque hay hoy\b", r"\bqu[eé] hay hoy\b", r"\bpartidos de\b",
              r"\bjuegos de\b", r"\bque se juega\b", r"\bqu[eé] se juega\b"]

COBERTURA = [r"\bque ligas\b", r"\bqu[eé] ligas\b", r"\bque deportes\b",
             r"\bqu[eé] deportes\b", r"\bque datos tienes\b",
             r"\bqu[eé] datos tienes\b", r"\bcobertura\b", r"\bde que hay datos\b",
             r"\bcuantos partidos tienes\b", r"\bcu[aá]ntos partidos tienes\b",
             r"\bque cubres\b", r"\bqu[eé] cubres\b"]

RENDIMIENTO = [r"\bcomo va el sistema\b", r"\bc[oó]mo va el sistema\b",
               r"\bcuanto acierta\b", r"\bcu[aá]nto acierta\b",
               r"\bque tan bueno es el sistema\b", r"\bqu[eé] tal va\b",
               r"\bresultados del sistema\b", r"\brendimiento\b",
               r"\bacierto\b", r"\bhistorial\b", r"\bva ganando\b",
               r"\broi\b", r"\bganancia\b", r"\bvas ganando\b",
               r"\bcomo vas\b", r"\bc[oó]mo vas\b"]

MODELOS = [r"\bque tan bueno es\b", r"\bqu[eé] tan bueno es\b",
           r"\bque tan confiable\b", r"\bqu[eé] tan confiable\b",
           r"\bque modelos\b", r"\bqu[eé] modelos\b", r"\bque tan fiable\b",
           r"\ben cual conf[ií]o\b", r"\ben cu[aá]l conf[ií]o\b",
           r"\bcual es el mejor\b", r"\bcu[aá]l es el mejor\b",
           r"\bmejor mercado\b"]

# "¿cual es el partido con mas probabilidad de ganarse hoy?" es EXACTAMENTE
# esta pregunta, y una regla la contesta al instante y con el numero exacto.
# Mandarla a un modelo de lenguaje seria mas lento y menos fiable.
PICKS = [r"\bque recomiendas\b", r"\bqu[eé] recomiendas\b", r"\btop picks\b",
         r"\bmejores picks\b", r"\bque apuesto\b", r"\bqu[eé] apuesto\b",
         r"\bmejores opciones\b", r"\bque me sugieres\b", r"\bqu[eé] me sugieres\b",
         r"\bmas probabilidad\b", r"\bm[aá]s probabilidad\b",
         r"\bmayor probabilidad\b", r"\bmas probable\b", r"\bm[aá]s probable\b",
         r"\bmas seguro\b", r"\bm[aá]s seguro\b", r"\bmejor probabilidad\b",
         r"\bmas alta\b", r"\bm[aá]s alta\b", r"\bmas % \b", r"\bmas porcentaje\b",
         r"\bm[aá]s porcentaje\b", r"\bel que mas\b", r"\bel que m[aá]s\b",
         r"\bla que mas\b", r"\bla que m[aá]s\b", r"\bcual es el mejor pick\b",
         r"\bmejor pick\b", r"\bmas facil\b", r"\bm[aá]s f[aá]cil\b",
         r"\bcual conviene mas\b", r"\bmejor opcion\b", r"\bmejor opci[oó]n\b"]

HOY = [r"\bhoy\b", r"\bahora\b", r"\beste dia\b"]
MANANA = [r"\bmanana\b", r"\bmañana\b"]


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _hay(t, pats) -> bool:
    return any(re.search(p, t) for p in pats)


def _deporte(t):
    for s, palabras in DEPORTES.items():
        if any(w in t for w in palabras):
            return s
    return None


def clasificar(texto: str) -> str | None:
    """¿Es una pregunta sobre el SISTEMA? Devuelve el tipo o None."""
    t = _norm(texto)
    if _hay(t, CALENDARIO):
        return "calendario"
    if _hay(t, PICKS):
        return "picks"
    if _hay(t, RENDIMIENTO):
        return "rendimiento"
    if _hay(t, COBERTURA):
        return "cobertura"
    if _hay(t, MODELOS):
        return "modelos"
    return None


# --------------------------------------------------------------------------

def _fmt_hora(s):
    if not s:
        return ""
    v = str(s).replace("T", " ")
    return v[11:16]


def calendario(texto: str) -> dict:
    from CHAT import games
    t = _norm(texto)
    sport = _deporte(t)
    hoy = date.today()
    manana = hoy + timedelta(days=1)
    solo_manana = _hay(t, MANANA)
    solo_hoy = _hay(t, HOY) and not solo_manana

    js = games.listar(10, sport)
    if solo_hoy:
        js = [j for j in js if str(j.get("game_date") or j.get("start_utc") or "")[:10] == str(hoy)]
        cuando = "hoy"
    elif solo_manana:
        js = [j for j in js if str(j.get("game_date") or j.get("start_utc") or "")[:10] == str(manana)]
        cuando = "mañana"
    else:
        cuando = "en los próximos días"

    etiqueta = {"MLB": "béisbol", "SOCCER": "fútbol", "NFL": "NFL",
                "NBA": "NBA", "TENIS": "tenis"}.get(sport, "")
    if not js:
        return {"ok": False, "tipo": "calendario", "status": "INSUFFICIENT DATA",
                "respuesta": (f"No tengo ningún partido{' de ' + etiqueta if etiqueta else ''} "
                              f"{cuando} en el calendario."),
                "partidos": []}

    lineas = [f"{len(js)} partido{'s' if len(js) != 1 else ''}"
              f"{' de ' + etiqueta if etiqueta else ''} {cuando}:", ""]
    for j in js[:20]:
        h = _fmt_hora(j.get("start_utc"))
        lineas.append(f"  {h or '  ·  '}  {j['nombre']}"
                      + (f"   ({len(j['mercados'])} mercados)" if j.get("mercados") else ""))
    if len(js) > 20:
        lineas.append(f"  … y {len(js) - 20} más")
    lineas.append("")
    lineas.append("Dime el nombre de cualquiera y te doy sus números.")
    return {"ok": True, "tipo": "calendario", "status": None,
            "respuesta": "\n".join(lineas), "partidos": js[:20], "sport": sport}


def cobertura(texto: str) -> dict:
    from CHAT import games, registry
    from shared import ledger
    cob = ledger.cobertura()["deportes"]
    js = games.listar(10)
    por_deporte = {}
    for j in js:
        por_deporte[j["sport"]] = por_deporte.get(j["sport"], 0) + 1

    lineas = ["Esto es lo que tengo en el sistema:", ""]
    for s, v in cob.items():
        mercados = [d["etiqueta"] for d in registry.listar(s)]
        prox = por_deporte.get(s, 0)
        lineas.append(
            f"▸ {s}: {prox} partidos próximos · {v['predicciones']} predicciones "
            f"({v['calificadas']} ya calificadas)")
        if mercados:
            lineas.append(f"    mercados: {', '.join(mercados)}")
    lineas.append("")
    lineas.append("En fútbol cubro Premier, LaLiga, Serie A, Bundesliga, Ligue 1, "
                  "Liga MX, Champions y Arabia. Tarjetas y córners solo en las "
                  "cinco europeas: son las únicas con esa fuente.")
    return {"ok": True, "tipo": "cobertura", "status": None,
            "respuesta": "\n".join(lineas), "detalle": cob}


def rendimiento(texto: str) -> dict:
    from shared import tracking
    f = tracking.favoritos()
    d = tracking.discrepancias()
    lineas = ["Cómo va el sistema, con los números reales:", ""]
    lineas.append(f"De {f['n_predicciones_totales']} predicciones ya calificadas, "
                  f"{f['n_por_encima_del_umbral']} pasaron del 60 %.")
    lineas.append("")
    for b in f["buckets"]:
        if not b["n"]:
            continue
        wr = f"{b['win_rate']:.0%}" if b["win_rate"] is not None else "—"
        lineas.append(f"  {b['bucket']}%: {b['wins']}-{b['losses']} · acierto {wr}"
                      + (f" · anunciaba {b['prob_media_anunciada']:.1%}"
                         if b["prob_media_anunciada"] else ""))
    lineas.append("")
    lineas.append("Y ahora la parte que importa y que no te voy a maquillar:")
    lineas.append(
        f"NO se ha demostrado ventaja rentable en ningún mercado ni en ningún "
        f"tramo. No es que el sistema pierda: es que solo hay "
        f"{sum(b['n_apostables'] for b in f['buckets'])} apuestas liquidadas con "
        f"cuota registrada. Hacen falta unas 100 por tramo para que el ROI "
        f"signifique algo. Los aciertos ya se pueden leer; el dinero todavía no.")
    return {"ok": True, "tipo": "rendimiento", "status": f["estado"].upper(),
            "respuesta": "\n".join(lineas), "favoritos": f,
            "veredicto": d["veredicto"]["conclusion"]}


def modelos(texto: str) -> dict:
    from CHAT import gating, registry
    t = _norm(texto)
    sport = _deporte(t)
    g = gating.todos()

    filas = []
    for d in registry.listar(sport):
        ev = d.get("evidencia") or {}
        ll, base = ev.get("holdout_log_loss"), ev.get("baseline")
        gg = g.get(d["market_id"]) or {}
        if gg.get("holdout"):
            ll = ll or gg["holdout"]["log_loss_modelo"]
            base = base or gg["holdout"]["log_loss_baseline"]
        filas.append((d, ll, base, ev))

    # Se ordena por cuanto le gana el modelo a su baseline: eso es "bueno".
    con_ev = [(d, ll, base, ev) for d, ll, base, ev in filas
              if ll is not None and base is not None]
    con_ev.sort(key=lambda x: -(x[2] - x[1]))

    lineas = ["Qué tan buenos son mis modelos, medido fuera de muestra "
              "(cuanto menor el log loss, mejor):", ""]
    for d, ll, base, ev in con_ev:
        ganancia = base - ll
        acc = (f" · acierto {ev['accuracy']:.1%} frente a "
               f"{ev['baseline_accuracy']:.1%}" if ev.get("accuracy") else "")
        lineas.append(f"▸ {d['etiqueta']} ({d['sport']}): {ll} vs {base} del "
                      f"baseline → gana {ganancia:+.5f}{acc}")
    sin_ev = [d for d, ll, base, ev in filas if ll is None or base is None]
    if sin_ev:
        lineas.append("")
        lineas.append("Sin evidencia registrada en el walk-forward: "
                      + ", ".join(f"{d['etiqueta']} ({d['sport']})" for d in sin_ev))
    bloqueados = [k for k, v in g.items()
                  if k != "version" and v.get("estado") == "no_pick"]
    if bloqueados:
        lineas.append("")
        lineas.append("Y estos NO publican nada porque no le ganan a su baseline: "
                      + ", ".join(bloqueados) + ".")
    lineas.append("")
    lineas.append("Todos son PROJECTION: buenas probabilidades no es lo mismo que "
                  "ganar dinero, y lo segundo no está demostrado.")
    return {"ok": True, "tipo": "modelos", "status": None,
            "respuesta": "\n".join(lineas), "gating": g}


def picks(texto: str) -> dict:
    from CHAT import games
    from shared import ledger
    t = _norm(texto)
    sport = _deporte(t)
    rs = [r for r in ledger.records(sport, solo_activas=True)
          if r.get("model_probability_calibrated") is not None]
    rs.sort(key=lambda r: -float(r["model_probability_calibrated"]))
    if not rs:
        return {"ok": False, "tipo": "picks", "status": "INSUFFICIENT DATA",
                "respuesta": "No tengo predicciones activas ahora mismo."}

    # Se busca en una ventana mas amplia: una prediccion puede ser de un partido
    # que ya no esta en los proximos 10 dias, y enseniar el id en crudo
    # ('c9bbedd626635edd...') en lugar del nombre no le sirve a nadie.
    nombres = {(j["sport"], j["game_id"]): j["nombre"] for j in games.listar(30)}
    lineas = ["Las probabilidades más altas que tengo publicadas ahora:", ""]
    vistos = 0
    for r in rs:
        n = nombres.get((r["sport"], str(r["event_id"])))
        if not n:
            # Sin nombre no se muestra el hash: se dice la liga, que si informa.
            n = f"{r.get('league') or r['sport']}"
        lineas.append(f"  {float(r['model_probability_calibrated']):.1%}  "
                      f"{r['selection']}  ({r['market']} · {n})")
        vistos += 1
        if vistos >= 10:
            break
    lineas.append("")
    lineas.append(
        "Ojo con cómo lees esto: es la lista de probabilidades más ALTAS, no de "
        "las más rentables. Una probabilidad alta con una cuota baja puede perder "
        "dinero igual. El sistema no tiene histórico suficiente para decirte cuál "
        "conviene apostar, y prefiero decírtelo a dejar que lo supongas.")
    return {"ok": True, "tipo": "picks", "status": "PROJECTION",
            "respuesta": "\n".join(lineas)}


DESPACHO = {"calendario": calendario, "cobertura": cobertura,
            "rendimiento": rendimiento, "modelos": modelos, "picks": picks}


def responder(texto: str) -> dict | None:
    """Contesta una pregunta sobre el sistema. None si no lo es."""
    tipo = clasificar(texto)
    if tipo is None:
        return None
    try:
        r = DESPACHO[tipo](texto)
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "tipo": tipo, "status": "ERROR", "version": VERSION,
                "respuesta": f"No pude consultarlo: {type(e).__name__}: {e}"}
    return {"version": VERSION, "pregunta": texto, "intencion": tipo, **r}
