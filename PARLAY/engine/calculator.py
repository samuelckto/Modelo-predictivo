"""Calculadora de momios y combinadas. Version PARLAY_ENGINE_v1.

Herramienta manual: el usuario mete las selecciones que quiera y la calculadora
devuelve la cuota combinada, el pago, la probabilidad implicita del mercado y,
cuando existe, la probabilidad del modelo.

La parte que importa de verdad, y la razon de que este modulo tenga tanto
comentario para tan poca matematica: LA MULTIPLICACION SUPONE INDEPENDENCIA.

    P(A y B) = P(A) x P(B)   solo si A y B son independientes.

Casi nunca lo son del todo. Dos patas del mismo partido estan atadas: "gana el
Madrid" y "over 2.5 goles" se mueven juntas. Dos partidos de la misma liga el
mismo dia comparten arbitraje y clima. Cuando hay correlacion POSITIVA la
multiplicacion INFRAvalora la combinada; cuando es NEGATIVA la SOBREvalora, que
es el caso peligroso porque hace parecer barata una apuesta cara.

Este modulo no inventa una correccion. Detecta los casos en que la
independencia es dudosa, lo dice con nombre y apellidos, y marca la
probabilidad combinada como estimacion bajo independencia. Presentar un numero
de parlay como exacto cuando hay correlacion sin modelar seria mentir con
decimales.
"""
from __future__ import annotations

from shared.roi import to_american, to_decimal

VERSION = "PARLAY_ENGINE_v1"

# Nivel de aviso segun que comparten dos patas.
CORRELACION = {
    "mismo_evento": ("ALTA", "son dos mercados del MISMO partido: los resultados "
                             "estan atados y la multiplicacion no vale."),
    "mismo_participante": ("ALTA", "repiten equipo o jugador: si a uno le va mal, "
                                   "las dos patas caen a la vez."),
    "misma_liga_mismo_dia": ("BAJA", "misma liga y misma fecha: comparten condiciones "
                                     "(arbitraje, clima, calendario), pero el efecto "
                                     "es pequeno."),
}


def _norm(sel: dict) -> dict:
    """Normaliza una seleccion. Acepta cuota americana o decimal, no las dos a medias."""
    d = to_decimal(american=sel.get("odds_american"), decimal=sel.get("odds_decimal"))
    p_modelo = sel.get("model_probability")
    return {
        "sport": sel.get("sport"), "event_id": sel.get("event_id"),
        "event": sel.get("event"), "league": sel.get("league"),
        "market": sel.get("market"), "selection": sel.get("selection"),
        "date": sel.get("date"),
        "participants": [str(p) for p in (sel.get("participants") or [])],
        "odds_decimal": d, "odds_american": to_american(d) if d else None,
        "implied": round(1.0 / d, 6) if d else None,
        "model_probability": float(p_modelo) if p_modelo is not None else None,
        "valida": d is not None,
        "motivo": None if d is not None else "cuota ausente o invalida",
    }


def correlaciones(sels: list[dict]) -> list[dict]:
    """Pares de patas cuya independencia es dudosa, con el motivo."""
    avisos = []
    for i in range(len(sels)):
        for j in range(i + 1, len(sels)):
            a, b = sels[i], sels[j]
            tipo = None
            if a.get("event_id") and a["event_id"] == b.get("event_id"):
                tipo = "mismo_evento"
            elif set(a["participants"]) & set(b["participants"]):
                tipo = "mismo_participante"
            elif (a.get("league") and a["league"] == b.get("league")
                  and a.get("date") and a["date"] == b.get("date")):
                tipo = "misma_liga_mismo_dia"
            if tipo:
                nivel, texto = CORRELACION[tipo]
                avisos.append({
                    "tipo": tipo, "nivel": nivel,
                    "pata_a": f"{a.get('selection')} ({a.get('event') or a.get('sport')})",
                    "pata_b": f"{b.get('selection')} ({b.get('event') or b.get('sport')})",
                    "explicacion": texto,
                    "comun": sorted(set(a["participants"]) & set(b["participants"])) or None,
                })
    return avisos


def calcular(selecciones: list[dict], stake: float = 1.0) -> dict:
    """Combina las selecciones y devuelve todo el desglose.

    No rechaza una combinada correlacionada: la calcula y la marca. Decidir es
    del usuario; ocultarle el aviso no lo seria.
    """
    sels = [_norm(s) for s in (selecciones or [])]
    validas = [s for s in sels if s["valida"]]

    res = {
        "version": VERSION, "stake": float(stake),
        "selecciones": sels,
        "n_selecciones": len(sels), "n_validas": len(validas),
        "descartadas": [{"selection": s.get("selection"), "motivo": s["motivo"]}
                        for s in sels if not s["valida"]],
        "cuota_decimal": None, "cuota_americana": None,
        "probabilidad_implicita_combinada": None,
        "probabilidad_modelo_combinada": None,
        "cobertura_modelo": None,
        "gap_pp": None,
        "payout": None, "profit": None,
        "supuesto": "Probabilidad combinada estimada BAJO INDEPENDENCIA.",
        "avisos_correlacion": [], "nivel_correlacion": "NINGUNA",
        "advertencia": None,
    }
    if not validas:
        res["advertencia"] = ("ninguna seleccion tiene cuota valida; no se puede "
                              "calcular la combinada.")
        return res

    cuota = 1.0
    for s in validas:
        cuota *= s["odds_decimal"]
    res["cuota_decimal"] = round(cuota, 6)
    res["cuota_americana"] = to_american(cuota)
    res["probabilidad_implicita_combinada"] = round(1.0 / cuota, 6)
    res["payout"] = round(stake * cuota, 6)
    res["profit"] = round(stake * (cuota - 1.0), 6)

    # Probabilidad del modelo: SOLO si todas las patas la tienen. Multiplicar
    # unas cuantas y rellenar el resto con la implicita mezclaria dos fuentes y
    # el resultado no seria ni del modelo ni del mercado.
    con_modelo = [s for s in validas if s["model_probability"] is not None]
    res["cobertura_modelo"] = f"{len(con_modelo)}/{len(validas)}"
    if len(con_modelo) == len(validas):
        pm = 1.0
        for s in validas:
            pm *= s["model_probability"]
        res["probabilidad_modelo_combinada"] = round(pm, 6)
        res["gap_pp"] = round((pm - res["probabilidad_implicita_combinada"]) * 100, 3)
        res["cuota_justa_modelo"] = round(1.0 / pm, 4) if pm > 0 else None
    else:
        res["nota_modelo"] = (
            f"solo {len(con_modelo)} de {len(validas)} patas tienen probabilidad del "
            f"modelo. No se calcula la combinada del modelo: mezclar probabilidades "
            f"del modelo con implicitas del mercado daria un numero que no es de "
            f"ninguno de los dos.")

    res["avisos_correlacion"] = correlaciones(validas)
    if any(a["nivel"] == "ALTA" for a in res["avisos_correlacion"]):
        res["nivel_correlacion"] = "ALTA"
        res["advertencia"] = (
            "Hay patas correlacionadas. La probabilidad combinada mostrada NO es "
            "correcta: supone independencia y aqui no la hay. Uselo como referencia, "
            "no como probabilidad real.")
    elif res["avisos_correlacion"]:
        res["nivel_correlacion"] = "BAJA"
        res["advertencia"] = (
            "Hay patas que comparten liga y fecha. El efecto suele ser pequeno, pero "
            "la independencia no es exacta.")

    # Recordatorio permanente: mas patas siempre baja la probabilidad.
    if len(validas) >= 2:
        res["recordatorio"] = (
            f"{len(validas)} patas: la probabilidad se multiplica, no se suma. "
            f"{' x '.join(f'{s['implied']:.0%}' for s in validas)} "
            f"= {res['probabilidad_implicita_combinada']:.1%} segun el mercado.")
    return res


def convertir(american=None, decimal=None) -> dict:
    """Conversor suelto entre formatos, para la interfaz."""
    d = to_decimal(american=american, decimal=decimal)
    if d is None:
        return {"valida": False,
                "motivo": ("cuota americana invalida: no existe ningun precio entre "
                           "-100 y +100" if american is not None else
                           "cuota decimal invalida: debe ser mayor que 1")}
    return {"valida": True, "decimal": round(d, 6), "americana": to_american(d),
            "implicita": round(1.0 / d, 6),
            "beneficio_por_unidad": round(d - 1.0, 6)}
