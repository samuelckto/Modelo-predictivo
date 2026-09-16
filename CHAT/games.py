"""Catalogo de partidos para el chat. Version CHAT_GAMES_v1.

Nadie deberia tener que saberse un `event_id` de 32 caracteres para preguntar
por un partido. Este modulo hace dos cosas:

  listar()    todos los partidos proximos con su NOMBRE, ordenados por hora
  resolver()  encuentra el partido a partir de lo que escribio el usuario
              ("¿cuantas tarjetas en el Barcelona?")

La resolucion es por TOKENS del nombre del equipo, no por parecido de cadenas.
Es la misma leccion que en el resolver de clubes de futbol: la similitud de
cadenas daba 0.81 entre 'manchester united' y 'manchester city' y los fusionaba.
Aqui el error equivalente seria contestar del partido equivocado, que es peor
que no contestar.

Cuando varios partidos encajan igual de bien NO se elige uno: se devuelven todos
para que el usuario desempate. Adivinar en un empate es exactamente como se
acaba dando el dato de otro partido con total seguridad.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta

VERSION = "CHAT_GAMES_v1"

# Palabras que no distinguen a un equipo de otro y solo anaden ruido al cruce.
RUIDO = {"fc", "cf", "sc", "ac", "afc", "cd", "ud", "sd", "club", "de", "del",
         "the", "los", "las", "el", "la", "y", "and", "vs", "contra", "en",
         "partido", "juego", "del", "que", "para"}


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s) -> set[str]:
    return {t for t in _norm(s).split() if t not in RUIDO and len(t) > 2}


# Cache corto. Recolectar el calendario cuesta ~5 s porque pregunta a los cinco
# motores, y el desplegable lo pide en cada pulsacion. El calendario no cambia
# de un minuto a otro, asi que 60 s de cache no oculta nada y hace la interfaz
# usable. TTL corto a proposito: si un ciclo publica predicciones nuevas, se ven
# al minuto siguiente sin reiniciar el servidor.
TTL = 60
_cache: dict[tuple, tuple[float, list]] = {}


def limpiar_cache() -> None:
    _cache.clear()


def listar(dias: int = 10, sport: str | None = None) -> list[dict]:
    """Partidos proximos con nombre legible, uno por partido (no por mercado).

    Se apoya en el mismo `_collect` del dashboard, asi que si un partido sale en
    el calendario tambien sale aqui: no hay dos fuentes de verdad que se puedan
    desincronizar.
    """
    import time

    from dashboard.backend.main import _collect
    clave = (dias, (sport or "all").upper())
    hit = _cache.get(clave)
    if hit and (time.time() - hit[0]) < TTL:
        return hit[1]
    hoy = date.today()
    cards, warns = _collect((sport or "all").upper() if sport else "all",
                            str(hoy), str(hoy + timedelta(days=dias)))
    juegos: dict[tuple, dict] = {}
    for c in cards:
        clave = (c.get("sport"), str(c.get("game_id")))
        if clave in juegos:
            juegos[clave]["mercados"].append(c.get("market"))
            continue
        home = c.get("home_name") or c.get("home") or "?"
        away = c.get("away_name") or c.get("away") or "?"
        juegos[clave] = {
            "sport": c.get("sport"), "game_id": str(c.get("game_id")),
            "home": home, "away": away,
            "nombre": f"{home} vs {away}" if c.get("sport") != "MLB"
                      else f"{away} @ {home}",
            "liga": (c.get("extra") or {}).get("league")
                    or (c.get("extra") or {}).get("liga") or c.get("sport"),
            "start_utc": c.get("start_utc"), "game_date": c.get("game_date"),
            "venue": c.get("venue"), "mercados": [c.get("market")],
        }
    out = sorted(juegos.values(), key=lambda j: (str(j.get("start_utc") or ""),
                                                 j["sport"], j["game_id"]))
    for j in out:
        j["mercados"] = sorted({m for m in j["mercados"] if m})
    _cache[clave] = (time.time(), out)
    return out


def resolver(texto: str, sport: str | None = None, dias: int = 10) -> dict:
    """Encuentra el partido que menciona el texto. Nunca adivina en un empate."""
    juegos = listar(dias, sport)
    if not juegos:
        return {"encontrado": False, "version": VERSION, "candidatos": [],
                "motivo": "no hay partidos en el calendario de los proximos dias."}
    tt = _tokens(texto)
    if not tt:
        return {"encontrado": False, "version": VERSION, "candidatos": [],
                "motivo": "no mencionas ningun equipo."}

    puntuados = []
    for j in juegos:
        th, ta = _tokens(j["home"]), _tokens(j["away"])
        # Se puntua por equipo: nombrar a los DOS vale mas que nombrar a uno.
        p = (2 if (th & tt) else 0) + (2 if (ta & tt) else 0)
        # y coincidir con mas de un token del mismo equipo tambien suma
        p += len(th & tt) + len(ta & tt)
        if p:
            puntuados.append((p, j))
    if not puntuados:
        return {"encontrado": False, "version": VERSION, "candidatos": [],
                "motivo": "no reconozco ningun equipo de los proximos partidos "
                          "en lo que escribiste."}
    puntuados.sort(key=lambda kv: -kv[0])
    mejor = puntuados[0][0]
    empatados = [j for p, j in puntuados if p == mejor]
    if len(empatados) > 1:
        return {"encontrado": False, "version": VERSION,
                "candidatos": empatados[:6], "ambiguo": True,
                "motivo": f"hay {len(empatados)} partidos que encajan igual de bien. "
                          f"Dime cual."}
    return {"encontrado": True, "version": VERSION, "partido": empatados[0],
            "puntuacion": mejor,
            "otros": [j for _, j in puntuados[1:4]]}


def buscar(q: str, dias: int = 10, limite: int = 20) -> list[dict]:
    """Filtro simple por texto para el selector de la interfaz."""
    juegos = listar(dias)
    if not q:
        return juegos[:limite]
    n = _norm(q)
    return [j for j in juegos
            if n in _norm(j["nombre"]) or n in _norm(j["liga"] or "")
            or n in _norm(j["sport"])][:limite]
