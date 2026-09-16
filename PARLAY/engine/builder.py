"""Generador de combinadas (parlays) de 2 y 3 patas.

REGLAS, y por que existen:

 1. Solo entran mercados con estado PICK (ventaja demostrada fuera de muestra).
    Las proyecciones no se combinan: su probabilidad no esta validada como
    ventaja, y multiplicar dos numeros no validados da un tercero peor.
 2. NUNCA dos patas del MISMO partido ni con un EQUIPO/JUGADOR repetido.
    "Gana Sabalenka" y "Sabalenka -2.5" no son independientes; tampoco lo son
    "gana NYY el martes" y "gana NYY el miercoles" (mismo rival, misma serie,
    misma rotacion). Multiplicar probabilidades correlacionadas exagera la
    combinada. Con equipos y partidos distintos la independencia es una
    aproximacion razonable.
 3. La probabilidad de la combinada es el PRODUCTO de las probabilidades
    calibradas de sus patas. Es lo que dice la matematica, no una estimacion
    nueva: si cada pata esta calibrada, el producto lo esta salvo correlacion.
 4. Se ordenan por probabilidad de acertar, que es lo que pediste. Aviso
    permanente: mas patas = MENOS probabilidad (0.70 x 0.70 = 0.49).
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from itertools import combinations

from shared.timeutil import utcnow

MIN_LEG_PROB = 0.55          # una pata por debajo del umbral de pick no entra
MAX_PER_SIZE = 12            # cuantas combinadas se guardan por tamano
MAX_LEGS_POOL = 14           # candidatas por partido (las mejores por probabilidad)


def _cards(date_from: str, date_to: str) -> list[dict]:
    """Tarjetas de todos los deportes, en solo lectura."""
    out = []
    for mod, name in (("NFL.adapter", "NFL"), ("MLB.engine.provider", "MLB"),
                      ("NFL.markets.provider", "NFL"), ("NBA.markets.provider", "NBA"),
                      ("TENIS.markets.provider", "TENIS")):
        try:
            m = __import__(mod, fromlist=["games"])
            out.extend(m.games(date_from, date_to))
        except Exception:                                  # noqa: BLE001
            continue
    return out


def eligible_legs(cards: list[dict]) -> list[dict]:
    """Patas validas: mercado PICK, probabilidad suficiente y partido futuro."""
    legs = []
    for c in cards:
        ex = c.get("extra") or {}
        p = c.get("ensemble_probability")
        if p is None or c.get("result") is not None:
            continue
        # PICK: MLB/NFL moneyline vienen sin `mode`; el resto lo declara
        modo = ex.get("mode")
        es_pick = (modo == "pick") if modo else (ex.get("publish_pick", True) is not False)
        if not es_pick or float(p) < MIN_LEG_PROB:
            continue
        legs.append({
            "sport": c["sport"], "game_id": str(c["game_id"]), "prediction_id": c.get("prediction_id"),
            "market": c.get("market"), "market_key": ex.get("market_key") or "moneyline",
            "selection": c.get("selection"), "probability": float(p),
            "market_probability": c.get("market_probability"),
            "start_utc": c.get("start_utc"), "teams": f"{c.get('away')} @ {c.get('home')}"
            if c["sport"] != "TENIS" else f"{c.get('home')} vs {c.get('away')}",
            "participants": {f"{c['sport']}:{x}" for x in (c.get("home"), c.get("away")) if x},
            "date": c.get("game_date")})
    legs.sort(key=lambda l: -l["probability"])
    # una pata por partido Y por equipo/jugador: dos partidos de la misma serie
    # (o del mismo jugador) no son sucesos independientes
    vistos_partido, vistos_part, pool = set(), set(), []
    for l in legs:
        k = (l["sport"], l["game_id"])
        if k in vistos_partido or (l["participants"] & vistos_part):
            continue
        vistos_partido.add(k); vistos_part |= l["participants"]; pool.append(l)
    return pool[:MAX_LEGS_POOL]


def _key(legs) -> str:
    s = "|".join(sorted(f"{l['sport']}:{l['game_id']}:{l['market_key']}:{l['selection']}" for l in legs))
    return hashlib.md5(s.encode()).hexdigest()


def _ts(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).replace("T", " ")[:19]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def combos(pool: list[dict], sizes=(2, 3), max_per_size: int = MAX_PER_SIZE) -> list[dict]:
    out = []
    for size in sizes:
        cand = []
        for combo in combinations(pool, size):
            if len({(l["sport"], l["game_id"]) for l in combo}) != size:
                continue                                   # nunca dos patas del mismo partido
            parts = set()
            for l in combo:
                parts |= l["participants"]
            if len(parts) != sum(len(l["participants"]) for l in combo):
                continue                                   # ni con un equipo/jugador repetido
            p = 1.0
            for l in combo:
                p *= l["probability"]
            mp = 1.0
            for l in combo:
                if l["market_probability"] is None:
                    mp = None; break
                mp *= float(l["market_probability"])
            starts = [t for t in (_ts(l["start_utc"]) for l in combo) if t is not None]
            sports = sorted({l["sport"] for l in combo})
            cand.append({
                "legs": list(combo), "size": size, "probability": p, "fair_odds": (1 / p) if p else None,
                "market_probability": mp, "market_odds": (1 / mp) if mp else None,
                "min_leg_probability": min(l["probability"] for l in combo),
                "sports": ",".join(sports), "mixed": len(sports) > 1,
                "first_start_utc": min(starts) if starts else None,
                "last_start_utc": max(starts) if starts else None,
                "match_date": (min(starts).date() if starts else None),
                "key": _key(combo)})
        cand.sort(key=lambda c: -c["probability"])
        out.extend(cand[:max_per_size])
    return out


def generate(date_from: str | None = None, date_to: str | None = None, sizes=(2, 3),
             max_per_size: int = MAX_PER_SIZE) -> dict:
    """Genera y guarda las combinadas con mayor probabilidad para el rango dado."""
    from sqlalchemy import select
    from PARLAY.engine.db import Parlay, ParlayLeg, ParlayRun, init_db, session_scope
    init_db()
    hoy = date.today()
    a = date_from or str(hoy)
    b = date_to or str(hoy + timedelta(days=2))
    cards = _cards(a, b)
    pool = eligible_legs(cards)
    cs = combos(pool, sizes, max_per_size)
    now = utcnow()
    nuevas = 0
    with session_scope() as s:
        run = ParlayRun(started_at=now); s.add(run); s.flush()
        for i, c in enumerate(cs, 1):
            if s.execute(select(Parlay).where(Parlay.parlay_key == c["key"])).scalars().first():
                continue
            p = Parlay(parlay_key=c["key"], size=c["size"], mixed=c["mixed"], sports=c["sports"],
                       created_at=now, first_start_utc=c["first_start_utc"], last_start_utc=c["last_start_utc"],
                       match_date=c["match_date"], probability=c["probability"], fair_odds=c["fair_odds"],
                       market_probability=c["market_probability"], market_odds=c["market_odds"],
                       min_leg_probability=c["min_leg_probability"], rank=i, status="open",
                       legs_total=c["size"], legs_won=0,
                       notes=("Patas independientes de partidos distintos; probabilidad = producto de las "
                              "probabilidades calibradas."))
            s.add(p); s.flush()
            for l in c["legs"]:
                s.add(ParlayLeg(parlay_id=p.id, sport=l["sport"], game_id=l["game_id"],
                                prediction_id=l["prediction_id"], market=l["market"],
                                market_key=l["market_key"], selection=l["selection"],
                                probability=l["probability"], market_probability=l["market_probability"],
                                start_utc=_ts(l["start_utc"]), teams=l["teams"], result=None))
            nuevas += 1
        run.finished_at = utcnow()
        run.summary = {"desde": a, "hasta": b, "tarjetas": len(cards), "patas_elegibles": len(pool),
                       "combinadas_generadas": len(cs), "nuevas": nuevas}
        resumen = dict(run.summary)
    return resumen
