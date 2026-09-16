"""Datos ACTUALES: calendario, cuotas y resultados desde The Odds API.

Esto es lo que convierte al modulo en un sistema que predice partidos proximos y
no en un backtester. Tres funciones, cada una idempotente:

    fixtures()  proximos partidos por liga (alta y actualizacion, sin duplicar)
    odds()      snapshot de cuotas con el momento REAL de recepcion
    results()   marcadores finales -> crecen `fixtures` y el historico propio

Reglas que se respetan aqui:
- `fetched_at` es la hora de ESTA maquina al recibir la respuesta. Nunca se
  inventa ni se copia de otro sitio.
- `bookmaker_update` es la hora que declara la casa; si no viene, queda NULL.
- Si la API falla, se registra el error y se sigue con las demas ligas: una
  fuente caida no puede tumbar el ciclo entero.
- Nada de lo que se guarda aqui entra en las features de un partido ya jugado.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from sqlalchemy import select

from SOCCER.db import (SoccerFixture, SoccerOdds, SoccerSourceLog, init_db, session_scope)
from SOCCER.leagues import LIGAS, POR_ODDS_KEY
from shared.odds import decimal_to_prob, devig
from shared.timeutil import utcnow

API = "https://api.the-odds-api.com/v4/sports/{key}"
H = {"User-Agent": "SportsPredictionCenter/1.0", "Accept": "application/json"}
# Se piden en grupos: si un mercado no esta soportado la API devuelve 422 y
# tumbaria la peticion entera. Verificado en vivo: pedir 'btts' junto a
# 'h2h,totals' daba 422 en las siete ligas. Ahora cada grupo va por su cuenta y
# el que falle se registra sin arrastrar a los demas.
GRUPOS_MERCADO = ("h2h,totals", "btts")
REGIONES = "eu,uk,us"


class OddsAPIError(RuntimeError):
    pass


def api_key() -> str | None:
    return os.getenv("THE_ODDS_API_KEY") or None


def _get(url: str, timeout: int = 40):
    """Devuelve (datos, cabeceras, error). Nunca lanza: el ciclo debe sobrevivir."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")), dict(r.headers), None
    except urllib.error.HTTPError as e:
        return None, {}, f"HTTP {e.code}: {e.reason}"
    except Exception as e:                                   # noqa: BLE001
        return None, {}, f"{type(e).__name__}: {e}"


def _log(s, source, scope, url, status, n=0, err=None):
    s.add(SoccerSourceLog(source=source, scope=scope,
                          url=(url.split("&apiKey")[0] if url else None),
                          status=status, records=n, error=err, retrieved_at=utcnow()))


def _dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(
            timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# 1. Calendario
# --------------------------------------------------------------------------- #
def fixtures(codes: list[str] | None = None, dias: int = 8) -> dict:
    """Alta y actualiza los proximos partidos. No duplica: la clave es el event_id."""
    init_db()
    key = api_key()
    if not key:
        return {"ok": False, "motivo": "THE_ODDS_API_KEY no configurada", "nuevos": 0,
                "actualizados": 0, "por_liga": {}}
    ahora = utcnow()
    nuevos = act = 0
    por_liga: dict[str, int] = {}
    errores: dict[str, str] = {}
    with session_scope() as s:
        vistos = {f.event_id: f for f in s.execute(select(SoccerFixture)).scalars()}
        for lg in LIGAS:
            if codes and lg.code not in [c.upper() for c in codes]:
                continue
            url = f"{API.format(key=lg.odds_key)}/events/?apiKey={key}"
            data, _, err = _get(url)
            if data is None:
                _log(s, "the_odds_api", f"events {lg.code}", url, "error", 0, err)
                errores[lg.code] = err or "sin datos"
                continue
            _log(s, "the_odds_api", f"events {lg.code}", url, "ok", len(data))
            por_liga[lg.code] = len(data)
            for ev in data:
                eid, ini = ev.get("id"), _dt(ev.get("commence_time"))
                if not eid or ini is None:
                    continue                     # respuesta incompleta: se ignora, no se rellena
                f = vistos.get(eid)
                if f is None:
                    f = SoccerFixture(event_id=eid, primera_vez=ahora)
                    s.add(f)
                    vistos[eid] = f
                    nuevos += 1
                else:
                    act += 1
                f.league_code, f.odds_key = lg.code, lg.odds_key
                f.commence_utc = ini
                f.home_team = ev.get("home_team") or ""
                f.away_team = ev.get("away_team") or ""
                f.fetched_at = ahora
            s.flush()
    return {"ok": True, "nuevos": nuevos, "actualizados": act, "por_liga": por_liga,
            "errores": errores, "consultado_en": ahora.isoformat()}


# --------------------------------------------------------------------------- #
# 2. Cuotas
# --------------------------------------------------------------------------- #
def _guardar_mercado(s, ev_id, code, market, book, titulo, upd, ahora, selecciones) -> int:
    """`selecciones` = [(nombre, linea, precio_decimal)]. Devig solo si esta completo."""
    precios = [decimal_to_prob(p) for _, _, p in selecciones]
    sin_vig = devig(precios, "proportional") if all(x is not None for x in precios) else None
    n = 0
    for i, (nombre, linea, precio) in enumerate(selecciones):
        s.add(SoccerOdds(event_id=ev_id, league_code=code, market=market, bookmaker=titulo or book,
                         selection=nombre, line=linea, price=float(precio),
                         implied=precios[i], novig=(sin_vig[i] if sin_vig else None),
                         bookmaker_update=upd, fetched_at=ahora, snapshot_kind="current"))
        n += 1
    return n


def odds(codes: list[str] | None = None) -> dict:
    """Un snapshot por casa y mercado, con la hora real de recepcion."""
    init_db()
    key = api_key()
    if not key:
        return {"ok": False, "motivo": "THE_ODDS_API_KEY no configurada", "snapshots": 0}
    ahora = utcnow()
    total = 0
    por_liga: dict[str, int] = {}
    eventos_con_cuotas: set[str] = set()
    errores: dict[str, str] = {}
    restantes = None
    with session_scope() as s:
        for lg in LIGAS:
            if codes and lg.code not in [c.upper() for c in codes]:
                continue
            eventos: dict[str, dict] = {}
            fallos = []
            for grupo in GRUPOS_MERCADO:
                url = (f"{API.format(key=lg.odds_key)}/odds/?apiKey={key}"
                       f"&regions={REGIONES}&markets={grupo}&oddsFormat=decimal")
                data, cab, err = _get(url)
                if data is None:
                    _log(s, "the_odds_api", f"odds {lg.code} [{grupo}]", url, "error", 0, err)
                    fallos.append(f"{grupo}: {err}")
                    continue
                restantes = cab.get("x-requests-remaining", restantes)
                for ev in data:                 # se acumulan las casas de cada grupo
                    eid = ev.get("id")
                    if not eid:
                        continue
                    if eid not in eventos:
                        eventos[eid] = {"id": eid, "bookmakers": []}
                    eventos[eid]["bookmakers"].extend(ev.get("bookmakers") or [])
            if fallos:
                errores[lg.code] = "; ".join(fallos)
            if not eventos:
                continue
            data = list(eventos.values())
            n_liga = 0
            for ev in data:
                eid = ev.get("id")
                if not eid:
                    continue
                for bk in ev.get("bookmakers") or []:
                    upd = _dt(bk.get("last_update"))
                    for mk in bk.get("markets") or []:
                        clave = mk.get("key")
                        outs = mk.get("outcomes") or []
                        if clave == "h2h" and len(outs) == 3:
                            sel = [(o.get("name"), None, o.get("price")) for o in outs]
                        elif clave == "totals" and len(outs) == 2:
                            sel = [(o.get("name"), o.get("point"), o.get("price")) for o in outs]
                        elif clave == "btts" and len(outs) == 2:
                            sel = [(o.get("name"), None, o.get("price")) for o in outs]
                        else:
                            continue
                        if any(p is None for _, _, p in sel):
                            continue             # respuesta incompleta: no se rellena
                        n_liga += _guardar_mercado(s, eid, lg.code, clave, bk.get("key"),
                                                   bk.get("title"), upd, ahora, sel)
                        eventos_con_cuotas.add(eid)
            _log(s, "the_odds_api", f"odds {lg.code}", None, "ok", n_liga)
            por_liga[lg.code] = n_liga
            total += n_liga
            s.flush()
    return {"ok": True, "snapshots": total, "por_liga": por_liga,
            "eventos_con_cuotas": len(eventos_con_cuotas), "errores": errores,
            "creditos_restantes": restantes, "consultado_en": ahora.isoformat()}


# --------------------------------------------------------------------------- #
# 3. Resultados
# --------------------------------------------------------------------------- #
def results(codes: list[str] | None = None, days_from: int = 3) -> dict:
    """Marcadores finales. Cierran el fixture y alimentan el historico propio."""
    init_db()
    key = api_key()
    if not key:
        return {"ok": False, "motivo": "THE_ODDS_API_KEY no configurada", "cerrados": 0}
    ahora = utcnow()
    cerrados = 0
    errores: dict[str, str] = {}
    with session_scope() as s:
        for lg in LIGAS:
            if codes and lg.code not in [c.upper() for c in codes]:
                continue
            url = (f"{API.format(key=lg.odds_key)}/scores/?daysFrom={days_from}&apiKey={key}")
            data, _, err = _get(url)
            if data is None:
                _log(s, "the_odds_api", f"scores {lg.code}", url, "error", 0, err)
                errores[lg.code] = err or "sin datos"
                continue
            _log(s, "the_odds_api", f"scores {lg.code}", url, "ok", len(data))
            for ev in data:
                f = s.get(SoccerFixture, ev.get("id"))
                if f is None or not ev.get("completed"):
                    continue
                marc = {x.get("name"): x.get("score") for x in (ev.get("scores") or [])}
                try:
                    hg, ag = int(marc[f.home_team]), int(marc[f.away_team])
                except (KeyError, TypeError, ValueError):
                    continue                     # marcador incompleto: queda pendiente
                if f.completed and f.home_goals == hg and f.away_goals == ag:
                    continue
                f.completed, f.home_goals, f.away_goals, f.fetched_at = True, hg, ag, ahora
                cerrados += 1
            s.flush()
    return {"ok": True, "cerrados": cerrados, "errores": errores,
            "consultado_en": ahora.isoformat()}
