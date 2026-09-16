"""Resultados reales de los partidos del calendario (el archivo esta congelado).

Fuente principal: TennisExplorer (`/results/`). Publica el marcador SET A SET,
que es lo unico que permite calificar TOTAL DE JUEGOS y HANDICAP DE JUEGOS. De
ahi se cuentan los juegos reales de cada jugador; no se estima nada.

Fuente de respaldo: The Odds API `/scores` (misma clave). Solo da ganador y sets,
asi que cierra el mercado de GANADOR y deja juegos en None. Cuando despues
TennisExplorer publica el detalle, `fetch` vuelve por ese partido y completa los
juegos (por eso se reintentan los resultados sin juegos).

ESPN queda como fuente opcional (TENIS_ESPN=1); hoy responde 403.

Regla que no se rompe: si no hay marcador por juegos, la prediccion queda
PENDIENTE. Nunca se cierra a ojo.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import urllib.request
from datetime import date, datetime, timedelta

from sqlalchemy import select

from TENIS.markets.db import TenisResult, TenisSchedule, TenisSourceLog, init_db
from shared.timeutil import utcnow

TE = "https://www.tennisexplorer.com/results/?type={tipo}-single&year={y}&month={m:02d}&day={d:02d}"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard?dates={fecha}"
ODDS = "https://api.the-odds-api.com/v4/sports/{sport}/scores/?daysFrom={dias}&apiKey={key}"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
     "Accept": "application/json"}
H_HTML = {"User-Agent": H["User-Agent"], "Accept": "text/html,application/xhtml+xml",
          "Accept-Language": "en-US,en;q=0.9"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def _apellido(s: str) -> str:
    n = _norm(s).split()
    return n[-1] if n else ""


def _get(url, timeout=25):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")), None
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _log(s, dominio, url, status, n=0, err=None):
    s.add(TenisSourceLog(source="resultados", domain=dominio, url=url, status=status, records=n,
                         error=err, retrieved_at=utcnow()))


def _guardar(s, sc: TenisSchedule, ganador, g1, g2, s1, s2, estado, detalle, fuente):
    r = s.get(TenisResult, sc.event_id)
    if r is None:
        r = TenisResult(event_id=sc.event_id); s.add(r)
    r.tour, r.match_date = sc.tour, sc.start_utc.date()
    r.p1_name, r.p2_name, r.winner_name = sc.p1_name, sc.p2_name, ganador
    r.p1_games, r.p2_games = g1, g2
    r.games_total = (g1 + g2) if (g1 is not None and g2 is not None) else None
    r.games_margin = (g1 - g2) if (g1 is not None and g2 is not None) else None
    r.sets_p1, r.sets_p2 = s1, s2
    r.status, r.detail, r.source, r.fetched_at = estado, detalle, fuente, utcnow()


def _html(url, timeout=30):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H_HTML), timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore"), None
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _celdas(fila: str) -> list[str]:
    return [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").replace("&nbsp;", "").strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", fila, re.S)]


def _te_apellido(nombre: str) -> str:
    """'Romero Gormaz L. (4)' -> 'gormaz'. TennisExplorer pone las iniciales al final."""
    limpio = re.sub(r"\(.*?\)", " ", str(nombre or ""))
    partes = [p for p in _norm(limpio).split() if len(p) > 1]
    return partes[-1] if partes else ""


def _juegos(valor: str):
    """'6'->6, '68'->6 (perdio el tiebreak 6-8), '10'->10. Vacio -> None."""
    v = re.sub(r"[^0-9]", "", str(valor or ""))
    if not v:
        return None
    if v.startswith("10"):
        return 10
    return int(v[0])


def _te_partidos(html_txt: str) -> list[dict]:
    """Extrae los partidos terminados de una pagina de resultados."""
    filas = re.findall(r"<tr[^>]*>(.*?)</tr>", html_txt, re.S)
    partidos, i = [], 0
    while i < len(filas) - 1:
        a = _celdas(filas[i])
        if not a or not re.match(r"^\d{1,2}:\d{2}$", a[0]):
            i += 1
            continue
        b = _celdas(filas[i + 1])
        i += 2
        if len(a) < 4 or len(b) < 3:
            continue
        n1, n2 = a[1], b[0]
        try:
            sets1, sets2 = int(a[2]), int(b[1])
        except (TypeError, ValueError):
            continue                                   # abandono, W.O. o partido sin marcador
        g1 = [_juegos(x) for x in a[3:8]]
        g2 = [_juegos(x) for x in b[2:7]]
        pares = [(x, y) for x, y in zip(g1, g2) if x is not None and y is not None]
        if not pares or sets1 == sets2:
            continue
        partidos.append({"n1": n1, "n2": n2, "sets1": sets1, "sets2": sets2,
                         "juegos1": sum(x for x, _ in pares), "juegos2": sum(y for _, y in pares),
                         "n_sets": len(pares)})
    return partidos


def _tennisexplorer(s, pendientes) -> int:
    """Marcador set a set: es la unica fuente que permite calificar juegos.

    La web agrupa por su propio huso horario, asi que un partido de madrugada UTC
    puede salir en la pagina del dia anterior o del siguiente: se miran los tres.
    """
    n = 0
    dias = sorted({sc.start_utc.date() + timedelta(days=k) for sc in pendientes for k in (-1, 0, 1)})
    cerrados: set[str] = set()
    for f in dias:
        for tipo, tour in (("atp", "ATP"), ("wta", "WTA")):
            faltan = [sc for sc in pendientes if sc.tour == tour and sc.event_id not in cerrados
                      and abs((sc.start_utc.date() - f).days) <= 1]
            if not faltan:
                continue
            url = TE.format(tipo=tipo, y=f.year, m=f.month, d=f.day)
            txt, err = _html(url)
            if txt is None:
                _log(s, f"tennisexplorer-{tipo}", url, "error", 0, err)
                continue
            partidos = _te_partidos(txt)
            _log(s, f"tennisexplorer-{tipo}", url, "ok", len(partidos))
            for sc in faltan:
                a1, a2 = _apellido(sc.p1_name), _apellido(sc.p2_name)
                for p in partidos:
                    b1, b2 = _te_apellido(p["n1"]), _te_apellido(p["n2"])
                    if {a1, a2} != {b1, b2} or "" in {a1, a2, b1, b2}:
                        continue
                    directo = a1 == b1
                    j1 = p["juegos1"] if directo else p["juegos2"]
                    j2 = p["juegos2"] if directo else p["juegos1"]
                    s1 = p["sets1"] if directo else p["sets2"]
                    s2 = p["sets2"] if directo else p["sets1"]
                    ganador = sc.p1_name if s1 > s2 else sc.p2_name
                    _guardar(s, sc, ganador, j1, j2, s1, s2, "final",
                             f"{p['n_sets']} sets", "tennisexplorer")
                    cerrados.add(sc.event_id)
                    n += 1
                    break
    s.flush()
    return n


def _espn(s, pendientes) -> int:
    fechas = sorted({sc.start_utc.date() for sc in pendientes})
    n = 0
    for f in fechas:
        for tour_key, tour in (("atp", "ATP"), ("wta", "WTA")):
            if not any(sc.tour == tour and sc.start_utc.date() == f for sc in pendientes):
                continue
            url = ESPN.format(tour=tour_key, fecha=f.strftime("%Y%m%d"))
            data, err = _get(url)
            if data is None:
                _log(s, f"espn-{tour_key}", url, "error", 0, err); continue
            eventos = data.get("events", [])
            _log(s, f"espn-{tour_key}", url, "ok", len(eventos))
            for ev in eventos:
                comp = (ev.get("competitions") or [{}])[0]
                st = ((ev.get("status") or comp.get("status") or {}).get("type") or {})
                if not st.get("completed"):
                    continue
                jug = []
                for c in comp.get("competitors") or []:
                    ath = c.get("athlete") or {}
                    nombre = ath.get("displayName") or ath.get("shortName") or ""
                    juegos = sum(int(ls.get("value") or 0) for ls in (c.get("linescores") or []))
                    sets = sum(1 for ls in (c.get("linescores") or []) if ls.get("winner"))
                    jug.append({"nombre": nombre, "gano": bool(c.get("winner")), "juegos": juegos,
                                "sets": sets, "n_sets": len(c.get("linescores") or [])})
                if len(jug) != 2:
                    continue
                ap = {_apellido(j["nombre"]) for j in jug}
                for sc in pendientes:
                    if sc.tour != tour or sc.start_utc.date() != f:
                        continue
                    if {_apellido(sc.p1_name), _apellido(sc.p2_name)} != ap:
                        continue
                    a = next(j for j in jug if _apellido(j["nombre"]) == _apellido(sc.p1_name))
                    b = next(j for j in jug if _apellido(j["nombre"]) == _apellido(sc.p2_name))
                    ganador = sc.p1_name if a["gano"] else sc.p2_name
                    con_juegos = a["juegos"] > 0 or b["juegos"] > 0
                    estado = "retired" if "ret" in str(st.get("detail", "")).lower() else "final"
                    _guardar(s, sc, ganador, a["juegos"] if con_juegos else None,
                             b["juegos"] if con_juegos else None, a["sets"], b["sets"],
                             estado, st.get("detail"), "espn")
                    n += 1
    s.flush()
    return n


def _odds_api(s, pendientes) -> int:
    key = os.getenv("THE_ODDS_API_KEY")
    if not key:
        return 0
    n = 0
    for sport in sorted({sc.sport_key for sc in pendientes if sc.sport_key}):
        url = ODDS.format(sport=sport, dias=3, key=key)
        data, err = _get(url, 40)
        if data is None:
            _log(s, f"odds-{sport}", url.split("&apiKey")[0], "error", 0, err); continue
        _log(s, f"odds-{sport}", url.split("&apiKey")[0], "ok", len(data))
        por_id = {e.get("id"): e for e in data}
        for sc in pendientes:
            ev = por_id.get(sc.event_id)
            if not ev or not ev.get("completed") or not ev.get("scores"):
                continue
            marc = {_apellido(x.get("name")): x.get("score") for x in ev["scores"]}
            a, b = marc.get(_apellido(sc.p1_name)), marc.get(_apellido(sc.p2_name))
            if a is None or b is None:
                continue
            try:
                s1, s2 = int(a), int(b)
            except (TypeError, ValueError):
                continue
            ganador = sc.p1_name if s1 > s2 else sc.p2_name
            _guardar(s, sc, ganador, None, None, s1, s2, "final", "sets", "the_odds_api")
            n += 1
    s.flush()
    return n


def fetch(session, days_back: int = 4) -> dict:
    """Trae resultados de los partidos jugados que aun no los tienen.

    Tambien reintenta los que ya tienen ganador pero se quedaron sin juegos (por
    ejemplo los cerrados por The Odds API), para poder calificar total y handicap.
    """
    init_db()
    ahora = utcnow()
    scs = session.execute(select(TenisSchedule).where(
        TenisSchedule.start_utc <= ahora - timedelta(hours=1),
        TenisSchedule.start_utc >= ahora - timedelta(days=days_back))).scalars().all()
    res = {r.event_id: r for r in session.execute(select(TenisResult)).scalars()}
    sin_nada = [sc for sc in scs if sc.start_utc and sc.event_id not in res]
    sin_juegos = [sc for sc in scs if sc.start_utc and sc.event_id in res
                  and res[sc.event_id].games_total is None]
    pendientes = sin_nada + sin_juegos
    if not pendientes:
        return {"pendientes": 0, "tennisexplorer": 0, "espn": 0, "odds_api": 0, "sin_resultado": 0}

    n_te = _tennisexplorer(session, pendientes)
    con_juegos = {r.event_id for r in session.execute(select(TenisResult)).scalars()
                  if r.games_total is not None}
    faltan = [sc for sc in pendientes if sc.event_id not in con_juegos]
    n_espn = _espn(session, faltan) if (faltan and os.getenv("TENIS_ESPN") == "1") else 0

    cerrados = {r.event_id for r in session.execute(select(TenisResult)).scalars()}
    faltan_todo = [sc for sc in sin_nada if sc.event_id not in cerrados]
    n_odds = _odds_api(session, faltan_todo) if faltan_todo else 0
    return {"pendientes": len(pendientes), "tennisexplorer": n_te, "espn": n_espn,
            "odds_api": n_odds,
            "sin_resultado": len(pendientes) - n_te - n_espn - n_odds,
            "nota": ("TennisExplorer da el marcador set a set: con el se califican ganador, "
                     "total de juegos y handicap. The Odds API solo cierra el ganador.")}
