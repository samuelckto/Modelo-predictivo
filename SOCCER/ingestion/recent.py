"""Temporadas recientes. Sin esto el modulo no sirve para predecir hoy.

El espejo `footballcsv` se detiene en junio de 2024. Si la forma de un equipo se
calculara solo con eso, una prediccion de hoy usaria datos de hace mas de un ano:
seria un backtester disfrazado. Esta fuente cierra ese hueco.

Fuentes, ambas verificadas en vivo:
  - openfootball/football.json  -> 5 ligas europeas, 2024-25 en adelante, con
    fecha, HORA de saque, marcador final y de medio tiempo.
  - openfootball/world (.txt)   -> Liga MX, formato de texto por jornadas.

openfootball SI trae hora de saque en las europeas: cuando viene, se guarda en
`kickoff_utc` y el corte anti-leakage puede ser mas fino. Cuando no viene, queda
NULL y se usa 00:00 del dia, como en el resto del historico.
"""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import date, datetime

from sqlalchemy import select

from SOCCER.db import SoccerMatch, SoccerSourceLog, init_db, session_scope
from SOCCER.ingestion.historical import _mid, _norm
from shared.timeutil import utcnow

JSON_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master/{temp}/{f}.json"
TXT_BASE = "https://raw.githubusercontent.com/openfootball/world/master/{ruta}"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124"}

# liga -> archivo en football.json
JSON_FILE = {"EPL": "en.1", "LALIGA": "es.1", "SERIEA": "it.1",
             "BUNDES": "de.1", "LIGUE1": "fr.1"}
# liga -> plantilla de ruta en openfootball/world (una por temporada)
TXT_RUTA = {"LIGAMX": "north-america/mexico/{temp}_mx1.txt",
            "SAUDI": "middle-east/saudi-arabia/{temp}_sa1.txt"}

MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def temporadas_recientes(desde: int = 2024, hasta: int | None = None) -> list[str]:
    fin = hasta if hasta is not None else utcnow().year
    return ["%d-%02d" % (y, (y + 1) % 100) for y in range(desde, fin + 1)]


def _get(url, timeout=40):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore"), None
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _log(s, source, scope, url, status, n=0, err=None):
    s.add(SoccerSourceLog(source=source, scope=scope, url=url, status=status, records=n,
                          error=err, retrieved_at=utcnow()))


def _parse_json(txt: str) -> list[dict]:
    """football.json: {date, time, team1, team2, score:{ft:[h,a], ht:[h,a]}}."""
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        return []
    out = []
    for m in d.get("matches", []):
        sc = m.get("score")
        if not isinstance(sc, dict):        # algunas temporadas traen otra forma: no se adivina
            continue
        ft = sc.get("ft")
        if not (isinstance(ft, list) and len(ft) == 2):
            continue                              # sin jugar: no se inventa marcador
        try:
            f = date.fromisoformat(str(m.get("date")))
        except (TypeError, ValueError):
            continue
        ko = None
        hora = str(m.get("time") or "")
        if re.match(r"^\d{1,2}:\d{2}$", hora):
            hh, mm = hora.split(":")
            ko = datetime(f.year, f.month, f.day, int(hh), int(mm))
        ht = sc.get("ht") if isinstance(sc.get("ht"), list) and len(sc.get("ht") or []) == 2 else None
        out.append({"fecha": f, "kickoff": ko, "home": m.get("team1"), "away": m.get("team2"),
                    "hg": int(ft[0]), "ag": int(ft[1]),
                    "hthg": int(ht[0]) if ht else None, "htag": int(ht[1]) if ht else None,
                    "etapa": m.get("round")})
    return out


_TXT_FECHA = re.compile(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_TXT_PARTIDO = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?(.+?)\s{2,}v\s+(.+?)\s{2,}(\d+)-(\d+)(?:\s*\((\d+)-(\d+)\))?\s*$")
_TXT_ETAPA = re.compile(r"^\s*[▪●-]\s*(.+?)\s*$")


def _parse_txt(txt: str, anio_defecto: int) -> list[dict]:
    """Formato de openfootball/world: cabecera de fecha y lineas de partido."""
    out, f_actual, etapa, anio = [], None, None, anio_defecto
    for ln in txt.splitlines():
        mf = _TXT_FECHA.match(ln)
        if mf:
            mes, dia, an = mf.group(2), int(mf.group(3)), mf.group(4)
            if an:
                anio = int(an)
            if mes in MESES:
                try:
                    f_actual = date(anio, MESES[mes], dia)
                except ValueError:
                    f_actual = None
            continue
        mp = _TXT_PARTIDO.match(ln)
        if mp and f_actual:
            hora, home, away, hg, ag, hthg, htag = mp.groups()
            ko = None
            if hora:
                hh, mm = hora.split(":")
                ko = datetime(f_actual.year, f_actual.month, f_actual.day, int(hh), int(mm))
            out.append({"fecha": f_actual, "kickoff": ko, "home": home.strip(), "away": away.strip(),
                        "hg": int(hg), "ag": int(ag),
                        "hthg": int(hthg) if hthg else None, "htag": int(htag) if htag else None,
                        "etapa": etapa})
            continue
        me = _TXT_ETAPA.match(ln)
        if me and "v " not in ln and len(ln.strip()) < 60:
            etapa = me.group(1)
    return out


def ingest(codes: list[str] | None = None, desde: int = 2024, progress=None) -> dict:
    """Rellena las temporadas que el espejo no tiene. Idempotente."""
    init_db()
    ahora = utcnow()
    resumen: dict[str, dict] = {}
    with session_scope() as s:
        existentes = set(s.execute(select(SoccerMatch.match_id)).scalars())
        objetivo = {**{k: ("json", v) for k, v in JSON_FILE.items()},
                    **{k: ("txt", v) for k, v in TXT_RUTA.items()}}
        for code, (tipo, plantilla) in objetivo.items():
            if codes and code not in [c.upper() for c in codes]:
                continue
            r = {"nuevos": 0, "temporadas_ok": 0, "sin_datos": 0}
            for temp in temporadas_recientes(desde):
                if tipo == "json":
                    url = JSON_BASE.format(temp=temp, f=plantilla)
                else:
                    url = TXT_BASE.format(ruta=plantilla.format(temp=temp))
                txt, err = _get(url)
                if txt is None:
                    _log(s, "openfootball", f"{code} {temp}", url, "error", 0, err)
                    r["sin_datos"] += 1
                    continue
                anio = int(temp.split("-")[0])
                ms = _parse_json(txt) if tipo == "json" else _parse_txt(txt, anio)
                _log(s, "openfootball", f"{code} {temp}", url, "ok", len(ms))
                if not ms:
                    r["sin_datos"] += 1
                    continue
                r["temporadas_ok"] += 1
                for p in ms:
                    if not p["home"] or not p["away"]:
                        continue
                    mid = _mid(code, p["fecha"], p["home"], p["away"])
                    if mid in existentes:
                        continue
                    existentes.add(mid)
                    s.add(SoccerMatch(
                        match_id=mid, league_code=code, season=temp, stage=p["etapa"],
                        match_date=p["fecha"], kickoff_utc=p["kickoff"],
                        home_team=p["home"], away_team=p["away"],
                        home_goals=p["hg"], away_goals=p["ag"],
                        ht_home_goals=p["hthg"], ht_away_goals=p["htag"],
                        source="openfootball", source_url=url, fetched_at=ahora))
                    r["nuevos"] += 1
                if progress:
                    progress(f"{code} {temp}: {len(ms)} partidos")
                s.flush()
            resumen[code] = r
    return resumen
