"""Ingesta desde la MLB Stats API (statsapi.mlb.com).

Cubre: equipos, calendario, marcadores (incluidas las primeras 5 entradas),
pitchers probables/confirmados, alineaciones y estado de roster (lesiones).

Todo lo que entra queda registrado en `data_source_logs`. Ningun valor se
inventa: si la API no trae un campo, la columna queda en NULL.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from MLB.database.models import (Game, Injury, Lineup, ProbablePitcher, Team)
from MLB.ingest.http import get_json, log_fetch
from shared.timeutil import utcnow

BASE = "https://statsapi.mlb.com/api/v1"
SOURCE = "mlb_stats_api"
HYDRATE = "probablePitcher,linescore,venue,team,decisions"

# Estados que cuentan como partido jugado y utilizable para entrenar
FINAL_STATES = {"Final", "Game Over", "Completed Early"}


def _dt(s):
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ equipos
def ingest_teams(session, season: int) -> dict:
    url = f"{BASE}/teams?sportId=1&season={season}&hydrate=venue"
    data, f = get_json(url, "MLB", SOURCE, "schedule")
    log_fetch(session, f)
    if not data:
        return {"status": f.status, "error": f.error, "teams": 0}
    n = 0
    for t in data.get("teams", []):
        row = session.get(Team, t["id"]) or Team(id=t["id"])
        row.abbr = t.get("abbreviation")
        row.name = t.get("name")
        row.league = (t.get("league") or {}).get("nameShort") or (t.get("league") or {}).get("name")
        row.division = (t.get("division") or {}).get("name")
        row.venue_id = (t.get("venue") or {}).get("id")
        row.venue_name = (t.get("venue") or {}).get("name")
        session.merge(row)
        n += 1
    session.flush()
    return {"status": "ok", "teams": n}


# --------------------------------------------------------------- calendario
def ingest_schedule(session, season: int, start: str | None = None,
                    end: str | None = None) -> dict:
    """Descarga el calendario completo de una temporada (o un rango de fechas)."""
    a = start or f"{season}-02-20"
    b = end or f"{season}-11-15"
    url = (f"{BASE}/schedule?sportId=1&startDate={a}&endDate={b}"
           f"&gameType=R,F,D,L,W&hydrate={HYDRATE}")
    data, f = get_json(url, "MLB", SOURCE, "schedule", timeout=180)
    log_fetch(session, f)
    if not data:
        return {"status": f.status, "error": f.error, "games": 0}

    seen = {g.id for g in session.execute(select(Game.id)).all()} if False else None
    n_games = n_pp = 0
    now = utcnow()
    for day in data.get("dates", []):
        for g in day.get("games", []):
            pk = g["gamePk"]
            row = session.get(Game, pk) or Game(id=pk)
            ls = g.get("linescore") or {}
            innings = ls.get("innings") or []
            home, away = g["teams"]["home"], g["teams"]["away"]
            row.season = int(g.get("season") or season)
            row.game_date = g.get("officialDate") or day.get("date")
            row.start_utc = _dt(g.get("gameDate"))
            row.game_type = g.get("gameType")
            row.status = (g.get("status") or {}).get("detailedState")
            row.doubleheader = g.get("doubleHeader")
            row.game_number = g.get("gameNumber") or 1
            row.home_team_id = home["team"]["id"]
            row.away_team_id = away["team"]["id"]
            row.home_abbr = home["team"].get("abbreviation")
            row.away_abbr = away["team"].get("abbreviation")
            row.venue_id = (g.get("venue") or {}).get("id")
            row.venue_name = (g.get("venue") or {}).get("name")
            row.home_score = home.get("score")
            row.away_score = away.get("score")
            row.innings = len(innings) or None
            if row.status in FINAL_STATES and len(innings) >= 5:
                hf = [i["home"].get("runs") for i in innings[:5]]
                af = [i["away"].get("runs") for i in innings[:5]]
                # solo se calcula si TODAS las entradas traen carreras
                row.home_score_f5 = sum(hf) if all(v is not None for v in hf) else None
                row.away_score_f5 = sum(af) if all(v is not None for v in af) else None
            row.source = SOURCE
            row.retrieved_at = now
            session.merge(row)
            n_games += 1
            n_pp += _probables(session, pk, g, now)
    session.flush()
    return {"status": "ok", "games": n_games, "probable_pitchers": n_pp,
            "range": f"{a}..{b}", "url": url}


def _probables(session, game_pk: int, g: dict, now) -> int:
    """Guarda el pitcher probable como una fila nueva si cambio respecto a la ultima."""
    n = 0
    for side, is_home in (("home", True), ("away", False)):
        pp = (g["teams"][side] or {}).get("probablePitcher")
        if not pp:
            continue
        team_id = g["teams"][side]["team"]["id"]
        last = session.execute(
            select(ProbablePitcher).where(ProbablePitcher.game_id == game_pk,
                                          ProbablePitcher.team_id == team_id)
            .order_by(ProbablePitcher.available_at.desc()).limit(1)).scalars().first()
        if last and last.player_id == pp.get("id"):
            continue                                  # sin cambio: no se duplica
        session.add(ProbablePitcher(
            game_id=game_pk, team_id=team_id, is_home=is_home,
            player_id=pp.get("id"), player_name=pp.get("fullName"),
            state="probable", available_at=now, source=SOURCE))
        n += 1
    return n


# ------------------------------------------------------------- jugadores
def ingest_players(session, season: int) -> dict:
    """Catalogo de jugadores: nombre, posicion, mano de batear y de lanzar.

    Un unico endpoint masivo. Sin esto el dashboard solo puede mostrar numeros de
    identificacion en vez de nombres.
    """
    from MLB.database.models import Player
    url = f"{BASE}/sports/1/players?season={season}"
    data, f = get_json(url, "MLB", SOURCE, "players", timeout=120)
    log_fetch(session, f)
    if not data:
        return {"status": f.status, "error": f.error, "players": 0}
    now = utcnow()
    n = 0
    for p in data.get("people", []):
        row = session.get(Player, p["id"]) or Player(id=p["id"])
        row.full_name = p.get("fullName")
        row.primary_position = (p.get("primaryPosition") or {}).get("abbreviation")
        row.bats = (p.get("batSide") or {}).get("code")
        row.throws = (p.get("pitchHand") or {}).get("code")
        row.team_id = (p.get("currentTeam") or {}).get("id")
        row.updated_at = now
        session.merge(row)
        n += 1
    session.flush()
    return {"status": "ok", "players": n, "season": season}


# ------------------------------------------------------------- alineaciones
def ingest_lineups(session, game_pks: list[int]) -> dict:
    """Alineaciones desde el boxscore. `battingOrder` no vacio => confirmada."""
    now = utcnow()
    ok = err = added = 0
    for pk in game_pks:
        url = f"{BASE}/game/{pk}/boxscore"
        data, f = get_json(url, "MLB", SOURCE, "lineups", retries=2, timeout=45)
        log_fetch(session, f)
        if not data:
            err += 1
            continue
        ok += 1
        for side, is_home in (("home", True), ("away", False)):
            t = (data.get("teams") or {}).get(side) or {}
            order = [int(x) for x in (t.get("battingOrder") or [])]
            if not order:
                continue
            team_id = (t.get("team") or {}).get("id")
            last = session.execute(
                select(Lineup).where(Lineup.game_id == pk, Lineup.team_id == team_id)
                .order_by(Lineup.available_at.desc()).limit(1)).scalars().first()
            if last and list(last.batting_order or []) == order:
                continue
            session.add(Lineup(game_id=pk, team_id=team_id, is_home=is_home,
                               state="confirmed", batting_order=order,
                               available_at=now, source=SOURCE))
            added += 1
    session.flush()
    return {"status": "ok", "games_ok": ok, "games_error": err, "lineups_added": added}


# ----------------------------------------------------------------- lesiones
# Codigos de lista de lesionados de la MLB Stats API.
# OJO: "RM" (Reassigned to Minors) y "MIN" (Minor League Contract) NO son
# lesiones, son movimientos de roster. Estaban incluidos por error y por eso el
# recuento de "lesiones" salia inflado. Detectado al revisar los datos.
IL_CODES = {"D7", "D10", "D15", "D60", "DL", "IL", "BRV", "PL", "FME"}
IL_LABEL = {"D7": "IL 7 dias", "D10": "IL 10 dias", "D15": "IL 15 dias",
            "D60": "IL 60 dias", "DL": "lista de lesionados", "IL": "lista de lesionados",
            "BRV": "permiso por duelo", "PL": "permiso de paternidad",
            "FME": "emergencia familiar"}


def ingest_injuries(session, season: int) -> dict:
    """Estado de roster actual. La API NO expone historico de lesiones: solo el
    presente. Por eso las features historicas de lesion quedan marcadas como no
    disponibles en vez de reconstruirse a ojo."""
    teams = session.execute(select(Team.id)).scalars().all()
    if not teams:
        return {"status": "skipped", "reason": "no hay equipos ingeridos"}
    now, added, err = utcnow(), 0, 0
    # Limpieza de filas antiguas que no son lesiones (reasignaciones a menores,
    # RM/MIN) guardadas por una version anterior del ingestor.
    session.query(Injury).filter(Injury.status.notin_(list(IL_CODES))).delete(
        synchronize_session=False)
    for tid in teams:
        url = f"{BASE}/teams/{tid}/roster?rosterType=fullRoster&season={season}"
        data, f = get_json(url, "MLB", SOURCE, "injuries", retries=2, timeout=45)
        log_fetch(session, f)
        if not data:
            err += 1
            continue
        for p in data.get("roster", []):
            st = (p.get("status") or {})
            code = st.get("code")
            if code not in IL_CODES:
                continue
            session.add(Injury(team_id=tid, player_id=(p.get("person") or {}).get("id"),
                               player_name=(p.get("person") or {}).get("fullName"),
                               status=code, description=st.get("description"),
                               available_at=now, source=SOURCE))
            added += 1
    session.flush()
    return {"status": "ok", "teams": len(teams), "injuries": added, "errors": err,
            "caveat": "solo estado actual; la API no publica historico de lesiones"}


def games_needing_results(session, season: int) -> list[int]:
    return list(session.execute(
        select(Game.id).where(Game.season == season, Game.status.in_(FINAL_STATES),
                              Game.home_score.is_(None))).scalars().all())


def today_range(days_ahead: int = 2) -> tuple[str, str]:
    t = datetime.utcnow().date()
    return str(t - timedelta(days=1)), str(t + timedelta(days=days_ahead))
