"""AUDITORIA DE FUENTES (§2).

No basta con que el codigo diga que usa una fuente: aqui se vuelve a pedir el
dato a la API original y se compara con lo que hay guardado. Si no coincide, se
reporta la discrepancia.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB = Path(__file__).resolve().parents[1] / "MLB" / "database" / "mlb.sqlite3"
UA = {"User-Agent": "Sports-Prediction-Center-audit/1.0"}


def get_json(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def inventory() -> dict:
    """Lo que dice el registro de descargas, sin interpretarlo."""
    c = sqlite3.connect(DB)
    rows = c.execute("""
        select source, domain, status, count(*), min(retrieved_at), max(retrieved_at),
               sum(records), sum(case when checksum is not null then 1 else 0 end)
        from data_source_logs group by source, domain, status
        order by source, domain, status""").fetchall()
    out = []
    for s, d, st, n, a, b, rec, chk in rows:
        out.append({"source": s, "domain": d, "status": st, "calls": n,
                    "first": a, "last": b, "records": rec or 0, "with_checksum": chk})
    urls = c.execute("""select source, url from data_source_logs
                        group by source having count(*) > 0""").fetchall()
    hosts = defaultdict(set)
    for s, u in c.execute("select source, url from data_source_logs").fetchall():
        try:
            hosts[s].add(u.split("/")[2])
        except IndexError:
            hosts[s].add("(url mal formada)")
    c.close()
    return {"log": out, "hosts": {k: sorted(v) for k, v in hosts.items()},
            "sample_urls": dict(urls)}


def verify_schedule() -> dict:
    """Re-pide 3 dias de calendario y compara marcador y sede con lo guardado."""
    c = sqlite3.connect(DB)
    fechas = [r[0] for r in c.execute(
        "select distinct game_date from games where season in (2023,2024,2025) "
        "and status='Final' order by random() limit 3").fetchall()]
    res = {"checked_dates": fechas, "games": 0, "match": 0, "mismatch": []}
    for f in fechas:
        d = get_json("https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=" + f
                     + "&hydrate=linescore,venue")
        api = {}
        for day in d.get("dates", []):
            for g in day.get("games", []):
                api[g["gamePk"]] = {
                    "home": g["teams"]["home"].get("score"),
                    "away": g["teams"]["away"].get("score"),
                    "venue": (g.get("venue") or {}).get("id")}
        for pk, hs, as_, vid in c.execute(
                "select id, home_score, away_score, venue_id from games where game_date=?",
                (f,)).fetchall():
            if pk not in api:
                continue
            res["games"] += 1
            a = api[pk]
            if a["home"] == hs and a["away"] == as_ and a["venue"] == vid:
                res["match"] += 1
            else:
                res["mismatch"].append({"game": pk, "guardado": [hs, as_, vid],
                                        "api": [a["home"], a["away"], a["venue"]]})
    c.close()
    return res


def verify_boxscore() -> dict:
    """Re-pide 5 boxscores y compara linea de pitcheo (outs, K, BB, ER)."""
    c = sqlite3.connect(DB)
    pks = [r[0] for r in c.execute(
        "select distinct game_id from pitcher_game_logs order by random() limit 5").fetchall()]
    res = {"checked_games": pks, "lines": 0, "match": 0, "mismatch": []}
    for pk in pks:
        d = get_json(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
        api = {}
        for side in ("home", "away"):
            for _, p in ((d.get("teams") or {}).get(side, {}).get("players") or {}).items():
                st = (p.get("stats") or {}).get("pitching") or {}
                if st.get("inningsPitched") is None:
                    continue
                ip = str(st["inningsPitched"])
                w, _, fr = ip.partition(".")
                api[(p.get("person") or {}).get("id")] = {
                    "outs": int(w) * 3 + int(fr or 0), "k": st.get("strikeOuts"),
                    "bb": st.get("baseOnBalls"), "er": st.get("earnedRuns")}
        for pid, outs, k, bb, er in c.execute(
                "select player_id, outs, strikeouts, walks, earned_runs "
                "from pitcher_game_logs where game_id=?", (pk,)).fetchall():
            if pid not in api:
                continue
            res["lines"] += 1
            a = api[pid]
            if (a["outs"], a["k"], a["bb"], a["er"]) == (outs, k, bb, er):
                res["match"] += 1
            else:
                res["mismatch"].append({"game": pk, "player": pid,
                                        "guardado": [outs, k, bb, er],
                                        "api": [a["outs"], a["k"], a["bb"], a["er"]]})
    c.close()
    return res


def verify_statcast() -> dict:
    """Re-descarga un dia de Statcast y recalcula los agregados de 3 lanzadores."""
    import io

    import pandas as pd
    c = sqlite3.connect(DB)
    fecha = c.execute("select game_date from statcast_agg where season=2024 "
                      "order by random() limit 1").fetchone()[0]
    url = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfSea=2024%7C"
           f"&player_type=pitcher&game_date_gt={fecha}&game_date_lt={fecha}"
           "&type=details&min_pitches=0&min_results=0&group_by=name&sort_col=pitches"
           "&player_event_sort=api_p_release_speed&sort_order=desc")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    res = {"date": fecha, "pitches_downloaded": int(len(df)), "checked": 0, "match": 0,
           "mismatch": [], "truncated": bool(len(df) >= 25000)}
    g = df.groupby("pitcher").agg(pitches=("description", "size"),
                                  velo=("release_speed", "mean"))
    for pid, pitches, velo in c.execute(
            "select player_id, pitches, velocity from statcast_agg "
            "where scope='pitcher' and game_date=? limit 3", (fecha,)).fetchall():
        if pid not in g.index:
            continue
        res["checked"] += 1
        exp = g.loc[pid]
        ok = (int(exp.pitches) == pitches and
              (velo is None or abs(float(exp.velo) - velo) < 0.01))
        if ok:
            res["match"] += 1
        else:
            res["mismatch"].append({"player": pid, "guardado": [pitches, velo],
                                    "recalculado": [int(exp.pitches), round(float(exp.velo), 3)]})
    c.close()
    return res


def verify_no_secrets() -> dict:
    """Ninguna URL guardada puede contener una clave de API."""
    c = sqlite3.connect(DB)
    bad = c.execute("select count(*) from data_source_logs "
                    "where url like '%apiKey=%' or url like '%api_key=%'").fetchone()[0]
    c.close()
    return {"urls_con_clave": bad, "ok": bad == 0}


if __name__ == "__main__":
    out = {"inventario": inventory(), "calendario": verify_schedule(),
           "boxscores": verify_boxscore(), "statcast": verify_statcast(),
           "secretos": verify_no_secrets()}
    Path("audit/out").mkdir(exist_ok=True, parents=True)
    Path("audit/out/sources.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
