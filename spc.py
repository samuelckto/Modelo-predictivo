"""CLI del Sports Prediction Center.

    python spc.py serve          # levanta el dashboard unificado
    python spc.py status         # estado de los dos motores
    python spc.py init           # crea el esquema MLB
    python spc.py test           # ejecuta la bateria de pruebas
"""
from __future__ import annotations

import argparse
import json
from datetime import date
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared.paths  # noqa: E402,F401  -- carga el .env antes de cualquier comando


def cmd_init(_):
    from MLB.database.session import init_db, table_names
    print("base MLB:", init_db())
    print(len(table_names()), "tablas:", ", ".join(table_names()))


def cmd_status(_):
    import NFL.adapter as nfl
    from MLB.engine.provider import status
    ok, msg = nfl.available()
    print(f"NFL  ready={ok}  {msg}")
    if ok:
        print("     aislamiento:", nfl.read_only_check())
    print(json.dumps(status(), indent=1, default=str))


def _spc_running(host, port) -> bool:
    """True si ya hay un Sports Prediction Center respondiendo en ese puerto."""
    import urllib.request
    check_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    try:
        with urllib.request.urlopen(f"http://{check_host}:{port}/api/status", timeout=3) as r:
            return b"Sports Prediction Center" in r.read() or r.status == 200
    except Exception:
        return False


def cmd_serve(a):
    import uvicorn
    host = getattr(a, "host", None) or os.getenv("SPC_API_HOST", "0.0.0.0")
    port = a.port or int(os.getenv("SPC_API_PORT", "8100"))
    if _spc_running(host, port):
        if not a.restart:
            print(f"El servidor YA esta corriendo en el puerto {port} (quiza en segundo plano).\n"
                  f"Para reiniciarlo con el codigo nuevo: python spc.py serve --restart")
            return
        _kill_port(port)
    print(f"Iniciando Sports Prediction Center en http://{host}:{port}...")
    uvicorn.run("dashboard.backend.main:app", host=host, port=port, reload=False)


def _kill_port(port: int):
    import subprocess, sys, time
    if sys.platform.startswith("win"):
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Get-NetTCPConnection -LocalPort {port} -State Listen "
                        f"-ErrorAction SilentlyContinue | ForEach-Object "
                        f"{{ Stop-Process -Id $_.OwningProcess -Force }}"],
                       capture_output=True)
    else:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
    time.sleep(2)
    print(f"servidor anterior detenido; arrancando de nuevo en el puerto {port}")


def cmd_mlb_daily(a):
    from MLB.engine.daily import run
    print(json.dumps(run(a.days_ahead, a.days_back, a.skip_statcast), indent=1, default=str))


def cmd_mlb_predict(a):
    from datetime import date, timedelta
    from MLB.database.session import session_scope
    from MLB.engine.predict import predict_range
    t = date.today()
    with session_scope() as s:
        print(json.dumps(predict_range(s, a.start or str(t),
                                       a.end or str(t + timedelta(days=7)),
                                       reason=a.reason), indent=1, default=str))


def cmd_odds_snapshot(_):
    """Toma un snapshot de cuotas. Pensado para ejecutarse varias veces al dia
    (09:00, 12:00, 15:00, 18:00 y antes del primer partido) y asi construir
    nuestro propio historico de mercado."""
    from MLB.database.session import session_scope
    from MLB.ingest.odds import ingest, mark_closing, snapshot_stats
    with session_scope() as s:
        r = ingest(s)
        r["cierre"] = mark_closing(s)
        r["historico"] = snapshot_stats(s)
    print(json.dumps(r, indent=1, default=str))


def cmd_score(a):
    """Califica las predicciones de los partidos que ya terminaron. Rapido (~5 s):
    solo baja los marcadores finales y marca acierto/fallo. No reconstruye
    features ni reentrena. El servidor (`serve`) hace esto solo cada 10 min."""
    from MLB.engine.daily import refresh_results
    r = refresh_results(a.days_back)
    try:
        from NFL.markets.pipeline import refresh_results as nfl_refresh
        print(f"NFL total/spread calificadas ahora: {nfl_refresh().get('scored')}")
    except Exception as e:
        print(f"NFL total/spread: no se pudo calificar ({type(e).__name__}: {e})")
    try:
        from NBA.markets.pipeline import refresh_results as nba_refresh
        print(f"NBA calificadas ahora: {nba_refresh().get('scored')}")
    except Exception as e:
        print(f"NBA: no se pudo calificar ({type(e).__name__}: {e})")
    try:
        from TENIS.markets.pipeline import refresh_results as ten_refresh
        print(f"TENIS calificadas ahora: {ten_refresh().get('scored')}")
    except Exception as e:
        print(f"TENIS: no se pudo calificar ({type(e).__name__}: {e})")
    try:
        from SOCCER.markets.pipeline import score as soccer_score
        print(f"SOCCER calificadas ahora: {soccer_score().get('calificadas')}")
    except Exception as e:
        print(f"SOCCER: no se pudo calificar ({type(e).__name__}: {e})")
    try:
        from PARLAY.engine.scoring import score as score_parlays
        print(f"combinadas: {score_parlays()}")
    except Exception as e:
        print(f"combinadas: no se pudo calificar ({type(e).__name__}: {e})")
    print(f"marcadores {r['desde']} a {r['hasta']}: {r['status']} - {r['partidos']} partidos")
    print(f"predicciones calificadas ahora: {r['calificadas']}")
    from MLB.engine.history import history
    h = history()
    print(f"historico: {h['total']['aciertos']} aciertos - {h['total']['fallos']} fallos "
          f"- {h['total']['n']} evaluadas")


def cmd_nfl_train(a):
    """Investigacion walk-forward + entrenamiento de NFL_TOTAL / NFL_SPREAD (solo lee el motor NFL)."""
    import subprocess, sys
    from shared.paths import ROOT
    if not a.skip_research:
        for scr in ("NFL/markets/research.py", "NFL/markets/research_lines.py"):
            print(f"== {scr} ==")
            subprocess.run([sys.executable, str(ROOT / scr)], check=True)
    from NFL.markets.train import main
    main()


def cmd_nfl_cycle(a):
    """Cuotas NFL + predicciones de total/spread + calificacion. Programar a diario."""
    from NFL.markets.pipeline import run
    print(json.dumps(run(a.days_ahead, not a.skip_odds), indent=1, default=str))


def cmd_nba_train(a):
    """Datos NBA (pbpstats) + features + walk-forward + calibracion + modelos. Reanudable."""
    import subprocess, sys
    from shared.paths import ROOT
    from NBA.markets.ingest import run as ingest
    from NBA.markets.pipeline import current_season
    if not a.skip_ingest:
        seasons = list(range(a.from_season, current_season() + 1))
        while True:
            r = ingest(seasons, budget_s=a.budget, progress=print)
            print(r)
            if r["pending"] == 0:
                break
    from NBA.markets.features import build
    build(progress=print)
    if not a.skip_research:
        for scr in ("NBA/markets/research.py", "NBA/markets/research_lines.py"):
            print(f"== {scr} ==")
            subprocess.run([sys.executable, str(ROOT / scr)], check=True)
    from NBA.markets.train import main
    main()


def cmd_nba_cycle(a):
    """Partidos/logs nuevos + cuotas NBA + predicciones + calificacion. Programar a diario."""
    from NBA.markets.pipeline import run
    print(json.dumps(run(a.days_ahead, not a.skip_odds), indent=1, default=str))


def cmd_tenis_train(a):
    """Archivo ATP/WTA + features + walk-forward + calibracion + modelos."""
    import subprocess, sys
    from shared.paths import ROOT
    from TENIS.markets.ingest import run as ingest
    if not a.skip_ingest:
        seasons = list(range(a.from_season, date.today().year + 1)) if False else list(range(a.from_season, 2027))
        while True:
            r = ingest(seasons, budget_s=a.budget, progress=print)
            print(r)
            if r["pending"] == 0:
                break
    from TENIS.markets.features import build
    build(progress=print)
    if not a.skip_research:
        for scr in ("TENIS/markets/research.py", "TENIS/markets/research_lines.py"):
            for tour in ("ATP", "WTA"):
                print(f"== {scr} {tour} ==")
                subprocess.run([sys.executable, str(ROOT / scr), "--tour", tour], check=True)
    from TENIS.markets.train import main
    main()


def cmd_tenis_cycle(a):
    """Calendario y cuotas de tenis + predicciones + calificacion. Programar a diario."""
    from TENIS.markets.pipeline import run
    print(json.dumps(run(a.days_ahead, not a.skip_odds), indent=1, default=str))


def cmd_soccer_train(a):
    """Futbol: historico + corners + features + auditoria + modelos. Tarda varios minutos."""
    from SOCCER.markets.pipeline import train
    print(json.dumps(train(desde=a.from_season, progress=lambda m: print("  ->", m)),
                     indent=1, default=str))


def cmd_soccer_cycle(a):
    """Futbol: calendario, cuotas, resultados y predicciones ACTUALES. Programar a diario."""
    from SOCCER.markets.pipeline import cycle
    print(json.dumps(cycle(a.days_ahead, progress=lambda m: print("  ->", m)),
                     indent=1, default=str))


def cmd_soccer_gating(a):
    """Recalcula que mercado publica cada liga, con los numeros del walk-forward."""
    from SOCCER.markets.pipeline import refrescar_gating
    print(json.dumps(refrescar_gating(), indent=1, default=str))


def cmd_parlays(a):
    """Genera las combinadas de mayor probabilidad y califica las anteriores."""
    from PARLAY.engine.builder import generate
    from PARLAY.engine.scoring import history, score
    print(json.dumps(generate(a.start, a.end, sizes=tuple(a.sizes)), indent=1, default=str))
    print(json.dumps(score(), indent=1, default=str))
    h = history()
    print(f"combinadas abiertas: {len(h['abiertas'])} | calificadas: {h['total']['n']} "
          f"({h['total']['aciertos']} aciertos)")


def cmd_repair(a):
    """Recupera lo que una ejecucion con fallos de red no descargo: calendario
    con abridores anunciados, alineaciones y cuotas de los proximos dias. Va
    despacio y con muchos reintentos a proposito; despues reconstruye features
    y predicciones."""
    from datetime import date, timedelta
    from sqlalchemy import select
    from MLB.database.models import Game
    from MLB.database.session import init_db, session_scope
    from MLB.ingest.statsapi import FINAL_STATES, ingest_lineups, ingest_schedule
    from MLB.ingest.odds import ingest as ingest_odds
    init_db()
    hoy = date.today()
    season = hoy.year
    x, y = str(hoy - timedelta(days=1)), str(hoy + timedelta(days=a.days))
    print(f"=== REPARACION {x} → {y} ===")
    with session_scope() as s:
        for intento in range(1, 4):
            r = ingest_schedule(s, season, x, y)
            print(f"  calendario (intento {intento}): {r.get('status')} "
                  f"partidos={r.get('games')} abridores={r.get('probable_pitchers')}")
            if r.get("status") == "ok":
                break
        pks = list(s.execute(select(Game.id).where(
            Game.game_date >= str(hoy), Game.game_date <= y,
            ~Game.status.in_(FINAL_STATES))).scalars().all())
        r = ingest_lineups(s, pks[:60])
        print(f"  alineaciones: {r}")
        r = ingest_odds(s)
        print(f"  cuotas: {r.get('status')} filas={r.get('rows', 0)}")
    from sqlalchemy import func
    from MLB.database.models import ProbablePitcher
    with session_scope() as s:
        n = s.scalar(select(func.count(func.distinct(ProbablePitcher.game_id)))
                     .where(ProbablePitcher.game_id.in_(
                         select(Game.id).where(Game.game_date >= str(hoy),
                                               Game.game_date <= y))))
        tot = s.scalar(select(func.count()).select_from(Game).where(
            Game.game_date >= str(hoy), Game.game_date <= y))
    print(f"  partidos con abridor anunciado: {n}/{tot}")
    print("\n=== FEATURES Y PREDICCIONES ===")
    from MLB.features.builder import build
    X = build(sorted({season - 5, season - 4, season - 3, season - 2, season - 1, season}))
    print(f"  features: {X.shape}")
    from MLB.engine.predict import predict_range
    with session_scope() as s:
        r = predict_range(s, str(hoy), y, reason="reparacion")
    print(f"  predicciones: {r.get('predictions')} nuevas, {r.get('revisions')} revisadas")
    print("\nListo. Recarga el dashboard con Ctrl+Shift+R.")


def cmd_doctor(_):
    """Comprueba conectividad, credenciales y estado de los datos.

    Es lo primero que hay que ejecutar cuando algo no carga: distingue un
    problema de red de un problema del sistema.
    """
    import socket
    import urllib.request

    print("=== CONECTIVIDAD ===")
    hosts = [("statsapi.mlb.com", "MLB Stats API"),
             ("baseballsavant.mlb.com", "Baseball Savant"),
             ("api.the-odds-api.com", "The Odds API")]
    red_ok = True
    for host, nombre in hosts:
        try:
            socket.gethostbyname(host)
            dns = "DNS ok"
        except OSError as e:
            dns = f"DNS FALLA ({e})"
            red_ok = False
            print(f"  {nombre:<20}{dns}")
            continue
        try:
            req = urllib.request.Request(f"https://{host}/",
                                         headers={"User-Agent": "spc-doctor/1.0"})
            urllib.request.urlopen(req, timeout=15)
            print(f"  {nombre:<20}{dns} · responde")
        except Exception as e:                              # noqa: BLE001
            code = getattr(e, "code", None)
            if code:                                        # responde aunque de 4xx
                print(f"  {nombre:<20}{dns} · responde (HTTP {code})")
            else:
                red_ok = False
                print(f"  {nombre:<20}{dns} · NO RESPONDE ({type(e).__name__})")

    print("\n=== CREDENCIALES ===")
    from MLB.ingest.odds import api_key
    print(f"  THE_ODDS_API_KEY     {'configurada' if api_key() else 'NO configurada'}")

    print("\n=== DATOS ===")
    from dashboard.backend.performance import data_health
    h = data_health()
    r = h.get("resumen_mlb", {})
    print(f"  boxscores            {r.get('boxscores')} "
          f"{'ok' if r.get('boxscores_ok') else 'INCOMPLETO'}")
    print(f"  dias sin statcast    {r.get('n_dias_faltantes')}")
    d = h.get("diagnostico") or {}
    if d.get("problema"):
        print(f"\n  PROBLEMA DETECTADO: {d['problema']}")
        print(f"  {d.get('detalle')}")
        if d.get("consejo"):
            print(f"  -> {d['consejo']}")
    else:
        print(f"\n  {d.get('detalle', 'sin problemas')}")

    print("\n=== MODELOS ===")
    from MLB.engine.models import UNAVAILABLE, _algos
    a = _algos()
    print(f"  algoritmos usables   {sorted(a)}")
    if UNAVAILABLE:
        for k, v in UNAVAILABLE.items():
            print(f"  DESCARTADO {k}: {v[:120]}")
    print("\n=== VEREDICTO ===")
    if not red_ok:
        print("  RED: hay fuentes que no responden AHORA. Nada funcionara hasta arreglarlo.")
    elif d.get("problema"):
        print("  RED: ahora responde. Los fallos fueron pasados y la cobertura esta "
              "completa; vuelve a ejecutar `python spc.py mlb-daily` para recuperar lo "
              "que aquella ejecucion no descargo (abridores, alineaciones).")
    else:
        print("  RED: en orden.")
    if not api_key():
        print("  CUOTAS: sin clave -> las predicciones salen sin mercado. Revisa .env.")
    if UNAVAILABLE:
        print(f"  MODELOS: {', '.join(UNAVAILABLE)} descartado(s); el ensemble sigue con "
              f"{len(a)} algoritmos. Moneyline (Elo) no se ve afectado.")
    if not r.get("boxscores_ok") or r.get("n_dias_faltantes"):
        print("  DATOS: hay huecos, revisa la tabla de Data health.")
    else:
        print("  DATOS: cobertura completa.")


def cmd_markets(_):
    from MLB.engine.gating import decide
    d = decide()
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        print(f"{k:<14}{'HABILITADO' if v.get('enabled') else 'BLOQUEADO':<12}{v.get('reason','')}")


def cmd_test(_):
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))


def cmd_value(a):
    """Capa de valor: cobertura, favoritos >60 % y backtest de discrepancias."""
    import json

    from shared import ledger, tracking
    sport = None if getattr(a, "sport", "all") == "all" else a.sport.upper()
    cob = ledger.cobertura()
    print("== COBERTURA ==")
    for s, v in cob["deportes"].items():
        print(f"  {s:7s} predicciones={v['predicciones']:5d} calificadas={v['calificadas']:4d} "
              f"con_cuota={v['con_cuota']:5d} calificadas_con_cuota={v['calificadas_con_cuota']:4d}")
    for e in cob["errores_de_consulta"]:
        print(f"  ! consulta fallida: {e}")

    f = tracking.favoritos(sport)
    print(f"\n== FAVORITOS >60 % == estado: {f['estado']}")
    print(f"  {f['n_por_encima_del_umbral']} de {f['n_predicciones_totales']} "
          f"predicciones calificadas superaron el 60 %")
    for b in f["buckets"]:
        print(f"  {b['bucket']:6s} n={b['n']:4d}  W-L-P {b['wins']}-{b['losses']}-{b['pushes']}"
              f"  acierto={b['win_rate']}  anunciada={b['prob_media_anunciada']}"
              f"  apostables={b['n_apostables']}  ROI={b['roi']}")
    if f.get("nota"):
        print(f"  NOTA: {f['nota']}")

    d = tracking.discrepancias(sport)
    print(f"\n== DISCREPANCIAS == estado: {d['estado']}")
    for b in d["buckets"]:
        print(f"  {b['bucket']:9s} n={b['n']:4d}  acierto={b['win_rate']}  ROI={b['roi']}"
              f"  IC95={b['roi_ic95']}  muestra_suficiente={b['suficiente']}")
    print(f"  VEREDICTO: {d['veredicto']['conclusion']}")
    if getattr(a, "json", False):
        print(json.dumps({"cobertura": cob, "favoritos": f, "discrepancias": d},
                         indent=1, default=str))


def cmd_chat(a):
    """Pregunta al chat desde la terminal."""
    from CHAT import answer
    r = answer.responder(a.texto, getattr(a, "game", None), getattr(a, "sport", None))
    print(r.get("respuesta", ""))
    for k in ("motivo", "aviso", "advertencia"):
        if r.get(k):
            print(f"\n{k}: {r[k]}")
    if r.get("status"):
        print(f"\nestado: {r['status']}  modelo: {r.get('model_version') or '-'}")


def cmd_soccer_cards(a):
    """Walk-forward del mercado de tarjetas."""
    from SOCCER.evaluation import wf_cards
    res = wf_cards.ejecutar(progress=print)
    for bloque in ("seleccion", "holdout"):
        print(f"\n== {bloque.upper()} ==")
        print(wf_cards.resumen(res, bloque).to_string(index=False))


def cmd_mlb_runs(a):
    """Walk-forward de la distribucion de carreras (completo y F5)."""
    from MLB.backtests import wf_runs
    res = wf_runs.ejecutar(progress=print)
    for tramo in ("completo", "f5"):
        for bloque in ("seleccion", "holdout"):
            print(f"\n== {tramo.upper()} / {bloque.upper()} ==")
            print(wf_runs.resumen(res, tramo, bloque).to_string(index=False))


def main():
    p = argparse.ArgumentParser(prog="spc")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("serve")
    s.add_argument("--host", default=None, help="host donde escuchar (por defecto 0.0.0.0 o SPC_API_HOST)")
    s.add_argument("--port", type=int, help="puerto donde escuchar (por defecto 8100 o SPC_API_PORT)")
    s.add_argument("--restart", action="store_true", help="detiene el servidor que ya corre y arranca otro")
    s.set_defaults(fn=cmd_serve)
    d = sub.add_parser("mlb-daily"); d.add_argument("--days-ahead", type=int, default=7)
    d.add_argument("--days-back", type=int, default=3)
    d.add_argument("--skip-statcast", action="store_true"); d.set_defaults(fn=cmd_mlb_daily)
    q = sub.add_parser("mlb-predict"); q.add_argument("--start"); q.add_argument("--end")
    q.add_argument("--reason", default="manual"); q.set_defaults(fn=cmd_mlb_predict)
    sub.add_parser("markets").set_defaults(fn=cmd_markets)
    sub.add_parser("odds-snapshot").set_defaults(fn=cmd_odds_snapshot)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    rp = sub.add_parser("repair"); rp.add_argument("--days", type=int, default=4)
    rp.set_defaults(fn=cmd_repair)
    sc = sub.add_parser("score"); sc.add_argument("--days-back", type=int, default=1)
    sc.set_defaults(fn=cmd_score)
    nt = sub.add_parser("nfl-train"); nt.add_argument("--skip-research", action="store_true")
    nt.set_defaults(fn=cmd_nfl_train)
    nc = sub.add_parser("nfl-cycle"); nc.add_argument("--days-ahead", type=int, default=10)
    nc.add_argument("--skip-odds", action="store_true"); nc.set_defaults(fn=cmd_nfl_cycle)
    bt = sub.add_parser("nba-train"); bt.add_argument("--from-season", type=int, default=2016)
    bt.add_argument("--budget", type=float, default=600); bt.add_argument("--skip-ingest", action="store_true")
    bt.add_argument("--skip-research", action="store_true"); bt.set_defaults(fn=cmd_nba_train)
    bc = sub.add_parser("nba-cycle"); bc.add_argument("--days-ahead", type=int, default=3)
    bc.add_argument("--skip-odds", action="store_true"); bc.set_defaults(fn=cmd_nba_cycle)
    tt = sub.add_parser("tenis-train"); tt.add_argument("--from-season", type=int, default=2000)
    tt.add_argument("--budget", type=float, default=600); tt.add_argument("--skip-ingest", action="store_true")
    tt.add_argument("--skip-research", action="store_true"); tt.set_defaults(fn=cmd_tenis_train)
    tc = sub.add_parser("tenis-cycle"); tc.add_argument("--days-ahead", type=int, default=3)
    tc.add_argument("--skip-odds", action="store_true"); tc.set_defaults(fn=cmd_tenis_cycle)
    st = sub.add_parser("soccer-train"); st.add_argument("--from-season", type=int, default=2012)
    st.set_defaults(fn=cmd_soccer_train)
    scy = sub.add_parser("soccer-cycle"); scy.add_argument("--days-ahead", type=int, default=8)
    scy.set_defaults(fn=cmd_soccer_cycle)
    sub.add_parser("soccer-gating").set_defaults(fn=cmd_soccer_gating)

    v = sub.add_parser("value"); v.add_argument("--sport", default="all")
    v.add_argument("--json", action="store_true"); v.set_defaults(fn=cmd_value)
    ch = sub.add_parser("chat"); ch.add_argument("texto")
    ch.add_argument("--game"); ch.add_argument("--sport"); ch.set_defaults(fn=cmd_chat)
    sub.add_parser("soccer-cards").set_defaults(fn=cmd_soccer_cards)
    sub.add_parser("mlb-runs").set_defaults(fn=cmd_mlb_runs)

    pl = sub.add_parser("parlays"); pl.add_argument("--start"); pl.add_argument("--end")
    pl.add_argument("--sizes", nargs="*", type=int, default=[2, 3]); pl.set_defaults(fn=cmd_parlays)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
