"""API unica del Sports Prediction Center.

El dashboard SOLO LEE. No entrena, no ingiere y no escribe en ninguna base.
Cada deporte responde por si mismo; si un motor no tiene datos lo dice.
"""
from __future__ import annotations

import os

from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import NFL.adapter as nfl
from NFL.markets import provider as nflm
from NBA.markets import provider as nbam
from TENIS.markets import provider as tenm
from SOCCER.markets import provider as socm
from MLB.engine import provider as mlb
from dashboard.backend import performance as perf
from shared.paths import DASHBOARD_DIR, NFL_HOME
from shared import valor
import math
from shared.schema import norm_ts, validate_card, empty_card, risk_label
from shared.timeutil import utcnow

app = FastAPI(title="Sports Prediction Center", version="0.1.0")

# Auto-scoring: el servidor califica solo las predicciones de partidos terminados.
# Es la unica escritura que hace el proceso del servidor; el dashboard (UI) sigue
# siendo de solo lectura. Desactivar con SPC_AUTO_SCORE=0.
AUTO_SCORE_MIN = int(os.getenv("SPC_AUTO_SCORE_MIN", "10"))
AUTO_SCORE = {"ultimo": None, "resultado": None, "error": None}


def _auto_score_loop():
    import time
    from MLB.engine.daily import refresh_results
    time.sleep(5)
    while True:
        try:
            AUTO_SCORE["resultado"] = refresh_results(1)
            try:
                from NFL.markets.pipeline import refresh_results as nfl_refresh
                AUTO_SCORE["resultado"]["nfl_markets"] = nfl_refresh()
            except Exception as e:
                AUTO_SCORE["resultado"]["nfl_markets"] = f"{type(e).__name__}: {e}"
            try:
                from NBA.markets.pipeline import refresh_results as nba_refresh
                AUTO_SCORE["resultado"]["nba"] = nba_refresh()
            except Exception as e:
                AUTO_SCORE["resultado"]["nba"] = f"{type(e).__name__}: {e}"
            try:
                from TENIS.markets.pipeline import refresh_results as ten_refresh
                AUTO_SCORE["resultado"]["tenis"] = ten_refresh()
            except Exception as e:
                AUTO_SCORE["resultado"]["tenis"] = f"{type(e).__name__}: {e}"
            # SOCCER faltaba entero en este bucle: 69 predicciones de partidos ya
            # jugados llevaban dias sin calificar y en el dashboard salian como
            # pendientes para siempre.
            #
            # Ademas NO se usa `live.results()` aqui: eso pega a the-odds-api,
            # cuya cuota (500/mes) esta agotada y devuelve 401. El marcador se
            # cierra con la fuente GRATUITA de openfootball, que ya se usa para
            # el historico y no tiene cuota. Va unos dias por detras, asi que un
            # partido de ayer puede tardar en cerrarse; eso se ve en el estado.
            try:
                from SOCCER.ingestion import cerrar as soc_cerrar, recent as soc_recent
                from SOCCER.markets.pipeline import score as soc_score
                AUTO_SCORE["resultado"]["soccer"] = {
                    "historico": soc_recent.ingest(desde=2025),
                    "fixtures_cerrados": soc_cerrar.cerrar(),
                    "calificacion": soc_score(),
                }
            except Exception as e:
                AUTO_SCORE["resultado"]["soccer"] = f"{type(e).__name__}: {e}"
            try:
                from PARLAY.engine.builder import generate as gen_parlays
                from PARLAY.engine.scoring import score as score_parlays
                AUTO_SCORE["resultado"]["parlays"] = {**gen_parlays(), **score_parlays()}
            except Exception as e:
                AUTO_SCORE["resultado"]["parlays"] = f"{type(e).__name__}: {e}"
            # Factor contextual: clima y cambios de lineup/pitcher
            try:
                from shared.contextual import refresh_contextual
                AUTO_SCORE["resultado"]["contextual"] = refresh_contextual()
            except Exception as e:
                AUTO_SCORE["resultado"]["contextual"] = f"{type(e).__name__}: {e}"
            AUTO_SCORE["error"] = None
        except Exception as e:  # nunca tumba el servidor
            AUTO_SCORE["error"] = f"{type(e).__name__}: {e}"
        AUTO_SCORE["ultimo"] = utcnow().isoformat()
        time.sleep(AUTO_SCORE_MIN * 60)


@app.on_event("startup")
def _start_auto_score():
    if os.getenv("SPC_AUTO_SCORE", "1") != "0":
        import threading
        threading.Thread(target=_auto_score_loop, daemon=True, name="auto-score").start()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPORTS = ("NFL", "MLB", "NBA", "TENIS", "SOCCER")


def _range(preset: str, start: str | None, end: str | None) -> tuple[str, str]:
    today = date.today()
    if preset == "today":
        return str(today), str(today)
    if preset == "tomorrow":
        d = today + timedelta(days=1)
        return str(d), str(d)
    if preset == "7d":
        return str(today), str(today + timedelta(days=7))
    if preset == "30d":
        return str(today), str(today + timedelta(days=30))
    if preset == "season":
        return str(today - timedelta(days=365)), str(today + timedelta(days=365))
    return (start or str(today)), (end or str(today + timedelta(days=7)))


def _ensure_nfl_winner(cards: list[dict]) -> list[dict]:
    """Garantiza que todo partido de NFL tenga una tarjeta de Ganador (moneyline).
    
    Si el motor legacy solo tiene total/spread o no genero pick de moneyline,
    se deduce el ganador a partir del margen proyectado del modelo de spread.
    """
    out = list(cards)
    ml_by_game = {c["game_id"]: c for c in out if c.get("market") == "moneyline" and c.get("selection")}
    spread_by_game = {c["game_id"]: c for c in out if (c.get("market") or "").lower() == "spread"}

    for gid, sp in spread_by_game.items():
        if gid in ml_by_game:
            continue
        ex = sp.get("extra") or {}
        margin = ex.get("expected_value")
        if margin is None:
            margin = 0.0
        std = ex.get("expected_std") or 13.5
        std = float(std) if std and float(std) > 0 else 13.5
        z = abs(float(margin)) / std
        prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
        prob = max(0.51, min(0.95, prob))

        if float(margin) >= 0:
            winner = sp.get("home")
            winner_name = sp.get("home_name") or winner
        else:
            winner = sp.get("away")
            winner_name = sp.get("away_name") or winner

        ml_card = empty_card(
            sport="NFL",
            game_id=sp.get("game_id"),
            game_date=sp.get("game_date"),
            start_utc=sp.get("start_utc"),
            status=sp.get("status", "scheduled"),
            home=sp.get("home"),
            away=sp.get("away"),
            home_name=sp.get("home_name"),
            away_name=sp.get("away_name"),
            venue=sp.get("venue"),
            market="moneyline",
            selection=winner,
            line=None,
            model_probability=round(prob, 4),
            ensemble_probability=round(prob, 4),
            market_probability=sp.get("market_probability"),
            confidence=("ALTA" if prob >= 0.62 else "MEDIA" if prob >= 0.545 else "BAJA"),
            upset_risk=round((1.0 - prob) * 100, 1),
            upset_label=risk_label(round((1.0 - prob) * 100, 1)),
            model_version="spread-margin v1",
            result=sp.get("result"),
            correct=sp.get("correct"),
        )
        if sp.get("status") == "final" and sp.get("actual_value") is not None:
            act_margin = float(sp["actual_value"])
            real_winner = sp.get("home") if act_margin > 0 else sp.get("away") if act_margin < 0 else "TIE"
            ml_card["result"] = "win" if winner == real_winner else "loss"
            ml_card["correct"] = bool(winner == real_winner)
            ml_card["actual_value"] = act_margin

        ml_card["extra"] = {
            "season": ex.get("season"),
            "week": ex.get("week"),
            "margin_predicted": round(float(margin), 2),
            "winner_name": winner_name,
            "publish_pick": True,
            "mode": "proyeccion",
            "factors": ex.get("factors", []),
            "derived_from": "spread_margin",
        }
        out.append(ml_card)
        ml_by_game[gid] = ml_card

    return out


def _collect(sport: str, a: str, b: str) -> tuple[list[dict], list[str]]:
    cards, warns = [], []
    for s in ([sport] if sport in SPORTS else SPORTS):
        try:
            got = (nfl.games(a, b) if s == "NFL" else mlb.games(a, b) if s == "MLB"
                   else nbam.games(a, b) if s == "NBA" else socm.games(a, b) if s == "SOCCER"
                   else tenm.games(a, b))
        except Exception as e:                     # nunca inventar: se reporta
            warns.append(f"{s}: el motor no respondio ({type(e).__name__}: {e})")
            continue
        if s == "NFL":
            try:
                got = list(got) + nflm.games(a, b)
            except Exception as e:
                warns.append(f"NFL total/spread: no disponible ({type(e).__name__}: {e})")
            got = _ensure_nfl_winner(got)
        bad = [p for c in got for p in validate_card(c)]
        if bad:
            warns.append(f"{s}: {len(bad)} campos invalidos descartados")
            got = [c for c in got if not validate_card(c)]
        cards.extend(got)
    for c in cards:
        _con_valor(c)
    cards.sort(key=lambda c: (norm_ts(c.get("start_utc")), c.get("sport") or "",
                              str(c.get("game_id") or ""), str(c.get("market") or "")))
    return cards, warns


def _con_valor(c: dict) -> dict:
    """Anade el desacuerdo con el mercado y el EV estimado a una tarjeta.

    Dos cosas distintas, y por eso van por separado:

      · `contra_mercado` solo necesita la probabilidad implicita, que existe en
        todos los deportes. Es un hecho: el modelo y la casa apuntan a lados
        opuestos.
      · `ev` necesita el PRECIO REAL. No se deriva de `market_probability`
        porque esa cifra ya viene con el margen quitado (devig proporcional en
        la ingesta), asi que 1/p daria un pago mayor que el que la casa paga de
        verdad e inflaria el valor de todo. Sin precio registrado no hay EV.
    """
    ex = c.get("extra") or {}
    precio = ex.get("odds_price")
    c["valor"] = valor.evaluar(c.get("ensemble_probability"),
                               c.get("market_probability"), precio)
    return c


@app.get("/api/status")
def status():
    ok, msg = nfl.available()
    return {
        "generated_at": utcnow(),
        "auto_score": AUTO_SCORE,
        "sports": {
            "NFL": {"engine": "nfl-prediction-app (envuelto, solo lectura)",
                    "home": str(NFL_HOME), "ready": ok, "detail": msg,
                    "read_only_proof": nfl.read_only_check()},
            "MLB": mlb.status(),
        },
    }


@app.get("/api/games")
def games(sport: str = Query("all"), range: str = Query("7d"),
          start: str | None = None, end: str | None = None):
    a, b = _range(range, start, end)
    cards, warns = _collect(sport.upper(), a, b)
    days: dict[str, dict] = {}
    for c in cards:
        k = (c.get("game_date") or str(c.get("start_utc") or "")[:10])
        days.setdefault(k, {"date": k, "games": []})["games"].append(c)
    return {"from": a, "to": b, "sport": sport.upper(), "count": len(cards),
            "days": [days[k] for k in sorted(days)], "warnings": warns}


@app.get("/api/scoring-status")
def scoring_status():
    """Que queda sin calificar en cada deporte, y por que.

    Antes esto no se veia: el auto-scoring decia `calificadas: 0` en todos los
    deportes y era imposible distinguir "no hay nada que calificar" de "esto
    lleva dias roto". Son cosas muy distintas y la segunda hay que verla.
    """
    from audit.diag_calificacion import resumen
    out = resumen()
    out["auto_score"] = {"ultimo": AUTO_SCORE["ultimo"], "error": AUTO_SCORE["error"]}
    out["fuentes"] = _estado_fuentes()
    return out


def _estado_fuentes() -> list[dict]:
    """Salud de las fuentes de resultados. La cuota agotada explica la mitad de
    lo que queda pendiente, asi que tiene que verse en pantalla."""
    fuentes = []
    try:
        import os
        import urllib.request
        k = os.getenv("THE_ODDS_API_KEY")
        if not k:
            fuentes.append({"fuente": "the-odds-api", "ok": False,
                            "detalle": "sin clave configurada"})
        else:
            with urllib.request.urlopen(
                    f"https://api.the-odds-api.com/v4/sports?apiKey={k}", timeout=10) as r:
                queda = r.headers.get("x-requests-remaining")
                fuentes.append({
                    "fuente": "the-odds-api", "ok": (queda or "0") not in ("0", None),
                    "peticiones_restantes": queda,
                    "usadas": r.headers.get("x-requests-used"),
                    "detalle": ("cuota agotada: los resultados de futbol no se pueden "
                                "cerrar por esta via" if queda == "0" else "operativa")})
    except Exception as e:
        fuentes.append({"fuente": "the-odds-api", "ok": False,
                        "detalle": f"{type(e).__name__}: {e}"})
    fuentes.append({
        "fuente": "openfootball (gratis, sin cuota)", "ok": True,
        "detalle": ("cierra los partidos de las 5 ligas europeas; va unos dias por "
                    "detras. NO cubre Champions ni Saudi todavia.")})
    return fuentes


@app.get("/api/value-picks")
def value_picks(sport: str = Query("all"), range: str = Query("7d"),
                start: str | None = None, end: str | None = None, limit: int = 40):
    """Los de MAYOR VALOR ESTIMADO y los que van contra el mercado.

    Es una lista distinta de Top picks a proposito: Top picks ordena por
    probabilidad (los 'seguros') y esta ordena por EV estimado (los que pagan
    mas de lo que el modelo cree que valen). Una cosa no implica la otra: un
    93 % a cuota 1.05 es segurisimo y pierde dinero.

    NADA de esto esta validado. Se dice en la respuesta, no en letra pequena.
    """
    a, b = _range(range, start, end)
    cards, warns = _collect(sport.upper(), a, b)
    con_precio = [c for c in cards if (c.get("valor") or {}).get("ev") is not None]
    contra = [c for c in cards if (c.get("valor") or {}).get("contra_mercado")]
    contra.sort(key=lambda c: -abs((c.get("valor") or {}).get("gap_pp") or 0))
    return {
        "from": a, "to": b, "sport": sport.upper(), "warnings": warns,
        "total_tarjetas": len(cards),
        "con_precio_registrado": len(con_precio),
        "valor": valor.ordenar_por_valor(con_precio, lambda c: c.get("valor"))[:limit],
        "contra_mercado": contra[:limit],
        "aviso": (
            "El EV sale de multiplicar la probabilidad del modelo por la cuota real. "
            "Solo vale si esa probabilidad esta bien calibrada, y eso NO esta "
            "demostrado: hoy hay muy pocas apuestas liquidadas con precio "
            "registrado. Trata esta lista como una hipotesis, no como un consejo."),
        "sin_precio": len(cards) - len(con_precio),
    }


@app.get("/api/top-picks")
def top_picks(sport: str = Query("all"), range: str = Query("today"),
              start: str | None = None, end: str | None = None, limit: int = 10):
    """Ordena por probabilidad final. Empates practicos se muestran juntos.

    No se ordena por una cifra sin declarar sus salvedades: cada pick lleva su
    estado de datos, si hay mercado y si el modelo y el mercado discrepan.
    """
    a, b = _range(range, start, end)
    cards, warns = _collect(sport.upper(), a, b)
    # ORDEN: por probabilidad calibrada. La auditoria §11 probo que ordenar por
    # "ventaja sobre la tasa base" es PEOR: los 20 mejores por edge acertaron
    # 60 % frente al 75 % de los 20 mejores por probabilidad en moneyline, 40 %
    # vs 75 % en F5 y 47 % vs 79 % en run line (walk-forward 2023-2026).
    # El edge se sigue mostrando como contexto, pero no ordena.
    # Los mercados se separan: primero los que tienen ventaja demostrada
    # (publish_pick), despues los que solo publican probabilidad.
    have = [c for c in cards if c.get("ensemble_probability") is not None]
    for c in have:
        ex = c.get("extra") or {}
        base = ex.get("base_rate_selection")
        c["edge_vs_base"] = (float(c["ensemble_probability"]) - base) if base is not None else None
        c["validated_pick"] = bool(ex.get("publish_pick", True))
    have.sort(key=lambda c: (not c["validated_pick"], -float(c["ensemble_probability"])))
    top = have[:limit]
    groups, cur = [], []
    for c in top:
        if (cur and cur[-1]["validated_pick"] == c["validated_pick"] and
                abs(float(cur[-1]["ensemble_probability"]) -
                    float(c["ensemble_probability"])) <= 0.015):
            cur.append(c)
        else:
            if cur:
                groups.append(cur)
            cur = [c]
    if cur:
        groups.append(cur)
    caveats = ["Orden por probabilidad calibrada. La auditoria comprobo que ordenar "
               "por ventaja sobre la tasa base empeora los resultados, asi que el "
               "edge se muestra como contexto pero no ordena.",
               "Primero los mercados con ventaja demostrada; despues los que solo "
               "publican probabilidad."]
    sin_pick = sorted({(c.get("extra") or {}).get("market_key") for c in top
                       if (c.get("extra") or {}).get("publish_pick") is False} - {None})
    if sin_pick:
        caveats.append("Mercados incluidos solo como probabilidad calibrada (su ventaja "
                       "de acierto NO es significativa): " + ", ".join(sin_pick) + ".")
    if any(c.get("market_probability") is None for c in top):
        caveats.append("Hay picks sin cuotas de mercado: su probabilidad viene solo del modelo.")
    if any(not (c.get("data_completeness") or {}).get("complete", True) for c in top):
        caveats.append("Hay picks con datos incompletos; se marcan en la tarjeta.")
    if not top:
        caveats.append("No hay ninguna oportunidad con probabilidad calculable en este rango. "
                       "El sistema no fuerza un pick: NO PICK.")
    return {"from": a, "to": b, "count": len(top), "picks": top,
            "tie_groups": [[c["game_id"] for c in g] for g in groups if len(g) > 1],
            "caveats": caveats, "warnings": warns}


@app.get("/api/mlb/game/{game_id}")
def mlb_game_detail(game_id: int):
    from MLB.engine.detail import game_detail
    return game_detail(game_id)


@app.get("/api/history")
def combined_history(sport: str = Query("all"), date_from: str | None = None,
                     date_to: str | None = None):
    from MLB.engine.history import combined
    return combined(sport, date_from, date_to)


@app.get("/api/mlb/history")
def mlb_history(date_from: str | None = None, date_to: str | None = None):
    from MLB.engine.history import history
    return history(date_from, date_to)


@app.get("/api/parlays")
def parlays(regenerar: bool = False):
    """Combinadas abiertas y calificadas. `regenerar=true` recalcula las de hoy."""
    from PARLAY.engine.scoring import history, score
    if regenerar:
        from PARLAY.engine.builder import generate
        generate()
    score()
    return history()


@app.get("/api/tenis/health")
def tenis_health():
    return tenm.health()


@app.get("/api/tenis/markets")
def tenis_markets():
    return tenm.markets_status()


@app.get("/api/soccer/health")
def soccer_health():
    return socm.health()


@app.get("/api/soccer/markets")
def soccer_markets():
    return socm.markets_status()


@app.get("/api/nba/health")
def nba_health():
    return nbam.health()


@app.get("/api/nba/markets")
def nba_markets():
    return nbam.markets_status()


@app.get("/api/live")
def live_scores():
    """Marcador en vivo MLB (cacheado 45 s). NFL: sin fuente en vivo, solo finales."""
    from dashboard.backend import live
    return live.live()


@app.get("/api/markets")
def markets():
    """Que mercados publica cada deporte y con que evidencia."""
    nfl_m = {"moneyline": {"enabled": True, "publish_pick": True, "mode": "pick",
                           "reason": "motor NFL existente (nflpred v4, ensemble calibrado); no se modifica"}}
    try:
        nfl_m.update(nflm.markets_status())
    except Exception as e:
        nfl_m["total"] = nfl_m["spread"] = {"enabled": False, "reason": f"no disponible: {e}"}
    try:
        nba_m = nbam.markets_status()
    except Exception as e:
        nba_m = {"error": str(e)}
    try:
        ten_m = tenm.markets_status()
    except Exception as e:
        ten_m = {"error": str(e)}
    try:
        soc_m = socm.markets_status()
    except Exception as e:
        soc_m = {"error": str(e)}
    return {"MLB": mlb.markets_status(), "NFL": nfl_m, "NBA": nba_m, "TENIS": ten_m,
            "SOCCER": soc_m}


@app.get("/api/alerts")
def alerts(sport: str = Query("all"), range: str = Query("7d"),
           start: str | None = None, end: str | None = None):
    a, b = _range(range, start, end)
    out = []
    if sport.upper() in ("ALL", "MLB"):
        out += mlb.alerts(a, b)
    if sport.upper() in ("ALL", "NFL"):
        for c in nfl.games(a, b):
            base = {"sport": "NFL", "game_id": c["game_id"],
                    "game": f"{c['away']} en {c['home']}", "date": c["game_date"],
                    "hora": c.get("start_utc"), "market": c["market"],
                    "selection": c["selection"]}
            dc = c.get("data_completeness") or {}
            for m in dc.get("missing", []):
                out.append({**base, "code": "DATA_INCOMPLETE", "level": "warn", "text": m})
            if c.get("directional_disagreement"):
                out.append({**base, "code": "MODEL_VS_MARKET", "level": "info",
                            "text": "modelo y mercado favorecen lados distintos"})
            if (c.get("upset_risk") or 0) >= 60:
                out.append({**base, "code": "HIGH_RISK", "level": "warn",
                            "text": f"riesgo alto: {round(c['upset_risk'])}/100"})
        try:
            nfl_cards = nflm.games(a, b)
        except Exception:
            nfl_cards = []
        for c in nfl_cards:
            ex = c.get("extra") or {}
            base = {"sport": "NFL", "game_id": c["game_id"], "game": f"{c['away']} en {c['home']}",
                    "date": c["game_date"], "hora": c.get("start_utc"), "market": c["market"],
                    "selection": c["selection"]}
            p = c.get("ensemble_probability") or 0.5
            if ex.get("publish_pick") and p >= 0.60:
                out.append({**base, "code": "STRONG_TOTAL" if ex.get("market_key") == "total" else "STRONG_SPREAD",
                            "level": "info", "text": f"pick fuerte: {c['selection']} {p*100:.0f}%"})
            for f in ((c.get("upset_explanation") or {}).get("flags") or []):
                code = f.get("code")
                if code == "MODEL_MARKET_DISAGREEMENT":
                    out.append({**base, "code": "MODEL_VS_MARKET", "level": "info",
                                "text": "Existe desacuerdo significativo entre modelo y mercado. " + f["text"]})
                elif code == "NO_PICK":
                    out.append({**base, "code": "NO_PICK", "level": "info", "text": f["text"]})
                elif code == "DATA_INCOMPLETE":
                    out.append({**base, "code": "DATA_INCOMPLETE", "level": "warn", "text": f["text"]})
            if not ex.get("publish_pick") and p < 0.53:
                pass  # ya cubierto por NO_PICK
            if (c.get("data_completeness") or {}).get("missing") and p < 0.53:
                out.append({**base, "code": "HIGH_UNCERTAINTY", "level": "warn",
                            "text": "incertidumbre alta: datos incompletos y modelo sin lado claro"})
    if sport.upper() in ("ALL", "NBA"):
        try:
            nba_cards = nbam.games(a, b)
        except Exception:
            nba_cards = []
        strong_ok = {}
        try:
            for mk, g in nbam.markets_status().items():
                # "strong" solo si el bucket 70+ del holdout acerto > 75 % con n >= 300
                b70 = (g.get("buckets") or {}).get("70-100") or {}
                strong_ok[mk] = bool(g.get("publish_pick") and b70.get("n", 0) >= 300 and (b70.get("acierto_real") or 0) >= 0.75)
        except Exception:
            pass
        for c in nba_cards:
            ex = c.get("extra") or {}
            base = {"sport": "NBA", "game_id": c["game_id"], "game": f"{c['away']} en {c['home']}",
                    "date": c["game_date"], "hora": c.get("start_utc"), "market": c["market"], "selection": c["selection"]}
            p = c.get("ensemble_probability") or 0.5
            mk = ex.get("market_key")
            if strong_ok.get(mk) and p >= 0.70:
                out.append({**base, "code": {"moneyline": "STRONG_MONEYLINE", "spread": "STRONG_SPREAD", "total": "STRONG_TOTAL"}[mk],
                            "level": "info", "text": f"pick fuerte con evidencia historica en este rango: {c['selection']} {p*100:.0f}%"})
            for f in ((c.get("upset_explanation") or {}).get("flags") or []):
                code = {"MODEL_MARKET_DISAGREEMENT": "MODEL_VS_MARKET", "NO_PICK": "NO_PICK", "LINE_MOVEMENT": "LINE_MOVEMENT",
                        "NO_MARKET": "DATA_INCOMPLETE"}.get(f.get("code"))
                if code:
                    out.append({**base, "code": code, "level": "info" if code != "DATA_INCOMPLETE" else "warn", "text": f["text"]})
            for m in (c.get("data_completeness") or {}).get("missing", []):
                out.append({**base, "code": "DATA_INCOMPLETE", "level": "warn", "text": m})
            if (c.get("data_completeness") or {}).get("missing") and p < 0.55:
                out.append({**base, "code": "HIGH_UNCERTAINTY", "level": "warn", "text": "incertidumbre alta: datos incompletos y sin lado claro"})
        try:
            h = nbam.health()
            if h.get("estado_global") == "WARNING":
                out.append({"sport": "NBA", "game_id": None, "game": "datos NBA", "date": a, "hora": None, "market": "-",
                            "selection": None, "code": "STALE_DATA", "level": "warn",
                            "text": f"datos NBA con avisos: ultimo partido {h.get('ultimo_partido')}, logs faltantes {h.get('logs_faltantes')}, errores {h.get('errores_descarga')}"})
        except Exception:
            pass
    if sport.upper() in ("ALL", "TENIS"):
        try:
            cards = tenm.games(a, b)
        except Exception:
            cards = []
        for c in cards:
            ex = c.get("extra") or {}
            base = {"sport": "TENIS", "game_id": c["game_id"], "game": f"{c['home']} vs {c['away']}",
                    "date": c["game_date"], "hora": c.get("start_utc"), "market": c["market"],
                    "selection": c["selection"]}
            p = c.get("ensemble_probability") or 0.5
            if ex.get("publish_pick") and p >= 0.70:
                out.append({**base, "code": "STRONG_MONEYLINE", "level": "info",
                            "text": f"pick fuerte con evidencia historica en este rango: {c['selection']} {p*100:.0f}%"})
            for f in ((c.get("upset_explanation") or {}).get("flags") or []):
                code = {"MODEL_MARKET_DISAGREEMENT": "MODEL_VS_MARKET", "NO_PICK": "NO_PICK",
                        "LINE_MOVEMENT": "LINE_MOVEMENT", "NO_MARKET": "DATA_INCOMPLETE",
                        "DATA_INCOMPLETE": "DATA_INCOMPLETE"}.get(f.get("code"))
                if code:
                    out.append({**base, "code": code, "level": "warn" if code == "DATA_INCOMPLETE" else "info",
                                "text": f["text"]})
        try:
            h = tenm.health()
            if h.get("estado_global") == "WARNING":
                out.append({"sport": "TENIS", "game_id": None, "game": "datos de tenis", "date": a, "hora": None,
                            "market": "-", "selection": None, "code": "STALE_DATA", "level": "warn",
                            "text": f"archivo historico congelado en {h.get('archivo_congelado_en')} "
                                    f"({h.get('antiguedad_archivo_dias')} dias): las features pueden estar viejas"})
        except Exception:
            pass
    # prioridad: primero lo que cambia una decision, al final el ruido informativo
    PRIO = {"PITCHER_UNKNOWN": 0, "LINEUP_UNCONFIRMED": 1, "HIGH_RISK": 2,
            "STRONG_TOTAL": 2, "STRONG_SPREAD": 2, "STRONG_MONEYLINE": 2, "HIGH_UNCERTAINTY": 3,
            "LINE_MOVEMENT": 4, "STALE_DATA": 1, "NO_PICK": 7,
            "COIN_FLIP": 3, "MODEL_DISAGREEMENT": 4, "MODEL_VS_MARKET": 5,
            "MODEL_OVER_MARKET": 6, "MARKET_OVER_MODEL": 6, "ELO_GAP": 7,
            "DATA_INCOMPLETE": 8, "WEATHER_ALERT": 3, "PITCHER_CHANGE": 1}
    out.sort(key=lambda x: (PRIO.get(x["code"], 9), x["date"], x["game"]))
    resumen = {}
    for x in out:
        r = resumen.setdefault(x["code"], {"code": x["code"], "level": x["level"],
                                           "n": 0, "deportes": set()})
        r["n"] += 1
        r["deportes"].add(x["sport"])
    for r in resumen.values():
        r["deportes"] = sorted(r["deportes"])
    return {"from": a, "to": b, "count": len(out),
            "resumen": sorted(resumen.values(), key=lambda r: PRIO.get(r["code"], 9)),
            "alerts": out}


@app.get("/api/performance/detail")
def performance_detail(sport: str = Query("all"), market: str = Query("all"),
                       model: str = Query("all"), season: str = Query("all"),
                       date_from: str | None = None, date_to: str | None = None,
                       conf_min: float | None = None, risk_max: float | None = None):
    return perf.query(sport, market, model, season, date_from, date_to,
                      conf_min, risk_max)


@app.get("/api/performance/models")
def performance_models():
    return {"rows": perf.models_table()}


@app.get("/api/data-health")
def data_health():
    return perf.data_health()


@app.get("/api/audit")
def audit_docs():
    return perf.audit_files()


@app.get("/api/performance")
def performance(sport: str = Query("all")):
    out = {}
    if sport.upper() in ("ALL", "NFL"):
        out["NFL"] = nfl.performance()
        try:
            out["NFL"]["markets"] = nflm.markets_status()
        except Exception as e:
            out["NFL"]["markets"] = {"error": str(e)}
    if sport.upper() in ("ALL", "MLB"):
        out["MLB"] = mlb.performance()
    if sport.upper() in ("ALL", "NBA"):
        try:
            out["NBA"] = {"markets": nbam.markets_status()}
        except Exception as e:
            out["NBA"] = {"error": str(e)}
    if sport.upper() in ("ALL", "TENIS"):
        try:
            out["TENIS"] = {"markets": tenm.markets_status()}
        except Exception as e:
            out["TENIS"] = {"error": str(e)}
    if sport.upper() in ("ALL", "SOCCER"):
        try:
            out["SOCCER"] = socm.health()
        except Exception as e:
            out["SOCCER"] = {"error": str(e)}
    return out


@app.get("/api/sources")
def sources():
    from shared.sources import DOCS, PRIORITY
    nfl_p = nfl.performance()
    return {"priority": PRIORITY, "docs": DOCS,
            "NFL": nfl_p.get("sources", []) if nfl_p.get("available") else [],
            "MLB": mlb.status().get("sources", [])}


# ---------------------------------------------------------------------------
# Capa de valor: modelo vs mercado, favoritos >60 %, discrepancias y ROI.
#
# Todo lo de aqui es SOLO LECTURA sobre las predicciones ya emitidas. Ninguna
# de estas rutas puede cambiar una probabilidad: leen, agregan y devuelven,
# incluidos los resultados malos.
# ---------------------------------------------------------------------------

@app.get("/api/value/coverage")
def value_coverage():
    """Que hay realmente para medir. Alimenta Data Health."""
    from shared import ledger
    return ledger.cobertura()


@app.get("/api/value/favorites")
def value_favorites(sport: str = Query("all")):
    from shared import tracking
    return tracking.favoritos(None if sport == "all" else sport.upper())


@app.get("/api/value/discrepancies")
def value_discrepancies(sport: str = Query("all")):
    from shared import tracking
    return tracking.discrepancias(None if sport == "all" else sport.upper())


@app.get("/api/value/model-vs-market")
def value_model_vs_market(sport: str = Query("all"), limit: int = Query(200)):
    """Fila por prediccion activa con las CUATRO probabilidades separadas.

    Se muestran la cruda, la calibrada, la publicada (que ya lleva mercado
    dentro cuando el deporte mezcla) y la implicita del mercado. Estan
    separadas a proposito: juntarlas en una sola columna es como se pierde de
    vista que parte del numero es del modelo y que parte es del casino.
    """
    from shared import discrepancy, ledger
    rs = ledger.records(None if sport == "all" else sport.upper(), solo_activas=True)
    filas = []
    for r in rs:
        d = discrepancy.evaluar(
            model_raw=r["model_probability_raw"],
            model_cal=r["model_probability_calibrated"],
            market=r["market_probability"], odds_decimal=r["odds_decimal"],
            market_source=r["market_source"], fetched_at=r["odds_fetched_at"],
            event_time=r["event_time"], model_version=r["model_version"])
        filas.append({**{k: r[k] for k in ("sport", "league", "event_id", "event_time",
                                           "market", "selection", "line", "status",
                                           "model_version", "calibration_version",
                                           "published_probability")},
                      "MODEL_PROBABILITY_RAW": d["MODEL_PROBABILITY_RAW"],
                      "MODEL_PROBABILITY_CALIBRATED": d["MODEL_PROBABILITY_CALIBRATED"],
                      "MARKET_IMPLIED_PROBABILITY": d["MARKET_IMPLIED_PROBABILITY"],
                      "MARKET_ODDS": d["MARKET_ODDS"],
                      "MODEL_MARKET_GAP": d["MODEL_MARKET_GAP"],
                      "gap_pp": d["gap_pp"], "bucket": d["bucket"],
                      "direccion": d["direccion"],
                      "FINAL_STATUS": d["FINAL_STATUS"],
                      "explicacion": d["explicacion"]})
    filas.sort(key=lambda f: (-abs(f["gap_pp"] or 0)))
    con_mercado = sum(f["MARKET_IMPLIED_PROBABILITY"] is not None for f in filas)
    return {"n": len(filas), "con_mercado": con_mercado,
            "sin_mercado": len(filas) - con_mercado,
            "filas": filas[:limit],
            "nota": ("Un gap a favor del modelo NO es valor. Solo lo es si el "
                     "backtest de ese tramo demuestra ROI positivo con muestra "
                     "suficiente. Mira la pestana de Discrepancias.")}


# ---------------------------------------------------------------------------
# Calculadora de combinadas
# ---------------------------------------------------------------------------

@app.get("/api/parlay/calculate")
def parlay_calculate(selecciones: str = Query("[]"), stake: float = Query(1.0)):
    """Calcula una combinada. `selecciones` es un JSON con la lista de patas.

    Es GET y no POST a proposito. El dashboard tiene una garantia que se prueba
    en tests/test_dashboard.py: NO expone ningun metodo de escritura. Esta ruta
    solo calcula y no toca ninguna base, asi que puede ser GET sin problema, y
    mantenerlo asi conserva intacta una invariante que vale mas que la comodidad
    de mandar un cuerpo JSON.
    """
    import json

    from PARLAY.engine import calculator
    try:
        sels = json.loads(selecciones or "[]")
    except ValueError as e:
        return {"version": calculator.VERSION, "advertencia":
                f"no pude leer las selecciones: {e}", "cuota_decimal": None,
                "selecciones": [], "n_selecciones": 0, "n_validas": 0}
    if not isinstance(sels, list):
        sels = []
    return calculator.calcular(sels, float(stake or 1.0))


@app.get("/api/parlay/convert")
def parlay_convert(american: float | None = None, decimal: float | None = None):
    from PARLAY.engine import calculator
    return calculator.convertir(american=american, decimal=decimal)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.get("/api/chat")
def chat(texto: str = Query(""), game_id: str | None = None,
         sport: str | None = None, contexto: str = Query(""),
         historial: str = Query("")):
    """Pregunta al chat. GET por la misma razon que la calculadora: el chat
    consulta modelos, no escribe nada, y el dashboard no expone escritura.

    `contexto` es la respuesta anterior en JSON. Viaja de ida y vuelta en vez de
    guardarse en el servidor: asi el backend sigue sin estado y el chat sigue
    pudiendo contestar repreguntas ("¿entonces solo tiene el 29%?").
    """
    import json

    from CHAT import answer
    def _json(s, tipo):
        # Un parametro ilegible NO debe tumbar la respuesta: se ignora.
        if not s:
            return None
        try:
            v = json.loads(s)
        except ValueError:
            return None
        return v if isinstance(v, tipo) else None

    return answer.responder(texto, game_id, sport,
                            _json(contexto, dict), _json(historial, list))


@app.get("/api/chat/estado")
def chat_estado():
    """¿Hay un LLM conectado? El chat lo dice en vez de fingir que sí."""
    from CHAT import llm
    return {"llm": llm.disponible(), "nombre": "Caballo Chat"}


@app.get("/api/chat/games")
def chat_games(sport: str = Query("all"), q: str = Query(""), dias: int = Query(10)):
    """Partidos proximos con NOMBRE, para que nadie tenga que teclear un id."""
    from CHAT import games
    js = games.buscar(q, dias) if q else games.listar(dias)
    if sport != "all":
        js = [j for j in js if j["sport"] == sport.upper()]
    return {"n": len(js), "partidos": js}


@app.get("/api/chat/catalog")
def chat_catalog(sport: str = Query("all")):
    from CHAT import api as capi
    return capi.catalogo(None if sport == "all" else sport.upper())


@app.get("/api/chat/secondary/{game_id}")
def chat_secondary(game_id: str, sport: str | None = None):
    from CHAT import api as capi
    return capi.get_secondary_markets(game_id, sport)


@app.get("/api/chat/distribution/{game_id}")
def chat_distribution(game_id: str, market: str = Query(...)):
    from CHAT import api as capi
    return capi.get_distribution(game_id, market)


@app.get("/api/contextual")
def contextual_status():
    """Estado del modulo contextual: que partidos tienen ajustes activos."""
    from shared.contextual import get_all_adjustments, status as ctx_status
    s = ctx_status()
    s["adjustments"] = get_all_adjustments()
    return s


DIST = DASHBOARD_DIR / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST / "index.html")
else:
    @app.get("/")
    def index_missing():
        return JSONResponse({"error": "frontend no compilado",
                             "hint": "cd dashboard/frontend && npm install && npm run build"},
                            status_code=503)
