"""Performance Center (§18-§20) y Data Health (§21). SOLO LECTURA.

Fuente de las metricas MLB: las predicciones walk-forward partido a partido de la
auditoria (variante limpia, sin features del abridor). Son las mismas cifras que
gatean el sistema, no unas distintas.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from shared.calibration import summary
from shared.paths import ROOT

PRE = ROOT / "audit" / "preds"
OUT = ROOT / "audit" / "out"
from MLB.engine.markets import LABEL as MARKET_LABEL, HIDDEN as HIDDEN_MARKETS  # noqa: E402
MODEL_COL = {"elo": "p_home_elo", "ensemble": "p_home_model", "forest": "p_home_forest"}


@lru_cache(maxsize=1)
def _wf() -> pd.DataFrame:
    """Predicciones walk-forward de MLB, en formato 'seleccion'."""
    fs = sorted(PRE.glob("*.parquet"))
    if not fs:
        return pd.DataFrame()
    frames = []
    for f in fs:
        d = pd.read_parquet(f)
        for model, col in MODEL_COL.items():
            if col not in d.columns or d[col].isna().all():
                continue
            p = d[col].values
            home = p >= 0.5
            frames.append(pd.DataFrame({
                "sport": "MLB", "season": d.season, "market": d.market, "model": model,
                "game_id": d.game_id, "game_date": d.game_date, "start_utc": d.start_utc,
                "home": d.home, "away": d.away,
                "selection": np.where(home, d.home, d.away),
                "p": np.where(home, p, 1 - p),
                "hit": np.where(home, d.y_home == 1, d.y_home == 0).astype(int),
                "dispersion": d.dispersion,
                "base": np.where(home, d.base_rate_train, 1 - d.base_rate_train)}))
    # NFL total/spread: walk-forward propio (NFL/markets/out/lines_*.parquet),
    # probabilidad SIN calibrar frente a la linea de cierre (lo que se midio).
    from NFL.markets.gating import OUT as NFL_OUT
    for mk in ("total", "spread"):
        f = NFL_OUT / f"lines_{mk}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        d = d[~d.push]
        pcol = "p_norm"
        p = d[pcol].values
        over = p >= 0.5
        gid = d.game_id.astype(str)
        parts = gid.str.split("_", expand=True)
        gdate = pd.to_datetime(gid.str[:4] + "-09-01", errors="coerce")
        frames.append(pd.DataFrame({
            "sport": "NFL", "season": d.season, "market": mk, "model": "ridge",
            "game_id": d.game_id, "game_date": gdate, "start_utc": gdate,
            "home": parts[3] if parts.shape[1] > 3 else None, "away": parts[2] if parts.shape[1] > 2 else None,
            "selection": np.where(over, "over/local", "under/visitante"),
            "p": np.where(over, p, 1 - p),
            "hit": np.where(over, d.over, ~d.over).astype(int),
            "dispersion": np.nan, "base": 0.5}))
    # NBA: walk-forward propio (NBA/markets/out/lines_*.parquet)
    from shared.paths import NBA_OUT_DIR
    f = NBA_OUT_DIR / "lines_moneyline.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        frames.append(pd.DataFrame({
            "sport": "NBA", "season": d.season, "market": "moneyline", "model": "rf",
            "game_id": d.game_id, "game_date": pd.to_datetime(d.date), "start_utc": pd.to_datetime(d.date),
            "home": None, "away": None, "selection": np.where(d.p_cal >= 0.5, "local", "visitante"),
            "p": d.p_pick, "hit": d.hit.astype(int), "dispersion": np.nan, "base": 0.5}))
    for mk in ("spread", "total"):
        f = NBA_OUT_DIR / f"lines_{mk}.parquet"
        if f.exists():
            d = pd.read_parquet(f); d = d[d.off == 0]
            pc = d.p_norm if mk == "spread" else d.p_emp
            over = pc >= 0.5
            gdate = pd.to_datetime(d.season.astype(str) + "-01-01")
            frames.append(pd.DataFrame({
                "sport": "NBA", "season": d.season, "market": mk, "model": "ridge",
                "game_id": np.arange(len(d)), "game_date": gdate, "start_utc": gdate, "home": None, "away": None,
                "selection": np.where(over, "over/local", "under/visitante"), "p": np.where(over, pc, 1 - pc),
                "hit": np.where(over, d.over, 1 - d.over).astype(int), "dispersion": np.nan, "base": 0.5}))
    # TENIS: walk-forward propio por circuito (ganador, probabilidad calibrada)
    from shared.paths import TENIS_OUT_DIR
    for tour in ("ATP", "WTA"):
        f = TENIS_OUT_DIR / f"preds_{tour}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        col = "p_elo_model" if "p_elo_model" in d.columns else "elo_prob"
        d = d.dropna(subset=[col])
        side1 = d[col] >= 0.5
        frames.append(pd.DataFrame({
            "sport": "TENIS", "season": d.season, "market": f"winner_{tour}", "model": "elo_logistica",
            "game_id": d.match_id, "game_date": pd.to_datetime(d.date), "start_utc": pd.to_datetime(d.date),
            "home": None, "away": None, "selection": np.where(side1, "jugador 1", "jugador 2"),
            "p": np.where(side1, d[col], 1 - d[col]),
            "hit": np.where(side1, d.y, 1 - d.y).astype(int), "dispersion": np.nan, "base": 0.5}))
    out = pd.concat(frames, ignore_index=True)
    out["month"] = pd.to_datetime(out.game_date).dt.to_period("M").astype(str)
    out["week"] = pd.to_datetime(out.game_date).dt.to_period("W").astype(str)
    out["confianza"] = (out.p - 0.5) * 2
    out["riesgo"] = (1 - out.p) * 100
    return out


def _metrics(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"n": 0}
    s = summary(d.hit.values, d.p.values)
    return {"n": int(len(d)), "wins": int(d.hit.sum()), "losses": int((1 - d.hit).sum()),
            "accuracy": s["accuracy"], "log_loss": s["log_loss"], "brier": s["brier"],
            "ece": s["ece"], "prob_media": float(d.p.mean()),
            "base_media": float(d.base.mean()),
            "reliability": s["reliability"]}


def query(sport="all", market="all", model="all", season="all",
          date_from=None, date_to=None, conf_min=None, risk_max=None) -> dict:
    d = _wf()
    if d.empty:
        return {"available": False,
                "reason": "no hay predicciones walk-forward generadas todavia"}
    f = d
    if sport.upper() != "ALL":
        f = f[f.sport == sport.upper()]
    if market != "all":
        f = f[f.market == market]
    if model != "all":
        f = f[f.model == model]
    if season != "all":
        f = f[f.season == int(season)]
    if date_from:
        f = f[f.game_date >= date_from]
    if date_to:
        f = f[f.game_date <= date_to]
    if conf_min is not None:
        f = f[f.confianza >= float(conf_min)]
    if risk_max is not None:
        f = f[f.riesgo <= float(risk_max)]
    return {"available": True, "filtros": {
        "sport": sport, "market": market, "model": model, "season": season,
        "date_from": date_from, "date_to": date_to,
        "conf_min": conf_min, "risk_max": risk_max},
        "total": _metrics(f),
        "roi": {"disponible": False,
                "motivo": "sin historico de cuotas MLB no se puede calcular ROI, "
                          "yield ni closing line value"},
        "por_modelo": _by(f, ["sport", "market", "model"]),
        "por_temporada": _by(f, ["season"]),
        "por_mes": _by(f, ["month"]),
        "por_semana": _by(f, ["week"]),
        "por_confianza": _bucketed(f, "confianza", [0, .05, .1, .2, .3, 1.01]),
        "por_riesgo": _bucketed(f, "riesgo", [0, 30, 35, 40, 45, 51])}


def _by(d: pd.DataFrame, keys: list[str]) -> list[dict]:
    if d.empty:
        return []
    out = []
    for k, g in d.groupby(keys, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, [str(x) for x in k]))
        m = _metrics(g)
        m.pop("reliability", None)
        out.append({**row, **m})
    return sorted(out, key=lambda r: tuple(r[k] for k in keys))


def _bucketed(d: pd.DataFrame, col: str, edges: list[float]) -> list[dict]:
    if d.empty:
        return []
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = d[(d[col] >= lo) & (d[col] < hi)]
        m = _metrics(g)
        m.pop("reliability", None)
        out.append({"rango": f"{lo}-{hi}", **m})
    return out


def models_table() -> list[dict]:
    """§19: una fila por modelo, deporte y mercado."""
    d = _wf()
    rows = _by(d, ["sport", "market", "model"]) if not d.empty else []
    for r in rows:
        r["market_label"] = ({"total": "Total", "spread": "Spread"}.get(r["market"], r["market"]) if r["sport"] != "MLB"
                             else MARKET_LABEL.get(r["market"], r["market"]))
    # NFL: del backtest almacenado del motor existente
    try:
        import NFL.adapter as nfl
        p = nfl.performance()
        st = ((p.get("latest_backtest") or {}).get("summary") or {}).get("strategies") or {}
        for name, v in st.items():
            rows.append({"sport": "NFL", "market": "moneyline", "market_label": "Moneyline",
                         "model": name, "n": None, "accuracy": v.get("accuracy"),
                         "log_loss": v.get("log_loss"), "brier": v.get("brier"),
                         "ece": None, "wins": None, "losses": None,
                         "fantasy_points_pct": v.get("points_pct")})
    except Exception:                                    # noqa: BLE001
        pass
    return rows


def _clasificar_error(msg: str | None) -> tuple[str | None, str | None]:
    """Traduce el error tecnico a algo accionable.

    El caso mas comun no es un fallo del sistema sino de la conexion: si el DNS
    no resuelve, ninguna descarga funciona y el usuario merece que se lo digan
    asi, no con un `getaddrinfo failed`.
    """
    if not msg:
        return None, None
    m = msg.lower()
    if "getaddrinfo" in m or "name or service not known" in m or "nodename" in m:
        return ("sin conexion / DNS",
                "el equipo no pudo resolver el nombre del servidor. Revisa tu "
                "conexion, VPN o firewall y vuelve a ejecutar `python spc.py mlb-daily`: "
                "la ingesta es reanudable y recuperara lo que falte")
    if "timed out" in m or "timeout" in m:
        return ("tiempo agotado",
                "el servidor tardo demasiado. Suele resolverse repitiendo la ejecucion")
    if "429" in m or "rate" in m:
        return ("limite de peticiones",
                "se alcanzo el limite de la API. Espera unos minutos y reintenta")
    if "401" in m or "403" in m or "apikey" in m:
        return ("credencial",
                "la clave de API falta o no es valida. Revisa el archivo .env")
    if "truncada" in m:
        return ("respuesta truncada",
                "la fuente devolvio el maximo de filas; el rango se partio y se "
                "reintento automaticamente")
    return ("otro", None)


def data_health() -> dict:
    """§21: estado de cada fuente.

    Un error de descarga NO significa que falten datos: casi siempre es un fallo
    de red que el reintento resolvio. Por eso se separan los errores RECIENTES
    (ultimos 7 dias) de los historicos, y sobre todo se mide la COBERTURA REAL:
    cuantos partidos terminados se quedaron sin boxscore y cuantos dias de juego
    sin Statcast. Esa es la pregunta que importa.
    """
    from datetime import timedelta

    from sqlalchemy import func, select

    from MLB.database.models import DataSourceLog, Game, PitcherGameLog, StatcastAgg
    from MLB.database.session import init_db, session_scope
    from shared.timeutil import utcnow
    init_db()
    FINAL = ("Final", "Game Over", "Completed Early")
    corte = utcnow() - timedelta(days=7)
    out = {"MLB": [], "NFL": []}
    with session_scope() as s:
        rows = s.execute(select(
            DataSourceLog.source, DataSourceLog.domain,
            func.max(DataSourceLog.retrieved_at), func.count(),
            func.sum(func.iif(DataSourceLog.status == "error", 1, 0)),
            func.sum(func.iif((DataSourceLog.status == "error") &
                              (DataSourceLog.retrieved_at >= corte), 1, 0)),
            func.sum(DataSourceLog.records))
            .group_by(DataSourceLog.source, DataSourceLog.domain)).all()
        ultimo_error = {}
        for src, dom, msg, when in s.execute(select(
                DataSourceLog.source, DataSourceLog.domain, DataSourceLog.error,
                DataSourceLog.retrieved_at).where(DataSourceLog.status == "error")
                .order_by(DataSourceLog.retrieved_at.desc()).limit(200)).all():
            ultimo_error.setdefault((src, dom), (msg, when))

        # --- cobertura REAL
        finales = s.scalar(select(func.count()).select_from(Game)
                           .where(Game.status.in_(FINAL))) or 0
        con_box = len(set(s.execute(select(
            PitcherGameLog.game_id.distinct())).scalars().all()))
        dias_juego = set(s.execute(select(Game.game_date.distinct())
                                   .where(Game.status.in_(FINAL))).scalars().all())
        dias_sc = set(s.execute(select(StatcastAgg.game_date.distinct())).scalars().all())
        from datetime import date as _date
        hoy = str(_date.today())
        # el dia en curso no es un hueco: los partidos todavia no han terminado
        faltan_sc = sorted(d for d in dias_juego
                           if d and d not in dias_sc and d < hoy)
        n_games = s.scalar(select(func.count()).select_from(Game)) or 0

    cobertura = {
        "mlb_stats_api": {
            "boxscore": (f"{con_box}/{finales} partidos terminados con boxscore",
                         con_box >= finales),
            "schedule": (f"{n_games} partidos en la base", n_games > 0),
        },
        "baseball_savant": {
            "statcast": (f"{len(dias_juego) - len(faltan_sc)}/{len(dias_juego)} dias con "
                         f"partido cubiertos" + (f" · faltan {len(faltan_sc)}"
                                                 if faltan_sc else ""),
                         not faltan_sc),
        },
    }

    for src, dom, last, calls, errs, errs_rec, recs in rows:
        errs, errs_rec = int(errs or 0), int(errs_rec or 0)
        cov_txt, cov_ok = cobertura.get(src, {}).get(dom, ("—", True))
        if errs_rec and not cov_ok:
            estado = "CON ERRORES"
        elif errs_rec:
            estado = "REINTENTOS"
        elif not cov_ok:
            estado = "DATOS INCOMPLETOS"
        else:
            estado = "ONLINE"
        msg, when = ultimo_error.get((src, dom), (None, None))
        tipo, consejo = _clasificar_error(msg)
        out["MLB"].append({
            "fuente": src, "dominio": dom, "ultima_descarga": last,
            "descargas": calls, "errores": errs, "errores_recientes": errs_rec,
            "registros": int(recs or 0), "estado": estado, "cobertura": cov_txt,
            "cobertura_ok": bool(cov_ok),
            "ultimo_error": ({"mensaje": (msg or "")[:200], "cuando": when,
                              "tipo": tipo, "consejo": consejo} if msg else None),
            "nota": ("los fallos de descarga se reintentaron y la cobertura esta "
                     "completa" if (errs and cov_ok) else None),
        })
    out["MLB"].sort(key=lambda r: (r["estado"] == "ONLINE", r["fuente"], r["dominio"]))
    recientes = [r for r in out["MLB"] if r["errores_recientes"]]
    red = [r for r in recientes
           if (r.get("ultimo_error") or {}).get("tipo") == "sin conexion / DNS"]
    out["diagnostico"] = ({
        "problema": "sin conexion / DNS",
        "detalle": (f"{sum(r['errores_recientes'] for r in red)} descargas fallaron en "
                    f"los ultimos 7 dias porque el equipo no pudo resolver el nombre "
                    f"del servidor. No es un fallo del sistema, es la red."),
        "fuentes": sorted({r["fuente"] for r in red}),
        "consejo": (red[0]["ultimo_error"]["consejo"] if red else None),
    } if red else ({"problema": "fallos recientes sin clasificar",
                    "fuentes": sorted({r["fuente"] for r in recientes}),
                    "detalle": "hubo fallos de descarga en los ultimos 7 dias"}
                   if recientes else {"problema": None,
                                      "detalle": "sin fallos en los ultimos 7 dias"}))
    out["resumen_mlb"] = {
        "boxscores": f"{con_box}/{finales}",
        "boxscores_ok": con_box >= finales,
        "dias_statcast_faltantes": faltan_sc[:20],
        "n_dias_faltantes": len(faltan_sc),
        "veredicto": ("cobertura completa: los errores de descarga fueron transitorios"
                      if (con_box >= finales and not faltan_sc)
                      else "hay huecos de datos, revisar la tabla"),
    }
    try:
        import NFL.adapter as nfl
        p = nfl.performance()
        for s_ in (p.get("sources") or []):
            out["NFL"].append({"fuente": s_["source"], "dominio": "-",
                               "ultima_descarga": s_["last_fetch"],
                               "descargas": None, "errores": None,
                               "errores_recientes": None, "registros": None,
                               "estado": ("ONLINE" if s_.get("status") == "ok"
                                          else str(s_.get("status"))),
                               "cobertura": "motor NFL existente", "cobertura_ok": True,
                               "ultimo_error": None, "nota": None})
    except Exception:                                    # noqa: BLE001
        pass
    try:
        from MLB.database.session import session_scope as ss
        from MLB.ingest.odds import api_key, snapshot_stats
        with ss() as s:
            st = snapshot_stats(s)
        out["odds"] = {**st, "clave_configurada": bool(api_key()),
                       "estado": ("ONLINE" if api_key() else
                                  "SIN CLAVE: no se descargan cuotas")}
    except Exception as e:                               # noqa: BLE001
        out["odds"] = {"estado": f"error: {e}"}
    # NFL por mercado (moneyline = motor existente; total/spread = mercados propios)
    try:
        from NFL.markets import provider as nflm
        h = nflm.health()
        ml_ok = bool(out["NFL"]) and all(r["estado"] == "ONLINE" for r in out["NFL"])
        out["nfl_mercados"] = {
            "moneyline": {"estado": "verde" if ml_ok else "amarillo", "modo": "pick",
                          "modelo": "nflpred v4 (motor existente)", "nota": "no se modifica"},
            **h}
    except Exception as e:                               # noqa: BLE001
        out["nfl_mercados"] = {"error": str(e)}
    try:
        from TENIS.markets import provider as tenm
        out["tenis"] = tenm.health()
    except Exception as e:                               # noqa: BLE001
        out["tenis"] = {"error": str(e)}
    try:
        from SOCCER.markets import provider as socm
        out["SOCCER"] = socm.health()
    except Exception as e:                       # una fuente caida no tumba el panel
        out["SOCCER"] = {"error": str(e)}
    try:
        from NBA.markets import provider as nbam
        out["nba"] = nbam.health()
    except Exception as e:                               # noqa: BLE001
        out["nba"] = {"error": str(e)}
    return out


def audit_files() -> dict:
    """Deja consultables los JSON de la auditoria desde el dashboard."""
    res = {}
    for f in sorted(OUT.glob("*.json")):
        try:
            res[f.stem] = json.loads(f.read_text())
        except Exception:                                # noqa: BLE001
            res[f.stem] = {"error": "no se pudo leer"}
    return res
