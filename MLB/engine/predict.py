"""Predicciones MLB en vivo.

Para cada partido futuro se calculan los mercados HABILITADOS por la evidencia
(`gating.decide()`), se combinan con el mercado si hay cuotas anteriores al
momento de la prediccion, y se guarda una fila nueva por version. Nunca se borra
ni se sobrescribe una prediccion anterior.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from MLB.database.models import Game, Lineup, ModelVersion, Prediction, PredictionSnapshot
from MLB.engine import uncertainty as unc
from MLB.engine.explain import narrative, narrative_total
from MLB.engine.gating import decide
from MLB.engine.markets import HIDDEN as HIDDEN_MARKETS
from MLB.engine.models import dispersion
from MLB.ingest.odds import consensus
from shared.blend import DEFAULT_POLICY, disagreement, signal_label
from shared.paths import MLB_MODELS_DIR, MLB_PROCESSED_DIR
from shared.timeutil import json_safe, utcnow

PICK_LABEL = {
    "moneyline": lambda r, home, away, p: (home if p >= .5 else away, None),
    "f5_moneyline": lambda r, home, away, p: (f"{home} (5 entradas)" if p >= .5
                                              else f"{away} (5 entradas)", None),
    "run_line": lambda r, home, away, p: ((f"{home} -1.5", -1.5) if p >= .5
                                          else (f"{away} +1.5", 1.5)),
    "total": lambda r, home, away, p: ((f"Over {r._line:g}", r._line) if p >= .5
                                       else (f"Under {r._line:g}", r._line)),
}


def _ref_line(mu: float) -> float:
    """Linea de referencia cuando no hay mercado: la media proyectada al .5 mas cercano."""
    return float(np.floor(mu) + 0.5)


def load_models(version: str = "v2") -> dict:
    out = {}
    for f in MLB_MODELS_DIR.glob(f"*_{version}.joblib"):
        key = f.stem.rsplit(f"_{version}", 1)[0]
        out[key] = joblib.load(f)
    return out


class _RowLine:
    """Envuelve la fila con la linea del total (para PICK_LABEL)."""
    def __init__(self, row, line):
        self._row, self._line = row, line

    def __getattr__(self, k):
        return getattr(self._row, k)


def _with_line(row, line):
    return _RowLine(row, line)


def _completeness(row, has_odds: bool, lineup_confirmed: bool) -> tuple[list, list]:
    missing, notes = [], []
    if not bool(getattr(row, "h_sp_known", np.nan) == 1.0):
        missing.append("el local no ha anunciado abridor")
    if not bool(getattr(row, "a_sp_known", np.nan) == 1.0):
        missing.append("el visitante no ha anunciado abridor")
    if not lineup_confirmed:
        missing.append("alineaciones sin confirmar")
    if not has_odds:
        missing.append("sin cuotas de casas de apuestas")
    for c, label in (("h_off_games_s", "pocos partidos del local esta temporada"),
                     ("a_off_games_s", "pocos partidos del visitante esta temporada")):
        v = getattr(row, c, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)) or v < 15:
            missing.append(label)
    if getattr(row, "park_runs_factor", None) is None or \
            (isinstance(getattr(row, "park_runs_factor", np.nan), float) and
             np.isnan(getattr(row, "park_runs_factor"))):
        notes.append("sin park factor de temporadas anteriores para esta sede")
    return missing, notes


def predict_range(session, date_from: str, date_to: str, version: str = "v2",
                  reason: str = "manual", progress=None) -> dict:
    X = pd.read_parquet(MLB_PROCESSED_DIR / "features.parquet")
    elo_art = joblib.load(MLB_MODELS_DIR / f"elo_{version}.joblib")
    # Las calificaciones Elo se RECALCULAN aqui con los resultados actuales, usando
    # la configuracion (k, ventaja, regresion) elegida al entrenar. Si se usara la
    # tabla guardada en el artefacto, las calificaciones quedarian congeladas en el
    # dia del entrenamiento y cada partido nuevo se predeciria con datos viejos.
    # Es la unica parte del modelo que "aprende" a diario; tarda ~2 segundos.
    from MLB.engine.elo import run as run_elo
    elo_now = run_elo(X, elo_art["cfg"])
    Xe = X.merge(elo_now[["game_id", "p_home_elo"]], on="game_id", how="left")
    Xe["d_elo"] = Xe.p_home_elo - 0.5

    gate = decide()
    models = {k: v for k, v in load_models(version).items() if k != "elo"}
    # moneyline se sirve con Elo (decision de la auditoria §5/§30), no con ML
    ELO_MARKETS = {"moneyline"}
    mv = {m.market: m for m in session.execute(
        select(ModelVersion).where(ModelVersion.is_production.is_(True))).scalars()}

    sub = Xe[(Xe.game_date >= date_from) & (Xe.game_date <= date_to)].copy()
    if sub.empty:
        return {"status": "empty", "reason": f"no hay partidos entre {date_from} y {date_to}",
                "predictions": 0}
    now = utcnow()
    made, revised, skipped = 0, 0, []
    lineups = {(l.game_id, l.team_id): l for l in session.execute(
        select(Lineup).where(Lineup.game_id.in_(sub.game_id.tolist()))).scalars()}

    for market in sorted(set(models) | ELO_MARKETS):
        g = gate.get(market, {})
        if market in HIDDEN_MARKETS:
            skipped.append({"market": market, "reason": "retirado de la vista: sustituido por Over/Under"})
            continue
        if not g.get("enabled"):
            skipped.append({"market": market, "reason": g.get("reason", "no habilitado")})
            continue
        if market in ELO_MARKETS:
            mm = None
            p_all = sub.p_home_elo.values
            members = {"elo": p_all}
            disp = np.zeros(len(sub))
        else:
            mm = models.get(market)
            if mm is None:
                skipped.append({"market": market, "reason": "sin artefacto entrenado"})
                continue
            if market == "total":
                mu_all = mm.predict_total(sub)
                p_all = np.full(len(sub), np.nan)
                members, disp = {}, np.zeros(len(sub))
            else:
                p_all = mm.predict_proba(sub)
                members = mm.member_probs(sub)
                disp = dispersion(members)
        for i, row in enumerate(sub.itertuples()):
            cut = pd.Timestamp(row.start_utc)
            cons = consensus(session, int(row.game_id), market, cut)
            p_market = cons["p_home"] if cons else None
            total_info = None
            if market == "total":
                mu = float(mu_all[i])
                if np.isnan(mu):
                    continue
                # mediana de la distribucion proyectada (las carreras son asimetricas:
                # la media queda por encima del punto 50/50)
                mu = mu + float(np.median(mm.residuals))
                line = cons["line"] if (cons and cons.get("line") is not None) else _ref_line(mu)
                row = _with_line(row, line)
                p_all[i] = float(mm.prob_over(sub.iloc[[i]], line)[0])
                total_info = {"proyeccion": round(mu, 2), "linea": line,
                              "linea_origen": "mercado" if (cons and cons.get("line") is not None)
                              else "referencia (sin mercado)"}
            if p_all[i] is None or (isinstance(p_all[i], float) and np.isnan(p_all[i])):
                continue
            p_model = float(p_all[i])
            d = float(disp[i]) if len(disp) else 0.0
            pol = DEFAULT_POLICY
            p_final = pol.blend(p_model, p_market)
            home, away = row.home_abbr, row.away_abbr
            pick, line = PICK_LABEL[market](row, home, away, p_final)
            # Todas las probabilidades que se guardan son las DE LA SELECCION, no
            # las del local. Mostrar 49.6% junto a un pick visitante seria falso.
            pick_is_home = p_final >= 0.5
            flip = (lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                    else (float(v) if pick_is_home else 1.0 - float(v)))
            p_model_sel = flip(p_model)
            p_market_sel = flip(p_market)
            p_final_sel = flip(p_final)
            p_elo_sel = None if market == "total" else flip(row.p_home_elo)
            lc = any(k[0] == row.game_id and lineups[k].state == "confirmed" for k in lineups
                     if k[0] == row.game_id)
            missing, notes = _completeness(row, cons is not None, lc)
            starter_known = bool(getattr(row, "h_sp_known", 0) == 1.0 and
                                 getattr(row, "a_sp_known", 0) == 1.0)
            risk = unc.uncertainty(p_final_sel)
            fl = unc.flags(dispersion=d, p_elo=p_elo_sel, p_selection=p_final_sel,
                           missing=missing, lineup_confirmed=lc,
                           starter_known=starter_known, p_market=p_market_sel)
            dis = disagreement(p_model_sel, p_market_sel, p_final_sel, d, p_elo_sel)
            if market == "total":
                expl = narrative_total(row, home, away, pick, p_final_sel, total_info,
                                       missing, p_market_sel)
            else:
                expl = narrative(row, home, away, pick.split(" ")[0], p_final_sel,
                                 p_elo_sel, d, missing, p_market_sel)
            expl["flags"] = json_safe(fl)
            expl["uncertainty_note"] = (
                "El riesgo es simplemente la probabilidad de que el pick falle "
                "(100 menos la probabilidad del pick).")
            made_one, was_rev = _store(
                session, row, market, pick, line, p_model_sel, p_market_sel, p_final_sel,
                p_elo_sel, risk, expl, dis, pol, missing, notes, d,
                mv.get(market), now, reason, g)
            made += made_one
            revised += was_rev
        if progress:
            progress(f"{market}: {len(sub)} partidos")
    return {"status": "ok", "from": date_from, "to": date_to, "games": int(len(sub)),
            "predictions": made, "revisions": revised, "skipped_markets": skipped,
            "market_available": any(True for _ in []) or None}


def _store(session, row, market, pick, line, p_model, p_market, p_final, p_elo,
           risk, expl, dis, pol, missing, notes, disp, mv, now, reason, gate) -> tuple[int, int]:
    prev = session.execute(
        select(Prediction).where(Prediction.game_id == int(row.game_id),
                                 Prediction.market == market,
                                 Prediction.status != "superseded")
        .order_by(Prediction.version.desc())).scalars().first()
    changed = True
    if prev is not None:
        changed = (abs((prev.ensemble_probability or 0) - p_final) > 0.005 or
                   prev.selection != pick)
        if not changed:
            return 0, 0
        prev.status = "superseded"
    p = Prediction(
        game_id=int(row.game_id), season=int(row.season), game_date=row.game_date,
        game_start_utc=pd.Timestamp(row.start_utc).to_pydatetime(),
        home_abbr=row.home_abbr, away_abbr=row.away_abbr,
        market=market, selection=pick, line=line,
        model_probability=p_model, market_probability=p_market,
        ensemble_probability=p_final,
        elo_probability=p_elo,
        confidence=abs(p_final - 0.5) * 2, upset_risk=risk,
        upset_label=unc.band(p_final),
        upset_explanation=json_safe(expl),
        model_market_gap=dis.get("market_model_gap"),
        directional_disagreement=dis.get("directional_disagreement"),
        agreement_bucket=dis.get("agreement_bucket"), model_dispersion=disp,
        market_signal=signal_label(dis),
        blend_policy=json_safe(pol.to_dict()),
        data_completeness=json_safe({"complete": not missing, "missing": missing,
                                     "notes": notes}),
        status="provisional" if missing else "published",
        status_note=("; ".join(missing) if missing else None) or
                    (None if gate.get("publish_pick") else
                     "mercado habilitado solo como probabilidad calibrada: la ventaja de "
                     "acierto no es estadisticamente significativa"),
        version=(prev.version + 1) if prev else 1,
        parent_prediction_id=prev.id if prev else None,
        revision_diff=json_safe({"antes": prev.ensemble_probability,
                                 "ahora": p_final, "motivo": reason}) if prev else None,
        model_version=f"{mv.name} {mv.version}" if mv else None,
        feature_version=mv.code_version if mv else None,
        sources=["mlb_stats_api", "baseball_savant"] + (["the_odds_api"] if p_market else []),
        prediction_timestamp=now, result=None)
    session.add(p)
    session.flush()
    session.add(PredictionSnapshot(prediction_id=p.id, features=json_safe({
        k: getattr(row, k) for k in ("h_sp_era_s", "a_sp_era_s", "h_off_obp_d30",
                                     "a_off_obp_d30", "h_bp_pitches_d3", "a_bp_pitches_d3",
                                     "park_runs_factor", "p_home_elo")
        if hasattr(row, k)}),
        inputs=json_safe({"p_model": p_model, "p_market": p_market, "p_final": p_final,
                          "dispersion": disp, "gate": gate})))
    return 1, (1 if prev else 0)
