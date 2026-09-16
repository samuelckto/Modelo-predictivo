"""Entrena los modelos de produccion MLB y los registra con su version.

Se entrena con TODAS las temporadas disponibles hasta la actual. Las metricas
que se publican NO salen de aqui (seria entrenar y medir en lo mismo): salen del
backtest walk-forward, que es lo unico que se reporta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib                                                  # noqa: E402
import pandas as pd                                            # noqa: E402

from MLB.database.models import ModelVersion                   # noqa: E402
from MLB.database.session import init_db, session_scope        # noqa: E402
from MLB.engine.elo import fit_params, run as run_elo          # noqa: E402
from MLB.engine.gating import decide                           # noqa: E402
from MLB.engine.models import feature_columns, train           # noqa: E402
from MLB.features.builder import FEATURE_VERSION               # noqa: E402
from shared.paths import MLB_BACKTEST_DIR, MLB_MODELS_DIR, MLB_PROCESSED_DIR  # noqa: E402
from shared.timeutil import json_safe, utcnow                             # noqa: E402

VERSION = "v2"

# Decision de la auditoria (§3, §5, §30):
#  * Los modelos de produccion se entrenan SIN las features del abridor, que es
#    exactamente la variante con la que se midio el rendimiento publicado.
#    Ship what you measured.
#  * moneyline NO usa modelo de ML: usa Elo, que fue el mejor fuera de muestra
#    (55.82 % frente a 55.38 % del mejor ML) y ademas no depende del abridor.
EXCLUDE_STARTER = True
ELO_MARKETS = {"moneyline"}


def add_elo(X: pd.DataFrame):
    seasons = sorted(X.season.unique())
    cfg = fit_params(X, seasons[:-2], seasons[-2])
    elo = run_elo(X, cfg)
    Xe = X.merge(elo[["game_id", "p_home_elo"]], on="game_id", how="left")
    Xe["d_elo"] = Xe.p_home_elo - 0.5
    return Xe, cfg, elo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=VERSION)
    ap.add_argument("--markets", nargs="*", default=None)
    a = ap.parse_args()
    init_db()
    X = pd.read_parquet(MLB_PROCESSED_DIR / "features.parquet")
    gate = decide()
    markets = a.markets or [k for k, v in gate.items()
                            if isinstance(v, dict) and v.get("enabled")]
    markets = [m for m in markets if m not in ELO_MARKETS]   # moneyline -> Elo
    Xe, elo_cfg, elo = add_elo(X)
    joblib.dump({"cfg": elo_cfg, "elo": elo}, MLB_MODELS_DIR / f"elo_{a.version}.joblib")
    bt = json.loads((MLB_BACKTEST_DIR / "walkforward_v1.json").read_text())
    # registro del Elo como modelo de produccion de moneyline
    with session_scope() as s:
        s.query(ModelVersion).filter(ModelVersion.market == "moneyline",
                                     ModelVersion.is_production.is_(True)) \
            .update({"is_production": False})
        g_ml = gate.get("moneyline", {})
        s.add(ModelVersion(
            name="MLB_model_moneyline_elo", version=a.version, market="moneyline",
            training_period=f"{sorted(int(x) for x in Xe.season.unique())[0]}-"
                            f"{sorted(int(x) for x in Xe.season.unique())[-1]}",
            validation_period="rejilla k/ventaja/regresion validada en T-1",
            test_period=str(json_safe(g_ml.get("n"))),
            features=["rating Elo por equipo", "ventaja de local", "margen de carreras"],
            hyperparameters=json_safe({"k": elo_cfg.k, "home_adv": elo_cfg.home_adv,
                                       "regress": elo_cfg.regress, "mov": elo_cfg.mov}),
            metrics=json_safe({"gating": g_ml}),
            artifact_path=str(MLB_MODELS_DIR / f"elo_{a.version}.joblib"),
            code_version=FEATURE_VERSION, is_production=True,
            notes=("Elo elegido por la auditoria: mejor accuracy y log loss fuera de "
                   "muestra que cualquier variante de ML, y sin dependencia del "
                   "abridor. La simplicidad gana cuando el desempeno es igual o mejor."),
            created_at=utcnow()))
    seasons = sorted(int(s) for s in Xe.season.unique())
    out = {}
    for m in markets:
        mm = train(Xe, m, seasons, exclude_starter=EXCLUDE_STARTER)
        path = MLB_MODELS_DIR / f"{m}_{a.version}.joblib"
        joblib.dump(mm, path)
        with session_scope() as s:
            s.query(ModelVersion).filter(ModelVersion.market == m,
                                         ModelVersion.is_production.is_(True)) \
                .update({"is_production": False})
            mv = ModelVersion(
                name=f"MLB_model_{m}", version=a.version, market=m,
                training_period=f"{seasons[0]}-{seasons[-1]}",
                validation_period="walk-forward: cada T se valida en T-1",
                test_period=str(json_safe(bt["summary"].get(m, {}).get("seasons", []))),
                features=json_safe(mm.features),
                hyperparameters=json_safe({"weights": mm.weights, "elo": vars(elo_cfg)}),
                metrics=json_safe({"backtest_walkforward": bt["summary"].get(m),
                                   "gating": gate.get(m), "oof_train": mm.oof}),
                blend_policy=None, artifact_path=str(path),
                code_version=FEATURE_VERSION, is_production=True,
                notes=("metricas publicadas = backtest walk-forward, nunca el entrenamiento. "
                       + (gate.get(m, {}).get("reason", ""))),
                created_at=utcnow())
            s.add(mv); s.flush()
            out[m] = {"model_version_id": mv.id, "n_train": mm.n_train,
                      "features": len(mm.features), "weights": mm.weights,
                      "publish_pick": gate.get(m, {}).get("publish_pick"),
                      "algoritmos_usados": sorted(mm.members),
                      "algoritmos_descartados": mm.degraded or None}
    out["_blocked"] = [k for k, v in gate.items()
                       if isinstance(v, dict) and not v.get("enabled")]
    out["_elo"] = {"k": elo_cfg.k, "home_adv": elo_cfg.home_adv, "regress": elo_cfg.regress}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
