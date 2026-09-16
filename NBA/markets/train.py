"""Entrena NBA_ML_v1, NBA_SPREAD_v1 y NBA_TOTAL_v1 con la configuracion elegida en
la investigacion walk-forward. Metricas publicadas = walk-forward, nunca train."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib                                       # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from NBA.markets.db import NbaModelVersion, init_db, session_scope  # noqa: E402
from NBA.markets.features import FEATURES_PARQUET, FEATURE_VERSION, numeric  # noqa: E402
from NBA.markets.gating import decide               # noqa: E402
from NBA.markets.research import TARGET, clf_algos, reg_algos  # noqa: E402
from NBA.markets.research_lines import oof_residuals  # noqa: E402
from shared.paths import NBA_MODELS_DIR, NBA_OUT_DIR, assert_writable  # noqa: E402
from shared.timeutil import json_safe, utcnow       # noqa: E402

VERSION = "v1"
NAME = {"moneyline": "NBA_ML", "spread": "NBA_SPREAD", "total": "NBA_TOTAL"}


def train_market(market, X, gate):
    research = json.loads((NBA_OUT_DIR / "research.json").read_text())["targets"][market]
    lines = json.loads((NBA_OUT_DIR / "research_lines.json").read_text())["targets"][market]
    cols, algo, target = research["features"], research["algoritmo_elegido"], TARGET[market]
    tr = X[X[target].notna() & (X.season_type == "Regular Season")]
    art = {"market": market, "algorithm": algo, "features": cols, "version": VERSION,
           "name": f"{NAME[market]}_{VERSION}", "trained_at": utcnow(), "n_train": int(len(tr)),
           "train_seasons": sorted(int(s) for s in tr.season.unique()), "feature_version": FEATURE_VERSION}
    if market == "moneyline":
        m = clf_algos()[algo](); m.fit(numeric(tr, cols), tr[target].astype(int).values)
        art["model"] = m
        art["calibration"] = {"method": lines["calibracion_elegida"], "params": lines["calibrador"],
                              "fitted_on": "walk-forward OOS 2019-2022"}
        art["threshold"] = gate["moneyline"]["threshold"]
    else:
        m = reg_algos()[algo](); m.fit(numeric(tr, cols), tr[target].values)
        res = oof_residuals(reg_algos()[algo], tr, cols, target)
        art.update({"model": m, "residuals": res, "res_std": float(res.std()),
                    "res_median": float(np.median(res)), "distribution": lines["distribucion_elegida"]})
    NBA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = assert_writable(NBA_MODELS_DIR / f"{market}_{VERSION}.joblib", "NBA")
    joblib.dump(art, path)
    with session_scope() as s:
        s.query(NbaModelVersion).filter(NbaModelVersion.market == market,
                                        NbaModelVersion.is_production.is_(True)).update({"is_production": False})
        s.add(NbaModelVersion(
            name=art["name"], market=market, version=VERSION, algorithm=algo,
            training_period=f"{art['train_seasons'][0]}-{art['train_seasons'][-1]} (temporada regular)",
            validation_period="walk-forward: seleccion 2019-2022, holdout 2023-2026",
            features=cols, calibration=json_safe(art.get("calibration")),
            metrics=json_safe({"gating": gate[market]}), artifact_path=str(path), is_production=True,
            notes=gate[market]["reason"], created_at=utcnow()))
    return {"name": art["name"], "algorithm": algo, "n_train": art["n_train"], "n_features": len(cols),
            "mode": gate[market]["mode"]}


def main():
    init_db()
    X = pd.read_parquet(FEATURES_PARQUET)
    gate = decide()
    out = {m: train_market(m, X, gate) for m in ("moneyline", "spread", "total")}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
