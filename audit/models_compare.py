"""AUDITORIA DE MODELOS (§4, §5, §6, §9) + cuantificacion del leakage del abridor.

Por cada (temporada de prueba, mercado, variante) entrena UNA vez y evalua por
separado TODOS los algoritmos y las lineas base. Dos variantes:

  con_abridor  : todas las features (las publicadas hasta ahora)
  sin_abridor  : se quitan las 116 features del abridor, que en el historico
                 dependen de saber quien lanzo realmente. Es la cota LIMPIA.

La diferencia entre ambas es exactamente cuanto puede estar inflada la metrica.
Reanudable: guarda una parte por combinacion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from MLB.engine.elo import fit_params, run as run_elo         # noqa: E402
from MLB.engine.models import TARGET, dispersion, train       # noqa: E402
from shared.calibration import summary                        # noqa: E402
from shared.paths import MLB_PROCESSED_DIR                    # noqa: E402
from shared.timeutil import json_safe                         # noqa: E402

PARTS = Path(__file__).resolve().parent / "parts"
PARTS.mkdir(exist_ok=True, parents=True)
SEASONS_TEST = [2023, 2024, 2025, 2026]
MARKETS = ["moneyline", "f5_moneyline", "run_line"]
VARIANTS = ["con_abridor", "sin_abridor"]


def prep(X: pd.DataFrame, T: int):
    seasons = sorted(int(s) for s in X.season.unique())
    past = [s for s in seasons if s < T]
    cfg = fit_params(X[X.season < T], past[:-1], past[-1])
    elo = run_elo(X, cfg)
    Xe = X.merge(elo[["game_id", "p_home_elo"]], on="game_id", how="left")
    Xe["d_elo"] = Xe.p_home_elo - 0.5
    return Xe, cfg, past


def evaluate(T: int, market: str, variant: str) -> dict:
    X = pd.read_parquet(MLB_PROCESSED_DIR / "features.parquet")
    Xe, cfg, train_seasons = prep(X, T)
    tgt = TARGET[market]
    mm = train(Xe, market, train_seasons, exclude_starter=(variant == "sin_abridor"))
    te = Xe[(Xe.season == T) & Xe[tgt].notna()].copy()
    y = te[tgt].astype(int).values
    members = mm.member_probs(te)
    ens = mm.predict_proba(te)
    p_elo = te.p_home_elo.values
    rng = np.random.default_rng(7)
    base = float(y.mean())

    strat = {**{k: v for k, v in members.items()}, "ensemble": ens, "elo": p_elo,
             "siempre_local": np.full(len(y), 0.5 + 1e-6),
             "siempre_visitante": np.full(len(y), 0.5 - 1e-6),
             "aleatorio": rng.random(len(y))}
    out = {"season": T, "market": market, "variant": variant, "n": int(len(y)),
           "base_rate": base, "n_features": len(mm.features),
           "train_seasons": train_seasons,
           "elo_cfg": {"k": cfg.k, "home_adv": cfg.home_adv, "regress": cfg.regress},
           "oof_train": mm.oof, "weights": mm.weights,
           "mean_dispersion": float(np.mean(dispersion(members))),
           "strategies": {}}
    for name, p in strat.items():
        s = summary(y, p)
        if name == "siempre_local":
            s["accuracy"] = base
        elif name == "siempre_visitante":
            s["accuracy"] = 1 - base
        s["reliability"] = s["reliability"]
        out["strategies"][name] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=2.4)
    ap.add_argument("--markets", nargs="*", default=MARKETS)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS_TEST)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    todo = [(T, m, v) for T in a.seasons for m in a.markets for v in a.variants
            if not (PARTS / f"{T}_{m}_{v}.json").exists()]
    if a.status:
        print(json.dumps({"pendientes": todo, "hechas": sorted(p.stem for p in
                                                               PARTS.glob("*.json"))},
                         indent=1, default=str))
        return
    deadline = time.time() + a.minutes * 60
    for T, m, v in todo:
        if time.time() > deadline:
            print(f"tiempo agotado; quedan {len(todo)} combinaciones", flush=True)
            return
        print(f"→ {T} {m} {v}", flush=True)
        r = evaluate(T, m, v)
        (PARTS / f"{T}_{m}_{v}.json").write_text(json.dumps(json_safe(r), indent=1))
        print(f"  ok n={r['n']} feats={r['n_features']}", flush=True)
    print("todas las combinaciones completas", flush=True)


if __name__ == "__main__":
    main()
