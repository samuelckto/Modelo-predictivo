"""AUDITORIA DEL MOTOR NFL (§15). SOLO LECTURA.

No se modifica nada. La conexion se abre con mode=ro, asi que es el propio sqlite
quien impide escribir. Antes de cualquier cambio futuro, `backup()` deja una copia
fechada.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                    # noqa: E402

import NFL.adapter as nfl                             # noqa: E402
from shared.calibration import summary                # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True, parents=True)


def _j(v, default=None):
    if v in (None, ""):
        return default
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:                                  # noqa: BLE001
        return default


def backup(dest: Path | None = None) -> dict:
    """Copia fechada de la base NFL. Se hace ANTES de tocar nada, nunca despues."""
    src = nfl.db_path()
    if not src.exists():
        return {"status": "no existe la base NFL", "path": str(src)}
    dest = dest or (src.parent / "backups")
    dest.mkdir(exist_ok=True, parents=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    tgt = dest / f"nflpred_{stamp}.sqlite3"
    shutil.copy2(src, tgt)
    h = hashlib.sha256(tgt.read_bytes()).hexdigest()[:32]
    return {"status": "ok", "origen": str(src), "copia": str(tgt),
            "bytes": tgt.stat().st_size, "sha256_32": h}


def audit() -> dict:
    ok, msg = nfl.available()
    if not ok:
        return {"disponible": False, "motivo": msg}
    out = {"disponible": True, "solo_lectura": nfl.read_only_check()}
    with nfl._conn() as c:
        c.row_factory = sqlite3.Row
        # --- inventario
        out["tablas"] = {r[0]: c.execute(f"select count(*) from {r[0]}").fetchone()[0]
                         for r in c.execute("select name from sqlite_master "
                                            "where type='table' order by name")}
        # --- modelo en produccion
        mv = c.execute("select name,version,algorithm,created_at,train_seasons,metrics,"
                       "blend_policy from model_versions where is_production=1 "
                       "order by id desc limit 1").fetchone()
        out["modelo_produccion"] = (
            {"name": mv["name"], "version": mv["version"], "algorithm": mv["algorithm"],
             "created_at": mv["created_at"], "train_seasons": mv["train_seasons"],
             "metrics": _j(mv["metrics"], {}), "blend_policy": _j(mv["blend_policy"], {})}
            if mv else None)
        # --- backtest almacenado
        bt = c.execute("select id,label,created_at,seasons,summary from backtest_runs "
                       "order by id desc limit 1").fetchone()
        out["backtest"] = ({"id": bt["id"], "label": bt["label"],
                            "created_at": bt["created_at"], "seasons": bt["seasons"],
                            "summary": _j(bt["summary"], {})} if bt else None)
        # --- calibracion de las predicciones ya evaluadas
        rows = c.execute("select pick_prob, correct, home_prob_final, home_prob_market, "
                         "home_prob_ensemble, points, points_earned, season, week "
                         "from predictions where correct is not null and is_active=1").fetchall()
        if rows:
            p = np.array([r["pick_prob"] for r in rows], dtype=float)
            y = np.array([r["correct"] for r in rows], dtype=float)
            s = summary(y, p)
            out["calibracion_en_vivo"] = {k: s[k] for k in
                                          ("n", "accuracy", "log_loss", "brier", "ece")}
            out["calibracion_en_vivo"]["reliability"] = s["reliability"]
            pts = np.array([r["points_earned"] or 0 for r in rows], dtype=float)
            mx = np.array([r["points"] or 0 for r in rows], dtype=float)
            out["fantasy"] = {"puntos": float(pts.sum()), "maximo": float(mx.sum()),
                              "pct": float(pts.sum() / mx.sum()) if mx.sum() else None}
        else:
            out["calibracion_en_vivo"] = {"n": 0,
                                          "nota": "no hay predicciones NFL ya evaluadas"}
        # --- leakage: data_cutoff frente a kickoff
        leak = c.execute("""
            select count(*) tot,
                   sum(case when data_cutoff_utc is null then 1 else 0 end) sin_cutoff,
                   sum(case when data_cutoff_utc >= kickoff_utc then 1 else 0 end) posterior,
                   sum(case when predicted_at >= kickoff_utc then 1 else 0 end) pred_posterior
            from predictions where is_active=1 and kickoff_utc is not null""").fetchone()
        out["leakage_predicciones"] = {
            "predicciones": leak["tot"], "sin_data_cutoff": leak["sin_cutoff"],
            "cutoff_posterior_al_kickoff": leak["posterior"],
            "prediccion_posterior_al_kickoff": leak["pred_posterior"],
            "ok": (leak["posterior"] or 0) == 0 and (leak["pred_posterior"] or 0) == 0}
        # --- cuotas usadas: ninguna posterior a la prediccion
        odds = c.execute("""
            select count(*) tot, sum(case when o.captured_at > p.predicted_at then 1 else 0 end) post
            from predictions p join odds_snapshots o on o.game_id = p.game_id
            where p.is_active=1""").fetchone()
        out["odds_posteriores_a_la_prediccion"] = {
            "snapshots_asociados": odds["tot"] or 0,
            "capturados_despues": odds["post"] or 0,
            "nota": ("existen snapshots posteriores en la base; lo que importa es que el "
                     "motor filtre por captured_at < predicted_at al construir features")}
        # --- politica de combinacion
        pol = c.execute("select blend_policy from predictions where blend_policy is not null "
                        "order by id desc limit 1").fetchone()
        out["blend_policy"] = _j(pol["blend_policy"], {}) if pol else None
        # --- fuentes
        out["fuentes"] = [dict(r) for r in c.execute(
            "select source, max(fetched_at) ultima, count(*) descargas, "
            "sum(case when status!='ok' then 1 else 0 end) errores "
            "from data_sources_log group by source order by ultima desc")]
    return out


if __name__ == "__main__":
    r = {"backup": backup(), "auditoria": audit()}
    (OUT / "nfl_audit.json").write_text(json.dumps(r, indent=1, default=str))
    a = r["auditoria"]
    print("BACKUP:", json.dumps(r["backup"], indent=1, default=str))
    if not a.get("disponible"):
        print(a)
        raise SystemExit(0)
    print("\nSOLO LECTURA:", a["solo_lectura"])
    print("\nMODELO EN PRODUCCION:", (a["modelo_produccion"] or {}).get("name"),
          (a["modelo_produccion"] or {}).get("version"))
    bp = (a["modelo_produccion"] or {}).get("blend_policy") or {}
    print("  politica de mercado:", bp.get("description"), "|", (bp.get("rationale") or "")[:120])
    print("\nLEAKAGE:", json.dumps(a["leakage_predicciones"], indent=1))
    print("\nCALIBRACION EN VIVO:", json.dumps(
        {k: v for k, v in a["calibracion_en_vivo"].items() if k != "reliability"}, indent=1))
    st = ((a.get("backtest") or {}).get("summary") or {}).get("strategies") or {}
    if st:
        print("\nBACKTEST NFL ALMACENADO:")
        print(f"{'estrategia':<16}{'acc':>9}{'logloss':>10}{'brier':>9}{'puntos':>9}")
        for k, v in sorted(st.items(), key=lambda kv: -(kv[1].get("accuracy") or 0)):
            print(f"{k:<16}{(v.get('accuracy') or 0)*100:>8.2f}%{v.get('log_loss',0):>10.4f}"
                  f"{v.get('brier',0):>9.4f}"
                  f"{(v.get('points_pct') or 0)*100:>8.2f}%")
    print("\nFUENTES:", json.dumps(a["fuentes"], indent=1, default=str)[:600])
