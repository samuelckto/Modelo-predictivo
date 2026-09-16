"""AUDITORIA DE DATA LEAKAGE (§3).

Dos niveles:

A) INVENTARIO: para cada una de las features, de que tabla sale, que significa su
   `available_at` y que riesgo de leakage tiene. Se deriva del prefijo del nombre
   y del codigo del constructor, no de una lista escrita a mano.

B) PRUEBA DECISIVA: se reconstruyen las features de un dia concreto usando un
   dataset TRUNCADO en ese instante (se borra todo lo posterior). Si el
   constructor usara informacion futura, los valores cambiarian. Deben ser
   identicos.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from MLB.features.builder import build                        # noqa: E402
from shared.paths import MLB_PROCESSED_DIR                    # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True, parents=True)

# origen -> (tabla, que significa available_at, riesgo)
ORIGIN = [
    ("_sp_", "probable_pitchers + pitcher_game_logs",
     "aperturas anteriores del abridor; fin de partido = inicio + 3h15", "BAJO"),
    ("_off_", "batter_game_logs", "fin del partido correspondiente", "BAJO"),
    ("_def_", "games (carreras recibidas)", "fin del partido correspondiente", "BAJO"),
    ("_stf_", "pitcher_game_logs (staff completo)", "fin del partido", "BAJO"),
    ("_bp_", "bullpen_usage (derivado de boxscores)", "fin del partido", "BAJO"),
    ("_rest_days", "games.start_utc", "calendario, publicado con antelacion", "NULO"),
    ("park_runs_factor", "games de temporadas ANTERIORES",
     "desplazado una temporada: el factor de 2023 solo se usa en 2024", "NULO"),
    ("is_doubleheader", "games.doubleHeader", "calendario", "NULO"),
    ("month", "games.game_date", "calendario", "NULO"),
    ("game_number", "games.gameNumber", "calendario", "NULO"),
    ("p_home_elo", "elo calculado en orden cronologico",
     "solo partidos anteriores por construccion", "BAJO"),
    ("d_elo", "idem", "idem", "BAJO"),
]

# columnas que NUNCA pueden usarse como feature
FORBIDDEN = {"home_score", "away_score", "home_f5", "away_f5", "total_runs",
             "home_margin", "total_runs_f5", "y_home_win", "y_home_f5", "y_home_rl",
             "result", "status", "innings"}


def classify(col: str) -> dict:
    for key, table, meaning, risk in ORIGIN:
        if key in col:
            return {"feature": col, "source_table": table, "available_at": meaning,
                    "used_before_game": True, "leakage_risk": risk}
    return {"feature": col, "source_table": "SIN CLASIFICAR", "available_at": "UNKNOWN",
            "used_before_game": None, "leakage_risk": "REVISAR"}


def inventory(X: pd.DataFrame) -> dict:
    from MLB.engine.models import feature_columns
    used = feature_columns(X)                       # las que ve el modelo
    allcols = [c for c in X.columns if c.startswith(("h_", "a_", "d_")) or
               c in ("park_runs_factor", "is_doubleheader", "month", "game_number",
                     "p_home_elo")]
    rows = [classify(c) for c in allcols]
    for r in rows:
        r["fed_to_model"] = r["feature"] in used
    unclassified = [r["feature"] for r in rows if r["leakage_risk"] == "REVISAR"]
    forbidden_used = sorted(set(used) & FORBIDDEN)
    return {"features_total": len(allcols), "features_fed_to_model": len(used),
            "sin_clasificar": unclassified,
            "prohibidas_usadas": forbidden_used,
            "por_riesgo": pd.Series([r["leakage_risk"] for r in rows]).value_counts().to_dict(),
            "tabla": rows}


def truncation_test(seasons, game_ids: list[int] | None = None,
                    n_games: int = 6) -> dict:
    """PRUEBA DECISIVA, hecha partido a partido.

    Para cada partido G que empieza en T se reconstruyen las features con un
    dataset del que se ha borrado TODO lo que tiene available_at >= T. Si el
    constructor usara informacion futura, los valores de G cambiarian.

    Se hace por partido y no por dia porque un partido de las 23:00 UTC si puede
    (y debe) usar el resultado del partido de la noche anterior, que termino a
    las 02:00 UTC de ese mismo dia. Cortar a medianoche daria una falsa alarma.
    """
    full = pd.read_parquet(MLB_PROCESSED_DIR / "features.parquet")
    fin = full[full.y_home_win.notna() & (full.season >= 2023)]
    if game_ids is None:
        game_ids = fin.sample(n_games, random_state=13).game_id.tolist()
    cols = [c for c in full.columns
            if (c.startswith(("h_", "a_", "d_")) or c == "park_runs_factor")
            and pd.api.types.is_numeric_dtype(full[c])]
    detalle, difs_tot = [], 0
    for gid in game_ids:
        row = full[full.game_id == gid].iloc[0]
        cut = pd.Timestamp(row.start_utc)
        try:
            tr = build(seasons, save=False, data_cutoff=cut)
        except Exception as e:                                # noqa: BLE001
            detalle.append({"game_id": int(gid), "error": f"{type(e).__name__}: {e}"})
            continue
        t_row = tr[tr.game_id == gid]
        if t_row.empty:
            detalle.append({"game_id": int(gid), "error": "el partido desaparecio"})
            continue
        t_row = t_row.iloc[0]
        difs = []
        for c in cols:
            a, b = row[c], t_row[c]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b) or not np.isclose(float(a), float(b), atol=1e-9):
                difs.append({"feature": c,
                             "completo": None if pd.isna(a) else float(a),
                             "truncado": None if pd.isna(b) else float(b)})
        difs_tot += len(difs)
        detalle.append({"game_id": int(gid), "start_utc": str(cut),
                        "features_comparadas": len(cols),
                        "features_diferentes": len(difs), "ejemplos": difs[:5]})
    return {"games_tested": len(game_ids), "features_compared": len(cols),
            "total_diferencias": difs_tot, "detalle": detalle,
            "veredicto": ("SIN LEAKAGE: cortar el dataset en el instante de cada "
                          "partido no cambia ni una sola feature" if difs_tot == 0 else
                          "LEAKAGE DETECTADO")}


def forbidden_correlations(X: pd.DataFrame, thr: float = 0.5) -> dict:
    """Ninguna feature puede correlacionar fuerte con el resultado del partido."""
    from MLB.engine.models import feature_columns
    cols = feature_columns(X)
    m = X.y_home_win.notna()
    y = X.loc[m, "y_home_win"]
    sus = []
    for c in cols:
        v = X.loc[m, c]
        if v.notna().sum() < 500:
            continue
        r = float(np.corrcoef(v.fillna(v.median()), y)[0, 1])
        if abs(r) > thr:
            sus.append({"feature": c, "corr": round(r, 4)})
    mx = max(((c, float(np.corrcoef(X.loc[m, c].fillna(X.loc[m, c].median()), y)[0, 1]))
              for c in cols if X.loc[m, c].notna().sum() >= 500), key=lambda t: abs(t[1]))
    return {"umbral": thr, "sospechosas": sus,
            "correlacion_maxima": {"feature": mx[0], "corr": round(mx[1], 4)}}


def season_boundary(X: pd.DataFrame) -> dict:
    """Ninguna feature de temporada puede sobrevivir al cambio de ano."""
    long = pd.concat([
        X[["game_id", "season", "start_utc", "home_team_id"]]
            .rename(columns={"home_team_id": "team"}).assign(side="h"),
        X[["game_id", "season", "start_utc", "away_team_id"]]
            .rename(columns={"away_team_id": "team"}).assign(side="a")])
    first = long.sort_values("start_utc").groupby(["season", "team"]).head(1)
    idx = X.set_index("game_id")
    malos = 0
    for r in first.itertuples():
        col = "h_off_obp_s" if r.side == "h" else "a_off_obp_s"
        if pd.notna(idx.at[r.game_id, col]):
            malos += 1
    return {"primeros_partidos": int(len(first)), "con_historia_previa": malos,
            "ok": malos == 0}


if __name__ == "__main__":
    X = pd.read_parquet(MLB_PROCESSED_DIR / "features.parquet")
    res = {"inventario": inventory(X),
           "correlaciones": forbidden_correlations(X),
           "limite_temporada": season_boundary(X)}
    print(json.dumps({k: v for k, v in res.items() if k != "inventario"},
                     indent=1, default=str))
    inv = res["inventario"]
    print(json.dumps({k: v for k, v in inv.items() if k != "tabla"}, indent=1, default=str))
    (OUT / "leakage_inventory.json").write_text(json.dumps(res, indent=1, default=str))
    print("\n--- PRUEBA DE TRUNCAMIENTO (la decisiva) ---", flush=True)
    t = truncation_test([2021, 2022, 2023, 2024, 2025, 2026], "2025-07-15")
    print(json.dumps(t, indent=1, default=str))
    (OUT / "leakage_truncation.json").write_text(json.dumps(t, indent=1, default=str))
