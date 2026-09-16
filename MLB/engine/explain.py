"""Explicacion por partido, construida con los numeros reales de ese partido.

No hay plantillas fijas: cada frase solo aparece si el dato existe y supera un
umbral. Si no hay nada que decir, se dice que no hay nada que destacar.
"""
from __future__ import annotations

import numpy as np

# (etiqueta en lenguaje sencillo, umbral, "menor es mejor")
FMT = {
    "d_sp_era_s": ("su abridor permite menos carreras esta temporada", 0.7, True),
    "d_sp_kbb_s": ("su abridor domina mas (ponches vs bases por bolas)", 0.03, False),
    "d_sp_xwoba_s": ("su abridor permite contacto mas debil", 0.020, True),
    "d_sp_csw_s": ("su abridor consigue mas strikes", 0.02, False),
    "d_off_obp_d30": ("su ofensiva se embasa mas en el ultimo mes", 0.020, False),
    "d_off_rpg_d30": ("su ofensiva anota mas carreras en el ultimo mes", 0.6, False),
    "d_stf_era_d30": ("su pitcheo en conjunto viene mejor en el ultimo mes", 0.6, True),
    "d_bp_pitches_d3": ("su bullpen llega mas descansado", 25, True),
    "d_rest_days": ("llega con mas dias de descanso", 1.0, False),
}


def drivers(row, home: str, away: str, top: int = 5) -> list[dict]:
    """Factores con ventaja clara, ordenados por magnitud relativa al umbral."""
    out = []
    for col, (name, thr, lower_better) in FMT.items():
        v = getattr(row, col, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if abs(v) < thr:
            continue
        favors = away if (v > 0) == lower_better else home
        fuerza = abs(v) / thr
        grado = "mucho " if fuerza >= 2 else ""
        out.append({"feature": col, "label": name, "value": float(v),
                    "text": f"{favors}: {name} ({grado}mejor que el rival)",
                    "favors": favors, "strength": fuerza})
    return sorted(out, key=lambda d: -d["strength"])[:top]


def narrative(row, home, away, pick, p_final, p_elo, disp, missing, market_p=None) -> dict:
    ds = drivers(row, home, away)
    conf = abs(p_final - 0.5) * 2
    partes = []
    rival = away if pick == home else home
    if conf < 0.06:
        partes.append(f"Partido casi parejo: {pick} tiene {p_final*100:.0f}% y "
                      f"{rival} {100-p_final*100:.0f}%. Es practicamente un volado.")
    elif conf < 0.2:
        partes.append(f"{pick} es favorito ligero ({p_final*100:.0f}%). "
                      f"Aun asi, {rival} gana {100-p_final*100:.0f} de cada 100 veces.")
    else:
        partes.append(f"{pick} es favorito claro ({p_final*100:.0f}%).")
    a_favor = [d for d in ds if d["favors"] == pick]
    en_contra = [d for d in ds if d["favors"] != pick]
    if a_favor:
        partes.append("Lo que ayuda a " + pick + ": " +
                      "; ".join(d["label"] for d in a_favor[:3]) + ".")
    if en_contra:
        partes.append("Lo que juega en contra: " + rival + " " +
                      "; ".join(d["label"] for d in en_contra[:2]) + ".")
    if not ds:
        partes.append("Ningun equipo destaca claramente en pitcheo, bateo o descanso; "
                      "la ventaja sale de la suma de detalles pequenos.")
    if p_elo is not None and abs(p_final - p_elo) > 0.08:
        partes.append(f"El rating de equipos (Elo) da {p_elo*100:.0f}%: no coincide del "
                      f"todo con el modelo, senal de menos certeza.")
    if disp and disp > 0.04:
        partes.append("Los distintos algoritmos no se ponen de acuerdo: menos certeza.")
    if market_p is not None and abs(p_final - market_p) > 0.06:
        quien = "mas" if p_final > market_p else "menos"
        partes.append(f"Las casas de apuestas le dan {market_p*100:.0f}% a {pick}: "
                      f"el modelo es {quien} optimista que ellas.")
    if missing:
        partes.append("Falta informacion: " + "; ".join(missing) + ".")
    return {"summary": " ".join(partes), "drivers": ds,
            "verdict": (f"Ojo: hay mas incertidumbre de lo normal. Puede fallar "
                        f"aunque el pick sea correcto."
                        if (conf < 0.08 or (disp or 0) > 0.05 or missing)
                        else "Nada raro: la incertidumbre es la normal de un partido de MLB.")}


def narrative_total(row, home, away, pick, p_final, info, missing, market_p=None) -> dict:
    """Explicacion del Over/Under en lenguaje sencillo."""
    mu, linea = info["proyeccion"], info["linea"]
    lado = "mas" if pick.lower().startswith("over") else "menos"
    origen = "la de las casas de apuestas" if info["linea_origen"] == "mercado" else "de referencia, no hay cuotas"
    partes = [f"El modelo espera unas {mu:.1f} carreras entre los dos equipos. "
              f"Con la linea en {linea:g} ({origen}) ve {lado} probable que se pase: "
              f"{pick} con {p_final*100:.0f}%."]
    ds = []
    for col, (name, thr, lower_better) in FMT.items():
        if not col.startswith("d_off") and not col.startswith("d_sp") and not col.startswith("d_stf"):
            continue
        v = getattr(row, col, None)
        if v is None or (isinstance(v, float) and np.isnan(v)) or abs(v) < thr:
            continue
        favors = away if (v > 0) == lower_better else home
        ds.append({"feature": col, "label": name, "value": float(v),
                   "text": f"{favors}: {name}", "favors": favors, "strength": abs(v) / thr})
    pf = getattr(row, "park_runs_factor", None)
    if pf is not None and not (isinstance(pf, float) and np.isnan(pf)) and abs(pf - 1) > 0.05:
        partes.append("El estadio favorece " + ("las carreras." if pf > 1 else "el pitcheo."))
    if market_p is not None and abs(p_final - market_p) > 0.06:
        partes.append(f"Las casas de apuestas dan {market_p*100:.0f}% al {pick.split()[0]}.")
    partes.append("Importante: en las pruebas el modelo no proyecto carreras mejor que la "
                  "media historica, por eso el Over/Under es solo informativo, no un pick.")
    if missing:
        partes.append("Falta informacion: " + "; ".join(missing) + ".")
    return {"summary": " ".join(partes), "drivers": ds[:4], "total": info,
            "verdict": "Over/Under informativo: sin ventaja demostrada."}
