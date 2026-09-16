"""Anti-leakage de la capa de valor.

La capa nueva toca cuotas y resultados, que son justo los dos datos capaces de
contaminar una prediccion. Estas pruebas fijan la frontera:

    la cuota puede usarse para EVALUAR (ROI, discrepancia, comparacion)
    la cuota NO puede entrar como feature de un modelo pregame

y comprueban ademas el corte point-in-time del cruce de cuotas: una cuota
posterior al momento de la prediccion es informacion del futuro.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from shared.paths import ROOT

# Palabras que delatan una cuota o una linea de mercado dentro de un modulo de
# features. Si aparecen ahi, hay que mirarlo.
MERCADO = re.compile(
    r"\b(odds|moneyline|implied_prob|closing|price_american|price_decimal|"
    r"market_probability|vegas|spread_line|total_line)\b", re.I)

# Modulos que construyen features pregame de cada deporte.
FEATURES = [
    ROOT / "MLB" / "features" / "builder.py",
    ROOT / "SOCCER" / "features" / "build.py",
    ROOT / "NBA" / "markets" / "features.py",
    ROOT / "TENIS" / "markets" / "features.py",
    ROOT / "NFL" / "markets" / "data.py",
]


@pytest.mark.parametrize("f", [p for p in FEATURES if p.exists()],
                         ids=lambda p: p.parent.parent.name)
def test_ninguna_feature_pregame_lee_cuotas(f: Path):
    """Ninguna linea EJECUTABLE de un constructor de features menciona cuotas.

    Se ignoran comentarios y docstrings: NFL/markets/data.py declara a proposito
    una lista de columnas de mercado PROHIBIDAS, y eso es exactamente lo
    contrario de un problema.
    """
    lineas = f.read_text(encoding="utf-8", errors="ignore").splitlines()
    sospechosas, en_doc = [], False
    for i, ln in enumerate(lineas, 1):
        s = ln.strip()
        if s.count('"""') == 1 or s.count("'''") == 1:
            en_doc = not en_doc
            continue
        if en_doc or s.startswith("#") or not s:
            continue
        codigo = s.split("#", 1)[0]
        if MERCADO.search(codigo) and "PROHIB" not in codigo.upper():
            sospechosas.append(f"{i}: {s[:100]}")
    # NFL declara MARKET_COLS como lista negra: es una declaracion, no un uso.
    if f.name == "data.py":
        sospechosas = [x for x in sospechosas if "MARKET_COLS" not in x
                       and not re.match(r"^\d+: \"", x)]
    assert not sospechosas, f"{f} podria estar leyendo mercado:\n" + "\n".join(sospechosas)


def test_el_ledger_solo_cruza_cuotas_anteriores_a_la_prediccion():
    """El cruce de cuotas de MLB y NFL lleva un corte point-in-time explicito."""
    src = (ROOT / "shared" / "ledger.py").read_text(encoding="utf-8")
    # Debe aparecer la comparacion available_at <= prediction_timestamp
    assert src.count('o.get("available_at") or ""') >= 2
    assert src.count('p.get("prediction_timestamp")') >= 2


def test_el_ledger_no_escribe():
    """Abre todas las bases en modo solo lectura."""
    src = (ROOT / "shared" / "ledger.py").read_text(encoding="utf-8")
    assert "mode=ro" in src
    for prohibido in ("INSERT", "UPDATE ", "DELETE", "DROP", "CREATE TABLE"):
        assert prohibido not in src.upper(), prohibido


def test_la_capa_de_valor_no_escribe_en_ninguna_base():
    for m in ("roi", "discrepancy", "tracking", "calibrators"):
        src = (ROOT / "shared" / f"{m}.py").read_text(encoding="utf-8")
        for prohibido in ("INSERT INTO", "UPDATE ", "DELETE FROM", "session.add",
                          "commit()"):
            assert prohibido not in src, (m, prohibido)


def test_el_modulo_chat_no_escribe():
    for f in (ROOT / "CHAT").glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for prohibido in ("INSERT INTO", "UPDATE ", "DELETE FROM", "session.add",
                          "commit()", "to_parquet", "joblib.dump"):
            assert prohibido not in src, (f.name, prohibido)


def test_la_cuota_tiene_que_ser_de_la_misma_linea():
    """Bug real, partido 824063 (KC-AZ): la prediccion era 'AZ +1.5' y el mercado
    solo tenia AZ -1.5. Comparar sin mirar la linea daba +30 pp de desacuerdo
    contra un precio que no existia, y eran los dos gaps mas grandes del panel."""
    from shared.ledger import _misma_linea
    assert _misma_linea({"line": 1.5}, {"line": 1.5})
    assert not _misma_linea({"line": -1.5}, {"line": 1.5})
    assert not _misma_linea({"line": None}, {"line": 1.5})
    assert not _misma_linea({"line": 8.5}, {"line": None})
    assert _misma_linea({"line": None}, {"line": None})       # moneyline


def test_las_cuotas_de_evaluacion_van_marcadas_como_tales():
    """El registro de discrepancia guarda la procedencia y el momento de la cuota."""
    from shared import discrepancy
    r = discrepancy.evaluar(model_cal=0.6, market=0.5, odds_decimal=2.0,
                            market_source="pinnacle", fetched_at="2026-09-08 10:00:00",
                            event_time="2026-09-08 18:00:00", overround=0.04)
    assert r["market_source"] == "pinnacle"
    assert r["fetched_at"] == "2026-09-08 10:00:00"
    assert r["event_time"] == "2026-09-08 18:00:00"
    # y la cuota nunca se convierte en la probabilidad del modelo
    assert r["MODEL_PROBABILITY_CALIBRATED"] == 0.6


def test_la_probabilidad_publicada_no_sustituye_a_la_del_modelo():
    """MLB publica una mezcla 50/50 con el mercado. La discrepancia debe medirse
    contra la probabilidad del MODELO, no contra la mezcla: si no, el mercado
    estaria en los dos lados de la resta y el gap se encogeria solo."""
    from shared import ledger
    src = (ROOT / "dashboard" / "backend" / "main.py").read_text(encoding="utf-8")
    i = src.index("def value_model_vs_market")
    bloque = src[i:i + 2500]
    assert 'model_cal=r["model_probability_calibrated"]' in bloque
    assert 'model_cal=r["published_probability"]' not in bloque
    assert "published_probability" in ledger.CAMPOS
    assert "model_probability_calibrated" in ledger.CAMPOS
