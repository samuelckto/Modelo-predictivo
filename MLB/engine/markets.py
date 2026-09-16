"""Mercados MLB que se muestran y se generan.

`f5_moneyline` sigue medido en el backtest y sus predicciones antiguas se
conservan (nunca se borran), pero se dejo de generar y de mostrar a peticion
del usuario: en su lugar se publica Over/Under de carreras (`total`).
"""
HIDDEN = {"f5_moneyline"}

LABEL = {"moneyline": "Moneyline", "total": "Over/Under carreras",
         "run_line": "Run line", "f5_moneyline": "Primeras 5 entradas"}

ORDER = ["moneyline", "total", "run_line"]
