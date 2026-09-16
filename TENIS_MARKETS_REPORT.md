# Informe · Módulo TENIS del Sports Prediction Center

Fecha: 2026-09-08. Circuitos ATP y WTA, mercados ganador / total de juegos /
hándicap de juegos. NFL, MLB y NBA sin cambios (hashes verificados en test).
Tests: **167 passed** (137 previos + 30 nuevos de tenis).

## Veredicto

| circuito · mercado | estado | evidencia (holdout 2020-2026) |
|---|---|---|
| ATP · Ganador | **PICK** (umbral 0.55-0.65 según año) | 12,967 picks, **69.1 %** de acierto (z=43.4 vs 50 %) y **por encima del favorito Elo** en los mismos partidos (67.8 %, z=+3.04). ECE 0.012 |
| WTA · Ganador | **PICK** (umbral 0.50-0.55) | 14,059 picks, **67.3 %** (z=40.9) y por encima del favorito Elo (66.1 %, z=+2.91). ECE 0.0095 |
| ATP · Total de juegos | PROYECCIÓN | MAE 6.04 vs 6.68 de la media histórica; P(over) mal calibrada (ECE 0.122) |
| ATP · Hándicap de juegos | PROYECCIÓN | MAE del margen 4.26 vs 4.92 (suponer empate); P(cubre) ECE 0.042 |
| WTA · Total de juegos | **BLOQUEADO** | MAE 5.44 **peor** que la media histórica (4.82) |
| WTA · Hándicap de juegos | PROYECCIÓN | MAE 4.65 vs 5.34; P(cubre) ECE 0.037 |

Es el primer deporte del sistema donde el modelo **supera a su baseline fuerte
con significancia estadística**. Aviso que se mantiene: nada de esto está medido
contra las cuotas; no hay historial de líneas de tenis.

---

## 1. Arquitectura
```
TENIS/
  markets/  db.py ingest.py features.py pointmodel.py research.py research_lines.py
            gating.py train.py predict.py odds.py pipeline.py provider.py
  markets/out/  features.parquet research.json research_lines.json preds_{ATP,WTA}.parquet lines_*.parquet
  database/tenis_markets.sqlite3   (matches, players, schedule, model_versions,
                                    predictions, odds_snapshots, pipeline_runs, source_log)
  models/  ATP_winner_v1.joblib  WTA_winner_v1.joblib
  tests/test_tenis.py
```
Comparte solo `shared/`. `spc.py` añade `tenis-train` y `tenis-cycle`; `score` y
el auto-scoring del servidor califican también tenis.

## 2. Fuente de datos
Los repos originales de Jeff Sackmann (`tennis_atp`, `tennis_wta`) **ya no son
públicos**. Se usa el espejo `Aneeshers/tennis-sackmann-archive`:
**152,013 partidos** (ATP 79,299 · WTA 72,714), 2000-2026, con superficie, nivel
de torneo, ronda, ranking, edad, altura, mano y estadísticas de saque
(aces, dobles faltas, puntos al saque, primeros dentro, ganados con 1º y 2º,
juegos al saque, break points). 93 % de los partidos desde 2010 traen saque.
Calendario y cuotas en vivo: The Odds API (`tennis_atp_*`, `tennis_wta_*`).

**Limitación grave y documentada:** el espejo está congelado en **2026-05-25**
(106 días de antigüedad hoy). Data Health lo marca en amarillo y cada predicción
afectada lleva el aviso "histórico congelado". Si el usuario recupera una fuente
viva, solo hay que cambiar `BASE` en `ingest.py`.

## 3. Features as-of
Corte: 00:00 del día del partido (el histórico no trae hora fiable), así que un
partido nunca ve otro del mismo día. Por jugador: ventana de 12 meses, últimos 20
partidos y ventana de 24 meses **de esa superficie**:
puntos ganados al saque (spw) y al resto (rpw), spw ajustado por la calidad al
resto del rival, aces y dobles faltas por punto, break points salvados,
% de victorias, juegos ganados, juegos por partido; Elo global y **Elo por
superficie** (K por nivel de torneo); descanso, partidos en 14 días, ranking y
puntos, edad, altura, mano, best-of.

**Simetría (clave anti-fuga):** los datos históricos vienen como
ganador/perdedor. Cada partido se convierte en una fila donde p1/p2 se asignan
**por id de jugador**, nunca por resultado (test: `p1_win` = 50.6 %, todos los
`p1_id < p2_id`). Sin esto el modelo aprendería el resultado.

## 4. Modelo punto a punto (`pointmodel.py`)
Dado p1 = probabilidad de ganar un punto con su saque y p2 la del rival, se
calcula de forma **exacta y cerrada** (sin Monte Carlo): P(ganar el juego al
saque) con deuce, P(ganar el tie-break) por estados, y la distribución completa
de marcadores de set → P(ganar el partido), distribución de **juegos totales** y
de **margen de juegos**. Los tres mercados salen del mismo objeto, así que nunca
se contradicen. Validado en tests: simetría exacta (0.5 con jugadores iguales),
probabilidades que suman 1, best-of-5 amplifica la ventaja, `game_prob(0.62) =
0.7759` (valor clásico del tenis masculino).

## 5. Motores comparados y elegido
Comparación en la selección 2015-2019, **ya calibrados** (comparación justa):

| motor | log loss ATP | log loss WTA |
|---|---|---|
| **logística sobre Elo + Elo superficie + ranking** | **0.6224** | **0.6215** |
| Elo simple | 0.6276 | 0.6309 |
| Elo por superficie | 0.6315 | 0.6434 |
| punto a punto puro (spw ajustado por rival) | 0.6564 | 0.6579 |

Gana la logística. El punto a punto **no se descarta**: es el que convierte la
probabilidad calibrada de partido en juegos y hándicap (vía `solve_serve_probs`),
que es donde aporta. Modelos: `TENIS_ATP_WINNER_v1`, `TENIS_WTA_WINNER_v1`
(5 features, calibración Platt en ATP e isotónica en WTA).

## 6. Walk-forward
Prueba por temporada 2015→2026, entrenando siempre con lo anterior.
**Selección** 2015-2019 (motor, calibración, umbrales); **holdout** 2020-2026.
Se excluyen los retiros. Umbral de NO PICK elegido cada año con T-1, por buckets
(un bucket entra si acierta significativamente más del 50 % y todos los
superiores también).

## 7. Resultados por temporada (ganador)

**ATP** — acierto del modelo vs favorito Elo, y picks del umbral:

| T | n | modelo | Elo | umbral | picks · acierto | Elo en esos picks |
|---|---|---|---|---|---|---|
| 2015 | 2843 | 70.2 % | 69.3 % | — | 0 | — |
| 2016 | 2846 | 69.6 % | 68.5 % | 0.50 | 2846 · 69.6 % | 68.5 % |
| 2017 | 2814 | 67.2 % | 66.2 % | 0.50 | 2814 · 67.2 % | 66.2 % |
| 2018 | 2827 | 65.2 % | 64.5 % | 0.60 | 1896 · 70.4 % | 69.4 % |
| 2019 | 2747 | 64.9 % | 63.9 % | 0.55 | 2255 · 67.4 % | 66.6 % |
| 2020 | 1437 | 65.5 % | 64.3 % | 0.55 | 1141 · 68.5 % | 67.3 % |
| 2021 | 2671 | 64.7 % | 63.5 % | 0.65 | 1210 · 76.0 % | 74.8 % |
| 2022 | 2846 | 66.3 % | 64.9 % | 0.55 | 2266 · 69.6 % | 68.2 % |
| 2023 | 2910 | 64.8 % | 63.4 % | 0.55 | 2361 · 68.1 % | 66.7 % |
| 2024 | 2995 | 64.7 % | 64.2 % | 0.55 | 2375 · 68.4 % | 67.9 % |
| 2025 | 2835 | 65.3 % | 63.8 % | 0.55 | 2197 · 68.6 % | 67.4 % |
| 2026 | 1417 | 66.1 % | 65.8 % | 0.50 | 1417 · 66.1 % | 65.8 % |

Holdout ATP: 65.3 % · log loss 0.6150 · Brier 0.2138 · ECE 0.012.

**WTA** — holdout 65.9 % · log loss 0.6173 · Brier 0.2144 · ECE 0.0095; picks
14,059 con 67.3 %. Por temporada oscila entre 64.2 % y 67.0 %, siempre por
encima del favorito Elo (0.9-1.9 puntos).

## 8. Baselines (selección)
ATP: favorito por ranking 65.2 %, favorito Elo 66.5 %, favorito Elo superficie
65.9 %, moneda 50 %. WTA: 63.2 % / 64.3 % / 63.2 % / 50 %.
El modelo gana a todos ellos; la comparación que importa (favorito Elo) se hace
además **partido a partido sobre los mismos picks**, con z = +3.04 (ATP) y
+2.91 (WTA).

## 9. Calibración (holdout, probabilidad calibrada → acierto real)

| bucket | ATP n · acierto | WTA n · acierto |
|---|---|---|
| 50-55 % | 3546 · 52.1 % | 1678 · 53.5 % |
| 55-60 % | 3021 · 56.5 % | 4135 · 56.9 % |
| 60-65 % | 2606 · 61.7 % | 3755 · 63.4 % |
| 65-70 % | 2328 · 67.7 % | 887 · 70.3 % |
| 70 %+ | 5610 · 78.9 % | 5139 · 78.4 % |

Casi perfecta: cuando dice 65-70 %, acierta 67.7 %. Métodos probados: ninguna,
Platt e isotónica; se eligió por log loss en 2018-2019 y se ajustó el calibrador
final solo con la selección.

## 10. Juegos y hándicap
Se derivan invirtiendo la probabilidad calibrada a (p1, p2) de saque y usando la
distribución exacta. Holdout: ATP juegos MAE 6.04 (media 6.68), hándicap 4.26
(vs 4.92); WTA juegos 5.44 (media 4.82 → **bloqueado**), hándicap 4.65 (vs 5.34).
La P(over/cubre) se evalúa contra líneas sintéticas (proyección ± k): dice si la
probabilidad es fiable en abstracto, **no** si hay ventaja sobre las casas.
En ATP la P(over) sale mal calibrada (ECE 0.122): se muestra la proyección de
juegos y la probabilidad marcada como orientativa.

## 11. Auditoría de fuga de datos
- p1/p2 por id, nunca por resultado (test).
- El **debut** de cada jugador (5,641 jugadores) no tiene ninguna estadística
  previa: 100 % con historial vacío (test).
- `max_source_available_at < cutoff` en las 152,013 filas (test).
- Ninguna feature contiene resultado, marcador, minutos, cuotas ni líneas (test).
- Walk-forward con selección y holdout separados y verificados (test).
- Bug encontrado y corregido durante el desarrollo: un `NaN` de pandas
  contaminaba de forma permanente las sumas acumuladas de un jugador; se
  detectó al auditar la cobertura (8 % de spw) y se corrigió (94 %).

## 12. Mercado
Sin líneas históricas de tenis. En vivo: The Odds API con snapshots propios
(apertura, actual, cierre a 3 h del inicio) y no-vig proporcional. El dashboard
muestra MODELO / MERCADO / GAP y la alerta dice "desacuerdo significativo",
nunca "valor". Emparejado de nombres del mercado con el histórico por apellido +
inicial, con test.

## 12 bis. Calificación de resultados en vivo
El archivo histórico está congelado, así que los partidos del calendario se
califican con fuentes vivas, en este orden:

1. **TennisExplorer** (`/results/`, HTML): marcador **set a set**. De ahí salen
   los juegos reales de cada jugador, y con ellos se cierran los **tres**
   mercados: ganador, total de juegos y hándicap. Es la fuente principal.
2. **The Odds API `/scores`**: solo ganador y sets. Cierra el ganador y deja los
   juegos en `None`.
3. ESPN: opcional (`TENIS_ESPN=1`), hoy responde 403.

Si un partido se cerró por la vía 2, `results.fetch` vuelve a buscarlo mientras
siga sin juegos, y cuando aparece el detalle completa total y hándicap
retroactivamente. Si no hay marcador por juegos, la predicción queda
**pendiente**: nunca se cierra a ojo.

Lectura del marcador: `68` significa 6 juegos perdiendo el tiebreak 6-8, no 68
juegos (test `TENIS/tests/test_resultados.py`). Los partidos sin marcador
(abandono, W.O.) se ignoran en vez de calificarse.

## 13. Limitaciones
- Archivo congelado en mayo de 2026: las features de jugadores se degradan con
  el tiempo hasta que haya fuente viva.
- Sin lesiones, retiros ni bajas de última hora (en tenis pesan mucho).
- Los retiros a mitad de partido se excluyen del entrenamiento; en vivo un
  retiro anula el mercado de juegos en la práctica.
- Dobles, Challengers, ITF y Copa Davis/BJK: los partidos de equipos entran en
  el histórico pero no hay calendario en vivo para ellos.
- La superficie del calendario en vivo se infiere del nombre del torneo.
- Sin historial de cuotas: no hay prueba de ventaja frente al mercado.

## 14. Operación
```
python spc.py tenis-train    # archivo + features + walk-forward + modelos (~15 min)
python spc.py tenis-cycle    # diario: calendario, cuotas, predicciones, calificación
python spc.py score          # califica MLB + NFL + NBA + TENIS
```
