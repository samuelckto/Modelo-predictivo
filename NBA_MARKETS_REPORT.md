# Informe · Módulo NBA del Sports Prediction Center

Fecha: 2026-09-08. Alcance: módulo NBA nuevo, aislado. NFL, motor NFL original y MLB
sin cambios funcionales (hash del motor NFL y de la base MLB verificados en test;
los 113 tests previos siguen pasando; total 137).

## Veredicto

| mercado | estado | evidencia |
|---|---|---|
| NBA Moneyline | **PICK** (con umbral 55 %) | 69.4 % de acierto en 3,682 picks del holdout 2023-2026 (z = 23.5 frente a 50 %); calibración ECE 0.028 |
| NBA Spread | **PROYECCIÓN** | margen proyectado con MAE 10.85 vs 12.26 de la media; P(cubre) calibrada (ECE 0.005) frente a líneas neutras; **sin líneas históricas** → no se puede demostrar ventaja contra el mercado |
| NBA Total | **PROYECCIÓN** | total proyectado con MAE 14.97 vs 17.68; P(over) calibrada (ECE 0.031); misma limitación de mercado |

Advertencia honesta sobre el moneyline: gana claramente a "siempre local" (55.5 %) y al 50 %,
pero **empata con "siempre el favorito Elo"** en esos mismos partidos (69.8 %, z = −0.54).
Su valor añadido frente a Elo es la calidad de la probabilidad (log loss 0.619 vs 0.626,
ECE 0.019 vs 0.043), no más aciertos. Y **no está medido contra las cuotas**: ganar al
55 % de tasa local no es ganar a las casas de apuestas.

---

## 1. Arquitectura
```
NBA/
  markets/ db.py ingest.py features.py research.py research_lines.py gating.py
           train.py predict.py odds.py pipeline.py provider.py
  markets/out/ features.parquet research.json research_lines.json preds_*.parquet lines_*.parquet
  database/nba_markets.sqlite3   (8 tablas: games, team_game_logs, teams, model_versions,
                                  predictions, odds_snapshots, availability, pipeline_runs, source_log)
  models/ moneyline_v1.joblib spread_v1.joblib total_v1.joblib
  tests/test_nba_markets.py
```
Integración: `dashboard/backend/main.py` (`SPORTS` incluye NBA; `/api/games`, `/api/top-picks`,
`/api/alerts`, `/api/markets`, `/api/performance`, `/api/data-health`, `/api/history`,
`/api/nba/health`, `/api/nba/markets`), `performance.py` (walk-forward NBA), `history.py`
(`nba_history`), `spc.py` (`nba-train`, `nba-cycle`, `score`), auto-scoring del servidor.
Compartido: solo `shared/` (no-vig, calibración, rutas). `shared/paths.py` añade `NBA_DIR`
y el aislamiento de escritura por deporte.

## 2. Fuente de datos
**pbpstats.com API** (derivada del play-by-play oficial de la NBA): 14,128 partidos
2015-16 → 2025-26 (regular + playoffs) y 28,256 logs equipo-partido con posesiones,
pace, tiros por zona, rebotes, pérdidas, faltas, TL. 528 descargas, 0 errores.
Descartados: stats.nba.com (timeout desde el entorno), basketball-reference (403),
balldontlie (requiere clave). Hora de tipoff: no la publica pbpstats → el corte as-of es
00:00 del día del partido (solo entran días anteriores). En vivo, el calendario con
tipoff viene de The Odds API.

## 3. Features (as-of, versión `nba_features_v1`)
Por equipo, acumulado de temporada con encogimiento a la temporada anterior (10 partidos)
y últimos 10: ORtg, DRtg, pace, margen, victorias, puntos, eFG%, TS%, tasa y acierto de
triples, ataques al aro, tiros libres, pérdidas propias y forzadas, rebote ofensivo y
defensivo, tiro permitido, asistencias, robos, tapones, faltas, calidad de tiro; descanso,
back-to-back, partidos en 7 días; Elo (k=20, ventaja local 70, regresión 25 %); contexto
(playoffs, mes, partidos jugados). Se construyen diferencias local−visitante y sumas.
Familias: elo, strength, pace, shooting, turnovers, rebounding, free_throws, defense_misc,
rest, context.

## 4. Features eliminadas
- Total: familia **shooting** (48 columnas): quitarla mejoró el MAE fuera de muestra
  15.13 → 15.04 en 2019-2022. Redundante con ORtg/DRtg para el total.
- Moneyline y spread: ninguna familia mejoró > tolerancia al quitarse (se conservan las 10).
- Nunca candidatas: puntos, margen, total, posesiones del partido, resultado, cualquier
  columna de cuotas o líneas. **Lesiones, disponibilidad y quintetos: excluidos** (ver §16-17).

## 5. Modelos probados (selección 2019-2022, todas las familias)

Moneyline (log loss / acierto): **random forest 0.6329 / 64.5 %**, logística 0.6350 / 64.5 %,
XGBoost 0.6369 / 64.3 %, HGB 0.6379 / 64.2 %, Elo 0.6415 / 64.4 %, tasa local 0.6871 / 55.9 %.
Spread (MAE): **ridge 10.69**, RF 10.72, HGB 10.75, XGB 10.76, media 11.73.
Total (MAE): **ridge 15.13**, XGB 15.19, RF 15.22, HGB 15.23, media 17.69.
Ensemble: no se usó (ningún modelo aportaba lo bastante distinto para justificarlo).

## 6-8. Modelos elegidos
- `NBA_ML_v1`: random forest (300 árboles, hoja mínima 25), 119 features, calibración Platt.
- `NBA_SPREAD_v1`: ridge, 119 features, distribución normal de residuos (σ 12.9).
- `NBA_TOTAL_v1`: ridge, 83 features, distribución empírica de residuos (σ 18.6).

## 9-11. Walk-forward, selección y holdout
Temporadas de prueba 2019 → 2026 (2019 = 2018-19), entrenando con todo lo anterior desde
2015-16. **Selección**: 2019-2022 (algoritmo, familias, calibración, distribución).
**Holdout**: 2023-2026, no tocado hasta cerrar la configuración. Solo temporada regular
(los playoffs se ingieren pero se excluyen: régimen distinto). Umbral de NO PICK elegido
cada año con la temporada T-1.

## 12-13. Baselines y métricas por temporada

**Moneyline**

| T | n | acierto | log loss | Brier | ECE | AUC | Elo acierto | Elo log loss | tasa local | umbral (T-1) | picks · acierto | Elo en esos picks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | 1230 | 66.2 % | 0.6135 | — | 0.041 | 0.706 | 65.5 % | 0.6272 | 59.3 % | — | 0 | — |
| 2020 | 1059 | 65.0 % | 0.6284 | — | 0.033 | 0.697 | 65.5 % | 0.6327 | 55.1 % | 0.55 | 842 · 67.8 % | 67.2 % |
| 2021 | 1080 | 62.4 % | 0.6500 | — | 0.036 | 0.667 | 62.2 % | 0.6574 | 54.4 % | 0.55 | 833 · 65.2 % | 64.1 % |
| 2022 | 1230 | 64.4 % | 0.6410 | — | 0.036 | 0.683 | 64.3 % | 0.6495 | 54.4 % | 0.55 | 942 · 68.3 % | 67.1 % |
| 2023 | 1230 | 62.0 % | 0.6474 | — | 0.039 | 0.655 | 62.9 % | 0.6625 | 58.0 % | 0.55 | 854 · 66.3 % | 67.2 % |
| 2024 | 1230 | 64.9 % | 0.6171 | — | 0.035 | 0.716 | 64.6 % | 0.6213 | 54.3 % | 0.55 | 947 · 69.3 % | 69.4 % |
| 2025 | 1230 | 65.3 % | 0.6116 | — | 0.039 | 0.723 | 66.3 % | 0.6182 | 54.5 % | 0.55 | 950 · 69.8 % | 70.1 % |
| 2026 | 1230 | 68.1 % | 0.5983 | — | 0.036 | 0.739 | 68.1 % | 0.6038 | 55.4 % | 0.55 | 931 · 71.9 % | 72.2 % |

Holdout 2023-2026 (4,920 partidos): modelo 65.1 % / log loss 0.6186 / Brier 0.2149 / ECE 0.019
/ AUC 0.708 · Elo 65.4 % / 0.6264 / 0.2181 / 0.043 · tasa local 55.5 %.

**Spread** (MAE / RMSE del margen; sin líneas históricas no hay ATS contra mercado)

| T | n | modelo | media |
|---|---|---|---|
| 2019 | 1230 | 10.18 / 13.11 | 11.37 |
| 2020 | 1059 | 10.19 / 12.92 | 11.24 |
| 2021 | 1080 | 11.21 / 14.25 | 12.06 |
| 2022 | 1230 | 11.17 / 14.18 | 12.22 |
| 2023 | 1230 | 10.13 / 12.88 | 10.90 |
| 2024 | 1230 | 10.99 / 14.13 | 12.45 |
| 2025 | 1230 | 10.94 / 14.00 | 12.59 |
| 2026 | 1230 | 11.36 / 14.55 | 13.10 |

Holdout: 10.85 / 13.90 vs media 12.26 / 15.43.

**Total**

| T | n | modelo | media |
|---|---|---|---|
| 2019 | 1230 | 15.21 / 19.42 | 19.02 |
| 2020 | 1059 | 15.29 / 19.33 | 18.01 |
| 2021 | 1080 | 14.91 / 18.91 | 17.40 |
| 2022 | 1230 | 14.77 / 18.63 | 16.34 |
| 2023 | 1230 | 14.61 / 18.79 | 18.25 |
| 2024 | 1230 | 15.16 / 18.76 | 17.64 |
| 2025 | 1230 | 14.61 / 18.32 | 17.09 |
| 2026 | 1230 | 15.51 / 19.40 | 17.76 |

Holdout: 14.97 / 18.82 vs media 17.68 / 22.38. Pushes: no medibles sin líneas reales; con
líneas sintéticas enteras el sistema los excluye (test).

## 14. Calibración
Moneyline, elección en 2022 con ajuste en 2019-2021: ninguna 0.6410, **Platt 0.6406**,
isotónica 0.6406 → Platt (pendiente 0.923, intercepto −0.017). Buckets del holdout
(probabilidad calibrada → acierto real): 50-55 % → 53.1 % (n 1238) · 55-60 % → 59.0 % (1111)
· 60-65 % → 64.9 % (833) · 65-70 % → 71.1 % (686) · 70 %+ → 82.7 % (1052). Ligeramente
conservador en la zona alta; nunca sobreconfiado.
Spread P(cubre k), normal (log loss 0.6296 vs empírica 0.6298): 55-60 → 57.5 %, 60-65 → 65.2 %,
70+ → 77.5 %; ECE 0.005. Total P(over k), empírica (0.6100 vs 0.6108): 55-60 → 55.9 %,
60-65 → 65.1 %, 65-70 → 66.1 %, 70+ → 77.6 %; ECE 0.031. Calibrado frente a líneas
neutras (proyección ± k); **frente a líneas reales del mercado no está medido** y la
experiencia NFL indica que sería sobreconfiado.

## 15. Leakage audit
- Cada log lleva `available_at` = fin del día del partido; el corte de cada partido es
  00:00 de su día (`max_source_available_at < cutoff` en las 14,128 filas; test).
- Test de truncación: se reconstruyen las features con la base cortada en 2024-01-15 y
  coinciden exactamente (diferencia < 1e-9) con las del parquet completo para todos los
  partidos anteriores.
- Ninguna feature es resultado, posesiones del partido, cuota ni línea (test por familia).
- Walk-forward reproducible fila a fila (test recalcula 2024 y compara).
- Partidos futuros: fila sin métricas, reciben as-of y no alimentan a nadie.

## 16. Injury timestamp audit
No existe fuente gratuita de reportes de lesiones históricos con timestamp. **No se
fingieron**: ninguna feature de lesión entra al modelo. La tabla `nba_availability`
existe para anotar en vivo (out / questionable / probable / available) con timestamp
propio; sin historial no se usa para predecir.

## 17. Lineup timestamp audit
Igual: el quinteto real del partido no se usa como si se conociera antes. Excluido y
documentado. Es la limitación principal del módulo (la NBA es muy sensible a ausencias).

## 18. Market comparison
Sin cuotas históricas → sin comparación modelo/mercado en el backtest. El dashboard muestra
MODELO / MERCADO / GAP con cuotas en vivo, sin llamarlo "value". La alerta dice
"desacuerdo significativo", nunca "ventaja".

## 19. Odds coverage
The Odds API `basketball_nba` (misma clave): h2h, spreads, totals; snapshots con
timestamp, apertura (primer snapshot), actual, cierre (último dentro de las 3 h previas),
no-vig proporcional. Los eventos sirven también de calendario (tipoff). Cobertura histórica: 0.

## 20. Line movement
`movement()` compara apertura vs actual por partido/mercado; alerta `LINE_MOVEMENT` si
≥ 1.5 pts (spread) o ≥ 2.5 (total). Sin validación histórica de su valor predictivo.

## 21. PICK / PROJECTION / BLOCKED
Moneyline: PICK si p ≥ 0.55 (umbral que en cada T-1 fue el primer bucket con acierto
significativamente > 50 %, z ≥ 1.96, y todos los superiores también); NO PICK por debajo.
Spread y total: PROJECTION. Ningún mercado BLOCKED (todos mejoran la media histórica y
están calibrados). `gating.decide()` bloquearía automáticamente si dejara de cumplirse.

## 22. Limitaciones
- Sin lesiones ni quintetos históricos: el modelo no sabe si falta la estrella.
- Sin cuotas históricas: no hay evidencia de ventaja contra el mercado en ningún mercado.
- Corte as-of por día (sin hora): conservador; un partido de la tarde no ve el de la
  mañana del mismo día.
- Temporada 2026-27 aún no empieza: no hay predicciones reales hasta octubre.
- La temporada 2020 (burbuja) y 2021 (72 partidos) tienen menos datos.
- Playoffs excluidos del entrenamiento y de la evaluación.

## 23. Reproducibilidad
`python spc.py nba-train` reingesta (reanudable), reconstruye features, ejecuta
`research.py` (por etapas, reanudable) y `research_lines.py`, y entrena. Semillas fijas.
Tests: `python -m pytest` → **137 passed** (24 NBA nuevos que cubren los 34 casos exigidos:
ML win/loss, spread ±, .5, 0, push, over/under, cover, no-vig, timestamps, truncación,
exclusión de mercado y resultado, reproducibilidad walk-forward, calibración, buckets,
no-pick, mercado bloqueado, versionado, unicidad, cambio de línea, dashboard, top picks,
performance, data health, endpoints, aislamiento NBA↔NFL/MLB, hash del motor NFL y de la base MLB).

## Operación
```
python spc.py nba-train      # solo si quieres regenerar todo (≈50 min: 528 descargas + walk-forward)
python spc.py nba-cycle      # diario en temporada: partidos nuevos, cuotas, predicciones, calificación
python spc.py score          # califica MLB + NFL + NBA (el servidor lo hace solo cada 10 min)
```
