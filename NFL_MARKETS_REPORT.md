# Informe · Mercados NFL Total y Spread en Sports Prediction Center

Fecha: 2026-09-08. Alcance: SOLO el módulo NFL del Sports Prediction Center.
`C:\nfl-prediction-app` no se modificó (solo lectura, verificado por hash en test).
MLB no se modificó (los 84 tests MLB/aislamiento pasan idénticos).

## Veredicto en una línea

Total y Spread quedan como **PROYECCIÓN**, no como PICK. La proyección de puntos
sí aporta (mejora la media histórica en todas las temporadas), pero frente a la
línea de cierre del mercado el modelo **no** acierta más que una moneda
(50.7 % y 51.6 %), y una probabilidad alta sin calibrar no acertaba más que una
baja. Publicarlos como picks sería inventar.

---

## 1. Modelo usado para Moneyline
Sin cambios: `nflpred v4` (ensemble calibrado del motor NFL existente, walk-forward
2016-2025). Se lee de su base en modo `ro`. No se reentrenó ni se recalibró.

## 2. Modelo usado para Totals
`NFL_TOTAL_v1`: regresión **ridge** (α por validación interna) sobre 113 features
as-of + distribución **normal** de residuos (σ = 13.65 puntos, residuos fuera de
muestra por KFold) + calibrador logístico sin intercepto ajustado en el
walk-forward (pendiente 0.034: en la práctica encoge P(over) hacia 50 %).

Selección de algoritmo (MAE, temporadas de selección 2020-2022, todas las familias):

| algoritmo | MAE |
|---|---|
| **ridge** | **10.76** |
| random forest | 10.87 |
| xgboost | 11.00 |
| hist. gradient boosting | 11.03 |
| media histórica (baseline) | 11.23 |
| línea de cierre (referencia) | 10.51 |

## 3. Modelo usado para Spread
`NFL_SPREAD_v1`: ridge sobre 78 features de diferencia local−visitante, normal
de residuos (σ = 13.18), calibrador sin intercepto (pendiente 0.171).

| algoritmo | MAE margen (2020-2022) |
|---|---|
| **ridge** | **10.10** |
| random forest | 10.14 |
| hist. gradient boosting | 10.36 |
| xgboost | 10.42 |
| media histórica | 10.91 |
| línea de cierre | 9.75 |

LightGBM no se probó: en la máquina del usuario falla (access violation) y se
descartó ya en la auditoría MLB.

## 4. Features utilizadas
Todas vienen del `features.parquet` as-of del motor NFL (leído, no modificado)
más sumas derivadas local+visitante para totales. Familias finales:

- **Total (10 familias, 113 columnas):** EPA conjunto of/def (temporada y últimos 5),
  eficiencia conjunta (success, explosivas, 3er down, zona roja, yardas, CPOE),
  presión/turnovers conjuntos, ritmo de anotación (puntos a favor/en contra de ambos),
  nivel de puntos por equipo, QB (calidad, experiencia, out/questionable, cambio),
  lesiones (out ponderado, conteos), contexto (dome, césped, viento, frío,
  temperatura, primetime, jueves/lunes, playoffs, divisional, semana, descanso),
  Elo (rating y suma), historial (partidos jugados).
- **Spread (7 familias, 78 columnas):** Elo (diff y prob), diff EPA of/def,
  diff eficiencia, diff resultados (puntos, margen, victorias), QB, lesiones,
  historial.

## 5. Features descartadas (ablación hacia atrás, solo 2020-2022)
- Total: `level_epa` (EPA por equipo por separado; redundante con las sumas).
- Spread: `context` (clima/horario), `diff_pressure` (sacks/presión/turnovers), `h2h`.
  Quitarlas mejoró el MAE fuera de muestra; quitar más ya no mejoraba > 0.02.
- Nunca fueron candidatas: marcadores, resultado, `spread_line`, `total_line`,
  moneylines, `market_*`, `line_move`, timestamps de fuente.

## 6. Walk-forward
Temporadas de prueba 2020→2025; para cada T se entrena con 2015..T-1.
Selección de algoritmo y familias **solo con 2020-2022**; 2023-2025 se miró una
vez cerrada la configuración. Umbral de NO PICK elegido en T-1 y aplicado en T.

## 7. Resultados por temporada

**TOTAL** (acierto = lado del modelo frente a la línea de cierre)

| T | n | MAE modelo | MAE media | MAE cierre | acierto vs línea | log loss | siempre over / under | umbral (T-1) | picks · acierto |
|---|---|---|---|---|---|---|---|---|---|
| 2020 | 269 | 10.74 | 11.34 | 10.30 | 45.5 % | 0.7169 | 50.8 / 49.2 | — | 0 |
| 2021 | 285 | 10.83 | 11.08 | 10.81 | 53.5 % | 0.6975 | 45.7 / 54.3 | — | 0 |
| 2022 | 284 | 10.63 | 11.26 | 10.40 | 51.6 % | 0.7083 | 44.5 / 55.5 | 0.52 | 240 · 52.1 % |
| 2023 | 285 | 10.44 | 10.98 | 10.17 | 50.9 % | 0.7080 | 46.3 / 53.7 | — | 0 |
| 2024 | 285 | 9.96 | 10.12 | 9.77 | 49.6 % | 0.7097 | 53.9 / 46.1 | — | 0 |
| 2025 | 285 | 10.57 | 11.00 | 10.42 | 52.6 % | 0.7025 | 51.9 / 48.1 | — | 0 |

Agregado 2020-2025: n = 1675 (26 pushes), acierto 50.7 %, log loss 0.7070, Brier 0.2566.

**SPREAD**

| T | n | MAE modelo | MAE media | MAE cierre | acierto vs línea | log loss | favorito / underdog cubre | umbral (T-1) | picks · acierto |
|---|---|---|---|---|---|---|---|---|---|
| 2020 | 269 | 10.06 | 11.09 | 9.79 | 50.6 % | 0.7071 | 44.2 / 55.8 | — | 0 |
| 2021 | 285 | 10.96 | 12.03 | 10.67 | 51.2 % | 0.7037 | 46.6 / 53.4 | — | 0 |
| 2022 | 284 | 8.98 | 9.60 | 8.78 | 51.1 % | 0.6993 | 45.3 / 54.7 | — | 0 |
| 2023 | 285 | 10.34 | 11.11 | 9.98 | 50.6 % | 0.7091 | 52.0 / 48.0 | — | 0 |
| 2024 | 285 | 9.75 | 11.15 | 9.70 | 54.1 % | 0.6902 | 52.7 / 47.3 | — | 0 |
| 2025 | 285 | 10.07 | 11.00 | 9.67 | 51.8 % | 0.7129 | 48.6 / 51.4 | 0.54 | 178 · 47.8 % |

Agregado 2020-2025: n = 1660 (41 pushes), acierto 51.6 %, log loss 0.7037, Brier 0.2547.

## 8. Baselines
- Puntos: la media histórica pierde contra el modelo en las 6 temporadas (ambos
  mercados). La línea de cierre gana al modelo en las 6 (es información posterior
  a la nuestra; se muestra como referencia, nunca es feature).
- Acierto vs línea: "siempre under" (52.4 % agregado) y "siempre underdog"
  (51.8 %) rinden igual o mejor que el modelo. Ninguno supera el 52.4 % que
  compensa el vig típico (-110).

## 9. Calibración (probabilidad sin calibrar del lado del modelo → acierto real, 2020-2025)

| bucket | Total n · acierto | Spread n · acierto |
|---|---|---|
| 50-55 % | 716 · 51.7 % | 780 · 51.8 % |
| 55-60 % | 548 · 49.6 % | 504 · 48.6 % |
| 60-65 % | 287 · 50.5 % | 246 · 56.1 % |
| 65-70 % | 96 · 49.0 % | 77 · 49.4 % |
| 70 %+ | 28 · 53.6 % | 53 · 58.5 % |

Un "65-70 %" acertó 49 %. Por eso el calibrador (ajustado sobre estos datos, sin
intercepto para no cambiar de lado) deja las probabilidades publicadas entre
50 % y ~53 %. Moneyline conserva su calibración propia; no se reutiliza nada.

**Key numbers.** Masa real / masa que da una normal: spread 3 → ×2.72, 7 → ×1.75,
14 → ×1.49, 6 → ×1.36; total 51 → ×1.46, 37 → ×1.45, 44 → ×1.31, 41 → ×1.29.
Una normal simple NO representa bien los key numbers del spread. Aun así, la
normal y la distribución empírica dieron el mismo log loss frente a la línea
(0.7070 vs 0.7076 total; 0.7037 vs 0.7068 spread), así que se usa la normal por
regla; la σ y la mediana publicadas son las empíricas. Con un modelo que no
bate a la línea, afinar la forma de la distribución no cambia la conclusión.

## 10. Leakage audit
- Ninguna columna de resultado ni de mercado es feature (test automático sobre
  cada familia).
- Los 6 timestamps de fuente (stats, depth chart, lesiones × local/visitante) son
  estrictamente anteriores al kickoff en todas las filas históricas (test).
- Las líneas de nflverse son de cierre sin timestamp previo: se usan solo como
  referencia y para evaluar; nunca como entrada.
- `cutoff_timestamp` de cada predicción = máximo timestamp de fuente; test
  comprueba `cutoff < kickoff`.
- El walk-forward se puede reproducir fila a fila (test recalcula 2023 y compara).
- Limitación heredada: el motor NFL construye el clima con `<=` kickoff y el QB
  esperado desde depth charts semanales; SPC lo consume tal cual, no lo audita
  de nuevo.

## 11. Mercado vs modelo
- Histórico: solo hay línea de cierre. El cierre bate al modelo en MAE todas las
  temporadas (≈0.3 puntos). Reportado, no maquillado.
- En vivo: The Odds API (misma clave) → tabla propia `nfl_odds` con snapshot,
  apertura, actual, cierre (último snapshot dentro de las 3 h previas) y
  no-vig proporcional. El dashboard muestra MODELO / MERCADO / GAP por separado.
- El gap modelo−mercado **no está validado como valor**: no existe historial de
  cuotas pre-kickoff para medirlo. La alerta dice "desacuerdo", nunca "valor".

## 12. Mercados habilitados como PICK
- NFL Moneyline (motor existente).
- MLB Moneyline (Elo, auditoría previa).

## 13. Mercados como PROYECCIÓN / informativos
- NFL Total: proyección de puntos + P(over/under) calibrada. Etiqueta PROYECCIÓN.
- NFL Spread: margen proyectado + P(cubre) calibrada. Etiqueta PROYECCIÓN.
- MLB Over/Under y Run line (sin cambios).

## 14. Mercados bloqueados
Ninguno nuevo. (Si en el futuro un mercado no mejorara ni la media histórica,
`gating.decide()` lo bloquearía y `predict` no lo generaría.)

## 15. Limitaciones
- Sin cuotas históricas pre-kickoff: el rendimiento vs línea se mide contra el
  cierre, el más difícil de batir. Cuando haya meses de snapshots propios se
  podrá medir contra apertura.
- 2026: solo semana 1 con líneas; las features de esas filas dependen del último
  refresco del motor NFL (`weekly_update` de nfl-prediction-app).
- Las probabilidades publicadas están casi en 50 % por diseño (calibración
  honesta). No es un fallo de visualización.
- Umbrales de NO PICK: solo dos temporadas encontraron umbral significativo en
  T-1 (total 2022 con 0.52; spread 2025 con 0.54) y en ambas los picks fallaron
  fuera de muestra (52.1 % y 47.8 %). Por eso hoy no existe umbral de pick.
- Pace real (jugadas/drive) no está en los datos del motor; se usa un proxy de
  ritmo de anotación.

## 16. Tests finales
`python -m pytest` → **113 passed** (10 aislamiento, 75 MLB/shared/dashboard/auditoría,
**28 nuevos NFL**: over, under, push, líneas .5, spread favorito/underdog,
pick'em, etiquetas con signo, no-vig y cuotas posteriores excluidas, sin
features de resultado/mercado, timestamps < kickoff, reproducibilidad del
walk-forward, gating sin evidencia, buckets, baselines, calibrador que no
cambia de lado, versionado, unicidad, cambio de línea → nueva versión, hash del
motor NFL antes/después, moneyline sin dependencia de NFL.markets, NFL.markets
sin importar MLB, dashboard con 3 mercados, Top picks sin mezclar, endpoint de
mercados).

## Cómo se opera
```
python spc.py nfl-train          # investigación walk-forward + entrenamiento (≈5 min)
python spc.py nfl-cycle          # cuotas + predicciones + calificación (programar a diario)
python spc.py score              # califica MLB y NFL (el servidor lo hace solo cada 10 min)
```
Artefactos: `NFL/markets/out/research.json`, `research_lines.json`, `preds_*.parquet`,
`lines_*.parquet`; modelos en `NFL/models/`; base en `NFL/database/nfl_markets.sqlite3`.
