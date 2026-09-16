> ⚠️ **DOCUMENTO SUPERADO.** Este informe es anterior a la auditoría final.
> Varias de sus conclusiones fueron refutadas por `FINAL_AUDIT_REPORT.md`:
> F5 Moneyline ya **no** es un pick validado, moneyline se sirve con **Elo** y
> el orden de Top Picks por «ventaja sobre la tasa base» resultó **peor** que
> por probabilidad. Se conserva para poder comparar qué cambió y por qué.

---

# AUDIT_REPORT — Sports Prediction Center

Generado a partir de comandos realmente ejecutados. **Ninguna cifra de este
documento está estimada.** Donde una prueba falló, se dice cuál y por qué.

---

## 0. Resumen en una frase

El motor MLB predice mejor que el azar y mejor que las estrategias tontas, pero
**el margen es pequeño** (+3.1 pp en moneyline sobre decir siempre "local"), el
mercado de totales **no aporta nada y está bloqueado**, el run line **no supera a
"decir siempre que no" en acierto**, y el índice compuesto de Upset Risk **falló
su validación y fue retirado**.

---

## 1. Auditoría exigida (13 puntos del §44)

| # | Prueba | Estado | Evidencia |
|---|---|---|---|
| 1 | Tests unitarios | ✅ | 57 pruebas pasan |
| 2 | Tests de integración | ✅ | ciclo diario completo ejecutado (`spc.py mlb-daily`) |
| 3 | Aislamiento NFL/MLB | ✅ | 8 pruebas; sqlite rechaza escrituras en la base NFL |
| 4 | Data leakage | ✅ | 9 sintéticas + 6 contra datos reales |
| 5 | Backtest temporal NFL | ✅ | del motor NFL existente (2020-2025) |
| 6 | Backtest temporal MLB | ✅ | walk-forward 2023-2026, n=9.574 por mercado |
| 7 | Calibración | ✅ | ECE 0.016-0.027 según mercado |
| 8 | Comparación con mercado | ❌ **NO SE PUDO** | no existe histórico gratuito de cuotas MLB |
| 9 | Análisis por bucket de probabilidad | ✅ | terciles: 46.2% / 46.3% / 40.3% de error |
| 10 | Análisis de Upset Risk | ❌ **FALLÓ** | el índice no separa los fallos; retirado |
| 11 | Reproducibilidad | ✅ | 317 snapshots de features, 1 por predicción |
| 12 | Validación de fuentes | ✅ | 15.063 descargas, 0 errores, 0 truncamientos |
| 13 | Validación del dashboard | ✅ | render JSDOM de las 6 pestañas, 0 errores |

---

## 2. Datos ingeridos (ejecutado, no estimado)

| Temporada | Partidos | Terminados | Boxscores | Días con Statcast | Lanzamientos |
|---|---|---|---|---|---|
| 2021 | 2 467 | 2 466 | 2 466 | 206 | 723 172 |
| 2022 | 2 470 | 2 470 | 2 470 | 200 | 721 984 |
| 2023 | 2 471 | 2 471 | 2 471 | 205 | 732 562 |
| 2024 | 2 473 | 2 472 | 2 472 | 209 | 729 918 |
| 2025 | 2 477 | 2 477 | 2 477 | 210 | 733 376 |
| 2026 | 2 483 | 2 155 | 2 155 | 188 | 740 625 |

**Cobertura de boxscores: 100 %** de los partidos terminados.
Statcast cubre **todos** los días con partido de cada temporada.

Filas: 125 096 líneas de pitcheo · 294 410 de bateo · 433 575 agregados Statcast ·
28 361 de bullpen · 29 037 alineaciones · 29 426 pitchers probables · 2 620 lesiones.

**15 063 descargas registradas · 0 errores · 2 `skipped`** (las cuotas, por no
haber clave de API configurada en el entorno de construcción).

### Verificación contra la realidad conocida
| Métrica | Valor obtenido | Valor esperado (MLB real) |
|---|---|---|
| Ventaja de local 2021-2025 | **53.13 %** | ~53 % |
| Carreras por partido | **8.90** | 8.6-9.0 |
| Local gana las 5 primeras entradas | 45.02 % | ~45 % (25 % son empates) |

### Un fallo real que se detectó y corrigió
Baseball Savant **corta el CSV en 25 000 filas**. Los primeros bloques de 8 días
superaban ese tope y llegaban **truncados**: 328 134 filas contaminadas. Se
detectó comparando `2 días → 5 997 filas`, `4 días → 15 140`, `8 días → 25 000`
(tope exacto). Se **borró todo lo descargado y se rehízo** con bloques de 3 días
y una guarda que parte el rango automáticamente si vuelve a tocar el tope.
Resultado actual: **0 truncamientos**.

---

## 3. Anti-leakage

Regla: para un partido que empieza en T solo entran filas con `available_at < T`
(estrictamente menor: un dato con el mismo instante se considera no disponible).

### Pruebas contra datos reales (`MLB/tests/test_leakage_real.py`)
1. Ninguna línea de un partido tiene `available_at` anterior a su propio inicio.
2. El OBP ofensivo se recalculó a mano para 150 partidos con un filtro
   independiente: **coincide exactamente**.
3. La ERA del abridor se recalculó para 120 partidos: **coincide**, y se verifica
   que la apertura del propio partido nunca está incluida.
4. El primer partido de la temporada de cada equipo **no tiene** medias acumuladas.
5. Los agregados Statcast de un día no están disponibles hasta el día siguiente.
6. Ninguna columna objetivo aparece entre las features.

**Un fallo detectado por estas pruebas:** la ventana "temporada" arrastraba datos
del año anterior (no era leakage, pero la etiqueta era falsa). Se corrigió
agrupando también por temporada.

### Supuestos declarados (no son datos)
- **Fin de partido**: la API no publica la hora exacta; se usa `inicio + 3 h 15 min`.
- **Statcast disponible**: día siguiente a las 06:00 UTC (conservador).
- **Identidad del abridor histórico**: para partidos anteriores a la primera
  ingesta no hay snapshot del anuncio, así que se toma **quién lanzó realmente**.
  La MLB publica los probables con días de antelación y coinciden casi siempre,
  pero **no siempre**. El supuesto afecta solo a *quién* lanza; **todas** sus
  estadísticas siguen siendo estrictamente anteriores al partido. La columna
  `sp_source` distingue `actual_starter_assumed` (14 504 filas) de
  `probable_announced` (34, las de partidos futuros reales).

---

## 4. Backtest walk-forward MLB

Esquema: para cada temporada de prueba T se entrena con todo lo anterior a T-1,
se valida en T-1 (Elo y política de combinación) y se mide en T, que no se toca
antes. Temporadas de prueba: **2023, 2024, 2025, 2026**. n = 9 574 por mercado.

### Moneyline (tasa base: el local gana el 52.72 %)
| Estrategia | Accuracy | Log loss | Brier | ECE |
|---|---|---|---|---|
| **Elo** | **55.82 %** | **0.6819** | 0.2444 | 0.0153 |
| Random forest | 55.59 % | 0.6832 | 0.2451 | 0.0163 |
| Ensemble (4 algoritmos) | 55.46 % | 0.6838 | 0.2454 | 0.0197 |
| Siempre el local | 52.73 % | 0.6931 | 0.2500 | 0.0272 |
| Aleatorio | 49.94 % | 0.9834 | 0.3315 | 0.2513 |

> **El Elo simple gana al ensemble de machine learning.** No es lo que se
> esperaba y se reporta tal cual. La diferencia (0.36 pp) está dentro del ruido,
> pero en ningún caso el ensemble demuestra ser mejor.

### Primeras 5 entradas (tasa base 45.00 %; clase mayoritaria acertaría 55.00 %)
| Estrategia | Accuracy | Log loss | ECE |
|---|---|---|---|
| **Random forest** | **56.85 %** | 0.6818 | 0.0146 |
| Ensemble | 56.32 % | 0.6821 | 0.0165 |
| Elo | 51.41 % | 0.6979 | 0.0809 |
| Siempre el local | 45.00 % | 0.6932 | 0.0501 |

> Aquí el modelo **sí** aporta: +5.4 pp sobre Elo. Tiene sentido — las primeras 5
> entradas dependen sobre todo del abridor, que es justo lo que el modelo mide.

### Run line (local −1.5; tasa base 35.73 %; clase mayoritaria 64.27 %)
| Estrategia | Accuracy | Log loss | ECE |
|---|---|---|---|
| Random forest | 64.53 % | 0.6430 | 0.0222 |
| Ensemble | 64.41 % | 0.6440 | 0.0270 |
| Decir siempre "no cubre" | 64.27 % | — | — |

> **Aviso importante:** ese 64.5 % suena bien pero **decir siempre "no cubre"
> acierta 64.27 %**. La ventaja real son **0.26 pp (z = 0.54, no significativa)**.
> Lo que sí mejora es el log loss (+0.0079), o sea que las *probabilidades*
> informan aunque la *decisión* no acierte más. Por eso el run line se publica
> **solo como probabilidad calibrada, nunca como pick**.

### Total de carreras — **BLOQUEADO**
| Métrica | Modelo | Baseline (predecir la media) |
|---|---|---|
| RMSE | 4.501 | 4.506 |
| MAE | 3.568 | — |

| Línea | "Over" ocurre | Clase mayoritaria | Modelo | Ventaja |
|---|---|---|---|---|
| 7.5 | 57.02 % | 57.02 % | 56.90 % | **−0.12 pp** |
| 8.5 | 49.53 % | 50.47 % | 51.45 % | +0.98 pp |
| 9.5 | 39.78 % | 60.22 % | 60.33 % | +0.11 pp |

> El modelo de totales **mejora el RMSE en 0.005 carreras**. Eso es nada.
> Aplicando la regla del §5 ("no generar probabilidades para mercados que el
> sistema no pueda modelar adecuadamente"), **el mercado de totales está
> bloqueado por código** (`MLB/engine/gating.py`) y no aparece en el dashboard.

### Decisión automática de qué se publica
| Mercado | Estado | z vs clase mayoritaria | Mejora log loss |
|---|---|---|---|
| moneyline | **PICK** | 6.07 | +0.0079 |
| f5_moneyline | **PICK** | 3.65 | +0.0060 |
| run_line | solo probabilidad | 0.54 | +0.0079 |
| total | **BLOQUEADO** | — | — |

---

## 5. Calibración

| Mercado | ECE | Lectura |
|---|---|---|
| moneyline | 0.0197 | cuando dice 60 %, ocurre ~60 % (±2 pp) |
| f5_moneyline | 0.0165 | bien calibrado |
| run_line | 0.0270 | aceptable |

Por terciles de probabilidad (n = 4 631, prueba en 2025-2026):

| Tercil | Probabilidad media | Tasa de error real |
|---|---|---|
| parejo | 51.9 % | 46.18 % |
| medio | 55.9 % | 46.34 % |
| claro | 62.7 % | **40.28 %** |

La probabilidad **sí** discrimina: los picks "claros" fallan 6 pp menos.

---

## 6. Upset Risk — **LA VALIDACIÓN FALLÓ**

Se construyó un índice 0-100 al estilo del sistema NFL (cercanía al 50 %,
dispersión entre algoritmos, distancia al Elo, datos faltantes, desacuerdo con el
mercado). Se validó fuera de muestra (entreno 2021-2024, prueba 2025-2026,
n = 4 631):

| Tercil de riesgo declarado | n | Tasa de error real |
|---|---|---|
| bajo | 1 544 | 43.65 % |
| medio | 1 543 | 44.72 % |
| alto | 1 544 | 44.43 % |

**Correlación riesgo-error: 0.008.** Prácticamente cero. Declarar "riesgo alto"
no identificaba predicciones más equivocadas. Se probó también ajustando los
pesos con una regresión logística sobre el error: `monotonic: false`,
`useful: false`.

**Qué se hizo:** se retiró el índice compuesto. En su lugar el sistema publica
(1) la **probabilidad de fallar que estima el propio modelo** (100 − probabilidad
del pick), que sí está validada y calibrada, y (2) los avisos de incertidumbre
como **hechos separados y sin puntuar**: `COIN_FLIP`, `MODEL_DISAGREEMENT`,
`ELO_GAP`, `PITCHER_UNKNOWN`, `LINEUP_UNCONFIRMED`, `DATA_INCOMPLETE`,
`MODEL_OVER_MARKET`, `MARKET_OVER_MODEL`.

Las bandas LOW/MEDIUM/HIGH se fijaron con los **terciles observados de MLB**
(≥62 %, ≥54.5 %, resto), no con los cortes de la NFL, porque en MLB casi todas
las probabilidades caen entre 50 % y 76 % y los cortes de la NFL habrían dejado
todo en la misma banda.

---

## 7. Comparación modelo vs mercado — **NO SE PUDO HACER EN MLB**

The Odds API **no ofrece histórico de cuotas MLB en el plan gratuito**. Sin
histórico no hay forma honesta de medir si el mercado es mejor que el modelo.

Consecuencias, todas declaradas en el sistema:
- El backtest MLB **no incluye** ninguna estrategia de mercado.
- La política de combinación es la **a priori 50/50**, marcada como tal en cada
  predicción (`"por defecto a priori (50/50): sin historial suficiente no se
  optimiza nada"`). **No está validada** y no se presenta como si lo estuviera.
- Cuando no hay cuotas, la probabilidad final es la del modelo y la tarjeta
  muestra **SIN CUOTAS**.

En NFL sí existe esa comparación (motor existente): mercado 66.50 % vs ensemble
66.87 %, con **desacuerdo direccional en 243 partidos** donde el ensemble acertó
59.3 % frente al 55.6 % del mercado y el 44.4 % del modelo puro.

---

## 8. Reproducibilidad y trazabilidad

- **317 predicciones**, cada una con `prediction_timestamp`, `model_version`,
  `feature_version`, `sources` y un **snapshot de features** (317 snapshots).
- **14 predicciones en versión 2**, generadas cuando se confirmaron alineaciones.
  La v1 queda como `superseded`; **nunca se borra**. Verificado por la prueba
  `test_las_predicciones_no_se_sobrescriben`.
- Cada descarga deja fila en `data_source_logs` con `source`, `url`, `status`,
  `records`, `retrieved_at`, `checksum` y `error`. La clave de API **nunca** se
  guarda (se recorta de la URL antes de registrar).

---

## 9. WHAT WE KNOW

- Los datos son reales, completos y verificados contra la realidad conocida.
- No hay data leakage: comprobado recalculando features a mano contra la base.
- El modelo de **primeras 5 entradas** aporta valor claro sobre el Elo (+5.4 pp).
- El **moneyline** supera a las estrategias tontas por +3.1 pp, significativo.
- Las probabilidades están **calibradas** (ECE < 0.03).
- El sistema **bloquea** el mercado que no sabe modelar.

## 10. WHAT WE DON'T KNOW

- **Si el modelo bate al mercado en MLB.** No hay histórico de cuotas: no se sabe.
- Si el ensemble aporta algo sobre el Elo en moneyline. Los datos dicen que no,
  pero la diferencia está dentro del ruido.
- Si la política 50/50 modelo-mercado es la correcta para MLB. No está validada.
- Cuánto se pierde por el supuesto del abridor histórico. No se ha cuantificado.
- Si estos resultados se mantienen en 2027. Un backtest no es el futuro.

## 11. LIMITACIONES

1. Sin histórico de cuotas MLB → sin comparación de mercado, sin ROI, sin CLV.
2. Sin histórico de lesiones MLB (la API solo publica el estado actual).
3. Sin datos de umpire, viento direccional ni dimensiones de estadio.
4. Park factors solo desde la 2ª temporada de cada sede (80.6 % de cobertura).
5. Los márgenes son pequeños. MLB es un deporte muy aleatorio.
6. Las predicciones antes de que se confirmen alineaciones son **provisionales**.

## 12. IMPORTANT CAVEATS

- **Una accuracy alta no significa un buen modelo.** El run line acierta 64.5 %
  y no aporta prácticamente nada. Siempre hay que mirar la tasa base.
- **El sistema no dice quién va a ganar.** Dice con qué frecuencia debería
  ocurrir algo si el modelo está bien calibrado.
- **Nadie ha verificado que esto sea rentable apostando.** No se ha medido ROI
  porque no hay cuotas históricas, y sin cuotas la rentabilidad es incalculable.
- El margen de +3.1 pp sobre "siempre el local" es real pero **pequeño**; en una
  muestra de 100 partidos es indistinguible de la suerte.
