# FINAL_AUDIT_REPORT — Sports Prediction Center (NFL + MLB)

Auditoría ejecutada sobre el sistema ya construido. **No se reinventó nada**: se
verificó, se midió y, donde la evidencia lo obligó, se corrigió.

Todas las cifras salen de comandos realmente ejecutados y son reproducibles con
los comandos que se indican. Donde una prueba falló, se dice cuál y por qué.

---

## 1. EXECUTIVE SUMMARY

La auditoría encontró **cuatro problemas reales** y los corrigió:

| # | Hallazgo | Gravedad | Estado |
|---|---|---|---|
| 1 | Las 116 features del abridor usan la identidad del lanzador tomada del boxscore del propio partido | **Alta** | Cuantificado y aislado. El gating pasó a usar la cota limpia |
| 2 | F5 Moneyline se publicaba como PICK; con la cota limpia su ventaja no es significativa (z 3.65 → **1.29**) | **Alta** | Degradado a «solo probabilidad» |
| 3 | Top Picks ordenaba por «ventaja sobre la tasa base»; la auditoría demuestra que eso da **peores** picks | **Alta** | Revertido a probabilidad calibrada |
| 4 | El calendario unificado desordenaba MLB y NFL por el formato del timestamp (`T` vs espacio) | Media | Corregido con `norm_ts()` |

Además: `d_season` se colaba como feature (constante, inofensiva) → eliminada.

**Cambio de modelo de producción (§30):** moneyline MLB ya **no** usa machine
learning. Usa **Elo**, que ganó fuera de muestra (55.82 % frente al 55.38 % del
mejor ML), no depende del abridor y tiene 3 parámetros en vez de 86 features.
*La simplicidad ganó porque tuvo mejor desempeño, no por gusto.*

**Lo que sigue sin poder demostrarse:** si el modelo bate al mercado en MLB. No
hay histórico de cuotas. Se construyó la infraestructura para empezar a
acumularlo desde hoy, pero hasta que exista, cualquier afirmación sería inventada.

Pruebas: **79 pasan, 0 fallan.**

---

## 2. DATA SOURCES

### Verificación independiente (no basta con que el código lo diga)
Se volvió a pedir el dato a la API original y se comparó con lo guardado:

| Comprobación | Resultado |
|---|---|
| Marcadores y sede (3 fechas al azar) | **45 de 45 coinciden** |
| Líneas de pitcheo (5 partidos al azar) | **34 de 34 coinciden** (outs, K, BB, ER) |
| Agregados Statcast recalculados desde una descarga nueva | **3 de 3 coinciden** |
| ¿La descarga de prueba venía truncada? | No (4 593 lanzamientos, tope 25 000) |
| Claves de API guardadas en alguna URL | **0** |

Hosts contactados en toda la historia del sistema — solo tres:
`statsapi.mlb.com`, `baseballsavant.mlb.com`, `api.the-odds-api.com`.

| Fuente | Dominio | Descargas | Errores | Cobertura |
|---|---|---|---|---|
| mlb_stats_api | schedule / boxscore / lineups / injuries | 14 641 | 0 | 14 841 partidos |
| baseball_savant | statcast | 420 | 0 | 1 218 / 1 194 días con partido |
| the_odds_api | odds | 3 | 0 (2 `skipped`) | sin clave configurada |
| nflverse | NFL | 45 | 3 | motor NFL existente |

Reproducir: `python -m audit.sources`

---

## 3. DATA QUALITY

| Temporada | Partidos | Terminados | Boxscores | Días Statcast | Lanzamientos |
|---|---|---|---|---|---|
| 2021 | 2 467 | 2 466 | 2 466 | 206 | 723 172 |
| 2022 | 2 470 | 2 470 | 2 470 | 200 | 721 984 |
| 2023 | 2 471 | 2 471 | 2 471 | 205 | 732 562 |
| 2024 | 2 473 | 2 472 | 2 472 | 209 | 729 918 |
| 2025 | 2 477 | 2 477 | 2 477 | 210 | 733 376 |
| 2026 | 2 483 | 2 155 | 2 155 | 188 | 740 625 |

Cobertura de boxscores: **100 %** de los partidos terminados.

Contraste con la realidad conocida: ventaja de local **53.13 %** (real ≈53 %),
**8.90** carreras/partido (real 8.6-9.0), local gana las 5 primeras **45.02 %**.

**Protección de truncamiento (§27) vigente y verificada por prueba**: el tope real
de Baseball Savant son 25 000 filas; si un bloque lo alcanza se marca como error y
el rango se parte en dos automáticamente. Ningún día de la base llega al tope.

No se creó ningún «Data Quality Score» (§24): no había forma de justificarlo
estadísticamente. Se muestra `Complete / Partial / Missing`, que son hechos.

---

## 4. LEAKAGE AUDIT

Regla: `available_at < prediction_timestamp`, estrictamente. Timestamp idéntico →
**no se usa**. Fila sin timestamp → **se descarta** (no se puede probar que fuera
anterior).

### Inventario de features
359 columnas clasificadas por origen; **0 features prohibidas** (resultado del
partido, marcadores, totales) llegan al modelo. Riesgo declarado: 7 NULO, 352
BAJO, 0 sin clasificar tras eliminar `d_season`.

### La prueba decisiva: truncamiento partido a partido
Para cada partido que empieza en T se reconstruyen sus features con un dataset del
que se ha borrado **todo** lo que tiene `available_at >= T`. Si el constructor
usara información futura, los valores cambiarían.

| Bloque de features | Nº | Diferencias al truncar |
|---|---|---|
| Todo excepto el abridor | 243 | **0** |
| Features del abridor | 116 | 412 en 4 partidos |

**Conclusión: 243 features están perfectamente limpias.** El leakage se limita al
bloque del abridor y tiene una causa concreta y documentada: para un partido
histórico la **identidad** del lanzador se toma del boxscore de ese mismo partido,
porque el sistema no capturó los anuncios previos. Las *estadísticas* del abridor
sí son estrictamente anteriores (verificado recalculando la ERA a mano en 120
partidos).

### ¿Cuánto infla eso las métricas?
Se rehízo todo el backtest en dos variantes:

| Mercado | Con abridor | Sin abridor | Inflación |
|---|---|---|---|
| moneyline (ensemble) | 55.50 % | 55.11 % | **+0.39 pp** |
| moneyline (Elo) | 55.82 % | 55.82 % | **0.00 pp** |
| f5_moneyline (forest) | 56.91 % | 55.63 % | **+1.28 pp** |
| f5_moneyline (ensemble) | 56.51 % | 55.67 % | **+0.85 pp** |
| run_line (ensemble) | 64.40 % | 64.62 % | −0.23 pp |

### Un dato que el sistema NO tiene
Se intentó medir cuántas veces el abridor anunciado coincide con el real:
**0 anuncios con timestamp anterior al partido**. Todos los anuncios históricos se
ingirieron después de que los partidos se jugaran. Por eso **no se puede estimar**
el impacto real en producción, y el gating usa la cota conservadora.

Desde ahora el ciclo diario guarda los anuncios con su timestamp real, así que en
unas semanas esta cifra podrá medirse.

Reproducir: `python -m audit.leakage`

### NFL
| Comprobación | Resultado |
|---|---|
| Predicciones con `data_cutoff` posterior al kickoff | **0** |
| Predicciones generadas después del kickoff | **0** |
| Snapshots de cuotas capturados después de la predicción que los usó | **0** |
| Conexión de solo lectura | sqlite rechaza la escritura |

---

## 5. MLB BACKTEST

Walk-forward: para cada temporada de prueba T se entrena con todo lo anterior a
T−1, se valida en T−1 (Elo y política) y se mide en T. **Variante limpia**, sin
features del abridor. n = 9 575 (moneyline).

### Moneyline — tasa base 52.73 %
| Algoritmo | Accuracy | Log loss | Brier | ECE | z vs mayoritaria |
|---|---|---|---|---|---|
| **Elo** | **55.82 %** | **0.6819** | **0.2444** | 0.0153 | **6.06** |
| Random forest | 55.38 % | 0.6850 | 0.2459 | 0.0137 | 5.20 |
| Ensemble | 55.11 % | 0.6851 | 0.2460 | 0.0214 | 4.67 |
| Logistic | 54.83 % | 0.6852 | 0.2461 | 0.0152 | 4.11 |
| LightGBM | 54.48 % | 0.6870 | 0.2469 | 0.0122 | 3.42 |
| XGBoost | 54.15 % | 0.6872 | 0.2470 | 0.0116 | 2.78 |
| Siempre local | 52.73 % | 0.6931 | 0.2500 | 0.0273 | 0.00 |
| Siempre visitante | 47.27 % | 0.6931 | 0.2500 | 0.0273 | −10.71 |
| Aleatorio | 50.04 % | 0.9820 | 0.3309 | 0.2501 | −5.28 |

Por temporada (accuracy): Elo 55.65 / 56.07 / 56.40 / 55.08 (2023-2026).
Gana en 3 de 4 temporadas; en 2026 el forest lo supera por 0.51 pp.

### F5 Moneyline — clase mayoritaria («siempre visitante») 55.01 %
| Algoritmo | Accuracy | Log loss | ECE | z |
|---|---|---|---|---|
| Ensemble | 55.67 % | 0.6845 | 0.0159 | **1.29** |
| Random forest | 55.63 % | 0.6842 | 0.0111 | 1.21 |
| Siempre visitante | 55.01 % | 0.6931 | 0.0501 | 0.00 |
| Elo | 51.40 % | 0.6979 | 0.0810 | −7.09 |

**No significativo.** En 2026 el modelo (54.11 %) es incluso peor que decir
siempre «visitante» (54.76 %).

### Run Line — clase mayoritaria 64.39 %
Ensemble 64.62 %, **z = 0.41**. Sin ventaja de acierto. Pero el log loss mejora
+0.0092 sobre predecir la tasa base: las probabilidades informan aunque la
decisión no acierte más.

### Totales — **BLOQUEADO**
RMSE 4.501 vs 4.506 de predecir la media (mejora de **0.005 carreras**). Se
reexaminó y sigue sin aportar. Permanece bloqueado por código.

Reproducir: `python -m audit.models_compare` y `python -m audit.report_models <mercado>`

### Por qué Elo gana al machine learning
Brecha entrenamiento (OOF) → prueba, moneyline:

| Algoritmo | Brecha de accuracy | Brecha de log loss |
|---|---|---|
| Random forest | +0.98 pp | +0.0046 |
| Ensemble | +0.72 pp | +0.0028 |
| Logistic | +0.87 pp | −0.0007 |
| XGBoost | +0.61 pp | +0.0017 |

Los modelos con 86-124 features sobreajustan; Elo tiene 3 parámetros y no.

### Variantes de Elo probadas (§9)
| Variante | Accuracy | Log loss | ECE |
|---|---|---|---|
| **base ajustado por validación temporal** | 55.82 % | **0.6819** | **0.0153** |
| con ajuste por descanso (0.02) | 55.79 % | 0.6819 | 0.0157 |
| k=4, ventaja=24, regresión=0.25 (fijo) | 55.99 % | 0.6828 | 0.0247 |
| sin MOV | 55.91 % | 0.6830 | 0.0205 |
| sin regresión entre temporadas | 55.95 % | 0.6844 | 0.0341 |
| sin ventaja de local | 55.84 % | 0.6832 | 0.0288 |
| k alto (8) | 55.68 % | 0.6863 | 0.0340 |

Algunas variantes tienen accuracy marginalmente mayor pero peor calibración. Se
mantiene el Elo ajustado por validación temporal. **No se añadió el abridor al
Elo**: reintroduciría exactamente el leakage que la auditoría acaba de aislar.

---

## 6. NFL BACKTEST

Motor existente, leído en modo `ro`. Copia de seguridad creada antes de auditar:
`nflpred_20260907T212342.sqlite3` (71.8 MB).

| Estrategia | Accuracy | Log loss | Brier | Puntos fantasy |
|---|---|---|---|---|
| Modelo (producción, 90/10 mercado) | **66.87 %** | 0.6119 | 0.2119 | 71.78 % |
| Blend 80 % mercado | 66.69 % | 0.6090 | 0.2107 | 72.27 % |
| **Mercado puro** | 66.50 % | **0.6076** | **0.2102** | **72.38 %** |
| Modelo puro | 64.83 % | 0.6320 | 0.2209 | 69.14 % |
| Favorito | 63.47 % | 0.6522 | 0.2299 | 67.90 % |
| Elo | 63.28 % | 0.6377 | 0.2231 | 68.08 % |
| Aleatorio | 50.22 % | 0.9962 | 0.3350 | 49.77 % |

**Hallazgo incómodo y honesto:** el modelo NFL gana en accuracy, pero el
**mercado puro gana en log loss y en puntos fantasy** (72.38 % vs 71.78 %). En el
juego que importa —los puntos— seguir al mercado habría rendido más.

No hay predicciones NFL evaluadas todavía (la temporada acaba de empezar), así que
no hay calibración en vivo que reportar. **No se modificó nada del motor NFL.**

Reproducir: `python -m audit.nfl`

---

## 7. CALIBRATION

Buckets exigidos en §10, moneyline con Elo (n = 9 575):

| Bucket | N | Predicho | Real | Diferencia |
|---|---|---|---|---|
| 50-55 % | 4 492 | 52.4 % | 52.6 % | +0.1 pp |
| 55-60 % | 3 006 | 57.2 % | 55.4 % | −1.8 pp |
| 60-65 % | 1 458 | 62.1 % | 61.5 % | −0.7 pp |
| 65-70 % | 471 | 67.2 % | 66.5 % | −0.7 pp |
| 70-75 % | 134 | 71.9 % | 73.1 % | +1.2 pp |
| 75-80 % | 14 | 75.8 % | 64.3 % | −11.5 pp |
| 80 %+ | 0 | — | — | — |

**Sí: «70 %» significa aproximadamente 70 %.** El único bucket muy desviado tiene
n = 14, es ruido y así se marca. El sistema casi nunca supera el 75 % en MLB.

F5: el bucket 60-65 % (n = 1 678) se desvía **−3.4 pp** — es la peor calibración
del sistema y queda anotada. Run line: bien calibrado, ±1.2 pp salvo el 50-55 %
(+3.5 pp).

---

## 8. MODEL COMPARISON

Resumen de qué gana en cada mercado, con la cota limpia:

| Mercado | Mejor | Accuracy | Clase mayoritaria | Ventaja | z | Decisión |
|---|---|---|---|---|---|---|
| moneyline | **Elo** | 55.82 % | 52.73 % | +3.09 pp | 6.06 | **PICK** |
| f5_moneyline | ensemble | 55.67 % | 55.01 % | +0.66 pp | 1.29 | solo probabilidad |
| run_line | ensemble | 64.62 % | 64.39 % | +0.23 pp | 0.41 | solo probabilidad |
| total | — | — | — | — | — | **BLOQUEADO** |

`Elo` es ahora el modelo de producción de moneyline (`MLB_model_moneyline_elo v2`).
Los modelos de f5 y run line se reentrenaron **sin** features del abridor: se
publica exactamente lo que se midió.

---

## 9. MARKET COMPARISON

**MLB: no se puede hacer.** The Odds API no ofrece histórico gratuito. No hay
ninguna comparación modelo-vs-mercado en MLB en este informe porque cualquier
cifra sería inventada.

Lo que sí se hizo (§13-§14): infraestructura para construir el histórico desde hoy.
- Cada snapshot guarda `bookmaker, market, selection, line, price, timestamp, game_id`.
- Solo se inserta fila si el precio o la línea **cambiaron** (histórico de movimientos).
- `mark_closing()` etiqueta el último snapshot anterior al partido.
- **Etiquetar el cierre no lo hace utilizable**: el filtro `available_at <
  prediction_timestamp` sigue siendo lo único que decide, y hay una prueba que lo
  blinda.
- Comando: `python spc.py odds-snapshot` (pensado para 09:00, 12:00, 15:00, 18:00
  y antes del primer partido).

Estado actual: **0 snapshots**, porque no hay clave de API configurada en este
entorno. La pestaña *Data health* lo muestra y dice explícitamente que la
comparación no está lista.

**NFL sí tiene comparación** y está en la sección 6: el mercado gana en log loss y
en puntos.

---

## 10. UPSET / CASINAZO ANALYSIS

El índice compuesto se había retirado antes de esta auditoría por fallar su
validación (correlación con el error 0.008). **No se recuperó ni se inventó otro.**

Lo que se publica es la probabilidad de fallo estimada por el propio modelo
(100 − probabilidad del pick). Validación de §12:

| Mercado | Error medio declarado vs real | Monótono |
|---|---|---|
| moneyline | **0.68 pp** | sí |
| f5_moneyline | 1.69 pp | sí |
| run_line | 1.33 pp | sí |

Ejemplo (moneyline): riesgo declarado 37.9 % → fallo real 38.5 %; declarado 47.5 %
→ real 47.4 %. **Está calibrado y significa lo que dice.**

Las alertas factuales (`COIN_FLIP`, `MODEL_DISAGREEMENT`, `ELO_GAP`,
`PITCHER_UNKNOWN`, `LINEUP_UNCONFIRMED`, `DATA_INCOMPLETE`) se mantienen
**separadas** y sin puntuar, como exige §12.

---

## 11. DASHBOARD AUDIT

### El orden de Top Picks estaba mal — y lo demuestra la propia auditoría
§11 pedía comprobar si «ventaja sobre la tasa base» predice mejor rendimiento.
**No lo hace:**

| Mercado | Separación entre cuartiles por edge | Por probabilidad |
|---|---|---|
| moneyline | +8.69 pp | **+11.45 pp** |
| f5_moneyline | +3.87 pp | **+8.96 pp** |
| run_line | +8.48 pp (no monótono) | **+15.41 pp** |

Los 20 mejores picks:

| Mercado | Por edge | Por probabilidad | Diferencia |
|---|---|---|---|
| moneyline | 60.0 % | 75.0 % | **−15 pp** |
| f5_moneyline | 40.0 % | 75.0 % | **−35 pp** |
| run_line | 70.0 % | 85.0 % | **−15 pp** |

**Se revirtió.** Top Picks ordena por probabilidad calibrada, primero los mercados
con ventaja demostrada. El edge sigue visible como contexto, pero no ordena. Hay
una prueba (`test_el_edge_no_ordena_mejor_que_la_probabilidad`) que avisará si
algún día eso cambia.

### Otras comprobaciones
| Comprobación | Resultado |
|---|---|
| Endpoints de escritura expuestos | **0** (solo GET) |
| Tarjetas que cumplen el esquema único | 100 % |
| Filtro TODOS / NFL / MLB | `total = NFL + MLB` verificado |
| Orden cronológico dentro de cada día | corregido (bug del formato `T`) |
| Agrupación fecha → deporte → partidos | implementada |
| Sin oportunidades → NO PICK | verificado |
| Render de las 8 pestañas | 0 errores de consola |

Pestañas: Calendario · Top picks · Alertas · Mercados · **Performance** ·
**Modelos** · **Data health** · Estado.

---

## 12. TEST RESULTS

```
$ python spc.py test
79 passed
```

Ninguno de los 57 anteriores se eliminó. Se añadieron 22, incluidos:

- solo el bloque del abridor muestra leakage (243 features limpias)
- el gating usa la variante limpia
- F5 ya no se publica como pick
- moneyline se sirve con Elo
- `season` no es una feature
- el edge no ordena mejor que la probabilidad
- el riesgo publicado está calibrado y es monótono
- las cuotas de cierre no entran en una predicción anterior
- no se guardan claves de API en las URLs
- la guarda de truncamiento es el tope real y ningún día lo alcanza
- mismo snapshot → misma predicción
- cada predicción guarda su snapshot reproducible
- el versionado supersede y no borra (v1 → v2, sin cambio → no crea versión)
- provisional vs published según datos faltantes
- el calendario está en orden cronológico
- la base NFL sigue siendo de solo lectura y tiene copia de seguridad
- las fuentes verificadas coinciden con el original

**Dos pruebas fallaron durante la auditoría y ambas destaparon problemas reales**
(el orden del calendario y el estado de las predicciones superseded). Las dos
están corregidas.

---

## 13. KNOWN LIMITATIONS

1. **No hay comparación modelo vs mercado en MLB.** Sin histórico de cuotas no hay
   ROI, yield ni closing line value. Empezará a construirse desde el primer
   snapshot con clave configurada.
2. **La fiabilidad del abridor anunciado es desconocida.** 0 anuncios históricos
   con timestamp anterior. Por eso el gating usa la cota conservadora, que
   probablemente **subestima** el rendimiento real en producción.
3. **La política 50/50 modelo-mercado no está validada** para MLB. Se declara como
   «a priori» en cada predicción.
4. Sin histórico de lesiones MLB (la API solo publica el estado actual).
5. Sin datos de umpire, viento direccional ni dimensiones de estadio.
6. Park factors solo desde la 2ª temporada de cada sede (80.6 % de cobertura).
7. **Los márgenes son pequeños.** +3.09 pp sobre «siempre el local». En 100
   partidos eso es indistinguible de la suerte.
8. El bucket 60-65 % de F5 se desvía −3.4 pp: es la peor calibración del sistema.
9. Supuestos declarados que no son datos: fin de partido = inicio + 3 h 15 min;
   Statcast disponible al día siguiente 06:00 UTC.

---

## 14. RECOMMENDATIONS

**Inmediato**
1. Configurar `THE_ODDS_API_KEY` y programar `spc.py odds-snapshot` varias veces
   al día. Sin eso, la comparación con el mercado nunca existirá.
2. Usar los picks de **moneyline** (los únicos con ventaja demostrada). Leer F5 y
   run line como probabilidades, no como recomendaciones.

**En 4-8 semanas**
3. Medir la coincidencia entre abridor anunciado y real (el sistema ya guarda los
   datos). Si supera el 95 %, se podrá reevaluar el gating de F5 con la variante
   completa y probablemente recupere el estatus de pick.
4. Reejecutar `audit.models_compare` con la temporada 2027 como prueba.

**No hacer**
5. No desbloquear los totales para tener más picks.
6. No volver a ordenar Top Picks por edge.
7. No añadir el abridor al Elo: reintroduce el leakage aislado.
8. No sustituir Elo por machine learning en moneyline sin evidencia fuera de
   muestra que lo justifique.

---

## RESPUESTA FINAL HONESTA

### QUÉ ESTÁ DEMOSTRADO
- Los datos son reales y verificados contra la fuente original (45/45, 34/34, 3/3).
- **243 de las 359 features no tienen ningún leakage**, probado truncando el
  dataset en el instante de cada partido.
- **Moneyline MLB con Elo bate a las estrategias tontas: +3.09 pp, z = 6.06.**
- Las probabilidades están calibradas: «70 %» significa ~70 % (±1.8 pp).
- El riesgo publicado está calibrado (error medio 0.68 pp) y es monótono.
- Ordenar por probabilidad es mejor que ordenar por edge (medido, no opinado).
- El motor NFL no se tocó y no tiene leakage en sus predicciones.

### QUÉ NO ESTÁ DEMOSTRADO
- Que el modelo bata al mercado en MLB. **No hay datos para saberlo.**
- Que F5 y run line aporten ventaja de acierto (z = 1.29 y 0.41).
- Que el sistema sea rentable apostando. Nunca se ha medido.
- Que el ensemble aporte algo sobre Elo en moneyline. Los datos dicen que no.
- Cuán fiable es el abridor anunciado.

### QUÉ CAMBIÓ
- Moneyline pasa de machine learning a **Elo**.
- F5 pasa de PICK a **solo probabilidad**.
- Top Picks vuelve a ordenar por **probabilidad calibrada**.
- El gating usa la **cota limpia**, no el backtest contaminado.
- `d_season` eliminada de las features; timestamps normalizados.
- Modelos de producción reentrenados sin features del abridor (v2).
- Dashboard: Performance Center, Modelos, Data health; calendario agrupado por
  fecha → deporte.
- Infraestructura de snapshots de cuotas para construir el histórico.

### QUÉ FALLÓ
- La prueba de truncamiento reveló el leakage del abridor (116 features).
- La validación del edge tumbó mi propia decisión anterior de ordenación.
- La validación de F5 tumbó su estatus de pick.
- Dos pruebas nuevas fallaron y destaparon el bug de ordenación del calendario y
  un fallo de clasificación de estados.
- El primer intento de prueba de truncamiento dio **falsa alarma** por cortar a
  medianoche en vez de al inicio de cada partido; se corrigió el método.

### QUÉ QUEDA PENDIENTE
- Acumular histórico de cuotas (bloqueante para todo lo de mercado, ROI y CLV).
- Medir la fiabilidad del abridor anunciado.
- Reevaluar F5 cuando exista esa medición.
- Validar la política de combinación modelo-mercado en MLB.
- Calibración en vivo de NFL (la temporada acaba de empezar).

**El sistema está auditado en el sentido que pediste**: se puede demostrar que las
predicciones usan solo información disponible en el momento correcto —con la única
excepción, aislada y cuantificada, de la identidad del abridor histórico— y que las
métricas reportadas se reproducen con los comandos indicados.
