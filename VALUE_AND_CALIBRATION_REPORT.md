# VALUE_AND_CALIBRATION_REPORT

Fase A/B/C del Sports Prediction Center: valor, calibración, discrepancias, ROI,
calculadora de combinadas y chatbot analista.

Ejecutado el 2026-09-08 sobre `C:\Sports-Prediction-Center`.
Todos los números de este informe salen de ejecuciones reales. Los negativos
también están, y son la mayoría.

---

## 0. Resumen en una página

**Lo que funciona y está medido:**

- La capa de calibración, discrepancia y ROI existe, corre sobre las cinco bases
  y está probada con 94 tests nuevos.
- Se añadió un mercado nuevo con evidencia real: **tarjetas amarillas de fútbol**,
  25 237 partidos, gana a su baseline por 0.032 de log loss en selección y 0.021
  en holdout.
- La calculadora de combinadas y el chatbot funcionan de punta a punta.

**Lo que NO se ha demostrado, y hay que decirlo primero:**

- **NO existe ninguna evidencia de ventaja rentable.** En ningún deporte, en
  ningún mercado, en ningún tramo de discrepancia. La razón no es que el sistema
  pierda: es que **solo hay 11 apuestas liquidadas con cuota registrada** en todo
  el histórico. Con 11 no se demuestra nada.
- **El modelo de distribución de carreras de MLB (partido completo y F5) NO le
  gana a su baseline** y por eso no publica ninguna probabilidad.
- No hay histórico de cuotas en ningún deporte: las cuotas más antiguas de todo
  el sistema son del **2026-09-08**, o sea de hoy.

**Dos bugs encontrados y corregidos durante esta fase**, los dos capaces de
inventar ventaja donde no la hay:

1. La comparación con el mercado en `run_line` y `spread` no filtraba por línea.
   Producía los dos desacuerdos más grandes del panel (+30 pp) contra precios que
   no existían.
2. Cuotas americanas imposibles (un `-9.0` en un hándicap de tenis) se convertían
   a cuota decimal 12.1 e inventaban una ganancia del 1 100 %.

---

## 1. Arquitectura

Nada de lo existente se reescribió. Se añadió una capa que **solo lee**.

```
shared/                     capa de valor, compartida y sin estado
  calibrators.py            CALIBRATION_ENGINE_v1  Platt / isotónica / beta
  discrepancy.py            DISCREPANCY_ENGINE_v1  modelo vs mercado
  roi.py                    VALUE_TRACKER_v1       unidades, ROI, yield
  ledger.py                 LEDGER_v1              lectura unificada, mode=ro
  tracking.py               FAVORITES_60_TRACKER_v1

CHAT/                       no es un deporte: solo lee, como PARLAY
  registry.py               CHAT_MARKETS_v1   registro de mercados
  api.py                    CHAT_API_v1       API interna controlada
  intent.py                 CHAT_INTENT_v1    detección de intención
  answer.py                 CHAT_ANSWER_v1    respuestas estructuradas
  gating.py                 CHAT_GATING_v1    ¿puede publicar este mercado?

PARLAY/engine/calculator.py PARLAY_ENGINE_v1
SOCCER/models/cards.py      SOCCER_CARDS_v1
MLB/engine/distributions.py MLB_RUNS_DIST_v1
```

**Aislamiento.** `CHAT` y `PARLAY` no tienen base de datos ni modelos propios, a
propósito: así no pueden convertirse en una vía lateral de escritura entre
deportes. `shared/ledger.py` abre las cinco SQLite con `mode=ro` en la URI, de
modo que la garantía no depende de la buena voluntad del que llame.
`shared/paths.assert_writable` sigue vigilando el resto. NFL Fantasy
(`NFL_HOME`) no se toca: sigue siendo solo lectura.

**El dashboard sigue sin exponer escritura.** Los endpoints nuevos del chat y de
la calculadora se implementaron como `GET` justamente para no romper esa
garantía, que está comprobada en `tests/test_dashboard.py` y que **no se
modificó** para que pasara.

---

## 2. Calibración

`shared/calibrators.py`. Tres familias (Platt, isotónica, beta) y **tres bloques
temporales disjuntos**, no dos:

```
    ajuste (50 %)      selección (25 %)     holdout (25 %)
  se ajustan los       decide cuál gana     no se toca hasta
  tres calibradores                          el final
```

Ajustar y elegir sobre el mismo bloque garantiza que *siempre* gane algún
calibrador aunque ninguno sirva. Con tres bloques, el holdout es el único número
publicable porque es el único que no participó en ninguna decisión.

Si ninguno mejora al crudo por más de `MIN_GANANCIA = 0.0005` de log loss en el
bloque de selección, **gana el crudo** y queda documentado. Con menos de 200
casos no se calibra nada.

Todo se serializa a JSON plano, sin pickle: el calibrador se puede leer a mano.

### Crudo vs calibrado — resultado real sobre MLB

El único sitio donde había muestra suficiente para probarlo de verdad fue el
walk-forward de carreras de MLB (14 521 partidos, 2021-2026), con el calibrador
ajustado en la temporada *anterior* a la de test:

| Bloque | Config | Log loss crudo | Log loss calibrado | Mejora |
|---|---|---:|---:|---:|
| Selección | total_nbinom | 0.70695 | **0.69639** | +0.01056 |
| Selección | equipos_nbinom | 0.70982 | **0.69687** | +0.01295 |
| Selección | equipos_poisson | 0.72925 | **0.69688** | +0.03237 |
| Holdout | total_poisson | 0.70688 | **0.69123** | +0.01565 |
| Holdout | total_nbinom | 0.69431 | **0.69126** | +0.00305 |

**La calibración funciona y su efecto es grande** (hasta 0.032 de log loss). El
síntoma que la motivó era de manual: el modelo *acertaba más* que el baseline
(53.8 % vs 51.4 %) pero tenía *peor* log loss, que es exactamente lo que ocurre
cuando el lado elegido es bueno y la confianza está inflada.

Aun así no bastó para superar al baseline en el bloque de selección. Ver §7.

---

## 3. Motor de discrepancias

`shared/discrepancy.py`. Registra, por predicción y por separado:

```
MODEL_PROBABILITY_RAW          MARKET_ODDS
MODEL_PROBABILITY_CALIBRATED   MODEL_MARKET_GAP
MARKET_IMPLIED_PROBABILITY     FINAL_STATUS
market_source · fetched_at · event_time · model_version
```

**El gap se mide siempre contra la probabilidad del MODELO, nunca contra la
publicada.** MLB publica una mezcla 50/50 con el mercado; si se restara esa, el
mercado estaría en los dos lados de la resta y el gap se encogería solo. Es la
forma silenciosa de que un sistema termine siguiendo al casino, y hay un test
que lo impide (`test_la_probabilidad_publicada_no_sustituye_a_la_del_modelo`).

### Estados y de dónde salen sus umbrales

| Estado | Cuándo |
|---|---|
| `ALIGNED` | \|gap\| ≤ medio overround del mercado |
| `MINOR DISAGREEMENT` | por encima de eso y < 5 pp |
| `STRONG MODEL-MARKET DISAGREEMENT` | ≥ 5 pp |
| `UNVALIDATED` | hay desacuerdo pero el backtest no lo respalda |
| `VALIDATED VALUE` | el tramo tiene ≥ 100 casos y ROI con IC 95 % por encima de cero |
| `NO MARKET` | no hay cuota con la que comparar |

`ALIGNED` no usa un umbral inventado: se ancla al **vigorish real**. Si la
diferencia es menor que el margen que cobra la casa, no se está discrepando con
el mercado sino con su comisión. Cuando no se conoce el overround se usa 4.5 %
como cota conservadora y queda marcado en `vig_conocido`.

**`VALIDATED VALUE` es hoy inalcanzable en todos los mercados**, porque exige 100
casos liquidados con cuota y el máximo disponible es 11.

---

## 4. ROI y yield

`shared/roi.py`. Una unidad por pick por defecto. Americana y decimal. `win`,
`loss`, `push` y `void` con tratamiento distinto: `push` ocurrió y cuenta en la
muestra con profit 0; `void` es apuesta anulada y **no cuenta como apuesta**.

**Sin cuota registrada no hay ROI.** Una predicción acertada sin precio se cuenta
aparte en `sin_cuota` y jamás se le asigna una cuota típica.

Se separan explícitamente:

- **rendimiento del modelo** — log loss, Brier, ECE, acierto
- **rendimiento apostando** — ROI, yield, unidades

Un modelo perfectamente calibrado pierde dinero si siempre apuesta a cuotas
peores que su probabilidad, porque el vig se cobra en cada apuesta.

### Bug corregido: cuotas americanas imposibles

Una cuota americana no existe entre −100 y +100. Los datos reales traían un
`-9.0` en un hándicap de tenis (Sabalenka +0). Convertido daba cuota decimal
12.1, o sea una ganancia inventada del 1 100 % en una sola apuesta. Ahora se
rechaza y se cuenta como dato roto.

---

## 5. Tracking de favoritos > 60 %

`shared/tracking.py`. Buckets 60-65 / 65-70 / 70-75 / 75-80 / 80+, separados por
deporte, liga, mercado, temporada y versión de modelo. **Nunca se mezclan
mercados sin etiquetar**: el 65 % de un moneyline y el 65 % de un over de
carreras no son la misma apuesta.

Se sigue la probabilidad **calibrada del modelo**, no la publicada.

### Resultado real

```
30 de 100 predicciones calificadas superaron el 60 %

bucket   n    W-L-P   acierto   anunciada   con cuota   ROI
60-65   12    4-0-0    100 %      63.3 %         4     +56.3 %
65-70   14    4-1-0     80 %      66.8 %         5     +16.7 %
70-75    4    2-0-0    100 %      71.5 %         2     +19.1 %
75-80    0        —        —          —          0        —
80+      0        —        —          —          0        —
```

**Estado: INSUFFICIENT DATA.** 11 picks liquidados con cuota, de 30 que superaron
el umbral. Hacen falta 100 para que el ROI signifique algo.

Los aciertos sí se pueden leer (10 de 11) y son consistentes con lo anunciado.
El ROI de +56 % del primer bucket **no significa nada**: son 4 apuestas.

---

## 6. Backtesting de discrepancias

Tramos 0-3 / 3-5 / 5-8 / 8-10 / 10-15 / 15+ pp.

```
tramo       n   acierto      ROI   IC 95 % del ROI          muestra
0-3 pp     13    50.0 %   -12.1 %  -60.3 % … +36.0 %        no
3-5 pp     13    53.8 %    -9.5 %  -57.2 % … +38.2 %        no
5-8 pp      8    50.0 %   -20.1 %  -80.6 % … +40.4 %        no
8-10 pp     2   100.0 %   +64.4 %  +27.3 % … +101.5 %       no
10-15 pp    4    75.0 %   +14.5 %  -64.2 % … +93.1 %        no
15+ pp      1   100.0 %    +8.3 %  —                        no
```

**VEREDICTO: NO SE HA DEMOSTRADO VENTAJA RENTABLE en ningún tramo.**

Fíjate en el tramo 8-10 pp: ROI +64 % con un intervalo que excluye el cero. Y aun
así el sistema lo marca como insuficiente, porque **la puerta del tamaño de
muestra se comprueba antes que el intervalo**. Un intervalo calculado sobre 2
apuestas no dice nada por mucho que excluya el cero.

### Modelo contra el favorito del mercado

Muestra actual: 6 casos. Ganó el modelo 2, ganó el mercado 4. ROI siguiendo al
modelo −27.9 %, IC 95 % de −117.7 % a +61.8 %. **No permite concluir nada.**

### Bug corregido: la línea tiene que coincidir

Los dos desacuerdos más grandes del panel eran artefactos:

> Partido 824063, KC local contra AZ visitante. La predicción era **AZ +1.5** y el
> mercado solo tenía **AZ −1.5** y **KC +1.5**. Comparando sin mirar la línea
> salía un desacuerdo de **+30 pp** contra un precio que no existe.

`shared/ledger._misma_linea` ahora exige que la cuota sea del mismo lado **y de la
misma línea**. Sin coincidencia, `market_probability` queda en `None` y el estado
es `NO MARKET`, que es la verdad.

Efecto del arreglo: el tramo 15+ pp pasó de 4 casos a 1, y NFL de 68 predicciones
con cuota a 54.

---

## 7. Mercados nuevos: qué pasó el filtro y qué no

El gating (`CHAT/gating.py`) lee los walk-forward del disco y decide solo. No hay
ningún estado escrito a mano: si se reentrena y los números cambian, el estado
cambia. La regla es que el modelo le gane a su **baseline** en el bloque de
**selección** por más de 0.002 de log loss.

### ✅ Tarjetas amarillas de fútbol — PROJECTION

25 237 partidos con amarillas reales, 2012-13 a 2025-26, cinco ligas europeas.

| Bloque | Modelo | Baseline | Ganancia |
|---|---:|---:|---:|
| Selección (14 432 partidos) | 0.63881 | 0.67120 | **+0.03239** |
| Holdout (3 503 partidos) | 0.63649 | 0.65761 | **+0.02112** |

Acierto en holdout 63.72 % contra 63.60 % del baseline. Es decir: **el modelo
mejora mucho las probabilidades y casi nada la dirección** sobre la línea 4.5. Es
un mercado para calibrar expectativas, no para elegir lado.

Lo que dicen los datos y por qué el modelo está montado así:

```
por equipo y partido    media 2.0314   varianza 1.8938   INFRAdisperso
total del partido       media 4.0627   varianza 4.5442   SOBREdisperso
```

Si las tarjetas de los dos equipos fueran independientes, la varianza del total
sería 3.79 y se observa 4.54. Sobra varianza: **las tarjetas de un equipo y las
del rival están correlacionadas**, que es lo que uno esperaría de un partido
áspero. Por eso `ambos_reciben` se declara explícitamente como **cota inferior**
en lugar de corregirlo con un factor inventado.

Configuración elegida: `hgb_equipos_poisson`. Las cinco configuraciones probadas
se separaban por el **cuarto decimal** en selección (0.63854 la mejor, 0.63893 la
peor), así que elegir "la mejor" habría sido elegir ruido. Se eligió por tres
razones independientes de esa cifra: el camino por equipos es el único que da los
mercados por equipo; a nivel de equipo los datos están infradispersos y la
binomial negativa colapsa a Poisson (el walk-forward lo confirma: dan números
idénticos, 0.63649); y el GLM se descartó por no converger en 800 iteraciones, no
por su puntuación.

**Limitación que hay que decir: no hay árbitro en la base**, y es el factor
conocido más fuerte de este mercado. Cualquier ventaja que este modelo reclamara
sin árbitro sería sospechosa. Se publica como PROJECTION.

Cobertura: EPL, LALIGA, SERIEA, BUNDES, LIGUE1. **Champions, Liga MX y Saudi no
tienen fuente de tarjetas** y quedan INSUFFICIENT DATA.

### ❌ Distribución de carreras de MLB — NO PICK

14 521 partidos con marcador, 2021-2026. Se probaron 2 caminos × 2 familias × con
y sin calibración, contra el baseline de la frecuencia histórica.

| Tramo | Bloque | Mejor modelo | Baseline | Diferencia |
|---|---|---:|---:|---:|
| Completo | Selección | 0.69639 | 0.69418 | **−0.00222** |
| Completo | Holdout | 0.69123 | 0.69280 | +0.00157 |
| F5 | Selección | 0.69391 | 0.69394 | **+0.00002** |
| F5 | Holdout | 0.70254 | 0.69320 | −0.00934 |

**Los dos quedan NO PICK.** El modelo no le gana a "la frecuencia histórica de
esta línea" en el bloque de selección.

El caso de F5 es el que mejor justifica que exista el margen mínimo: ganaba por
**0.00002** en selección, y en holdout perdía por 0.00934. Sin margen se habría
publicado un modelo que en la temporada siguiente era peor que no tener modelo.

Los datos sí están y son buenos (`home_score_f5`/`away_score_f5` de 14 529
partidos reales; F5 **no** se deriva del partido completo, tiene objetivo propio).
Lo que falla es que el modelo no aporta información sobre el baseline.

Consecuencia práctica, y es exactamente lo que pediste en el punto 11:

> **Usuario:** ¿Habrá carreras en las primeras 5 entradas?
> **Chat:** No existe actualmente un modelo validado de las primeras 5 entradas.
> El modelo NO le gana al baseline en el bloque de selección (modelo 0.69391
> frente a baseline 0.69394, diferencia +0.00002). No se publica ninguna
> probabilidad de este mercado. No voy a darte un porcentaje: repartir el total
> entre entradas o usar una media serían números inventados.

---

## 8. Calculadora de combinadas

`PARLAY/engine/calculator.py`. Americana y decimal, cuota combinada, payout,
profit, probabilidad implícita combinada y probabilidad del modelo combinada.

**La probabilidad del modelo solo se calcula si TODAS las patas la tienen.**
Multiplicar unas cuantas del modelo y rellenar el resto con implícitas del
mercado daría un número que no es ni del modelo ni del mercado.

### Correlación

Siempre se declara `"Probabilidad combinada estimada BAJO INDEPENDENCIA."`, y
además se detectan tres situaciones:

| Nivel | Situación |
|---|---|
| ALTA | dos patas del **mismo partido** |
| ALTA | **participante repetido** (Sabalenka gana + Sabalenka −2.5) |
| BAJA | misma liga y misma fecha |

Con nivel ALTA el aviso es explícito: *"La probabilidad combinada mostrada NO es
correcta: supone independencia y aquí no la hay."* No se corrige con un factor
inventado, se avisa. Presentar un número de parlay como exacto habiendo
correlación sin modelar sería mentir con decimales.

Verificado en vivo: dos patas del mismo `event_id`, cuota 4.7727, payout 47.73
sobre stake 10, correlación ALTA detectada.

---

## 9. Caballo Chat

Reglas explícitas, no un modelo de lenguaje.

### El partido se elige por nombre, nunca por id

`CHAT/games.py` (CHAT_GAMES_v1) publica los **220 partidos** de los próximos 10
días con su nombre legible, sacados del mismo `_collect` que alimenta el
calendario — así no hay dos fuentes de verdad que se desincronicen.

Dos formas de indicar el partido, y ninguna pide un identificador:

- un **buscador desplegable** por equipo, liga o deporte;
- **nombrando al equipo en la propia pregunta**: *«¿cuántas tarjetas en el Union
  Berlin?»* resuelve solo.

La resolución es por **tokens del nombre**, no por parecido de cadenas. Es la
misma lección que en el resolver de clubes: la similitud de cadenas daba 0.81
entre *manchester united* y *manchester city*. Aquí el error equivalente sería
contestar del partido equivocado con total seguridad, que es peor que no
contestar.

**En un empate no se elige.** *«¿tarjetas del Manchester?»* devuelve los dos
partidos para que el usuario desempate; *«del Manchester United»* resuelve. Con
datos reales, *«el Barcelona»* devuelve 3 candidatos en vez de adivinar.

Si no se nombra ningún equipo, el chat **ofrece la lista de partidos como
botones** en vez de pedir un id que nadie puede conocer.

Caché de 60 s: recolectar el calendario cuesta ~5 s (pregunta a los cinco
motores) y el desplegable lo consulta en cada pulsación. Es una decisión: un clasificador
acertaría más en la superficie pero fallaría **en silencio**, y aquí un fallo
silencioso significa contestar con el modelo equivocado.

El chat **no consulta las bases**. Pasa por `CHAT/api.py`:

```
get_prediction · get_probability · get_distribution
get_market_comparison · get_explanation · get_secondary_markets
```

### Preguntas que ya contesta (verificadas contra el servidor)

**«¿Cuántas tarjetas se esperan?»** — partido real Union Berlin vs Schalke:

```
TARJETAS AMARILLAS
Tarjetas esperadas: 4.419
  Over 2.5: 81.7%   Under 2.5: 18.3%
  Over 3.5: 64.4%   Under 3.5: 35.6%
  Over 4.5: 45.2%   Under 4.5: 54.8%
  Over 5.5: 28.3%   Under 5.5: 71.7%
  Over 6.5: 15.9%   Under 6.5: 84.1%
Ambos reciben tarjeta: 78.8%
  local 86.5% · visitante 91.1%

modelo SOCCER_CARDS_v1 · PROJECTION · confianza moderada
holdout 0.63649 frente a 0.65761 del baseline
aviso: no hay árbitro en los datos, y es el factor conocido más fuerte.
```

**«¿Qué probabilidad hay de over 2.5 goles?»**

```
TOTAL DE GOLES
Selección: OVER 2.5
Probabilidad del modelo (calibrada): 57.6%
Implícita del mercado: 59.6%
Diferencia: -2.0 pp
aviso: la diferencia no supera la mitad del margen de la casa (2.2 pp),
       así que no es un desacuerdo real.
```

**«¿Gana o empata?»**

```
GANA O EMPATA
Selección: Union Berlin o empate
Probabilidad del modelo (calibrada): 76.9%
Mercado: sin cuota registrada para este partido.
modelo SOCCER_DC_v1 · holdout 0.57024 frente a 0.62715 · acierto 70.1 % vs 68.1 %
```

**Preguntas que rechaza, y cómo:**

| Pregunta | Respuesta |
|---|---|
| ¿Cuántas tarjetas en Liga MX? | *no hay fuente de tarjetas para LIGAMX. Solo tienen amarillas EPL, LALIGA, SERIEA, BUNDES, LIGUE1.* |
| ¿Habrá carreras en las primeras 5? | *No existe actualmente un modelo validado…* (§7) |
| ¿Cuántos home runs pega Judge? | *Entiendo lo que quieres saber, pero no de qué mercado.* + catálogo |
| texto ininteligible | *No tengo un modelo validado para eso, y prefiero decírtelo a darte un número inventado.* |

### Explicabilidad

`get_explanation` devuelve los factores **guardados con la predicción**. Si la
predicción no guardó explicación, lo dice:

> *esta predicción no guardó su explicación, y no voy a redactar una a
> posteriori: sonaría convincente sin corresponder a lo que el modelo usó de
> verdad.*

---

## 10. Mercados secundarios

Registro en `CHAT/registry.py`. Cada mercado declara `market_id`, `sport`,
`league`, `visibilidad`, `estado`, `model_version`, `distribution_type`,
`supported_lines`, `probability_method`, `calibration_status`,
`data_requirements`, `ligas_con_datos`, `evidencia` y `limitaciones`.

**Los secundarios NO aparecen en Top Picks ni en el reporte principal.** Solo se
consultan por chat. Hay un test que lo fija (`test_los_secundarios_no_son_primarios`).

Ser secundario **no rebaja el listón**: pasan por el mismo gating.

| Mercado | Visibilidad | Estado medido |
|---|---|---|
| `soccer.cards` | secondary | **projection** |
| `soccer.corners` | secondary | projection |
| `mlb.runs_dist` | secondary | **no_pick** |
| `mlb.f5_total` | secondary | **no_pick** |

Añadir tiros, tiros a puerta, faltas o props de jugador es añadir una entrada al
registro y un motor de distribución, sin tocar el chat.

---

## 11. Auditoría anti-leakage

`tests/test_leakage_value.py`, 12 pruebas.

- **Ningún constructor de features pregame lee cuotas.** Comprobado línea a línea
  sobre MLB, SOCCER, NBA, TENIS y NFL, ignorando comentarios y docstrings (NFL
  declara a propósito una lista negra `MARKET_COLS`, que es lo contrario de un
  problema).
- **El cruce de cuotas es point-in-time**: solo `available_at ≤ prediction_timestamp`.
  Una cuota posterior a la predicción es información del futuro.
- **La cuota debe ser de la misma línea** (§6).
- El ledger abre todo en `mode=ro` y no contiene `INSERT`, `UPDATE`, `DELETE`,
  `DROP` ni `CREATE TABLE`.
- Ni la capa de valor ni `CHAT/` contienen `session.add`, `commit()`,
  `to_parquet` ni `joblib.dump`.
- El registro de discrepancia guarda `market_source`, `fetched_at` y `event_time`
  de cada cuota, y la cuota nunca se convierte en la probabilidad del modelo.

Las cuotas se usan para **evaluar** (ROI, discrepancia, comparación), nunca como
feature pregame.

---

## 12. Data Health ampliado

`/api/value/coverage` responde qué hay realmente para medir:

```
deporte  predicciones  calificadas  con mercado  con cuota  calificadas+cuota
MLB               547           93           39         39                 39
NFL                68            0           34         54                  0
NBA                 0            0            0          0                  0
TENIS              23            7           23         17                  5
SOCCER           1278            0          328        328                  0
```

La última columna es la que manda: para calcular ROI hace falta que la predicción
tenga **resultado** y **cuota registrada al emitirla**. Sin las dos cosas no hay
contabilidad posible.

NBA está en cero porque no tiene predicciones activas. Fútbol tiene 1 278
predicciones pero ninguna calificada todavía (son de partidos futuros).

Los errores de consulta ya **no se tragan en silencio**: `ledger.ERRORES` los
recoge y el dashboard los muestra. Así se detectó que `nfl_odds` no tiene columna
`price_decimal` y la consulta devolvía cero filas, indistinguible de "no hay
datos".

---

## 13. Tests

```
94 tests nuevos, todos pasando:
  tests/test_value_layer.py     26   ROI, discrepancia, calibración
  tests/test_leakage_value.py   12   anti-leakage y aislamiento
  CHAT/tests/test_chat.py       46   registro, intención, API, calculadora
  CHAT/tests/test_gating.py     10   gating de mercados secundarios
```

Más `CHAT/tests/test_games.py` (11): catálogo de partidos y resolución por nombre.

**Suite completa ejecutada: `352 passed` en 129 s. 0 fallos.**
Antes de esta fase eran 247. Ninguno de los antiguos se tocó.

**No se modificó ningún test existente para hacerlo pasar.** Al contrario: el
test `test_el_dashboard_no_expone_endpoints_de_escritura` falló con mis endpoints
`POST` del chat y de la calculadora, y la solución fue **cambiar los endpoints a
`GET`**, no relajar el test. La garantía de solo-lectura vale más que la
comodidad de mandar un cuerpo JSON.

Tres bugs los encontraron los propios tests: la intención se perdía cuando el
mercado no se reconocía; el catálogo se marcaba como "no entendido"
contradiciendo su propia respuesta; y los mercados con predicción publicada
(goles, gana-o-empata) se enrutaban al motor de distribución y contestaban "no
disponible" teniendo el dato a mano.

---

## 14. Versionado

```
CALIBRATION_ENGINE_v1   DISCREPANCY_ENGINE_v1   VALUE_TRACKER_v1
LEDGER_v1               FAVORITES_60_TRACKER_v1 PARLAY_ENGINE_v1
CHAT_MARKETS_v1         CHAT_API_v1             CHAT_INTENT_v1
CHAT_ANSWER_v1          CHAT_GATING_v1
SOCCER_CARDS_v1         MLB_RUNS_DIST_v1
```

---

## 15. Comandos nuevos

```
python spc.py value [--sport MLB] [--json]   cobertura, favoritos, discrepancias
python spc.py chat "¿cuántas tarjetas?" --game <id>
python spc.py soccer-cards                    walk-forward de tarjetas
python spc.py mlb-runs                        walk-forward de carreras y F5
```

Pestañas nuevas del dashboard: **Chat**, **Modelo vs mercado**, **Calculadora**.

---

## 16. Limitaciones, en orden de importancia

1. **No hay histórico de cuotas.** Las más antiguas de todo el sistema son del
   2026-09-08. Sin eso no se puede validar ninguna discrepancia como valor, ni
   hoy ni en semanas. El sistema ya guarda snapshots hacia el futuro; es cuestión
   de dejarlo correr.
2. **11 apuestas liquidadas con cuota.** Todo el aparato de ROI está construido y
   probado, pero alimentado con 11 casos. Los porcentajes que salen no son
   ventaja, son ruido.
3. **MLB no tiene modelo de carreras publicable.** Ni partido completo ni F5.
4. **No hay árbitro para el modelo de tarjetas**, que es su factor más fuerte.
5. **Fútbol no tiene ninguna predicción calificada todavía**, así que sus 1 278
   predicciones no aportan nada a la validación.
6. **NBA no tiene predicciones activas.**
7. La combinada supone independencia entre patas de deportes distintos, calibrados
   con métodos distintos, sin verificación empírica. Se avisa, no se corrige.

---

## 17. Siguiente paso recomendado

**Dejar correr los snapshots de cuotas y no tocar nada más del motor.**

Todo lo que falta para responder "¿tengo ventaja?" es tiempo, no código. El
sistema ya registra cada predicción con su cuota y su timestamp; en cuanto haya
~100 apuestas liquidadas por tramo, `spc value` contestará solo, y contestará con
un intervalo de confianza en vez de con una opinión.

Lo segundo más rentable sería **conseguir el árbitro de cada partido de fútbol**,
que convertiría el modelo de tarjetas de PROJECTION en algo con posibilidades
reales.
