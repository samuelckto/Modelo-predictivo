# Sports Prediction Center — NFL + MLB

Dos motores de predicción independientes bajo un solo dashboard.
**NFL** (sistema existente, envuelto en solo lectura) y **MLB** (motor nuevo,
construido con datos reales de MLB Stats API y Baseball Savant).

> Los números de este documento salen de comandos realmente ejecutados.
> La auditoría completa, incluidas **las pruebas que fallaron y los hallazgos que
> obligaron a cambiar decisiones**, está en
> [`FINAL_AUDIT_REPORT.md`](FINAL_AUDIT_REPORT.md).
> El informe anterior a la auditoría se conserva en `AUDIT_REPORT.md` para poder
> comparar qué cambió y por qué.

---

## 1. Qué hace el sistema hoy

```
$ python spc.py markets
moneyline     PICK con Elo         z=6.06  ventaja +3.09 pp sobre "siempre el local"
f5_moneyline  SOLO PROBABILIDAD    z=1.29  NO significativo tras la auditoria
run_line      SOLO PROBABILIDAD    z=0.41  sin ventaja de acierto
total         BLOQUEADO            no mejora frente a predecir la media
```

> **Estas cifras son posteriores a la auditoria** y salen de la variante limpia,
> sin las features del abridor. Ver [`FINAL_AUDIT_REPORT.md`](FINAL_AUDIT_REPORT.md).

El sistema **decide solo** qué mercados puede publicar, comparando su backtest
contra la estrategia de decir siempre lo mismo. Los totales están bloqueados por
código porque no aportan nada.

```
$ python spc.py test
79 passed
```

---

## 2. Datos (ejecutado, no estimado)

| Temporada | Partidos | Boxscores | Días Statcast | Lanzamientos |
|---|---|---|---|---|
| 2021 | 2 467 | 2 466 | 206 | 723 172 |
| 2022 | 2 470 | 2 470 | 200 | 721 984 |
| 2023 | 2 471 | 2 471 | 205 | 732 562 |
| 2024 | 2 473 | 2 472 | 209 | 729 918 |
| 2025 | 2 477 | 2 477 | 210 | 733 376 |
| 2026 | 2 483 | 2 155 | 188 | 740 625 |

**14 841 partidos · 390 features · 15 063 descargas registradas · 0 errores.**

Verificación contra la realidad: ventaja de local **53.13 %** (real ~53 %),
**8.90** carreras por partido (real 8.6-9.0).

---

## 3. Resultados reales del backtest walk-forward

Temporadas de prueba **2023-2026**, n = 9 574 por mercado. Cada temporada de
prueba nunca se usó para entrenar ni para elegir nada.

### Moneyline — el local gana el 52.73 % de las veces
| Algoritmo | Accuracy | Log loss | Brier | ECE | z |
|---|---|---|---|---|---|
| **Elo** (produccion) | **55.82 %** | **0.6819** | **0.2444** | 0.0153 | **6.06** |
| Random forest | 55.38 % | 0.6850 | 0.2459 | 0.0137 | 5.20 |
| Ensemble | 55.11 % | 0.6851 | 0.2460 | 0.0214 | 4.67 |
| Logistic | 54.83 % | 0.6852 | 0.2461 | 0.0152 | 4.11 |
| LightGBM | 54.48 % | 0.6870 | 0.2469 | 0.0122 | 3.42 |
| XGBoost | 54.15 % | 0.6872 | 0.2470 | 0.0116 | 2.78 |
| Siempre local | 52.73 % | 0.6931 | 0.2500 | 0.0273 | 0.00 |
| Aleatorio | 50.04 % | 0.9820 | 0.3309 | 0.2501 | −5.28 |

**El Elo simple gana a todo el machine learning**, y ademas no depende del
abridor. Por eso es el modelo de produccion. La simplicidad gano porque tuvo
mejor desempeno fuera de muestra, no por gusto.

### Primeras 5 entradas — degradado por la auditoria
| Algoritmo | Accuracy | Log loss | z |
|---|---|---|---|
| Ensemble | 55.67 % | 0.6845 | **1.29** |
| Random forest | 55.63 % | 0.6842 | 1.21 |
| Siempre visitante | 55.01 % | 0.6931 | 0.00 |

Antes de la auditoria F5 se publicaba como PICK con z=3.65. Esa ventaja venia de
las features del abridor, que en el historico usan informacion del propio partido.
Con la cota limpia **deja de ser significativa** y pasa a «solo probabilidad».

### Run line y totales
- **Run line**: acierta 64.62 %… pero *decir siempre «no cubre» acierta 64.39 %*.
  Ventaja real **+0.23 pp (z=0.41)**. Solo probabilidad, nunca pick.
- **Totales**: RMSE 4.501 vs 4.506 de predecir la media. **Bloqueado.**

---

## 4. Lo que falló (y por qué está aquí)

| Cosa | Resultado | Qué se hizo |
|---|---|---|
| Features del abridor | **Leakage real**: la identidad del lanzador sale del boxscore del propio partido | Aislado y cuantificado (+0.39 a +1.28 pp). El gating usa la cota limpia |
| F5 como PICK | **Falló** al quitar el leakage: z 3.65 → 1.29 | Degradado a «solo probabilidad» |
| Orden de Top Picks por «edge» | **Peor** que por probabilidad (−15 a −35 pp en el top-20) | Revertido a probabilidad calibrada |
| Orden del calendario NFL+MLB | **Bug**: formato `T` vs espacio desordenaba | Corregido con `norm_ts()` |
| `d_season` como feature | Constante colada por el bucle de diferenciales | Eliminada |
| Índice compuesto de Upset Risk | **Falló**: correlación con el error 0.008 | Retirado. Se publica la probabilidad de fallar, validada |
| Comparación modelo vs mercado MLB | **Imposible**: no hay histórico gratuito de cuotas | Declarado; infraestructura lista para acumularlo |
| Descarga de Statcast en bloques de 8 días | **Truncada** en 25 000 filas | 328 134 filas borradas y rehechas con guarda automática |
| Ventana "temporada" en las features | Arrastraba el año anterior | Corregido; lo detectó una prueba |

---

## 5. Anti-leakage

Para un partido que empieza en T solo entran filas con `available_at < T`
(estrictamente: mismo instante = no disponible). Filas sin timestamp se descartan.

Verificado **contra los datos reales**, no solo con datos sintéticos:
- El OBP se recalculó a mano en 150 partidos con un filtro independiente: coincide.
- La ERA del abridor se recalculó en 120 partidos: coincide, y su propia apertura
  nunca está dentro.
- El primer partido de cada equipo no tiene medias acumuladas.

**Supuestos declarados** (no son datos): fin de partido = inicio + 3 h 15 min;
Statcast disponible al día siguiente 06:00 UTC; identidad del abridor histórico =
quien lanzó realmente (la MLB anuncia los probables con antelación, pero no
siempre coinciden — la columna `sp_source` lo distingue).

---

## 6. Arquitectura

```
Sports-Prediction-Center/
├── NFL/adapter.py       envuelve nfl-prediction-app en SOLO LECTURA
├── MLB/
│   ├── database/        21 tablas propias, SQLite independiente
│   ├── ingest/          MLB Stats API, Baseball Savant, The Odds API, boxscores
│   ├── features/        constructor point-in-time (390 columnas)
│   ├── engine/          elo · models · gating · uncertainty · explain · predict · daily
│   ├── backtests/       walk-forward por partes + validación del riesgo
│   └── tests/
├── shared/              odds · calibration · blend · asof · sources · schema · paths
├── dashboard/           API de solo lectura + React (6 pestañas)
└── tests/
```

**No se comparte** entre deportes: bases, modelos, features, predicciones,
entrenamiento ni configuración. `shared/` solo tiene matemática sin estado.

El sistema NFL **no se tocó**: se abre su SQLite con `mode=ro`, así que es el
propio sqlite quien rechaza cualquier escritura.

---

## 7. Cómo se ejecuta

```powershell
cd Sports-Prediction-Center
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # editar NFL_HOME y THE_ODDS_API_KEY

python spc.py init          # esquema MLB
python spc.py test          # 57 pruebas
python spc.py markets       # qué mercados publica y por qué
python spc.py serve         # http://127.0.0.1:8100
```

### Primera carga de datos MLB (varias horas)
```powershell
python -m MLB.ingest.resume --seasons 2021 2022 2023 2024 2025 2026 --minutes 600
python -m MLB.ingest.fill_statcast --seasons 2021 2022 2023 2024 2025 2026 --minutes 600
```
Es **reanudable**: si se corta, se vuelve a lanzar y sigue donde iba.

### Uso diario
```powershell
python spc.py mlb-daily     # datos + features + predicciones + resultados
```
Actualiza calendario, lesiones, cuotas, resultados, bullpen y alineaciones;
reconstruye features; predice; y puntúa lo ya jugado. Cuando cambia algo
importante (se confirma una alineación, cambia el abridor) **crea una versión
nueva** de la predicción y conserva la anterior.

### Reentrenar
```powershell
python -m MLB.backtests.run_wf --minutes 600      # backtest walk-forward
python -m MLB.backtests.run_wf --merge
python -m MLB.engine.train_production             # modelos de producción (v2)
```

### Reproducir la auditoría completa
```powershell
python -m audit.sources           # verifica las fuentes re-descargando muestras
python -m audit.leakage           # inventario + prueba de truncamiento
python -m audit.models_compare    # todos los algoritmos, con y sin abridor
python -m audit.report_models moneyline
python -m audit.elo_variants      # por qué gana Elo
python -m audit.gen_preds         # predicciones walk-forward partido a partido
python -m audit.calibration_edge  # calibración, edge y riesgo
python -m audit.nfl               # auditoría NFL (crea copia de seguridad)
```

### Construir el histórico de mercado
```powershell
python spc.py odds-snapshot       # ejecutar varias veces al día
```

---

## 8. El dashboard

| Pestaña | Qué muestra |
|---|---|
| **Calendario** | agrupado por día → deporte → partidos, en hora local |
| **Top picks** | ordenados por probabilidad calibrada; primero los mercados validados |
| **Alertas** | avisos factuales agrupados por tipo |
| **Mercados** | qué publica el sistema y con qué evidencia |
| **Performance** | filtros por mercado/modelo/temporada/fecha + curva de calibración, accuracy por mes, Brier y log loss, acierto acumulado |
| **Modelos** | una fila por deporte, mercado y modelo |
| **Data health** | estado de cada fuente, cobertura y estado del histórico de mercado |
| **Estado** | motores, prioridad de fuentes, prueba de aislamiento |

**Por qué Top Picks ordena por probabilidad y no por «edge»**: se probó. Los 20
mejores picks por edge acertaron 60 % frente al 75 % de los 20 mejores por
probabilidad en moneyline (−35 pp en F5, −15 pp en run line). El edge se muestra
como contexto, pero no ordena.

---

## 9. Fuentes

| Deporte | Dominio | Prioridad |
|---|---|---|
| MLB | calendario, boxscore, alineaciones, lesiones | MLB Stats API |
| MLB | Statcast (xwOBA, barrels, velocidad, spin, CSW) | Baseball Savant |
| MLB | cuotas | The Odds API |
| NFL | todo | nflverse |

Cada descarga deja fila en `data_source_logs` con fuente, URL, estado, filas,
`retrieved_at`, `checksum` y error. Si dos fuentes discrepan se guarda el
conflicto en `source_conflicts`; **nunca se decide en silencio**. La clave de API
se recorta de la URL antes de registrarla.

---

## 10. Lo que este sistema NO es

- **No dice quién va a ganar.** Da probabilidades calibradas.
- **No se ha demostrado que sea rentable apostando.** Sin cuotas históricas MLB
  no se puede calcular ROI ni closing line value. No se calcula, no se afirma.
- **El margen es pequeño.** +3.1 pp sobre "siempre el local". En 100 partidos eso
  es indistinguible de la suerte.
- **No sustituye tu criterio.** Cuando faltan datos lo dice y marca la predicción
  como provisional en vez de rellenar el hueco.

Si el mercado supera al modelo, el informe lo dice. Si el Elo supera al machine
learning —como ocurre en moneyline— el informe también lo dice.

## Mercados NFL propios: Total y Spread (2026-09)

`NFL/markets/` añade proyección de puntos totales y de margen para cada partido
NFL, con P(over/under) y P(cubre) calibradas frente a la línea, en la MISMA
tarjeta que el moneyline del motor NFL existente. El motor original no se toca
(solo lectura, verificado por hash). Estado por evidencia (walk-forward
2020-2025 contra la línea de cierre): **PROYECCIÓN, no pick** — ver
`NFL_MARKETS_REPORT.md`.

```
python spc.py nfl-train     # investigación walk-forward + entrenamiento
python spc.py nfl-cycle     # cuotas NFL + predicciones + calificación (diario)
```

## Módulo NBA (2026-09)

`NBA/` añade moneyline, spread y total con base, modelos y features propios (datos de
pbpstats.com 2015-16 → 2025-26). Estado por evidencia walk-forward: **moneyline PICK
(umbral 55 %)**, spread y total **PROYECCIÓN**. Detalle en `NBA_MARKETS_REPORT.md`.
```
python spc.py nba-train      # regenerar datos, investigación y modelos
python spc.py nba-cycle      # ciclo diario en temporada
```

## Módulo TENIS (2026-09)

`TENIS/` añade ATP y WTA con tres mercados derivados de un modelo punto a punto
exacto: ganador, total de juegos y hándicap de juegos. Estado por evidencia
walk-forward (holdout 2020-2026): **ganador PICK en ambos circuitos** (69.1 % ATP
y 67.3 % WTA en los picks, por encima del favorito Elo con significancia),
juegos y hándicap PROYECCIÓN, y total de juegos WTA BLOQUEADO.
Detalle en `TENIS_MARKETS_REPORT.md`.
```
python spc.py tenis-train    # datos, investigación y modelos
python spc.py tenis-cycle    # ciclo diario
```

## Combinadas (parlays)

`PARLAY/` genera combinadas de 2 y 3 patas con la mayor probabilidad de acertar,
del mismo deporte o mezcladas, y las sigue hasta calificarlas. Reglas fijas:
solo mercados con estado **PICK**, nunca dos patas del mismo partido **ni con un
equipo/jugador repetido** (evita correlación), y la probabilidad mostrada es el
producto de las probabilidades calibradas. Base propia; no escribe en ningún
deporte. Pestaña «Combinadas» y gráficas de acierto real vs anunciado en Histórico.
```
python spc.py parlays        # genera y califica (el servidor lo hace solo cada 10 min)
```
