# SOCCER — informe de mercados

Módulo de fútbol del Sports Prediction Center. Aislado de NFL, MLB, NBA y TENIS.
Mercados publicados: **córners over/under**, **total de goles over/under** y
**ambos marcan (BTTS)**. El 1X2 **no se publica**: se sustituyó por córners a
petición del usuario, y además era el mercado con menos ventaja sobre su baseline.

Todos los números de este informe salen de ejecuciones reales sobre 29,917
partidos. Ninguno está estimado ni redondeado a conveniencia.

---

## 1. Ligas y datos

| Liga | Partidos | Desde | Hasta | Córners | Estado |
|---|---:|---|---|---:|---|
| Premier League | 5,340 | 2012-08-18 | 2026-08-31 | 5,320 | OK |
| LaLiga | 5,348 | 2012-08-18 | 2026-08-31 | 5,320 | OK |
| Serie A | 5,340 | 2012-08-25 | 2026-08-31 | 5,319 | OK |
| Ligue 1 | 5,013 | 2012-08-10 | 2026-08-30 | 4,996 | OK |
| Bundesliga | 4,292 | 2012-08-24 | 2026-08-30 | 4,283 | OK |
| Liga MX | 4,317 | 2012-07-20 | 2025-05-25 | **0** | **STALE** |
| Saudi Pro League | 267 | 2024-08-22 | 2025-05-02 | **0** | **INSUFFICIENT DATA** |

**Total: 29,917 partidos. 25,238 con córners.**

### Fuentes, y por qué estas
Probadas en vivo antes de elegir:

| Fuente | Resultado | Uso |
|---|---|---|
| football-data.co.uk | **503 caído** | ninguno |
| FBref (xG) | **403 bloqueado** | ninguno |
| Understat (xG) | bloqueado | ninguno |
| football-data.org | clave requerida; sin Liga MX ni Saudi | ninguno |
| `footballcsv/cache.footballdata` | funciona | histórico 2012-2024, 6 ligas |
| `openfootball` | funciona | temporadas recientes + hora de saque |
| `datasets/football-datasets` | funciona | **córners**, tiros, faltas, tarjetas |
| The Odds API | funciona (7 ligas) | calendario y cuotas actuales |

**No hay xG histórico gratuito accesible.** Las features de xG del plan original
no existen en v1 y no se han sustituido por invenciones. Cuando aparezca una
fuente, `matchstats.py` acepta columnas nuevas sin rehacer el módulo.

**No hay cuotas históricas.** Ninguna fuente accesible las trae. Esto tiene una
consecuencia dura y es la razón principal del apartado 6.

---

## 2. Problemas encontrados y corregidos

No son detalles: cada uno invalidaba resultados.

**a) Nombres de equipo entre fuentes.** `Arsenal` vs `Arsenal FC`. Cada equipo
perdía toda su historia al cambiar de fuente. Detectado porque el log loss de
2024-25 saltó de 1.06 a 1.15. Resuelto con `ingestion/teams.py`.

**b) Colisión Manchester.** La primera versión del resolvedor usaba similitud de
cadena, que da 0.81 entre `manchester united` y `manchester city`: **los fusionó
en un solo equipo**. Corregido comparando solo por tokens (da 0.5) más una regla
de margen sobre el segundo candidato.

**c) `Nott'm Forest`** se normalizaba a `nott m forest` y nunca casaba con
`Nottingham Forest`.

**d) Partidos duplicados.** Las fuentes cargaban el mismo partido con nombres
distintos: 733 filas en una temporada de 380. Los equipos actualizaban su estado
dos veces por partido. Detectado porque el holdout mejoraba de golpe sin razón.
`ingestion/dedupe.py` fusionó **3,368 duplicados con 0 conflictos de marcador**
—las fuentes coincidían en el resultado, lo que confirma que la unión era correcta.

Verificación posterior: la continuidad de equipos entre 2023-24 y 2024-25 es
17/20 en EPL, LaLiga y Serie A, 16/18 en Bundesliga, 15/18 en Ligue 1 y 18/18 en
Liga MX. Los que faltan son exactamente los ascendidos reales.

---

## 3. Anti-leakage

La garantía es estructural: el histórico se recorre en orden cronológico y para
cada partido se **lee** el estado de los equipos antes de **actualizarlo** con el
resultado. No hay ventanas calculadas sobre el dataset completo.

`available_at` = hora de saque cuando la fuente la da (3,994 partidos), 00:00 del
día del partido cuando no.

| Prueba | Resultado |
|---|---|
| Columnas prohibidas | 109 features, 0 son objetivo disfrazado |
| Correlación sospechosa (r ≥ 0.90 con el marcador) | 0 |
| `available_at` ≤ saque | 0 violaciones |
| Ventana de forma excluye el partido actual | 300 casos recalculados, 0 fallos |
| **Truncamiento** | 400 partidos reconstruidos solo con el pasado: **idénticos** |

La auditoría además **caza fugas inyectadas**: hay 4 tests que meten datos del
futuro (una feature igual al resultado, una columna `closing_odds`, un cutoff
posterior al saque, una ventana que incluye el partido actual) y exigen que la
auditoría los detecte. Una auditoría que solo dice que sí a todo no vale nada.

---

## 4. Modelos

### Distribución de goles (bloque de selección, 8 temporadas, 17,076 partidos)

| Familia | 1X2 log loss | BTTS ll | O2.5 ll | MAE goles |
|---|---:|---:|---:|---:|
| **dixon_coles** | **0.99217** | 0.68867 | 0.68276 | 1.3150 |
| bivariate_poisson | 0.99220 | **0.68860** | 0.68284 | 1.3150 |
| poisson | 0.99295 | 0.68903 | 0.68276 | 1.3150 |
| negative_binomial | 0.99417 | 0.69227 | 0.68313 | 1.3149 |

ρ de Dixon-Coles: **−0.05**, estable en las 9 temporadas. φ ≈ 11-13
(sobredispersión leve; por eso la binomial negativa no aporta en goles).
Estimador de λ elegido: **hgb_poisson** (gradient boosting con pérdida Poisson),
que batió a `ratings` (Elo/fuerzas) y a `poisson_glm`.

### Córners (holdout, 2 temporadas, 3,503 partidos)

| Configuración | log loss | Brier | Accuracy | MAE |
|---|---:|---:|---:|---:|
| hgb_total_nbinom | **0.68476** | 0.24585 | 0.5461 | 2.6924 |
| hgb_equipos_poisson | 0.68517 | 0.24605 | **0.5495** | 2.6926 |
| hgb_equipos_nbinom | 0.68564 | 0.24629 | 0.5461 | 2.6926 |
| glm_equipos_nbinom | 0.68774 | 0.24730 | 0.5449 | 2.6980 |
| **BASELINE frecuencia** | 0.69560 | — | 0.4930 | 2.7378 |

**Honestidad sobre la selección:** `glm_equipos_nbinom` ganó el bloque de
selección (0.68654) y es **el peor de los cuatro en el holdout**. Sigue batiendo
al baseline, pero es un caso claro de sobreajuste a la selección y queda escrito
aquí en vez de esconderse.

---

## 5. Contra baselines (holdout)

| Mercado | Modelo | Baseline | Mejora | Acc. modelo | Acc. baseline |
|---|---:|---:|---:|---:|---:|
| Córners O/U 9.5 | 0.68476 | 0.69560 | **−0.0108** | 54.6% | 49.3% |
| Total goles O/U 2.5 | 0.67917 | 0.69119 | **−0.0120** | 56.7% | 53.6% |
| BTTS | 0.68529 | 0.68936 | −0.0041 | 55.2% | 54.6% |

Los tres superan a su baseline. La ventaja es **pequeña**: entre 0.004 y 0.012
nats. En accuracy, córners y total de goles ganan de forma visible; **BTTS apenas
mejora medio punto**, que es poco más que ruido.

---

## 6. Estado de cada mercado — y por qué ninguno es PICK

| Mercado | Estado | Motivo |
|---|---|---|
| Córners | **PROYECCIÓN** | supera al baseline, pero sin histórico de cuotas no se puede demostrar ventaja frente al mercado |
| Total de goles | **PROYECCIÓN** | igual |
| BTTS | **PROYECCIÓN** | igual, y con la mejora más pequeña de los tres |

Esta regla está escrita en `markets/gating.py` a propósito: **mientras no haya
histórico de cuotas propias, ningún mercado puede pasar de PROYECCIÓN.** Superar
a una frecuencia histórica no es lo mismo que superar a una casa de apuestas.
Llamar PICK a esto sería exactamente el error que el sistema debe evitar.

El sistema empezó a guardar snapshots de cuotas con timestamp real. Cuando haya
unos 3 meses acumulados se podrá medir contra el cierre y reevaluar el estado.

### Por liga

| Liga | Córners | Total goles | BTTS |
|---|---|---|---|
| Premier League | PROYECCIÓN | PROYECCIÓN | PROYECCIÓN |
| LaLiga | PROYECCIÓN | PROYECCIÓN | PROYECCIÓN |
| Serie A | PROYECCIÓN | PROYECCIÓN | PROYECCIÓN |
| Bundesliga | PROYECCIÓN | PROYECCIÓN | PROYECCIÓN |
| Ligue 1 | PROYECCIÓN | PROYECCIÓN | PROYECCIÓN |
| Liga MX | **INSUFFICIENT DATA** | PROYECCIÓN | PROYECCIÓN |
| Saudi Pro League | **INSUFFICIENT DATA** | **INSUFFICIENT DATA** | **INSUFFICIENT DATA** |

---

## 7. Auditoría por liga (holdout) — dónde funciona y dónde no

Formato: log loss / accuracy / baseline.

| Liga | n | Córners | Total 2.5 | BTTS |
|---|---:|---|---|---|
| Serie A | 760 | **0.67995 / 57.0% / 0.70316** | 0.69691 / 51.4% / 0.69598 ✗ | 0.69685 / 49.7% / 0.69682 ✗ |
| EPL | 760 | 0.68281 / 55.5% / 0.68531 | 0.68165 / 57.5% / 0.68962 | 0.68573 / 56.0% / 0.68691 |
| LaLiga | 760 | 0.68637 / 55.5% / 0.69795 | 0.68021 / 56.3% / 0.69427 | 0.68599 / 55.0% / 0.68848 |
| Bundesliga | 612 | 0.69224 / 54.0% / 0.69288 ≈ | **0.64858 / 63.9% / 0.68531** | **0.67283 / 57.8% / 0.68379** |
| Ligue 1 | 612 | 0.69035 / 52.1% / 0.69985 | 0.67863 / 57.0% / 0.69085 | 0.68937 / 53.4% / 0.69044 |
| Liga MX | 338 | sin datos | 0.68545 / 57.1% / 0.69038 | 0.68339 / 56.8% / 0.68963 |
| Saudi | 267 | sin datos | 0.68244 / 56.9% / 0.68880 | 0.68052 / 56.9% / 0.68799 |

Lo que hay que leer aquí, sin adornos:

- **Serie A es la mejor liga para córners** (57.0% frente a un baseline de 50%)
  y a la vez **la peor para goles y BTTS: el modelo pierde contra el baseline**.
  Marcado con ✗.
- **Bundesliga es lo contrario**: el mejor total de goles del sistema (63.9%) y
  córners que **no aportan nada** (0.69224 vs 0.69288 ≈ empate).
- Ligue 1 en córners está en 52.1%: prácticamente una moneda.
- Saudi y Liga MX aparecen aquí solo como referencia; sus mercados están
  bloqueados por falta de datos y **no generan predicciones publicables**.

Ninguna liga se ha ocultado.

---

## 8. Tiempo real

`SOCCER/ingestion/live.py` está escrito contra las 7 claves de The Odds API,
verificadas activas: `soccer_epl`, `soccer_spain_la_liga`, `soccer_italy_serie_a`,
`soccer_germany_bundesliga`, `soccer_france_ligue_one`, `soccer_mexico_ligamx`,
`soccer_saudi_arabia_pro_league`.

Guarda `fetched_at` con la hora real de recepción y `bookmaker_update` solo si la
casa la declara. Si una liga falla, se registra el error y el ciclo continúa.

### API EJECUTADA DE VERDAD — 2026-09-09 01:51:06 UTC

| | |
|---|---|
| **A) Endpoint** | `api.the-odds-api.com/v4/sports/{liga}/events \| odds \| scores` |
| **B) Última consulta correcta** | 2026-09-09 01:51:06 |
| **C) Partidos próximos encontrados** | **119** |
| **D) Snapshots de cuotas guardados** | **17,802** de **49 casas** |
| | 119 de 119 partidos con cuotas (100%) |
| **E) Predicciones actuales generadas** | **234** |
| | 172 PROYECCIÓN · 52 INSUFFICIENT DATA · 10 NO PICK |
| **F) Créditos restantes de la API** | 308 |

Por liga: EPL 20, LaLiga 20, Serie A 20, Ligue 1 18, Bundesliga 18, Saudi 12,
Liga MX 11 — **todas con cuotas**.

**Error real encontrado y corregido:** pedir `markets=h2h,totals,btts` en una sola
llamada devolvía **HTTP 422 en las siete ligas** y dejaba el ciclo sin ninguna
cuota. Ahora los mercados se piden por grupos independientes. Resultado: `h2h` y
`totals` funcionan; **`btts` da 422 en las siete ligas — The Odds API no ofrece
ese mercado para fútbol**. Es una limitación real de la fuente, no un fallo del
módulo: BTTS se publica sin comparación de mercado.

### Ejemplos reales del calendario actual

```
SERIE A · Venezia vs Fiorentina · saque 2026-09-11 18:45 UTC
  CÓRNERS  (9.5)  UNDER 9.5   50.1%  PROYECCIÓN
                  esperados 9.87 | O7.5=70% O8.5=60% O9.5=50% O10.5=40% O11.5=31%
  TOTAL    (2.5)  UNDER 2.5   55.7%  PROYECCIÓN
                  goles esperados 2.45 (0.97-1.48)
                  mercado 53.1% (Matchbook) gap +2.6 pp [INFORMATIVO]
  BTTS            NO          51.4%  PROYECCIÓN
  modelos SOCCER_CORNERS_v1 / SOCCER_TOTAL_v1 / SOCCER_BTTS_v1
  prediction_timestamp 2026-09-09 01:51:07

LALIGA · Sevilla vs Valencia · saque 2026-09-11 19:00 UTC
  CÓRNERS  (9.5)  UNDER 9.5   53.7%  PROYECCIÓN  (esperados 9.51)
  TOTAL    (2.5)  OVER 2.5    50.6%  PROYECCIÓN  (2.70 = 1.44-1.26)
                  mercado 40.3% (1xBet) gap +10.3 pp [INFORMATIVO]
  BTTS            SI          55.2%  PROYECCIÓN
```

Saudi aparece en el calendario con sus 12 partidos y sus cuotas, pero sus tres
mercados salen **INSUFFICIENT DATA** y no publican probabilidad. Funciona como se
diseñó.

---

## 9. Data Health

| Liga | Estado | Última fecha | Antigüedad | Córners |
|---|---|---|---:|---:|
| Premier League | OK | 2026-08-31 | 8 días | 5,320 |
| LaLiga | OK | 2026-08-31 | 8 días | 5,320 |
| Serie A | OK | 2026-08-31 | 8 días | 5,319 |
| Bundesliga | OK | 2026-08-30 | 9 días | 4,283 |
| Ligue 1 | OK | 2026-08-30 | 9 días | 4,996 |
| Liga MX | **STALE** | 2025-05-25 | **471 días** | 0 |
| Saudi | **INSUFFICIENT DATA** | 2025-05-02 | 494 días | 0 |

Avisos que el dashboard muestra tal cual:
- Liga MX: el histórico termina el 2025-05-25 (471 días): la forma de los equipos está vieja.
- Liga MX: sin fuente de córners, ese mercado no se publica.
- Saudi Pro League: histórico insuficiente (267 partidos), no se generan picks.
- Todavía no hay ningún snapshot de cuotas guardado.

Saudi y Liga MX se activan solas cuando acumulen datos: `MIN_PARTIDOS_ENTRENAR`
es un umbral y `_estado_ligas()` recalcula el estado en cada ingesta. Los partidos
terminados del calendario en vivo pasan al histórico propio (`absorber_resultados`),
así que el hueco se cierra con el uso, sin tocar arquitectura.

---

## 10. Tests

**58 tests propios de SOCCER.** Cubren aislamiento (5), ligas (3), anti-leakage
con inyección de fugas (11), nombres de equipo (4), distribución de goles (14),
córners (7), gating (4) y datos reales cargados (4).

**Suite completa en la máquina del usuario: 249 pasan, 0 fallan** (191 previos +
58 de fútbol). NFL, MLB, NBA, TENIS y PARLAY siguen intactos.

`pytest.ini` se amplió para incluir `SOCCER/tests`; sin eso los 58 no se recogían
y la suite reportaba 191 como si nada hubiera cambiado.

---

## 11. Comandos

```
python spc.py soccer-train     # histórico + córners + dedupe + features + auditoría + modelos
python spc.py soccer-cycle     # calendario + cuotas + resultados + predicciones ACTUALES
python spc.py soccer-gating    # recalcula qué publica cada liga
python spc.py score            # califica lo jugado (incluye fútbol)
```

---

## 12. Lo que falta y lo que no se hizo

- **La API no se ha ejecutado** con clave real. Es lo primero que hay que hacer.
- **Sin xG**: no hay fuente gratuita accesible. No se ha sustituido por nada.
- **Sin cuotas históricas**: por eso ningún mercado pasa de PROYECCIÓN.
- **Liga MX sin córners** y con histórico de hace 15 meses.
- **Saudi con 267 partidos**: bloqueada.
- El estimador de córners elegido en selección resultó el peor en holdout.
- Calibración por bucket calculada dentro del walk-forward; falta publicarla como
  tabla independiente en el informe.

## 13. Veredicto

El mercado de **córners es el más prometedor** de los tres (54.6% en holdout
frente a 49.3% del baseline), y el de **total de goles el más consistente por
liga**. **BTTS apenas aporta** y sería el primer candidato a bloquear si el
patrón se mantiene.

Pero la ventaja medida es frente a una frecuencia histórica, **no frente a una
casa de apuestas**. Con esta evidencia, ninguno de los tres mercados justifica un
PICK, y el sistema no lo permite. Eso no es un fallo del módulo: es el resultado
honesto.
