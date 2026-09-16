import React from 'react'
import { pct, localTime } from '../format'

// TARJETA DE PARTIDO — una por partido, con sus mercados.
//
// Rediseño: la tarjeta enseña lo que se mira de un vistazo (equipos, hora,
// selección y probabilidad) y guarda lo técnico dentro de un desplegable.
//
// Lo que NO se escondió, a propósito:
//   · el estado (NO PICK, BLOQUEADO, SIN DATOS) — es la diferencia entre
//     "el modelo dice esto" y "el modelo no se pronuncia";
//   · el resultado real cuando ya se jugó;
//   · la proyección numérica, que es la predicción en sí.
// Lo que sí se escondió es la procedencia (modelo/mercado/gap, origen de la
// línea): sigue a un clic, deja de competir con la respuesta.

const ORDER = { NFL: ['moneyline', 'total', 'spread'], MLB: ['moneyline', 'total', 'run_line'],
                NBA: ['moneyline', 'spread', 'total'], TENIS: ['winner', 'total_games', 'handicap_games'],
                SOCCER: ['double_chance', 'total_goals', 'btts'] }
const TITLE = { moneyline: 'Ganador', total: 'Total', spread: 'Spread', run_line: 'Run line',
                winner: 'Ganador', total_games: 'Total de juegos', handicap_games: 'Hándicap',
                corners: 'Córners', total_goals: 'Total de goles', btts: 'Ambos marcan',
                double_chance: 'Gana o empata' }
const ACCENT = { NFL: 'var(--nfl)', MLB: 'var(--mlb)', NBA: 'var(--nba)', TENIS: 'var(--tenis)',
                 SOCCER: 'var(--soccer)' }
const EMOJI = { NFL: '🏈', MLB: '⚾', NBA: '🏀', TENIS: '🎾', SOCCER: '⚽' }
const CONF_CLS = { ALTA: 'risk-low', MEDIA: 'risk-med', BAJA: 'risk-high' }

// Confianza = la misma banda para todos los mercados y deportes, derivada de la
// probabilidad calibrada del lado mostrado (terciles reales de MLB/NFL).
export const confianza = p => p == null ? null : p >= 0.62 ? 'ALTA' : p >= 0.545 ? 'MEDIA' : 'BAJA'

const UNIDAD = { TENIS: 'juegos', MLB: 'carreras', NBA: 'puntos', NFL: 'puntos', SOCCER: 'goles' }
const TOTALES = ['total', 'total_games', 'total_goals', 'corners']
const MARGENES = ['spread', 'run_line', 'handicap_games']

// Que paso DE VERDAD en ese mercado. Sale del valor real guardado al calificar
// (actual_value), nunca de un texto fijo: sirve igual para cualquier partido.
function real(c, ex, k, sp) {
  const av = ex.actual_value != null ? ex.actual_value : c.actual_value
  if (!c.result || av == null) return null
  const u = UNIDAD[sp] || 'puntos'
  const v = Number(av)
  const ln = ex.line != null ? Number(ex.line) : (c.line != null ? Number(c.line) : null)
  if (TOTALES.includes(k)) {
    const cmp = ln == null ? '' : ` · línea ${ln}: quedó ${v > ln ? 'OVER' : v < ln ? 'UNDER' : 'EMPATE'}`
    return `Resultado real: ${v} ${u}${cmp}`
  }
  if (MARGENES.includes(k)) {
    const need = ln == null ? '' : ` · con la línea ${ln >= 0 ? '+' : ''}${ln} hacía falta un margen mayor que ${-ln}`
    return `Margen real: ${v >= 0 ? '+' : ''}${v} ${u}${need}`
  }
  return null
}

function Res({ c }) {
  if (c.result === 'win') return <span className="res ok">✓</span>
  if (c.result === 'loss') return <span className="res no">✗</span>
  if (c.result === 'push') return <span className="res pend">=</span>
  return null
}

function Marcador({ c, lv }) {
  if (!lv || lv.estado === 'programado') return null
  const st = lv.estado
  const cls = st === 'en_vivo' ? 'live' : st === 'finalizado' ? 'final' : 'post'
  const label = st === 'en_vivo' ? 'En vivo' : st === 'finalizado' ? 'Final' : 'Pospuesto'
  return (
    <div className="score">
      <div>
        <div className="teams">{c.away} {lv.away ?? '–'} — {lv.home ?? '–'} {c.home}</div>
        {st === 'en_vivo' && <div className="inn">{lv.entrada || ''}{lv.outs != null ? ` · ${lv.outs} out${lv.outs === 1 ? '' : 's'}` : ''}</div>}
        {st === 'finalizado' && lv.entradas_jugadas > 9 && <div className="inn">{lv.entradas_jugadas} entradas</div>}
        {st === 'pospuesto' && <div className="inn">{lv.detalle}</div>}
      </div>
      <span className={'st ' + cls}>{label}</span>
    </div>
  )
}

export function agrupar(cards) {
  const m = new Map()
  for (const c of cards) {
    const k = `${c.sport}-${c.game_id}`
    if (!m.has(k)) m.set(k, { key: k, sport: c.sport, game_id: c.game_id, home: c.home, away: c.away,
                             start_utc: c.start_utc, venue: c.venue, tour: (c.extra || {}).tour, markets: {} })
    const mk = (c.extra && c.extra.market_key) || String(c.market || '').toLowerCase()
    m.get(k).markets[mk] = c
  }
  return [...m.values()]
}

// Estado del mercado cuando no hay tarjeta: distingue "bloqueado por evidencia"
// de "todavia sin prediccion". Sin esto un hueco parece un fallo del sistema.
function estadoMercado(mk, sport, tour, key) {
  if (!mk) return null
  const d = sport === 'TENIS' ? (mk.TENIS || {})[tour] : mk[sport]
  const g = d && d[key]
  if (!g) return null
  return { mode: g.mode, reason: g.reason }
}

// La frase de proyección de cada deporte/mercado. Es la predicción en sí, así
// que se queda a la vista; lo que se colapsa es de dónde salió.
function proyeccion(sp, k, ex, game) {
  const n = (v, d = 1) => Number(v).toFixed(d)
  if (sp === 'NFL' && k === 'total') return `Proyección: ${n(ex.expected_value)} puntos`
  if (sp === 'NFL' && k === 'spread') return `Margen proyectado: ${game.home} ${ex.expected_value >= 0 ? '+' : ''}${n(ex.expected_value)}`
  if (sp === 'MLB' && k === 'total' && ex.total) return `Carreras esperadas: ${n(ex.total.proyeccion)}`
  if (sp === 'NBA' && k === 'total' && ex.expected_value != null) return `Total proyectado: ${n(ex.expected_value)}`
  if (sp === 'NBA' && k === 'spread' && ex.expected_value != null) return `Margen proyectado: ${ex.expected_value >= 0 ? game.home : game.away} +${n(Math.abs(ex.expected_value))}`
  if (sp === 'TENIS' && k === 'total_games' && ex.expected_value != null) return `Juegos proyectados: ${n(ex.expected_value)}`
  if (sp === 'TENIS' && k === 'handicap_games' && ex.expected_value != null) return `Margen proyectado: ${ex.expected_value >= 0 ? game.home : game.away} +${n(Math.abs(ex.expected_value))} juegos`
  if (sp === 'TENIS' && k === 'winner' && ex.p_serve_1 != null) return `Saque proyectado: ${n(ex.p_serve_1 * 100, 0)}% vs ${n(ex.p_serve_2 * 100, 0)}%`
  if (sp === 'SOCCER' && k === 'double_chance' && ex.lineas)
    return `Local ${n(ex.lineas.local * 100, 0)}% · Empate ${n(ex.lineas.empate * 100, 0)}% · Visitante ${n(ex.lineas.visitante * 100, 0)}%`
  if (sp === 'SOCCER' && k === 'corners' && ex.expected_value != null) return `Córners proyectados: ${n(ex.expected_value)}`
  if (sp === 'SOCCER' && k === 'total_goals' && ex.expected_value != null) return `Goles proyectados: ${n(ex.expected_value, 2)} (${n(ex.lambda_home, 2)} – ${n(ex.lambda_away, 2)})`
  if (sp === 'SOCCER' && k === 'btts' && ex.lambda_home != null) return `Goles esperados: ${n(ex.lambda_home, 2)} – ${n(ex.lambda_away, 2)}`
  return null
}

// Fila de un mercado sin predicción: hueco explicado, nunca un hueco mudo.
function FilaVacia({ k, titulo, tag, texto, razon }) {
  return (
    <div className="pickbox mute">
      <div className="mkt-label">{titulo}{tag}</div>
      <div className="sub">{texto}</div>
      {razon && <div className="sub" style={{ opacity: .8, marginTop: 3 }}>{razon}</div>}
    </div>
  )
}

function Fila({ k, game, sp, onOpen, markets }) {
  const c = game.markets[k]
  const titulo = TITLE[k] || k

  if (!c) {
    const st = estadoMercado(markets, sp, game.tour, k)
    const bloqueado = st && st.mode === 'blocked'
    return <FilaVacia k={k} titulo={titulo}
      tag={bloqueado ? <span className="tag warn">Bloqueado</span> : null}
      texto={bloqueado
        ? 'No se publica: en las pruebas el modelo no superó al baseline.'
        : 'Todavía sin predicción para este partido.'}
      razon={bloqueado ? st.reason : null} />
  }

  const ex = c.extra || {}
  // Sin datos no es una proyeccion del 0 %: es que no hay fuente. Mostrarlo
  // como 'PROYECCION 0.0%' hacia parecer que el modelo predice cero.
  if (ex.mode === 'insufficient_data') {
    return <FilaVacia k={k} titulo={titulo} tag={<span className="tag dis">Sin datos</span>}
      texto="No hay fuente de datos para este mercado en esta competición."
      razon={c.status_note} />
  }

  const p = c.ensemble_probability
  const conf = confianza(p)
  const esPick = ex.publish_pick && ['NBA', 'TENIS', 'SOCCER'].includes(sp)
  const val = c.valor || {}
  const proj = proyeccion(sp, k, ex, game)
  const resultado = real(c, ex, k, sp)
  const lineSrc = sp === 'NFL'
    ? (ex.line_source && ex.line_source !== 'the_odds_api' ? ex.line_source.split(' ')[0] : null)
    : (ex.total && ex.total.linea_origen && ex.total.linea_origen !== 'mercado' ? 'referencia' : null)

  // «Procedencia» solo se ofrece si al ABRIRLA hay algo que no estuviera ya
  // arriba. Sin cuota registrada, la probabilidad publicada ES la del modelo:
  // el desplegable repetia el mismo numero y parecia que no habia datos. En ese
  // caso se dice de una vez, en una linea, y no se ofrece nada que abrir.
  const sinCuota = c.market_probability == null
  const modeloDistinto = c.model_probability != null && p != null
    && Math.abs(c.model_probability - p) > 0.0005
  // «Procedencia» se ofrece siempre: aunque no haya cuota, dentro hay algo que
  // decir (que no la hay, la versión del modelo, el origen de la línea). Lo que
  // NO se hace es anunciarlo en la cara de la tarjeta con una etiqueta ámbar:
  // sin cuotas en casi todos los partidos, ese aviso salía en todas las filas a
  // la vez y un aviso que aparece siempre deja de avisar de nada.

  return (
    <div className="pickbox" onClick={e => { e.stopPropagation(); onOpen(c) }}
      style={{ cursor: 'pointer' }}>
      <div className="mkt-label">
        {titulo}
        {/* El chip «Sin pick» salía en casi todas las filas, así que dejaba de
            informar. El motivo exacto vive en «Procedencia». */}
        {esPick && <span className="tag acc">Pick</span>}
        {val.contra_mercado && <span className="tag dis" title={
          `El modelo da ${(p * 100).toFixed(1)}% y la casa ${(c.market_probability * 100).toFixed(1)}%`
        }>Contra el mercado {val.gap_pp >= 0 ? '+' : ''}{val.gap_pp} pp</span>}
        {val.ev_pct != null && val.ev_pct > 0 &&
          <span className="tag risk-low">Valor est. +{val.ev_pct}%</span>}
      </div>

      <div className="pickrow">
        <div style={{ minWidth: 0 }}>
          <div className="sel">{c.selection || 'Sin pick'} <Res c={c} /></div>
          {proj && <div className="sub">{proj}</div>}
          {resultado && <div className="sub" style={{
            color: c.result === 'win' ? 'var(--ok)' : c.result === 'loss' ? 'var(--bad)' : 'var(--dim)',
          }}>{resultado}</div>}
          {!c.result && c.status === 'final' &&
            <div className="sub" style={{ color: 'var(--warn)' }}>
              sin resultado disponible para este mercado</div>}
        </div>
        <div className="odds">
          <div className="pct" style={{ color: ACCENT[sp] }}>{pct(p)}</div>
          {/* Solo se etiqueta la banda ALTA. «BAJA» junto a un 53 % no añade
              nada: repite en palabras lo que el número ya dice. */}
          {conf === 'ALTA' && <span className={'tag ' + CONF_CLS[conf]}>Alta</span>}
        </div>
      </div>

      {p != null && <div className="meter" style={{ color: ACCENT[sp] }}>
        <i style={{ width: `${Math.max(0, Math.min(100, p * 100))}%` }} /></div>}

      <details className="tech" onClick={e => e.stopPropagation()}>
          <summary>Procedencia</summary>
          <div className="tech-body">
            <div className="tech-row">
              {modeloDistinto && <span>Modelo <b>{pct(c.model_probability)}</b></span>}
              <span>Mercado <b>{sinCuota ? 'sin cuota' : pct(c.market_probability)}</b></span>
              {!sinCuota && c.model_market_gap != null &&
                <span>Diferencia <b>{c.model_market_gap >= 0 ? '+' : ''}
                  {(c.model_market_gap * 100).toFixed(1)} pp</b></span>}
            </div>
            {val.cuota_decimal != null && <div className="tech-row">
              <span>Cuota <b>{val.cuota_decimal}</b></span>
              {val.ev_pct != null && <span>Valor estimado <b style={{
                color: val.ev_pct > 0 ? 'var(--ok)' : 'var(--bad)',
              }}>{val.ev_pct >= 0 ? '+' : ''}{val.ev_pct}%</b></span>}
            </div>}
            {lineSrc && <div className="tech-row"><span>Línea <b>{lineSrc}</b></span></div>}
            {c.model_version && <div className="tech-row">
              <span>Versión <b>{c.model_version}</b></span></div>}
            {ex.publish_pick === false && <div className="tech-row"
              style={{ color: 'var(--warn)' }}>
              Se publica la probabilidad, no una recomendación.
              {ex.motivo_estado ? ` ${ex.motivo_estado}` : ''}</div>}
            {val.aviso && <div className="tech-row" style={{ color: 'var(--dim-2)' }}>
              {val.aviso}</div>}
            {!sinCuota && <div className="tech-row" style={{ color: 'var(--dim-2)' }}>
              Una diferencia a favor del modelo no es «valor»: no hay histórico de
              cuotas suficiente para demostrarlo.
            </div>}
        </div>
      </details>
    </div>
  )
}

export default function GroupedGameCard({ game, onOpen, live, markets }) {
  const sp = game.sport
  const first = Object.values(game.markets)[0]
  const lv = live && live[game.game_id]
  // Solo se marca lo ACERTADO. Un fallo no se anuncia en la cabecera: ya sale
  // su ✗ dentro del mercado que fallo, y ahi si aporta. Repetirlo arriba
  // convertia la tarjeta entera en un cartel de derrota.
  const calificados = Object.values(game.markets).filter(c => c.result === 'win' || c.result === 'loss')
  const aciertos = calificados.filter(c => c.result === 'win').length
  const liga = sp === 'TENIS' ? (game.tour || 'Tenis')
    : sp === 'SOCCER' ? ((first || {}).extra || {}).league_name || 'Fútbol'
    : sp

  return (
    <div className={'card' + (aciertos > 0 ? ' hit' : '')}
      onClick={() => first && onOpen(first)}>
      <div className="bar" style={{ background: `linear-gradient(90deg, ${ACCENT[sp]}, transparent)` }} />
      <div className="body">
        <div className="card-head">
          <span className={'pill ' + sp.toLowerCase()}>{EMOJI[sp]} {liga}</span>
          <span className="when">
            {calificados.length > 0 &&
              <span className={aciertos > 0 ? 'res ok' : 'res pend'}>
                {aciertos}/{calificados.length}</span>}
            {localTime(game.start_utc)}
          </span>
        </div>

        <div className="teams">{sp === 'TENIS'
          ? <>{game.home} <small>vs</small> {game.away}</>
          : <>{game.away} <small>en</small> {game.home}</>}</div>
        {game.venue && <div className="mkt">{game.venue}</div>}

        <Marcador c={game} lv={lv} />
        {ORDER[sp].map(k =>
          <Fila key={k} k={k} game={game} sp={sp} onOpen={onOpen} markets={markets} />)}
      </div>
    </div>
  )
}
