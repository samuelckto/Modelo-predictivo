import React from 'react'
import { pct, localTime } from '../format'

// Una tarjeta por partido con sus mercados (NFL: moneyline/total/spread;
// MLB: moneyline/over-under/run line). Cada bloque abre el detalle de ese mercado.
const ORDER = { NFL: ['moneyline', 'total', 'spread'], MLB: ['moneyline', 'total', 'run_line'],
                NBA: ['moneyline', 'spread', 'total'], TENIS: ['winner', 'total_games', 'handicap_games'] }
const ICON = { moneyline: '🏆', total: '🔥', spread: '📊', run_line: '📊', winner: '🏆', total_games: '🔥', handicap_games: '📊' }
const TITLE = { moneyline: 'MONEYLINE', total: 'TOTAL', spread: 'SPREAD', run_line: 'RUN LINE',
                winner: 'GANADOR', total_games: 'TOTAL DE JUEGOS', handicap_games: 'HÁNDICAP DE JUEGOS' }
const ACCENT = { NFL: 'var(--nfl)', MLB: 'var(--mlb)', NBA: 'var(--nba)', TENIS: 'var(--tenis)' }
const CONF_CLS = { ALTA: 'risk-low', MEDIA: 'risk-med', BAJA: 'risk-high' }

// Confianza = la misma banda para todos los mercados y deportes, derivada de la
// probabilidad calibrada del lado mostrado (terciles reales de MLB/NFL).
export const confianza = p => p == null ? null : p >= 0.62 ? 'ALTA' : p >= 0.545 ? 'MEDIA' : 'BAJA'

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
  const label = st === 'en_vivo' ? 'EN VIVO' : st === 'finalizado' ? 'FINALIZADO' : 'POSPUESTO'
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

export default function GroupedGameCard({ game, onOpen, live, markets }) {
  const sp = game.sport
  const first = Object.values(game.markets)[0]
  const lv = live && live[game.game_id]
  const ml = game.markets.moneyline
  const res = ml && ml.result
  const cls = res === 'win' ? ' hit' : res === 'loss' ? ' miss' : ''
  const Row = ({ k }) => {
    const c = game.markets[k]
    if (!c) {
      const st = estadoMercado(markets, sp, game.tour, k)
      const bloqueado = st && st.mode === 'blocked'
      return (
        <div className="pickbox" style={{ opacity: .6 }}>
          <div style={{ font: '600 10px Inter', color: 'var(--dim)', letterSpacing: 1 }}>
            {ICON[k]} {TITLE[k]}
            {bloqueado && <span className="tag warn" style={{ marginLeft: 6 }}>BLOQUEADO</span>}
          </div>
          <div style={{ fontSize: 12, color: 'var(--dim)', lineHeight: 1.5 }}>
            {bloqueado
              ? 'No se publica: en las pruebas el modelo no superó al baseline en este mercado.'
              : 'Todavía sin predicción para este partido.'}
          </div>
          {bloqueado && st.reason &&
            <div style={{ fontSize: 11, color: 'var(--dim)', marginTop: 4, opacity: .85 }}>{st.reason}</div>}
        </div>)
    }
    const ex = c.extra || {}
    const conf = confianza(c.ensemble_probability)
    const noPick = ex.publish_pick === false
    const tag = ex.mode === 'no_pick' ? 'NO PICK' : ex.mode === 'blocked' ? 'BLOQUEADO' : (k === 'run_line' ? 'PROBABILIDAD' : 'PROYECCIÓN')
    let proj = null
    if (sp === 'NFL' && k === 'total') proj = `Proyección: ${Number(ex.expected_value).toFixed(1)} puntos`
    if (sp === 'NFL' && k === 'spread') proj = `Margen proyectado: ${game.home} ${ex.expected_value >= 0 ? '+' : ''}${Number(ex.expected_value).toFixed(1)}`
    if (sp === 'MLB' && k === 'total' && ex.total) proj = `Carreras esperadas: ${Number(ex.total.proyeccion).toFixed(1)}`
    if (sp === 'NBA' && k === 'total' && ex.expected_value != null) proj = `Total proyectado: ${Number(ex.expected_value).toFixed(1)}`
    if (sp === 'NBA' && k === 'spread' && ex.expected_value != null) proj = `Margen proyectado: ${ex.expected_value >= 0 ? game.home : game.away} +${Math.abs(Number(ex.expected_value)).toFixed(1)}`
    if (sp === 'TENIS' && k === 'total_games' && ex.expected_value != null) proj = `Juegos proyectados: ${Number(ex.expected_value).toFixed(1)}`
    if (sp === 'TENIS' && k === 'handicap_games' && ex.expected_value != null) proj = `Margen proyectado: ${ex.expected_value >= 0 ? game.home : game.away} +${Math.abs(Number(ex.expected_value)).toFixed(1)} juegos`
    if (sp === 'TENIS' && k === 'winner' && ex.p_serve_1 != null) proj = `Saque proyectado: ${(ex.p_serve_1 * 100).toFixed(0)}% vs ${(ex.p_serve_2 * 100).toFixed(0)}%`
    const lineSrc = sp === 'NFL' ? (ex.line_source && ex.line_source !== 'the_odds_api' ? ` · línea ${ex.line_source.split(' ')[0]}` : '')
      : (ex.total && ex.total.linea_origen && ex.total.linea_origen !== 'mercado' ? ' · línea de referencia' : '')
    return (
      <div className="pickbox" onClick={e => { e.stopPropagation(); onOpen(c) }} style={{ cursor: 'pointer' }}>
        <div className="row">
          <div>
            <div style={{ font: '600 10px Inter', color: 'var(--dim)', letterSpacing: 1 }}>
              {ICON[k]} {TITLE[k]}
              {noPick && <span className="tag warn" style={{ marginLeft: 6 }}>{tag}</span>}
              {ex.publish_pick && (sp === 'NBA' || sp === 'TENIS') && <span className="tag risk-low" style={{ marginLeft: 6 }}>PICK</span>}
            </div>
            <div className="sel">{c.selection || 'SIN PICK'} <Res c={c} /></div>
            {(proj || lineSrc) && <div style={{ fontSize: 12, color: 'var(--dim)' }}>{proj}{lineSrc}</div>}
          </div>
          <div style={{ textAlign: 'right' }}>
            <div className="pct" style={{ color: ACCENT[sp] }}>{pct(c.ensemble_probability)}</div>
            {conf && <span className={'tag ' + CONF_CLS[conf]} style={{ fontSize: 10 }}>{conf}</span>}
          </div>
        </div>
        {(c.market_probability != null || c.model_probability != null) && <div className="srcline">
          <span>Modelo <b>{pct(c.model_probability)}</b></span>
          <span>Mercado <b>{c.market_probability == null ? 'n/d' : pct(c.market_probability)}</b></span>
          {c.market_probability != null && c.model_market_gap != null &&
            <span>Gap <b>{c.model_market_gap >= 0 ? '+' : ''}{(c.model_market_gap * 100).toFixed(1)} pp</b></span>}
        </div>}
      </div>
    )
  }
  return (
    <div className={'card' + cls} onClick={() => first && onOpen(first)}>
      <div className="bar" style={{ background: `linear-gradient(90deg, ${ACCENT[sp]}, transparent)` }} />
      <div className="body">
        <div className="row">
          <span className={'pill ' + sp.toLowerCase()}>
            {sp === 'NFL' ? '🏈' : sp === 'NBA' ? '🏀' : sp === 'TENIS' ? '🎾' : '⚾'} {sp === 'TENIS' ? (game.tour || 'TENIS') : sp}</span>
          <span className="mono" style={{ fontSize: 12, color: 'var(--dim)' }}>{localTime(game.start_utc)}</span>
        </div>
        <div className="teams">{sp === 'TENIS'
          ? <>{game.home} <small>vs</small> {game.away}</>
          : <>{game.away} <small>en</small> {game.home}</>}</div>
        <div className="mkt">{game.venue || '—'}</div>
        <Marcador c={game} lv={lv} />
        {ORDER[sp].map(k => <Row key={k} k={k} />)}
      </div>
    </div>
  )
}
