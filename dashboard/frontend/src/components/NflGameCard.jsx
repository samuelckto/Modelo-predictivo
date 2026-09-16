import React from 'react'
import { pct, localTime } from '../format'

// Una tarjeta por partido NFL con sus tres mercados: moneyline (motor NFL
// existente), total y spread (mercados propios de SPC).
const CONF = { ALTA: 'risk-low', MEDIA: 'risk-med', BAJA: 'risk-high' }
const ICON = { moneyline: '🏆', total: '🔥', spread: '📊' }
const TITLE = { moneyline: 'MONEYLINE', total: 'TOTAL', spread: 'SPREAD' }

function confML(c) {
  const p = c.ensemble_probability
  if (p == null) return null
  return p >= 0.62 ? 'ALTA' : p >= 0.56 ? 'MEDIA' : 'BAJA'
}

function Res({ c }) {
  if (c.result === 'win') return <span className="res ok">✓</span>
  if (c.result === 'loss') return <span className="res no">✗</span>
  if (c.result === 'push') return <span className="res pend">=</span>
  return null
}

export default function NflGameCard({ game, onOpen }) {
  const { ml, total, spread, home, away, start_utc, venue } = game
  const any = ml || total || spread
  const res = ml && ml.result
  const cls = res === 'win' ? ' hit' : res === 'loss' ? ' miss' : ''
  const Row = ({ k, c }) => {
    if (!c) return (
      <div className="pickbox" style={{ opacity: .55 }}>
        <div style={{ font: '600 10px Inter', color: 'var(--dim)', letterSpacing: 1 }}>{ICON[k]} {TITLE[k]}</div>
        <div style={{ fontSize: 12, color: 'var(--dim)' }}>sin predicción</div>
      </div>)
    const ex = c.extra || {}
    const conf = k === 'moneyline' ? confML(c) : c.confidence
    const proj = k === 'total' ? `Proyección: ${Number(ex.expected_value).toFixed(1)} puntos`
      : k === 'spread' ? `Margen proyectado: ${home} ${ex.expected_value >= 0 ? '+' : ''}${Number(ex.expected_value).toFixed(1)}` : null
    const noPick = k !== 'moneyline' && ex.publish_pick === false
    return (
      <div className="pickbox" onClick={e => { e.stopPropagation(); onOpen(c) }} style={{ cursor: 'pointer' }}>
        <div className="row">
          <div>
            <div style={{ font: '600 10px Inter', color: 'var(--dim)', letterSpacing: 1 }}>
              {ICON[k]} {TITLE[k]} {noPick && <span className="tag warn" style={{ marginLeft: 6 }}>PROYECCIÓN</span>}
            </div>
            <div className="sel">{c.selection || 'SIN PICK'} <Res c={c} /></div>
            {proj && <div style={{ fontSize: 12, color: 'var(--dim)' }}>{proj}
              {ex.line_source && ex.line_source !== 'the_odds_api' && <span> · línea {ex.line_source.split(' ')[0]}</span>}</div>}
          </div>
          <div style={{ textAlign: 'right' }}>
            <div className="pct" style={{ color: 'var(--nfl)' }}>{pct(c.ensemble_probability)}</div>
            {conf && <span className={'tag ' + CONF[conf]} style={{ fontSize: 10 }}>{conf}</span>}
          </div>
        </div>
        {c.market_probability != null && <div className="srcline">
          <span>Modelo <b>{pct(c.model_probability)}</b></span>
          <span>Mercado <b>{pct(c.market_probability)}</b></span>
          {ex.gap_pp != null && <span>Gap <b>{ex.gap_pp >= 0 ? '+' : ''}{Number(ex.gap_pp).toFixed(1)} pp</b></span>}
        </div>}
      </div>
    )
  }
  return (
    <div className={'card' + cls} onClick={() => any && onOpen(ml || total || spread)}>
      <div className="bar" style={{ background: 'linear-gradient(90deg, var(--nfl), transparent)' }} />
      <div className="body">
        <div className="row">
          <span className="pill nfl">🏈 NFL</span>
          <span className="mono" style={{ fontSize: 12, color: 'var(--dim)' }}>{localTime(start_utc)}</span>
        </div>
        <div className="teams">{away} <small>en</small> {home}</div>
        <div className="mkt">{venue || '—'}</div>
        <Row k="moneyline" c={ml} />
        <Row k="total" c={total} />
        <Row k="spread" c={spread} />
      </div>
    </div>
  )
}
