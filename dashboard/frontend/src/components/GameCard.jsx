import React from 'react'
import { pct, localDateTime, riskLabel } from '../format'
import { confianza } from './GroupedGameCard'

// TARJETA DE UN SOLO PICK (Top picks). Misma regla que la tarjeta agrupada:
// arriba la selección y su probabilidad; la procedencia y las etiquetas
// técnicas, dentro del desplegable.

const ACCENT = { NFL: 'var(--nfl)', MLB: 'var(--mlb)', NBA: 'var(--nba)',
                 TENIS: 'var(--tenis)', SOCCER: 'var(--soccer)' }
const EMOJI = { NFL: '🏈', MLB: '⚾', NBA: '🏀', TENIS: '🎾', SOCCER: '⚽' }

function Marcador({ c, lv }) {
  // NFL: sin fuente en vivo; solo se muestra el final si ya está en la base
  if (c.sport === 'NFL') return null
  if (!lv) return null
  const st = lv.estado
  if (st === 'programado') return null
  const cls = st === 'en_vivo' ? 'live' : st === 'finalizado' ? 'final' : 'post'
  const label = st === 'en_vivo' ? 'En vivo' : st === 'finalizado' ? 'Final' : 'Pospuesto'
  return (
    <div className="score">
      <div>
        <div className="teams">{c.away} {lv.away ?? '–'} — {lv.home ?? '–'} {c.home}</div>
        {st === 'en_vivo' && <div className="inn">
          {lv.entrada || ''}{lv.outs != null ? ` · ${lv.outs} out${lv.outs === 1 ? '' : 's'}` : ''}</div>}
        {st === 'finalizado' && lv.entradas_jugadas > 9 &&
          <div className="inn">{lv.entradas_jugadas} entradas</div>}
        {st === 'pospuesto' && <div className="inn">{lv.detalle}</div>}
      </div>
      <span className={'st ' + cls}>{label}</span>
    </div>
  )
}

export default function GameCard({ c, onOpen, live }) {
  const sp = c.sport
  const lv = live && live[c.game_id]
  const dc = c.data_completeness || {}
  const ex = c.extra || {}
  const edge = c.edge_vs_base ?? ex.edge_vs_base ?? null
  const incomplete = dc.complete === false
  const res = c.result            // win | loss | push | null
  const cls = res === 'win' ? ' hit' : res === 'loss' ? ' miss' : ''
  const p = c.ensemble_probability
  const conf = confianza(p)
  const linea = c.line !== null && c.line !== undefined && ex.market_key !== 'total'
    ? ` ${c.line}` : ''

  // «Datos incompletos» no dice nada: hay que decir QUÉ falta. En MLB lo que
  // suele faltar es el abridor, y eso explica por qué los tres partidos de una
  // misma serie salen con la MISMA probabilidad: sin abridor anunciado el
  // modelo solo conoce la fuerza de los equipos, que no cambia entre días.
  const falta = (dc.missing || [])
  const sinAbridor = falta.some(m => /abridor/i.test(m))
  const val = c.valor || {}

  // Estas advertencias ya NO van en la cara de la tarjeta. Con casi todos los
  // partidos sin cuota y sin abridor, la fila de etiquetas ámbar salía en las
  // doce tarjetas a la vez: cuando un aviso aparece siempre, deja de avisar de
  // nada y solo ensucia. Siguen todas dentro de «Procedencia», que es donde se
  // va a mirar cuando el número importe. No se borra ninguna.

  return (
    <div className={'card' + cls} onClick={() => onOpen(c)}>
      <div className="bar" style={{ background: `linear-gradient(90deg, ${ACCENT[sp]}, transparent)` }} />
      <div className="body">
        <div className="card-head">
          <span className={'pill ' + sp.toLowerCase()}>{EMOJI[sp]} {sp}</span>
          <span className="when">
            {res === 'win' && <span className="res ok">✓</span>}
            {res === 'loss' && <span className="res no">✗</span>}
            {res === 'push' && <span className="res pend">=</span>}
            {localDateTime(c.start_utc)}
          </span>
        </div>

        <div className="teams">{c.away} <small>en</small> {c.home}</div>
        {c.venue && <div className="mkt">{c.venue}</div>}
        <Marcador c={c} lv={lv} />

        <div className="pickbox">
          <div className="mkt-label">
            {(c.market || '').toUpperCase()}{linea}
            {val.contra_mercado && <span className="tag dis">
              Contra el mercado {val.gap_pp >= 0 ? '+' : ''}{val.gap_pp} pp</span>}
            {val.ev_pct != null && val.ev_pct > 0 &&
              <span className="tag risk-low">Valor est. +{val.ev_pct}%</span>}
          </div>
          <div className="pickrow">
            <div style={{ minWidth: 0 }}>
              <div className="sel">{c.selection || 'Sin pick'}</div>
              {ex.total && ex.total.proyeccion != null && <div className="sub">
                Carreras esperadas: {Number(ex.total.proyeccion).toFixed(1)} · línea {ex.total.linea}
              </div>}
            </div>
            <div className="odds">
              <div className="pct" style={{ color: ACCENT[sp] }}>{pct(p)}</div>
              {conf && <span className={'tag ' + (conf === 'ALTA' ? 'risk-low'
                : conf === 'MEDIA' ? 'risk-med' : 'risk-high')}>{conf}</span>}
            </div>
          </div>
          {p != null && <div className="meter" style={{ color: ACCENT[sp] }}>
            <i style={{ width: `${Math.max(0, Math.min(100, p * 100))}%` }} /></div>}
        </div>

        <details className="tech" onClick={e => e.stopPropagation()}>
          <summary>Procedencia</summary>
          <div className="tech-body">
            <div className="tech-row">
              <span>Modelo <b>{pct(c.model_probability)}</b></span>
              <span>Mercado <b>{c.market_probability === null ? 'n/d' : pct(c.market_probability)}</b></span>
              {ex.base_rate_selection != null && <span>Tasa base <b>{pct(ex.base_rate_selection, 0)}</b></span>}
            </div>
            {edge != null && <div className="tech-row">
              <span>Ventaja sobre la tasa base <b style={{
                color: edge > 0 ? 'var(--ok)' : 'var(--bad)',
              }}>{(edge * 100 >= 0 ? '+' : '') + (edge * 100).toFixed(1)} pp</b></span>
            </div>}
            {val.cuota_decimal != null && <div className="tech-row">
              <span>Cuota <b>{val.cuota_decimal}</b></span>
              <span>Valor estimado <b style={{
                color: val.ev_pct > 0 ? 'var(--ok)' : 'var(--bad)',
              }}>{val.ev_pct >= 0 ? '+' : ''}{val.ev_pct}%</b></span>
            </div>}
            {val.aviso && <div className="tech-row" style={{ color: 'var(--dim-2)' }}>
              {val.aviso}</div>}
            {ex.total && ex.total.linea_origen && <div className="tech-row">
              <span>Origen de la línea <b>{ex.total.linea_origen}</b></span></div>}
            {/* Todo lo que antes gritaba en la cara de la tarjeta vive aquí. */}
            {falta.length > 0 && <div className="tech-row" style={{ color: 'var(--warn)' }}>
              Falta: {falta.join(' · ')}
              {sinAbridor && <> — sin abridor el modelo solo ve la fuerza de los
                equipos, así que el número no cambia entre días de una misma serie.</>}
            </div>}
            {incomplete && falta.length === 0 && <div className="tech-row"
              style={{ color: 'var(--warn)' }}>Los datos de este partido están incompletos.</div>}
            {c.directional_disagreement && <div className="tech-row"
              style={{ color: 'var(--warn)' }}>El modelo y el mercado apuntan a lados
              distintos.</div>}
            {ex.publish_pick === false && <div className="tech-row"
              style={{ color: 'var(--warn)' }}>Se publica la probabilidad, no una
              recomendación: este mercado no pasó el filtro de evidencia.</div>}
            <div className="tech-row">
              <span>Riesgo de fallar <b>{c.upset_risk != null
                ? Math.round(c.upset_risk) + '%' : 'n/d'}</b></span>
              <span>{riskLabel(c.upset_label, c.upset_risk)}</span>
            </div>
          </div>
        </details>
      </div>
    </div>
  )
}
