import React, { useEffect, useState } from 'react'
import GameDetail from './GameDetail'
import { pct, num, localTime, riskLabel } from '../format'

function LiveLine({ c }) {
  const [lv, setLv] = useState(null)
  useEffect(() => {
    if (!c || c.sport !== 'MLB') return
    let vivo = true
    const tick = () => fetch('/api/live').then(r => r.json())
      .then(d => vivo && setLv((d.games || {})[c.game_id] || null)).catch(() => {})
    tick(); const id = setInterval(tick, 60000)
    return () => { vivo = false; clearInterval(id) }
  }, [c && c.game_id])
  if (!lv || lv.estado === 'programado') return null
  const live = lv.estado === 'en_vivo'
  return (
    <div className="score" style={{ margin: '0 0 12px' }}>
      <div>
        <div className="teams">{c.away} {lv.away ?? '–'} — {lv.home ?? '–'} {c.home}</div>
        {live && <div className="inn">{lv.entrada || ''}{lv.outs != null ? ` · ${lv.outs} outs` : ''}</div>}
      </div>
      <span className={'st ' + (live ? 'live' : lv.estado === 'finalizado' ? 'final' : 'post')}>
        {live ? 'EN VIVO' : lv.estado === 'finalizado' ? 'FINALIZADO' : 'POSPUESTO'}</span>
    </div>
  )
}

function AbridoresLinea({ c }) {
  const [d, setD] = useState(null)
  useEffect(() => {
    if (!c || c.sport !== 'MLB') return
    fetch(`/api/mlb/game/${c.game_id}`).then(r => r.json()).then(setD).catch(() => {})
  }, [c && c.game_id])
  if (!d || !d.abridores) return null
  const stat = (a, key) => (a.stats || []).find(s => s.value != null && String(s.label || '').toLowerCase().startsWith(key))
  const Uno = ({ a, abbr, color }) => {
    const era = stat(a, 'era'), kbb = stat(a, 'k-bb')
    return (
      <div style={{ flex: 1, minWidth: 200 }}>
        <div style={{ font: '700 10px Inter', color: 'var(--dim)', letterSpacing: 1 }}>ABRE POR {abbr}</div>
        <div style={{ fontSize: 16, fontWeight: 700, color }}>{a.name || 'Sin anunciar todavía'}</div>
        <div style={{ fontSize: 12, color: 'var(--dim)' }}>
          {a.name ? (a.throws ? (a.throws === 'R' ? 'derecho' : 'zurdo') : '') : 'la MLB aún no lo anuncia'}
          {era ? ` · ERA ${era.value}` : ''}{kbb ? ` · K-BB ${kbb.value}` : ''}
        </div>
      </div>
    )
  }
  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', padding: '10px 12px', background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 10, marginBottom: 14 }}>
      <Uno a={d.abridores.away || {}} abbr={c.away} color="#4aa3ff" />
      <Uno a={d.abridores.home || {}} abbr={c.home} color="#39d353" />
    </div>
  )
}

export default function GameModal({ c, onClose }) {
  const [tab, setTab] = useState(c && c._detalle ? 'ficha' : 'resumen')
  if (!c) return null
  const dc = c.data_completeness || {}
  const ue = c.upset_explanation || null
  const ex = c.extra || {}
  const edge = c.edge_vs_base ?? ex.edge_vs_base ?? null
  return (
    <div className="modalbg" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <button className="close" onClick={onClose}>cerrar</button>
        <span className={'pill ' + (c.sport || '').toLowerCase()}>{c.sport}</span>
        <h2>{c.away} en {c.home}</h2>
        <div style={{ color: 'var(--dim)', fontSize: 13, marginBottom: 10 }}>
          {c.game_date} · {localTime(c.start_utc)} · {c.venue || 'sede no registrada'}
          {c.result === 'win' && <span className="res ok" style={{ marginLeft: 10 }}>✓ ACERTÓ</span>}
          {c.result === 'loss' && <span className="res no" style={{ marginLeft: 10 }}>✗ FALLÓ</span>}
        </div>
        <LiveLine c={c} />
        {c.sport === 'MLB' && <div className="seg" style={{ marginBottom: 14 }}>
          <button className={tab === 'resumen' ? 'on' : ''} onClick={() => setTab('resumen')}>Resumen</button>
          <button className={tab === 'ficha' ? 'on' : ''} onClick={() => setTab('ficha')}>Ficha completa</button>
        </div>}
        {c.sport === 'MLB' && tab === 'ficha' && <GameDetail gameId={c.game_id} />}
        {(c.sport !== 'MLB' || tab === 'resumen') && <>
        <AbridoresLinea c={c} />
        {c.sport === 'NBA' && <div className="note">
          {ex.market_key === 'moneyline'
            ? <>Estado: <b>{ex.mode === 'pick' ? 'PICK' : ex.mode === 'no_pick' ? 'NO PICK' : 'PROYECCIÓN'}</b>{ex.threshold ? ` · umbral validado ${(ex.threshold * 100).toFixed(0)} %` : ''}</>
            : ex.market_key === 'spread'
              ? <>Margen proyectado: <b>{ex.expected_value >= 0 ? c.home : c.away} +{Math.abs(Number(ex.expected_value)).toFixed(1)}</b> (desviación {Number(ex.expected_std).toFixed(1)}) · línea {c.home} {ex.line >= 0 ? '+' : ''}{ex.line} ({ex.line_source}) · <b>PROYECCIÓN</b></>
              : <>Total proyectado: <b>{Number(ex.expected_value).toFixed(1)}</b> (desviación {Number(ex.expected_std).toFixed(1)}) · línea {ex.line} ({ex.line_source}) · <b>PROYECCIÓN</b></>}
          {ex.odds_home != null && <div style={{ fontSize: 12, color: 'var(--dim)', marginTop: 4 }}>Cuotas: {ex.odds_home > 0 ? '+' : ''}{ex.odds_home} / {ex.odds_away > 0 ? '+' : ''}{ex.odds_away}</div>}
        </div>}
        {c.sport === 'NFL' && ex.expected_value != null && <div className="note">
          {ex.market_key === 'total'
            ? <>Proyección: <b>{Number(ex.expected_value).toFixed(1)} puntos</b> (mediana {Number(ex.expected_median).toFixed(1)}, desviación {Number(ex.expected_std).toFixed(1)}) · línea {ex.line} ({ex.line_source})</>
            : <>Margen proyectado: <b>{c.home} {ex.expected_value >= 0 ? '+' : ''}{Number(ex.expected_value).toFixed(1)}</b> (desviación {Number(ex.expected_std).toFixed(1)}) · línea {c.home} {ex.line != null ? (-ex.line > 0 ? '+' : '') + (-ex.line) : '—'} ({ex.line_source})</>}
          {ex.p_raw != null && <div style={{ fontSize: 12, color: 'var(--dim)', marginTop: 4 }}>
            Probabilidad sin calibrar del lado mostrado: {(Math.max(ex.p_raw, 1 - ex.p_raw) * 100).toFixed(0)}% · calibrada: {pct(c.ensemble_probability)}</div>}
          {ex.mode === 'proyeccion' && <div style={{ fontSize: 12, marginTop: 4, color: 'var(--warn)' }}>PROYECCIÓN, no pick.</div>}
        </div>}
        {ex.total && ex.total.proyeccion != null && <div className="note">
          Carreras esperadas: <b>{Number(ex.total.proyeccion).toFixed(1)}</b> · línea {ex.total.linea} ({ex.total.linea_origen})</div>}

        <h3 style={{ fontSize: 15 }}>¿Qué dice cada fuente?</h3>
        <dl className="kv">
          <dt>Modelo (ensemble propio)</dt><dd>{pct(c.model_probability)}</dd>
          <dt>Casas de apuestas</dt><dd>{c.market_probability === null ? 'no disponible' : pct(c.market_probability)}</dd>
          {c.elo_probability !== null && <><dt>Rating de equipos (Elo)</dt><dd>{pct(c.elo_probability)}</dd></>}
          <dt>Probabilidad final</dt><dd>{pct(c.ensemble_probability)}</dd>
          {c.model_market_gap !== null && <><dt>Modelo vs casas</dt><dd>{(c.model_market_gap * 100).toFixed(1) + ' pp · ' + (c.directional_disagreement ? 'no coinciden en el favorito' : 'coinciden')}</dd></>}
          <dt>Tasa base del mercado</dt>
          <dd>{ex.base_rate_selection == null ? '—' : pct(ex.base_rate_selection)}</dd>
          <dt>Ventaja sobre la tasa base</dt>
          <dd>{edge == null ? '—' : ((edge * 100 >= 0 ? '+' : '') + (edge * 100).toFixed(1) + ' pp')}</dd>
          <dt>Riesgo</dt>
          <dd>{c.upset_risk == null ? '—' : `${Math.round(c.upset_risk)}% de que el pick falle · ${riskLabel(c.upset_label, c.upset_risk)}`}</dd>
        </dl>
        {c.market_signal && <div className="note">{c.market_signal}</div>}
        {ex.market_caveat && <div className="note">{ex.market_caveat}</div>}



        {ue && <>
          <h3 style={{ fontSize: 15, marginTop: 20 }}>¿Por qué este pick y qué puede fallar?</h3>
          {ue.summary && <p style={{ fontSize: 13, lineHeight: 1.6 }}>{ue.summary}</p>}
          {Array.isArray(ue.drivers) && <ul style={{ fontSize: 13, lineHeight: 1.6 }}>
            {ue.drivers.map((d, i) => <li key={i}>{typeof d === 'string' ? d : d.text}</li>)}
          </ul>}
          {ue.verdict && <div className="note">{ue.verdict}</div>}
          {Array.isArray(ue.flags) && ue.flags.length > 0 && <>
            <h3 style={{ fontSize: 15, marginTop: 16 }}>Avisos</h3>
            <ul style={{ fontSize: 13, lineHeight: 1.6 }}>
              {ue.flags.map((f, i) => <li key={i}>{f.text}</li>)}
            </ul></>}
        </>}

        {dc.complete === false && <>
          <h3 style={{ fontSize: 15, marginTop: 20 }}>⚠️ Datos incompletos</h3>
          <ul style={{ fontSize: 13 }}>{(dc.missing || []).map((m, i) => <li key={i}>{m}</li>)}</ul>
        </>}

        <h3 style={{ fontSize: 15, marginTop: 20 }}>Trazabilidad</h3>
        <dl className="kv">
          <dt>Modelo</dt><dd>{c.model_version || '—'}</dd>
          <dt>Predicción generada</dt><dd>{c.prediction_timestamp || '—'}</dd>
          <dt>Versión de la predicción</dt><dd>{c.version ?? '—'}</dd>
          <dt>ID</dt><dd>{c.prediction_id ?? '—'}</dd>
          <dt>Fuentes</dt><dd>{(c.sources || []).length ? (c.sources || []).map(s => typeof s === 'string' ? s : s.source).join(', ') : '—'}</dd>
        </dl>
        </>}
      </div>
    </div>
  )
}
