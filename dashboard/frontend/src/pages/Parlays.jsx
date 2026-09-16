import React, { useEffect, useState } from 'react'
import { pct, localTime } from '../format'

const SPC = { NFL: 'nfl', MLB: 'mlb', NBA: 'nba', TENIS: 'tenis', SOCCER: 'soccer' }

function Combinada({ p }) {
  const res = p.resultado
  const cls = res === 'win' ? ' hit' : res === 'loss' ? ' miss' : ''
  return (
    <div className={'card' + cls}>
      <div className="bar" style={{ background: 'linear-gradient(90deg, var(--warn), transparent)' }} />
      <div className="body">
        <div className="row">
          <span className="pill" style={{ color: 'var(--warn)', borderColor: '#ffb02055', background: '#ffb02012' }}>
            🎟️ {p.size} PATAS{p.mixed ? ' · MIXTA' : ''}</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {res === 'win' && <span className="res ok">✓ ACERTÓ</span>}
            {res === 'loss' && <span className="res no">✗ FALLÓ</span>}
            {res === 'push' && <span className="res pend">= ANULADA</span>}
            <span className="mono" style={{ fontSize: 12, color: 'var(--dim)' }}>{p.fecha}</span>
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, margin: '6px 0 2px' }}>
          <span className="mono" style={{ fontSize: 30, color: 'var(--warn)' }}>{pct(p.probabilidad, 1)}</span>
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>de que se cumpla</span>
        </div>
        <div style={{ fontSize: 12, color: 'var(--dim)' }}>
          Cuota justa: <b className="mono">{p.cuota_justa ? p.cuota_justa.toFixed(2) : '—'}</b>
          {p.cuota_mercado && <> · cuota del mercado (sin vig): <b className="mono">{p.cuota_mercado.toFixed(2)}</b></>}
          {p.estado === 'open' && p.patas_ganadas > 0 && <> · {p.patas_ganadas}/{p.patas} patas ganadas</>}
        </div>
        <div style={{ marginTop: 10, display: 'grid', gap: 8 }}>
          {p.legs.map((l, i) => (
            <div key={i} className="pickbox" style={{ padding: '8px 10px' }}>
              <div className="row">
                <div>
                  <span className={'pill ' + (SPC[l.sport] || '')} style={{ fontSize: 10 }}>{l.sport}</span>{' '}
                  <span style={{ fontSize: 12, color: 'var(--dim)' }}>{l.teams} · {l.market}</span>
                  <div className="sel" style={{ fontSize: 15 }}>
                    {l.selection}{' '}
                    {l.resultado === 'win' && <span className="res ok">✓</span>}
                    {l.resultado === 'loss' && <span className="res no">✗</span>}
                    {l.resultado === 'push' && <span className="res pend">=</span>}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div className="mono" style={{ fontSize: 16 }}>{pct(l.probabilidad, 1)}</div>
                  <div style={{ fontSize: 11, color: 'var(--dim)' }}>{localTime(l.inicio)}</div>
                </div>
              </div>
            </div>))}
        </div>
      </div>
    </div>
  )
}

export default function Parlays() {
  const [d, setD] = useState(null)
  const [tam, setTam] = useState('all')
  const [cargando, setCargando] = useState(false)
  const cargar = (regenerar) => {
    setCargando(true)
    fetch('/api/parlays' + (regenerar ? '?regenerar=true' : ''))
      .then(r => r.json()).then(x => { setD(x); setCargando(false) })
      .catch(() => setCargando(false))
  }
  useEffect(() => { cargar(false) }, [])
  if (!d) return <div className="empty">Cargando…</div>
  const filtro = l => tam === 'all' || String(l.size) === tam
  const abiertas = (d.abiertas || []).filter(filtro)
  const cerradas = (d.calificadas || []).filter(filtro)
  const t = d.total
  return (
    <>
      <div className="panel">
        <h2>🎟️ Combinadas</h2>
        <div className="controls" style={{ margin: '8px 0' }}>
          <div className="seg">
            {[['all', 'Todas'], ['2', '2 patas'], ['3', '3 patas']].map(([k, l]) =>
              <button key={k} className={tam === k ? 'on' : ''} onClick={() => setTam(k)}>{l}</button>)}
          </div>
          <button className="btn" onClick={() => cargar(true)} disabled={cargando}>
            {cargando ? 'generando…' : 'regenerar con los partidos de hoy'}</button>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', margin: '10px 0' }}>
          {[['Abiertas', abiertas.length], ['Calificadas', t.n],
            ['Aciertos', t.aciertos], ['Fallos', t.fallos],
            ['Acierto real', t.accuracy == null ? '—' : pct(t.accuracy, 1)],
            ['Prob. media anunciada', t.prob_media_anunciada == null ? '—' : pct(t.prob_media_anunciada, 1)]].map(([l, v]) => (
              <div key={l} style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 10, padding: '10px 14px', minWidth: 120 }}>
                <div style={{ font: '600 10px Inter', color: 'var(--dim)', letterSpacing: 1, textTransform: 'uppercase' }}>{l}</div>
                <div className="mono" style={{ fontSize: 20 }}>{v}</div>
              </div>))}
        </div>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: 'var(--dim)' }}>
          {[2, 3].map(s => d.por_tamano[s] && (
            <span key={s}>{s} patas: <b>{d.por_tamano[s].aciertos}</b>✓ / <b>{d.por_tamano[s].fallos}</b>✗
              {d.por_tamano[s].n ? ` (${pct(d.por_tamano[s].accuracy, 1)} real vs ${pct(d.por_tamano[s].prob_media_anunciada, 1)} anunciado)` : ''}</span>))}
        </div>
        <div className="note" style={{ marginTop: 10 }}>{d.nota}</div>
      </div>

      {abiertas.length > 0 && <>
        <h3 style={{ margin: '18px 0 8px' }}>Pendientes</h3>
        <div className="grid">{abiertas.map(p => <Combinada key={p.id} p={p} />)}</div>
      </>}
      {cerradas.length > 0 && <>
        <h3 style={{ margin: '18px 0 8px' }}>Ya resueltas</h3>
        <div className="grid">{cerradas.map(p => <Combinada key={p.id} p={p} />)}</div>
      </>}
      {abiertas.length === 0 && cerradas.length === 0 &&
        <div className="empty">No hay combinadas: hacen falta al menos dos picks validados de partidos
          y equipos distintos. Pulsa «regenerar» cuando haya partidos del día.</div>}
    </>
  )
}
