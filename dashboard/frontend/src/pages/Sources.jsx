import React, { useEffect, useState } from 'react'
import { getSources, getStatus } from '../api'

export default function Sources() {
  const [d, setD] = useState(null), [st, setSt] = useState(null)
  const [cal, setCal] = useState(null)
  useEffect(() => {
    getSources().then(setD); getStatus().then(setSt)
    fetch('/api/scoring-status').then(r => r.json()).then(setCal).catch(() => { })
  }, [])
  if (!d || !st) return <div className="empty">Cargando…</div>
  return (
    <>
      {/* Lo primero de la pantalla: qué quedó sin calificar. Antes esto no se
          veía en ningún sitio y un deporte podía llevar días sin cerrarse sin
          que nada lo dijera. */}
      {cal && (
        <div className="panel">
          <h2>Calificación de lo ya jugado</h2>
          <table>
            <thead><tr><th>Deporte</th><th className="num">Ya jugadas</th>
              <th className="num">Calificadas</th><th className="num">Pendientes</th>
              <th>Estado</th></tr></thead>
            <tbody>{cal.por_deporte.map(p => (
              <tr key={p.deporte}>
                <td><span className={'pill ' + p.deporte.toLowerCase()}>{p.deporte}</span></td>
                <td className="num">{p.ya_jugadas ?? '—'}</td>
                <td className="num">{p.calificadas ?? '—'}</td>
                <td className="num">{p.pendientes ?? '—'}</td>
                <td>{p.error ? <span className="tag dis">{p.error}</span>
                  : p.al_dia ? <span className="tag risk-low">al día</span>
                    : <span className="tag warn">quedan sin cerrar</span>}</td>
              </tr>))}
            </tbody>
          </table>
          <div className="note" style={{ marginTop: 10 }}>{cal.nota}</div>
          {(cal.fuentes || []).map((f, i) => (
            <div key={i} className={'note ' + (f.ok ? 'ok' : 'bad')}>
              <b>{f.fuente}</b>
              {f.peticiones_restantes != null && <> · {f.peticiones_restantes} peticiones
                restantes de {Number(f.peticiones_restantes) + Number(f.usadas || 0)}</>}
              {' — '}{f.detalle}
            </div>))}
          {cal.auto_score?.ultimo && <div className="sub" style={{ marginTop: 8 }}>
            Última pasada automática: {cal.auto_score.ultimo}
            {cal.auto_score.error ? ` · error: ${cal.auto_score.error}` : ''}</div>}
        </div>)}

      <div className="panel">
        <h2>Estado de los motores</h2>
        {Object.entries(st.sports).map(([sp, s]) => (
          <div key={sp} style={{ marginBottom: 14 }}>
            <span className={'pill ' + sp.toLowerCase()}>{sp}</span>
            <dl className="kv" style={{ marginTop: 8 }}>
              <dt>Listo</dt><dd>{s.ready ? 'sí' : 'no'}</dd>
              {s.engine && <><dt>Motor</dt><dd>{s.engine}</dd></>}
              {s.home && <><dt>Ubicación</dt><dd>{s.home}</dd></>}
              {s.read_only_proof && <><dt>Aislamiento</dt><dd>{s.read_only_proof}</dd></>}
              {s.phase && <><dt>Fase</dt><dd>{s.phase}</dd></>}
              {s.games !== undefined && <><dt>Partidos en base</dt><dd>{s.games}</dd></>}
              {s.predictions !== undefined && <><dt>Predicciones</dt><dd>{s.predictions}</dd></>}
            </dl>
            {(s.missing || []).map((m, i) => <div className="note" key={i}>Falta: {m}</div>)}
          </div>))}
      </div>

      <div className="panel">
        <h2>Prioridad de fuentes declarada</h2>
        <table><thead><tr><th>Deporte</th><th>Dominio</th><th>Orden de prioridad</th></tr></thead>
          <tbody>{Object.entries(d.priority).flatMap(([sp, doms]) =>
            Object.entries(doms).map(([dom, list]) =>
              <tr key={sp + dom}><td><span className={'pill ' + sp.toLowerCase()}>{sp}</span></td>
                <td>{dom}</td><td className="mono" style={{ fontSize: 12 }}>{list.join(' → ')}</td></tr>))}
          </tbody></table>
        <div className="note">Si dos fuentes discrepan no se decide en silencio: se guarda el conflicto
          en <code>source_conflicts</code> y se aplica esta prioridad.</div>
      </div>

      <div className="panel">
        <h2>Últimas descargas</h2>
        {['NFL', 'MLB'].map(sp => (
          <div key={sp}>
            <h3 style={{ fontSize: 14 }}>{sp}</h3>
            {(d[sp] || []).length === 0 ? <div className="empty" style={{ padding: 12 }}>Sin registros.</div> :
              <table><thead><tr><th>Fuente</th><th>Última descarga</th><th>Estado</th></tr></thead>
                <tbody>{d[sp].map((s, i) => <tr key={i}><td>{s.source}</td>
                  <td className="mono">{s.last_fetch || '—'}</td><td>{s.status ?? s.calls ?? '—'}</td></tr>)}</tbody></table>}
          </div>))}
      </div>
    </>
  )
}
