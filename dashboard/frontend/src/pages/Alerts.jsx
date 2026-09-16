import React, { useEffect, useState } from 'react'

const ICON = {
  COIN_FLIP: '⚖️', MODEL_DISAGREEMENT: '🔀', ELO_GAP: '📐',
  MODEL_OVER_MARKET: '📈', MARKET_OVER_MODEL: '📉', MODEL_VS_MARKET: '🔁',
  PITCHER_UNKNOWN: '⚾', LINEUP_UNCONFIRMED: '🧢', DATA_INCOMPLETE: '⚠️',
  HIGH_RISK: '🚨', STRONG_TOTAL: '🔥', STRONG_SPREAD: '📊', STRONG_MONEYLINE: '🏆', NO_PICK: '🟡', HIGH_UNCERTAINTY: '🚨', LINE_MOVEMENT: '📈', STALE_DATA: '⏳',
}

const EXPLICA = {
  PITCHER_UNKNOWN: 'La MLB aún no anuncia al abridor. Es lo que más cambia una predicción de béisbol.',
  LINEUP_UNCONFIRMED: 'La alineación no está publicada. La predicción es provisional.',
  HIGH_RISK: 'Riesgo por encima de lo normal.',
  COIN_FLIP: 'Partido muy parejo: la probabilidad está cerca del 50 %.',
  MODEL_DISAGREEMENT: 'Los algoritmos del ensemble no coinciden entre sí.',
  MODEL_VS_MARKET: 'El modelo y el mercado favorecen lados distintos.',
  MODEL_OVER_MARKET: 'El modelo da más probabilidad que el mercado.',
  MARKET_OVER_MODEL: 'El mercado da más probabilidad que el modelo.',
  ELO_GAP: 'El rating Elo discrepa del modelo. Informativo.',
  DATA_INCOMPLETE: 'Falta algún dato de entrada.',
  STRONG_TOTAL: 'Total con pick validado y probabilidad alta.',
  STRONG_SPREAD: 'Spread con pick validado y probabilidad alta.',
  NO_PICK: 'Modelo y línea casi coinciden: no hay pick.',
  HIGH_UNCERTAINTY: 'Datos incompletos y sin lado claro.',
  STRONG_MONEYLINE: 'Moneyline con pick validado y probabilidad alta con evidencia histórica en ese rango.',
  LINE_MOVEMENT: 'La línea se movió de forma notable desde la apertura.',
  STALE_DATA: 'Los datos del deporte no están al día.',
}

export default function Alerts({ sport, range, start, end }) {
  const [d, setD] = useState(null)
  const [sel, setSel] = useState('all')
  useEffect(() => {
    setD(null); setSel('all')
    const q = new URLSearchParams({ sport, range })
    if (start) q.set('start', start); if (end) q.set('end', end)
    fetch('/api/alerts?' + q).then(r => r.json()).then(setD)
  }, [sport, range, start, end])
  if (!d) return <div className="empty">Cargando…</div>
  const groups = {}
  for (const a of d.alerts) (groups[a.code] = groups[a.code] || []).push(a)
  const visibles = sel === 'all' ? Object.keys(groups) : [sel]
  return (
    <>
      <div className="panel">
        <h2>Alertas · {d.from} → {d.to}</h2>

        {d.count === 0
          ? <div className="note ok">Ninguna alerta en este rango y con este filtro.
            Eso significa que no falta ningún dato y que ninguna predicción tiene
            señales de incertidumbre inusual.</div>
          : <>
            <div className="seg" style={{ flexWrap: 'wrap' }}>
              <button className={sel === 'all' ? 'on' : ''} onClick={() => setSel('all')}>
                Todas ({d.count})</button>
              {(d.resumen || []).map(r =>
                <button key={r.code} className={sel === r.code ? 'on' : ''}
                  onClick={() => setSel(r.code)}>
                  {ICON[r.code] || '•'} {r.code} ({r.n})</button>)}
            </div>
          </>}
      </div>
      {visibles.filter(c => groups[c]).map(code => {
        const list = groups[code]
        return (
          <div className="panel" key={code}>
            <h2>{ICON[code] || '•'} {code}
              <span style={{ color: 'var(--dim)', fontSize: 13 }}> ({list.length})</span></h2>
            {EXPLICA[code] && <p style={{ fontSize: 12, color: 'var(--dim)', marginTop: 0 }}>
              {EXPLICA[code]}</p>}
            <table><thead><tr><th>Deporte</th><th>Fecha</th><th>Partido</th>
              <th>Mercados afectados</th><th>Detalle</th></tr></thead>
              <tbody>{list.slice(0, 60).map((a, i) =>
                <tr key={i}>
                  <td><span className={'pill ' + a.sport.toLowerCase()}>{a.sport}</span></td>
                  <td className="mono">{a.date}</td><td>{a.game}</td>
                  <td style={{ fontSize: 12 }}>{a.market}</td>
                  <td style={{ color: 'var(--dim)' }}>{a.text}</td></tr>)}
              </tbody></table>
            {list.length > 60 && <div style={{ fontSize: 12, color: 'var(--dim)', marginTop: 8 }}>
              …y {list.length - 60} más</div>}
          </div>)
      })}
    </>
  )
}
