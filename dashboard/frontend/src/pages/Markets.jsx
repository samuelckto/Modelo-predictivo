import React, { useEffect, useState } from 'react'
import { pct, num } from '../format'

export default function Markets() {
  const [d, setD] = useState(null)
  useEffect(() => { fetch('/api/markets').then(r => r.json()).then(setD) }, [])
  if (!d) return <div className="empty">Cargando…</div>
  const mlb = d.MLB || {}
  const nfl = d.NFL || {}
  return (
    <>
      <div className="panel">
        <h2>Qué mercados publica el sistema y por qué</h2>

        <table>
          <thead><tr><th>Mercado</th><th>Estado</th><th>n</th><th>Tasa base</th>
            <th>Mejor estrategia</th><th>Acierto</th><th>Ventaja</th><th>z</th><th>ECE</th></tr></thead>
          <tbody>{Object.entries(mlb).filter(([, v]) => typeof v === 'object').map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{v.enabled
                ? (v.publish_pick ? <span style={{ color: 'var(--ok)' }}>PICK</span>
                  : <span style={{ color: 'var(--warn)' }}>solo probabilidad</span>)
                : <span style={{ color: 'var(--bad)' }}>BLOQUEADO</span>}</td>
              {v.rmse != null
                ? <td className="num" colSpan={7} style={{ color: 'var(--dim)' }}>
                    modelo de regresión — RMSE <b className="mono">{num(v.rmse, 3)}</b> frente a
                    {' '}<b className="mono">{num(v.baseline_rmse, 3)}</b> de predecir la media
                    {' '}(mejora de solo <b className="mono">{num(v.rmse_gain, 3)}</b> carreras)
                  </td>
                : <>
                  <td className="num">{v.n ?? '—'}</td>
                  <td className="num">{v.base_rate == null ? '—' : pct(v.base_rate, 2)}</td>
                  <td>{v.best_strategy ?? '—'}</td>
                  <td className="num">{v.best_accuracy == null ? '—' : pct(v.best_accuracy, 2)}</td>
                  <td className="num">{v.edge_vs_majority_pp == null ? '—' : v.edge_vs_majority_pp + ' pp'}</td>
                  <td className="num">{v.z ?? '—'}</td>
                  <td className="num">{v.ece == null ? '—' : num(v.ece, 4)}</td>
                </>}
            </tr>))}
          </tbody></table>
      </div>
      <div className="panel">
        <h2>NFL</h2>
        <table><thead><tr><th>Mercado</th><th>Estado</th><th>Motivo</th></tr></thead>
          <tbody>{Object.entries(nfl).map(([k, v]) => (
            <tr key={k}><td>{k}</td>
              <td style={{ color: v.enabled ? 'var(--ok)' : 'var(--bad)' }}>
                {v.enabled ? 'PICK' : 'BLOQUEADO'}</td>
              <td style={{ color: 'var(--dim)' }}>{v.reason}</td></tr>))}
          </tbody></table>

      </div>
    </>
  )
}
