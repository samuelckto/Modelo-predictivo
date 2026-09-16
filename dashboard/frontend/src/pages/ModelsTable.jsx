import React, { useEffect, useState } from 'react'
import { pct, num } from '../format'

export default function ModelsTable() {
  const [rows, setRows] = useState(null)
  useEffect(() => { fetch('/api/performance/models').then(r => r.json()).then(d => setRows(d.rows)) }, [])
  if (!rows) return <div className="empty">Cargando…</div>
  return (
    <div className="panel">
      <h2>Rendimiento por modelo</h2>
      <p style={{ fontSize: 12, color: 'var(--dim)', marginTop: 0 }}>
        MLB: walk-forward 2023-2026, variante limpia. NFL: backtest almacenado del motor existente.
      </p>
      <table>
        <thead><tr><th>Deporte</th><th>Mercado</th><th>Modelo</th><th>N</th>
          <th>Accuracy</th><th>Log loss</th><th>Brier</th><th>ECE</th><th>Fantasy</th></tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i}>
            <td><span className={'pill ' + r.sport.toLowerCase()}>{r.sport}</span></td>
            <td>{r.market_label || r.market}</td><td>{r.model}</td>
            <td className="num">{r.n ?? '—'}</td>
            <td className="num">{pct(r.accuracy, 2)}</td>
            <td className="num">{num(r.log_loss, 4)}</td>
            <td className="num">{num(r.brier, 4)}</td>
            <td className="num">{r.ece == null ? '—' : num(r.ece, 4)}</td>
            <td className="num">{r.fantasy_points_pct == null ? '—' : pct(r.fantasy_points_pct, 2)}</td>
          </tr>))}</tbody></table>
    </div>
  )
}
