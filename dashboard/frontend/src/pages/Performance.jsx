import React, { useEffect, useState } from 'react'
import { getPerformance } from '../api'
import { pct, num } from '../format'

function Block({ sport, p }) {
  if (!p) return null
  if (p.available === false)
    return <div className="panel"><h2>{sport}</h2>
      <div className="note">Sin desempeño evaluado todavía: {p.reason}.
        No se muestran métricas porque no existen datos reales que las respalden.</div></div>
  const bt = p.latest_backtest, st = bt?.summary?.strategies
  return (
    <div className="panel">
      <h2>{sport}</h2>
      <dl className="kv">
        <dt>Predicciones evaluadas</dt><dd>{p.graded_predictions ?? 0}</dd>
        <dt>Accuracy</dt><dd>{pct(p.accuracy)}</dd>
        {p.fantasy_points_pct !== undefined &&
          <><dt>Puntos fantasy</dt><dd>{p.fantasy_points ?? '—'} / {p.fantasy_points_max ?? '—'} ({pct(p.fantasy_points_pct)})</dd></>}
        <dt>Modelo en producción</dt>
        <dd>{p.production_model ? `${p.production_model.name} ${p.production_model.version}` : '—'}</dd>
      </dl>
      {st && <>
        <h3 style={{ fontSize: 14, marginTop: 18 }}>Backtest walk-forward — {bt.label}</h3>
        <table><thead><tr><th>Estrategia</th><th>Acc</th><th>Log loss</th><th>Brier</th><th>Puntos</th></tr></thead>
          <tbody>{Object.entries(st).map(([k, v]) =>
            <tr key={k}><td>{k}</td><td className="num">{pct(v.accuracy, 2)}</td>
              <td className="num">{num(v.log_loss, 4)}</td><td className="num">{num(v.brier, 4)}</td>
              <td className="num">{v.points_pct != null ? pct(v.points_pct, 2) : '—'}</td></tr>)}</tbody></table>
        <div className="note">Estas cifras salen del backtest realmente ejecutado y guardado en la base
          del motor {sport}. No son estimaciones.</div>
      </>}
    </div>
  )
}

export default function Performance({ sport }) {
  const [d, setD] = useState(null)
  useEffect(() => { setD(null); getPerformance(sport).then(setD) }, [sport])
  if (!d) return <div className="empty">Cargando…</div>
  return <>{Object.entries(d).map(([k, v]) => <Block key={k} sport={k} p={v} />)}</>
}
