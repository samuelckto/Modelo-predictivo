import React, { useEffect, useState } from 'react'
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis, ReferenceLine, Bar, BarChart
} from 'recharts'
import { pct, num } from '../format'

const MARKETS = [['all', 'Todos'], ['moneyline', 'Moneyline'],
                 ['total', 'Total / O-U'], ['run_line', 'Run line'], ['spread', 'Spread']]
const MODELS = [['all', 'Todos'], ['elo', 'Elo'], ['ensemble', 'Ensemble'], ['forest', 'Random forest'], ['ridge', 'Ridge (NFL total/spread)']]
const SEASONS = [['all', 'Todas'], ['2023', '2023'], ['2024', '2024'], ['2025', '2025'], ['2026', '2026']]
const AX = { fontSize: 11, fill: 'var(--dim)' }

function NflMarkets({ m }) {
  const mk = ['total', 'spread'].filter(k => m[k] && m[k].por_temporada)
  if (!mk.length) return null
  return (
    <div className="panel">
      <h2>NFL · Total y Spread (walk-forward 2020-2025, frente a la línea de cierre)</h2>
      {mk.map(k => {
        const g = m[k]
        const seasons = Object.keys(g.por_temporada)
        return (
          <div key={k} style={{ marginTop: 10 }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <b style={{ textTransform: 'uppercase' }}>{k}</b>
              <span className={'tag ' + (g.mode === 'pick' ? 'risk-low' : 'warn')}>{g.mode === 'pick' ? 'PICK' : 'PROYECCIÓN'}</span>
              <span style={{ fontSize: 12, color: 'var(--dim)' }}>{g.algorithm} · {g.n_features} features · distribución {g.distribution}</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
            <table style={{ marginTop: 6 }}>
              <thead><tr><th>Temporada</th><th>n</th><th>Acierto vs línea</th><th>Baselines</th><th>Log loss</th><th>Brier</th>
                <th>MAE modelo</th><th>MAE media</th><th>MAE cierre</th><th>Umbral (T-1)</th><th>Picks</th><th>Acierto picks</th></tr></thead>
              <tbody>{seasons.map(s => {
                const r = g.por_temporada[s], v = r.vs_linea, b = v.baselines || {}
                return <tr key={s}><td>{s}</td><td className="num">{v.n}</td>
                  <td className="num">{pct(v.acc_todos, 1)}</td>
                  <td style={{ fontSize: 11 }}>{Object.entries(b).map(([n, x]) => `${n.replace('siempre_', '')} ${(x * 100).toFixed(1)}%`).join(' · ')}</td>
                  <td className="num">{num(r.log_loss, 4)}</td><td className="num">{num(r.brier, 4)}</td>
                  <td className="num">{num(r.regresion.modelo.mae, 2)}</td><td className="num">{num(r.regresion.media.mae, 2)}</td>
                  <td className="num">{num(r.regresion.mercado_cierre.mae, 2)}</td>
                  <td className="num">{v.umbral == null ? 'sin pick' : v.umbral}</td>
                  <td className="num">{v.n_picks}</td><td className="num">{v.acc_picks == null ? '—' : pct(v.acc_picks, 1)}</td></tr>
              })}</tbody></table></div>
            <div style={{ fontSize: 12, marginTop: 6 }}>
              <span style={{ color: 'var(--dim)' }}>Calibración (prob. sin calibrar → acierto real): </span>
              {Object.entries(g.calibration || {}).map(([b, c]) => `${b}%: ${c.n ? (c.acierto_real * 100).toFixed(1) + '%' : '—'} (n=${c.n})`).join(' · ')}
            </div>
            <div className="note" style={{ marginTop: 6 }}>{g.reason}</div>
          </div>)
      })}
    </div>
  )
}

function TenisMarkets({ m }) {
  return (
    <div className="panel">
      <h2>Tenis · Performance (walk-forward: selección 2015-2019, holdout 2020-2026)</h2>
      {['ATP', 'WTA'].filter(t => m[t]).map(tour => {
        const w = m[tour].winner
        return (
          <div key={tour} style={{ marginTop: 12 }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <b>{tour} · GANADOR</b>
              <span className={'tag ' + (w.mode === 'pick' ? 'risk-low' : 'warn')}>{w.mode === 'pick' ? 'PICK' : w.mode.toUpperCase()}</span>
              <span style={{ fontSize: 12, color: 'var(--dim)' }}>{w.engine} · calibración {w.calibration_method} · umbral {w.threshold}</span>
            </div>
            <div style={{ overflowX: 'auto' }}><table style={{ marginTop: 6 }}>
              <thead><tr><th>Temporada</th><th>n</th><th>Acierto</th><th>Favorito Elo</th><th>Umbral (T-1)</th>
                <th>Picks</th><th>Acierto picks</th><th>Elo en esos picks</th></tr></thead>
              <tbody>{Object.entries(w.por_temporada || {}).map(([s, r]) => <tr key={s}>
                <td>{s}</td><td className="num">{r.n}</td><td className="num">{pct(r.acc, 1)}</td>
                <td className="num">{pct(r.acc_elo, 1)}</td><td className="num">{r.umbral == null ? 'sin pick' : r.umbral}</td>
                <td className="num">{r.n_picks}</td><td className="num">{r.acc_picks == null ? '—' : pct(r.acc_picks, 1)}</td>
                <td className="num">{r.acc_elo_en_picks == null ? '—' : pct(r.acc_elo_en_picks, 1)}</td></tr>)}</tbody></table></div>
            <div style={{ fontSize: 12, marginTop: 6 }}>
              <span style={{ color: 'var(--dim)' }}>Calibración holdout: </span>
              {Object.entries(w.buckets || {}).map(([b, c]) => `${b}%: ${c.n ? (c.acierto_real * 100).toFixed(1) + '%' : '—'} (n=${c.n})`).join(' · ')}
              {' · ECE '}{num((w.holdout || {}).ece, 4)}
            </div>
            <div className="note" style={{ marginTop: 6 }}>{w.reason}</div>
            {['total_games', 'handicap_games'].map(k => {
              const g = m[tour][k]
              return (
                <div key={k} style={{ marginTop: 8 }}>
                  <b style={{ fontSize: 13 }}>{k === 'total_games' ? 'TOTAL DE JUEGOS' : 'HÁNDICAP DE JUEGOS'}</b>{' '}
                  <span className={'tag ' + (g.mode === 'projection' ? 'warn' : 'risk-high')}>
                    {g.mode === 'projection' ? 'PROYECCIÓN' : 'BLOQUEADO'}</span>
                  <div style={{ fontSize: 12, color: 'var(--dim)', marginTop: 4 }}>{g.reason}</div>
                </div>)
            })}
          </div>)
      })}
    </div>
  )
}

function NbaMarkets({ m }) {
  const ml = m.moneyline
  const rows = ml && ml.por_temporada ? Object.entries(ml.por_temporada) : []
  return (
    <div className="panel">
      <h2>NBA Performance (walk-forward: selección 2019-2022, holdout 2023-2026)</h2>
      {ml && <div style={{ marginTop: 10 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <b>MONEYLINE</b>
          <span className={'tag ' + (ml.mode === 'pick' ? 'risk-low' : 'warn')}>{ml.mode === 'pick' ? 'PICK' : ml.mode.toUpperCase()}</span>
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>{ml.algorithm} · {ml.n_features} features · calibración {ml.calibration_method} · umbral {ml.threshold}</span>
        </div>
        <div style={{ overflowX: 'auto' }}><table style={{ marginTop: 6 }}>
          <thead><tr><th>Temporada</th><th>n</th><th>Acierto</th><th>Log loss</th><th>Brier</th><th>ECE</th><th>AUC</th>
            <th>Elo acierto</th><th>Elo log loss</th><th>Tasa local</th><th>Umbral (T-1)</th><th>Picks</th><th>Acierto picks</th><th>Elo en esos picks</th></tr></thead>
          <tbody>{rows.map(([s, r]) => <tr key={s}><td>{s}</td><td className="num">{r.n}</td>
            <td className="num">{pct(r.accuracy, 1)}</td><td className="num">{num(r.log_loss, 4)}</td><td className="num">{num(r.brier, 4)}</td>
            <td className="num">{num(r.ece, 3)}</td><td className="num">{num(r.auc, 3)}</td>
            <td className="num">{pct(r.elo_accuracy, 1)}</td><td className="num">{num(r.elo_log_loss, 4)}</td><td className="num">{pct(r.tasa_local, 1)}</td>
            <td className="num">{r.umbral == null ? 'sin pick' : r.umbral}</td><td className="num">{r.n_picks ?? '—'}</td>
            <td className="num">{r.acc_picks == null ? '—' : pct(r.acc_picks, 1)}</td><td className="num">{r.acc_fav_elo_en_picks == null ? '—' : pct(r.acc_fav_elo_en_picks, 1)}</td></tr>)}</tbody></table></div>
        <div style={{ fontSize: 12, marginTop: 6 }}><span style={{ color: 'var(--dim)' }}>Calibración holdout (prob. calibrada → acierto real): </span>
          {Object.entries(ml.buckets || {}).map(([b, c]) => `${b}%: ${c.n ? (c.acierto_real * 100).toFixed(1) + '%' : '—'} (n=${c.n})`).join(' · ')}</div>
        <div className="note" style={{ marginTop: 6 }}>{ml.reason}</div>
      </div>}
      {['spread', 'total'].map(k => m[k] && <div key={k} style={{ marginTop: 16 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <b style={{ textTransform: 'uppercase' }}>{k}</b>
          <span className={'tag ' + (m[k].mode === 'projection' ? 'warn' : 'risk-high')}>{m[k].mode === 'projection' ? 'PROYECCIÓN' : 'BLOQUEADO'}</span>
          <span style={{ fontSize: 12, color: 'var(--dim)' }}>{m[k].algorithm} · {m[k].n_features} features · distribución {m[k].distribution} · σ {num(m[k].res_std, 1)}</span>
        </div>
        <div style={{ overflowX: 'auto' }}><table style={{ marginTop: 6 }}>
          <thead><tr><th>Temporada</th><th>n</th><th>MAE modelo</th><th>RMSE modelo</th><th>MAE media</th><th>RMSE media</th></tr></thead>
          <tbody>{Object.entries(m[k].por_temporada || {}).map(([s, r]) => <tr key={s}><td>{s}</td><td className="num">{r.modelo.n}</td>
            <td className="num">{num(r.modelo.mae, 2)}</td><td className="num">{num(r.modelo.rmse, 2)}</td>
            <td className="num">{num(r.media.mae, 2)}</td><td className="num">{num(r.media.rmse, 2)}</td></tr>)}</tbody></table></div>
        <div style={{ fontSize: 12, marginTop: 6 }}><span style={{ color: 'var(--dim)' }}>Calibración de P(y &gt; k) en holdout: </span>
          {Object.entries(m[k].buckets || {}).map(([b, c]) => `${b}%: ${c.n ? (c.acierto_real * 100).toFixed(1) + '%' : '—'} (n=${c.n})`).join(' · ')}
          {' · ECE '}{num((m[k].holdout_p_over_k || {}).ece, 3)}</div>
        <div className="note" style={{ marginTop: 6 }}>{m[k].reason}</div>
      </div>)}
    </div>
  )
}

function Metric({ label, value, sub }) {
  return (
    <div style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 10, padding: '10px 14px', minWidth: 120 }}>
      <div style={{ font: '600 10px Inter', color: 'var(--dim)', letterSpacing: 1, textTransform: 'uppercase' }}>{label}</div>
      <div className="mono" style={{ fontSize: 20 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--dim)' }}>{sub}</div>}
    </div>
  )
}

export default function PerformanceCenter({ sport }) {
  const [market, setMarket] = useState('moneyline')
  const [model, setModel] = useState('elo')
  const [season, setSeason] = useState('all')
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [d, setD] = useState(null)
  const [nfl, setNfl] = useState(null)
  const [nba, setNba] = useState(null)
  const [ten, setTen] = useState(null)
  useEffect(() => {
    fetch('/api/performance?sport=NFL').then(r => r.json()).then(x => setNfl(x.NFL && x.NFL.markets)).catch(() => {})
    fetch('/api/nba/markets').then(r => r.json()).then(setNba).catch(() => {})
    fetch('/api/tenis/markets').then(r => r.json()).then(setTen).catch(() => {})
  }, [])

  useEffect(() => {
    setD(null)
    const q = new URLSearchParams({ sport, market, model, season })
    if (from) q.set('date_from', from); if (to) q.set('date_to', to)
    fetch('/api/performance/detail?' + q).then(r => r.json()).then(setD)
  }, [sport, market, model, season, from, to])

  if (!d) return <div className="empty">Cargando…</div>
  if (d.available === false)
    return <div className="panel"><div className="note">{d.reason}</div></div>
  const t = d.total
  const rel = (t.reliability || []).filter(b => b.n > 20)
    .map(b => ({ bucket: b.bucket, predicho: b.predicted * 100, real: b.observed * 100, n: b.n }))
  const mes = (d.por_mes || []).filter(m => m.n >= 30)
    .map(m => ({ mes: m.month, acc: m.accuracy * 100, brier: m.brier, ll: m.log_loss, n: m.n }))
  let acum = 0, i = 0
  const acumulado = (d.por_mes || []).filter(m => m.n >= 1).map(m => {
    acum += m.wins; i += m.n
    return { mes: m.month, acierto_acumulado: (acum / i) * 100, picks: i }
  })

  return (
    <>
      {nfl && (sport === 'all' || sport === 'NFL') && <NflMarkets m={nfl} />}
      {nba && !nba.error && (sport === 'all' || sport === 'NBA') && <NbaMarkets m={nba} />}
      {ten && !ten.error && (sport === 'all' || sport === 'TENIS') && <TenisMarkets m={ten} />}
      <div className="panel">
        <h2>Performance Center</h2>

        <div className="controls" style={{ margin: '10px 0' }}>
          <div className="seg">{MARKETS.map(([k, l]) =>
            <button key={k} className={market === k ? 'on' : ''} onClick={() => setMarket(k)}>{l}</button>)}</div>
          <div className="seg">{MODELS.map(([k, l]) =>
            <button key={k} className={model === k ? 'on' : ''} onClick={() => setModel(k)}>{l}</button>)}</div>
          <select value={season} onChange={e => setSeason(e.target.value)}>
            {SEASONS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
          <input type="date" value={from} onChange={e => setFrom(e.target.value)} />
          <input type="date" value={to} onChange={e => setTo(e.target.value)} />
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <Metric label="Picks" value={t.n ?? 0} />
          <Metric label="Aciertos" value={t.wins ?? 0} sub={`${t.losses ?? 0} fallos`} />
          <Metric label="Accuracy" value={pct(t.accuracy, 2)} sub={`base ${pct(t.base_media, 1)}`} />
          <Metric label="Brier" value={num(t.brier, 4)} />
          <Metric label="Log loss" value={num(t.log_loss, 4)} />
          <Metric label="ECE" value={num(t.ece, 4)} />
          <Metric label="Prob. media" value={pct(t.prob_media, 1)} />
        </div>

      </div>

      {rel.length > 0 && <div className="panel">
        <h2>Curva de calibración</h2>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={rel}>
            <CartesianGrid stroke="#22304a" strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={AX} /><YAxis tick={AX} domain={[35, 85]} />
            <Tooltip contentStyle={{ background: '#121924', border: '1px solid #22304a' }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line dataKey="predicho" stroke="#8fa3bf" strokeDasharray="4 4" dot={false} name="Predicho" />
            <Line dataKey="real" stroke="#4aa3ff" strokeWidth={2} name="Real" />
          </LineChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 12, color: 'var(--dim)' }}>
          Si las dos líneas coinciden, «60 %» significa de verdad 60 %.
        </div>
      </div>}

      {mes.length > 0 && <>
        <div className="panel">
          <h2>Accuracy por mes</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={mes}>
              <CartesianGrid stroke="#22304a" strokeDasharray="3 3" />
              <XAxis dataKey="mes" tick={AX} /><YAxis tick={AX} domain={[40, 70]} />
              <Tooltip contentStyle={{ background: '#121924', border: '1px solid #22304a' }} />
              <ReferenceLine y={(t.base_media || 0.5) * 100} stroke="#ffb020" strokeDasharray="4 4"
                label={{ value: 'tasa base', fill: '#ffb020', fontSize: 11 }} />
              <Line dataKey="acc" stroke="#39d353" strokeWidth={2} name="Accuracy %" dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <h2>Brier y log loss por mes</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={mes}>
              <CartesianGrid stroke="#22304a" strokeDasharray="3 3" />
              <XAxis dataKey="mes" tick={AX} /><YAxis tick={AX} domain={[0.2, 0.75]} />
              <Tooltip contentStyle={{ background: '#121924', border: '1px solid #22304a' }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line dataKey="brier" stroke="#4aa3ff" name="Brier" dot={false} />
              <Line dataKey="ll" stroke="#ffb020" name="Log loss" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <h2>Acierto acumulado</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={acumulado}>
              <CartesianGrid stroke="#22304a" strokeDasharray="3 3" />
              <XAxis dataKey="mes" tick={AX} /><YAxis tick={AX} domain={[45, 65]} />
              <Tooltip contentStyle={{ background: '#121924', border: '1px solid #22304a' }} />
              <Line dataKey="acierto_acumulado" stroke="#39d353" strokeWidth={2} dot={false}
                name="Acierto acumulado %" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </>}

      <div className="panel">
        <h2>Por nivel de confianza</h2>
        <table><thead><tr><th>Rango</th><th>N</th><th>Accuracy</th><th>Brier</th><th>Log loss</th></tr></thead>
          <tbody>{(d.por_confianza || []).map((b, i) =>
            <tr key={i}><td className="mono">{b.rango}</td><td className="num">{b.n}</td>
              <td className="num">{pct(b.accuracy, 2)}</td><td className="num">{num(b.brier, 4)}</td>
              <td className="num">{num(b.log_loss, 4)}</td></tr>)}</tbody></table>
      </div>

      <div className="panel">
        <h2>Por nivel de riesgo</h2>
        <table><thead><tr><th>Riesgo declarado</th><th>N</th><th>Accuracy</th><th>Fallo real</th></tr></thead>
          <tbody>{(d.por_riesgo || []).map((b, i) =>
            <tr key={i}><td className="mono">{b.rango}%</td><td className="num">{b.n}</td>
              <td className="num">{pct(b.accuracy, 2)}</td>
              <td className="num">{b.accuracy == null ? '—' : pct(1 - b.accuracy, 2)}</td></tr>)}</tbody></table>
      </div>
    </>
  )
}
