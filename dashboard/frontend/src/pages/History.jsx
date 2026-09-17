import React, { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from 'recharts'
import { pct, num, localTime } from '../format'

const AX = { fontSize: 11, fill: 'var(--dim)' }
const SPCOLOR = { NFL: '#39d353', MLB: '#4aa3ff', NBA: '#ff8a3d', TENIS: '#d4e04a', SOCCER: '#00d4a0' }
const MKCOLOR = { moneyline: '#39d353', total: '#ffb020', run_line: '#4aa3ff', spread: '#ff8a3d' }

function Graficas({ deportes, todas, parlays }) {
  const evaluadas = todas.filter(p => p.acierto === true || p.acierto === false)
  if (evaluadas.length < 3) return null
  // 1) acierto por deporte y mercado
  const barras = []
  for (const [sp, x] of Object.entries(deportes))
    for (const [k, m] of Object.entries(x.por_mercado || {}))
      if (m.n) barras.push({ nombre: `${sp} ${CHIP[k] || k}`, acierto: m.accuracy * 100, n: m.n, sport: sp, market: k,
                             lo: m.ic95 ? m.ic95[0] * 100 : null, hi: m.ic95 ? m.ic95[1] * 100 : null })
  // 2) acierto acumulado en el tiempo (ordenado por fecha)
  const ord = [...evaluadas].sort((a, b) => String(a.fecha).localeCompare(String(b.fecha)))
  let h = 0
  const acumulado = ord.map((p, i) => { h += p.acierto ? 1 : 0; return { i: i + 1, fecha: p.fecha, acierto: (h / (i + 1)) * 100 } })
  // 3) aciertos y fallos por día
  const porDia = {}
  for (const p of ord) { const d = porDia[p.fecha] ||= { fecha: p.fecha, aciertos: 0, fallos: 0 }; p.acierto ? d.aciertos++ : d.fallos++ }
  const dias = Object.values(porDia)
  // 4) calibración: probabilidad anunciada vs acierto real
  const bks = [[0.5, 0.55], [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 1.01]]
  const calib = bks.map(([lo, hi]) => {
    const s = evaluadas.filter(p => p.probabilidad >= lo && p.probabilidad < hi)
    return { bucket: `${Math.round(lo * 100)}-${hi > 1 ? 100 : Math.round(hi * 100)}%`, n: s.length,
             anunciada: s.length ? (s.reduce((a, p) => a + p.probabilidad, 0) / s.length) * 100 : null,
             real: s.length ? (s.filter(p => p.acierto).length / s.length) * 100 : null }
  }).filter(b => b.n > 0)
  // combinadas: acierto real vs probabilidad anunciada, por tamaño
  const par = []
  if (parlays && parlays.total.n > 0) {
    for (const [lab, m] of [['2 patas', parlays.por_tamano[2]], ['3 patas', parlays.por_tamano[3]],
                            ['Todas', parlays.total]]) {
      if (m && m.n) par.push({ nombre: lab, real: m.accuracy * 100, anunciada: (m.prob_media_anunciada || 0) * 100, n: m.n })
    }
  }
  const tip = { contentStyle: { background: '#0f1520', border: '1px solid #26324a', fontSize: 12 } }
  return (
    <div className="panel">
      <h2>Gráficas</h2>
      <div className="dual graficas-grid" style={{ gap: 18, marginTop: 10 }}>
        <div>
          <div style={{ font: '600 11px Inter', color: 'var(--dim)', letterSpacing: 1, marginBottom: 6 }}>ACIERTO POR MERCADO (%)</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barras} margin={{ left: -10, right: 10 }}>
              <CartesianGrid stroke="#1d2739" vertical={false} />
              <XAxis dataKey="nombre" tick={{ ...AX, fontSize: 10 }} interval={0} angle={-15} textAnchor="end" height={50} />
              <YAxis domain={[0, 100]} tick={AX} />
              <Tooltip {...tip} formatter={(v, n, o) => [`${v.toFixed(1)}% (n=${o.payload.n})`, 'acierto']} />
              <ReferenceLine y={50} stroke="#ff5c5c" strokeDasharray="4 4" label={{ value: '50 %', fill: '#ff5c5c', fontSize: 10 }} />
              <Bar dataKey="acierto" radius={[4, 4, 0, 0]}>
                {barras.map((b, i) => <Cell key={i} fill={SPCOLOR[b.sport] || '#888'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div style={{ font: '600 11px Inter', color: 'var(--dim)', letterSpacing: 1, marginBottom: 6 }}>ACIERTO ACUMULADO (%) · {evaluadas.length} predicciones</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={acumulado} margin={{ left: -10, right: 10 }}>
              <CartesianGrid stroke="#1d2739" vertical={false} />
              <XAxis dataKey="i" tick={AX} />
              <YAxis domain={[0, 100]} tick={AX} />
              <Tooltip {...tip} labelFormatter={i => `predicción #${i} · ${(acumulado[i - 1] || {}).fecha || ''}`} formatter={v => [`${v.toFixed(1)}%`, 'acumulado']} />
              <ReferenceLine y={50} stroke="#ff5c5c" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="acierto" stroke="#39d353" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div style={{ font: '600 11px Inter', color: 'var(--dim)', letterSpacing: 1, marginBottom: 6 }}>ACIERTOS Y FALLOS POR DÍA</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={dias} margin={{ left: -10, right: 10 }}>
              <CartesianGrid stroke="#1d2739" vertical={false} />
              <XAxis dataKey="fecha" tick={{ ...AX, fontSize: 10 }} tickFormatter={f => String(f).slice(5)} />
              <YAxis tick={AX} allowDecimals={false} />
              <Tooltip {...tip} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="aciertos" stackId="a" fill="#39d353" name="✓ aciertos" />
              <Bar dataKey="fallos" stackId="a" fill="#ff5c5c" name="✗ fallos" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div style={{ font: '600 11px Inter', color: 'var(--dim)', letterSpacing: 1, marginBottom: 6 }}>CALIBRACIÓN · PROBABILIDAD ANUNCIADA VS ACIERTO REAL</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={calib} margin={{ left: -10, right: 10 }}>
              <CartesianGrid stroke="#1d2739" vertical={false} />
              <XAxis dataKey="bucket" tick={AX} />
              <YAxis domain={[0, 100]} tick={AX} />
              <Tooltip {...tip} formatter={(v, n, o) => [`${v == null ? '—' : v.toFixed(1)}% (n=${o.payload.n})`, n]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="anunciada" fill="#4aa3ff" name="anunciada" radius={[4, 4, 0, 0]} />
              <Bar dataKey="real" fill="#39d353" name="real" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 11, color: 'var(--dim)' }}>Si las barras verdes se parecen a las azules, la probabilidad dice la verdad. Con pocas predicciones las barras saltan mucho.</div>
        </div>
        {par.length > 0 && <div>
          <div style={{ font: '600 11px Inter', color: 'var(--dim)', letterSpacing: 1, marginBottom: 6 }}>COMBINADAS · ACIERTO REAL VS ANUNCIADO</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={par} margin={{ left: -10, right: 10 }}>
              <CartesianGrid stroke="#1d2739" vertical={false} />
              <XAxis dataKey="nombre" tick={AX} /><YAxis domain={[0, 100]} tick={AX} />
              <Tooltip {...tip} formatter={(v, n, o) => [`${v.toFixed(1)}% (n=${o.payload.n})`, n]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="anunciada" fill="#4aa3ff" name="anunciada" radius={[4, 4, 0, 0]} />
              <Bar dataKey="real" fill="#ffb020" name="real" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>}
      </div>
    </div>
  )
}

const CHIP = { moneyline: 'Moneyline', total: 'Total / Over-Under', run_line: 'Run line', spread: 'Spread',
               winner: 'Ganador', total_games: 'Total juegos', handicap_games: 'Hándicap juegos',
               total_goals: 'Total goles', btts: 'Ambos marcan', double_chance: 'Doble oportunidad', corners: 'Córners' }

function Tarjeta({ m }) {
  const acc = m.accuracy
  if (!m.n) return (
    <div className="sp" style={{ minWidth: 240, opacity: .75 }}>
      <h4>{m.market_label}</h4>
      <div className="mono" style={{ fontSize: 30, color: 'var(--dim)', margin: '6px 0' }}>—</div>
      <div style={{ fontSize: 12, color: 'var(--dim)' }}>{m.aviso || 'todavía sin partidos evaluados'}</div>
    </div>)
  const color = acc == null ? 'var(--dim)' : acc >= 0.55 ? 'var(--ok)' : acc >= 0.5 ? 'var(--warn)' : 'var(--bad)'
  return (
    <div className="sp" style={{ minWidth: 240 }}>
      <h4>{m.market_label}</h4>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, margin: '6px 0' }}>
        <span className="mono" style={{ fontSize: 30, color }}>{pct(acc, 1)}</span>
        <span style={{ fontSize: 12, color: 'var(--dim)' }}>
          {m.aciertos} ✓ · {m.fallos} ✗ de {m.n}</span>
      </div>
      {m.ic95 && <div style={{ fontSize: 12, color: 'var(--dim)' }}>
        Intervalo 95 %: {pct(m.ic95[0], 1)} – {pct(m.ic95[1], 1)}</div>}
      <div className="statrow"><span style={{ color: 'var(--dim)' }}>Prob. media anunciada</span>
        <b>{pct(m.prob_media, 1)}</b></div>
      <div className="statrow"><span style={{ color: 'var(--dim)' }}>Brier</span><b>{num(m.brier, 4)}</b></div>
      <div className="statrow"><span style={{ color: 'var(--dim)' }}>Log loss</span><b>{num(m.log_loss, 4)}</b></div>
      {m.aviso && <div style={{ marginTop: 6, fontSize: 11, color: 'var(--dim)' }}>muestra pequeña</div>}
    </div>
  )
}

export default function History({ sport = 'all', onOpenGame }) {
  const [data, setData] = useState(null)
  const [par, setPar] = useState(null)
  useEffect(() => { fetch('/api/parlays').then(r => r.json()).then(setPar).catch(() => {}) }, [])
  const [filtro, setFiltro] = useState('all')
  useEffect(() => {
    setData(null)
    fetch('/api/history?sport=' + sport).then(r => r.json()).then(setData)
  }, [sport])
  if (!data) return <div className="empty">Cargando…</div>
  const deportes = data.deportes || {}
  const t = data.total
  const todas = Object.values(deportes).flatMap(x => (x.predicciones || []))
  const lista = todas.filter(p => filtro === 'all' || p.market === filtro)
  const mercados = {}
  for (const [sp, x] of Object.entries(deportes))
    for (const [k, m] of Object.entries(x.por_mercado || {}))
      mercados[`${sp} · ${k}`] = { ...m, sport: sp }

  return (
    <>
      <div className="panel">
        <h2>Histórico real</h2>

        {Object.entries(deportes).map(([sp, x]) => (
          <div key={sp} className="note" style={{ borderColor: sp === 'NFL' ? 'var(--nfl)' : sp === 'NBA' ? 'var(--nba)' : sp === 'TENIS' ? 'var(--tenis)' : sp === 'SOCCER' ? 'var(--soccer)' : 'var(--mlb)' }}>
            <span className={'pill ' + sp.toLowerCase()}>{sp}</span>{' '}
            {x.disponible === false
              ? <>no disponible — {x.motivo}</>
              : x.total.n === 0
                ? <>todavía no hay ninguna predicción evaluada.
                  {' '}{Object.values(x.por_mercado || {})[0]?.aviso || ''}</>
                : <><b>{x.total.aciertos}</b> aciertos y <b>{x.total.fallos}</b> fallos
                  de <b>{x.total.n}</b> predicciones
                  {/* El total cuenta solo los picks validados. Las proyecciones se
                      califican igual, pero aparte, para no inflar ni hundir la cifra. */}
                  {(() => {
                    const ev = Object.values(x.por_mercado || {}).reduce((a, m) => a + (m.n || 0), 0)
                    return ev > x.total.n
                      ? <div style={{ fontSize: 12, color: 'var(--dim)', marginTop: 4 }}>
                        {x.total.nota || 'solo los picks cuentan en este total'}. En total se evaluaron{' '}
                        <b>{ev}</b> predicciones contando las proyecciones: cada mercado tiene su
                        propio % abajo.</div>
                      : null
                  })()}</>}
          </div>))}
        {t.n === 0
          ? <div className="empty">Todavía no hay predicciones evaluadas en el filtro
            seleccionado. Aparecerán aquí en cuanto terminen los primeros partidos.</div>
          : <>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, margin: '10px 0' }}>
              <span className="mono" style={{ fontSize: 40 }}>{pct(t.accuracy, 1)}</span>
              <span style={{ color: 'var(--dim)' }}>
                {t.aciertos} aciertos · {t.fallos} fallos · {t.n} predicciones</span>
            </div>
            {t.ic95 && <div style={{ fontSize: 12, color: 'var(--dim)' }}>
              Intervalo 95 %: {pct(t.ic95[0], 1)} – {pct(t.ic95[1], 1)}</div>}
          </>}
      </div>

      <Graficas deportes={deportes} todas={todas} parlays={par} />
      {par && par.total.n > 0 && <div className="panel">
        <h2>Combinadas · efectividad</h2>
        <div className="dual" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', marginTop: 10 }}>
          {[['Todas', par.total], ...[2, 3].map(s => [`${s} patas`, par.por_tamano[s]])]
            .filter(([, m]) => m && m.n > 0).map(([lab, m]) => (
              <div className="sp" key={lab} style={{ minWidth: 240 }}>
                <h4>{lab}</h4>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, margin: '6px 0' }}>
                  <span className="mono" style={{ fontSize: 30, color: m.accuracy >= (m.prob_media_anunciada || 0) ? 'var(--ok)' : 'var(--warn)' }}>
                    {pct(m.accuracy, 1)}</span>
                  <span style={{ fontSize: 12, color: 'var(--dim)' }}>{m.aciertos} ✓ · {m.fallos} ✗ de {m.n}</span>
                </div>
                <div className="statrow"><span style={{ color: 'var(--dim)' }}>Prob. media anunciada</span>
                  <b>{pct(m.prob_media_anunciada, 1)}</b></div>
                {m.ic95 && <div style={{ fontSize: 12, color: 'var(--dim)' }}>
                  Intervalo 95 %: {pct(m.ic95[0], 1)} – {pct(m.ic95[1], 1)}</div>}
              </div>))}
        </div>
      </div>}
      {Object.entries(deportes).filter(([, x]) => x.disponible !== false).map(([sp, x]) => (
        <div key={sp} className="panel">
          <h2><span className={'pill ' + sp.toLowerCase()}>{sp}</span> Efectividad por mercado</h2>
          <div className="dual" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', marginTop: 10 }}>
            {Object.entries(x.por_mercado || {}).map(([k, m]) =>
              <Tarjeta key={k} m={{ ...m, market_label: CHIP[k] || m.market_label || k }} />)}
          </div>
        </div>))}

      {t.n > 0 && <div className="panel">
        <h2>Todas las predicciones evaluadas</h2>
        <div className="seg" style={{ marginBottom: 10 }}>
          <button className={filtro === 'all' ? 'on' : ''} onClick={() => setFiltro('all')}>Todos</button>
          {[...new Set(todas.map(p => p.market))].map(k =>
            <button key={k} className={filtro === k ? 'on' : ''} onClick={() => setFiltro(k)}>
              {CHIP[k] || k}</button>)}
        </div>
        <table>
          <thead><tr><th></th><th>Fecha</th><th>Partido</th><th>Mercado</th><th>Selección</th>
            <th>Prob.</th><th>Riesgo</th><th>Modelo</th></tr></thead>
          <tbody>{lista.map((p, i) => (
            <tr key={i} style={{ cursor: onOpenGame ? 'pointer' : undefined }}
              onClick={() => onOpenGame && typeof p.game_id === 'number'
                && onOpenGame(p.game_id)}>
              <td>{p.acierto ? <span className="res ok">✓</span> : <span className="res no">✗</span>}</td>
              <td className="mono">{p.fecha}</td>
              <td>{p.away} @ {p.home}</td>
              <td>{p.market_label}</td>
              <td>{p.selection}</td>
              <td className="num">{pct(p.probabilidad, 1)}</td>
              <td className="num">{p.riesgo == null ? '—' : Math.round(p.riesgo) + '%'}</td>
              <td style={{ color: 'var(--dim)', fontSize: 11 }}>{p.modelo_version}</td>
            </tr>))}</tbody></table>
      </div>}
    </>
  )
}
