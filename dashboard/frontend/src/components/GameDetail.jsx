import React, { useEffect, useState } from 'react'
import { pct, num, localTime } from '../format'

const Stat = ({ s }) => s.value == null ? null : (
  <div className="statrow"><span style={{ color: 'var(--dim)' }}>{s.label}</span><b>{s.value}</b></div>
)

function Starter({ side, a, color }) {
  if (!a) return null
  const grupos = [...new Set((a.stats || []).filter(s => s.value != null).map(s => s.group))]
  return (
    <div className="sp">
      <div style={{ font: '700 10px Inter', color: 'var(--dim)', letterSpacing: 1 }}>
        ABRIDOR {side}</div>
      <h4 style={{ color }}>{a.name || 'Sin anunciar'}</h4>
      <div className="meta">
        {a.throws ? `Lanza con la ${a.throws === 'R' ? 'derecha' : 'izquierda'} · ` : ''}
        {a.estado}
        {a.anunciado_at ? ` · anunciado ${String(a.anunciado_at).replace('T', ' ').slice(0, 16)}` : ''}
      </div>
      {!a.name && <div style={{ fontSize: 12, color: 'var(--warn)' }}>
        La MLB todavía no ha anunciado al abridor. La predicción es provisional.</div>}
      {grupos.length === 0 && a.name &&
        <div style={{ fontSize: 12, color: 'var(--warn)' }}>
          Sin estadísticas suficientes de este lanzador todavía (pocas aperturas en la
          temporada o datos aún no descargados).</div>}
      {grupos.map(g => (
        <div key={g}>
          <div className="grp">{g}</div>
          {(a.stats || []).filter(s => s.group === g).map((s, i) => <Stat key={i} s={s} />)}
        </div>))}
    </div>
  )
}

export default function GameDetail({ gameId }) {
  const [d, setD] = useState(null), [err, setErr] = useState(null)
  useEffect(() => {
    setD(null); setErr(null)
    fetch(`/api/mlb/game/${gameId}`).then(r => r.json())
      .then(x => x.error ? setErr(x.error) : setD(x)).catch(e => setErr(String(e)))
  }, [gameId])
  if (err) return <div className="note bad">No se pudo cargar la ficha: {err}.
    Comprueba que el servidor esté corriendo y ejecuta <code>python spc.py doctor</code>.</div>
  if (!d) return <div className="empty">Cargando ficha…</div>
  const p = d.partido, ctx = d.contexto
  const jugado = p.carreras != null || p.home.carreras != null

  return (
    <>
      {jugado && (p.home.carreras != null) && <div className="panel" style={{ margin: '12px 0' }}>
        <h2>Resultado final</h2>
        <div style={{ fontFamily: 'Barlow Condensed', fontSize: 28 }}>
          {p.away.abbr} {p.away.carreras} — {p.home.carreras} {p.home.abbr}
        </div>
        {p.home.carreras_f5 != null && <div style={{ fontSize: 12, color: 'var(--dim)' }}>
          Tras 5 entradas: {p.away.abbr} {p.away.carreras_f5} — {p.home.carreras_f5} {p.home.abbr}</div>}
      </div>}

      {(!d.abridores.home.name && !d.abridores.away.name) && <div className="note">
        Ningún abridor anunciado todavía para este partido. La MLB los publica normalmente
        entre 1 y 2 días antes. Si crees que ya deberían estar, ejecuta
        <code> python spc.py mlb-daily</code> y revisa <code>python spc.py doctor</code>.
      </div>}
      <h3 style={{ fontSize: 15, marginTop: 18 }}>Abridores</h3>
      <div className="dual">
        <Starter side={`VISITANTE · ${p.away.abbr}`} a={d.abridores.away} color="#4aa3ff" />
        <Starter side={`LOCAL · ${p.home.abbr}`} a={d.abridores.home} color="#39d353" />
      </div>

      <h3 style={{ fontSize: 15, marginTop: 18 }}>Equipos</h3>
      <div className="dual">
        {['away', 'home'].map(s => (
          <div className="sp" key={s}>
            <h4>{p[s].nombre || p[s].abbr}</h4>
            {(d.equipos[s] || []).map((x, i) => <Stat key={i} s={x} />)}
          </div>))}
      </div>

      <h3 style={{ fontSize: 15, marginTop: 18 }}>Bullpen (carga reciente)</h3>
      <div className="dual">
        {['away', 'home'].map(s => (
          <div className="sp" key={s}>
            <h4>{p[s].abbr}</h4>
            {(d.bullpen[s] || []).map((x, i) => <Stat key={i} s={x} />)}
          </div>))}
      </div>

      <h3 style={{ fontSize: 15, marginTop: 18 }}>Alineaciones</h3>
      <div className="dual">
        {['away', 'home'].map(s => {
          const l = d.alineaciones[s] || {}
          return (
            <div className="sp" key={s}>
              <h4>{p[s].abbr}</h4>
              <div className="meta">
                {l.estado === 'confirmed' ? '✓ confirmada' : `⚠️ ${l.estado}`}
                {l.available_at ? ` · ${String(l.available_at).replace('T', ' ').slice(0, 16)}` : ''}
              </div>
              {(l.orden || []).length === 0
                ? <div style={{ fontSize: 12, color: 'var(--warn)' }}>
                  Todavía no publicada. La predicción se marca como provisional.</div>
                : l.orden.map(b => (
                  <div className="statrow" key={b.pos}>
                    <span><b style={{ color: 'var(--dim)' }}>{b.pos}.</b> {b.name}</span>
                    <span style={{ color: 'var(--dim)', fontSize: 11 }}>{b.bats || ''}</span>
                  </div>))}
            </div>)
        })}
      </div>

      <h3 style={{ fontSize: 15, marginTop: 18 }}>Lesionados</h3>
      <div className="dual">
        {['away', 'home'].map(s => (
          <div className="sp" key={s}>
            <h4>{p[s].abbr} <span style={{ fontSize: 12, color: 'var(--dim)' }}>
              ({(d.lesiones[s] || []).length})</span></h4>
            {(d.lesiones[s] || []).length === 0
              ? <div style={{ fontSize: 12, color: 'var(--dim)' }}>Sin lesionados registrados.</div>
              : (d.lesiones[s] || []).slice(0, 14).map((x, i) => (
                <div className="inj" key={i}><b>{x.name}</b>
                  {x.posicion ? ` (${x.posicion})` : ''} — {x.estado}</div>))}
            {(d.lesiones[s] || []).length > 14 &&
              <div className="inj">…y {(d.lesiones[s] || []).length - 14} más</div>}
          </div>))}
      </div>
      <div className="note">Las lesiones son el estado ACTUAL del roster. La API no publica
        histórico, así que no se puede saber quién estaba lesionado en una fecha pasada.</div>

      <h3 style={{ fontSize: 15, marginTop: 18 }}>Contexto</h3>
      <dl className="kv">
        <dt>Park factor de carreras</dt>
        <dd>{ctx.park_runs_factor == null ? 'no disponible' : num(ctx.park_runs_factor, 3)}
          {ctx.park_runs_factor != null && (ctx.park_runs_factor > 1 ? ' (favorece anotar)' : ' (favorece pitcheo)')}</dd>
        <dt>Elo local / visitante</dt>
        <dd>{ctx.elo_home == null ? '—' : `${Math.round(ctx.elo_home)} / ${Math.round(ctx.elo_away)}`}</dd>
        <dt>Probabilidad Elo (local)</dt><dd>{pct(ctx.elo_prob_home)}</dd>
        <dt>Descanso local / visitante</dt>
        <dd>{ctx.descanso_home == null ? '—' : `${num(ctx.descanso_home, 1)} / ${num(ctx.descanso_away, 1)} días`}</dd>
        <dt>Clima</dt><dd>{ctx.clima?.motivo}</dd>
        <dt>Umpire</dt><dd>{ctx.umpire?.motivo}</dd>
      </dl>

      <h3 style={{ fontSize: 15, marginTop: 18 }}>Cuotas por casa</h3>
      {d.cuotas?.disponible === false
        ? <div className="note">{d.cuotas.motivo}</div>
        : Object.entries(d.cuotas).map(([mkt, casas]) => (
          <div key={mkt} style={{ marginBottom: 10 }}>
            <div className="grp">{mkt}</div>
            <table><thead><tr><th>Casa</th><th>Selección</th><th>Línea</th><th>Precio</th><th>Implícita</th></tr></thead>
              <tbody>{Object.entries(casas).flatMap(([bk, arr]) =>
                arr.map((o, i) => <tr key={bk + i}><td>{bk}</td><td>{o.seleccion}</td>
                  <td className="num">{o.linea ?? '—'}</td>
                  <td className="num">{o.precio > 0 ? '+' + o.precio : o.precio}</td>
                  <td className="num">{pct(o.prob_implicita, 1)}</td></tr>))}
              </tbody></table>
          </div>))}

      <h3 style={{ fontSize: 15, marginTop: 18 }}>Predicciones de este partido</h3>
      <table><thead><tr><th>Mercado</th><th>Selección</th><th>Final</th><th>Modelo</th>
        <th>Mercado</th><th>Elo</th><th>Riesgo</th><th>v</th><th>Resultado</th></tr></thead>
        <tbody>{d.predicciones.map((x, i) => (
          <tr key={i}>
            <td>{x.market}</td><td>{x.selection}</td>
            <td className="num">{pct(x.probabilidad)}</td>
            <td className="num">{pct(x.modelo)}</td>
            <td className="num">{x.mercado == null ? 'n/d' : pct(x.mercado)}</td>
            <td className="num">{x.elo == null ? '—' : pct(x.elo)}</td>
            <td className="num">{x.riesgo == null ? '—' : Math.round(x.riesgo) + '%'}</td>
            <td className="num">{x.version}</td>
            <td>{x.resultado === 'win' ? <span className="res ok">✓</span>
              : x.resultado === 'loss' ? <span className="res no">✗</span>
                : <span className="res pend">pendiente</span>}</td>
          </tr>))}</tbody></table>
    </>
  )
}
