import React, { useEffect, useState } from 'react'

const color = s => s === 'ONLINE' ? 'var(--ok)' : s && s.startsWith('SIN') ? 'var(--warn)' : 'var(--bad)'

export default function DataHealth() {
  const [d, setD] = useState(null)
  useEffect(() => { fetch('/api/data-health').then(r => r.json()).then(setD) }, [])
  if (!d) return <div className="empty">Cargando…</div>
  const Tabla = ({ sport, rows }) => (
    <div className="panel">
      <h2>{sport}</h2>
      {(!rows || rows.length === 0) ? <div className="empty" style={{ padding: 12 }}>Sin registros.</div> :
        <table><thead><tr><th>Fuente</th><th>Dominio</th><th>Estado</th><th>Última descarga</th>
          <th>Descargas</th><th>Fallos totales</th><th>Fallos 7 días</th><th>Registros</th>
          <th>Cobertura</th></tr></thead>
          <tbody>{rows.map((r, i) => (
            <React.Fragment key={i}>
              <tr>
                <td>{r.fuente}</td><td>{r.dominio}</td>
                <td style={{ color: color(r.estado), fontWeight: 600 }}>{r.estado}</td>
                <td className="mono">{(r.ultima_descarga || '—').toString().replace('T', ' ').slice(0, 19)}</td>
                <td className="num">{r.descargas ?? '—'}</td>
                <td className="num" style={{ color: r.errores ? 'var(--dim)' : undefined }}>{r.errores ?? '—'}</td>
                <td className="num" style={{ color: r.errores_recientes ? 'var(--warn)' : undefined }}>
                  {r.errores_recientes ?? '—'}</td>
                <td className="num">{r.registros ?? '—'}</td>
                <td style={{ color: r.cobertura_ok === false ? 'var(--bad)' : 'var(--dim)' }}>
                  {r.cobertura}</td>
              </tr>
              {(r.nota || r.ultimo_error) && <tr>
                <td colSpan={9} style={{ fontSize: 11, color: 'var(--dim)', paddingTop: 0 }}>
                  {r.nota && <span>✓ {r.nota}. </span>}
                  {r.ultimo_error && <span>
                    Último fallo{r.ultimo_error.tipo ? ` (${r.ultimo_error.tipo})` : ''}:
                    {' '}{r.ultimo_error.mensaje}
                    {r.ultimo_error.consejo && <> — <b style={{ color: 'var(--warn)' }}>
                      {r.ultimo_error.consejo}</b></>}
                  </span>}
                </td></tr>}
            </React.Fragment>))}</tbody></table>}
    </div>
  )
  const o = d.odds || {}
  const r = d.resumen_mlb || {}
  const diag = d.diagnostico || {}
  return (
    <>
      {diag.problema && <div className="panel">
        <h2 style={{ color: 'var(--warn)' }}>⚠️ {diag.problema}</h2>
        <p style={{ fontSize: 13, margin: '0 0 8px' }}>{diag.detalle}</p>
        {diag.fuentes?.length > 0 && <div style={{ fontSize: 12, color: 'var(--dim)' }}>
          Fuentes afectadas: {diag.fuentes.join(', ')}</div>}
        {diag.consejo && <div className="note">{diag.consejo}</div>}
        <div className="note ok">Para un diagnóstico completo desde la terminal:
          <code> python spc.py doctor</code></div>
      </div>}
      <div className="panel">
        <h2>¿Faltan datos?</h2>

        <dl className="kv">
          <dt>Boxscores de partidos terminados</dt>
          <dd style={{ color: r.boxscores_ok ? 'var(--ok)' : 'var(--bad)' }}>
            {r.boxscores} {r.boxscores_ok ? '✓ completo' : '✗ faltan'}</dd>
          <dt>Días de juego sin Statcast</dt>
          <dd style={{ color: r.n_dias_faltantes ? 'var(--warn)' : 'var(--ok)' }}>
            {r.n_dias_faltantes === 0 ? 'ninguno ✓'
              : `${r.n_dias_faltantes} — ${(r.dias_statcast_faltantes || []).join(', ')}`}</dd>
        </dl>
        <div className={'note ' + (r.veredicto?.startsWith('cobertura completa') ? 'ok' : '')}>
          {r.veredicto}</div>
      </div>
      <Tabla sport="MLB" rows={d.MLB} />
      <Tabla sport="NFL" rows={d.NFL} />
      {d.SOCCER && !d.SOCCER.error && <div className="panel">
        <h2><span className="pill soccer">FÚTBOL</span> Data Health</h2>
        <table className="tbl" style={{ marginTop: 10 }}>
          <thead><tr><th>Liga</th><th>Estado</th><th>Histórico</th><th>Última fecha</th>
            <th>Antigüedad</th><th>Córners</th><th>Próximos</th><th>Sin cuotas</th></tr></thead>
          <tbody>
            {Object.entries(d.SOCCER.ligas || {}).map(([k, v]) => (
              <tr key={k}>
                <td>{v.liga}</td>
                <td style={{ color: v.estado === 'OK' ? 'var(--ok)' : v.estado === 'STALE' ? 'var(--warn)' : 'var(--bad)' }}>
                  <b>{v.estado}</b></td>
                <td className="mono">{v.partidos_historicos}</td>
                <td className="mono">{v.ultima_fecha_historica || '—'}</td>
                <td className="mono" style={{ color: (v.antiguedad_dias || 0) > 45 ? 'var(--warn)' : 'inherit' }}>
                  {v.antiguedad_dias != null ? v.antiguedad_dias + ' d' : '—'}</td>
                <td className="mono" style={{ color: v.corners_partidos ? 'inherit' : 'var(--bad)' }}>
                  {v.corners_partidos || 'sin fuente'}</td>
                <td className="mono">{v.fixtures_proximos}</td>
                <td className="mono" style={{ color: v.fixtures_sin_cuotas ? 'var(--warn)' : 'var(--ok)' }}>
                  {v.fixtures_sin_cuotas}</td>
              </tr>))}
          </tbody>
        </table>
        <dl className="kv" style={{ marginTop: 10 }}>
          <dt>Snapshots de cuotas</dt><dd className="mono">{d.SOCCER.snapshots_de_cuotas}</dd>
          <dt>Último snapshot</dt>
          <dd className="mono" style={{ color: (d.SOCCER.edad_snapshot_horas || 0) > 6 ? 'var(--warn)' : 'var(--ok)' }}>
            {d.SOCCER.ultimo_snapshot || 'ninguno'}{d.SOCCER.edad_snapshot_horas != null
              ? ` (${d.SOCCER.edad_snapshot_horas} h)` : ''}</dd>
          <dt>Predicciones activas</dt><dd className="mono">{d.SOCCER.predicciones_activas}</dd>
          <dt>Por estado</dt><dd className="mono">{JSON.stringify(d.SOCCER.por_estado)}</dd>
        </dl>
        {(d.SOCCER.avisos || []).map((a, i) => (
          <div key={i} className="note bad" style={{ marginTop: 6 }}>{a}</div>))}
        {(d.SOCCER.errores_recientes || []).length > 0 && <div className="note" style={{ marginTop: 6 }}>
          Errores recientes de fuente: {d.SOCCER.errores_recientes.map(e => `${e.ambito}: ${e.error}`).join(' · ')}</div>}
      </div>}
      {d.tenis && !d.tenis.error && <div className="panel">
        <h2><span className="pill tenis">TENIS</span> Data Health · {d.tenis.estado_global}</h2>
        <dl className="kv" style={{ marginTop: 8 }}>
          <dt>Archivo histórico congelado en</dt>
          <dd style={{ color: (d.tenis.antiguedad_archivo_dias || 0) > 45 ? 'var(--warn)' : 'var(--ok)' }}>
            {d.tenis.archivo_congelado_en} ({d.tenis.antiguedad_archivo_dias} días)</dd>
          <dt>Calendario (eventos)</dt><dd>{d.tenis.calendario_eventos}</dd>
          <dt>Cuotas</dt><dd>{d.tenis.cuotas_snapshots} snapshots · última descarga {d.tenis.ultima_descarga || '—'}</dd>
          <dt>Errores de descarga</dt>
          <dd style={{ color: d.tenis.errores_descarga ? 'var(--warn)' : 'var(--ok)' }}>{d.tenis.errores_descarga}</dd>
          <dt>Cuotas históricas</dt><dd>{d.tenis.cuotas_historicas}</dd>
          <dt>Leakage</dt><dd>{d.tenis.leakage}</dd>
          {d.tenis.resultados && <>
            <dt>Resultados</dt>
            <dd>{d.tenis.resultados.fuente} · {d.tenis.resultados.partidos_con_resultado} partidos</dd>
            <dt>Jugadas sin calificar</dt>
            <dd style={{ color: d.tenis.resultados.predicciones_jugadas_sin_calificar ? 'var(--warn)' : 'var(--ok)' }}>
              {d.tenis.resultados.predicciones_jugadas_sin_calificar}
              {' '}({Object.entries(d.tenis.resultados.por_mercado_sin_calificar || {})
                .filter(([, v]) => v).map(([k, v]) => `${k}: ${v}`).join(' · ') || 'ninguna'})</dd>
            <dt>Limitación</dt><dd>{d.tenis.resultados.limitacion}</dd>
          </>}
        </dl>
        <table style={{ marginTop: 8 }}>
          <thead><tr><th>Circuito</th><th>Estado</th><th>Partidos</th><th>Último</th><th>Modelo</th>
            <th>Ganador</th><th>Total juegos</th><th>Hándicap</th><th>Predicciones</th></tr></thead>
          <tbody>{Object.entries(d.tenis.circuitos || {}).map(([k, c]) => <tr key={k}>
            <td>{k}</td><td>{c.estado === 'OK' ? '🟢' : c.estado === 'WARNING' ? '🟡' : '🔴'} {c.estado}</td>
            <td className="num">{c.partidos}</td><td className="mono" style={{ fontSize: 11 }}>{c.ultimo_partido}</td>
            <td>{c.modelo || '—'}</td>
            {['winner', 'total_games', 'handicap_games'].map(m => <td key={m}>
              {c.mercados[m].modo === 'pick' ? '✅ PICK' : c.mercados[m].modo === 'projection' ? '🟡 PROYECCIÓN' : '🔴 BLOQUEADO'}</td>)}
            <td className="num">{c.predicciones_activas} ({c.con_resultado} con resultado)</td></tr>)}</tbody></table>
      </div>}
      {d.nba && !d.nba.error && <div className="panel">
        <h2><span className="pill nba">NBA</span> Data Health · {d.nba.estado_global}</h2>
        <dl className="kv" style={{ marginTop: 8 }}>
          <dt>Partidos en la base</dt><dd>{d.nba.partidos} ({d.nba.partidos_terminados} terminados)</dd>
          <dt>Último partido</dt><dd>{d.nba.ultimo_partido || '—'}</dd>
          <dt>Última descarga</dt><dd>{d.nba.ultima_descarga || '—'}</dd>
          <dt>Logs faltantes</dt><dd style={{ color: d.nba.logs_faltantes ? 'var(--warn)' : 'var(--ok)' }}>{d.nba.logs_faltantes}</dd>
          <dt>Errores de descarga</dt><dd style={{ color: d.nba.errores_descarga ? 'var(--warn)' : 'var(--ok)' }}>{d.nba.errores_descarga}</dd>
          <dt>Cuotas</dt><dd>{d.nba.cuotas_snapshots} snapshots · {d.nba.cuotas_partidos} partidos · última {d.nba.cuotas_ultima || '—'}</dd>
          <dt>Cuotas históricas</dt><dd>{d.nba.cuotas_historicas}</dd>
          <dt>Lesiones (timestamp)</dt><dd>{d.nba.lesiones_timestamp}</dd>
          <dt>Quintetos (timestamp)</dt><dd>{d.nba.quintetos_timestamp}</dd>
          <dt>Leakage</dt><dd>{d.nba.leakage}</dd>
        </dl>
        <table style={{ marginTop: 8 }}><thead><tr><th>Mercado</th><th>Estado</th><th>Modo</th><th>Modelo</th><th>Features</th><th>Predicciones</th><th>Con resultado</th></tr></thead>
          <tbody>{Object.entries(d.nba.mercados || {}).map(([k, m]) => <tr key={k}>
            <td style={{ textTransform: 'uppercase' }}>{k}</td>
            <td>{m.estado === 'OK' ? '🟢' : m.estado === 'WARNING' ? '🟡' : '🔴'} {m.estado}</td>
            <td>{m.modo === 'pick' ? 'PICK' : m.modo === 'projection' ? 'PROYECCIÓN' : m.modo.toUpperCase()}</td>
            <td>{m.modelo || '—'}</td><td className="num">{m.features ?? '—'}</td>
            <td className="num">{m.predicciones_activas}</td><td className="num">{m.con_resultado}</td></tr>)}</tbody></table>
      </div>}
      {d.nfl_mercados && !d.nfl_mercados.error && <div className="panel">
        <h2>NFL por mercado</h2>
        <table><thead><tr><th>Mercado</th><th>Estado</th><th>Modo</th><th>Modelo</th><th>Features</th>
          <th>Predicciones</th><th>Última predicción</th><th>Cuotas</th><th>Con resultado</th></tr></thead>
          <tbody>{Object.entries(d.nfl_mercados).map(([k, m]) => <tr key={k}>
            <td style={{ textTransform: 'uppercase' }}>{k}</td>
            <td>{m.estado === 'verde' ? '🟢' : m.estado === 'amarillo' ? '🟡' : '🔴'} {m.estado}</td>
            <td>{m.modo === 'pick' ? 'PICK' : m.modo === 'proyeccion' ? 'PROYECCIÓN' : m.modo}</td>
            <td>{m.modelo || '—'}</td><td className="num">{m.features ?? '—'}</td>
            <td className="num">{m.predicciones_activas ?? '—'}</td>
            <td className="mono" style={{ fontSize: 11 }}>{m.ultima_prediccion ? String(m.ultima_prediccion).slice(0, 16) : (m.nota || '—')}</td>
            <td className="num">{m.cuotas_snapshots ?? '—'}{m.mercado_disponible === false ? ' (sin mercado)' : ''}</td>
            <td className="num">{m.con_resultado ?? '—'}</td>
          </tr>)}</tbody></table>
      </div>}
      <div className="panel">
        <h2>Histórico de mercado</h2>
        <dl className="kv">
          <dt>Estado</dt><dd style={{ color: color(o.estado) }}>{o.estado}</dd>
          <dt>Snapshots guardados</dt><dd>{o.rows ?? 0}</dd>
          <dt>Partidos con cuotas</dt><dd>{o.games ?? 0}</dd>
          <dt>Casas de apuestas</dt><dd>{o.bookmakers ?? 0}</dd>
          <dt>Primer snapshot</dt><dd>{o.first || '—'}</dd>
          <dt>Último snapshot</dt><dd>{o.last || '—'}</dd>
          <dt>Listo para comparar modelo vs mercado</dt>
          <dd>{o.listo_para_comparar_mercado ? 'sí' : 'todavía no'}</dd>
        </dl>

      </div>
    </>
  )
}
