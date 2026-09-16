import React, { useEffect, useState } from 'react'
import { getTopPicks, getValuePicks } from '../api'
import GameCard from '../components/GameCard'

// Esta pantalla enseñaba los MISMOS picks tres veces seguidas: una lista
// numerada, una tabla y las tarjetas. Tres formatos del mismo dato es ruido,
// no información. Ahora manda la tarjeta —que es lo que se mira— y la tabla
// queda a un clic para quien quiera compararlos en columnas.

const MK = { moneyline: 'Moneyline', total: 'Totales', spread: 'Spread', run_line: 'Run line',
             winner: 'Ganador', total_games: 'Total de juegos', handicap_games: 'Hándicap',
             total_goals: 'Total de goles', btts: 'Ambos marcan', double_chance: 'Gana o empata' }
const mkey = p => (p.extra && p.extra.market_key) || String(p.market || '').toLowerCase()

// Dos listas distintas, y la diferencia importa:
//   SEGUROS  -> mayor probabilidad. Es lo que mas veces ocurre.
//   VALOR    -> mayor EV estimado. Es lo que paga mas de lo que el modelo cree
//               que vale. Un 93 % a cuota 1.05 es segurísimo y pierde dinero.
// Ninguna de las dos está validada; la de valor, menos aún, porque depende de
// que la probabilidad del modelo esté bien calibrada.
const ORDENES = [['prob', 'Más seguros'], ['valor', 'Más valor'],
                 ['contra', 'Contra el mercado']]

export default function TopPicks({ sport, range, start, end, onOpen }) {
  const [d, setD] = useState(null), [err, setErr] = useState(null)
  const [mk, setMk] = useState('all')
  const [orden, setOrden] = useState('prob')
  const [v, setV] = useState(null)
  useEffect(() => {
    setD(null); setErr(null); setMk('all'); setV(null)
    getTopPicks(sport, range, start, end).then(setD).catch(e => setErr(String(e)))
    getValuePicks(sport, range, start, end).then(setV).catch(() => setV(null))
  }, [sport, range, start, end])

  if (err) return <div className="note bad">{err}</div>
  if (!d) return <div className="empty">Cargando…</div>

  const ties = new Set(d.tie_groups.flat())
  // La lista base cambia con el orden: «más valor» y «contra el mercado» NO son
  // un reordenamiento de los 12 más seguros, son otros partidos.
  const base = orden === 'valor' ? ((v && v.valor) || [])
    : orden === 'contra' ? ((v && v.contra_mercado) || [])
    : d.picks
  const mercados = [...new Set(base.map(mkey))]
  const picks = base.filter(p => mk === 'all' || mkey(p) === mk)
  const validados = picks.filter(p => p.validated_pick).length

  // Tres partidos de la MISMA serie (COL en DET los días 11, 12 y 13) salían
  // con la misma probabilidad exacta y parecían un pick duplicado. NO lo son:
  // son partidos distintos. Se repite el número porque aún no hay abridor
  // anunciado, y sin abridor el modelo solo ve la fuerza de los equipos, que
  // no cambia de un día a otro. Eso hay que decirlo, no dejar que se adivine.
  const porSerie = {}
  for (const p of picks) {
    const k = `${p.sport}|${p.away}@${p.home}|${mkey(p)}`
    ;(porSerie[k] = porSerie[k] || []).push(p)
  }
  const series = Object.values(porSerie).filter(l => l.length > 1)
  const seriesSinAbridor = series.filter(l =>
    l.some(p => ((p.data_completeness || {}).missing || [])
      .some(m => /abridor/i.test(m))))

  return (
    <>
      <div className="hero">
        <div>
          <h2>Top picks</h2>
          <p>{d.from} → {d.to} · {picks.length} {picks.length === 1 ? 'pick' : 'picks'}
            {picks.length > 0 && validados < picks.length && <> · {validados} validados;
              los otros {picks.length - validados} son probabilidades publicadas,
              no recomendaciones</>}</p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <div className="seg">
            {ORDENES.map(([k, l]) =>
              <button key={k} className={orden === k ? 'on' : ''}
                onClick={() => { setOrden(k); setMk('all') }}>{l}</button>)}
          </div>
          {mercados.length > 1 && (
            <div className="seg">
              <button className={mk === 'all' ? 'on' : ''} onClick={() => setMk('all')}>Todos</button>
              {mercados.map(m =>
                <button key={m} className={mk === m ? 'on' : ''} onClick={() => setMk(m)}>
                  {MK[m] || m}</button>)}
            </div>)}
        </div>
      </div>

      {orden === 'valor' && (
        <div className="note">
          <b>Valor no es lo mismo que seguridad.</b> Esta lista ordena por EV estimado
          = probabilidad del modelo × cuota real − 1. Un 93 % a cuota 1.05 es
          segurísimo y pierde dinero; un 40 % a cuota 4.00 es arriesgado y tiene valor.
          {v && <> Solo {v.con_precio_registrado} de {v.total_tarjetas} predicciones tienen
            cuota registrada: del resto no se puede calcular valor y no aparecen.</>}
          {' '}El EV solo vale si la probabilidad del modelo está bien calibrada,
          y eso <b>no está demostrado</b> con el histórico de cuotas que hay hoy.
        </div>)}

      {orden === 'contra' && (
        <div className="note">
          Partidos donde el modelo y la casa apuntan a <b>lados opuestos</b>. Que
          discrepen es un hecho observable; que el modelo tenga razón <b>no está
          demostrado</b>. Ordenados por tamaño de la discrepancia.
        </div>)}

      {seriesSinAbridor.length > 0 && (
        <div className="note">
          Verás {seriesSinAbridor.length === 1 ? 'una serie' : `${seriesSinAbridor.length} series`}
          {' '}con el mismo porcentaje repetido en varios días. No es un pick duplicado:
          son partidos distintos de la misma serie, y comparten número porque
          <b> todavía no hay abridor anunciado</b>. Sin abridor el modelo solo ve la fuerza
          de los equipos, que no cambia de un día a otro. El número cambiará cuando se
          anuncien los lanzadores.
        </div>)}

      {picks.length === 0
        ? <div className="empty">Ninguna oportunidad calculable en este rango.
            El sistema prefiere no publicar nada a publicar relleno.</div>
        : <>
          <div className="grid">
            {picks.map((p, i) => (
              <div key={i} style={ties.has(p.game_id)
                ? { outline: '1px dashed var(--warn)', borderRadius: 17 } : null}>
                <GameCard c={p} onOpen={onOpen} />
              </div>))}
          </div>

          <details className="panel" style={{ marginTop: 18 }}>
            <summary style={{ cursor: 'pointer', fontWeight: 700, fontSize: 13 }}>
              Ver los mismos picks como tabla
            </summary>
            <table style={{ marginTop: 12 }}>
              <thead><tr><th>#</th><th>Deporte</th><th>Mercado</th><th>Selección</th>
                <th className="num">Prob.</th><th className="num">Tasa base</th>
                <th className="num">Ventaja</th><th>Pick</th></tr></thead>
              <tbody>{picks.map((p, i) => {
                const ex = p.extra || {}
                return <tr key={i}>
                  <td className="num">{i + 1}</td>
                  <td><span className={'pill ' + p.sport.toLowerCase()}>{p.sport}</span></td>
                  <td>{MK[mkey(p)] || p.market}</td>
                  <td>{p.selection}</td>
                  <td className="num">{(p.ensemble_probability * 100).toFixed(1)}%</td>
                  <td className="num">{ex.base_rate_selection == null ? '—'
                    : (ex.base_rate_selection * 100).toFixed(1) + '%'}</td>
                  <td className="num">{p.edge_vs_base == null ? '—'
                    : ((p.edge_vs_base * 100 >= 0 ? '+' : '')
                       + (p.edge_vs_base * 100).toFixed(1) + ' pp')}</td>
                  <td>{p.validated_pick
                    ? <span className="tag risk-low">validado</span>
                    : <span className="tag warn">solo probabilidad</span>}</td>
                </tr>
              })}</tbody>
            </table>
          </details>
        </>}
    </>
  )
}
