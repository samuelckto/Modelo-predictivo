import React, { useEffect, useMemo, useState } from 'react'
import { getGames } from '../api'
import GroupedGameCard, { agrupar } from '../components/GroupedGameCard'
import { localDayKey, dayLabel } from '../format'

export default function Calendar({ sport, range, start, end, onOpen }) {
  const [data, setData] = useState(null), [err, setErr] = useState(null)
  const [live, setLive] = useState({})
  const [markets, setMarkets] = useState(null)
  useEffect(() => { fetch('/api/markets').then(r => r.json()).then(setMarkets).catch(() => {}) }, [])
  const [liveAt, setLiveAt] = useState(null)
  useEffect(() => {
    let vivo = true
    const tick = () => fetch('/api/live').then(r => r.json()).then(d => {
      if (!vivo) return
      setLive(d.games || {}); setLiveAt(d.actualizado || null)
    }).catch(() => {})
    tick()
    const id = setInterval(tick, 60000)
    return () => { vivo = false; clearInterval(id) }
  }, [])
  useEffect(() => {
    setData(null); setErr(null)
    getGames(sport, range, start, end).then(setData).catch(e => setErr(String(e)))
  }, [sport, range, start, end])

  const days = useMemo(() => {
    if (!data) return []
    const all = data.days.flatMap(d => d.games)
    const m = new Map()
    for (const g of all) {
      const k = localDayKey(g.start_utc)
      if (!m.has(k)) m.set(k, [])
      m.get(k).push(g)
    }
    return [...m.entries()].sort((a, b) => a[0] < b[0] ? -1 : 1)
      .map(([k, gs]) => [k, gs.sort((a, b) => String(a.start_utc).localeCompare(String(b.start_utc)))])
  }, [data])

  if (err) return <div className="note bad">No se pudo leer la API: {err}</div>
  if (!data) return <div className="empty">Cargando…</div>

  // Antes cada día repetía una cabecera por deporte con su píldora y su cuenta,
  // encima de unas tarjetas que YA llevan la píldora del deporte. Dos filas de
  // cronología por cada bloque para no decir nada nuevo. Ahora hay una sola
  // cabecera por día y las tarjetas se ordenan por deporte y hora.
  const total = days.reduce((n, [, gs]) => n + agrupar(gs).length, 0)
  const deportes = [...new Set(days.flatMap(([, gs]) => gs.map(g => g.sport)))]

  return (
    <>
      {(data.warnings || []).map((w, i) => <div className="note" key={i}>{w}</div>)}

      {days.length === 0
        ? <div className="empty">No hay partidos con predicción entre {data.from} y {data.to}.
            El sistema no inventa partidos.</div>
        : <div className="hero">
            <div>
              <h2>{total} {total === 1 ? 'partido' : 'partidos'}</h2>
              <p>{deportes.length} {deportes.length === 1 ? 'deporte' : 'deportes'} ·
                {' '}{data.from} → {data.to}
                {liveAt && <> · marcadores al minuto, últimos a las {liveAt}</>}</p>
            </div>
          </div>}

      {days.map(([k, gs]) => {
        const juegos = agrupar(gs).sort((a, b) =>
          a.sport === b.sport ? String(a.start_utc).localeCompare(String(b.start_utc))
            : a.sport.localeCompare(b.sport))
        return (
          <section key={k}>
            <div className="dayhead">
              <h3>{dayLabel(k)}</h3>
              <span className="n">{juegos.length}</span>
            </div>
            <div className="grid">{juegos.map(g =>
              <GroupedGameCard key={g.key} game={g} onOpen={onOpen}
                live={live} markets={markets} />)}</div>
          </section>)
      })}
    </>
  )
}
