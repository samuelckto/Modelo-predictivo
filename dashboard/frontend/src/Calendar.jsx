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
  return (
    <>
      {(data.warnings || []).map((w, i) => <div className="note" key={i}>{w}</div>)}
      {liveAt && <div style={{ fontSize: 11, color: 'var(--dim)', textAlign: 'right' }}>
        Marcadores MLB actualizados a las {liveAt} · se refrescan cada minuto</div>}
      {days.length === 0 &&
        <div className="empty">No hay partidos con predicción en este rango
          ({data.from} → {data.to}). El sistema no inventa partidos.</div>}
      {days.map(([k, gs]) => {
        const porDeporte = {}
        for (const g of gs) (porDeporte[g.sport] = porDeporte[g.sport] || []).push(g)
        return (
          <section key={k}>
            <div className="dayhead">
              <h3>{dayLabel(k)}</h3>
              <span className="n">{agrupar(gs).length} {agrupar(gs).length === 1 ? 'partido' : 'partidos'}</span>
            </div>
            {Object.entries(porDeporte).sort().map(([sp, lista]) => (
              <div key={sp} style={{ marginBottom: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '10px 0 8px' }}>
                  <span className={'pill ' + sp.toLowerCase()}>{sp}</span>
                  <span style={{ fontSize: 12, color: 'var(--dim)' }}>
                    {agrupar(lista).length} {agrupar(lista).length === 1 ? 'partido' : 'partidos'}</span>
                </div>
                <div className="grid">{agrupar(lista).map(g =>
                  <GroupedGameCard key={g.key} game={g} onOpen={onOpen} live={live} markets={markets} />)}</div>
              </div>))}
          </section>)
      })}
    </>
  )
}
