const j = async (u) => {
  const r = await fetch(u)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}
export const getStatus = () => j('/api/status')
export const getGames = (sport, range, start, end) => {
  const q = new URLSearchParams({ sport, range })
  if (start) q.set('start', start); if (end) q.set('end', end)
  return j('/api/games?' + q)
}
export const getTopPicks = (sport, range, start, end) => {
  const q = new URLSearchParams({ sport, range, limit: 12 })
  if (start) q.set('start', start); if (end) q.set('end', end)
  return j('/api/top-picks?' + q)
}
// Lista aparte de Top picks a proposito: aquella ordena por probabilidad y solo
// devuelve los 12 mas seguros, que casi nunca son los de mas valor.
export const getValuePicks = (sport, range, start, end) => {
  const q = new URLSearchParams({ sport, range, limit: 24 })
  if (start) q.set('start', start); if (end) q.set('end', end)
  return j('/api/value-picks?' + q)
}
export const getPerformance = (sport) => j('/api/performance?sport=' + sport)
export const getSources = () => j('/api/sources')
