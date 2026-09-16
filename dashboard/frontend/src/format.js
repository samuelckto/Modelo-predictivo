export const pct = (v, d = 1) => (v === null || v === undefined) ? '—' : (v * 100).toFixed(d) + '%'
export const num = (v, d = 3) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d)
const asUTC = (s) => {
  if (!s) return null
  let t = String(s).replace(' ', 'T')
  if (!/[zZ]|[+-]\d\d:?\d\d$/.test(t)) t += 'Z'
  const d = new Date(t)
  return isNaN(d) ? null : d
}
export const localTime = (s) => {
  const d = asUTC(s)
  return d ? d.toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit', hour12: false }) : '—'
}
// Fecha corta + hora. Sin la fecha, tres partidos de la MISMA serie (COL en DET
// los días 11, 12 y 13) se veían idénticos en Top picks: mismos equipos, misma
// probabilidad y solo la hora cambiando. Parecían un pick repetido tres veces.
export const localDateTime = (s) => {
  const d = asUTC(s)
  if (!d) return '—'
  const fecha = d.toLocaleDateString('es-MX', { day: '2-digit', month: 'short' })
  return `${fecha} · ${localTime(s)}`
}
export const localDayKey = (s) => {
  const d = asUTC(s); if (!d) return 'sin-fecha'
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}
export const dayLabel = (key) => {
  if (key === 'sin-fecha') return 'Sin fecha'
  const [y, m, dd] = key.split('-').map(Number)
  const d = new Date(y, m - 1, dd)
  const s = d.toLocaleDateString('es-MX', { weekday: 'long', day: 'numeric', month: 'long' })
  return s.charAt(0).toUpperCase() + s.slice(1)
}
export const riskClass = (r) => r === null || r === undefined ? '' :
  r >= 65 ? 'risk-high' : r >= 40 ? 'risk-med' : 'risk-low'
export const riskText = (r) => r === null || r === undefined ? 'SIN DATO' :
  r >= 65 ? 'RIESGO ALTO' : r >= 40 ? 'RIESGO MEDIO' : 'RIESGO BAJO'
const RISK_ES = { 'HIGH RISK': 'RIESGO ALTO', 'MEDIUM RISK': 'RIESGO MEDIO', 'LOW RISK': 'RIESGO BAJO' }
export const riskLabel = (label, r) => label ? (RISK_ES[label] || label) : riskText(r)
