import React, { useEffect, useState } from 'react'

// Calculadora manual de momios. La parte importante no es la multiplicacion:
// es el aviso de correlacion. Multiplicar probabilidades de patas ligadas da un
// numero que parece exacto y no lo es.

const pct = v => v == null ? '—' : (v * 100).toFixed(2) + '%'
const vacia = () => ({ selection: '', event: '', event_id: '', sport: '', league: '',
  date: '', participants: '', odds_american: '', odds_decimal: '', model_probability: '' })

export default function ParlayCalculator() {
  const [patas, setPatas] = useState([vacia(), vacia()])
  const [stake, setStake] = useState(1)
  const [res, setRes] = useState(null)
  const [cargando, setCargando] = useState(false)

  const set = (i, k, v) => setPatas(p => p.map((x, j) => j === i ? { ...x, [k]: v } : x))
  const quitar = i => setPatas(p => p.filter((_, j) => j !== i))

  const calcular = () => {
    setCargando(true)
    const sel = patas
      .filter(p => p.odds_american !== '' || p.odds_decimal !== '')
      .map(p => ({
        selection: p.selection || '(sin nombre)',
        event: p.event || null, event_id: p.event_id || null,
        sport: p.sport || null, league: p.league || null, date: p.date || null,
        participants: p.participants ? p.participants.split(',').map(s => s.trim()).filter(Boolean) : [],
        odds_american: p.odds_american === '' ? null : Number(p.odds_american),
        odds_decimal: p.odds_decimal === '' ? null : Number(p.odds_decimal),
        model_probability: p.model_probability === '' ? null : Number(p.model_probability) / 100,
      }))
    // GET, no POST: el dashboard no expone ningun metodo de escritura y esa
    // garantia se comprueba en los tests. Esto solo calcula.
    const url = '/api/parlay/calculate?selecciones=' +
      encodeURIComponent(JSON.stringify(sel)) + '&stake=' + (Number(stake) || 1)
    fetch(url).then(r => r.json()).then(d => { setRes(d); setCargando(false) })
      .catch(() => setCargando(false))
  }

  useEffect(() => { calcular() }, [])   // arranca con la tabla vacia y su aviso

  return (
    <>
      <div className="panel">
        <h2>Calculadora de momios y combinadas</h2>
        <div className="note">
          Introduce la cuota en <b>americana</b> o en <b>decimal</b>: se usa la que
          rellenes. La probabilidad del modelo es opcional, en porcentaje (por
          ejemplo <code>62.5</code>). Si rellenas <code>event_id</code> o los
          participantes, detecto patas que <b>no son independientes</b>.
        </div>
        <div style={{ overflowX: 'auto', marginTop: 10 }}>
          <table><thead><tr>
            <th>Selección</th><th>Evento</th><th>ID evento</th><th>Deporte</th>
            <th>Liga</th><th>Fecha</th><th>Participantes</th>
            <th>Americana</th><th>Decimal</th><th>Modelo %</th><th></th>
          </tr></thead><tbody>
            {patas.map((p, i) => (
              <tr key={i}>
                {['selection', 'event', 'event_id', 'sport', 'league', 'date',
                  'participants', 'odds_american', 'odds_decimal', 'model_probability'].map(k => (
                    <td key={k}><input value={p[k]} onChange={e => set(i, k, e.target.value)}
                      placeholder={k === 'participants' ? 'A, B' : ''}
                      style={{ width: k === 'selection' || k === 'event' ? 130 : 78 }} /></td>))}
                <td><button className="tab" onClick={() => quitar(i)}
                  disabled={patas.length <= 1}>✕</button></td>
              </tr>))}
          </tbody></table>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 10, flexWrap: 'wrap' }}>
          <button className="tab" onClick={() => setPatas(p => [...p, vacia()])}>+ Añadir pata</button>
          <label style={{ fontSize: 12 }}>Stake (unidades){' '}
            <input value={stake} onChange={e => setStake(e.target.value)} style={{ width: 70 }} /></label>
          <button className="tab on" onClick={calcular} disabled={cargando}>
            {cargando ? 'Calculando…' : 'Calcular'}</button>
        </div>
      </div>

      {res && <div className="panel">
        <h2>Resultado</h2>
        {res.advertencia && <div className={'note ' + (res.nivel_correlacion === 'ALTA' ? 'bad' : '')}>
          <b>⚠️ {res.advertencia}</b></div>}
        {res.cuota_decimal == null
          ? <div className="empty" style={{ padding: 12 }}>
            {res.advertencia || 'Añade al menos una pata con cuota válida.'}</div>
          : <>
            <dl className="kv">
              <dt>Patas válidas</dt><dd>{res.n_validas} de {res.n_selecciones}</dd>
              <dt>Cuota combinada (decimal)</dt><dd className="mono"><b>{res.cuota_decimal.toFixed(4)}</b></dd>
              <dt>Cuota combinada (americana)</dt><dd className="mono">
                {res.cuota_americana > 0 ? '+' : ''}{res.cuota_americana}</dd>
              <dt>Probabilidad implícita combinada</dt><dd>{pct(res.probabilidad_implicita_combinada)}</dd>
              <dt>Probabilidad del modelo combinada</dt>
              <dd>{res.probabilidad_modelo_combinada == null
                ? <span style={{ color: 'var(--dim)' }}>no calculada ({res.cobertura_modelo} patas con modelo)</span>
                : <b>{pct(res.probabilidad_modelo_combinada)}</b>}</dd>
              {res.gap_pp != null && <>
                <dt>Diferencia modelo − mercado</dt>
                <dd style={{ color: res.gap_pp > 0 ? 'var(--ok)' : 'var(--bad)' }}>
                  {res.gap_pp > 0 ? '+' : ''}{res.gap_pp.toFixed(2)} pp</dd>
                <dt>Cuota justa según el modelo</dt><dd className="mono">{res.cuota_justa_modelo}</dd>
              </>}
              <dt>Pago (stake {res.stake})</dt><dd className="mono">{res.payout.toFixed(2)}</dd>
              <dt>Beneficio</dt><dd className="mono" style={{ color: 'var(--ok)' }}>
                +{res.profit.toFixed(2)}</dd>
            </dl>
            {res.nota_modelo && <div className="note">{res.nota_modelo}</div>}
            {res.recordatorio && <div className="note">{res.recordatorio}</div>}
            <div className="note">{res.supuesto}</div>
            {res.descartadas?.length > 0 && <div className="note bad">
              Descartadas: {res.descartadas.map(d => `${d.selection} (${d.motivo})`).join(' · ')}</div>}
            {res.avisos_correlacion?.length > 0 && <div style={{ marginTop: 10 }}>
              <h3 style={{ fontSize: 13, margin: '0 0 6px' }}>Patas correlacionadas</h3>
              <table><thead><tr><th>Nivel</th><th>Pata A</th><th>Pata B</th><th>Por qué</th></tr></thead>
                <tbody>{res.avisos_correlacion.map((a, i) => (
                  <tr key={i}>
                    <td style={{ color: a.nivel === 'ALTA' ? 'var(--bad)' : 'var(--warn)', fontWeight: 600 }}>
                      {a.nivel}</td>
                    <td style={{ fontSize: 12 }}>{a.pata_a}</td>
                    <td style={{ fontSize: 12 }}>{a.pata_b}</td>
                    <td style={{ fontSize: 12, color: 'var(--dim)' }}>{a.explicacion}</td>
                  </tr>))}</tbody></table>
            </div>}
          </>}
      </div>}
    </>
  )
}
