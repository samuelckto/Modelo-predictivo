import React, { useEffect, useState } from 'react'

// Tres cosas que la gente confunde y que aqui se muestran SEPARADAS:
//   probabilidad cruda    lo que salio del modelo
//   probabilidad calibrada  la cruda corregida por su historial de confianza
//   probabilidad publicada  la que ya lleva mercado dentro (cuando el deporte mezcla)
//   implicita del mercado   la del casino, sin vig
// Juntarlas en una sola columna es exactamente como se pierde de vista cuanto
// del numero es del modelo y cuanto es del casino.

const pct = v => v == null ? '—' : (v * 100).toFixed(1) + '%'
const pp = v => v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(1)

const COLOR_ESTADO = {
  'VALIDATED VALUE': 'var(--ok)',
  'STRONG MODEL-MARKET DISAGREEMENT': 'var(--warn)',
  'MINOR DISAGREEMENT': 'var(--dim)',
  'UNVALIDATED': 'var(--warn)',
  'ALIGNED': 'var(--dim)',
  'NO MARKET': 'var(--dim)',
  'NO PICK': 'var(--bad)',
}

const SPORTS = ['all', 'MLB', 'NFL', 'NBA', 'TENIS', 'SOCCER']

export default function ValueCenter() {
  const [sport, setSport] = useState('all')
  const [mvm, setMvm] = useState(null)
  const [fav, setFav] = useState(null)
  const [dis, setDis] = useState(null)
  const [cob, setCob] = useState(null)

  useEffect(() => {
    setMvm(null); setFav(null); setDis(null)
    const q = `?sport=${sport}`
    fetch('/api/value/model-vs-market' + q).then(r => r.json()).then(setMvm)
    fetch('/api/value/favorites' + q).then(r => r.json()).then(setFav)
    fetch('/api/value/discrepancies' + q).then(r => r.json()).then(setDis)
  }, [sport])
  useEffect(() => { fetch('/api/value/coverage').then(r => r.json()).then(setCob) }, [])

  return (
    <>
      <div className="panel">
        <h2>Modelo vs mercado</h2>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '8px 0' }}>
          {SPORTS.map(s => (
            <button key={s} className={'tab' + (sport === s ? ' on' : '')}
              onClick={() => setSport(s)}>{s === 'all' ? 'Todos' : s}</button>))}
        </div>
        <div className="note">
          El modelo se calcula <b>antes</b> de mirar el mercado y nunca se ajusta
          hacia él. Que el modelo dé más probabilidad que la casa <b>no es valor</b>:
          es una hipótesis que solo se valida con el backtest de más abajo.
        </div>
        {!mvm ? <div className="empty">Cargando…</div> : <>
          <div style={{ fontSize: 12, color: 'var(--dim)', margin: '8px 0' }}>
            {mvm.n} predicciones activas · {mvm.con_mercado} con cuota registrada ·
            {' '}{mvm.sin_mercado} sin mercado con el que comparar
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table><thead><tr>
              <th>Deporte</th><th>Partido</th><th>Mercado</th><th>Selección</th>
              <th>Cruda</th><th>Calibrada</th><th>Publicada</th><th>Mercado</th>
              <th>Cuota</th><th>Gap</th><th>Tramo</th><th>Estado</th>
            </tr></thead><tbody>
              {mvm.filas.map((f, i) => (
                <tr key={i}>
                  <td><span className={'pill ' + String(f.sport).toLowerCase()}>{f.sport}</span></td>
                  <td className="mono" style={{ fontSize: 11 }}>{f.event_id}</td>
                  <td>{f.market}</td>
                  <td style={{ fontSize: 12 }}>{f.selection}</td>
                  <td className="num">{pct(f.MODEL_PROBABILITY_RAW)}</td>
                  <td className="num"><b>{pct(f.MODEL_PROBABILITY_CALIBRATED)}</b></td>
                  <td className="num" style={{ color: 'var(--dim)' }}>{pct(f.published_probability)}</td>
                  <td className="num">{pct(f.MARKET_IMPLIED_PROBABILITY)}</td>
                  <td className="num mono">{f.MARKET_ODDS ? f.MARKET_ODDS.toFixed(2) : '—'}</td>
                  <td className="num" style={{ color: f.gap_pp > 0 ? 'var(--ok)' : f.gap_pp < 0 ? 'var(--bad)' : undefined }}>
                    {pp(f.gap_pp)}</td>
                  <td style={{ fontSize: 11 }}>{f.bucket || '—'}</td>
                  <td style={{ color: COLOR_ESTADO[f.FINAL_STATUS], fontSize: 11, fontWeight: 600 }}>
                    {f.FINAL_STATUS}</td>
                </tr>))}
            </tbody></table>
          </div>
          <div className="note">{mvm.nota}</div>
        </>}
      </div>

      <div className="panel">
        <h2>Favoritos &gt; 60 %</h2>
        {!fav ? <div className="empty">Cargando…</div> : <>
          <div style={{ fontSize: 12, color: 'var(--dim)', marginBottom: 8 }}>
            De {fav.n_predicciones_totales} predicciones ya calificadas,
            {' '}<b>{fav.n_por_encima_del_umbral}</b> superaron el 60 % de probabilidad calibrada.
          </div>
          <table><thead><tr>
            <th>Bucket</th><th>n</th><th>W</th><th>L</th><th>P</th>
            <th>Acierto</th><th>IC 95 %</th><th>Anunciada</th>
            <th>Con cuota</th><th>Cuota media</th><th>Unidades</th><th>ROI</th>
          </tr></thead><tbody>
            {fav.buckets.map(b => (
              <tr key={b.bucket}>
                <td><b>{b.bucket}</b></td>
                <td className="num">{b.n}</td>
                <td className="num" style={{ color: 'var(--ok)' }}>{b.wins}</td>
                <td className="num" style={{ color: 'var(--bad)' }}>{b.losses}</td>
                <td className="num">{b.pushes}</td>
                <td className="num">{pct(b.win_rate)}</td>
                <td className="mono" style={{ fontSize: 11 }}>
                  {b.win_rate_ic95 ? `${pct(b.win_rate_ic95[0])}–${pct(b.win_rate_ic95[1])}` : '—'}</td>
                <td className="num" style={{ color: 'var(--dim)' }}>{pct(b.prob_media_anunciada)}</td>
                <td className="num">{b.n_apostables}</td>
                <td className="num mono">{b.cuota_media ?? '—'}</td>
                <td className="num">{b.profit_unidades?.toFixed(2) ?? '—'}</td>
                <td className="num" style={{ color: b.roi > 0 ? 'var(--ok)' : b.roi < 0 ? 'var(--bad)' : undefined }}>
                  {b.roi == null ? '—' : pct(b.roi)}</td>
              </tr>))}
          </tbody></table>
          {fav.nota && <div className="note bad">{fav.nota}</div>}
          <div className="note">
            <b>Acierto y dinero son cosas distintas.</b> El acierto se puede leer con
            pocas apuestas; el ROI no. Un ROI calculado con menos de 100 apuestas
            liquidadas no distingue ventaja de suerte.
          </div>
        </>}
      </div>

      <div className="panel">
        <h2>Backtest de discrepancias</h2>
        {!dis ? <div className="empty">Cargando…</div> : <>
          <div style={{ fontSize: 12, color: 'var(--dim)', marginBottom: 8 }}>
            {dis.n_registros} predicciones calificadas con gap medible ·
            {' '}{dis.sin_cuota} sin cuota registrada (no entran en el ROI)
          </div>
          <table><thead><tr>
            <th>Tramo</th><th>n</th><th>W</th><th>L</th><th>Acierto</th>
            <th>Cuota media</th><th>Implícita</th><th>Modelo</th>
            <th>Unidades</th><th>ROI</th><th>IC 95 % del ROI</th><th>¿Muestra?</th>
          </tr></thead><tbody>
            {dis.buckets.map(b => (
              <tr key={b.bucket}>
                <td><b>{b.bucket}</b></td>
                <td className="num">{b.n}</td>
                <td className="num">{b.wins}</td><td className="num">{b.losses}</td>
                <td className="num">{pct(b.win_rate)}</td>
                <td className="num mono">{b.cuota_media ?? '—'}</td>
                <td className="num" style={{ color: 'var(--dim)' }}>{pct(b.implied_media)}</td>
                <td className="num" style={{ color: 'var(--dim)' }}>{pct(b.prob_modelo_media)}</td>
                <td className="num">{b.profit_unidades?.toFixed(2) ?? '—'}</td>
                <td className="num" style={{ color: b.roi > 0 ? 'var(--ok)' : b.roi < 0 ? 'var(--bad)' : undefined }}>
                  {b.roi == null ? '—' : pct(b.roi)}</td>
                <td className="mono" style={{ fontSize: 11 }}>
                  {b.roi_ic95 ? `${pct(b.roi_ic95[0])} … ${pct(b.roi_ic95[1])}` : '—'}</td>
                <td style={{ color: b.suficiente ? 'var(--ok)' : 'var(--bad)' }}>
                  {b.suficiente ? 'sí' : 'no'}</td>
              </tr>))}
          </tbody></table>
          <div className={'note ' + (dis.veredicto?.con_evidencia_de_valor?.length ? 'ok' : 'bad')}>
            <b>Veredicto:</b> {dis.veredicto?.conclusion}
          </div>
          {dis.modelo_vs_favorito && <div style={{ marginTop: 12 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 6px' }}>
              ¿Qué pasa cuando el modelo contradice al favorito del mercado?</h3>
            {dis.modelo_vs_favorito.n === 0
              ? <div className="empty" style={{ padding: 10 }}>{dis.modelo_vs_favorito.nota}</div>
              : <dl className="kv">
                <dt>Casos</dt><dd>{dis.modelo_vs_favorito.n}</dd>
                <dt>Ganó el modelo</dt><dd>{dis.modelo_vs_favorito.gano_el_modelo}</dd>
                <dt>Ganó el mercado</dt><dd>{dis.modelo_vs_favorito.gano_el_mercado}</dd>
                <dt>ROI siguiendo al modelo</dt>
                <dd>{pct(dis.modelo_vs_favorito.roi_siguiendo_al_modelo)}
                  {dis.modelo_vs_favorito.roi_ic95_modelo &&
                    ` (IC 95 %: ${pct(dis.modelo_vs_favorito.roi_ic95_modelo[0])} … ${pct(dis.modelo_vs_favorito.roi_ic95_modelo[1])})`}</dd>
              </dl>}
            {dis.modelo_vs_favorito.nota && dis.modelo_vs_favorito.n > 0 &&
              <div className="note bad">{dis.modelo_vs_favorito.nota}</div>}
          </div>}
        </>}
      </div>

      {cob && <div className="panel">
        <h2>¿Con qué se puede medir?</h2>
        <table><thead><tr>
          <th>Deporte</th><th>Predicciones</th><th>Calificadas</th>
          <th>Con mercado</th><th>Con cuota</th><th>Calificadas + cuota</th><th>Mercados</th>
        </tr></thead><tbody>
          {Object.entries(cob.deportes || {}).map(([s, v]) => (
            <tr key={s}>
              <td><span className={'pill ' + s.toLowerCase()}>{s}</span></td>
              <td className="num">{v.predicciones}</td>
              <td className="num">{v.calificadas}</td>
              <td className="num">{v.con_mercado}</td>
              <td className="num">{v.con_cuota}</td>
              <td className="num" style={{ color: v.calificadas_con_cuota ? 'var(--ok)' : 'var(--bad)' }}>
                <b>{v.calificadas_con_cuota}</b></td>
              <td style={{ fontSize: 11, color: 'var(--dim)' }}>{(v.mercados || []).join(', ') || '—'}</td>
            </tr>))}
        </tbody></table>
        <div className="note">
          La última columna es la que manda: para calcular ROI hace falta que la
          predicción tenga <b>resultado</b> y <b>cuota registrada al emitirla</b>.
          Sin las dos cosas no hay contabilidad posible.
        </div>
        {(cob.errores_de_consulta || []).length > 0 && <div className="note bad">
          Consultas que fallaron al leer las bases: {cob.errores_de_consulta.join(' · ')}
        </div>}
      </div>}
    </>
  )
}
