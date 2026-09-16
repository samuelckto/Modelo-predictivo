import React, { useEffect, useMemo, useRef, useState } from 'react'

// CABALLO CHAT
// ---------------------------------------------------------------------------
// Rediseño. Tres cosas cambian respecto a la versión anterior:
//
// 1. Ocupa la pantalla como una app de mensajería (alto fijo, hilo con scroll
//    propio, composer abajo). Antes crecía hacia abajo y el campo de escribir
//    se iba fuera de la vista en cuanto había tres respuestas.
// 2. El globo del asistente enseña la RESPUESTA. Modelo, calibración, estado y
//    evidencia fuera de muestra siguen ahí, pero dentro de un desplegable: la
//    procedencia importa, no tiene por qué gritar más que el dato.
// 3. El textarea crece con el texto y Enter envía, Shift+Enter hace salto de
//    línea, como cualquier chat.
//
// Lo que NO cambia: el chat no calcula nada. Recibe cifras ya calculadas por
// los modelos y las presenta. Si no hay modelo validado, lo dice.

const EJEMPLOS = [
  '¿Qué partidos hay hoy?',
  '¿Qué tienes de este partido?',
  '¿Cuál tiene más probabilidad de ganarse hoy?',
  '¿Cómo va el sistema?',
  '¿Cuántas tarjetas se esperan?',
  '¿Qué ligas cubres?',
]

// El color del estado es SEMÁNTICO, no decorativo: dice si te puedes apoyar en
// la respuesta. VALIDATED VALUE en verde es el único "sí"; hoy no lo alcanza
// ningún mercado, y eso también es información.
const ESTADO = {
  'VALIDATED VALUE': 's-ok',
  PROJECTION: 's-warn', projection: 's-warn', UNVALIDATED: 's-warn',
  BLOCKED: 's-bad', 'NO PICK': 's-bad', no_pick: 's-bad', 'INSUFFICIENT DATA': 's-bad',
  'NO MARKET': 's-dim', ALIGNED: 's-dim', 'NO EXPLANATION': 's-dim',
}
const ETIQUETA = {
  PROJECTION: 'proyección', projection: 'proyección', BLOCKED: 'sin modelo',
  'NO MARKET': 'fuera de alcance', 'NO PICK': 'sin pick',
  'INSUFFICIENT DATA': 'sin datos', 'VALIDATED VALUE': 'valor validado',
  UNVALIDATED: 'sin validar', ALIGNED: 'de acuerdo con el mercado',
  'NO EXPLANATION': 'sin explicación',
}
const PILL = { MLB: 'mlb', NFL: 'nfl', NBA: 'nba', TENIS: 'tenis', SOCCER: 'soccer' }

const hora = s => {
  if (!s) return ''
  const d = new Date(String(s).replace(' ', 'T'))
  return isNaN(d) ? String(s).slice(5, 16)
    : d.toLocaleString('es-MX', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
}

// ---------------------------------------------------------------------------

function Buscador({ valor, onChange }) {
  const [abierto, setAbierto] = useState(false)
  const [q, setQ] = useState('')
  const [juegos, setJuegos] = useState(null)
  const caja = useRef(null)

  useEffect(() => {
    fetch('/api/chat/games?dias=10').then(r => r.json())
      .then(d => setJuegos(d.partidos || [])).catch(() => setJuegos([]))
  }, [])
  useEffect(() => {
    const fuera = e => { if (caja.current && !caja.current.contains(e.target)) setAbierto(false) }
    document.addEventListener('mousedown', fuera)
    return () => document.removeEventListener('mousedown', fuera)
  }, [])

  const filtrados = useMemo(() => {
    if (!juegos) return []
    const n = q.trim().toLowerCase()
    const base = n ? juegos.filter(j =>
      (j.nombre + ' ' + (j.liga || '') + ' ' + j.sport).toLowerCase().includes(n)) : juegos
    return base.slice(0, 60)
  }, [juegos, q])

  return (
    <div className="finder" ref={caja}>
      <button className="tab finder-btn" onClick={() => setAbierto(a => !a)}>
        {valor
          ? <><span className={'pill ' + (PILL[valor.sport] || '')}>{valor.sport}</span>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{valor.nombre}</span></>
          : <span style={{ color: 'var(--dim-2)' }}>Elegir partido…</span>}
      </button>
      {valor && <button className="tab ghost sm" onClick={() => onChange(null)}
        style={{ position: 'absolute', right: 4, top: 4 }} title="Quitar partido">✕</button>}

      {abierto && (
        <div className="finder-pop">
          <input autoFocus value={q} onChange={e => setQ(e.target.value)}
            placeholder="Buscar equipo, liga o deporte…"
            style={{ width: '100%', marginBottom: 6 }} />
          {juegos === null && <div className="empty" style={{ padding: 16 }}>Cargando…</div>}
          {juegos && filtrados.length === 0 &&
            <div className="empty" style={{ padding: 16 }}>Ningún partido coincide.</div>}
          {filtrados.map(j => (
            <div key={j.sport + j.game_id} className="finder-item"
              onClick={() => { onChange(j); setAbierto(false); setQ('') }}>
              <span className={'pill ' + (PILL[j.sport] || '')}>{j.sport}</span> {j.nombre}
              <div className="m">
                {j.liga}{j.liga ? ' · ' : ''}{hora(j.start_utc)}
                {j.mercados?.length ? ` · ${j.mercados.length} mercados` : ''}
              </div>
            </div>))}
        </div>)}
    </div>
  )
}

// ---------------------------------------------------------------------------

// La procedencia de la respuesta. Colapsada: sigue siendo auditable de un clic,
// pero deja de competir visualmente con lo que el usuario preguntó.
function Procedencia({ d }) {
  const ev = d.evidencia || {}
  const hay = d.model_version || d.status || ev.holdout_log_loss != null || d.limitaciones?.length
  if (!hay) return null
  return (
    <details className="tech">
      <summary>Cómo se obtuvo este número</summary>
      <div className="tech-body">
        {d.model_version && <div className="tech-row">
          <span>Modelo <b>{d.model_version}</b></span>
          {d.calibracion && <span>Calibración <b>{d.calibracion}</b></span>}
          {d.confianza && <span>Confianza <b>{d.confianza}</b></span>}
        </div>}
        {ev.holdout_log_loss != null && <div className="tech-row">
          <span>Fuera de muestra · log loss <b>{ev.holdout_log_loss}</b> frente
            a <b>{ev.baseline}</b> del baseline</span>
          {ev.accuracy != null && <span>Acierto <b>{(ev.accuracy * 100).toFixed(1)}%</b> frente
            a <b>{(ev.baseline_accuracy * 100).toFixed(1)}%</b></span>}
        </div>}
        {d.limitaciones?.length > 0 && <div className="tech-row" style={{ color: 'var(--warn)' }}>
          {d.limitaciones.join(' · ')}</div>}
        {d.factores && <pre style={{
          fontSize: 11, color: 'var(--dim)', overflowX: 'auto', margin: 0,
        }}>{JSON.stringify(d.factores, null, 1)}</pre>}
      </div>
    </details>
  )
}

function Turno({ m, onElegirPartido, onPreguntar }) {
  if (m.yo) return (
    <div className="turn me"><div className="bub me">{m.texto}</div></div>
  )

  const d = m.d || {}
  const cls = ESTADO[d.status] || 's-dim'
  return (
    <div className="turn">
      <div className="av sm">🐴</div>
      <div className="bubwrap">
        {(d.partido || d.status) && (
          <div className="bub-ctx">
            {d.partido && <>
              <span className={'pill ' + (PILL[d.partido.sport] || '')}>{d.partido.sport}</span>
              <span>{d.partido.nombre}</span>
              <span style={{ opacity: .7 }}>· {hora(d.partido.start_utc)}</span>
            </>}
            {d.status && <span className={'state ' + cls}>
              {ETIQUETA[d.status] || String(d.status).toLowerCase()}</span>}
          </div>)}

        <div className="bub bot">
          <pre>{d.respuesta}</pre>

          {d.motivo && d.tipo !== 'catalogo' &&
            <div className="sub" style={{ marginTop: 8 }}>{d.motivo}</div>}

          {/* Falta el partido: se elige aquí mismo y la pregunta se repite. */}
          {d.partidos?.length > 0 && (
            <div className="sugs">
              {d.partidos.slice(0, 8).map(j => (
                <button key={j.sport + j.game_id} className="sug"
                  onClick={() => onElegirPartido(j, m.pregunta)}>
                  {j.nombre} <span style={{ opacity: .6 }}>· {hora(j.start_utc)}</span>
                </button>))}
            </div>)}

          {/* El catálogo se vuelve botones: leerlo y usarlo es el mismo gesto. */}
          {(d.primarios || d.secundarios) && (
            <div style={{ marginTop: 10 }}>
              {[['Aparecen en Top Picks', d.primarios],
                ['Solo aquí, en el chat', d.secundarios]]
                .filter(([, l]) => l?.length).map(([titulo, lista]) => (
                  <div key={titulo} style={{ marginBottom: 10 }}>
                    <div className="mkt-label">{titulo}</div>
                    <div className="sugs" style={{ marginTop: 0 }}>
                      {lista.map(x => (
                        <button key={x.market_id} className="sug"
                          onClick={() => onPreguntar(x.pregunta)}
                          title={`${x.market_id} · ${x.estado}`}>
                          <span className={'pill ' + (PILL[x.sport] || '')}>{x.sport}</span>
                          {' '}{x.pregunta}
                        </button>))}
                    </div>
                  </div>))}
            </div>)}

          {d.sugerencia_llm && <div className="note info" style={{ marginTop: 10 }}>
            {d.sugerencia_llm}</div>}
          {d.aviso && <div className="note" style={{ marginTop: 10 }}>{d.aviso}</div>}
          {d.advertencia && <div className="note bad" style={{ marginTop: 8 }}>{d.advertencia}</div>}

          <Procedencia d={d} />
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------

// Con un modelo local una pregunta libre puede tardar dos minutos. Sin un
// contador uno cree que se colgó y recarga a los 20 segundos.
function Pensando({ local }) {
  const [seg, setSeg] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setSeg(s => s + 1), 1000)
    return () => clearInterval(t)
  }, [])
  return (
    <div className="turn">
      <div className="av sm">🐴</div>
      <div>
        <div className="typing"><i /><i /><i /></div>
        {seg >= 5 && <div className="sub" style={{ marginTop: 6 }}>
          {seg}s{local && seg >= 14 && ' · tu modelo corre en esta PC y las preguntas '
            + 'libres le cuestan'}{local && seg >= 45 && '. Puede llegar a un par de '
            + 'minutos: no se colgó'}
        </div>}
      </div>
    </div>
  )
}

function ComoConectar({ opciones }) {
  return (
    <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--line)' }}>
      <div style={{ fontSize: 13, marginBottom: 10, lineHeight: 1.6 }}>
        <b>Conectar un motor de lenguaje.</b> Elige uno, añade esas líneas
        a <code>C:\Sports-Prediction-Center\.env</code> y reinicia el servidor.
        Los números seguirán saliendo de tus modelos: el LLM solo los redacta.
      </div>
      <div style={{ display: 'grid', gap: 10,
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
        {opciones.map(o => (
          <div key={o.nombre} className="sp">
            <h4>{o.nombre} <span className="tag" style={{
              background: o.coste === 'gratis' ? '#22C55E1A' : '#FFFFFF0D',
              color: o.coste === 'gratis' ? 'var(--ok)' : 'var(--dim)',
            }}>{o.coste}{o.clave ? '' : ' · sin clave'}</span></h4>
            <ol style={{ margin: '8px 0 0', paddingLeft: 16, fontSize: 11.5,
              color: 'var(--dim)', lineHeight: 1.7 }}>
              {o.pasos.map((p, i) => <li key={i} style={{
                listStyle: p.startsWith('    ') ? 'none' : 'decimal',
                marginLeft: p.startsWith('    ') ? -16 : 0,
                fontFamily: p.startsWith('    ') ? 'ui-monospace, monospace' : 'inherit',
                color: p.startsWith('    ') ? 'var(--acc)' : 'inherit',
              }}>{p.trim()}</li>)}
            </ol>
            {o.nota && <div className="sub" style={{ marginTop: 7 }}>{o.nota}</div>}
          </div>))}
      </div>
      <div className="note info" style={{ marginTop: 10 }}>
        Sin esto el chat funciona igual: calendario, mercados, cómo va el sistema y
        repreguntas. La IA solo sirve para entender preguntas que no anticipé.
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------

export default function Chat() {
  const [msgs, setMsgs] = useState([])
  const [texto, setTexto] = useState('')
  const [partido, setPartido] = useState(null)
  const [esperando, setEsperando] = useState(false)
  const [estado, setEstado] = useState(null)
  const [verOpciones, setVerOpciones] = useState(false)
  // Memoria de la conversación: lo último que contestó el chat. Viaja de vuelta
  // en cada pregunta para poder entender una repregunta («entonces solo tiene
  // el 29%?») sin que el servidor guarde estado.
  const [ctx, setCtx] = useState(null)
  const fin = useRef(null)
  const ta = useRef(null)

  useEffect(() => { fin.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) },
    [msgs, esperando])
  useEffect(() => {
    fetch('/api/chat/estado').then(r => r.json()).then(setEstado).catch(() => { })
  }, [])

  // El textarea crece con el contenido hasta un tope. Un input de una línea
  // esconde la pregunta en cuanto pasa de diez palabras.
  const ajustar = el => {
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 150) + 'px'
  }

  const preguntar = (t, p) => {
    const q = (t ?? texto).trim()
    if (!q || esperando) return
    const juego = p !== undefined ? p : partido
    setMsgs(m => [...m, { yo: true, texto: q, pregunta: q }])
    setTexto(''); setEsperando(true)
    if (ta.current) { ta.current.style.height = 'auto' }

    const params = new URLSearchParams({ texto: q })
    if (juego) { params.set('game_id', juego.game_id); params.set('sport', juego.sport) }
    if (ctx) params.set('contexto', JSON.stringify(ctx))
    const hist = msgs.slice(-6).map(m => m.yo
      ? { yo: true, texto: m.texto }
      : { yo: false, respuesta: (m.d?.respuesta || '').slice(0, 1200) })
    if (hist.length) params.set('historial', JSON.stringify(hist))

    fetch('/api/chat?' + params).then(r => r.json())
      .then(d => {
        setMsgs(m => [...m, { yo: false, d, pregunta: q }])
        // Solo se sustituye la memoria si la respuesta trae contexto propio;
        // una repregunta conserva el de la respuesta que la originó.
        if (d.contexto) setCtx(d.contexto)
        setEsperando(false)
      })
      .catch(e => {
        setMsgs(m => [...m, { yo: false, pregunta: q,
          d: { respuesta: 'No pude contactar con el servidor: ' + e } }])
        setEsperando(false)
      })
  }

  const elegirYRepetir = (j, preguntaOriginal) => { setPartido(j); preguntar(preguntaOriginal, j) }
  const ia = estado?.llm

  return (
    <div className="chat">
      <div className="chat-head">
        <div className="av lg">🐴</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="chat-title">
            Caballo Chat
            {/* Con IA conectada no se anuncia nada: que el motor sea local o de
                pago es un detalle de instalación, no algo que el usuario tenga
                que ver en cada mensaje. Solo se avisa cuando FALTA, porque
                entonces sí cambia lo que el chat puede contestar. */}
            {ia && !ia.ok && (
              <span className="dot off" onClick={() => setVerOpciones(v => !v)}
                title="Cómo conectar una IA">conectar IA</span>)}
          </div>
          <div className="chat-sub">
            Pregunto a tus modelos, no al casino. Si no hay modelo validado, te lo digo.
          </div>
        </div>
        {msgs.length > 0 && <button className="tab ghost sm"
          onClick={() => { setMsgs([]); setCtx(null) }}>Limpiar</button>}
      </div>

      {verOpciones && ia?.opciones && <ComoConectar opciones={ia.opciones} />}

      <div className="chat-ctx">
        <Buscador valor={partido} onChange={setPartido} />
        {partido
          ? <button className="tab on sm"
            onClick={() => preguntar('¿Qué tienes de este partido?')}>Resumen completo</button>
          : <span className="sub" style={{ margin: 0 }}>
            opcional · también entiendo «¿tarjetas del Barcelona?»</span>}
      </div>

      <div className="thread">
        {msgs.length === 0 && (
          <div className="welcome">
            <div className="av lg" style={{ width: 52, height: 52, fontSize: 27 }}>🐴</div>
            <h3>¿Qué quieres saber?</h3>
            <p>
              Elige un partido arriba o nombra al equipo en la pregunta. Cada respuesta
              trae debajo el modelo que la produjo y cómo le fue fuera de muestra.
            </p>
            <div className="sugs">
              {EJEMPLOS.map(e =>
                <button key={e} className="sug" onClick={() => preguntar(e)}>{e}</button>)}
            </div>
          </div>)}

        {msgs.map((m, i) =>
          <Turno key={i} m={m} onElegirPartido={elegirYRepetir} onPreguntar={preguntar} />)}
        {esperando && <Pensando local={ia?.proveedor === 'local'} />}
        <div ref={fin} />
      </div>

      <div className="composer">
        <textarea ref={ta} value={texto} rows={1}
          onChange={e => { setTexto(e.target.value); ajustar(e.target) }}
          onKeyDown={e => {
            // Enter envía, Shift+Enter salta de línea: lo que hace cualquier chat.
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); preguntar() }
          }}
          placeholder={partido ? `Pregunta sobre ${partido.nombre}…`
            : 'Pregunta lo que quieras…'} />
        <button className="send" onClick={() => preguntar()}
          disabled={esperando || !texto.trim()} title="Enviar">↑</button>
      </div>
    </div>
  )
}
