import React, { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import Calendar from './pages/Calendar'
import Parlays from './pages/Parlays'
import TopPicks from './pages/TopPicks'
import SourcesPage from './pages/Sources'
import AlertsPage from './pages/Alerts'
import MarketsPage from './pages/Markets'
import PerformanceCenter from './pages/PerformanceCenter'
import ModelsTable from './pages/ModelsTable'
import DataHealth from './pages/DataHealth'
import History from './pages/History'
import ValueCenter from './pages/ValueCenter'
import ParlayCalculator from './pages/ParlayCalculator'
import Chat from './pages/Chat'
import GameModal from './components/GameModal'

// Trece pestañas en una sola fila obligaban a leerlas todas para encontrar una.
// Ahora hay CINCO a la vista —las que se usan a diario— y el resto vive en un
// menú agrupado por para qué sirve cada cosa. No se quitó ninguna pantalla.
const PRINCIPALES = [
  ['calendar', 'Partidos'],
  ['top', 'Top picks'],
  ['chat', 'Caballo Chat'],
  ['value', 'Vs mercado'],
  ['parlay', 'Combinadas'],
]
const MENU = [
  ['Herramientas', [['calc', 'Calculadora de momios'], ['alerts', 'Alertas'],
                    ['markets', 'Mercados disponibles']]],
  ['Resultados', [['hist', 'Histórico'], ['pc', 'Rendimiento']]],
  ['Sistema', [['models', 'Modelos'], ['health', 'Salud de los datos'],
               ['src', 'Estado y fuentes']]],
]
const TITULOS = Object.fromEntries(
  [...PRINCIPALES, ...MENU.flatMap(([, l]) => l)])

const DEPORTES = [['all', 'Todos', ''], ['NFL', 'NFL', 'nfl'], ['MLB', 'MLB', 'mlb'],
                  ['NBA', 'NBA', 'nba'], ['TENIS', 'Tenis', 'tenis'], ['SOCCER', 'Fútbol', 'soccer']]
const RANGES = [['today', 'Hoy'], ['tomorrow', 'Mañana'], ['7d', '7 días'],
                ['30d', '30 días'], ['custom', 'Fecha']]

// Pantallas que traen sus propios filtros: el selector de deporte solo estorba.
const SIN_FILTROS = ['chat', 'calc']
const CON_FECHAS = ['calendar', 'top', 'alerts']

// Contenido del dropdown compartido entre Menu y BottomNav
function MenuItems({ tab, setTab, onClose }) {
  return MENU.map(([grupo, items], i) => (
    <div key={grupo}>
      {i > 0 && <div className="sep" />}
      <div className="lbl">{grupo}</div>
      {items.map(([k, l]) => (
        <button key={k} className={tab === k ? 'on' : ''}
          onClick={() => { setTab(k); onClose() }}>{l}</button>
      ))}
    </div>
  ))
}

// Dropdown via Portal — se renderiza en document.body para escapar cualquier
// backdrop-filter o overflow en los padres. backdrop-filter crea un nuevo
// containing block que atrapa position:fixed, rompiendo el dropdown en móvil.
function DropdownPortal({ anchorRef, open, onClose, children, fromBottom }) {
  const [pos, setPos] = useState({ top: 0, right: 16, bottom: 'auto' })
  const menuRef = useRef(null)

  useEffect(() => {
    if (open && anchorRef.current) {
      const r = anchorRef.current.getBoundingClientRect()
      if (fromBottom) {
        setPos({ bottom: window.innerHeight - r.top + 8, top: 'auto', right: Math.max(8, window.innerWidth - r.right) })
      } else {
        setPos({ top: r.bottom + 8, bottom: 'auto', right: Math.max(8, window.innerWidth - r.right) })
      }
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const fuera = e => {
      if (menuRef.current && !menuRef.current.contains(e.target) &&
          anchorRef.current && !anchorRef.current.contains(e.target))
        onClose()
    }
    const t = setTimeout(() => {
      document.addEventListener('pointerdown', fuera)
    }, 200)
    return () => {
      clearTimeout(t)
      document.removeEventListener('pointerdown', fuera)
    }
  }, [open])

  if (!open) return null
  return createPortal(
    <div ref={menuRef} className="more-menu" style={{
      position: 'fixed',
      top: pos.top,
      bottom: pos.bottom,
      right: pos.right,
      left: 'auto',
      zIndex: 9999,
      maxWidth: 240,
      minWidth: 200,
    }}>
      {children}
    </div>,
    document.body
  )
}

function Menu({ tab, setTab }) {
  const [abierto, setAbierto] = useState(false)
  const btnRef = useRef(null)
  const dentro = MENU.some(([, l]) => l.some(([k]) => k === tab))
  return (
    <div className="more">
      <button ref={btnRef} className={dentro ? 'on' : ''} onClick={() => setAbierto(a => !a)}>
        {dentro ? TITULOS[tab] : 'Más'} ▾
      </button>
      <DropdownPortal anchorRef={btnRef} open={abierto} onClose={() => setAbierto(false)}>
        <MenuItems tab={tab} setTab={setTab} onClose={() => setAbierto(false)} />
      </DropdownPortal>
    </div>
  )
}

// Barra de navegación inferior para móvil. Solo muestra los 5 tabs principales
// + un botón "Más" que abre el mismo menú desplegable.
const BNAV_ICONS = {
  calendar: '📅', top: '⭐', chat: '💬', value: '📊', parlay: '🎟️'
}
function BottomNav({ tab, setTab }) {
  const [abierto, setAbierto] = useState(false)
  const btnRef = useRef(null)
  const dentro = MENU.some(([, l]) => l.some(([k]) => k === tab))
  return (
    <nav className="bottom-nav">
      {PRINCIPALES.map(([k, l]) => (
        <button key={k} className={'bnav-btn' + (tab === k ? ' on' : '')}
          onClick={() => setTab(k)}>
          <span className="ic">{BNAV_ICONS[k]}</span>
          {l}
        </button>
      ))}
      <button ref={btnRef} className={'bnav-btn' + (dentro ? ' on' : '')}
        style={{ flex: 1 }}
        onClick={() => setAbierto(a => !a)}>
        <span className="ic">☰</span>
        {dentro ? TITULOS[tab] : 'Más'}
      </button>
      <DropdownPortal anchorRef={btnRef} open={abierto} onClose={() => setAbierto(false)} fromBottom>
        <MenuItems tab={tab} setTab={setTab} onClose={() => setAbierto(false)} />
      </DropdownPortal>
    </nav>
  )
}

// El tema se guarda en el navegador: si eliges claro, sigue claro al volver.
// Se aplica sobre <html> y no sobre un contenedor de React para que el fondo
// de la pagina cambie tambien fuera del area de la app (rebote del scroll).
function useTema() {
  const [tema, setTema] = useState(() => {
    try { return localStorage.getItem('spc-tema') || 'oscuro' } catch { return 'oscuro' }
  })
  useEffect(() => {
    document.documentElement.dataset.tema = tema
    try { localStorage.setItem('spc-tema', tema) } catch { /* modo privado */ }
  }, [tema])
  return [tema, setTema]
}

export default function App() {
  const [tema, setTema] = useTema()
  const [tab, setTab] = useState('calendar')
  const [sport, setSport] = useState('all')
  // Se entra viendo HOY, no la semana. Con 7 días por defecto la primera
  // pantalla eran 200 partidos: nadie empieza el día queriendo revisar 200
  // partidos, y la semana sigue a un clic.
  const [range, setRange] = useState('today')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [open, setOpen] = useState(null)
  const s = range === 'custom' ? start : '', e = range === 'custom' ? end : ''

  return (
    <>
      <header className="top">
        <div className="topin">
          <h1 className="brand" onClick={() => setTab('calendar')}>
            <span className="mark">🐴</span>
            <span>
              <b>Sports <span>Prediction</span> Center</b>
            </span>
          </h1>
          <nav className="tabs">
            {PRINCIPALES.map(([k, l]) =>
              <button key={k} className={tab === k ? 'on' : ''}
                onClick={() => setTab(k)}>{l}</button>)}
            <Menu tab={tab} setTab={setTab} />
            <button className="tema" title={tema === 'oscuro' ? 'Tema claro' : 'Tema oscuro'}
              onClick={() => setTema(t => t === 'oscuro' ? 'claro' : 'oscuro')}>
              {tema === 'oscuro' ? '☀' : '☾'}
            </button>
          </nav>
        </div>
      </header>

      <div className="wrap">
        {!SIN_FILTROS.includes(tab) && (
          <div className="controls">
            <div className="seg">
              {DEPORTES.map(([k, l, c]) =>
                <button key={k} className={(sport === k ? 'on ' : '') + c}
                  onClick={() => setSport(k)}>{l}</button>)}
            </div>
            {CON_FECHAS.includes(tab) && (
              <div className="seg">
                {RANGES.map(([k, l]) =>
                  <button key={k} className={range === k ? 'on' : ''}
                    onClick={() => setRange(k)}>{l}</button>)}
              </div>)}
            {range === 'custom' && CON_FECHAS.includes(tab) && <>
              <input type="date" value={start} onChange={ev => setStart(ev.target.value)} />
              <input type="date" value={end} onChange={ev => setEnd(ev.target.value)} />
            </>}
          </div>)}

        {tab === 'calendar' && <Calendar sport={sport} range={range} start={s} end={e} onOpen={setOpen} />}
        {tab === 'top' && <TopPicks sport={sport} range={range} start={s} end={e} onOpen={setOpen} />}
        {tab === 'chat' && <Chat />}
        {tab === 'value' && <ValueCenter />}
        {tab === 'calc' && <ParlayCalculator />}
        {tab === 'parlay' && <Parlays />}
        {tab === 'alerts' && <AlertsPage sport={sport} range={range} start={s} end={e} />}
        {tab === 'markets' && <MarketsPage />}
        {tab === 'hist' && <History sport={sport} onOpenGame={gid => setOpen({
          sport: 'MLB', game_id: gid, game_date: '', start_utc: null, venue: null,
          market: '', selection: '', ensemble_probability: null, _detalle: true })} />}
        {tab === 'pc' && <PerformanceCenter sport={sport} />}
        {tab === 'models' && <ModelsTable />}
        {tab === 'health' && <DataHealth />}
        {tab === 'src' && <SourcesPage />}
      </div>
      <GameModal c={open} onClose={() => setOpen(null)} />
      <BottomNav tab={tab} setTab={setTab} />
    </>
  )
}
