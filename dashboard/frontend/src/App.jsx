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

const SIN_FILTROS = ['chat', 'calc']
const CON_FECHAS = ['calendar', 'top', 'alerts']

/* =========================================================================
   DESKTOP: dropdown "Más" con portal (funciona porque no hay backdrop-filter
   atrapando el menú en desktop)
   ========================================================================= */

function DesktopDropdown({ anchorRef, open, onClose, children }) {
  const [pos, setPos] = useState({ top: 0, right: 16 })
  const menuRef = useRef(null)

  useEffect(() => {
    if (open && anchorRef.current) {
      const r = anchorRef.current.getBoundingClientRect()
      setPos({ top: r.bottom + 8, right: Math.max(8, window.innerWidth - r.right) })
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const fuera = e => {
      if (menuRef.current && !menuRef.current.contains(e.target) &&
          anchorRef.current && !anchorRef.current.contains(e.target))
        onClose()
    }
    const t = setTimeout(() => document.addEventListener('pointerdown', fuera), 200)
    return () => { clearTimeout(t); document.removeEventListener('pointerdown', fuera) }
  }, [open])

  if (!open) return null
  return createPortal(
    <div ref={menuRef} className="more-menu" style={{
      position: 'fixed', top: pos.top, right: pos.right,
      left: 'auto', zIndex: 9999, maxWidth: 240, minWidth: 200,
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
      <DesktopDropdown anchorRef={btnRef} open={abierto} onClose={() => setAbierto(false)}>
        {MENU.map(([grupo, items], i) => (
          <div key={grupo}>
            {i > 0 && <div className="sep" />}
            <div className="lbl">{grupo}</div>
            {items.map(([k, l]) => (
              <button key={k} className={tab === k ? 'on' : ''}
                onClick={() => { setTab(k); setAbierto(false) }}>{l}</button>
            ))}
          </div>
        ))}
      </DesktopDropdown>
    </div>
  )
}

/* =========================================================================
   MÓVIL: bottom sheet — panel que sube desde abajo de la pantalla.
   Se renderiza directamente en App, NO dentro de BottomNav.
   No usa portales, no usa position calculations, no se rompe con
   backdrop-filter ni overflow. Es un div fijo que cubre toda la pantalla.
   ========================================================================= */

function MasSheet({ tab, setTab, onClose }) {
  return (
    <div className="mas-overlay" onClick={onClose}>
      <div className="mas-sheet" onClick={e => e.stopPropagation()}>
        <div className="mas-head">
          <span>Más opciones</span>
          <button onClick={onClose}>✕</button>
        </div>
        {MENU.map(([grupo, items], i) => (
          <div key={grupo}>
            {i > 0 && <div className="mas-sep" />}
            <div className="mas-grupo">{grupo}</div>
            {items.map(([k, l]) => (
              <button key={k} className={'mas-item' + (tab === k ? ' on' : '')}
                onClick={() => { setTab(k); onClose() }}>{l}</button>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

/* =========================================================================
   MÓVIL: barra inferior — sin dropdown, solo abre el sheet via callback
   ========================================================================= */

const BNAV_ICONS = {
  calendar: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
      <line x1="16" y1="2" x2="16" y2="6"/>
      <line x1="8" y1="2" x2="8" y2="6"/>
      <line x1="3" y1="10" x2="21" y2="10"/>
    </svg>
  ),
  top: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
    </svg>
  ),
  chat: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  ),
  value: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10"/>
      <line x1="12" y1="20" x2="12" y2="4"/>
      <line x1="6" y1="20" x2="6" y2="14"/>
    </svg>
  ),
  parlay: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="20" height="14" rx="2" ry="2"/>
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>
    </svg>
  ),
  more: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="1"/>
      <circle cx="12" cy="5" r="1"/>
      <circle cx="12" cy="19" r="1"/>
    </svg>
  ),
}

function BottomNav({ tab, setTab, onMas }) {
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
      <button className={'bnav-btn' + (dentro ? ' on' : '')} onClick={onMas}>
        <span className="ic">{BNAV_ICONS.more}</span>
        {dentro ? TITULOS[tab] : 'Más'}
      </button>
    </nav>
  )
}

/* =========================================================================
   TEMA
   ========================================================================= */

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

/* =========================================================================
   APP
   ========================================================================= */

export default function App() {
  const [tema, setTema] = useTema()
  const [tab, setTab] = useState('calendar')
  const [sport, setSport] = useState('all')
  const [range, setRange] = useState('today')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [open, setOpen] = useState(null)
  const [masAbierto, setMasAbierto] = useState(false)
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
      <BottomNav tab={tab} setTab={setTab} onMas={() => setMasAbierto(true)} />
      {masAbierto && <MasSheet tab={tab} setTab={setTab} onClose={() => setMasAbierto(false)} />}
    </>
  )
}
