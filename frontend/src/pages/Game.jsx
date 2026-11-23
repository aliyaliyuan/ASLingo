import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import Modal from '../components/Modal.jsx'
import SignPrompt from '../components/SignPrompt.jsx'
import { makeShoppingList } from '../game/items.js'
import hedgehog from '../assets/hedgehog.png'
import fruitImg from '../assets/fruit.jpg'
import vegImg from '../assets/veg.png'
import dessertImg from '../assets/dessert.png'
import cashierImg from '../assets/cashier.png'

const CELL_SIZE = 60
const COLS = 8
const ROWS = 8
const BOARD_W = COLS * CELL_SIZE
const BOARD_H = ROWS * CELL_SIZE
const SPEED = 3
const AGENT_RADIUS = 18
const START = { x: BOARD_W / 2, y: BOARD_H / 2 }

// Store sections, in board pixels.
const ZONES = [
  { id: 'fruit', label: 'Fruit', img: fruitImg, x: 80, y: 2, w: 150, h: 50 },
  { id: 'vegetable', label: 'Vegetables', img: vegImg, x: 2, y: 160, w: 50, h: 150 },
  { id: 'dessert', label: 'Dessert', img: dessertImg, x: 300, y: 2, w: 150, h: 50 },
  { id: 'cashier', label: 'Cashier', img: cashierImg, x: 328, y: 250, w: 150, h: 50 },
]

const MOVES = {
  ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0],
  w: [0, -1], s: [0, 1], a: [-1, 0], d: [1, 0],
}

function touchingZone(x, y) {
  return ZONES.find((z) => {
    const dx = x - Math.max(z.x, Math.min(x, z.x + z.w))
    const dy = y - Math.max(z.y, Math.min(y, z.y + z.h))
    return Math.hypot(dx, dy) < AGENT_RADIUS
  })
}

export default function Game() {
  const agentRef = useRef(null)
  const keys = useRef({})
  const pos = useRef({ ...START })
  // The zone whose prompt is open right now (also mirrored in state for rendering).
  const openZoneRef = useRef(null)
  // The zone whose prompt was just closed. It can't re-open until the player
  // steps away from it, otherwise the prompt would pop straight back up.
  const dismissedZoneRef = useRef(null)
  const [openZone, setOpenZone] = useState(null)
  const [shoppingList, setShoppingList] = useState(makeShoppingList)
  const [cart, setCart] = useState([]) // ids of collected items

  const remaining = shoppingList.filter((item) => !cart.includes(item.id))
  // The item to sign for in the open section, if the list has one there.
  const wanted = openZone && remaining.find((item) => item.section === openZone.id)

  useEffect(() => {
    const down = (e) => {
      if (!(e.key in MOVES)) return
      keys.current[e.key] = true
      if (e.key.startsWith('Arrow')) e.preventDefault()
    }
    const up = (e) => {
      keys.current[e.key] = false
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)

    let frame
    const loop = () => {
      frame = requestAnimationFrame(loop)
      if (openZoneRef.current) return // paused while a prompt is open

      const p = pos.current
      let moving = false
      for (const [key, [dx, dy]] of Object.entries(MOVES)) {
        if (!keys.current[key]) continue
        p.x += dx * SPEED
        p.y += dy * SPEED
        moving = true
      }
      p.x = Math.max(AGENT_RADIUS, Math.min(BOARD_W - AGENT_RADIUS, p.x))
      p.y = Math.max(AGENT_RADIUS, Math.min(BOARD_H - AGENT_RADIUS, p.y))

      const agent = agentRef.current
      agent.style.left = `${p.x}px`
      agent.style.top = `${p.y}px`
      agent.classList.toggle('moving', moving)

      const zone = touchingZone(p.x, p.y)
      if (!zone) {
        dismissedZoneRef.current = null // walked away: every zone is live again
      } else if (zone.id !== dismissedZoneRef.current) {
        openZoneRef.current = zone
        keys.current = {}
        setOpenZone(zone)
      }
    }
    frame = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  const closePrompt = useCallback(() => {
    dismissedZoneRef.current = openZoneRef.current?.id ?? null
    openZoneRef.current = null
    keys.current = {}
    setOpenZone(null)
  }, [])

  const collect = (item) => {
    setCart((c) => [...c, item.id])
    closePrompt()
  }

  const newGame = () => {
    setShoppingList(makeShoppingList())
    setCart([])
    closePrompt()
  }

  return (
    <main className="page centered game-page">
      <div className="game-layout">
      <div className="game-wrapper" style={{ width: BOARD_W, height: BOARD_H }}>
        <div className="board" style={{ gridTemplateColumns: `repeat(${COLS}, ${CELL_SIZE}px)` }}>
          {Array.from({ length: ROWS * COLS }, (_, i) => {
            const pink = (Math.floor(i / COLS) + (i % COLS)) % 2 === 0
            return <div key={i} className={pink ? 'cell pink' : 'cell'} style={{ height: CELL_SIZE }} />
          })}
        </div>

        {ZONES.map((z) => (
          <div
            key={z.id}
            className="zone"
            title={z.label}
            style={{ left: z.x, top: z.y, width: z.w, height: z.h, backgroundImage: `url(${z.img})` }}
          />
        ))}

        <div
          ref={agentRef}
          className="agent"
          style={{
            left: START.x,
            top: START.y,
            width: AGENT_RADIUS * 2,
            height: AGENT_RADIUS * 2,
            backgroundImage: `url(${hedgehog})`,
          }}
        />
      </div>

        <aside className="shopping-list">
          <h2>Shopping list</h2>
          <ul>
            {shoppingList.map((item) => {
              const done = cart.includes(item.id)
              return (
                <li key={item.id} className={done ? 'done' : ''}>
                  <span className="check">{done ? '✓' : ''}</span>
                  {item.icon} {item.word}
                </li>
              )
            })}
          </ul>
          <p className="cart-count">
            In cart: {cart.length} / {shoppingList.length}
          </p>
          <p className="list-tip">
            {remaining.length ? 'Walk to a section to pick up an item.' : 'Everything collected. Go to the cashier!'}
          </p>
        </aside>
      </div>

      <p className="game-hint">
        Move with the arrow keys · <Link to="/menu">Menu</Link>
      </p>

      {openZone && openZone.id === 'cashier' && (
        <Modal title="Cashier" onClose={closePrompt}>
          {remaining.length ? (
            <>
              <p>You still need: {remaining.map((item) => `${item.icon} ${item.word}`).join(', ')}</p>
              <button autoFocus onClick={closePrompt}>Keep shopping</button>
            </>
          ) : (
            <>
              <p className="win">🎉 You got everything on your list!</p>
              <button autoFocus onClick={newGame}>Play again</button>
            </>
          )}
        </Modal>
      )}

      {openZone && openZone.id !== 'cashier' && (
        <Modal title={`${openZone.label} section`} onClose={closePrompt}>
          {wanted ? (
            <SignPrompt item={wanted} onSuccess={() => collect(wanted)} onCancel={closePrompt} />
          ) : (
            <>
              <img className="modal-image" src={openZone.img} alt="" />
              <p>Nothing left on your list from this section.</p>
              <button autoFocus onClick={closePrompt}>Close</button>
            </>
          )}
        </Modal>
      )}
    </main>
  )
}
