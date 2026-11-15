import { Link } from 'react-router-dom'
import logo from '../assets/logo_fdc1d6.jpg'

export default function Menu() {
  return (
    <main className="page centered">
      <img className="logo" src={logo} alt="ASLingo" />
      <h2>Choose a Mode:</h2>
      <Link className="button" to="/flashcards">Flashcards</Link>
      <Link className="button" to="/live">Live Recognition</Link>
      <Link className="button" to="/game">Game</Link>
    </main>
  )
}
