import { Route, Routes } from 'react-router-dom'
import Login from './pages/Login.jsx'
import Menu from './pages/Menu.jsx'
import Flashcards from './pages/Flashcards.jsx'
import LiveRecognition from './pages/LiveRecognition.jsx'
import Game from './pages/Game.jsx'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/menu" element={<Menu />} />
      <Route path="/flashcards" element={<Flashcards />} />
      <Route path="/live" element={<LiveRecognition />} />
      <Route path="/game" element={<Game />} />
    </Routes>
  )
}
