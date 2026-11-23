import { useState } from 'react'
import { Link } from 'react-router-dom'
import CameraView from '../components/CameraView.jsx'
import { useHandCamera } from '../hooks/useHandCamera.js'
import { usePrediction } from '../hooks/usePrediction.js'

const FLASHCARDS = ['A', 'L', 'X']

export default function Flashcards() {
  const [index, setIndex] = useState(0)
  const camera = useHandCamera()
  const { prediction } = usePrediction(camera.landmarksRef, 500)

  const target = FLASHCARDS[index]
  const sawHand = prediction && prediction.toLowerCase() !== 'none'
  const correct = prediction === target

  return (
    <main className="page centered">
      <div className="card">
        <h1>ASLingo</h1>
        <div className="flashcard">Sign: {target}</div>

        <CameraView camera={camera} />

        <p className="prediction">
          {sawHand ? `Current sign: ${prediction}` : 'Please sign the letter'}
        </p>

        {sawHand && (
          <div className={`feedback ${correct ? 'correct' : 'incorrect'}`}>
            {correct ? 'Correct!' : `Incorrect, that is ${prediction}, not ${target}`}
          </div>
        )}

        <button onClick={() => setIndex((index + 1) % FLASHCARDS.length)}>Next Card</button>
        <Link className="back-link" to="/menu">← Menu</Link>
      </div>
    </main>
  )
}
