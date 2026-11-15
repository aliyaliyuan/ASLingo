import { useState } from 'react'
import { Link } from 'react-router-dom'
import CameraView from '../components/CameraView.jsx'
import { useCamera } from '../hooks/useCamera.js'
import { usePrediction } from '../hooks/usePrediction.js'

const FLASHCARDS = ['A', 'L', 'X']

export default function Flashcards() {
  const [index, setIndex] = useState(0)
  const { videoRef, error: cameraError } = useCamera()
  const { prediction, error } = usePrediction(videoRef, 1500)

  const target = FLASHCARDS[index]
  const sawHand = prediction && prediction.toLowerCase() !== 'none'
  const correct = prediction === target

  return (
    <main className="page centered">
      <div className="card">
        <h1>ASLingo</h1>
        <div className="flashcard">Sign: {target}</div>

        <CameraView videoRef={videoRef} error={cameraError} />

        <p className="prediction">
          {sawHand ? `Current sign: ${prediction}` : 'Please sign the letter'}
        </p>

        {sawHand && (
          <div className={`feedback ${correct ? 'correct' : 'incorrect'}`}>
            {correct ? 'Correct!' : `Incorrect, that is ${prediction}, not ${target}`}
          </div>
        )}
        {error && <p className="camera-error">{error}</p>}

        <button onClick={() => setIndex((index + 1) % FLASHCARDS.length)}>Next Card</button>
        <Link className="back-link" to="/menu">← Menu</Link>
      </div>
    </main>
  )
}
