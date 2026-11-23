import { Link } from 'react-router-dom'
import CameraView from '../components/CameraView.jsx'
import { useHandCamera } from '../hooks/useHandCamera.js'
import { usePrediction } from '../hooks/usePrediction.js'

export default function LiveRecognition() {
  const camera = useHandCamera()
  const { prediction } = usePrediction(camera.landmarksRef, 500)

  return (
    <main className="page centered">
      <div className="card">
        <h1>Live Recognition</h1>
        <p className="prediction">Letter: {prediction ?? '…'}</p>
        <CameraView camera={camera} />
        <Link className="back-link" to="/menu">← Menu</Link>
      </div>
    </main>
  )
}
