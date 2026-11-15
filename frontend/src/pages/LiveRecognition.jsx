import { Link } from 'react-router-dom'
import CameraView from '../components/CameraView.jsx'
import { useCamera } from '../hooks/useCamera.js'
import { usePrediction } from '../hooks/usePrediction.js'

export default function LiveRecognition() {
  const { videoRef, error: cameraError } = useCamera()
  const { prediction, error } = usePrediction(videoRef, 1000)

  return (
    <main className="page centered">
      <div className="card">
        <h1>Live Recognition</h1>
        <p className="prediction">Letter: {prediction ?? '…'}</p>
        <CameraView videoRef={videoRef} error={cameraError} />
        {error && <p className="camera-error">{error}</p>}
        <Link className="back-link" to="/menu">← Menu</Link>
      </div>
    </main>
  )
}
