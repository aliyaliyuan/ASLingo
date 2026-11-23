import CameraView from './CameraView.jsx'
import { useHandCamera } from '../hooks/useHandCamera.js'

// Asks the player to sign one word, with the live camera and hand skeleton.
//
// The model can't recognise words yet, so for now the player confirms the
// sign themselves once their hand is in view. When the word model exists,
// call onSuccess from the recogniser here instead of from the button.
export default function SignPrompt({ item, onSuccess, onCancel }) {
  const camera = useHandCamera()

  return (
    <div className="sign-prompt">
      <p className="sign-target">
        <span className="sign-icon">{item.icon}</span> Sign: <strong>{item.word}</strong>
      </p>

      <CameraView camera={camera} />

      <p className="sign-hint">
        {camera.handVisible
          ? 'Hand detected. Sign the word, then confirm.'
          : 'Hold your hand up so the camera can see it.'}
      </p>

      <button disabled={!camera.handVisible} onClick={onSuccess}>I signed it</button>
      <button className="secondary" onClick={onCancel}>Not now</button>
    </div>
  )
}
