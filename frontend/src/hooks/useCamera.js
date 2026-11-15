import { useEffect, useRef, useState } from 'react'

// Starts the webcam and attaches it to the returned videoRef.
// The camera is released when the component unmounts.
export function useCamera() {
  const videoRef = useRef(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let stream
    let cancelled = false

    navigator.mediaDevices
      ?.getUserMedia({ video: true })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop())
          return
        }
        stream = s
        if (videoRef.current) videoRef.current.srcObject = s
      })
      .catch((err) => setError(err.message || 'Could not open the camera'))

    return () => {
      cancelled = true
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  return { videoRef, error }
}
