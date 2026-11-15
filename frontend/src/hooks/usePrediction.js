import { useEffect, useState } from 'react'

// Sends a webcam frame to the Flask /predict endpoint every `intervalMs`
// and returns the latest predicted sign ('none' when no hand is found,
// null before the first answer).
export function usePrediction(videoRef, intervalMs = 1000) {
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    const canvas = document.createElement('canvas')
    let busy = false
    let cancelled = false

    const tick = () => {
      const video = videoRef.current
      if (busy || !video || !video.videoWidth) return
      busy = true

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      canvas.getContext('2d').drawImage(video, 0, 0)

      canvas.toBlob(async (blob) => {
        try {
          const body = new FormData()
          body.append('image', blob)
          const res = await fetch('/predict', { method: 'POST', body })
          if (!res.ok) throw new Error(`Server answered ${res.status}`)
          const data = await res.json()
          if (!cancelled) {
            setPrediction(data.letter)
            setError(null)
          }
        } catch (err) {
          if (!cancelled) setError('Cannot reach the prediction server. Is Flask running?')
          console.error(err)
        } finally {
          busy = false
        }
      }, 'image/jpeg')
    }

    const id = setInterval(tick, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [videoRef, intervalMs])

  return { prediction, error }
}
