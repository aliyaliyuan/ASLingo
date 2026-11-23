import { useEffect, useState } from 'react'

// Every `intervalMs`, sends the latest hand landmarks (from useHandCamera)
// to the Flask /predict endpoint and returns the predicted sign.
// 'none' means no hand is in view; null means no answer yet.
export function usePrediction(landmarksRef, intervalMs = 1000) {
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let busy = false
    let cancelled = false

    const tick = async () => {
      if (busy) return
      const hand = landmarksRef.current
      if (!hand) {
        setPrediction('none')
        return
      }
      busy = true
      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ landmarks: hand.flatMap((p) => [p.x, p.y, p.z]) }),
        })
        if (!res.ok) throw new Error(`Server answered ${res.status}`)
        const data = await res.json()
        if (!cancelled) {
          setPrediction(data.letter)
          setError(null)
        }
      } catch (err) {
        console.error(err)
        if (!cancelled) setError('Cannot reach the prediction server. Is Flask running?')
      } finally {
        busy = false
      }
    }

    const id = setInterval(tick, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [landmarksRef, intervalMs])

  return { prediction, error }
}
