import { useEffect, useState } from 'react'
import { predictSign } from '../model/forest.js'

// Every `intervalMs`, runs the sign model on the latest hand landmarks
// (from useHandCamera) and returns the predicted sign. Runs entirely in the
// browser. 'none' means no hand is in view; null means no answer yet.
export function usePrediction(landmarksRef, intervalMs = 500) {
  const [prediction, setPrediction] = useState(null)

  useEffect(() => {
    const id = setInterval(() => {
      const hand = landmarksRef.current
      setPrediction(hand ? predictSign(hand).label : 'none')
    }, intervalMs)
    return () => clearInterval(id)
  }, [landmarksRef, intervalMs])

  return { prediction }
}
