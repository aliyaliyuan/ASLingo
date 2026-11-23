import { useEffect, useRef, useState } from 'react'
import { DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision'
import wasmLoaderPath from '@mediapipe/tasks-vision/vision_wasm_internal.js?url'
import wasmBinaryPath from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url'

// The hand model is loaded once and shared by every page.
let landmarkerPromise
function loadLandmarker() {
  landmarkerPromise ??= HandLandmarker.createFromOptions(
    { wasmLoaderPath, wasmBinaryPath },
    {
      baseOptions: { modelAssetPath: `${import.meta.env.BASE_URL}hand_landmarker.task` },
      runningMode: 'VIDEO',
      numHands: 1,
    },
  ).catch((err) => {
    landmarkerPromise = undefined // allow a retry on the next page
    throw err
  })
  return landmarkerPromise
}

// Opens the webcam, tracks one hand in the browser and draws its skeleton.
//
// Attach `videoRef` to a <video> and `canvasRef` to a <canvas> laid over it
// (the CameraView component does this). `landmarksRef.current` always holds
// the latest hand as 21 {x, y, z} points, or null when no hand is in view.
export function useHandCamera() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const landmarksRef = useRef(null)
  const [handVisible, setHandVisible] = useState(false)
  const [status, setStatus] = useState('loading') // 'loading' | 'ready' | 'error'
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    let stream
    let frame

    async function start() {
      const [landmarker, s] = await Promise.all([
        loadLandmarker(),
        navigator.mediaDevices.getUserMedia({ video: true }),
      ])
      stream = s
      if (cancelled) return
      const video = videoRef.current
      video.srcObject = stream
      await video.play().catch(() => {})
      if (cancelled) return
      setStatus('ready')

      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      const draw = new DrawingUtils(ctx)
      let lastTime = -1
      let wasVisible = false

      const loop = () => {
        frame = requestAnimationFrame(loop)
        if (!video.videoWidth || video.currentTime === lastTime) return
        lastTime = video.currentTime

        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }

        const result = landmarker.detectForVideo(video, performance.now())
        const hand = result.landmarks[0] ?? null
        landmarksRef.current = hand

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        if (hand) {
          draw.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: '#ffffff', lineWidth: 4 })
          draw.drawLandmarks(hand, { color: '#0097b2', fillColor: '#fdc1d6', lineWidth: 2, radius: 5 })
        }

        if (!!hand !== wasVisible) {
          wasVisible = !!hand
          setHandVisible(wasVisible)
        }
      }
      loop()
    }

    start().catch((err) => {
      if (cancelled) return
      console.error(err)
      setError(err.message || 'Could not start the camera')
      setStatus('error')
    })

    return () => {
      cancelled = true
      cancelAnimationFrame(frame)
      stream?.getTracks().forEach((t) => t.stop())
      landmarksRef.current = null
    }
  }, [])

  return { videoRef, canvasRef, landmarksRef, handVisible, status, error }
}
