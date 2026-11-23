// Mirrored webcam preview with the hand skeleton drawn on top.
// Pass it the object returned by useHandCamera().
export default function CameraView({ camera }) {
  const { videoRef, canvasRef, status, error } = camera

  return (
    <div className="camera">
      <video ref={videoRef} autoPlay playsInline muted />
      <canvas ref={canvasRef} />
      {status === 'loading' && <p className="camera-status">Starting camera…</p>}
      {status === 'error' && <p className="camera-status error">Camera unavailable: {error}</p>}
    </div>
  )
}
