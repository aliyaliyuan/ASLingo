// Mirrored webcam preview. Pass the videoRef from useCamera().
export default function CameraView({ videoRef, error }) {
  if (error) return <p className="camera-error">Camera unavailable: {error}</p>
  return <video ref={videoRef} className="camera" autoPlay playsInline muted />
}
