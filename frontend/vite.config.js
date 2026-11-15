import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, API calls are forwarded to the Flask backend (see ../main.py).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict': 'http://127.0.0.1:5001',
    },
  },
})
