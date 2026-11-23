import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Relative asset paths, so the same build works at the site root (Flask)
// and under /ASLingo/ on GitHub Pages.
export default defineConfig({
  base: './',
  plugins: [react()],
})
