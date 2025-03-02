import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/predict': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        secure: false,
        cors: true
      }
    }
  }
})
