import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    basicSsl(),
  ],
  server: {
    host: true, // Listen on all local IPs
    proxy: {
      '/api': {
        target: 'http://172.20.233.159:8000',
        changeOrigin: true,
      },
    },
  },
})
