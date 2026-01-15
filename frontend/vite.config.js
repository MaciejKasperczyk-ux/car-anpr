import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: process.env.VITE_API_BASE || "http://localhost:8000",
        changeOrigin: true
      },
      "/ws": {
        target: process.env.VITE_API_BASE || "ws://localhost:8000",
        ws: true,
        changeOrigin: true
      }
    }
  }
});
