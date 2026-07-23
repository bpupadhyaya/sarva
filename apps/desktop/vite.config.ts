/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API calls to `sarva serve` (default 127.0.0.1:8000) so
// `npm run dev` and the FastAPI backend can run side by side without CORS
// config. The production build has no proxy — it's served BY the FastAPI
// app itself (see core/sarva/server/app.py), so requests are same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/chat": "http://127.0.0.1:8000",
      "/models": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/doctor": "http://127.0.0.1:8000",
      "/config": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: {
    outDir: "dist",
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/setupTests.ts"],
  },
});
