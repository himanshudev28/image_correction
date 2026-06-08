import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: React dev server on :5173 forwards /api -> FastAPI on :8000,
// so there are no cross-origin issues on the first `npm run dev` (no CORS needed,
// though the backend also allows the Vite origin as a backup).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
