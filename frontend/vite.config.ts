import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/app/",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "localhost", // to allow access to all network requests use 0.0.0.0
    port: 5173,
    strictPort: true,
    origin: "http://localhost:5173",
    // Configure CORS securely for the Vite dev server to allow requests from localhost,
    // supports additional hostnames (via regex). If you use another `project_tld`, adjust this.
    cors: {
      origin: "http://localhost:5173",
    },
    proxy: {
      // Proxy API requests to the backend server
      "/api": {
        target: "http://127.0.0.1:8000", // Default backend address
        changeOrigin: true,
        // Optionally rewrite path if needed (e.g., remove /api prefix if backend doesn't expect it)
        // rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
