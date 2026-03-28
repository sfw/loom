import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 1420,
    proxy: {
      // Proxy all API requests to loomd so there are no CORS issues in dev
      "/runtime": "http://127.0.0.1:9000",
      "/models": "http://127.0.0.1:9000",
      "/workspaces": "http://127.0.0.1:9000",
      "/settings": "http://127.0.0.1:9000",
      "/conversations": "http://127.0.0.1:9000",
      "/runs": "http://127.0.0.1:9000",
      "/tasks": "http://127.0.0.1:9000",
      "/approvals": "http://127.0.0.1:9000",
      "/notifications": "http://127.0.0.1:9000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    pool: "forks",
    fileParallelism: false,
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
  },
});
