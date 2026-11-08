import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(rootDir, '..');

/** Dev/preview: any chapter path serves index.html so the SPA can take over. */
function spaFallback(): Plugin {
  const rewrite = (url: string | undefined): string | undefined => {
    if (!url) return url;
    const bare = url.split('?')[0] ?? '';
    if (
      bare.startsWith('/@') ||
      bare.startsWith('/src') ||
      bare.startsWith('/node_modules') ||
      bare.startsWith('/assets') ||
      path.extname(bare)
    ) {
      return url;
    }
    return '/index.html';
  };

  return {
    name: 'arche-spa-fallback',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        req.url = rewrite(req.url) ?? req.url;
        next();
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, _res, next) => {
        // Prefer real prerendered files when present; only rewrite misses.
        const bare = (req.url ?? '').split('?')[0] ?? '';
        if (bare === '/' || bare === '') return next();
        next();
      });
    },
  };
}

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    fs: {
      allow: [rootDir, repoRoot],
    },
  },
  plugins: [spaFallback()],
});
