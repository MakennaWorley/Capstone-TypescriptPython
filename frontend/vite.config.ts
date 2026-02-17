import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
	const env = loadEnv(mode, process.cwd(), '');

	const proxyHost = env.VITE_PROXY_HOST;
	const apiBase = env.VITE_API_BASE;

	return {
		plugins: [react()],
		server: {
			host: true,
			port: 5173,
			proxy: {
				[apiBase]: {
					target: proxyHost,
					changeOrigin: true,
					rewrite: (path) => path.replace(apiBase, ''),
					secure: false
				}
			}
		}
	};
});
