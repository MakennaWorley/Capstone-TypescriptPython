import { useState } from 'react';
import DatasetModelForm from './components/DatasetModelForm.js';

export default function App() {
	const API_BASE = 'http://localhost:8000';
	const API_KEY = 'REACT_FRONTEND_REQUEST';

	const [msg, setMsg] = useState<string | null>(null);
	const [status, setStatus] = useState('');
	const [xApiKey, setXApiKey] = useState('');

	// --- Ping FastAPI ---
	async function pingBackend() {
		try {
			setStatus('Pinging...');
			const r = await fetch(`${API_BASE}/api/hello`);
			const j = await r.json();
			setMsg(j.message);
			setStatus('Success!');
		} catch {
			setStatus('Error contacting backend');
		}
	}

	return (
		<div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
			<button onClick={pingBackend}>Ping FastAPI</button>
			{msg && <p>Message: {msg}</p>}
			{status && <p>{status}</p>}

			<DatasetModelForm apiBase={API_BASE} xApiKey={API_KEY} />
		</div>
	);
}
