import { useState } from 'react';
import DatasetDashboard from './components/DatasetDisplayDashboard.js';
import DatasetModelCreationForm from './components/DatasetModelCreationForm.js';

type ApiSuccessDatasets = {
	status: 'success';
	message: string;
	data: {
		datasets: string[];
		count: number;
	};
};

type ApiError = {
	status: 'error';
	code?: string;
	message: string;
};

export default function App() {
	const API_BASE = import.meta.env.VITE_API_BASE;
	const API_KEY = import.meta.env.VITE_X_API_KEY;

	const [msg, setMsg] = useState<string | null>(null);
	const [status, setStatus] = useState('');
	const [datasets, setDatasets] = useState<string[]>([]);

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

	async function fetchDatasets() {
		try {
			setStatus('Fetching datasets...');

			const r = await fetch(`${API_BASE}/api/datasets/list`, {
				method: 'GET',
				headers: { 'x-api-key': API_KEY }
			});

			const j: ApiSuccessDatasets | ApiError = await r.json();

			if (j.status === 'success') {
				setDatasets(j.data.datasets);
				setMsg(`Loaded ${j.data.count} datasets`);
				setStatus('Success!');
			} else {
				setDatasets([]);
				setStatus(j.message || 'Failed to load datasets');
			}
		} catch {
			setDatasets([]);
			setStatus('Error fetching datasets');
		}
	}

	return (
		<div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
			<button onClick={pingBackend}>Ping FastAPI</button>
			<button onClick={fetchDatasets} style={{ marginLeft: '1rem' }}>
				List Datasets
			</button>

			{status && <p>{status}</p>}
			{msg && <p>Message: {msg}</p>}

			<DatasetDashboard apiBase={API_BASE} xApiKey={API_KEY} datasets={datasets} />

			<DatasetModelCreationForm apiBase={API_BASE} xApiKey={API_KEY} />
		</div>
	);
}
