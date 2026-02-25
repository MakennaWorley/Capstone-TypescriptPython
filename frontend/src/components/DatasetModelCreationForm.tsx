import React, { useMemo, useState } from 'react';

// ---------- helpers ----------
const ALNUM_RE = /^[A-Za-z0-9]+$/;

function isAlnumNoWhitespace(s: string) {
	return ALNUM_RE.test(s);
}

// ---------- types ----------
type SimConfig = {
	// basic
	name: string;
	full_data: boolean;

	// advanced (scaling)
	n_generations: number;
	samples_per_generation: number;
};

type GenerateRequest = {
	params: Pick<SimConfig, 'name' | 'full_data'> & Partial<SimConfig> & { n_diploid_samples?: number };
};

type Props = {
	apiBase: string;
	xApiKey: string;
	endpoint?: string;
};

// ---------- defaults ----------
const DEFAULTS: SimConfig = {
	name: '',
	full_data: false,
	n_generations: NaN,
	samples_per_generation: NaN
};

export default function DatasetModelCreationForm({ apiBase, xApiKey, endpoint = '/api/create/data' }: Props) {
	const [advanced, setAdvanced] = useState(false);
	const [sending, setSending] = useState(false);
	const [status, setStatus] = useState('');
	const [responseJson, setResponseJson] = useState<any>(null);

	const [cfg, setCfg] = useState<SimConfig>(DEFAULTS);

	function update<K extends keyof SimConfig>(key: K, value: SimConfig[K]) {
		setCfg((prev) => ({ ...prev, [key]: value }));
	}

	const derivedTotal =
		Number.isFinite(cfg.n_generations) && Number.isFinite(cfg.samples_per_generation)
			? cfg.n_generations * cfg.samples_per_generation
			: undefined;

	// ---------- validation ----------
	const errors = useMemo(() => {
		const e: string[] = [];

		if (!cfg.name.trim()) e.push('Dataset name is required.');
		if (cfg.name.trim() && !isAlnumNoWhitespace(cfg.name.trim())) {
			e.push('Dataset name must be alphanumeric only (no spaces).');
		}

		if (advanced) {
			if (!Number.isFinite(cfg.n_generations) || !Number.isInteger(cfg.n_generations) || cfg.n_generations <= 0) {
				e.push('Number of generations must be a positive integer.');
			}
			if (!Number.isFinite(cfg.samples_per_generation) || !Number.isInteger(cfg.samples_per_generation) || cfg.samples_per_generation <= 0) {
				e.push('Individuals per generation must be a positive integer.');
			}
		}

		return e;
	}, [cfg, advanced]);

	// ---------- submit ----------
	async function submit(e: React.FormEvent) {
		e.preventDefault();
		if (errors.length) return;

		setSending(true);
		setStatus('Sending...');
		setResponseJson(null);

		try {
			const params: GenerateRequest['params'] = {
				name: cfg.name.trim(),
				full_data: cfg.full_data
			};

			// Only include parameters if Advanced is enabled
			if (advanced) {
				params.n_generations = cfg.n_generations;
				params.samples_per_generation = cfg.samples_per_generation;

				if (derivedTotal !== undefined) {
					params.n_diploid_samples = derivedTotal;
				}
			}

			const payload: GenerateRequest = { params };

			const r = await fetch(`${apiBase}${endpoint}`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'X-API-Key': xApiKey
				},
				body: JSON.stringify(payload)
			});

			const text = await r.text().catch(() => '');
			let maybeJson: any = null;
			try {
				maybeJson = text ? JSON.parse(text) : null;
			} catch {}

			if (!r.ok) {
				setStatus(`Error ${r.status}`);
				setResponseJson(maybeJson ?? text);
				return;
			}

			setStatus('Success!');
			setResponseJson(maybeJson ?? text);
		} catch (err) {
			console.error(err);
			setStatus('Network error');
		} finally {
			setSending(false);
		}
	}

	return (
		<form onSubmit={submit} style={{ display: 'grid', gap: '0.9rem', maxWidth: 720 }}>
			<h2>Create Dataset</h2>

			{/* BASIC */}
			<fieldset style={{ padding: '1rem' }}>
				<legend>Basic</legend>

				<label style={{ display: 'grid', gap: 6 }}>
					Dataset name (alphanumeric, no spaces)
					<input
						value={cfg.name}
						onChange={(e) => update('name', e.target.value)}
						placeholder="mydataset01"
						style={{ padding: '0.5rem' }}
					/>
				</label>

				<label style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center', cursor: 'pointer' }}>
					<input type="checkbox" checked={cfg.full_data} onChange={(e) => update('full_data', e.target.checked)} />
					<span>Generate datasets for training a model</span>
				</label>

				<label style={{ display: 'flex', gap: 8, marginTop: 12 }}>
					<input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} />
					Advanced Settings (scale individuals)
				</label>
			</fieldset>

			{/* ADVANCED */}
			{advanced && (
				<fieldset style={{ padding: '1rem' }}>
					<legend>
						Advanced <span style={{ fontSize: '0.8rem', fontWeight: 'normal' }}>(optional)</span>
					</legend>

					<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
						<label>
							Number of generations
							<input
								type="number"
								value={Number.isFinite(cfg.n_generations) ? cfg.n_generations : ''}
								placeholder="5"
								onChange={(e) => update('n_generations', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								min={1}
								step={1}
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label>
							Individuals per generation
							<input
								type="number"
								value={Number.isFinite(cfg.samples_per_generation) ? cfg.samples_per_generation : ''}
								placeholder="50"
								onChange={(e) => update('samples_per_generation', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								min={1}
								step={1}
								style={{ padding: '0.5rem' }}
							/>
						</label>
					</div>

					<p style={{ fontSize: '0.9rem', marginTop: 10 }}>
						Total individuals: <strong>{derivedTotal ?? '—'}</strong>
					</p>
				</fieldset>
			)}

			<button type="submit" disabled={sending || errors.length > 0} style={{ padding: '0.75rem' }}>
				{sending ? 'Generating...' : 'Generate Data'}
			</button>

			{status && <p>{status}</p>}

			{responseJson && (
				<pre style={{ whiteSpace: 'pre-wrap', padding: '0.75rem', border: '1px solid #ddd' }}>
					{typeof responseJson === 'string' ? responseJson : JSON.stringify(responseJson, null, 2)}
				</pre>
			)}
		</form>
	);
}
