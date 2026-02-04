import React, { useMemo, useState } from 'react';

// ---------- helpers ----------
const ALNUM_RE = /^[A-Za-z0-9]+$/;

function isAlnumNoWhitespace(s: string) {
	return ALNUM_RE.test(s);
}

function parseOptionalNumber(raw: string): number | undefined {
	const t = raw.trim();
	if (t === '') return undefined;
	const n = Number(t);
	return Number.isFinite(n) ? n : undefined;
}

function clamp01(x: number) {
	if (!Number.isFinite(x)) return 0;
	return Math.max(0, Math.min(1, x));
}

// ---------- types ----------
type SimConfig = {
	// basic
	name: string;
	n_diploid_samples: number;
	masking_rate: number;

	// advanced
	Ne: number;
	sequence_length: number;
	recombination_rate: number;
	mutation_rate: number;

	seed?: number;
	min_variants: number;
	full_data: boolean;
};

type GenerateRequest = {
	params: Pick<SimConfig, 'name'> & Partial<SimConfig>;
};

type Props = {
	apiBase: string;
	xApiKey: string;
	endpoint?: string;
};

// ---------- defaults ----------
const DEFAULTS: SimConfig = {
	name: '',
	n_diploid_samples: NaN,
	masking_rate: NaN,

	Ne: NaN,
	sequence_length: NaN,
	recombination_rate: NaN,
	mutation_rate: NaN,

	min_variants: NaN,
	full_data: false
};

export default function DatasetModelCreationForm({ apiBase, xApiKey, endpoint = '/api/create/data' }: Props) {
	const [advanced, setAdvanced] = useState(false);
	const [sending, setSending] = useState(false);
	const [status, setStatus] = useState('');
	const [responseJson, setResponseJson] = useState<any>(null);

	const [cfg, setCfg] = useState<SimConfig>(DEFAULTS);
	const [seedText, setSeedText] = useState('');

	const [dirty, setDirty] = useState<Set<keyof SimConfig>>(() => new Set());
	const [seedDirty, setSeedDirty] = useState(false);

	function markDirty<K extends keyof SimConfig>(key: K) {
		setDirty((prev) => {
			const next = new Set(prev);
			next.add(key);
			return next;
		});
	}

	function update<K extends keyof SimConfig>(key: K, value: SimConfig[K]) {
		setCfg((prev) => ({ ...prev, [key]: value }));
		markDirty(key);
	}

	// ---------- validation ----------
	const errors = useMemo(() => {
		const e: string[] = [];

		// string validation
		if (!cfg.name.trim()) e.push('Dataset name is required.');
		if (cfg.name.trim() && !isAlnumNoWhitespace(cfg.name.trim())) {
			e.push('Dataset name must be alphanumeric only (no spaces).');
		}

		// numeric validation
		if (!Number.isFinite(cfg.n_diploid_samples) || !Number.isInteger(cfg.n_diploid_samples) || cfg.n_diploid_samples <= 0) {
			e.push('Number of samples must be a positive integer.');
		}

		if (!Number.isFinite(cfg.masking_rate) || cfg.masking_rate < 0 || cfg.masking_rate > 1) {
			e.push('Masking rate must be a number between 0 and 1.');
		}

		if (advanced) {
			if (Number.isFinite(cfg.Ne) && (!Number.isInteger(cfg.Ne) || cfg.Ne <= 0)) {
				e.push('Ne must be a positive integer.');
			}

			if (Number.isFinite(cfg.sequence_length) && (!Number.isInteger(cfg.sequence_length) || cfg.sequence_length <= 0)) {
				e.push('Sequence length must be a positive integer.');
			}

			if (Number.isFinite(cfg.recombination_rate) && cfg.recombination_rate <= 0) {
				e.push('Recombination rate must be a positive number.');
			}

			if (Number.isFinite(cfg.mutation_rate) && cfg.mutation_rate <= 0) {
				e.push('Mutation rate must be a positive number.');
			}

			if (Number.isFinite(cfg.min_variants) && (!Number.isInteger(cfg.min_variants) || cfg.min_variants <= 0)) {
				e.push('Min variants must be a positive integer.');
			}

			const seedVal = parseOptionalNumber(seedText);
			if (seedText.trim() !== '' && seedVal === undefined) {
				e.push('Seed must be numeric if provided.');
			}
			if (seedVal !== undefined && (!Number.isInteger(seedVal) || seedVal < 0)) {
				e.push('Seed must be a non-negative integer.');
			}
		}

		return e;
	}, [cfg, advanced, seedText]);

	// ---------- submit ----------
	async function submit(e: React.FormEvent) {
		e.preventDefault();
		if (errors.length) return;

		setSending(true);
		setStatus('Sending...');
		setResponseJson(null);

		try {
			const seedVal = parseOptionalNumber(seedText);

			const params: Pick<SimConfig, 'name'> & Partial<SimConfig> = {
				name: cfg.name.trim(),
				full_data: cfg.full_data
			};

			// REQUIRED fields: send if finite
			if (Number.isFinite(cfg.n_diploid_samples)) {
				params.n_diploid_samples = cfg.n_diploid_samples;
			}
			if (Number.isFinite(cfg.masking_rate)) {
				params.masking_rate = clamp01(cfg.masking_rate);
			}

			// OPTIONAL advanced: send any finite ones (user may fill 1 or many)
			if (advanced) {
				if (Number.isFinite(cfg.Ne)) params.Ne = cfg.Ne;
				if (Number.isFinite(cfg.sequence_length)) params.sequence_length = cfg.sequence_length;
				if (Number.isFinite(cfg.recombination_rate)) params.recombination_rate = cfg.recombination_rate;
				if (Number.isFinite(cfg.mutation_rate)) params.mutation_rate = cfg.mutation_rate;
				if (Number.isFinite(cfg.min_variants)) params.min_variants = cfg.min_variants;

				// Seed: only if they touched it AND provided a valid number
				if (seedDirty && seedVal !== undefined) params.seed = seedVal;
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

			setDirty(new Set());
			setSeedDirty(false);
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
					Dataset name (alphanumeric, no spaces) <span style={{ color: 'crimson' }}>*</span>
					<input
						value={cfg.name}
						onChange={(e) => update('name', e.target.value)}
						placeholder="mydataset01"
						style={{ padding: '0.5rem' }}
					/>
				</label>

				<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
					<label>
						Num diploid samples <span style={{ color: 'crimson' }}>*</span>
						<input
							type="number"
							value={Number.isFinite(cfg.n_diploid_samples) ? cfg.n_diploid_samples : ''}
							placeholder="200"
							onChange={(e) => update('n_diploid_samples', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
							min={1}
							step={1}
							style={{ width: '100%', padding: '0.5rem' }}
						/>
					</label>

					<label>
						Masking rate (0–1) <span style={{ color: 'crimson' }}>*</span>
						<input
							type="number"
							value={Number.isFinite(cfg.masking_rate) ? cfg.masking_rate : ''}
							placeholder="0.2"
							onChange={(e) => update('masking_rate', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
							min={0}
							max={1}
							step={0.01}
							style={{ width: '100%', padding: '0.5rem' }}
						/>
					</label>
				</div>

				<p style={{ fontSize: '0.85rem', color: '#555' }}>
					<span style={{ color: 'crimson' }}>*</span> Required fields
				</p>

				<label style={{ display: 'flex', gap: 8, marginTop: 12 }}>
					<input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} />
					Advanced features
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
							Ne
							<input
								type="number"
								value={Number.isFinite(cfg.Ne) ? cfg.Ne : ''}
								placeholder="10000"
								onChange={(e) => update('Ne', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								min={1}
								step={1}
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label>
							Sequence length
							<input
								type="number"
								value={Number.isFinite(cfg.sequence_length) ? cfg.sequence_length : ''}
								placeholder="100000"
								onChange={(e) => update('sequence_length', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								min={1}
								step={1}
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label>
							Recombination rate
							<input
								type="number"
								value={Number.isFinite(cfg.recombination_rate) ? cfg.recombination_rate : ''}
								placeholder="1e-8"
								onChange={(e) => update('recombination_rate', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								step={1e-9}
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label>
							Mutation rate
							<input
								type="number"
								value={Number.isFinite(cfg.mutation_rate) ? cfg.mutation_rate : ''}
								placeholder="1e-8"
								onChange={(e) => update('mutation_rate', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								step={1e-9}
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label>
							Min variants
							<input
								type="number"
								value={Number.isFinite(cfg.min_variants) ? cfg.min_variants : ''}
								placeholder="100"
								onChange={(e) => update('min_variants', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								min={1}
								step={1}
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label>
							Seed (blank = auto)
							<input
								value={seedText}
								onChange={(e) => {
									setSeedText(e.target.value);
									setSeedDirty(true);
								}}
								placeholder="42"
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center', cursor: 'pointer' }}>
							<input type="checkbox" checked={cfg.full_data} onChange={(e) => update('full_data', e.target.checked)} />
							<span style={{ fontWeight: 'bold' }}>Generate datasets for training a model</span>
						</label>
					</div>
				</fieldset>
			)}

			<button type="submit" disabled={sending || errors.length > 0} style={{ padding: '0.75rem' }}>
				{sending ? 'Generating...' : 'Send to backend'}
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
