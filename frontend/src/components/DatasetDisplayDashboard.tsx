import { useMemo, useState } from 'react';
import FamilyTreeVisualization from './FamilyTreeVisualization.js';

type DatasetDashboardProps = {
	apiBase: string;
	xApiKey: string;
	datasets: string[];
	maxPreviewRows?: number;
};

type CsvPreview = {
	headers: string[];
	rows: string[][];
	estimatedTotalRows?: number;
};

type DashboardState = {
	observedCsvRaw?: string;
	truthCsvRaw?: string;
};

type ApiEnvelope = { status: 'success'; data?: any; files?: any; message?: string } | { status: 'error'; message?: string; code?: string } | any;

function parseCsvPreview(csvText: string, maxRows: number): CsvPreview {
	const text = csvText.replace(/^\uFEFF/, ''); // strip BOM if present
	const lines: string[] = [];
	// Normalize newlines but keep it simple
	text.split(/\r\n|\n|\r/).forEach((l) => {
		// keep empty lines out
		if (l.trim().length > 0) lines.push(l);
	});

	if (lines.length === 0) return { headers: [], rows: [] };

	// Robust-ish CSV line parser (handles quoted commas)
	const parseLine = (line: string): string[] => {
		const out: string[] = [];
		let cur = '';
		let inQuotes = false;

		for (let i = 0; i < line.length; i++) {
			const ch = line[i];

			if (ch === '"') {
				// If we see "" inside quotes, that's an escaped quote
				if (inQuotes && line[i + 1] === '"') {
					cur += '"';
					i++;
				} else {
					inQuotes = !inQuotes;
				}
				continue;
			}

			if (ch === ',' && !inQuotes) {
				out.push(cur);
				cur = '';
				continue;
			}

			cur += ch;
		}
		out.push(cur);
		return out.map((s) => s.trim());
	};

	const headers = parseLine(lines[0]);
	const rows: string[][] = [];

	for (let i = 1; i < lines.length && rows.length < maxRows; i++) {
		rows.push(parseLine(lines[i]));
	}

	return {
		headers,
		rows,
		estimatedTotalRows: Math.max(0, lines.length - 1)
	};
}

function clampText(s: string, maxLen = 80): string {
	if (s.length <= maxLen) return s;
	return s.slice(0, maxLen - 1) + '…';
}

export default function DatasetDashboard({ apiBase, xApiKey, datasets, maxPreviewRows = 10 }: DatasetDashboardProps) {
	const [selected, setSelected] = useState<string>('');
	const [loading, setLoading] = useState(false);
	const [status, setStatus] = useState<string>('');
	const [data, setData] = useState<DashboardState>({});
	const [selectedIndId, setSelectedIndId] = useState<string>('');
	const [familyTreeData, setFamilyTreeData] = useState<any>(null);

	const canLoad = datasets.length > 0 && selected.trim().length > 0 && !loading;
	const hasLoadedDashboard = !!(data.observedCsvRaw || data.truthCsvRaw);

	// Previews
	const observedPreview = useMemo(() => {
		if (!data.observedCsvRaw) return null;
		return parseCsvPreview(data.observedCsvRaw, maxPreviewRows);
	}, [data.observedCsvRaw, maxPreviewRows]);

	const truthPreview = useMemo(() => {
		if (!data.truthCsvRaw) return null;
		return parseCsvPreview(data.truthCsvRaw, maxPreviewRows);
	}, [data.truthCsvRaw, maxPreviewRows]);

	// Individual selection
	const availableIds = useMemo(() => {
		if (!data.observedCsvRaw) return [];
		const firstLine = data.observedCsvRaw.split('\n')[0] || '';
		// Match headers like i_0000, i_0001 and convert to "0", "1"
		return firstLine
			.split(',')
			.map((h) => h.trim())
			.filter((h) => h.startsWith('i_'))
			.map((h) => h.replace('i_', '').replace(/^0+/, '') || '0');
	}, [data.observedCsvRaw]);

	async function loadDashboard() {
		if (!selected) return;

		setLoading(true);
		setStatus('Loading dashboard files...');
		setData({});

		try {
			const url = `${apiBase}/api/dataset/${encodeURIComponent(selected)}/dashboard`;
			const resp = await fetch(url, {
				method: 'GET',
				headers: {
					'x-api-key': xApiKey
				}
			});

			const contentType = resp.headers.get('content-type') || '';

			if (!resp.ok) {
				let errMsg = `Request failed (${resp.status})`;
				try {
					if (contentType.includes('application/json')) {
						const j = (await resp.json()) as ApiEnvelope;
						errMsg = j?.message || errMsg;
					} else {
						errMsg = (await resp.text()) || errMsg;
					}
				} catch {
					// ignore parse errors
				}
				setStatus(errMsg);
				return;
			}

			// JSON envelope
			if (contentType.includes('application/json')) {
				const j = (await resp.json()) as ApiEnvelope;

				// Accept multiple possible shapes:
				// A) { status:'success', data:{ observed_genotypes_csv:'...', truth_genotypes_csv:'...', trees_base64:'...' } }
				// B) { status:'success', files:{ ... } }
				// C) { observed_genotypes_csv:'...', truth_genotypes_csv:'...', trees_base64:'...' } (no envelope)
				const payload = j?.data ?? j?.files ?? j;

				const next: DashboardState = {};

				// Try common keys
				next.observedCsvRaw = payload?.observed_genotypes_csv ?? payload?.observed_csv ?? payload?.observedGenotypesCsv ?? payload?.observed;

				next.truthCsvRaw = payload?.truth_genotypes_csv ?? payload?.truth_csv ?? payload?.truthGenotypesCsv ?? payload?.truth;

				setData(next);
				setStatus('Loaded!');
				return;
			}

			const txt = await resp.text();
			setStatus(`Loaded non-JSON response (${contentType || 'unknown'}). Showing first bytes.`);
			setData({
				observedCsvRaw: txt
			});
		} catch (e: any) {
			setStatus(e?.message || 'Error fetching dashboard');
		} finally {
			setLoading(false);
		}
	}

	async function loadFamilyTree() {
		if (!selected || !selectedIndId) return;

		setLoading(true);
		setStatus(`Fetching family tree for ID ${selectedIndId}...`);
		try {
			const url = `${apiBase}/api/dataset/${encodeURIComponent(selected)}/tree/${selectedIndId}`;
			const resp = await fetch(url, {
				headers: { 'x-api-key': xApiKey }
			});

			if (!resp.ok) throw new Error(`Tree fetch failed: ${resp.status}`);

			const j = await resp.json();
			setFamilyTreeData(j.data);
			setStatus('Tree loaded.');
		} catch (e: any) {
			setStatus(e.message);
		} finally {
			setLoading(false);
		}
	}

	async function downloadAllDatasetZip() {
		if (!selected) return;

		setLoading(true);
		setStatus('Preparing download...');

		try {
			const url = `${apiBase}/api/dataset/${encodeURIComponent(selected)}/download`;

			const resp = await fetch(url, {
				method: 'GET',
				headers: {
					'x-api-key': xApiKey
				}
			});

			if (!resp.ok) {
				const contentType = resp.headers.get('content-type') || '';
				let errMsg = `Download failed (${resp.status})`;

				try {
					if (contentType.includes('application/json')) {
						const j = (await resp.json()) as ApiEnvelope;
						errMsg = j?.message || errMsg;
					} else {
						errMsg = (await resp.text()) || errMsg;
					}
				} catch {
					// ignore
				}

				setStatus(errMsg);
				return;
			}

			const blob = await resp.blob();

			// Try to respect Content-Disposition filename=...
			const dispo = resp.headers.get('content-disposition') || '';
			const match = dispo.match(/filename\*?=(?:UTF-8''|")?([^\";\n]+)\"?/i);
			const filename = (match?.[1] ? decodeURIComponent(match[1]) : `${selected}.zip`).replace(/[/\\]/g, '_');

			const href = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = href;
			a.download = filename;
			document.body.appendChild(a);
			a.click();
			a.remove();
			URL.revokeObjectURL(href);

			setStatus('Download started.');
		} catch (e: any) {
			setStatus(e?.message || 'Error downloading dataset');
		} finally {
			setLoading(false);
		}
	}

	return (
		<div style={{ marginTop: '1.25rem' }}>
			<h3>Dataset Dashboard</h3>

			{datasets.length === 0 ? (
				<p style={{ opacity: 0.8 }}>
					Click <b>List Datasets</b> first.
				</p>
			) : (
				<div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
					<label>
						<span style={{ display: 'block', fontSize: '0.9rem', marginBottom: '0.25rem' }}>Choose a dataset</span>
						<select value={selected} onChange={(e) => setSelected(e.target.value)} style={{ padding: '0.4rem', minWidth: '260px' }}>
							<option value="" disabled>
								Select…
							</option>
							{datasets.map((d) => (
								<option key={d} value={d}>
									{d}
								</option>
							))}
						</select>
					</label>

					<button onClick={loadDashboard} disabled={!canLoad} style={{ height: 'fit-content', padding: '0.55rem 0.9rem' }}>
						{loading ? 'Loading…' : 'Load Dashboard'}
					</button>
				</div>
			)}

			{status && <p style={{ marginTop: '0.75rem' }}>{status}</p>}

			{/* Download all dataset files */}
			{hasLoadedDashboard && (
				<div style={{ marginTop: '1rem', padding: '0.9rem', border: '1px solid #ddd', borderRadius: 10 }}>
					<h4 style={{ marginTop: 0 }}>Download</h4>
					<p style={{ marginTop: 0, opacity: 0.8 }}>
						Download a zip containing all files for <b>{selected}</b> (trees, truth, observed, pedigree, metadata).
					</p>

					<button onClick={downloadAllDatasetZip} disabled={loading || !selected}>
						{loading ? 'Preparing…' : 'Download all data (.zip)'}
					</button>
				</div>
			)}

			{/* CSV previews */}
			{observedPreview && <CsvTable title="observed_genotypes.csv (preview)" preview={observedPreview} maxRows={maxPreviewRows} />}

			{truthPreview && <CsvTable title="truth_genotypes.csv (preview)" preview={truthPreview} maxRows={maxPreviewRows} />}

			{/* Family tree */}
			{data.observedCsvRaw && (
				<div style={{ marginTop: '1.5rem', padding: '1rem', border: '2px solid #3b82f6', borderRadius: 10 }}>
					<h4>Family Tree Explorer</h4>
					<div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end' }}>
						<label>
							<span style={{ display: 'block', fontSize: '0.8rem' }}>Select Individual ID</span>
							<select
								value={selectedIndId}
								onChange={(e) => setSelectedIndId(e.target.value)}
								style={{ padding: '0.4rem', minWidth: '150px' }}
							>
								<option value="">-- Choose ID --</option>
								{availableIds.map((id) => (
									<option key={id} value={id}>
										Individual {id}
									</option>
								))}
							</select>
						</label>
						<button onClick={loadFamilyTree} disabled={!selectedIndId || loading}>
							Visualize Tree
						</button>
					</div>
				</div>
			)}

			{familyTreeData && <FamilyTreeVisualization data={familyTreeData} />}
		</div>
	);
}

function CsvTable({ title, preview, maxRows }: { title: string; preview: CsvPreview; maxRows: number }) {
	const { headers, rows, estimatedTotalRows } = preview;

	return (
		<div style={{ marginTop: '1rem', padding: '0.9rem', border: '1px solid #ddd', borderRadius: 10 }}>
			<h4 style={{ marginTop: 0 }}>{title}</h4>

			<p style={{ marginTop: 0, opacity: 0.8 }}>
				Showing first <b>{Math.min(rows.length, maxRows)}</b>
				{typeof estimatedTotalRows === 'number' ? (
					<>
						{' '}
						of about <b>{estimatedTotalRows.toLocaleString()}</b> rows
					</>
				) : null}
				.
			</p>

			<div style={{ overflowX: 'auto' }}>
				<table style={{ borderCollapse: 'collapse', width: '100%' }}>
					<thead>
						<tr>
							{headers.map((h, idx) => (
								<th
									key={idx}
									style={{
										textAlign: 'left',
										borderBottom: '1px solid #ccc',
										padding: '0.5rem',
										whiteSpace: 'nowrap'
									}}
									title={h}
								>
									{clampText(h, 40)}
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{rows.map((r, ridx) => (
							<tr key={ridx}>
								{headers.map((_, cidx) => (
									<td
										key={cidx}
										style={{
											borderBottom: '1px solid #eee',
											padding: '0.5rem',
											whiteSpace: 'nowrap'
										}}
										title={r[cidx] ?? ''}
									>
										{clampText(String(r[cidx] ?? ''), 60)}
									</td>
								))}
							</tr>
						))}
						{rows.length === 0 && (
							<tr>
								<td colSpan={headers.length || 1} style={{ padding: '0.5rem', opacity: 0.75 }}>
									No rows to display.
								</td>
							</tr>
						)}
					</tbody>
				</table>
			</div>

			<p style={{ marginBottom: 0, marginTop: '0.75rem', opacity: 0.75 }}>
				Tip: these CSVs are wide (lots of <code>ind_####</code> columns). Horizontal scroll is expected.
			</p>
		</div>
	);
}
