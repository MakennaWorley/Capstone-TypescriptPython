import Papa from "papaparse";
import { useState } from "react";

export default function App() {
  const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

  const [msg, setMsg] = useState(null);
  const [data, setData] = useState(null);
  const [name, setName] = useState("Makenna");
  const [status, setStatus] = useState("");

  // --- 1. Ping FastAPI ---
  async function pingBackend() {
    try {
      setStatus("Pinging...");
      const r = await fetch(`${API_BASE}/api/hello`);
      const j = await r.json();
      setMsg(j.message);
      setStatus("Success!");
    } catch (err) {
      setStatus("Error contacting backend");
    }
  }

  // --- 2. Load CSV ---
  async function loadCSV() {
    try {
      setStatus("Loading CSV...");
      const url = `${API_BASE}/poc/data/A_five_masked_out/csv`;
      const r = await fetch(url);
      const csvText = await r.text();

      const parsed = Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
      });

      setData(parsed.data);
      setStatus("CSV Loaded!");
    } catch (err) {
      console.error(err);
      setStatus("Error loading CSV");
    }
  }

  // --- 3. Compute summary statistics like df.describe() ---
  function summaryStats(rows) {
    if (!rows || rows.length === 0) return {};

    const numericCols = Object.keys(rows[0]).filter(
      (col) => typeof rows[0][col] === "number"
    );

    const stats = {};
    numericCols.forEach((col) => {
      const vals = rows.map((r) => r[col]).filter((v) => typeof v === "number");
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      stats[col] = { mean, min, max, count: vals.length };
    });

    return stats;
  }

  const stats = data ? summaryStats(data) : null;

  return (
    <div style={{ padding: "2rem", fontFamily: "system-ui, sans-serif" }}>
      <h1>Proof of Concept: Probabilistic Ancestral Inference</h1>
      <p>Backend: {API_BASE}</p>

      <button onClick={pingBackend}>Ping FastAPI</button>
      {msg && <p>Message: {msg}</p>}

      <br /><br />

      <button onClick={loadCSV}>Get me the DATA!</button>
      {status && <p>{status}</p>}

      {/* --- Data Table --- */}
      {data && (
        <>
          <h2>Data Table</h2>
          <table border="1" cellPadding={6} style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {Object.keys(data[0]).map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.slice(0, 50).map((row, i) => (
                <tr key={i}>
                  {Object.keys(row).map((col) => (
                    <td key={col}>{String(row[col])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontStyle: "italic" }}>
            Showing first 50 rows for performance.
          </p>

          <h2>Summary Statistics</h2>
          <table border="1" cellPadding={6} style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th>Column</th>
                <th>Mean</th>
                <th>Min</th>
                <th>Max</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stats).map(([col, s]) => (
                <tr key={col}>
                  <td>{col}</td>
                  <td>{s.mean.toFixed(4)}</td>
                  <td>{s.min}</td>
                  <td>{s.max}</td>
                  <td>{s.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <br /><br />

      {/* --- Name input section --- */}
      <h2>Misc</h2>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Name"
      />
      <button onClick={() => alert(`Hi, ${name}! (stub)`)}>Send</button>
    </div>
  );
}
