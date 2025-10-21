import React, { useState, useRef } from "react";
import type { CSSProperties } from "react";

// Pure React + inline CSS (no Tailwind)
const BACKEND_URL = (import.meta as any)?.env?.VITE_BACKEND_URL || "http://localhost:8000";

type TopK = { class: string; score: number; index: number }[];

// Global CSS to ensure full‑width background and responsive columns
const GlobalStyles = () => (
  <style>{`
    html, body, #root { height: 100%; }
    body { margin: 0; background: linear-gradient(135deg,#f7fafc,#edf2f7); color: #0f172a; }
    .container { max-width: 1280px; margin: 0 auto; padding: 48px 24px; }
    .cards { display: grid; grid-template-columns: 1fr; gap: 24px; }
    @media (min-width: 980px) { .cards { grid-template-columns: 1fr 1fr; } }
  `}</style>
);

const styles: Record<string, CSSProperties> = {
  title: { fontSize: 40, fontWeight: 700, margin: 0 },
  subtitle: { marginTop: 8, color: "#475569", fontSize: 15 },
  card: { background: "#fff", border: "1px solid #e2e8f0", borderRadius: 16, boxShadow: "0 1px 2px rgba(16,24,40,.05)" },
  cardBody: { padding: 24 },
  drop: { border: "2px dashed #cbd5e1", borderRadius: 12, padding: 24, textAlign: "center", cursor: "pointer" },
  dropTitle: { fontSize: 18, fontWeight: 600 },
  dropHint: { fontSize: 13, color: "#64748b" },
  previewImg: { maxHeight: 300, borderRadius: 12, objectFit: "contain" as const },
  row: { display: "flex", gap: 12, marginTop: 16 },
  btnPrimary: { padding: "10px 16px", borderRadius: 12, border: 0, background: "#0f172a", color: "#fff", fontSize: 14, cursor: "pointer" },
  btnGhost: { padding: "10px 16px", borderRadius: 12, border: "1px solid #cbd5e1", background: "#fff", color: "#0f172a", fontSize: 14, cursor: "pointer" },
  btnDisabled: { opacity: 0.5, cursor: "not-allowed" },
  error: { marginTop: 12, fontSize: 13, color: "#991b1b", background: "#fef2f2", border: "1px solid #fecaca", padding: 12, borderRadius: 12 },
  backend: { marginTop: 10, fontSize: 12, color: "#64748b" },
  sectionTitle: { fontWeight: 700, fontSize: 22, marginBottom: 12 },
  empty: { color: "#64748b", fontSize: 14 },
  item: { display: "flex", alignItems: "center", gap: 12 },
  rank: { width: 24, color: "#64748b", fontSize: 12 },
  itemHead: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, fontSize: 14 },
  itemLabel: { fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" as const },
  itemMeta: { color: "#64748b", fontWeight: 400 },
  itemScore: { width: 60, textAlign: "right", fontVariantNumeric: "tabular-nums" },
  barWrap: { marginTop: 8, height: 8, background: "#f1f5f9", borderRadius: 999, overflow: "hidden" },
  bar: { height: 8, background: "#0f172a", transition: "width .2s ease" },
  skeleton: { height: 40, borderRadius: 12, background: "#f1f5f9", animation: "pulse 1.2s ease-in-out infinite" },
  footer: { marginTop: 32, fontSize: 12, color: "#64748b" },
};

const Pulse = () => <style>{`@keyframes pulse{0%,100%{opacity:.6}50%{opacity:1}}`}</style>;

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topk, setTopk] = useState<TopK | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const pick = (f: File | null) => {
    if (!f) return;
    setFile(f);
    setTopk(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop: React.DragEventHandler = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) pick(f);
  };

  const onChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    pick(e.target.files?.[0] || null);
  };

  const predict = async () => {
    if (!file) return setError("Please choose an image first.");
    setLoading(true);
    setError(null);
    setTopk(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${BACKEND_URL}/predict`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`(${res.status}) ${await res.text()}`);
      const data = await res.json();
      setTopk(data.topk || []);
    } catch (err: any) {
      setError(err?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <GlobalStyles />
      <Pulse />
      <div className="container">
        <header style={{ marginBottom: 28 }}>
          <h1 style={styles.title}>SimCLR Linear Probe Demo</h1>
          <p style={styles.subtitle}>Upload an image to get Top‑K predictions from your FastAPI service.</p>
        </header>

        <div className="cards">
          {/* Upload card */}
          <section style={styles.card}>
            <div style={styles.cardBody}>
              <div onDragOver={(e) => e.preventDefault()} onDrop={onDrop} style={styles.drop} onClick={() => inputRef.current?.click()}>
                {preview ? (
                  <img src={preview} alt="preview" style={styles.previewImg} />
                ) : (
                  <>
                    <div style={styles.dropTitle}>Drop an image here</div>
                    <div style={styles.dropHint}>or click to choose a file</div>
                  </>
                )}
                <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={onChange} />
              </div>

              <div style={styles.row}>
                <button onClick={() => inputRef.current?.click()} style={styles.btnPrimary}>Choose Image</button>
                <button onClick={predict} style={{ ...styles.btnGhost, ...(loading || !file ? styles.btnDisabled : {}) }} disabled={loading || !file}>
                  {loading ? "Predicting…" : "Predict"}
                </button>
              </div>

              {error && <div style={styles.error}>{error}</div>}
              <div style={styles.backend}>Backend: <code>{BACKEND_URL}</code></div>
            </div>
          </section>

          {/* Results card */}
          <section style={styles.card}>
            <div style={styles.cardBody}>
              <h2 style={styles.sectionTitle}>Top Predictions</h2>
              {!topk && !loading && (
                <div style={styles.empty}>No predictions yet. Upload an image and click <em>Predict</em>.</div>
              )}

              {loading && (
                <div style={{ display: "grid", gap: 12 }}>
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} style={styles.skeleton} />
                  ))}
                </div>
              )}

              {topk && !loading && (
                <ul style={{ display: "grid", gap: 12, listStyle: "none", padding: 0, margin: 0 }}>
                  {topk.map((r, i) => (
                    <li key={i} style={styles.item}>
                      <div style={styles.rank}>{i + 1}.</div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={styles.itemHead}>
                          <div style={styles.itemLabel}>
                            {r.class} <span style={styles.itemMeta}>(#{r.index})</span>
                          </div>
                          <div style={styles.itemScore}>{(r.score * 100).toFixed(1)}%</div>
                        </div>
                        <div style={styles.barWrap}>
                          <div style={{ ...styles.bar, width: `${Math.max(3, Math.round(r.score * 100))}%` }} />
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        </div>

        <footer style={styles.footer}>Tip: For production, serve this UI from the same domain as your API and restrict allowed origins.</footer>
      </div>
    </>
  );
}
