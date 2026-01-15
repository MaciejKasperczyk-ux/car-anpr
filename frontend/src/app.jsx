import React, { useEffect, useMemo, useRef, useState } from "react";
import { getRun, listRuns, openRunWebSocket, startRun } from "./api.js";

function formatTs(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toISOString().replace("T", " ").replace("Z", "");
}

export default function App() {
  const [runs, setRuns] = useState([]);
  const [activeRunId, setActiveRunId] = useState("");
  const [activeRun, setActiveRun] = useState(null);
  const [logs, setLogs] = useState([]);
  const [busy, setBusy] = useState(false);

  const wsRef = useRef(null);
  const logEndRef = useRef(null);

  const canStart = useMemo(() => !busy, [busy]);

  async function refreshRuns() {
    const data = await listRuns();
    setRuns(data.items || []);
  }

  async function selectRun(runId) {
    setActiveRunId(runId);
    setLogs([]);
    setActiveRun(null);

    if (wsRef.current) {
      try { wsRef.current.close(); } catch {}
      wsRef.current = null;
    }

    const info = await getRun(runId);
    setActiveRun(info);

    wsRef.current = openRunWebSocket(runId, (line) => {
      setLogs((prev) => {
        const next = prev.length > 2000 ? prev.slice(prev.length - 2000) : prev;
        return [...next, line];
      });
    });
  }

  async function onStart() {
    setBusy(true);
    try {
      const res = await startRun();
      await refreshRuns();
      await selectRun(res.run_id);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refreshRuns();
    const id = setInterval(refreshRuns, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!logEndRef.current) return;
    logEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        try { wsRef.current.close(); } catch {}
      }
    };
  }, []);

  return (
    <div style={{ fontFamily: "Arial, sans-serif", padding: 16, maxWidth: 1200, margin: "0 auto" }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>ANPR Evaluate</h2>
        <button onClick={onStart} disabled={!canStart} style={{ padding: "8px 12px" }}>
          Start evaluate
        </button>
        <button onClick={refreshRuns} style={{ padding: "8px 12px" }}>
          Refresh
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 16, marginTop: 16 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 12 }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Runs</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {runs.map((r) => (
              <button
                key={r.run_id}
                onClick={() => selectRun(r.run_id)}
                style={{
                  textAlign: "left",
                  padding: 10,
                  borderRadius: 8,
                  border: activeRunId === r.run_id ? "2px solid #333" : "1px solid #ddd",
                  background: "#fff",
                  cursor: "pointer"
                }}
              >
                <div style={{ fontWeight: 700, fontSize: 12 }}>{r.run_id}</div>
                <div style={{ fontSize: 12 }}>state: {r.state}</div>
                <div style={{ fontSize: 12 }}>started: {formatTs(r.started_at)}</div>
              </button>
            ))}
            {runs.length === 0 ? <div style={{ fontSize: 12, color: "#666" }}>No runs yet</div> : null}
          </div>
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <div>
              <div style={{ fontWeight: 700 }}>Live logs</div>
              <div style={{ fontSize: 12, color: "#666" }}>
                {activeRun ? `state: ${activeRun.state} exit_code: ${activeRun.exit_code ?? ""}` : "Select a run"}
              </div>
            </div>
          </div>

          <div
            style={{
              marginTop: 10,
              height: 520,
              overflow: "auto",
              background: "#0b0b0b",
              color: "#e6e6e6",
              borderRadius: 8,
              padding: 10,
              fontFamily: "Consolas, monospace",
              fontSize: 12,
              whiteSpace: "pre-wrap"
            }}
          >
            {logs.map((l, idx) => (
              <div key={idx}>{l}</div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
