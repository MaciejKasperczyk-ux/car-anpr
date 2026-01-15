export function getApiBase() {
  return import.meta.env.VITE_API_BASE || "";
}

export async function startRun() {
  const res = await fetch(`/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({})
  });
  if (!res.ok) throw new Error(`startRun failed: ${res.status}`);
  return await res.json();
}

export async function listRuns() {
  const res = await fetch(`/api/runs`);
  if (!res.ok) throw new Error(`listRuns failed: ${res.status}`);
  return await res.json();
}

export async function getRun(runId) {
  const res = await fetch(`/api/runs/${runId}`);
  if (!res.ok) throw new Error(`getRun failed: ${res.status}`);
  return await res.json();
}

export function openRunWebSocket(runId, onLine) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host;

  const base = import.meta.env.VITE_WS_BASE;
  const url = base
    ? `${base.replace(/^http/, "ws")}/ws/runs/${runId}`
    : `${proto}://${host}/ws/runs/${runId}`;

  const ws = new WebSocket(url);

  ws.onmessage = (ev) => {
    onLine(String(ev.data));
  };

  ws.onerror = () => {
    onLine("[client] websocket error");
  };

  ws.onclose = () => {
    onLine("[client] websocket closed");
  };

  return ws;
}
