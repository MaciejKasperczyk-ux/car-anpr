from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, Any

from app.runner import Runner
from app.models import RunCreateRequest, RunCreateResponse, RunInfo, RunsList
from app.settings import WORKDIR, DEFAULT_EVALUATE_CMD, MAX_LOG_LINES_IN_MEMORY


app = FastAPI(title="ANPR API")

runner = Runner(workdir=WORKDIR, ring_max_lines=MAX_LOG_LINES_IN_MEMORY)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunCreateResponse)
async def create_run(req: RunCreateRequest) -> RunCreateResponse:
    run_id = await runner.create_run(cmd=req.cmd, env=req.env or {}, meta={"name": req.name or ""})
    return RunCreateResponse(run_id=run_id)


@app.post("/runs/evaluate", response_model=RunCreateResponse)
async def run_evaluate() -> RunCreateResponse:
    run_id = await runner.create_run(cmd=DEFAULT_EVALUATE_CMD, env={}, meta={"name": "evaluate"})
    return RunCreateResponse(run_id=run_id)


@app.get("/runs", response_model=RunsList)
async def list_runs() -> RunsList:
    runs = await runner.list_runs()
    out = []
    for r in runs.values():
        out.append(
            RunInfo(
                run_id=r.run_id,
                state=r.state,
                exit_code=r.exit_code,
                started_at=r.started_at,
                finished_at=r.finished_at,
                meta=r.meta,
            )
        )
    out.sort(key=lambda x: x.started_at, reverse=True)
    return RunsList(runs=out)


@app.get("/runs/{run_id}", response_model=RunInfo)
async def get_run(run_id: str) -> RunInfo:
    r = await runner.get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunInfo(
        run_id=r.run_id,
        state=r.state,
        exit_code=r.exit_code,
        started_at=r.started_at,
        finished_at=r.finished_at,
        meta=r.meta,
    )


@app.get("/runs/{run_id}/logs")
async def get_logs(run_id: str) -> Dict[str, Any]:
    lines = await runner.snapshot_lines(run_id)
    if lines is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "lines": lines}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ANPR</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    button { padding: 8px 12px; }
    pre { background: #111; color: #ddd; padding: 12px; height: 70vh; overflow: auto; }
    .row { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }
    input { width: 520px; padding: 8px; }
  </style>
</head>
<body>
  <h2>ANPR evaluate live</h2>

  <div class="row">
    <button id="btnEval">Run evaluate</button>
    <input id="cmd" value="python -u evaluate.py" />
    <button id="btnRun">Run custom</button>
  </div>

  <div class="row">
    <div>Run id:</div>
    <div id="runId"></div>
    <div id="state"></div>
  </div>

  <pre id="log"></pre>

<script>
const logEl = document.getElementById("log");
const runIdEl = document.getElementById("runId");
const stateEl = document.getElementById("state");
let pollTimer = null;
let lastLen = 0;

function append(line) {
  logEl.textContent += line + "\\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function resetLog() {
  logEl.textContent = "";
  lastLen = 0;
}

async function postJson(url, bodyObj) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj)
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t);
  }
  return await r.json();
}

async function poll(runId) {
  try {
    const r1 = await fetch(`/runs/${runId}`);
    if (r1.ok) {
      const info = await r1.json();
      stateEl.textContent = `state=${info.state}` + (info.exit_code !== null ? ` exit=${info.exit_code}` : "");
    }

    const r2 = await fetch(`/runs/${runId}/logs`);
    if (!r2.ok) return;
    const data = await r2.json();
    const lines = data.lines || [];

    if (lines.length < lastLen) {
      resetLog();
    }

    for (let i = lastLen; i < lines.length; i++) {
      append(lines[i]);
    }
    lastLen = lines.length;

    if (stateEl.textContent.includes("finished") || stateEl.textContent.includes("failed")) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (e) {
    append("[error] " + e.message);
  }
}

async function start(runId) {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  resetLog();
  runIdEl.textContent = runId;
  stateEl.textContent = "";
  await poll(runId);
  pollTimer = setInterval(() => poll(runId), 1000);
}

document.getElementById("btnEval").onclick = async () => {
  try {
    const data = await postJson("/runs/evaluate", {});
    await start(data.run_id);
  } catch (e) {
    append("[error] " + e.message);
  }
};

document.getElementById("btnRun").onclick = async () => {
  try {
    const cmd = document.getElementById("cmd").value.trim();
    const data = await postJson("/runs", { cmd: cmd });
    await start(data.run_id);
  } catch (e) {
    append("[error] " + e.message);
  }
};
</script>

</body>
</html>
"""
