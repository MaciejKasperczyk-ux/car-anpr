import asyncio
import shlex
import time
import uuid
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Deque
from collections import deque


@dataclass
class Run:
    run_id: str
    state: str = "queued"
    exit_code: Optional[int] = None
    started_at: float = field(default_factory=lambda: time.time())
    finished_at: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    process: Optional[asyncio.subprocess.Process] = None
    lines_ring: Deque[str] = field(default_factory=lambda: deque(maxlen=5000))


class Runner:
    def __init__(self, workdir: str, ring_max_lines: int = 5000) -> None:
        self._workdir = workdir
        self._runs: Dict[str, Run] = {}
        self._lock = asyncio.Lock()
        self._ring_max_lines = ring_max_lines

    async def create_run(self, cmd: str, env: Optional[Dict[str, str]] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        run_id = str(uuid.uuid4())
        run = Run(run_id=run_id, state="queued")
        run.lines_ring = deque(maxlen=self._ring_max_lines)
        if meta:
            run.meta.update(meta)

        async with self._lock:
            self._runs[run_id] = run

        asyncio.create_task(self._run_process(run, cmd=cmd, env=env or {}))
        return run_id

    async def get_run(self, run_id: str) -> Optional[Run]:
        async with self._lock:
            return self._runs.get(run_id)

    async def list_runs(self) -> Dict[str, Run]:
        async with self._lock:
            return dict(self._runs)

    async def snapshot_lines(self, run_id: str) -> Optional[list[str]]:
        run = await self.get_run(run_id)
        if run is None:
            return None
        return list(run.lines_ring)

    async def _push_line(self, run: Run, line: str) -> None:
        run.lines_ring.append(line)

    async def _pump_stream(self, run: Run, stream: asyncio.StreamReader, prefix: str) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            await self._push_line(run, f"{prefix}{text}")

    async def _run_process(self, run: Run, cmd: str, env: Dict[str, str]) -> None:
        run.state = "running"
        run.started_at = time.time()

        await self._push_line(run, f"[run_id={run.run_id}] started cmd={cmd}")

        args = shlex.split(cmd)

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=self._workdir,
                env={**os.environ, **env},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            run.exit_code = 127
            run.finished_at = time.time()
            run.state = "failed"
            await self._push_line(run, f"[run_id={run.run_id}] failed to start: {type(e).__name__}: {e}")
            return

        run.process = proc

        assert proc.stdout is not None
        assert proc.stderr is not None

        t1 = asyncio.create_task(self._pump_stream(run, proc.stdout, prefix=""))
        t2 = asyncio.create_task(self._pump_stream(run, proc.stderr, prefix="[stderr] "))

        exit_code = await proc.wait()
        await asyncio.gather(t1, t2, return_exceptions=True)

        run.exit_code = exit_code
        run.finished_at = time.time()
        run.state = "finished" if exit_code == 0 else "failed"
        await self._push_line(run, f"[run_id={run.run_id}] finished exit_code={exit_code} state={run.state}")
