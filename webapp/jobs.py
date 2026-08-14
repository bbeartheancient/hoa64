"""JobManager — thread-pool infrastructure for CPU-bound work.

Phase 1 endpoints are all fast and synchronous, but the search engines
of Phase 2 (micromag SA, tile SA, Williamson/GS PSD minimization, RNN
guidance) run for minutes to hours and must not block the event loop.
A Job is a small record with a `cancel` Event (engines poll it between
sweeps) and an unbounded `progress` Queue (producers never block;
consumers drain it into WebSocket messages).  The wrapper thread
catches exceptions into `job.error` and always posts a terminal
`{"type": "end", "status": ...}` message so clients can stop listening.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

_KEPT = 64  # bounded job history
_HISTORY = 500  # bounded per-job progress replay buffer


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"  # queued|running|done|error|cancelled
    created: float = field(default_factory=time.time)
    finished: float | None = None
    result: Any = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    progress: queue.Queue = field(default_factory=queue.Queue)
    history: deque = field(default_factory=lambda: deque(maxlen=_HISTORY))
    matrix: Any = None  # np.ndarray kept off the JSON result (see routes_search)


def report(job: Job, **fields) -> None:
    """Push a progress update; producers must not block (unbounded queue).

    Every message is also appended to `job.history` (bounded deque) so a
    WebSocket client that connects mid-run can replay what it missed.
    """
    msg = {"type": "progress", "t": time.time(), **fields}
    job.progress.put(msg)
    job.history.append(msg)


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Job], Any], params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:8], kind=kind, params=dict(params))
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > _KEPT:
                self._jobs.popitem(last=False)

        def _run() -> None:
            job.status = "running"
            try:
                job.result = fn(job)
                if job.cancel.is_set():
                    job.status = "cancelled"
                else:
                    job.status = "done"
            except Exception as e:  # noqa: BLE001 — errors belong on the job
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"
            finally:
                job.finished = time.time()
                end = {"type": "end", "status": job.status}
                job.progress.put(end)
                job.history.append(end)

        self._pool.submit(_run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.cancel.set()
        return True

    def list(self) -> list[Job]:
        with self._lock:
            return list(reversed(self._jobs.values()))


JOBS = JobManager()
