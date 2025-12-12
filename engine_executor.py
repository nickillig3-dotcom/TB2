from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

import multiprocessing as mp

from engine_config import ExecutionConfig

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ExecStats:
    mode: str  # single|threads|processes
    n_jobs: int
    n_tasks: int
    had_fallback: bool
    fallback_reason: Optional[str]


def _task_id(task: Any) -> str:
    if hasattr(task, "task_id"):
        return str(getattr(task, "task_id"))
    if isinstance(task, dict) and "task_id" in task:
        return str(task["task_id"])
    raise ValueError("Task must have a task_id attribute/key")


def _looks_like_pickling_error(exc: BaseException) -> bool:
    msg = (repr(exc) + " " + str(exc)).lower()
    return ("pickle" in msg) or ("pickling" in msg) or ("can't pickle" in msg) or ("cannot pickle" in msg)


def execute_tasks(
    worker_fn: Callable[[T], R],
    tasks: Sequence[T],
    exec_cfg: ExecutionConfig,
    *,
    process_initializer: Optional[Callable[..., Any]] = None,
    process_initargs: Sequence[Any] = (),
) -> tuple[List[R], ExecStats]:
    """
    Execute tasks with optional parallelism and graceful fallback.

    - Deterministic output order: results aligned with input task order.
    - Graceful degrade:
        n_jobs <= 1 => single
        processes fail => threads => single
    - Optional process initializer: set global worker context once (avoid per-task dataframe pickling).
    """
    n_tasks = int(len(tasks))
    n_jobs = int(exec_cfg.n_jobs)

    if n_tasks == 0:
        return [], ExecStats(mode="single", n_jobs=0, n_tasks=0, had_fallback=False, fallback_reason=None)

    if n_jobs <= 1:
        results: List[R] = []
        for t in tasks:
            try:
                results.append(worker_fn(t))
            except Exception as e:
                results.append({"task_id": _task_id(t), "status": "error", "error": repr(e)})  # type: ignore
        return results, ExecStats(mode="single", n_jobs=1, n_tasks=n_tasks, had_fallback=False, fallback_reason=None)

    prefer_processes = bool(exec_cfg.prefer_processes)

    modes = ["processes", "threads"] if prefer_processes else ["threads"]
    last_reason: Optional[str] = None
    had_fallback = False

    for mode in modes:
        try:
            if mode == "threads":
                res = _run_threadpool(worker_fn, tasks, n_jobs=max(1, n_jobs))
                return res, ExecStats(
                    mode="threads", n_jobs=n_jobs, n_tasks=n_tasks, had_fallback=had_fallback, fallback_reason=last_reason
                )

            if mode == "processes":
                # spawn is cross-platform safe (Windows default)
                ctx = mp.get_context("spawn")
                res = _run_processpool(
                    worker_fn,
                    tasks,
                    n_jobs=max(1, n_jobs),
                    mp_context=ctx,
                    initializer=process_initializer,
                    initargs=process_initargs,
                )
                return res, ExecStats(
                    mode="processes", n_jobs=n_jobs, n_tasks=n_tasks, had_fallback=had_fallback, fallback_reason=last_reason
                )

            raise RuntimeError(f"Unknown mode: {mode}")

        except Exception as e:
            had_fallback = True
            last_reason = f"{mode} failed: {repr(e)}"
            continue

    # Final fallback: single
    results: List[R] = []
    for t in tasks:
        try:
            results.append(worker_fn(t))
        except Exception as e:
            results.append({"task_id": _task_id(t), "status": "error", "error": repr(e)})  # type: ignore

    return results, ExecStats(mode="single", n_jobs=1, n_tasks=n_tasks, had_fallback=True, fallback_reason=last_reason)


def _run_threadpool(worker_fn: Callable[[T], R], tasks: Sequence[T], n_jobs: int) -> List[R]:
    order = [_task_id(t) for t in tasks]
    out: Dict[str, R] = {}

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(worker_fn, t): _task_id(t) for t in tasks}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"task_id": tid, "status": "error", "error": repr(e)}  # type: ignore
            out[tid] = res

    return [out[tid] for tid in order]


def _run_processpool(
    worker_fn: Callable[[T], R],
    tasks: Sequence[T],
    n_jobs: int,
    mp_context: Any,
    initializer: Optional[Callable[..., Any]],
    initargs: Sequence[Any],
) -> List[R]:
    order = [_task_id(t) for t in tasks]
    out: Dict[str, R] = {}

    try:
        from concurrent.futures.process import BrokenProcessPool
    except Exception:  # pragma: no cover
        BrokenProcessPool = RuntimeError  # type: ignore

    with ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=mp_context,
        initializer=initializer,
        initargs=tuple(initargs),
    ) as ex:
        futs = {ex.submit(worker_fn, t): _task_id(t) for t in tasks}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                res = fut.result()
            except BrokenProcessPool:
                # catastrophic => fallback
                raise
            except Exception as e:
                # pickling issues => fallback to threads
                if _looks_like_pickling_error(e):
                    raise
                res = {"task_id": tid, "status": "error", "error": repr(e)}  # type: ignore
            out[tid] = res

    return [out[tid] for tid in order]
