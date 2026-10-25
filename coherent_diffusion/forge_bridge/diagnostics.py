"""P07 privacy-bounded, rotating live diagnostics for installed Forge Neo.

No prompts, CRF documents, weights, request bodies, tokens, or traceback
arguments are captured. Logs persist across UI refreshes and process restarts.
The Gradio probe is read-only: it re-raises every original exception unchanged.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
from collections import deque
from datetime import datetime, timezone

_LOCK = threading.RLock()
_EVENTS = deque(maxlen=400)
_MAX_FILE_BYTES = 3 * 1024 * 1024
_ALLOWED = frozenset({
    "phase", "family", "mode", "width", "height", "tile", "steps",
    "type", "backend_fn", "fn_index", "targets", "kind",
    "elapsed_ms", "success", "partial_image", "model_type",
    "route", "api_name", "pid", "component_ids", "handler",
    "stage", "tile", "evaluation", "upper_bound", "completed", "step",
    "step_total", "region_states", "window_total",
})
_INSTALLED = False
_PROGRESS = {"active": False, "status": "idle", "completed": 0, "evaluation": 0,
             "upper_bound": 0, "stage": None, "tile": None, "start": None,
             "last": None, "sigma": None, "step": 0}


def start_progress(*, upper_bound: int, window_total: int,
                   region_states: int) -> None:
    """Upper bound counts tile/state evaluations, not true model forwards."""
    with _LOCK:
        _PROGRESS.update(active=True, status="starting", completed=0, evaluation=0,
                         upper_bound=max(0,int(upper_bound)), stage=None,
                         tile=None, step=0, sigma=None, start=time.monotonic(),
                         last=time.monotonic())
    record("progress_plan", upper_bound=upper_bound, window_total=window_total,
           region_states=region_states)


def finish_progress(*, success: bool) -> None:
    with _LOCK:
        _PROGRESS["active"] = False
        _PROGRESS["status"] = "complete" if success else "stopped_or_error"
        _PROGRESS["last"] = time.monotonic()
    record("progress_finished", success=success,
           evaluation=_PROGRESS["evaluation"], completed=_PROGRESS["completed"])


def step_progress(stage: str, info) -> None:
    record("step_completed", stage=stage, step=info.index+1,
           window_total=info.windows)


@contextmanager
def timed_prediction(*, stage: str, tile: int, sigma: float):
    """Capture what the denoiser is doing before a potentially slow GPU call."""
    begun = time.monotonic()
    with _LOCK:
        if _PROGRESS.get("sigma") != (stage, sigma):
            prior = _PROGRESS.get("sigma")
            if prior is None or prior[0] != stage:
                _PROGRESS["step"] = 0
            _PROGRESS["sigma"] = (stage, sigma)
            _PROGRESS["step"] = _PROGRESS.get("step",0)+1
        _PROGRESS["evaluation"] = _PROGRESS.get("evaluation",0)+1
        i = _PROGRESS["evaluation"]
        step = _PROGRESS["step"]
        _PROGRESS.update(stage=stage, tile=tile, status="prediction_started",
                         last=begun)
    record("prediction_begin", evaluation=i, stage=stage, tile=tile, step=step)
    try:
        yield
    except BaseException as exc:
        record("prediction_failure", evaluation=i, stage=stage, tile=tile,
               type=type(exc).__name__, elapsed_ms=(time.monotonic()-begun)*1000)
        raise
    else:
        duration=(time.monotonic()-begun)*1000
        with _LOCK:
            _PROGRESS["completed"]=_PROGRESS.get("completed",0)+1
            n=_PROGRESS["completed"]
            _PROGRESS.update(last=time.monotonic(), status="prediction_completed")
        record("prediction_end", evaluation=i, completed=n, stage=stage,
               tile=tile, step=step, elapsed_ms=duration)
        # Console progress at a bounded cadence; server panel logs EVERY eval.
        if i == 1 or i % 10 == 0 or duration >= 30000:
            print("[Coherent Diffusion] prediction {0} completed: stage={1} "
                  "step={2} tile={3}, last={4:.1f}s, upper-bound={5}".format(
                      n, stage, step, tile, duration/1000,
                      _PROGRESS["upper_bound"]), flush=True)


def status_text() -> str:
    with _LOCK:
        p=dict(_PROGRESS)
    if p["start"] is None:
        return "Idle — no coordinated generation started this process"
    now=time.monotonic()
    elapsed=round(now-p["start"],1)
    ago=round(now-p["last"],1) if p["last"] is not None else 0
    state="RUNNING" if p["active"] else p["status"].upper()
    upper=p["upper_bound"]
    approximate=f" / at most {upper} evaluations" if upper else ""
    return (f"{state} | {p['completed']} finished{approximate} | "
            f"stage={p['stage']} | step={p['step']} | tile={p['tile']} | "
            f"last={p['status']} {ago}s ago | elapsed={elapsed}s")


def _data_dir() -> Path:
    try:
        from modules import paths
        root = Path(paths.data_path)
    except (ImportError, AttributeError, TypeError):
        root = Path.home() / ".cache" / "forge-coherent-diffusion"
    target = root / "outputs" / "coherent-diffusion" / "diagnostics"
    target.mkdir(parents=True, exist_ok=True)
    return target


def diagnostic_file() -> Path:
    return _data_dir() / "events.jsonl"


def _clean(value):
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        return round(value, 3) if abs(value) < 1e15 else None
    if isinstance(value, str):
        # Event identifiers are internal program metadata, never user text.
        return value[:100] if all(ch.isalnum() or ch in "_-.:/" for ch in value[:100]) else "[redacted]"
    if isinstance(value, (list, tuple)):
        return [_clean(x) for x in value[:12]]
    return type(value).__name__


def record(event: str, **fields) -> None:
    """Append an allowlisted record without interrupting normal Forge execution."""
    try:
        if not isinstance(event, str) or not event.replace("_", "").isalnum():
            return
        row = {"at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "event": event[:64],
               "pid": os.getpid()}
        row.update({key: _clean(value) for key, value in fields.items()
                    if key in _ALLOWED})
        line = json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
        with _LOCK:
            _EVENTS.append(row)
            if _PROGRESS["active"] and event in (
                    "conditioning_begin", "conditioning_done",
                    "model_prepare_begin", "model_prepare_done",
                    "generation_stage", "decode_begin", "decode_done"):
                _PROGRESS["status"] = event
                _PROGRESS["last"] = time.monotonic()
                if "stage" in row: _PROGRESS["stage"] = row["stage"]
                if "tile" in row: _PROGRESS["tile"] = row["tile"]
            path = diagnostic_file()
            if path.exists() and path.stat().st_size + len(line.encode()) > _MAX_FILE_BYTES:
                previous = path.with_suffix(".jsonl.1")
                os.replace(path, previous)
            with path.open("a", encoding="utf-8") as output:
                output.write(line)
    except (OSError, ValueError, TypeError):
        # Diagnostics are non-invasive, even on read-only filesystem or full disk.
        return


def recent_text(limit=120) -> str:
    try:
        limit = max(1, min(int(limit), 400))
        with _LOCK:
            path = diagnostic_file()
            # Reopen persistent logs on extension restart; bound file size and
            # number of lines rather than exposing unrelated system logs.
            lines = path.read_text(encoding="utf-8").splitlines()[-limit:] if path.exists() else []
        return "\n".join(lines)
    except (OSError, ValueError):
        return "Diagnostic log temporarily unavailable"


def _missing_handler(self, fn):
    try:
        index = next((int(i) for i, candidate in self.fns.items()
                      if candidate is fn), None)
        dep = next((item for item in self.get_config_file().get("dependencies", [])
                    if item.get("id") == index), {})
        targets = dep.get("targets") or []
        record("gradio_missing_backend", fn_index=index, backend_fn=dep.get("backend_fn"),
               handler=type(fn).__name__,
               api_name=dep.get("api_name"),
               targets=[str(t[0]) + ":" + str(t[1]) for t in targets[:8]]
               if isinstance(targets, list) else [])
    except Exception:
        record("gradio_missing_backend", handler=type(fn).__name__)


def install_gradio_probe() -> bool:
    """Observe failures in Gradio's global dispatcher; no event is changed.

    Gradio 4.40 call_function(self, block_fn, processed_input, ...). A genuine
    frontend-only dependency has fn=None; dispatching it to Python causes the
    exact error seen on the user's Forge. The wrapper re-raises with the same
    exception/traceback, records only ID/targets, and installs once.
    """
    global _INSTALLED
    try:
        import gradio as gr
        blocks = gr.Blocks
        with _LOCK:
            if getattr(blocks.call_function, "_cd_diagnostic_wrapper", False):
                _INSTALLED = True
                return True
            original = blocks.call_function
            if not callable(original):
                return False

            @functools.wraps(original)
            async def observed(self, block_fn, *args, **kwargs):
                try:
                    return await original(self, block_fn, *args, **kwargs)
                except IndexError as exc:
                    if str(exc) == "function has no backend method.":
                        _missing_handler(self, block_fn)
                    raise

            observed._cd_diagnostic_wrapper = True
            blocks.call_function = observed
            _INSTALLED = True
        record("diagnostics_ready", kind="gradio_dispatch", type=str(getattr(gr, "__version__", "?")))
        return True
    except (ImportError, AttributeError, TypeError, OSError):
        return False
