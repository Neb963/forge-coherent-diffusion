"""P07 privacy-bounded, rotating live diagnostics for installed Forge Neo.

No prompts, CRF documents, weights, request bodies, tokens, or traceback
arguments are captured. Logs persist across UI refreshes and process restarts.
The Gradio probe is read-only: it re-raises every original exception unchanged.
"""
from __future__ import annotations

import functools
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
})
_INSTALLED = False


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
