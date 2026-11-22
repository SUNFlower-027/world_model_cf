"""List, download and inspect CarDreamer pre-trained checkpoints.

The official checkpoints live at
https://huggingface.co/ucd-dare/CarDreamer/tree/main/CarDreamer_checkpoints
and are Python pickles (see ``dreamerv3/embodied/core/checkpoint.py``), not
PyTorch files.  This module only needs ``numpy`` so it can run before the full
CARLA/JAX environment is installed.

Security note: ``pickle`` can execute arbitrary code while loading.  Only
inspect checkpoint files from the official repository or from your own runs.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

REPO_ID = "ucd-dare/CarDreamer"
SUBDIR = "CarDreamer_checkpoints"

#: Tasks shipped with a checkpoint in the official HuggingFace repository.
TASKS: tuple[str, ...] = (
    "four_lane",
    "lane_merge",
    "left_turn_hard",
    "left_turn_medium",
    "left_turn_simple",
    "navigation",
    "overtake",
    "right_turn_fov",
    "right_turn_hard",
    "right_turn_medium",
    "right_turn_raw",
    "right_turn_sfov",
    "right_turn_simple",
    "roundabout",
    "stop_sign",
    "traffic_light",
)

#: Expected size of one checkpoint in bytes (370,989,588 B ~= 354 MiB).
EXPECTED_BYTES = 370_989_588


def list_tasks() -> list[str]:
    """Return the checkpoint task names."""
    return list(TASKS)


def _endpoint() -> str:
    # Allow the usual HuggingFace mirror override, e.g. HF_ENDPOINT=https://hf-mirror.com
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def checkpoint_url(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(TASKS)}")
    return f"{_endpoint()}/{REPO_ID}/resolve/main/{SUBDIR}/{task}.ckpt"


def download(
    task: str,
    dest: str | Path = "checkpoints",
    force: bool = False,
    retries: int = 3,
) -> Path:
    """Download one checkpoint with resume support.

    Partial downloads are written to ``<name>.part`` and renamed on success, so
    an interrupted transfer can be resumed by calling this function again.
    """

    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"{task}.ckpt"
    partial = dest_dir / f"{task}.ckpt.part"

    if final.exists() and not force:
        size = final.stat().st_size
        if size == EXPECTED_BYTES:
            print(f"[ok] {final} already exists ({size:,} bytes)")
            return final
        print(f"[warn] {final} has unexpected size {size:,}; re-downloading")

    url = checkpoint_url(task)
    offset = partial.stat().st_size if partial.exists() and not force else 0
    if force and partial.exists():
        partial.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                if mode == "wb":
                    offset = 0
                total = EXPECTED_BYTES
                print(
                    f"[get] {task}: {url}\n"
                    f"      resuming at {offset:,} / {total:,} bytes -> {partial.name}"
                )
                with open(partial, mode) as handle:
                    downloaded = offset
                    last_report = time.time()
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_report > 5:
                            pct = 100.0 * downloaded / max(1, total)
                            print(f"      {downloaded / 2**20:8.1f} MiB ({pct:5.1f}%)")
                            last_report = time.time()
            size = partial.stat().st_size
            if size != EXPECTED_BYTES:
                raise IOError(f"size mismatch: got {size:,}, expected {EXPECTED_BYTES:,}")
            partial.replace(final)
            print(f"[ok] saved {final} ({size:,} bytes)")
            return final
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            offset = partial.stat().st_size if partial.exists() else 0
            print(f"[retry {attempt}/{retries}] {exc}")
            time.sleep(2 * attempt)

    raise RuntimeError(f"failed to download {task}") from last_error


class _Stub:
    """Placeholder for a class whose module is not installed (optax, jax...).

    Positional/keyword arguments from the pickle ``__reduce__`` are preserved so
    the tree walk can still reach the numpy arrays stored inside optimizer
    states such as ``optax.ScaleByAdamState(count, mu, nu)``.
    """

    def __init__(self, module: str, name: str, *args: Any, **kwargs: Any):
        self._module = module
        self._name = name
        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"<{self._module}.{self._name} stub>"


def _make_stub_class(module: str, name: str) -> type:
    """Create a real class that mimics a missing one for the unpickler.

    ``NEWOBJ`` requires a genuine ``type`` (not a factory), so we build a class
    whose ``__new__`` captures the constructor arguments into a stub payload.
    ``__init__`` is a no-op because the state is recorded in ``__new__``.
    """

    def __new__(cls, *args: Any, **kwargs: Any):  # noqa: D102
        self = object.__new__(cls)
        _Stub.__init__(self, module, name, *args, **kwargs)
        return self

    namespace = {
        "__new__": __new__,
        "__init__": lambda self, *args, **kwargs: None,
        "__module__": module,
    }
    return type(name, (_Stub,), namespace)


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler that substitutes missing classes with stubs instead of failing.

    Lets us inspect CarDreamer checkpoints (which embed optax/jax objects) in an
    environment that only has numpy installed.  Every substituted class is
    recorded so the summary can report what was stubbed.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.stubbed: set[str] = set()

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            self.stubbed.add(f"{module}.{name}")
            return _make_stub_class(module, name)


def _describe(value: Any) -> str:
    if isinstance(value, _Stub):
        return f"stub {value._module}.{value._name}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, (np.floating, np.integer, float, int, bool)):
        return f"{type(value).__name__} {value!r}"
    if isinstance(value, (bytes, str)):
        return f"{type(value).__name__} len={len(value)}"
    return type(value).__name__


def _walk(
    value: Any,
    prefix: str,
    depth: int,
    max_depth: int,
    max_entries: int,
    counter: list[int],
    stats: dict[str, int],
) -> None:
    if counter[0] >= max_entries:
        return
    if isinstance(value, _Stub):
        print(f"{'  ' * depth}{prefix or '<root>'}: <{value._module}.{value._name}>")
        counter[0] += 1
        for index, arg in enumerate(value._args):
            if counter[0] >= max_entries:
                return
            _walk(arg, f"{prefix}({index})", depth + 1, max_depth, max_entries, counter, stats)
        for key, child in value._kwargs.items():
            if counter[0] >= max_entries:
                return
            _walk(child, f"{prefix}.{key}", depth + 1, max_depth, max_entries, counter, stats)
        return
    if isinstance(value, np.ndarray):
        stats["arrays"] += 1
        stats["elements"] += int(value.size)
        stats["bytes"] += int(value.nbytes)
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            if key == "_timestamp":
                continue
            if counter[0] >= max_entries:
                print(f"{'  ' * depth}... (truncated, use --max-entries)")
                return
            child = value[key]
            label = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(child, (dict, list, tuple)) and depth < max_depth:
                n_child = len(child)
                print(f"{'  ' * depth}{label}/  ({n_child} entries)")
                counter[0] += 1
                _walk(child, label, depth + 1, max_depth, max_entries, counter, stats)
            else:
                print(f"{'  ' * depth}{label}: {_describe(child)}")
                counter[0] += 1
                _walk(child, label, depth + 1, max_depth, max_entries, counter, stats)
        return
    if isinstance(value, (list, tuple)):
        print(f"{'  ' * depth}{prefix or '<root>'}: {type(value).__name__} len={len(value)}")
        counter[0] += 1
        for index, child in enumerate(value):
            if counter[0] >= max_entries:
                return
            _walk(child, f"{prefix}[{index}]", depth + 1, max_depth, max_entries, counter, stats)
        return
    print(f"{'  ' * depth}{prefix or '<root>'}: {_describe(value)}")
    counter[0] += 1


def inspect_checkpoint(
    path: str | Path,
    max_depth: int = 4,
    max_entries: int = 200,
) -> dict[str, int]:
    """Load a CarDreamer checkpoint and print a compact tree summary."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as handle:
        unpickler = _TolerantUnpickler(handle)
        data = unpickler.load()

    print(f"[file] {path} ({path.stat().st_size:,} bytes)")
    if isinstance(data, dict) and "_timestamp" in data:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(data["_timestamp"])))
        print(f"[saved] {when}")
    if unpickler.stubbed:
        print(f"[stubbed] {', '.join(sorted(unpickler.stubbed))}")

    stats = {"arrays": 0, "elements": 0, "bytes": 0}
    _walk(data, "", 0, max_depth, max_entries, [0], stats)
    print(
        f"[stats] arrays={stats['arrays']:,} "
        f"elements={stats['elements']:,} "
        f"bytes={stats['bytes']:,} ({stats['bytes'] / 2**20:.1f} MiB)"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list available tasks")
    group.add_argument("--download", metavar="TASK", help="download one checkpoint")
    group.add_argument("--inspect", metavar="PATH", help="inspect a local .ckpt file")
    parser.add_argument("--dest", default="checkpoints", help="download directory")
    parser.add_argument("--force", action="store_true", help="re-download from scratch")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-entries", type=int, default=200)
    args = parser.parse_args(argv)

    if args.list:
        for name in TASKS:
            print(name)
        return 0

    if args.download:
        download(args.download, dest=args.dest, force=args.force)
        return 0

    inspect_checkpoint(args.inspect, max_depth=args.max_depth, max_entries=args.max_entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
