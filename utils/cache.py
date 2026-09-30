"""Per-video result cache so analysis.py only recomputes what changed.

Every analysed video gets one JSON file, <output>/analysis/<city>/<video>.json,
holding two stages:

- "detection": the accepted crossings (track id and frame bounds) and the
  candidate ids, keyed on the tracking CSV, the video fps, the road strip and
  the code of utils/crossing/detection.py.
- "metrics": one row per crossing, keyed on the detection *result* (not its
  key, so a detection change that finds the same crossings keeps the metrics),
  the video geometry, the hesitation and stature settings and the code of
  utils/crossing/metrics.py.

A stage is reused when its stored key equals the key computed now. Code is
fingerprinted from its syntax tree with docstrings removed, so editing
comments, docstrings or formatting does not invalidate anything. Limits,
summaries and figures are cheap and are always rebuilt from the cached rows.
"""

import ast
import hashlib
import inspect
import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

# Bump when the layout of the cache file or the glue in analysis.py changes.
CACHE_SCHEMA = 1


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def source_fingerprint(source: str) -> str:
    """Hash of Python source that ignores comments, docstrings and formatting."""
    tree = _strip_docstrings(ast.parse(source))
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def file_fingerprint(path: str) -> str:
    """source_fingerprint of a Python file, computed once per process."""
    with open(path) as f:
        return source_fingerprint(f.read())


def object_fingerprint(obj: Any) -> str:
    """source_fingerprint of a function or class."""
    import textwrap
    return source_fingerprint(textwrap.dedent(inspect.getsource(obj)))


def input_fingerprint(path: str) -> Dict[str, int]:
    """Size and modification time of an input file. Tracking replaces CSVs atomically, so both change on a redo."""
    stat = os.stat(path)
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def make_key(**parts: Any) -> str:
    """Stable hash of JSON-serialisable parts."""
    payload = json.dumps({"schema": CACHE_SCHEMA, **parts}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load(path: str) -> Dict[str, Any]:
    """Read a cache file; a missing or corrupt file is an empty cache."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and data.get("schema") == CACHE_SCHEMA else {}


def stage(cache: Dict[str, Any], name: str, key: str) -> Optional[Dict[str, Any]]:
    """Return the cached stage when its key matches, else None."""
    entry = cache.get(name)
    if isinstance(entry, dict) and entry.get("key") == key:
        return entry
    return None


def save(path: str, cache: Dict[str, Any]) -> None:
    """Write a cache file atomically, so an interrupted run never leaves a half-written file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({**cache, "schema": CACHE_SCHEMA}, f, indent=1, default=str)
    os.replace(tmp, path)
