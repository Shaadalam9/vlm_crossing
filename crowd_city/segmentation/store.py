"""Persistent store for derived road-surface labels.

Only the derived index is retained, never the masks: for every crossing track
the store holds one row per sampled frame giving the surface under the
pedestrian's footpoint. That keeps the store small enough to sit beside the
Parquet detection store, at the cost of being tied to the crossing decision
that produced it. The manifest therefore fingerprints the crossing
configuration, and a change to it invalidates the index rather than silently
reusing labels sampled for a different set of tracks.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import polars as pl

import common
from custom_logger import CustomLogger


logger = CustomLogger(__name__)

# v2: detection frames are mapped to video time with the video's probed frame
# rate instead of the integer rate in the file name. v1 labels were read under
# boxes that drift by one frame every ~33 s on 29.97 fps footage, so they are
# re-segmented rather than reused.
INDEX_SCHEMA = "crowd_surface_index_v2"
INDEX_FOLDER = "index"
MANIFEST_FOLDER = "manifest"
URL_CACHE_FILE = "video_urls.json"

INDEX_COLUMNS = {
    "unique-id": pl.Utf8,
    "frame-count": pl.Int64,
    "video_time_s": pl.Float64,
    "surface": pl.Utf8,
    "surface_code": pl.Int8,
    "confidence": pl.Float32,
}


def configured_segmentation_root() -> Optional[str]:
    """Return the configured segmentation store root, if any."""
    try:
        value = common.get_configs("seg_data")
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        root = os.fspath(value)
    elif isinstance(value, Sequence):
        roots = [os.fspath(item) for item in value if str(item).strip()]
        if not roots:
            return None
        if len(roots) > 1:
            logger.warning(
                "More than one 'seg_data' root is configured; using the first "
                f"({roots[0]}). Detection file names are globally unique, so a "
                "single store is sufficient."
            )
        root = roots[0]
    else:
        return None
    return root if str(root).strip() else None


@dataclass(frozen=True)
class SegmentationSettings:
    """Everything that changes the meaning of a stored surface label."""

    model_identifier: str
    coarse_hz: float
    refine_hz: float
    footpoint_band_fraction: float
    footpoint_width_fraction: float
    minimum_confidence: float
    crossing_fingerprint: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_identifier": self.model_identifier,
            "coarse_hz": float(self.coarse_hz),
            "refine_hz": float(self.refine_hz),
            "footpoint_band_fraction": float(self.footpoint_band_fraction),
            "footpoint_width_fraction": float(self.footpoint_width_fraction),
            "minimum_confidence": float(self.minimum_confidence),
            "crossing_fingerprint": self.crossing_fingerprint,
        }


def crossing_fingerprint(parameters: Dict[str, Any]) -> str:
    """Return a stable hash of the crossing configuration."""
    payload = json.dumps(parameters, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class SurfaceStore:
    """Read and write the derived surface index."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser()
        self.index_dir = self.root / INDEX_FOLDER
        self.manifest_dir = self.root / MANIFEST_FOLDER
        self._url_cache: Optional[Dict[str, str]] = None

    def ensure_directories(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        """Return whether the store root exists or can be created."""
        try:
            self.ensure_directories()
        except OSError as error:
            logger.warning(f"Segmentation store {self.root} is unavailable: {error}")
            return False
        return True

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def index_path(self, stem: str) -> Path:
        return self.index_dir / f"{stem}.parquet"

    def manifest_path(self, stem: str) -> Path:
        return self.manifest_dir / f"{stem}.json"

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def read_manifest(self, stem: str) -> Optional[Dict[str, Any]]:
        path = self.manifest_path(stem)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning(f"Unreadable segmentation manifest {path}: {error}")
            return None

    def write_manifest(
        self,
        stem: str,
        settings: SegmentationSettings,
        track_ids: Iterable[Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        manifest = {
            "schema": INDEX_SCHEMA,
            "settings": settings.as_dict(),
            "track_ids": sorted({str(value) for value in track_ids}),
        }
        if extra:
            manifest.update(extra)
        path = self.manifest_path(stem)
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        temporary.replace(path)

    def is_current(
        self,
        stem: str,
        settings: SegmentationSettings,
        required_track_ids: Iterable[Any],
    ) -> bool:
        """Return whether a stored index covers this request unchanged."""
        manifest = self.read_manifest(stem)
        if manifest is None or not self.index_path(stem).is_file():
            return False
        if manifest.get("schema") != INDEX_SCHEMA:
            return False
        if manifest.get("settings") != settings.as_dict():
            return False
        stored = set(manifest.get("track_ids") or [])
        needed = {str(value) for value in required_track_ids}
        return needed.issubset(stored)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    def read_index(self, stem: str) -> Optional[pl.DataFrame]:
        path = self.index_path(stem)
        if not path.is_file():
            return None
        try:
            return pl.read_parquet(path)
        except Exception as error:
            logger.warning(f"Unreadable surface index {path}: {error}")
            return None

    def write_index(self, stem: str, rows: List[Dict[str, Any]]) -> None:
        frame = (
            pl.DataFrame(rows, schema=INDEX_COLUMNS)
            if rows
            else pl.DataFrame(schema=INDEX_COLUMNS)
        )
        path = self.index_path(stem)
        temporary = path.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(path)

    # ------------------------------------------------------------------
    # Resolved video URLs
    # ------------------------------------------------------------------
    def _url_cache_path(self) -> Path:
        return self.root / URL_CACHE_FILE

    def cached_url(self, video_id: str) -> Optional[str]:
        if self._url_cache is None:
            path = self._url_cache_path()
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        self._url_cache = dict(json.load(handle))
                except (OSError, ValueError):
                    self._url_cache = {}
            else:
                self._url_cache = {}
        return self._url_cache.get(str(video_id))

    def store_url(self, video_id: str, url: str) -> None:
        if self._url_cache is None:
            self.cached_url(video_id)
        assert self._url_cache is not None
        self._url_cache[str(video_id)] = str(url)
        path = self._url_cache_path()
        temporary = path.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self._url_cache, handle, indent=2, sort_keys=True)
            temporary.replace(path)
        except OSError as error:
            logger.warning(f"Could not persist resolved video URLs: {error}")
