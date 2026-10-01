"""Turn SegFormer label maps into per-track surface timelines and intervals.

Two things are derived here. First, the surface under a pedestrian's feet in
each sampled frame, read from a small band around the bottom edge of the
bounding box rather than a single pixel, because a single footpoint lands on
the pedestrian's own shoes as often as on the ground. Second, the interval in
which the pedestrian is actually on the carriageway, which is what the crossing
speed and the hesitation time are measured against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from crowd_city.segmentation.constants import (
    NON_GROUND_TRAIN_IDS,
    OCCLUDING_TRAIN_IDS,
    SURFACE_ROAD,
    SURFACE_TO_CODE,
    SURFACE_UNKNOWN,
    surface_for_train_id,
)


# Fraction of the bounding-box height sampled above and below its bottom edge.
DEFAULT_FOOTPOINT_BAND_FRACTION = 0.06
# Fraction of the bounding-box width sampled either side of its centre.
DEFAULT_FOOTPOINT_WIDTH_FRACTION = 0.25
# Minimum SegFormer confidence for a pixel to contribute to the majority vote.
DEFAULT_MINIMUM_CONFIDENCE = 0.50
# Non-road samples tolerated inside an otherwise continuous on-road run.
DEFAULT_ROAD_GAP_TOLERANCE = 1


@dataclass(frozen=True)
class SurfaceSample:
    """One sampled frame of one track."""

    track_id: str
    frame: int
    video_time_s: float
    surface: str
    confidence: float

    def as_row(self) -> Dict[str, object]:
        return {
            "unique-id": self.track_id,
            "frame-count": int(self.frame),
            "video_time_s": float(self.video_time_s),
            "surface": self.surface,
            "surface_code": int(SURFACE_TO_CODE[self.surface]),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class RoadInterval:
    """The frames in which a track is on the carriageway."""

    entry_frame: int
    exit_frame: int
    road_samples: int
    total_samples: int


def sample_surface(
    labels: np.ndarray,
    confidence: np.ndarray,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    band_fraction: float = DEFAULT_FOOTPOINT_BAND_FRACTION,
    width_fraction: float = DEFAULT_FOOTPOINT_WIDTH_FRACTION,
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
) -> Tuple[str, float]:
    """Return the surface category under one bounding box and its confidence.

    ``labels`` and ``confidence`` are one SegFormer output grid. Normalised
    bounding-box coordinates address it directly, so the grid resolution never
    needs to match the source video.
    """
    if labels.ndim != 2 or labels.size == 0:
        return SURFACE_UNKNOWN, 0.0

    rows, columns = labels.shape
    bottom = float(y_center) + float(height) / 2.0
    band = max(float(band_fraction) * float(height), 1.0 / max(rows, 1))
    half_width = max(float(width_fraction) * float(width), 1.0 / max(columns, 1))

    row_low = int(np.clip((bottom - band) * (rows - 1), 0, rows - 1))
    row_high = int(np.clip((bottom + band) * (rows - 1), 0, rows - 1))
    column_low = int(np.clip((float(x_center) - half_width) * (columns - 1), 0, columns - 1))
    column_high = int(np.clip((float(x_center) + half_width) * (columns - 1), 0, columns - 1))

    patch_labels = labels[row_low:row_high + 1, column_low:column_high + 1].reshape(-1)
    patch_confidence = confidence[row_low:row_high + 1, column_low:column_high + 1].reshape(-1)
    if patch_labels.size == 0:
        return SURFACE_UNKNOWN, 0.0

    keep = patch_confidence >= float(minimum_confidence)
    for train_id in OCCLUDING_TRAIN_IDS | NON_GROUND_TRAIN_IDS:
        keep &= patch_labels != train_id

    kept_labels = patch_labels[keep]
    if kept_labels.size == 0:
        return SURFACE_UNKNOWN, 0.0

    values, counts = np.unique(kept_labels, return_counts=True)
    winner = int(values[int(np.argmax(counts))])
    winner_confidence = float(np.mean(patch_confidence[keep][kept_labels == winner]))
    return surface_for_train_id(winner), winner_confidence


def road_interval(
    frames: Sequence[int],
    surfaces: Sequence[str],
    gap_tolerance: int = DEFAULT_ROAD_GAP_TOLERANCE,
) -> Optional[RoadInterval]:
    """Return the longest on-road run, tolerating short interruptions.

    A single mis-sampled frame in the middle of a crossing should not split the
    crossing in two, so up to ``gap_tolerance`` consecutive non-road samples are
    absorbed into a run. Unknown samples never start or end a run.
    """
    if not frames or len(frames) != len(surfaces):
        return None

    order = np.argsort(np.asarray(frames, dtype=np.int64), kind="stable")
    ordered_frames = [int(frames[index]) for index in order]
    ordered_surfaces = [str(surfaces[index]) for index in order]

    best: Optional[Tuple[int, int, int]] = None
    start_index: Optional[int] = None
    last_road_index: Optional[int] = None
    road_count = 0
    pending_gap = 0

    for index, surface in enumerate(ordered_surfaces):
        if surface == SURFACE_ROAD:
            if start_index is None:
                start_index = index
                road_count = 0
            last_road_index = index
            road_count += 1
            pending_gap = 0
            continue

        if start_index is None:
            continue

        if surface == SURFACE_UNKNOWN:
            # Absorbed without counting against the tolerance: an unknown
            # sample is missing evidence, not evidence of leaving the road.
            continue

        pending_gap += 1
        if pending_gap > int(gap_tolerance):
            if best is None or road_count > best[2]:
                best = (start_index, last_road_index or start_index, road_count)
            start_index = None
            last_road_index = None
            road_count = 0
            pending_gap = 0

    if start_index is not None and (best is None or road_count > best[2]):
        best = (start_index, last_road_index or start_index, road_count)

    if best is None:
        return None

    return RoadInterval(
        entry_frame=ordered_frames[best[0]],
        exit_frame=ordered_frames[best[1]],
        road_samples=best[2],
        total_samples=len(ordered_frames),
    )


def surface_timeline(samples: Sequence[SurfaceSample]) -> Dict[str, List[SurfaceSample]]:
    """Group samples by track, each list sorted by frame."""
    grouped: Dict[str, List[SurfaceSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.track_id, []).append(sample)
    for values in grouped.values():
        values.sort(key=lambda item: item.frame)
    return grouped


def transition_brackets(
    frames: Sequence[int],
    surfaces: Sequence[str],
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Return the frame ranges that contain the road entry and road exit.

    A coarse pass only has to establish the shape of a track: off the road,
    then on it, then off it again. The exact moment of each change lies
    somewhere between the last sample of one kind and the first sample of the
    next, and that is the only place a finer pass needs to look.

    Unknown samples are skipped rather than treated as off-road, so an
    occluded moment widens the bracket instead of inventing a transition
    inside it. Returns ``(entry_bracket, exit_bracket)``, either of which is
    ``None`` when the track shows no such change.
    """
    if not frames or len(frames) != len(surfaces):
        return None, None

    order = np.argsort(np.asarray(frames, dtype=np.int64), kind="stable")
    known = [
        (int(frames[index]), str(surfaces[index]))
        for index in order
        if str(surfaces[index]) != SURFACE_UNKNOWN
    ]
    if not known:
        return None, None

    entry: Optional[Tuple[int, int]] = None
    exit_bracket: Optional[Tuple[int, int]] = None
    seen_road = False

    for position in range(1, len(known)):
        previous_frame, previous_surface = known[position - 1]
        current_frame, current_surface = known[position]

        if (
            entry is None
            and previous_surface != SURFACE_ROAD
            and current_surface == SURFACE_ROAD
        ):
            entry = (previous_frame, current_frame)

        if current_surface == SURFACE_ROAD:
            seen_road = True
        elif seen_road and previous_surface == SURFACE_ROAD:
            exit_bracket = (previous_frame, current_frame)

    return entry, exit_bracket
