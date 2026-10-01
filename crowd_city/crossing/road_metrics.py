"""Crossing metrics measured against the segmented carriageway.

The baseline metrics infer both quantities from bounding-box motion alone: the
hesitation time is the leading run of near-stationary samples in a track, and
the speed is fitted over the whole track. Neither knows where the kerb is.

Here the road-surface timeline supplies that boundary. The hesitation time
becomes the stationary interval immediately before the pedestrian steps onto
the carriageway, and the speed is fitted only over the frames spent on it.
Everything else, including the frozen Waymo speed model, is unchanged, so the
two sets of values stay directly comparable.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import polars as pl

import crowd_city.crossing.metrics as crossing_metrics
from custom_logger import CustomLogger
from crowd_city.segmentation.surface import RoadInterval, road_interval


logger = CustomLogger(__name__)

# Stationarity margin, expressed as a fraction of the median bounding-box
# height. Identical to the baseline metric so the two are comparable.
STATIONARY_MARGIN_FRACTION = 0.10
# Consecutive stationary samples required before a wait is reported at all.
MINIMUM_STATIONARY_SAMPLES = 3


def road_intervals_for_tracks(
    timelines: Mapping[str, List[Any]],
    gap_tolerance: int = 1,
) -> Dict[str, RoadInterval]:
    """Derive the on-road interval of every track that has one."""
    intervals: Dict[str, RoadInterval] = {}
    for track_id, samples in (timelines or {}).items():
        if not samples:
            continue
        interval = road_interval(
            [sample.frame for sample in samples],
            [sample.surface for sample in samples],
            gap_tolerance=gap_tolerance,
        )
        if interval is not None:
            intervals[str(track_id)] = interval
    return intervals


# ---------------------------------------------------------------------
# Hesitation time
# ---------------------------------------------------------------------

def hesitation_seconds(
    track: pl.DataFrame,
    entry_frame: int,
    fps: float,
    checks_per_second: float,
) -> Optional[float]:
    """Return the stationary interval ending when the pedestrian enters the road.

    The track is walked backwards from the frame at which the pedestrian first
    stands on the carriageway. Samples are taken at ``checks_per_second`` and
    counted while the horizontal displacement between consecutive samples stays
    within the stationarity margin. The count stops at the first sample that
    shows real movement, so an approach on foot is not counted as waiting.

    Returns ``None`` when the wait cannot be observed at all, which happens
    whenever the track begins with the pedestrian already on the carriageway.
    On real footage that is common: a dashcam frequently first sees someone
    mid-crossing. Such a track carries no information about waiting, and
    recording it as a zero would pull every city mean towards zero in
    proportion to how often the camera arrives late rather than to how little
    anyone waited. A genuine zero, meaning the pedestrian walked up and
    stepped straight out without stopping, is reported as ``0.0``; the
    segmentation pass leaves those out of the hesitation averages too, as
    the bounding-box metric does.
    """
    required = {"frame-count", "x-center", "height"}
    if track is None or track.height == 0 or not required.issubset(set(track.columns)):
        return None
    if fps <= 0 or checks_per_second <= 0:
        return None

    step = max(1, int(round(float(fps) / float(checks_per_second))))
    ordered = track.sort("frame-count")
    frames = ordered.get_column("frame-count").cast(pl.Int64, strict=False).to_numpy()
    x_values = ordered.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
    heights = ordered.get_column("height").cast(pl.Float64, strict=False).drop_nulls()

    if frames.size == 0 or x_values.size == 0 or heights.len() == 0:
        return None

    entry_index = int(np.searchsorted(frames, int(entry_frame)))
    entry_index = int(np.clip(entry_index, 0, frames.size - 1))
    if entry_index < step:
        # Not even one sampling interval of track exists before the kerb, so
        # there is nothing to measure. Unobservable, not zero.
        return None

    margin = STATIONARY_MARGIN_FRACTION * float(heights.median())
    stable_samples = 0
    index = entry_index

    while index - step >= 0:
        delta = abs(float(x_values[index]) - float(x_values[index - step]))
        if delta > margin:
            break
        stable_samples += 1
        index -= step

    if stable_samples < MINIMUM_STATIONARY_SAMPLES:
        return 0.0
    return float(stable_samples * step) / float(fps)


# ---------------------------------------------------------------------
# Road-restricted speed
# ---------------------------------------------------------------------

def _trim_track_rows(rows: List[Any], interval: RoadInterval) -> List[Any]:
    """Keep only the bbox rows inside the on-road interval."""
    return crossing_metrics.trim_rows_to_frame_range(
        rows, interval.entry_frame, interval.exit_frame,
    )


def road_restricted_speed(
    df_mapping: pl.DataFrame,
    df: pl.DataFrame,
    source_id: str,
    fps: float,
    intervals: Mapping[str, RoadInterval],
    aspect_ratio: Optional[float] = None,
) -> Tuple[Dict[str, float], Counter]:
    """Return crossing speed fitted over on-road frames only, plus reject reasons.

    The scene-motion reference is deliberately built from every detection in
    the segment, exactly as in the baseline. Only the measured pedestrian
    tracks are trimmed, so the camera-motion compensation the model relies on
    is unaffected by the restriction.

    When the frozen Waymo model is qualified, the returned values are metres
    per second, exactly like the baseline's qualified path. When it is not,
    this falls back to the same within-video relative-motion index the
    baseline itself falls back to (``predict_relative_bbox_rows``), computed
    from the on-road-only frames instead of the whole track. The reference
    tracks it is compared against are not similarly restricted (most carry no
    road interval at all), so the index is a coarser, self-consistent-only
    approximation, not a metric speed; treat it accordingly.
    """
    diagnostics: Counter = Counter()

    if not intervals:
        return {}, diagnostics
    if fps is None or float(fps) <= 0:
        diagnostics["invalid_fps"] += len(intervals)
        return {}, diagnostics

    if aspect_ratio is None or float(aspect_ratio) <= 0:
        aspect_ratio = crossing_metrics.DEFAULT_ASPECT_RATIO

    rows = crossing_metrics.bbox_rows_from_polars(df)
    if not rows:
        diagnostics["no_bbox_rows"] += len(intervals)
        return {}, diagnostics

    person_tracks = crossing_metrics.group_tracks(
        row for row in rows if row.class_id == crossing_metrics.PERSON_CLASS_ID
    )

    # group_tracks keys its output through normalise_id, which turns the
    # detection file's "171.0" into "171". The surface index is keyed by the
    # raw detection id, so the two must be reconciled before any lookup;
    # otherwise every track misses and nothing is ever trimmed. The baseline
    # calculate_speed_of_crossing normalises its selected ids for the same
    # reason, and the values returned here are keyed the same way so they
    # remain comparable with the baseline speeds.
    intervals_by_normalised_id = {}
    for raw_id, interval in intervals.items():
        normalised = crossing_metrics.normalise_id(raw_id)
        if normalised:
            intervals_by_normalised_id[normalised] = interval

    trimmed: Dict[str, List[Any]] = {}
    for track_id, track_rows in person_tracks.items():
        interval = intervals_by_normalised_id.get(str(track_id))
        if interval is None:
            trimmed[track_id] = track_rows
            continue
        kept = _trim_track_rows(track_rows, interval)
        if kept:
            trimmed[track_id] = kept
        else:
            diagnostics["road_interval_empty"] += 1

    if crossing_metrics._SPEED_MODEL:
        scene_profile = crossing_metrics.build_scene_motion_profile(rows, float(fps))
        features_by_track = crossing_metrics.contextual_track_features(
            trimmed,
            float(fps),
            str(source_id),
            float(aspect_ratio),
            scene_profile,
        )
        stature_scale = crossing_metrics.stature_scale_for_source(df_mapping, source_id)

        values: Dict[str, float] = {}
        for track_id in intervals_by_normalised_id:
            features = features_by_track.get(str(track_id))
            if features is None:
                diagnostics["no_features"] += 1
                continue
            prediction = crossing_metrics._predict_metric_speed(
                features,
                crossing_metrics._SPEED_MODEL,
            )
            if prediction.get("speed_status") == "valid":
                values[str(track_id)] = (
                    float(prediction["estimated_speed_mps"]) * stature_scale
                )
            else:
                diagnostics[str(prediction.get("reject_reason") or "rejected")] += 1

        return values, diagnostics

    # No qualified metric model: fall back to the CROWD relative-motion index,
    # restricted to the on-road frames of the crossing tracks. Non-person rows
    # carry the scene-motion reference and untouched person tracks act as the
    # within-video comparison set, exactly as predict_relative_bbox_rows
    # expects for the baseline's own unqualified path.
    diagnostics["relative_index_fallback_used"] += 1
    non_person_rows = [
        row for row in rows if row.class_id != crossing_metrics.PERSON_CLASS_ID
    ]
    relative_input_rows = non_person_rows + [
        row for track_rows in trimmed.values() for row in track_rows
    ]
    relative_predictions, _ = crossing_metrics.predict_relative_bbox_rows(
        relative_input_rows,
        float(fps),
        str(source_id),
        float(aspect_ratio),
    )
    relative_by_id = {
        crossing_metrics.normalise_id(row.get("prediction_track_id")): row
        for row in relative_predictions
    }

    values = {}
    for track_id in intervals_by_normalised_id:
        row = relative_by_id.get(str(track_id))
        if row is None:
            diagnostics["no_features"] += 1
            continue
        if row.get("relative_motion_status") != "valid":
            diagnostics[str(row.get("reject_reason") or "rejected")] += 1
            continue
        index_value = crossing_metrics.safe_float(row.get("relative_motion_index"))
        if index_value is None:
            diagnostics["relative_index_undefined"] += 1
            continue
        values[str(track_id)] = index_value

    return values, diagnostics
