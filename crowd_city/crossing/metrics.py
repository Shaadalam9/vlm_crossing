"""CROWD crossing metrics with algorithm selected Waymo speed calibration.

The production input remains a YOLO plus BoT SORT bounding box CSV and FPS.
Metric values are exposed only when the frozen bbox model passes untouched
Waymo validation; otherwise the output remains an explicitly relative index.
"""

import bisect
import csv
import json
import math
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

import common
from crowd_city.core.grouping import Grouping
from crowd_city.core.metadata import MetaData


METRICS_BUILD_ID = "crowd_algorithm_selected_waymo_speed_v32_20260825"
metadata_class = MetaData()
grouping_class = Grouping()

_PIPELINE_MODEL: Dict[str, Any] = {}
_SPEED_MODEL: Dict[str, Any] = {}


# The most recent detection table's box rows and scene-motion profile. Both
# are expensive for long CROWD segments and a detection worker needs them
# several times for the same file (detector picks, then rule-D candidates),
# so they are built once. Holding the table itself makes the identity check
# safe. Worker processes handle one file at a time; this is not used from the
# threaded segmentation pass.
_ROWS_AND_SCENE_CACHE: List[Any] = [None, None, None, None]


def rows_and_scene_profile(df: pl.DataFrame, fps: float) -> Tuple[List["BBoxRow"], "SceneMotionProfile"]:
    """Return the box rows of ``df`` and their scene-motion profile, built once per table."""
    cache = _ROWS_AND_SCENE_CACHE
    if cache[0] is df and cache[1] == float(fps):
        return cache[2], cache[3]
    rows = bbox_rows_from_polars(df)
    profile = build_scene_motion_profile(rows, float(fps))
    cache[:] = [df, float(fps), rows, profile]
    return rows, profile


def load_tuned_pipeline_model(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Load the frozen Waymo parameters used by the CROWD CSV analysis."""
    global _PIPELINE_MODEL, _SPEED_MODEL
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        _PIPELINE_MODEL = {}
        _SPEED_MODEL = {}
        return {}
    with model_path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("schema") not in {
        "crowd_waymo_pipeline_model_v32",
        "crowd_waymo_pipeline_model_v31",
        "crowd_waymo_pipeline_model_v30",
        "crowd_waymo_pipeline_model_v29",
        "crowd_waymo_pipeline_model_v28",
        "crowd_waymo_pipeline_model_v27",
    }:
        raise ValueError(f"Unsupported Waymo pipeline model schema: {model.get('schema')!r}")
    _PIPELINE_MODEL = model
    _SPEED_MODEL = {}
    speed_path_text = str(model.get("speed_production_model_path") or "").strip()
    if speed_path_text:
        speed_path = Path(speed_path_text).expanduser()
        if not speed_path.is_absolute():
            speed_path = model_path.parent / speed_path
        elif not speed_path.is_file():
            # The complete waymo_processed directory may have been moved from
            # the repository output directory to the raw Waymo dataset.  The
            # production model always lives at this stable location relative
            # to the frozen pipeline model.
            relocated_speed_path = (
                model_path.parent / "speed_validation" / "production_model.json"
            )
            if relocated_speed_path.is_file():
                speed_path = relocated_speed_path
        if speed_path.is_file():
            with speed_path.open("r", encoding="utf-8") as handle:
                speed_model = json.load(handle)
            if truthy(speed_model.get("calibration_qualified"), False) and truthy(
                speed_model.get("external_test_passed"),
                False,
            ):
                _SPEED_MODEL = speed_model
    os.environ["CROWD_CROSSING_SPEED_UNIT"] = "m/s" if _SPEED_MODEL else "relative"
    return dict(_PIPELINE_MODEL)


def tuned_crossing_parameters() -> Dict[str, Any]:
    parameters = _PIPELINE_MODEL.get("crossing_parameters", {})
    return dict(parameters) if isinstance(parameters, dict) else {}


def metric_speed_is_qualified() -> bool:
    return bool(_SPEED_MODEL)


# Consecutive stationary checks required before a wait is recorded at all.
MINIMUM_STATIONARY_CHECKS = 3


def leading_wait_check_units(
    x_values: np.ndarray,
    frames: Optional[np.ndarray],
    heights: np.ndarray,
    fps: float,
    checks_per_second: float,
) -> Optional[float]:
    """Return the leading stationary run of a track, in check_per_sec_time units.

    Positions are compared ``round(fps / checks_per_second)`` rows apart. A
    comparison moving no more than 10% of the median box height counts as a
    stationary check, and the first run of at least three such checks is the
    wait. Its length is measured from the frame numbers the checks actually
    span rather than assumed to be 1 / checks_per_second per check: the frame
    step is rounded (13 frames at 40 fps is 0.325 s, not 0.333 s) and missed
    detections make a check span more frames. The result is returned
    multiplied by checks_per_second, the unit every consumer of the baseline
    initiation time divides by, so its average comes out in true seconds.
    Returns None when the track has no such run.
    """
    if fps <= 0 or checks_per_second <= 0 or len(x_values) == 0 or len(heights) == 0:
        return None
    step = max(1, int(round(float(fps) / float(checks_per_second))))
    if frames is None or len(frames) != len(x_values):
        frames = np.arange(len(x_values), dtype=float)
    margin = 0.1 * float(np.median(heights))
    stable_checks = 0
    stable_frames = 0.0
    for index in range(0, len(x_values) - step, step):
        delta = abs(float(x_values[index + step]) - float(x_values[index]))
        if delta <= margin:
            stable_checks += 1
            stable_frames += float(frames[index + step]) - float(frames[index])
        elif stable_checks >= MINIMUM_STATIONARY_CHECKS:
            break
        else:
            stable_checks = 0
            stable_frames = 0.0
    if stable_checks < MINIMUM_STATIONARY_CHECKS:
        return None
    return stable_frames / float(fps) * float(checks_per_second)


class Metrics:
    """Public crossing metrics used by analysis.py."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _get_config(key: str):
        # Always the value in config: no hard-coded fallback.
        return common.get_configs(key)

    @staticmethod
    def _as_float(value, default=None):
        try:
            output = float(value)
        except (TypeError, ValueError):
            return default
        return output if math.isfinite(output) else default

    @staticmethod
    def _resolve_fps(df_mapping: pl.DataFrame, video_id: str) -> float | None:
        result = metadata_class.find_values_with_video_id(df_mapping, video_id)
        if result is None:
            return None
        fps = Metrics._as_float(result[17], None)
        return fps if fps is not None and fps > 0 else None

    @staticmethod
    def _track(df: pl.DataFrame, track_id: Any) -> pl.DataFrame:
        track = df.filter(pl.col("unique-id") == track_id).sort("frame-count")
        if track.height:
            return track
        return (
            df.filter(
                pl.col("unique-id").cast(pl.Utf8, strict=False) == str(track_id)
            )
            .sort("frame-count")
        )

    def time_to_cross(
        self,
        dataframe: pl.DataFrame,
        ids: list,
        video_id: str,
        df_mapping: pl.DataFrame,
    ) -> dict:
        """Return observed crossing track duration in seconds."""
        required = {"frame-count", "unique-id"}
        if not ids or not required.issubset(set(dataframe.columns)):
            return {}
        fps = self._resolve_fps(df_mapping, video_id)
        if fps is None:
            return {}

        output: dict = {}
        for track_id in ids:
            track = self._track(dataframe, track_id)
            if track.height < 2:
                continue
            frames = [
                self._as_float(value, None)
                for value in track.get_column("frame-count").to_list()
            ]
            frames = [value for value in frames if value is not None]
            if len(frames) < 2:
                continue
            duration = (max(frames) - min(frames)) / fps
            if duration > 0:
                output[track_id] = float(duration)
        return output

    def calculate_speed_of_crossing(
        self,
        df_mapping: pl.DataFrame,
        df: pl.DataFrame,
        data: dict,
        id_bounds: Optional[Dict[Any, Tuple[int, int]]] = None,
    ):
        """Return frozen metric speed when qualified, otherwise relative motion.

        ``id_bounds`` gives each crossing id's own ``(start_frame, end_frame)``
        from ``Detection.pedestrian_crossing``. A tracker id can be reused for
        an unrelated object elsewhere in the same video, so without this every
        row sharing that id would be treated as one track; the bound isolates
        the frames the crossing was actually accepted from.
        """
        if not data or not any(data.values()):
            return None
        source_id = next(iter(data))
        selected_ids = {
            normalise_id(value) for value in data[source_id] if normalise_id(value)
        }
        if not selected_ids:
            return None
        try:
            _video_id, _start, fps_text = str(source_id).rsplit("_", 2)
            fps = float(fps_text)
        except (TypeError, ValueError):
            fps = self._resolve_fps(df_mapping, source_id)
        if fps is None or fps <= 0:
            return None

        rows, scene_motion_profile = rows_and_scene_profile(df, float(fps))
        aspect_ratio = safe_float(os.environ.get("CROWD_BBOX_ASPECT_RATIO"))
        if aspect_ratio is None or aspect_ratio <= 0.0:
            aspect_ratio = DEFAULT_ASPECT_RATIO

        bounds_by_id = {
            normalise_id(key): value
            for key, value in (id_bounds or {}).items()
            if normalise_id(key) in selected_ids
        }

        if _SPEED_MODEL:
            person_tracks = group_tracks(
                row for row in rows if row.class_id == PERSON_CLASS_ID
            )
            if bounds_by_id:
                for track_id, bounds in bounds_by_id.items():
                    track_rows = person_tracks.get(track_id)
                    if not track_rows:
                        continue
                    trimmed = trim_rows_to_frame_range(track_rows, *bounds)
                    if trimmed:
                        person_tracks[track_id] = trimmed
            scene_profile = scene_motion_profile
            features_by_track = contextual_track_features(
                person_tracks,
                float(fps),
                str(source_id),
                aspect_ratio,
                scene_profile,
            )
            stature_scale = stature_scale_for_source(df_mapping, source_id)
            values: Dict[str, float] = {}
            for track_id in selected_ids:
                features = features_by_track.get(track_id)
                if features is None:
                    continue
                prediction = _predict_metric_speed(features, _SPEED_MODEL)
                if prediction.get("speed_status") == "valid":
                    values[track_id] = (
                        float(prediction["estimated_speed_mps"]) * stature_scale
                    )
            if not values:
                return None
            return grouping_class.locality_country_wrapper(
                input_dict={str(source_id): values},
                mapping=df_mapping,
            )

        # relative_rows and its cache measure the whole track, unbounded, since
        # analyse_crowd_sources' diagnostic dump reads this cache expecting one
        # row per full person track in the scene, not per accepted crossing.
        relative_rows, _ = predict_relative_bbox_rows(
            rows, float(fps), str(source_id), aspect_ratio
        )
        person_track_count = len(
            {row.track_id for row in rows if row.class_id == PERSON_CLASS_ID}
        )
        _RELATIVE_MEMORY_CACHE[str(source_id)] = (
            relative_rows,
            person_track_count,
        )
        values = {
            row["prediction_track_id"]: float(row["relative_motion_index"])
            for row in relative_rows
            if normalise_id(row.get("prediction_track_id")) in selected_ids
            and row.get("relative_motion_status") == "valid"
            and safe_float(row.get("relative_motion_index")) is not None
        }

        if bounds_by_id:
            # A second, bounded pass supplies the values actually reported for
            # the ids with a known crossing segment, so the cache above stays
            # whole-track for its own consumer while the reported speed is not
            # diluted by a reused id's unrelated frames.
            non_person_rows = [
                row for row in rows if row.class_id != PERSON_CLASS_ID
            ]
            person_tracks = group_tracks(
                row for row in rows if row.class_id == PERSON_CLASS_ID
            )
            bounded_rows = list(non_person_rows)
            for track_id, track_rows in person_tracks.items():
                bounds = bounds_by_id.get(track_id)
                if bounds is not None:
                    trimmed = trim_rows_to_frame_range(track_rows, *bounds)
                    bounded_rows.extend(trimmed or track_rows)
                else:
                    bounded_rows.extend(track_rows)
            bounded_relative_rows, _ = predict_relative_bbox_rows(
                bounded_rows, float(fps), str(source_id), aspect_ratio
            )
            for row in bounded_relative_rows:
                track_id = normalise_id(row.get("prediction_track_id"))
                if track_id not in bounds_by_id:
                    continue
                index_value = safe_float(row.get("relative_motion_index"))
                if row.get("relative_motion_status") != "valid" or index_value is None:
                    values.pop(track_id, None)
                    continue
                values[track_id] = index_value

        if not values:
            return None
        return grouping_class.locality_country_wrapper(
            input_dict={str(source_id): values},
            mapping=df_mapping,
        )

    def avg_speed_of_crossing_locality(
        self,
        df_mapping: pl.DataFrame,
        all_speed: dict,
    ):
        """Average the dimensionless crossing speed index by locality."""
        del df_mapping
        averages: dict = {}
        complete: dict = {}
        minimum = float(self._get_config("min_speed_limit"))
        maximum = float(self._get_config("max_speed_limit"))
        for locality, videos in (all_speed or {}).items():
            values = [
                float(value)
                for tracks in videos.values()
                for value in tracks.values()
                if minimum <= float(value) <= maximum
            ]
            if values:
                complete[locality] = values
                averages[locality] = sum(values) / len(values)
        return averages, complete

    def avg_speed_of_crossing_country(
        self,
        df_mapping: pl.DataFrame,
        all_speed: dict,
    ):
        """Average the dimensionless crossing speed index by country."""
        grouped: dict[str, list[float]] = {}
        minimum = float(self._get_config("min_speed_limit"))
        maximum = float(self._get_config("max_speed_limit"))
        for videos in (all_speed or {}).values():
            for video_id, tracks in videos.items():
                result = metadata_class.find_values_with_video_id(df_mapping, video_id)
                if result is None:
                    continue
                key = f"{result[8]}_{result[3]}"
                for value in tracks.values():
                    numeric = self._as_float(value, None)
                    if numeric is not None and minimum <= numeric <= maximum:
                        grouped.setdefault(key, []).append(float(numeric))
        averages = {
            key: sum(values) / len(values)
            for key, values in grouped.items()
            if values
        }
        return averages, grouped

    def time_to_start_cross(
        self,
        df_mapping: pl.DataFrame,
        df: pl.DataFrame,
        data: dict,
        person_id: int = 0,
    ):
        """Estimate the stationary interval immediately before motion starts."""
        del person_id
        if not data or not any(data.values()):
            return None
        required = {"unique-id", "frame-count", "x-center", "height"}
        if not required.issubset(set(df.columns)):
            return None
        source_id = next(iter(data))
        fps = self._resolve_fps(df_mapping, source_id)
        if fps is None:
            return None

        checks_per_second = float(self._get_config("check_per_sec_time"))
        if checks_per_second is None or checks_per_second <= 0:
            return None
        step = max(1, int(round(fps / checks_per_second)))
        durations: dict = {}

        for track_id in data[source_id]:
            track = self._track(df, track_id)
            if track.height <= step:
                continue
            x_values = track.get_column("x-center").cast(
                pl.Float64,
                strict=False,
            ).to_numpy()
            heights = track.get_column("height").cast(
                pl.Float64,
                strict=False,
            ).drop_nulls()
            wait = leading_wait_check_units(
                x_values,
                track.get_column("frame-count").cast(pl.Float64, strict=False).to_numpy(),
                heights.to_numpy(),
                float(fps),
                checks_per_second,
            )
            if wait is not None:
                durations[track_id] = wait

        if not durations:
            return None
        return grouping_class.locality_country_wrapper(
            input_dict={source_id: durations},
            mapping=df_mapping,
        )

    def avg_time_to_start_cross_locality(
        self,
        df_mapping: pl.DataFrame,
        all_time: dict,
    ):
        """Average crossing initiation time by locality."""
        del df_mapping
        averages: dict = {}
        complete: dict = {}
        checks = float(self._get_config("check_per_sec_time"))
        minimum = float(self._get_config("min_waiting_time"))
        maximum = float(self._get_config("max_waiting_time"))
        for locality, videos in (all_time or {}).items():
            values = [
                float(value) / checks
                for tracks in videos.values()
                if tracks
                for value in tracks.values()
                if minimum <= float(value) / checks <= maximum
            ]
            if values:
                complete[locality] = values
                averages[locality] = sum(values) / len(values)
        return averages, complete

    def avg_time_to_start_cross_country(
        self,
        df_mapping: pl.DataFrame,
        all_time: dict,
    ):
        """Average crossing initiation time by country."""
        grouped: dict[str, list[float]] = {}
        checks = float(self._get_config("check_per_sec_time"))
        minimum = float(self._get_config("min_waiting_time"))
        maximum = float(self._get_config("max_waiting_time"))
        for videos in (all_time or {}).values():
            for video_id, times in videos.items():
                if not times:
                    continue
                result = metadata_class.find_values_with_video_id(df_mapping, video_id)
                if result is None:
                    continue
                key = f"{result[8]}_{result[3]}"
                for value in times.values():
                    seconds = float(value) / checks
                    if minimum <= seconds <= maximum:
                        grouped.setdefault(key, []).append(seconds)
        averages = {
            key: sum(values) / len(values)
            for key, values in grouped.items()
            if values
        }
        return averages, grouped


REPORT_BUILD_ID = "crowd_waymo_crossing_report_v24_20260822"
PERSON_CLASS_ID = 0
DEFAULT_ASPECT_RATIO = 16.0 / 9.0
DEFAULT_PERSON_HEIGHT_M = 1.70

# Mean adult stature of the population the Waymo speed model was calibrated on.
# Waymo Open Dataset is recorded in the United States, whose mean stature in the
# project mapping is 170.0 cm, which is also DEFAULT_PERSON_HEIGHT_M.
WAYMO_CALIBRATION_STATURE_CM = 170.0

SCENE_MOTION_SETTINGS: Dict[str, float] = {
    "maximum_pair_gap_frames": 2,
    "window_radius_frames": 3,
    "minimum_reference_tracks": 3,
    "maximum_absolute_x_rate_per_second": 3.0,
    "outlier_mad_multiplier": 4.0,
}

# Feature windows in seconds rather than frames, so every feature describes
# the same span of time at any frame rate while still using every frame. At
# the 10 fps Waymo calibration rate each window equals the frame count it
# replaced (3, 5, 1, 2 and 3 frames), so calibrated features are unchanged.
FEATURE_TIME_WINDOWS_SECONDS: Dict[str, float] = {
    "smoothing": 0.3,
    "cleaning": 0.5,
    "rate_span": 0.1,
    "scene_pair_gap": 0.2,
    "scene_radius": 0.3,
}
FEATURE_REFERENCE_FPS = 10.0


def window_frames(seconds: float, fps: float, odd: bool = False) -> int:
    """Return the number of frames spanning ``seconds`` at ``fps`` (at least one)."""
    frames = max(1, int(round(float(seconds) * float(fps))))
    if odd and frames % 2 == 0:
        frames += 1
    return frames


def lagged_rates(time_values: np.ndarray, values: np.ndarray, lag: int) -> np.ndarray:
    """Return rates of change over ``lag`` frames, starting at every frame."""
    if len(values) <= lag:
        return np.zeros(0, dtype=float)
    delta_time = time_values[lag:] - time_values[:-lag]
    delta_value = values[lag:] - values[:-lag]
    valid = delta_time > 0.0
    return delta_value[valid] / delta_time[valid]


def reversal_fraction(x_values: np.ndarray, lag: int = 1) -> float:
    """Share of consecutive movement steps whose direction reverses.

    Each step spans ``lag`` frames. Every frame starts one sequence of
    back-to-back steps, and the sequences are pooled, so no frame is dropped.
    With ``lag`` 1 this is the original frame-to-frame measure.
    """
    changes = 0
    comparisons = 0
    for offset in range(max(1, int(lag))):
        series = x_values[offset::max(1, int(lag))]
        if len(series) < 4:
            continue
        delta = np.diff(series)
        noise = max(robust_scale(delta) * 0.25, 0.0005)
        signs = np.sign(delta[np.abs(delta) > noise])
        if len(signs) < 2:
            continue
        changes += int(np.sum(signs[1:] != signs[:-1]))
        comparisons += len(signs) - 1
    return float(changes / comparisons) if comparisons else 0.0


SOURCE_CONTEXT_SETTINGS: Dict[str, float] = {
    "minimum_reference_tracks": 3,
    "minimum_proxy_mps": 0.001,
    "maximum_absolute_log_ratio": 2.00,
}

BASE_GATES: Dict[str, float] = {
    "minimum_rows": 8,
    "minimum_duration_seconds": 0.50,
    "minimum_coverage": 0.50,
    "minimum_median_height": 0.01,
    "minimum_horizontal_range": 0.08,
    "minimum_x_fit_r2": 0.15,
    "maximum_height_ratio": 3.50,
    "maximum_edge_fraction": 0.60,
    "maximum_reversal_fraction": 0.50,
    "minimum_scene_motion_support": 3,
}

RELIABILITY_GATES: Dict[str, float] = {
    "minimum_duration_seconds": 2.00,
    "minimum_x_fit_r2": 0.90,
}

ASSOCIATION_REJECTION_REASONS = {
    "too_few_matched_frames",
    "mean_iou_below_minimum",
    "prediction_coverage_below_minimum",
    "ground_truth_coverage_below_minimum",
    "association_too_few_matched_frames",
    "association_mean_iou_too_low",
    "association_prediction_coverage_too_low",
    "association_ground_truth_coverage_too_low",
}

# analysis.py calculates its legacy aggregate fields before producing the new
# report. Reuse those extracted features during the same process instead of
# reading and fitting every bbox track a second time.
_RELATIVE_MEMORY_CACHE: Dict[str, Tuple[List[Dict[str, Any]], int]] = {}


@dataclass(frozen=True)
class BBoxRow:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    track_id: str
    confidence: float
    frame: int


@dataclass
class SceneMotionProfile:
    samples_by_frame: Dict[int, List[Tuple[float, str]]]
    # window_radius_frames at the reference rate; build_scene_motion_profile
    # sets it from FEATURE_TIME_WINDOWS_SECONDS for the video's frame rate.
    radius_frames: int = 3

    def rate_at(self, frame: float) -> Tuple[float, int]:
        radius = int(self.radius_frames)
        centre_frame = int(round(frame))
        samples: List[Tuple[float, str]] = []
        for candidate in range(centre_frame - radius, centre_frame + radius + 1):
            samples.extend(self.samples_by_frame.get(candidate, []))
        support = len({track_id for _, track_id in samples})
        if support < int(SCENE_MOTION_SETTINGS["minimum_reference_tracks"]):
            return 0.0, support
        rates = np.asarray([rate for rate, _ in samples], dtype=float)
        centre = float(np.median(rates))
        scale = robust_scale(rates)
        limit = float(SCENE_MOTION_SETTINGS["outlier_mad_multiplier"]) * scale
        retained = rates[np.abs(rates - centre) <= limit]
        if len(retained):
            centre = float(np.median(retained))
        return centre, support


@dataclass
class TrackFeatures:
    source_id: str
    prediction_track_id: str
    input_rows: int
    clean_rows: int
    removed_rows: int
    first_frame: int
    last_frame: int
    duration_seconds: float
    coverage: float
    gap_ratio: float
    median_confidence: float
    median_height: float
    median_width: float
    height_ratio: float
    horizontal_range: float
    vertical_range: float
    crosses_image_midline: bool
    overlaps_central_band: bool
    direction: str
    x_slope_per_second: float
    x_fit_r2: float
    q_slope_per_second: float
    q_fit_r2: float
    q_residual_mad: float
    log_height_rate_abs: float
    bottom_rate_abs: float
    edge_fraction: float
    truncation_fraction: float
    reversal_fraction: float
    raw_speed_proxy_mps: float
    q_speed_proxy_mps: float
    robust_speed_proxy_mps: float
    compensated_raw_speed_proxy_mps: float
    compensated_q_speed_proxy_mps: float
    compensated_robust_speed_proxy_mps: float
    scene_motion_rate_abs: float
    scene_motion_equivalent_speed_mps: float
    scene_motion_fraction: float
    scene_motion_support: float
    log_scene_motion_support: float
    compensated_proxy_disagreement_mps: float
    one_minus_x_r2: float
    log_height_ratio: float
    log_duration: float
    log_median_height: float
    source_context_tracks: float = 0.0
    source_context_available: float = 0.0
    source_relative_log_raw_proxy: float = 0.0
    source_relative_log_q_proxy: float = 0.0
    source_relative_log_robust_proxy: float = 0.0
    source_relative_log_compensated_robust_proxy: float = 0.0
    source_compensated_robust_percentile: float = 0.5


def _default_log(message: str) -> None:
    print(message)


def safe_float(value: Any) -> Optional[float]:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def safe_int(value: Any) -> Optional[int]:
    output = safe_float(value)
    return int(round(output)) if output is not None else None


def truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def stature_scale_for_source(df_mapping: "pl.DataFrame", source_id: Any) -> float:
    """Return the factor rescaling a Waymo-calibrated speed to local stature.

    Every geometric speed proxy is proportional to an assumed person height
    divided by the observed bounding box height, so a prediction inherits the
    stature of the population the model was calibrated on. Waymo is recorded in
    the United States, so an uncorrected prediction over-estimates the speed of
    shorter populations and under-estimates taller ones in direct proportion to
    the stature ratio. Across the analysed countries that ratio spans about 12
    per cent and is correlated with the very covariates the study regresses
    crossing speed against, so leaving it uncorrected biases those estimates.

    The factor is exactly 1.0 on the calibration population, so every reported
    Waymo validation metric remains valid.

    This correction is DISABLED by default because its strength cannot be
    determined from Waymo. A model that attenuates its proxy-to-speed response
    also attenuates any bias carried by that proxy, so the bias is closer to
    (stature ratio) ** beta, where beta is the model calibration slope (0.35 in
    the current frozen model), than to the full ratio. Applying the full ratio
    demonstrably overshoots: on the analysed cities it drove the correlation
    between city speed and national stature from +0.07 to +0.31 rather than
    towards zero. Waymo is recorded in one country, so it contains no stature
    variation with which to estimate beta. Measuring it requires ground truth
    from a second population; nuScenes (Singapore and Boston) is the obvious
    candidate and nuscenes-devkit is already a project dependency.

    Set CROWD_STATURE_CORRECTION=true to enable, and
    CROWD_STATURE_CORRECTION_EXPONENT to set the exponent (default 1.0, full
    strength). Treat any non-zero exponent as unvalidated until it is estimated
    against a second-population ground truth.
    """
    if not truthy(os.environ.get("CROWD_STATURE_CORRECTION", "false"), False):
        return 1.0
    result = metadata_class.find_values_with_video_id(df_mapping, str(source_id))
    if result is None:
        return 1.0
    stature_cm = safe_float(result[15])
    if stature_cm is None or stature_cm <= 0.0:
        return 1.0
    exponent = safe_float(os.environ.get("CROWD_STATURE_CORRECTION_EXPONENT"))
    if exponent is None:
        exponent = 1.0
    return (float(stature_cm) / WAYMO_CALIBRATION_STATURE_CM) ** exponent


def normalise_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    numeric = safe_float(text)
    if numeric is not None and abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return text


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "source"


def first_column(
    fieldnames: Sequence[str],
    aliases: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    if required:
        raise ValueError(
            "CSV is missing a required column. Expected one of: " + ", ".join(aliases)
        )
    return None


def read_dict_csv(path: os.PathLike[str] | str) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_dict_csv(path: os.PathLike[str] | str, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: os.PathLike[str] | str, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_bbox_csv(path: os.PathLike[str] | str) -> List[BBoxRow]:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Bounding box CSV has no header: {input_path}")
        names = reader.fieldnames
        class_col = first_column(names, ["yolo-id", "yolo_id", "class-id", "class_id", "class"])
        x_col = first_column(names, ["x-center", "x_center", "xcentre", "xcenter"])
        y_col = first_column(names, ["y-center", "y_center", "ycentre", "ycenter"])
        width_col = first_column(names, ["width", "bbox_width", "w"])
        height_col = first_column(names, ["height", "bbox_height", "h"])
        track_col = first_column(
            names,
            ["unique-id", "unique_id", "track-id", "track_id", "prediction_track_id"],
        )
        confidence_col = first_column(names, ["confidence", "conf", "score"], required=False)
        frame_col = first_column(
            names,
            ["frame-count", "frame_count", "frame-id", "frame_id", "frame"],
        )
        output: List[BBoxRow] = []
        for raw in reader:
            class_id = safe_int(raw.get(class_col))
            x = safe_float(raw.get(x_col))
            y = safe_float(raw.get(y_col))
            width = safe_float(raw.get(width_col))
            height = safe_float(raw.get(height_col))
            frame = safe_int(raw.get(frame_col))
            track_id = normalise_id(raw.get(track_col))
            confidence = safe_float(raw.get(confidence_col)) if confidence_col else 1.0
            if None in (class_id, x, y, width, height, frame, confidence) or not track_id:
                continue
            if float(width) <= 0.0 or float(height) <= 0.0:
                continue
            output.append(
                BBoxRow(
                    int(class_id),
                    float(x),
                    float(y),
                    float(width),
                    float(height),
                    track_id,
                    float(confidence),
                    int(frame),
                )
            )
    return output


def bbox_rows_from_polars(dataframe: pl.DataFrame) -> List[BBoxRow]:
    """Convert the existing analysis DataFrame without creating a second CSV."""
    names = dataframe.columns
    class_col = first_column(names, ["yolo-id", "yolo_id", "class-id", "class_id", "class"])
    x_col = first_column(names, ["x-center", "x_center", "xcentre", "xcenter"])
    y_col = first_column(names, ["y-center", "y_center", "ycentre", "ycenter"])
    width_col = first_column(names, ["width", "bbox-width", "bbox_width", "w"])
    height_col = first_column(names, ["height", "bbox-height", "bbox_height", "h"])
    track_col = first_column(names, ["unique-id", "unique_id", "track-id", "track_id", "id"])
    frame_col = first_column(names, ["frame-count", "frame_count", "frame", "frame-id", "frame_id"])
    confidence_col = first_column(
        names,
        ["confidence", "conf", "score"],
        required=False,
    )
    output: List[BBoxRow] = []
    for raw in dataframe.to_dicts():
        class_id = safe_int(raw.get(class_col))
        x_value = safe_float(raw.get(x_col))
        y_value = safe_float(raw.get(y_col))
        width = safe_float(raw.get(width_col))
        height = safe_float(raw.get(height_col))
        frame = safe_int(raw.get(frame_col))
        track_id = normalise_id(raw.get(track_col))
        confidence = safe_float(raw.get(confidence_col)) if confidence_col else 1.0
        if None in {class_id, x_value, y_value, width, height, frame} or not track_id:
            continue
        if width <= 0.0 or height <= 0.0:
            continue
        output.append(
            BBoxRow(
                class_id=int(class_id),
                x=float(x_value),
                y=float(y_value),
                width=float(width),
                height=float(height),
                track_id=track_id,
                confidence=float(confidence if confidence is not None else 1.0),
                frame=int(frame),
            )
        )
    return output


def group_tracks(rows: Iterable[BBoxRow]) -> Dict[str, List[BBoxRow]]:
    grouped: Dict[str, List[BBoxRow]] = defaultdict(list)
    for row in rows:
        grouped[row.track_id].append(row)
    return dict(grouped)


def trim_rows_to_frame_range(
    rows: Sequence[BBoxRow],
    start_frame: int,
    end_frame: int,
) -> List[BBoxRow]:
    """Keep only the rows of one track inside ``[start_frame, end_frame]``.

    A tracker id can be reused for an unrelated object elsewhere in the same
    video, so the rows sharing that id are not necessarily all one crossing.
    Callers that know the specific segment's frame bounds (from
    ``Detection.pedestrian_crossing`` or a road interval) use this to isolate
    it before computing duration or speed features.
    """
    low, high = int(start_frame), int(end_frame)
    return [row for row in rows if low <= row.frame <= high]


def rolling_median(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Centred rolling median; the window shrinks at the ends of the series.

    Vectorised with a sliding window view: for long CROWD tracks at 30 fps the
    per-element Python loop dominated the whole detection pass. The result is
    identical to taking np.median over values[max(0, i - r):i + r + 1].
    """
    if len(values) < 3 or window <= 1:
        return values.astype(float, copy=True)
    values = np.asarray(values, dtype=float)
    radius = window // 2
    count = len(values)
    output = np.empty(count, dtype=float)
    if count > 2 * radius:
        output[radius:count - radius] = np.median(
            np.lib.stride_tricks.sliding_window_view(values, 2 * radius + 1), axis=1
        )
        edges = list(range(radius)) + list(range(count - radius, count))
    else:
        edges = range(count)
    for index in edges:
        output[index] = float(np.median(values[max(0, index - radius):min(count, index + radius + 1)]))
    return output


def robust_scale(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    return max(1.4826 * mad, 1e-9)


def robust_line(time_values: np.ndarray, values: np.ndarray) -> Tuple[float, float, float, float]:
    if len(time_values) < 2 or float(np.ptp(time_values)) <= 0.0:
        first = float(values[0]) if len(values) else 0.0
        return 0.0, first, 0.0, 0.0
    design = np.column_stack([np.ones(len(time_values)), time_values])
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    weights = np.ones(len(values), dtype=float)
    for _ in range(20):
        residual = values - design @ beta
        scale = robust_scale(residual)
        threshold = 1.345 * scale
        weights = np.ones(len(values), dtype=float)
        large = np.abs(residual) > threshold
        weights[large] = threshold / np.maximum(np.abs(residual[large]), 1e-12)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = values * np.sqrt(weights)
        new_beta = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        if float(np.max(np.abs(new_beta - beta))) < 1e-9:
            beta = new_beta
            break
        beta = new_beta
    fitted = design @ beta
    residual = values - fitted
    weighted_mean = float(np.average(values, weights=weights))
    ss_res = float(np.sum(weights * residual * residual))
    ss_tot = float(np.sum(weights * (values - weighted_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return (
        float(beta[1]),
        float(beta[0]),
        float(max(-1.0, min(1.0, r2))),
        robust_scale(residual),
    )


def clean_track(
    rows: Sequence[BBoxRow],
    fps: float = FEATURE_REFERENCE_FPS,
) -> List[BBoxRow]:
    best_by_frame: Dict[int, BBoxRow] = {}
    for row in rows:
        if not (-0.25 <= row.x <= 1.25 and -0.25 <= row.y <= 1.25):
            continue
        if not (0.001 <= row.width <= 1.50 and 0.001 <= row.height <= 1.50):
            continue
        current = best_by_frame.get(row.frame)
        if current is None or row.confidence > current.confidence:
            best_by_frame[row.frame] = row
    ordered = [best_by_frame[key] for key in sorted(best_by_frame)]
    if len(ordered) < 7:
        return ordered
    values = np.asarray(
        [[row.x, row.y, row.width, row.height] for row in ordered], dtype=float
    )
    cleaning_window = window_frames(FEATURE_TIME_WINDOWS_SECONDS["cleaning"], fps, odd=True)
    baseline = np.column_stack(
        [rolling_median(values[:, column], cleaning_window) for column in range(4)]
    )
    residual = values - baseline
    keep = np.ones(len(ordered), dtype=bool)
    for column in range(4):
        scale = robust_scale(residual[:, column])
        if scale > 1e-8:
            keep &= np.abs(residual[:, column]) <= 6.0 * scale
    keep[0] = True
    keep[-1] = True
    return [row for index, row in enumerate(ordered) if bool(keep[index])]


def _scene_motion_pairs(rows: Sequence[BBoxRow], rate_span: int, maximum_gap: int):
    """Pair each reference detection with the first one ``rate_span`` or more frames later.

    Every detection starts a pair, so no frame is dropped; pairs further apart
    than ``maximum_gap`` frames are skipped. With a span of one frame this is
    the original consecutive-detection pairing.
    """
    frames = [row.frame for row in rows]
    for index, first in enumerate(rows):
        later = bisect.bisect_left(frames, first.frame + int(rate_span), index + 1)
        if later < len(rows) and rows[later].frame - first.frame <= int(maximum_gap):
            yield first, rows[later]


def build_scene_motion_profile(rows: Sequence[BBoxRow], fps: float) -> SceneMotionProfile:
    samples_by_frame: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
    grouped: Dict[Tuple[int, str], List[BBoxRow]] = defaultdict(list)
    for row in rows:
        if row.class_id != PERSON_CLASS_ID:
            grouped[(row.class_id, row.track_id)].append(row)
    maximum_gap = window_frames(FEATURE_TIME_WINDOWS_SECONDS["scene_pair_gap"], fps)
    rate_span = window_frames(FEATURE_TIME_WINDOWS_SECONDS["rate_span"], fps)
    maximum_rate = float(SCENE_MOTION_SETTINGS["maximum_absolute_x_rate_per_second"])
    for (class_id, track_id), track_rows in grouped.items():
        reference_id = f"{class_id}:{track_id}"
        cleaned = clean_track(track_rows, fps)
        for first, second in _scene_motion_pairs(cleaned, rate_span, maximum_gap):
            frame_gap = second.frame - first.frame
            rate = (second.x - first.x) / (frame_gap / fps)
            if not math.isfinite(rate) or abs(rate) > maximum_rate:
                continue
            midpoint = int(round((first.frame + second.frame) / 2.0))
            samples_by_frame[midpoint].append((float(rate), reference_id))
    return SceneMotionProfile(
        dict(samples_by_frame),
        radius_frames=window_frames(FEATURE_TIME_WINDOWS_SECONDS["scene_radius"], fps),
    )


def track_features(
    rows: Sequence[BBoxRow],
    fps: float,
    source_id: str,
    aspect_ratio: float,
    scene_motion_profile: SceneMotionProfile,
) -> Optional[TrackFeatures]:
    cleaned = clean_track(rows, fps)
    if len(cleaned) < 2:
        return None
    frames = np.asarray([row.frame for row in cleaned], dtype=float)
    x_raw = np.asarray([row.x for row in cleaned], dtype=float)
    y_raw = np.asarray([row.y for row in cleaned], dtype=float)
    width_raw = np.asarray([row.width for row in cleaned], dtype=float)
    height_raw = np.asarray([row.height for row in cleaned], dtype=float)
    confidence = np.asarray([row.confidence for row in cleaned], dtype=float)
    smoothing = window_frames(FEATURE_TIME_WINDOWS_SECONDS["smoothing"], fps, odd=True)
    rate_span = window_frames(FEATURE_TIME_WINDOWS_SECONDS["rate_span"], fps)
    x = rolling_median(x_raw, smoothing)
    y = rolling_median(y_raw, smoothing)
    width = rolling_median(width_raw, smoothing)
    height = np.maximum(rolling_median(height_raw, smoothing), 0.001)
    times = (frames - frames[0]) / fps
    duration = float(times[-1])
    if duration <= 0.0:
        return None
    median_height = float(np.median(height))
    median_width = float(np.median(width))
    bottom = y + 0.50 * height
    q_value = aspect_ratio * (x - 0.50) / height
    log_height = np.log(height)
    x_slope, _, x_r2, _ = robust_line(times, x)
    q_slope, _, q_r2, q_mad = robust_line(times, q_value)
    log_height_slope, _, _, _ = robust_line(times, log_height)
    bottom_slope, _, _, _ = robust_line(times, bottom)

    delta_time = np.diff(times)
    delta_x = np.diff(x)
    local_rates = (
        aspect_ratio * lagged_rates(times, x, rate_span) / max(median_height, 0.001)
    )
    raw_proxy = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * abs(x_slope)
        / max(median_height, 0.001)
    )
    q_proxy = DEFAULT_PERSON_HEIGHT_M * abs(q_slope)
    robust_proxy = (
        DEFAULT_PERSON_HEIGHT_M * abs(float(np.median(local_rates)))
        if len(local_rates)
        else 0.0
    )

    scene_rates: List[float] = []
    scene_support: List[int] = []
    for first_frame, second_frame in zip(frames, frames[1:]):
        rate, support = scene_motion_profile.rate_at((first_frame + second_frame) / 2.0)
        scene_rates.append(float(rate))
        scene_support.append(int(support))
    scene_array = np.asarray(scene_rates, dtype=float)
    corrected_delta_x = delta_x - scene_array * delta_time
    corrected_x = np.concatenate([[x[0]], x[0] + np.cumsum(corrected_delta_x)])
    corrected_x_slope, _, _, _ = robust_line(times, corrected_x)
    corrected_q = aspect_ratio * (corrected_x - 0.50) / height
    corrected_q_slope, _, _, _ = robust_line(times, corrected_q)
    corrected_local_rates = (
        aspect_ratio * lagged_rates(times, corrected_x, rate_span) / max(median_height, 0.001)
    )
    compensated_raw = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * abs(corrected_x_slope)
        / max(median_height, 0.001)
    )
    compensated_q = DEFAULT_PERSON_HEIGHT_M * abs(corrected_q_slope)
    compensated_robust = (
        DEFAULT_PERSON_HEIGHT_M * abs(float(np.median(corrected_local_rates)))
        if len(corrected_local_rates)
        else 0.0
    )
    supported_rates = [
        abs(rate)
        for rate, support in zip(scene_rates, scene_support)
        if support >= int(SCENE_MOTION_SETTINGS["minimum_reference_tracks"])
    ]
    scene_rate_abs = float(np.median(supported_rates)) if supported_rates else 0.0
    median_support = float(np.median(scene_support)) if scene_support else 0.0
    scene_equivalent = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * scene_rate_abs
        / max(median_height, 0.001)
    )
    scene_fraction = scene_equivalent / max(raw_proxy + scene_equivalent, 1e-9)

    first_frame = int(frames[0])
    last_frame = int(frames[-1])
    expected_rows = max(1, last_frame - first_frame + 1)
    coverage = min(1.0, len(cleaned) / expected_rows)
    left = x - 0.50 * width
    right = x + 0.50 * width
    top = y - 0.50 * height
    lower = y + 0.50 * height
    edge = (left <= 0.01) | (right >= 0.99)
    truncated = edge | (top <= 0.01) | (lower >= 0.99)
    height_low = max(float(np.quantile(height, 0.10)), 0.001)
    height_high = float(np.quantile(height, 0.90))
    direction = (
        "left_to_right"
        if x_slope > 0.0
        else "right_to_left"
        if x_slope < 0.0
        else "stationary"
    )
    disagreement = float(
        max(compensated_raw, compensated_q, compensated_robust)
        - min(compensated_raw, compensated_q, compensated_robust)
    )
    return TrackFeatures(
        source_id=source_id,
        prediction_track_id=cleaned[0].track_id,
        input_rows=len(rows),
        clean_rows=len(cleaned),
        removed_rows=len(rows) - len(cleaned),
        first_frame=first_frame,
        last_frame=last_frame,
        duration_seconds=duration,
        coverage=float(coverage),
        gap_ratio=float(1.0 - coverage),
        median_confidence=float(np.median(confidence)),
        median_height=median_height,
        median_width=median_width,
        height_ratio=float(height_high / height_low),
        horizontal_range=float(np.max(x) - np.min(x)),
        vertical_range=float(np.max(y) - np.min(y)),
        crosses_image_midline=bool(float(np.min(x)) <= 0.50 <= float(np.max(x))),
        overlaps_central_band=bool(float(np.min(x)) <= 0.60 and float(np.max(x)) >= 0.40),
        direction=direction,
        x_slope_per_second=float(x_slope),
        x_fit_r2=float(x_r2),
        q_slope_per_second=float(q_slope),
        q_fit_r2=float(q_r2),
        q_residual_mad=float(q_mad),
        log_height_rate_abs=abs(float(log_height_slope)),
        bottom_rate_abs=abs(float(bottom_slope)),
        edge_fraction=float(np.mean(edge)),
        truncation_fraction=float(np.mean(truncated)),
        reversal_fraction=reversal_fraction(x, rate_span),
        raw_speed_proxy_mps=float(raw_proxy),
        q_speed_proxy_mps=float(q_proxy),
        robust_speed_proxy_mps=float(robust_proxy),
        compensated_raw_speed_proxy_mps=float(compensated_raw),
        compensated_q_speed_proxy_mps=float(compensated_q),
        compensated_robust_speed_proxy_mps=float(compensated_robust),
        scene_motion_rate_abs=scene_rate_abs,
        scene_motion_equivalent_speed_mps=float(scene_equivalent),
        scene_motion_fraction=float(scene_fraction),
        scene_motion_support=median_support,
        log_scene_motion_support=float(math.log1p(median_support)),
        compensated_proxy_disagreement_mps=disagreement,
        one_minus_x_r2=float(max(0.0, 1.0 - x_r2)),
        log_height_ratio=float(math.log(max(height_high / height_low, 1.0))),
        log_duration=float(math.log1p(duration)),
        log_median_height=float(math.log(max(median_height, 1e-6))),
    )


def base_rejection_reason(features: TrackFeatures) -> str:
    checks = [
        (features.clean_rows < int(BASE_GATES["minimum_rows"]), "too_few_rows"),
        (
            features.duration_seconds < BASE_GATES["minimum_duration_seconds"],
            "duration_too_short",
        ),
        (features.coverage < BASE_GATES["minimum_coverage"], "track_too_fragmented"),
        (
            features.median_height < BASE_GATES["minimum_median_height"],
            "box_too_small",
        ),
        (
            features.horizontal_range < BASE_GATES["minimum_horizontal_range"],
            "insufficient_lateral_motion",
        ),
        (
            features.x_fit_r2 < BASE_GATES["minimum_x_fit_r2"],
            "nonlinear_or_unstable_motion",
        ),
        (
            features.height_ratio > BASE_GATES["maximum_height_ratio"],
            "excessive_scale_change",
        ),
        (
            features.edge_fraction > BASE_GATES["maximum_edge_fraction"],
            "track_truncated_at_image_edge",
        ),
        (
            features.reversal_fraction > BASE_GATES["maximum_reversal_fraction"],
            "too_many_direction_reversals",
        ),
        (
            features.scene_motion_support < BASE_GATES["minimum_scene_motion_support"],
            "insufficient_scene_motion_references",
        ),
    ]
    for rejected, reason in checks:
        if rejected:
            return reason
    return ""


def apply_source_context(
    features_by_track: Dict[str, TrackFeatures],
) -> Dict[str, TrackFeatures]:
    """Construct the V31 label-free context from one complete bbox CSV."""
    reference = [
        features
        for features in features_by_track.values()
        if base_rejection_reason(features) == ""
    ]
    reference_count = len(reference)
    minimum_count = int(SOURCE_CONTEXT_SETTINGS["minimum_reference_tracks"])
    context_available = reference_count >= minimum_count
    for features in features_by_track.values():
        features.source_context_tracks = float(reference_count)
        features.source_context_available = 1.0 if context_available else 0.0
        features.source_relative_log_raw_proxy = 0.0
        features.source_relative_log_q_proxy = 0.0
        features.source_relative_log_robust_proxy = 0.0
        features.source_relative_log_compensated_robust_proxy = 0.0
        features.source_compensated_robust_percentile = 0.5
    if not context_available:
        return features_by_track

    minimum_proxy = float(SOURCE_CONTEXT_SETTINGS["minimum_proxy_mps"])
    maximum_log_ratio = float(
        SOURCE_CONTEXT_SETTINGS["maximum_absolute_log_ratio"]
    )
    proxy_fields = {
        "source_relative_log_raw_proxy": "raw_speed_proxy_mps",
        "source_relative_log_q_proxy": "q_speed_proxy_mps",
        "source_relative_log_robust_proxy": "robust_speed_proxy_mps",
        "source_relative_log_compensated_robust_proxy": (
            "compensated_robust_speed_proxy_mps"
        ),
    }
    medians = {
        output_field: float(
            np.median(
                [
                    max(float(getattr(features, input_field)), minimum_proxy)
                    for features in reference
                ]
            )
        )
        for output_field, input_field in proxy_fields.items()
    }
    rank_values = np.asarray(
        [
            max(
                float(features.compensated_robust_speed_proxy_mps),
                minimum_proxy,
            )
            for features in reference
        ],
        dtype=float,
    )
    for features in features_by_track.values():
        for output_field, input_field in proxy_fields.items():
            value = max(float(getattr(features, input_field)), minimum_proxy)
            ratio = math.log(value / max(medians[output_field], minimum_proxy))
            setattr(
                features,
                output_field,
                float(np.clip(ratio, -maximum_log_ratio, maximum_log_ratio)),
            )
        rank_value = max(
            float(features.compensated_robust_speed_proxy_mps),
            minimum_proxy,
        )
        less = float(np.sum(rank_values < rank_value))
        equal = float(np.sum(rank_values == rank_value))
        features.source_compensated_robust_percentile = float(
            (less + 0.5 * equal) / max(len(rank_values), 1)
        )
    return features_by_track


def contextual_track_features(
    tracks: Dict[str, List[BBoxRow]],
    fps: float,
    source_id: str,
    aspect_ratio: float,
    scene_motion_profile: SceneMotionProfile,
) -> Dict[str, TrackFeatures]:
    features_by_track: Dict[str, TrackFeatures] = {}
    for track_id in sorted(tracks, key=str):
        features = track_features(
            tracks[track_id],
            fps,
            source_id,
            aspect_ratio,
            scene_motion_profile,
        )
        if features is not None:
            features_by_track[track_id] = features
    return apply_source_context(features_by_track)


def reliability_rejection_reason(features: TrackFeatures) -> str:
    if features.duration_seconds < RELIABILITY_GATES["minimum_duration_seconds"]:
        return "duration_below_reliable_minimum"
    if features.x_fit_r2 < RELIABILITY_GATES["minimum_x_fit_r2"]:
        return "horizontal_fit_below_reliable_minimum"
    return ""


def _direct_proxy_prediction(features: TrackFeatures, name: str) -> float:
    values = {
        "direct_raw": [features.raw_speed_proxy_mps],
        "direct_q": [features.q_speed_proxy_mps],
        "direct_robust": [features.robust_speed_proxy_mps],
        "direct_median": [
            features.raw_speed_proxy_mps,
            features.q_speed_proxy_mps,
            features.robust_speed_proxy_mps,
        ],
        "direct_compensated_raw": [features.compensated_raw_speed_proxy_mps],
        "direct_compensated_q": [features.compensated_q_speed_proxy_mps],
        "direct_compensated_robust": [features.compensated_robust_speed_proxy_mps],
        "direct_compensated_median": [
            features.compensated_raw_speed_proxy_mps,
            features.compensated_q_speed_proxy_mps,
            features.compensated_robust_speed_proxy_mps,
        ],
    }
    selected = values.get(name)
    if not selected:
        raise ValueError(f"Unsupported direct bbox proxy: {name}")
    return float(np.median(selected))


def _linear_spline_basis(value: float, knots: np.ndarray) -> np.ndarray:
    knot_values = np.asarray(knots, dtype=float).reshape(-1)
    if knot_values.size < 2 or bool(np.any(np.diff(knot_values) <= 0.0)):
        raise ValueError("invalid_monotonic_spline_knots")
    basis = np.zeros(knot_values.size, dtype=float)
    if value <= knot_values[0]:
        basis[0] = 1.0
        return basis
    if value >= knot_values[-1]:
        basis[-1] = 1.0
        return basis
    right = int(np.searchsorted(knot_values, value, side="right"))
    left = right - 1
    fraction = (value - knot_values[left]) / (
        knot_values[right] - knot_values[left]
    )
    basis[left] = 1.0 - fraction
    basis[right] = fraction
    return basis


def _monotonic_spline_prediction_design(
    feature_values: Mapping[str, Any],
    model: Mapping[str, Any],
) -> np.ndarray:
    primary_feature = str(model.get("primary_feature", ""))
    primary_value = float(feature_values[primary_feature])
    knots = np.asarray(model.get("spline_knots", []), dtype=float)
    basis = _linear_spline_basis(primary_value, knots)
    cumulative = (
        np.arange(knots.size)[:, None]
        > np.arange(max(knots.size - 1, 0))[None, :]
    ).astype(float)
    design = np.concatenate([[1.0], basis @ cumulative])
    quality_features = [
        str(value) for value in model.get("quality_features", [])
    ]
    if quality_features:
        quality = np.asarray(
            [float(feature_values[name]) for name in quality_features],
            dtype=float,
        )
        centre = np.asarray(model.get("quality_centre", []), dtype=float)
        scale = np.asarray(model.get("quality_scale", []), dtype=float)
        if len(quality) != len(centre) or len(quality) != len(scale):
            raise ValueError("invalid_monotonic_spline_quality_dimensions")
        scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
        design = np.concatenate([design, (quality - centre) / scale])
    return design


def _bounded_variance_prediction(
    base_prediction: float,
    model: Mapping[str, Any],
) -> Tuple[float, float]:
    prediction_centre = float(model["variance_prediction_centre_mps"])
    target_centre = float(model["variance_target_centre_mps"])
    expansion_factor = float(model["variance_expansion_factor"])
    cap = max(float(model["variance_correction_cap_mps"]), 0.0)
    delta = float(base_prediction) - prediction_centre
    unbounded_correction = (expansion_factor - 1.0) * delta
    correction = float(np.clip(unbounded_correction, -cap, cap))
    derivative = (
        expansion_factor
        if abs(unbounded_correction) < cap
        else 1.0
    )
    return target_centre + delta + correction, derivative


def _extra_trees_prediction(
    raw_vector: np.ndarray,
    model: Mapping[str, Any],
) -> Tuple[float, float]:
    """Evaluate the portable JSON tree ensemble without scikit-learn."""
    predictions: List[float] = []
    trees = model.get("extra_trees", [])
    if not isinstance(trees, list) or not trees:
        raise ValueError("invalid_extra_trees_model")
    for tree in trees:
        if not isinstance(tree, dict):
            raise ValueError("invalid_extra_trees_model")
        left = tree["children_left"]
        right = tree["children_right"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        value = tree["value"]
        node = 0
        while int(left[node]) >= 0:
            feature_index = int(feature[node])
            node = (
                int(left[node])
                if float(raw_vector[feature_index])
                <= float(threshold[node])
                else int(right[node])
            )
        predictions.append(float(value[node]))
    values = np.asarray(predictions, dtype=float)
    dispersion = (
        float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    )
    return float(np.mean(values)), dispersion


def _predict_metric_speed(
    features: TrackFeatures,
    model: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the qualified harness model without importing the CLI module."""
    reason = base_rejection_reason(features) or reliability_rejection_reason(features)
    feature_values = asdict(features)
    feature_names = [str(value) for value in model.get("feature_names", [])]
    try:
        raw_vector = np.asarray(
            [float(feature_values[name]) for name in feature_names],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return {
            "estimated_speed_mps": None,
            "speed_status": "rejected",
            "reject_reason": "unsupported_model_features",
        }
    centre = np.asarray(model.get("feature_centre", []), dtype=float)
    scale = np.asarray(model.get("feature_scale", []), dtype=float)
    coefficients = np.asarray(model.get("coefficients", []), dtype=float)
    if len(raw_vector) != len(centre) or len(raw_vector) != len(scale):
        return {
            "estimated_speed_mps": None,
            "speed_status": "rejected",
            "reject_reason": "invalid_model_dimensions",
        }
    scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
    design = np.concatenate([[1.0], (raw_vector - centre) / scale])
    variance_derivative = 1.0
    tree_prediction_dispersion = 0.0
    if model.get("prediction_mode") == "direct_proxy":
        prediction = _direct_proxy_prediction(
            features,
            str(model.get("direct_proxy_name", "")),
        )
    elif model.get("prediction_mode") in {
        "extra_trees_regression",
        "extra_trees_bounded_variance",
    }:
        try:
            prediction, tree_prediction_dispersion = (
                _extra_trees_prediction(raw_vector, model)
            )
        except (IndexError, KeyError, TypeError, ValueError):
            return {
                "estimated_speed_mps": None,
                "speed_status": "rejected",
                "reject_reason": "invalid_extra_trees_model",
            }
        design = np.empty(0, dtype=float)
        if model.get("prediction_mode") == "extra_trees_bounded_variance":
            try:
                prediction, variance_derivative = (
                    _bounded_variance_prediction(prediction, model)
                )
            except (KeyError, TypeError, ValueError):
                return {
                    "estimated_speed_mps": None,
                    "speed_status": "rejected",
                    "reject_reason": "invalid_variance_calibration_model",
                }
    elif model.get("prediction_mode") in {
        "monotonic_spline_gam",
        "monotonic_spline_physics_blend",
        "bounded_variance_calibration",
    }:
        try:
            design = _monotonic_spline_prediction_design(
                feature_values,
                model,
            )
        except (KeyError, TypeError, ValueError):
            return {
                "estimated_speed_mps": None,
                "speed_status": "rejected",
                "reject_reason": "invalid_monotonic_spline_model",
            }
        if len(coefficients) != len(design):
            return {
                "estimated_speed_mps": None,
                "speed_status": "rejected",
                "reject_reason": "invalid_model_coefficients",
            }
        spline_prediction = float(design @ coefficients)
        if model.get("prediction_mode") in {
            "monotonic_spline_physics_blend",
            "bounded_variance_calibration",
        }:
            blend_weight = float(model.get("blend_weight", 0.0))
            blend_proxy_feature = str(model.get("blend_proxy_feature", ""))
            try:
                proxy_prediction = float(
                    feature_values[blend_proxy_feature]
                )
            except (KeyError, TypeError, ValueError):
                return {
                    "estimated_speed_mps": None,
                    "speed_status": "rejected",
                    "reject_reason": "invalid_physics_blend_model",
                }
            prediction = float(
                (1.0 - blend_weight) * spline_prediction
                + blend_weight * proxy_prediction
            )
            if model.get("prediction_mode") == (
                "bounded_variance_calibration"
            ):
                try:
                    prediction, variance_derivative = (
                        _bounded_variance_prediction(prediction, model)
                    )
                except (KeyError, TypeError, ValueError):
                    return {
                        "estimated_speed_mps": None,
                        "speed_status": "rejected",
                        "reject_reason": "invalid_variance_calibration_model",
                    }
        else:
            prediction = spline_prediction
    else:
        if len(coefficients) != len(design):
            return {
                "estimated_speed_mps": None,
                "speed_status": "rejected",
                "reject_reason": "invalid_model_coefficients",
            }
        prediction = float(design @ coefficients)
    lower = np.asarray(model.get("feature_lower_bound", []), dtype=float)
    upper = np.asarray(model.get("feature_upper_bound", []), dtype=float)
    if (
        not reason
        and len(lower) == len(raw_vector)
        and len(upper) == len(raw_vector)
        and bool(np.any((raw_vector < lower) | (raw_vector > upper)))
    ):
        reason = "out_of_calibration_distribution"
    covariance = np.asarray(model.get("covariance_basis", []), dtype=float)
    residual = float(model.get("residual_sigma_mps", 0.0))
    leverage = (
        max(0.0, float(design @ covariance @ design))
        if covariance.shape == (len(design), len(design))
        else 0.0
    )
    if model.get("prediction_mode") in {
        "monotonic_spline_physics_blend",
        "bounded_variance_calibration",
    }:
        spline_weight = 1.0 - float(model.get("blend_weight", 0.0))
        leverage *= spline_weight * spline_weight
    leverage *= variance_derivative * variance_derivative
    uncertainty = max(
        residual * math.sqrt(1.0 + leverage),
        tree_prediction_dispersion
        / math.sqrt(
            max(float(model.get("extra_trees_n_estimators", 1)), 1.0)
        ),
        float(model.get("conformal_absolute_uncertainty_mps", 0.0)),
    )
    maximum_uncertainty = safe_float(model.get("maximum_absolute_uncertainty_mps"))
    if not reason and maximum_uncertainty is not None and uncertainty > maximum_uncertainty:
        reason = "absolute_uncertainty_too_high"
    gates = model.get("gates") if isinstance(model.get("gates"), dict) else {}
    minimum_speed = float(gates.get("minimum_predicted_speed_mps", 0.10))
    maximum_speed = float(gates.get("maximum_predicted_speed_mps", 3.50))
    if not reason and prediction < minimum_speed:
        reason = "predicted_speed_too_low"
    if not reason and prediction > maximum_speed:
        reason = "predicted_speed_too_high"
    return {
        "estimated_speed_mps": None if reason else float(prediction),
        "speed_uncertainty_mps": float(uncertainty),
        "speed_status": "rejected" if reason else "valid",
        "reject_reason": reason,
    }


def _track_features_from_mapping(row: Mapping[str, Any]) -> Optional[TrackFeatures]:
    integer_fields = {
        "input_rows",
        "clean_rows",
        "removed_rows",
        "first_frame",
        "last_frame",
    }
    boolean_fields = {"crosses_image_midline", "overlaps_central_band"}
    text_fields = {"source_id", "prediction_track_id", "direction"}
    values: Dict[str, Any] = {}
    for name in TrackFeatures.__dataclass_fields__:
        raw = row.get(name)
        if name in text_fields:
            values[name] = str(raw or "")
        elif name in boolean_fields:
            values[name] = truthy(raw, False)
        elif name in integer_fields:
            parsed = safe_int(raw)
            if parsed is None:
                return None
            values[name] = parsed
        else:
            parsed = safe_float(raw)
            if parsed is None:
                return None
            values[name] = parsed
    try:
        return TrackFeatures(**values)
    except TypeError:
        return None


def predict_relative_rows(
    bbox_csv: os.PathLike[str] | str,
    fps: float,
    source_id: str,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return the v22 style within video motion table and person track count."""
    return predict_relative_bbox_rows(
        load_bbox_csv(bbox_csv),
        fps,
        source_id,
        aspect_ratio,
    )


def predict_relative_bbox_rows(
    rows: Sequence[BBoxRow],
    fps: float,
    source_id: str,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> Tuple[List[Dict[str, Any]], int]:
    """Calculate relative motion from already loaded bounding box rows."""
    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError(f"FPS must be positive, received {fps}")
    person_tracks = group_tracks(row for row in rows if row.class_id == PERSON_CLASS_ID)
    scene_profile = build_scene_motion_profile(rows, fps)
    features_by_track = contextual_track_features(
        person_tracks,
        fps,
        source_id,
        aspect_ratio,
        scene_profile,
    )
    prepared: List[Tuple[TrackFeatures, float, str]] = []
    for track_id in sorted(features_by_track, key=str):
        features = features_by_track[track_id]
        if features.scene_motion_support >= int(
            SCENE_MOTION_SETTINGS["minimum_reference_tracks"]
        ):
            proxies = [
                features.compensated_raw_speed_proxy_mps,
                features.compensated_q_speed_proxy_mps,
                features.compensated_robust_speed_proxy_mps,
            ]
            role = "scene_motion_compensated_bbox_proxy"
        else:
            proxies = [
                features.raw_speed_proxy_mps,
                features.q_speed_proxy_mps,
                features.robust_speed_proxy_mps,
            ]
            role = "uncompensated_bbox_proxy"
        prepared.append((features, float(np.median(proxies)), role))

    eligible = [
        proxy
        for features, proxy, _ in prepared
        if not base_rejection_reason(features) and proxy > 0.0
    ]
    reference = float(np.median(eligible)) if eligible else None
    logs = np.log(np.maximum(np.asarray(eligible, dtype=float), 1e-9))
    log_centre = float(np.median(logs)) if len(logs) else None
    log_scale = robust_scale(logs) if len(logs) >= 3 else None
    output: List[Dict[str, Any]] = []
    for features, proxy, role in prepared:
        reason = base_rejection_reason(features)
        if not reason and len(eligible) < 3:
            reason = "too_few_comparable_tracks"
        relative_index = (
            proxy / reference if reference is not None and reference > 0.0 else None
        )
        robust_z = (
            (math.log(max(proxy, 1e-9)) - float(log_centre)) / float(log_scale)
            if log_centre is not None and log_scale is not None and log_scale > 1e-8
            else None
        )
        category = ""
        if not reason:
            if robust_z is None:
                category = "typical_within_video"
            elif robust_z <= -0.75:
                category = "slower_within_video"
            elif robust_z >= 0.75:
                category = "faster_within_video"
            else:
                category = "typical_within_video"
        row: Dict[str, Any] = {
            "source_id": source_id,
            "prediction_track_id": features.prediction_track_id,
            "bbox_motion_proxy": proxy,
            "relative_motion_index": relative_index,
            "relative_motion_robust_z": robust_z,
            "relative_motion_category": category,
            "relative_motion_status": "valid" if not reason else "rejected",
            "reject_reason": reason,
            "proxy_role": role,
            "speed_interpretation": "within-video bbox motion only; not metres per second",
            "comparable_tracks": len(eligible),
        }
        row.update(asdict(features))
        reliability_reason = reliability_rejection_reason(features)
        row["base_status"] = "rejected" if base_rejection_reason(features) else "eligible"
        row["base_reject_reason"] = base_rejection_reason(features)
        row["reliability_status"] = "rejected" if reliability_reason else "eligible"
        row["reliability_reject_reason"] = reliability_reason
        output.append(row)
    return output, len(person_tracks)


def _numeric_values(rows: Sequence[Mapping[str, Any]], field: str) -> List[float]:
    output: List[float] = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            output.append(value)
    return output


def _quantile(values: Sequence[float], fraction: float) -> Optional[float]:
    return float(np.quantile(np.asarray(values, dtype=float), fraction)) if values else None


def _source_cache_is_current(cache_path: Path, input_path: Path) -> bool:
    try:
        if not (
            cache_path.is_file()
            and cache_path.stat().st_mtime >= input_path.stat().st_mtime
        ):
            return False
        with cache_path.open("r", encoding="utf-8-sig") as handle:
            header = handle.readline()
        return "source_relative_log_q_proxy" in header
    except OSError:
        return False


def analyse_crowd_sources(
    sources: Sequence[Mapping[str, Any]],
    output_root: Path,
    force: bool,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    cache_root = output_root / "crowd_relative_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    track_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, source in enumerate(sources, start=1):
        source_id = str(source.get("source_id", "")).strip()
        bbox_path = Path(str(source.get("bbox_csv", ""))).expanduser()
        fps = safe_float(source.get("fps"))
        aspect_ratio = safe_float(source.get("aspect_ratio")) or DEFAULT_ASPECT_RATIO
        crossing_ids = {
            normalise_id(value)
            for value in source.get("crossing_track_ids", [])
            if normalise_id(value)
        }
        base_source = {
            "source_id": source_id,
            "locality": str(source.get("locality", "Unknown")),
            "state": str(source.get("state", "")),
            "country": str(source.get("country", "Unknown")),
            "fps": fps,
            "bbox_csv": str(bbox_path),
        }
        if not source_id or not bbox_path.is_file() or fps is None or fps <= 0.0:
            failure = {
                "source_id": source_id or f"row_{index}",
                "reason": "missing_bbox_csv_or_fps",
            }
            failures.append(failure)
            log(f"CROWD speed report skipped {failure['source_id']}: {failure['reason']}")
            continue
        cache_path = cache_root / f"{safe_name(source_id)}_relative_motion.csv"
        try:
            memory_result = None if force else _RELATIVE_MEMORY_CACHE.get(source_id)
            if memory_result is not None:
                relative_rows, person_track_count = memory_result
                write_dict_csv(cache_path, relative_rows)
            elif force or not _source_cache_is_current(cache_path, bbox_path):
                relative_rows, person_track_count = predict_relative_rows(
                    bbox_path, fps, source_id, aspect_ratio
                )
                write_dict_csv(cache_path, relative_rows)
            else:
                relative_rows = read_dict_csv(cache_path)
                person_track_count = len(
                    {
                        normalise_id(row.track_id)
                        for row in load_bbox_csv(bbox_path)
                        if row.class_id == PERSON_CLASS_ID
                    }
                )
        except Exception as error:
            failures.append({"source_id": source_id, "reason": str(error)})
            log(f"CROWD speed report failed for {source_id}: {error}")
            continue
        relative_by_track = {
            normalise_id(row.get("prediction_track_id")): dict(row)
            for row in relative_rows
            if normalise_id(row.get("prediction_track_id"))
        }
        valid_count = 0
        reliable_speed_count = 0
        for track_id in sorted(crossing_ids, key=str):
            row = relative_by_track.get(track_id)
            if row is None:
                row = {
                    "source_id": source_id,
                    "prediction_track_id": track_id,
                    "relative_motion_status": "rejected",
                    "reject_reason": "feature_extraction_failed",
                    "speed_interpretation": "within-video bbox motion only; not metres per second",
                }
            if _SPEED_MODEL:
                features = _track_features_from_mapping(row)
                prediction = (
                    _predict_metric_speed(features, _SPEED_MODEL)
                    if features is not None
                    else {
                        "estimated_speed_mps": None,
                        "speed_status": "rejected",
                        "reject_reason": "feature_extraction_failed",
                    }
                )
                row = {
                    **row,
                    "estimated_crossing_speed_mps": prediction.get("estimated_speed_mps"),
                    "speed_uncertainty_mps": prediction.get("speed_uncertainty_mps"),
                    "speed_status": prediction.get("speed_status"),
                    "speed_reject_reason": prediction.get("reject_reason"),
                    "metric_speed_qualified": 1,
                }
                if prediction.get("speed_status") == "valid":
                    reliable_speed_count += 1
            else:
                row = {
                    **row,
                    "estimated_crossing_speed_mps": None,
                    "speed_uncertainty_mps": None,
                    "speed_status": "unavailable",
                    "speed_reject_reason": "waymo_speed_model_not_qualified",
                    "metric_speed_qualified": 0,
                }
            output_row = {**base_source, **row, "detected_crossing": 1}
            output_row["prediction_track_id"] = track_id
            track_rows.append(output_row)
            if str(output_row.get("relative_motion_status", "")) == "valid":
                valid_count += 1
        source_rows.append(
            {
                **base_source,
                "person_tracks": person_track_count,
                "detected_crossing_tracks": len(crossing_ids),
                "analysed_crossing_tracks": valid_count,
                "rejected_crossing_tracks": len(crossing_ids) - valid_count,
                "reliable_metric_speed_tracks": reliable_speed_count,
            }
        )
        if index % 50 == 0 or index == len(sources):
            log(f"CROWD crossing speed analysed {index}/{len(sources)} bbox CSV files")

    by_city_sources: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_city_tracks: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        key = (str(row["locality"]), str(row["state"]), str(row["country"]))
        by_city_sources[key].append(row)
    for row in track_rows:
        key = (str(row["locality"]), str(row["state"]), str(row["country"]))
        by_city_tracks[key].append(row)

    city_rows: List[Dict[str, Any]] = []
    for key in sorted(by_city_sources, key=lambda item: (item[2], item[0], item[1])):
        locality, state, country = key
        city_sources = by_city_sources[key]
        city_tracks = by_city_tracks.get(key, [])
        valid = [
            row
            for row in city_tracks
            if str(row.get("relative_motion_status", "")) == "valid"
        ]
        values = _numeric_values(valid, "relative_motion_index")
        proxies = _numeric_values(valid, "bbox_motion_proxy")
        reliable_speed_rows = [
            row for row in city_tracks if str(row.get("speed_status", "")) == "valid"
        ]
        speed_values = _numeric_values(
            reliable_speed_rows,
            "estimated_crossing_speed_mps",
        )
        categories = Counter(str(row.get("relative_motion_category", "")) for row in valid)
        rejection_counts = Counter(
            str(row.get("reject_reason", "unknown")) or "unknown"
            for row in city_tracks
            if str(row.get("relative_motion_status", "")) != "valid"
        )
        crossing_count = sum(int(row["detected_crossing_tracks"]) for row in city_sources)
        analysed_count = len(valid)
        city_rows.append(
            {
                "locality": locality,
                "state": state,
                "country": country,
                "source_videos": len(city_sources),
                "person_tracks": sum(int(row["person_tracks"]) for row in city_sources),
                "detected_crossing_tracks": crossing_count,
                "analysed_crossing_tracks": analysed_count,
                "rejected_crossing_tracks": crossing_count - analysed_count,
                "analysis_coverage": analysed_count / crossing_count if crossing_count else None,
                "median_relative_motion_index": float(np.median(values)) if values else None,
                "relative_motion_index_q25": _quantile(values, 0.25),
                "relative_motion_index_q75": _quantile(values, 0.75),
                "median_bbox_motion_proxy": float(np.median(proxies)) if proxies else None,
                "reliable_metric_speed_tracks": len(reliable_speed_rows),
                "metric_speed_coverage": (
                    len(reliable_speed_rows) / crossing_count if crossing_count else None
                ),
                "median_crossing_speed_mps": (
                    float(np.median(speed_values)) if speed_values else None
                ),
                "crossing_speed_mps_q25": _quantile(speed_values, 0.25),
                "crossing_speed_mps_q75": _quantile(speed_values, 0.75),
                "slower_within_video": categories.get("slower_within_video", 0),
                "typical_within_video": categories.get("typical_within_video", 0),
                "faster_within_video": categories.get("faster_within_video", 0),
                "top_rejection_reason": rejection_counts.most_common(1)[0][0]
                if rejection_counts
                else "",
                "motion_unit": "m/s" if _SPEED_MODEL else "relative index",
                "interpretation": (
                    "Waymo qualified bbox crossing speed"
                    if _SPEED_MODEL
                    else "within-video bbox motion only; not metres per second"
                ),
            }
        )

    write_dict_csv(output_root / "crowd_crossing_relative_motion_tracks.csv", track_rows)
    write_dict_csv(output_root / "crowd_crossing_relative_motion_sources.csv", source_rows)
    write_dict_csv(output_root / "crowd_crossing_relative_motion_by_city.csv", city_rows)
    write_dict_csv(output_root / "crowd_crossing_speed_tracks.csv", track_rows)
    write_dict_csv(output_root / "crowd_crossing_speed_by_city.csv", city_rows)
    valid_tracks = [
        row for row in track_rows if str(row.get("relative_motion_status", "")) == "valid"
    ]
    rejections = Counter(
        str(row.get("reject_reason", "unknown")) or "unknown"
        for row in track_rows
        if str(row.get("relative_motion_status", "")) != "valid"
    )
    summary = {
        "status": "complete" if source_rows else "no_crowd_sources",
        "source_files_requested": len(sources),
        "source_files_analysed": len(source_rows),
        "cities_analysed": len(city_rows),
        "person_tracks": sum(int(row["person_tracks"]) for row in source_rows),
        "detected_crossing_tracks": len(track_rows),
        "valid_relative_motion_tracks": len(valid_tracks),
        "reliable_metric_speed_tracks": sum(
            str(row.get("speed_status", "")) == "valid" for row in track_rows
        ),
        "coverage": len(valid_tracks) / len(track_rows) if track_rows else 0.0,
        "relative_motion_categories": dict(
            Counter(str(row.get("relative_motion_category", "")) for row in valid_tracks)
        ),
        "rejection_counts": dict(rejections),
        "failures": failures,
        "speed_unit": "m/s" if _SPEED_MODEL else None,
        "metric_speed_qualified": bool(_SPEED_MODEL),
        "interpretation": (
            "Waymo qualified metric speed applied to CROWD crossing tracks"
            if _SPEED_MODEL
            else "CROWD bbox CSV supports relative apparent crossing motion; the Waymo speed model did not qualify"
        ),
    }
    write_json(output_root / "crowd_crossing_relative_motion_summary.json", summary)
    return {"summary": summary, "city_rows": city_rows, "track_rows": track_rows}


def _read_csv_from_zip(path: Path) -> Iterable[List[Dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv") or "/._" in name or name.startswith("._"):
                continue
            with archive.open(name) as raw:
                text = (line.decode("utf-8-sig") for line in raw)
                yield list(csv.DictReader(text))


def _candidate_role(path: Path) -> str:
    return "validation" if "validation" in str(path).lower() else "training"


def _manifest_score(path: Path) -> Tuple[int, int]:
    name = path.name.lower()
    preferred = int("crosswalk_axis" in name or "full_training" in name)
    try:
        rows = len(read_dict_csv(path))
    except Exception:
        rows = -1
    return preferred, rows


def _index_role_and_score(path: Path) -> Tuple[str, Tuple[int, int, int]]:
    """Infer the split from index content and prefer the current crosswalk target."""
    try:
        rows = read_dict_csv(path)
    except Exception:
        return _candidate_role(path), (0, 0, -1)
    sample = rows[0] if rows else {}
    searchable = " ".join(
        [
            str(path),
            str(sample.get("video_path", "")),
            str(sample.get("ground_truth_bbox_csv", "")),
        ]
    ).lower()
    role = "validation" if "validation" in searchable else "training"
    target = str(sample.get("ground_truth_target", "")).lower()
    target_score = int("crosswalk traversal axis" in target)
    confirmed = sum(int(safe_float(row.get("ground_truth_tracks")) or 0) for row in rows)
    return role, (target_score, confirmed, len(rows))


def _resolve_waymo_output_path(
    repository_root: Path,
    index_path: Path,
    value: Any,
) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()

    # Waymo exports are deliberately relocatable.  Older indices can contain
    # paths rooted at Docker's /workspace or at the former repository
    # _output/waymo_processed directory.  If the complete waymo_processed
    # directory has been moved beside the raw TFRecords, recover the suffix
    # below the training/validation directory before trying the old path.
    split_name = index_path.parent.name
    candidate_parts = candidate.parts
    if split_name in candidate_parts:
        split_position = len(candidate_parts) - 1 - list(
            reversed(candidate_parts)
        ).index(split_name)
        relocated = index_path.parent.joinpath(
            *candidate_parts[split_position + 1:]
        )
        if relocated.exists() or relocated.parent.exists():
            return relocated

    # Pre-move indices may contain repository-relative
    # data/speed_calibration/.../<source>/<file> values even though the whole
    # processed tree now lives beside the raw Waymo dataset.
    if len(candidate_parts) >= 2:
        relocated = index_path.parent / candidate_parts[-2] / candidate_parts[-1]
        if relocated.exists() or relocated.parent.exists():
            return relocated

    if candidate.is_absolute():
        try:
            relative = candidate.relative_to("/workspace")
        except ValueError:
            return candidate
        return repository_root / relative
    repository_candidate = repository_root / candidate
    if repository_candidate.exists():
        return repository_candidate
    return index_path.parent / candidate


def _raw_waymo_files(split_root: Path) -> List[Path]:
    if not split_root.is_dir():
        return []
    return sorted(
        path
        for path in split_root.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and (path.name.endswith(".tfrecord") or ".tfrecord-" in path.name)
    )


def _waymo_export_is_ready(
    repository_root: Path,
    raw_split_root: Path,
    processed_split_root: Path,
) -> bool:
    index_path = processed_split_root / "waymo_sequence_index.csv"
    raw_count = len(_raw_waymo_files(raw_split_root))
    if not index_path.is_file():
        return False

    summary_path = processed_split_root / "waymo_export_summary.json"
    try:
        export_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        index_rows = read_dict_csv(index_path)
    except (OSError, ValueError):
        return False
    selected = safe_int(export_summary.get("files_selected"))
    completed = safe_int(export_summary.get("files_completed"))
    failed = safe_int(export_summary.get("files_failed"))
    if raw_count and selected != raw_count:
        return False
    if selected is None or completed != selected or (failed or 0) != 0:
        return False

    return bool(index_rows)


def _current_tracking_settings() -> Dict[str, Any]:
    """Return the tracking settings from the harness, which reads config."""
    import speed_estimation_harness

    return speed_estimation_harness.current_tracking_settings()


def _waymo_tracking_is_ready(
    repository_root: Path,
    processed_split_root: Path,
) -> bool:
    """Every exported video needs a prediction CSV made with the current settings."""
    import speed_estimation_harness

    index_path = processed_split_root / "waymo_sequence_index.csv"
    if not index_path.is_file():
        return False
    try:
        index_rows = read_dict_csv(index_path)
    except OSError:
        return False
    for row in index_rows:
        prediction = _resolve_waymo_output_path(
            repository_root,
            index_path,
            row.get("prediction_bbox_csv"),
        )
        if prediction is None or not speed_estimation_harness.tracking_is_current(prediction):
            return False
    return bool(index_rows)


def _waymo_split_is_ready(
    repository_root: Path,
    raw_split_root: Path,
    processed_split_root: Path,
    split_name: str,
) -> bool:
    del split_name
    return _waymo_export_is_ready(
        repository_root,
        raw_split_root,
        processed_split_root,
    ) and _waymo_tracking_is_ready(repository_root, processed_split_root)


def _run_checked(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=str(cwd), check=True)


def _docker_waymo_export(
    repository_root: Path,
    raw_dataset_root: Path,
    processed_split_root: Path,
    split_name: str,
    log: Callable[[str], None],
) -> None:
    """Export Waymo into the dataset adjacent waymo_processed directory."""
    raw_mount_root = raw_dataset_root.parent
    raw_container_path = Path("/waymo_source") / raw_dataset_root.name / split_name
    try:
        output_relative = processed_split_root.relative_to(raw_mount_root)
    except ValueError as error:
        raise RuntimeError(
            "The Waymo processed directory must be inside the configured "
            f"dataset mount {raw_mount_root}: {processed_split_root}"
        ) from error
    output_container_path = Path("/waymo_source") / output_relative
    export_command = " ".join(
        [
            "python3 speed_estimation_harness.py waymo_export",
            shlex.quote(str(raw_container_path)),
            shlex.quote(str(output_container_path)),
            "FRONT 10 0 false",
        ]
    )
    container_command = (
        "pip install --no-cache-dir --no-deps "
        "waymo-open-dataset-tf-2-12-0==1.6.4 && "
        "pip install --no-cache-dir --no-deps "
        "opencv-python-headless==4.8.1.78 && "
        + export_command
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{raw_mount_root}:/waymo_source",
        "-v",
        f"{repository_root}:/workspace",
        "-w",
        "/workspace",
        "tensorflow/tensorflow:2.12.0",
        "bash",
        "-lc",
        container_command,
    ]
    log(f"Exporting raw Waymo {split_name} TFRecords with Docker")
    _run_checked(command, repository_root)


# Packages the raw TFRecord export needs, pinned to exactly what the Docker
# route installs into tensorflow/tensorflow:2.12.0, so both routes produce the
# same export. The Waymo package is installed without its dependencies, as in
# the container, because its own pins conflict with TensorFlow 2.12.
WAYMO_EXPORT_PYTHON = "3.10"
WAYMO_EXPORT_PACKAGES = ("tensorflow==2.12.0", "opencv-python-headless==4.8.1.78")
WAYMO_EXPORT_NO_DEPS_PACKAGES = ("waymo-open-dataset-tf-2-12-0==1.6.4",)
WAYMO_EXPORT_ENV_DIR = ".waymo_export_venv"


def _native_waymo_export_python(repository_root: Path, log: Callable[[str], None]) -> Path:
    """Return the interpreter of a local environment able to run the export.

    The environment is created once with uv and reused afterwards. It lives in
    the repository rather than beside the Waymo data because that data often
    sits on an exFAT drive, which cannot hold a virtualenv's symlinks.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Neither Docker nor uv is available, so the raw Waymo TFRecords "
            "cannot be exported. Install uv (https://docs.astral.sh/uv/) or Docker."
        )
    env_root = repository_root / WAYMO_EXPORT_ENV_DIR
    python = env_root / "bin" / "python"
    marker = env_root / "crowd_export_packages.json"
    wanted = {
        "python": WAYMO_EXPORT_PYTHON,
        "packages": list(WAYMO_EXPORT_PACKAGES),
        "no_deps_packages": list(WAYMO_EXPORT_NO_DEPS_PACKAGES),
    }
    try:
        if python.is_file() and json.loads(marker.read_text(encoding="utf-8")) == wanted:
            return python
    except (OSError, ValueError):
        pass

    log(
        f"Creating the Waymo export environment in {env_root} "
        f"(Python {WAYMO_EXPORT_PYTHON}, {', '.join(WAYMO_EXPORT_PACKAGES + WAYMO_EXPORT_NO_DEPS_PACKAGES)})"
    )
    # --no-config keeps the repository's own uv settings out of this separate
    # environment: pyproject.toml deliberately blocks opencv-python-headless,
    # which the export needs here.
    _run_checked(
        [uv, "venv", "--no-config", "--clear", "--python", WAYMO_EXPORT_PYTHON, str(env_root)],
        repository_root,
    )
    _run_checked(
        [uv, "pip", "install", "--no-config", "--python", str(python), *WAYMO_EXPORT_PACKAGES],
        repository_root,
    )
    _run_checked(
        [
            uv, "pip", "install", "--no-config", "--python", str(python),
            "--no-deps", *WAYMO_EXPORT_NO_DEPS_PACKAGES,
        ],
        repository_root,
    )
    # Only mark the environment ready once the export's imports really work.
    _run_checked(
        [
            str(python), "-c",
            "import cv2, tensorflow; from waymo_open_dataset import dataset_pb2, label_pb2",
        ],
        repository_root,
    )
    write_json(marker, wanted)
    return python


def _native_waymo_export(
    repository_root: Path,
    raw_dataset_root: Path,
    processed_split_root: Path,
    split_name: str,
    log: Callable[[str], None],
) -> None:
    """Export Waymo without Docker, using a local TensorFlow 2.12 environment."""
    python = _native_waymo_export_python(repository_root, log)
    log(f"Exporting raw Waymo {split_name} TFRecords without Docker")
    _run_checked(
        [
            str(python),
            str(repository_root / "speed_estimation_harness.py"),
            "waymo_export",
            str(raw_dataset_root / split_name),
            str(processed_split_root),
            "FRONT",
            "10",
            "0",
            "false",
        ],
        repository_root,
    )


def _docker_is_usable() -> bool:
    """Return whether the Docker CLI exists and its daemon answers."""
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        subprocess.run([docker, "info"], capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _waymo_export(
    repository_root: Path,
    raw_dataset_root: Path,
    processed_split_root: Path,
    split_name: str,
    log: Callable[[str], None],
) -> None:
    """Export one Waymo split with Docker when it works, otherwise natively."""
    if _docker_is_usable():
        _docker_waymo_export(repository_root, raw_dataset_root, processed_split_root, split_name, log)
    else:
        _native_waymo_export(repository_root, raw_dataset_root, processed_split_root, split_name, log)


def ensure_waymo_processed(
    raw_dataset_path: os.PathLike[str] | str,
    repository_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    process_if_missing: bool,
    log: Optional[Callable[[str], None]] = None,
) -> List[Path]:
    """Resume Waymo export/tracking, calibrate on training, then load the model."""
    logger = log or _default_log
    repository = Path(repository_root).expanduser().resolve()
    raw_path_text = os.path.expandvars(str(raw_dataset_path or "")).strip()
    raw_root = (
        Path(raw_path_text).expanduser().resolve()
        if raw_path_text
        else repository / "__waymo_dataset_not_configured__"
    )
    # Keep raw Waymo data and every derived artefact together.  When Waymo is
    # configured this resolves to <waymo_dataset_path>/waymo_processed.  The
    # repository output fallback is retained only for CROWD only installations
    # where no Waymo dataset path has been configured.
    processed_root = (
        raw_root / "waymo_processed"
        if raw_path_text
        else Path(output_root).expanduser().resolve() / "waymo_processed"
    )
    split_roots = {
        split: processed_root / split for split in ("training", "validation")
    }
    calibration_root = processed_root / "calibration_v32"
    pipeline_model_path = calibration_root / "crowd_waymo_pipeline_model.json"
    # Records the tracking settings the calibration was fitted on. A model
    # fitted on tracks from other settings is neither loaded nor reused.
    calibration_settings_path = calibration_root / "tracking_settings.json"

    def calibration_record() -> Dict[str, Any]:
        from crowd_city.crossing.waymo_calibration import CALIBRATION_BUILD_ID

        return {"tracking": _current_tracking_settings(), "calibration_build_id": CALIBRATION_BUILD_ID}

    def calibration_matches_tracking() -> bool:
        try:
            recorded = json.loads(calibration_settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return recorded == calibration_record()

    def log_speed_statistics() -> None:
        summary_path = (
            calibration_root
            / "figures"
            / "waymo_speed_distribution_statistics.json"
        )
        summary_pickle_path = (
            calibration_root / "figures" / "waymo_diagnostics.pickle"
        )
        summary: Dict[str, Any] = {}
        if summary_pickle_path.is_file():
            try:
                with summary_pickle_path.open("rb") as handle:
                    loaded_summary = pickle.load(handle)
                if isinstance(loaded_summary, dict):
                    summary = loaded_summary
            except (
                OSError,
                EOFError,
                AttributeError,
                ImportError,
                IndexError,
                ValueError,
                pickle.PickleError,
            ):
                summary = {}
        if not summary and summary_path.is_file():
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    loaded_summary = json.load(handle)
                if isinstance(loaded_summary, dict):
                    summary = loaded_summary
            except (OSError, ValueError):
                summary = {}
        if not summary:
            return

        def format_value(value: Any) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "NA"
            return f"{numeric:.2f}" if math.isfinite(numeric) else "NA"

        roles = summary.get("roles", {})
        validation_role_error_logged = False
        for role in (
            "training_fit",
            "training_cross_validation",
            "untouched_validation_test",
        ):
            payload = roles.get(role, {})
            if not isinstance(payload, Mapping):
                continue
            reference = payload.get("waymo_reference", {})
            estimate = payload.get("algorithm_estimate", {})
            error = payload.get("signed_error", {})
            count = reference.get("count", 0)
            if not count:
                continue
            label = payload.get("label", role.replace("_", " "))
            logger(
                f"Waymo speed distribution [{label}]: n={count}; "
                "reference "
                f"mean={format_value(reference.get('mean_mps'))}, "
                f"SD={format_value(reference.get('sample_sd_mps'))}, "
                f"median={format_value(reference.get('median_mps'))}, "
                f"IQR={format_value(reference.get('iqr_mps'))} m/s; "
                "algorithm "
                f"mean={format_value(estimate.get('mean_mps'))}, "
                f"SD={format_value(estimate.get('sample_sd_mps'))}, "
                f"median={format_value(estimate.get('median_mps'))}, "
                f"IQR={format_value(estimate.get('iqr_mps'))} m/s."
            )
            logger(
                f"Waymo signed error [{label}]: "
                f"mean={format_value(error.get('mean_mps'))}, "
                f"SD={format_value(error.get('sample_sd_mps'))}, "
                f"median={format_value(error.get('median_mps'))}, "
                f"IQR={format_value(error.get('iqr_mps'))}, "
                f"minimum={format_value(error.get('minimum_mps'))}, "
                f"maximum={format_value(error.get('maximum_mps'))} m/s."
            )
            role_error_metrics = payload.get("error_metrics", {})
            if isinstance(role_error_metrics, Mapping) and role_error_metrics:
                role_error_label = {
                    "training_fit": "Waymo training fit error",
                    "training_cross_validation": (
                        "Waymo test error "
                        "[source held out training cross validation]"
                    ),
                    "untouched_validation_test": (
                        "Waymo untouched validation error"
                    ),
                }.get(role, f"Waymo speed error [{label}]")
                try:
                    logger(
                        f"{role_error_label}: "
                        f"n={role_error_metrics.get('count', 0)}, "
                        "MAE="
                        f"{format_value(role_error_metrics.get('mae_mps'))}, "
                        "RMSE="
                        f"{format_value(role_error_metrics.get('rmse_mps'))}, "
                        "bias="
                        f"{format_value(role_error_metrics.get('bias_mps'))}, "
                        "median absolute error="
                        f"{format_value(role_error_metrics.get('median_absolute_error_mps'))} m/s, "
                        "within 0.25 m/s="
                        f"{format_value(100 * float(role_error_metrics.get('within_0_25_mps', 0)))}%, "
                        "within 0.50 m/s="
                        f"{format_value(100 * float(role_error_metrics.get('within_0_50_mps', 0)))}%."
                    )
                    if role == "untouched_validation_test":
                        validation_role_error_logged = True
                except (TypeError, ValueError):
                    pass

        metrics = summary.get("validation_error_metrics", {})
        validation_metrics_path = (
            calibration_root
            / "figures"
            / "waymo_validation_speed_metrics.json"
        )
        if not metrics and validation_metrics_path.is_file():
            try:
                with validation_metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
            except (OSError, ValueError):
                metrics = {}
        if metrics and not validation_role_error_logged:
            try:
                logger(
                    "Waymo untouched validation error: "
                    f"n={metrics.get('count', 0)}, "
                    f"MAE={format_value(metrics.get('mae_mps'))}, "
                    f"RMSE={format_value(metrics.get('rmse_mps'))}, "
                    f"bias={format_value(metrics.get('bias_mps'))}, "
                    "median absolute error="
                    f"{format_value(metrics.get('median_absolute_error_mps'))} m/s, "
                    "within 0.25 m/s="
                    f"{format_value(100 * float(metrics.get('within_0_25_mps', 0)))}%, "
                    "within 0.50 m/s="
                    f"{format_value(100 * float(metrics.get('within_0_50_mps', 0)))}%."
                )
            except (TypeError, ValueError):
                pass

    if pipeline_model_path.is_file() and not calibration_matches_tracking():
        logger(
            "The frozen Waymo calibration was built from other tracking settings or "
            f"calibration code than {calibration_record()}; it will be rebuilt."
        )
    elif pipeline_model_path.is_file():
        load_tuned_pipeline_model(pipeline_model_path)
        try:
            from crowd_city.crossing.waymo_calibration import (
                refresh_waymo_diagnostic_figures,
            )

            refreshed_figures = refresh_waymo_diagnostic_figures(calibration_root)
            if refreshed_figures:
                logger(
                    "Refreshed Waymo validation speed diagnostics: "
                    f"{calibration_root / 'figures'}"
                )
            log_speed_statistics()
        except (OSError, ValueError) as error:
            logger(f"Waymo diagnostic figure refresh failed: {error}")
        if raw_path_text and all(
            _waymo_split_is_ready(
                repository,
                raw_root / split,
                processed_split,
                split,
            )
            for split, processed_split in split_roots.items()
        ):
            logger(f"Reusing frozen Waymo calibration: {pipeline_model_path}")
            return [processed_root]
    if not process_if_missing:
        if processed_root.exists():
            logger(f"Waymo processing disabled; using available files in {processed_root}")
            return [processed_root]
        logger("Waymo raw processing is disabled in config")
        return []
    if not raw_path_text:
        logger("Waymo raw dataset path is not configured")
        return []
    if not raw_root.is_dir():
        logger(f"Waymo raw dataset path does not exist: {raw_root}")
        return []

    harness_path = repository / "speed_estimation_harness.py"
    if not harness_path.is_file():
        logger(
            "Waymo preprocessing cannot start because speed_estimation_harness.py "
            f"is not present in {repository}"
        )
        return []

    processed_root.mkdir(parents=True, exist_ok=True)
    processing_summary: Dict[str, Any] = {
        "metrics_build_id": METRICS_BUILD_ID,
        "raw_dataset_path": str(raw_root),
        "processed_root": str(processed_root),
        "process_if_missing": True,
        "splits": {},
    }
    for split_name, processed_split in split_roots.items():
        raw_split = raw_root / split_name
        if not _raw_waymo_files(raw_split):
            logger(f"No raw Waymo TFRecords found for {split_name}: {raw_split}")
            processing_summary["splits"][split_name] = {
                "status": "skipped",
                "reason": "no_tfrecord_files",
            }
            continue
        processed_split.mkdir(parents=True, exist_ok=True)
        index_path = processed_split / "waymo_sequence_index.csv"
        try:
            export_ready = _waymo_export_is_ready(
                repository,
                raw_split,
                processed_split,
            )
            if not export_ready:
                _waymo_export(
                    repository,
                    raw_root,
                    processed_split,
                    split_name,
                    logger,
                )
            else:
                logger(f"Reusing complete Waymo {split_name} export")
            _run_checked(
                [
                    sys.executable,
                    str(harness_path),
                    "track_index",
                    str(index_path),
                    "auto",
                    "0",
                    "false",
                    "true",
                ],
                repository,
            )
            if not _waymo_split_is_ready(
                repository,
                raw_split,
                processed_split,
                split_name,
            ):
                raise RuntimeError(
                    f"Waymo {split_name} tracking did not produce one CSV per exported sequence"
                )
            split_summary = {
                "metrics_build_id": METRICS_BUILD_ID,
                "status": "complete",
                "raw_tfrecords": len(_raw_waymo_files(raw_split)),
                "index": str(index_path),
                "prediction_csvs_complete": True,
            }
            write_json(
                processed_split / "waymo_processing_complete.json",
                split_summary,
            )
            processing_summary["splits"][split_name] = split_summary
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            logger(f"Waymo {split_name} preprocessing failed: {error}")
            processing_summary["splits"][split_name] = {
                "status": "failed",
                "reason": str(error),
            }

    training_ready = _waymo_split_is_ready(
        repository,
        raw_root / "training",
        split_roots["training"],
        "training",
    )
    validation_ready = _waymo_split_is_ready(
        repository,
        raw_root / "validation",
        split_roots["validation"],
        "validation",
    )
    if training_ready and validation_ready:
        try:
            _run_checked(
                [
                    sys.executable,
                    str(harness_path),
                    "calibrate_waymo_pipeline",
                    str(split_roots["training"] / "waymo_sequence_index.csv"),
                    str(split_roots["validation"] / "waymo_sequence_index.csv"),
                    str(calibration_root),
                ],
                repository,
            )
            processing_summary["calibration"] = {
                "status": "complete",
                "model": str(pipeline_model_path),
            }
            write_json(calibration_settings_path, calibration_record())
            load_tuned_pipeline_model(pipeline_model_path)
            log_speed_statistics()
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            logger(f"Waymo calibration failed: {error}")
            processing_summary["calibration"] = {
                "status": "failed",
                "reason": str(error),
            }
    else:
        processing_summary["calibration"] = {
            "status": "skipped",
            "reason": "training_or_validation_tracking_incomplete",
        }
    write_json(processed_root / "waymo_processing_summary.json", processing_summary)
    return [processed_root] if processed_root.exists() else []


def discover_waymo_files(
    repository_root: Path,
    explicit_roots: Optional[Sequence[os.PathLike[str] | str]],
) -> Dict[str, Any]:
    roots: List[Path] = []
    if explicit_roots:
        roots.extend(Path(value).expanduser() for value in explicit_roots)
    else:
        roots.extend(
            [
                repository_root / "data" / "speed_calibration" / "waymo",
                repository_root / "data" / "speed_calibration" / "waymo_validation_v18",
            ]
        )
    roots = [root.resolve() for root in roots if root.exists()]
    manifests: Dict[str, List[Path]] = defaultdict(list)
    indices: Dict[str, List[Path]] = defaultdict(list)
    relative_csvs: List[Path] = []
    relative_zips: List[Path] = []
    for root in roots:
        for candidate in root.rglob("*.csv"):
            lower = candidate.name.lower()
            if lower.startswith("waymo_sequence_index"):
                role, _ = _index_role_and_score(candidate)
                indices[role].append(candidate)
            elif "manifest" in lower:
                try:
                    with candidate.open("r", newline="", encoding="utf-8-sig") as handle:
                        fields = csv.DictReader(handle).fieldnames or []
                    if {"source_id", "prediction_track_id", "ground_truth_speed_mps"}.issubset(fields):
                        manifests[_candidate_role(candidate)].append(candidate)
                except OSError:
                    pass
            if lower == "relative_motion.csv" or lower.endswith("_relative_motion.csv"):
                relative_csvs.append(candidate)
        relative_zips.extend(root.rglob("*relative_motion*.zip"))
    selected_manifests = {
        role: max(paths, key=_manifest_score)
        for role, paths in manifests.items()
        if paths
    }
    selected_indices = {
        role: max(paths, key=lambda path: _index_role_and_score(path)[1])
        for role, paths in indices.items()
        if paths
    }
    return {
        "roots": roots,
        "manifests": selected_manifests,
        "indices": selected_indices,
        "relative_csvs": relative_csvs,
        "relative_zips": relative_zips,
    }


def _relative_row_map(files: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, str]]:
    output: Dict[Tuple[str, str], Dict[str, str]] = {}
    for path in files.get("relative_csvs", []):
        try:
            batches = [read_dict_csv(path)]
        except Exception:
            continue
        for rows in batches:
            for row in rows:
                key = (
                    str(row.get("source_id", "")).strip(),
                    normalise_id(row.get("prediction_track_id")),
                )
                if all(key):
                    output[key] = row
    for path in files.get("relative_zips", []):
        try:
            batches = _read_csv_from_zip(path)
            for rows in batches:
                for row in rows:
                    key = (
                        str(row.get("source_id", "")).strip(),
                        normalise_id(row.get("prediction_track_id")),
                    )
                    if all(key):
                        output[key] = row
        except Exception:
            continue
    return output


def _resolve_manifest_bbox(
    raw_path: str,
    source_id: str,
    roots: Sequence[Path],
) -> Optional[Path]:
    candidate = Path(raw_path).expanduser()
    if candidate.is_file():
        return candidate
    for root in roots:
        for name in ("crowd_yolo_botsort_bbox.csv", "imptc_yolo_botsort_bbox.csv"):
            candidate = root / source_id / name
            if candidate.is_file():
                return candidate
    return None


def _ensure_waymo_relative_rows(
    manifests: Mapping[str, Path],
    files: Mapping[str, Any],
    relative_map: Dict[Tuple[str, str], Dict[str, str]],
    output_root: Path,
    force: bool,
    log: Callable[[str], None],
) -> None:
    cache_root = output_root / "waymo_relative_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for role, manifest_path in manifests.items():
        rows = read_dict_csv(manifest_path)
        by_source: Dict[str, Dict[str, str]] = {}
        for row in rows:
            source_id = str(row.get("source_id", "")).strip()
            if source_id:
                by_source.setdefault(source_id, row)
        for source_id, row in by_source.items():
            required_tracks = {
                normalise_id(item.get("prediction_track_id"))
                for item in rows
                if str(item.get("source_id", "")).strip() == source_id
            }
            if required_tracks and all((source_id, track) in relative_map for track in required_tracks):
                continue
            bbox_path = _resolve_manifest_bbox(
                str(row.get("bbox_csv", "")), source_id, files.get("roots", [])
            )
            fps = safe_float(row.get("fps"))
            aspect = safe_float(row.get("aspect_ratio")) or DEFAULT_ASPECT_RATIO
            if bbox_path is None or fps is None or fps <= 0.0:
                continue
            cache_path = cache_root / f"{safe_name(source_id)}_relative_motion.csv"
            try:
                if force or not _source_cache_is_current(cache_path, bbox_path):
                    relative_rows, _ = predict_relative_rows(bbox_path, fps, source_id, aspect)
                    write_dict_csv(cache_path, relative_rows)
                else:
                    relative_rows = read_dict_csv(cache_path)
                for relative in relative_rows:
                    track_id = normalise_id(relative.get("prediction_track_id"))
                    if track_id:
                        relative_map[(source_id, track_id)] = {
                            key: str(value) if value is not None else ""
                            for key, value in relative.items()
                        }
            except Exception as error:
                log(f"Waymo relative motion generation failed for {source_id}: {error}")
        log(f"Waymo {role} relative motion rows prepared for {len(by_source)} sources")


def regression_fit(rows: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    pairs = [
        (safe_float(row.get("relative_motion_index")), safe_float(row.get("ground_truth_speed_mps")))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_value = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_value = np.asarray([pair[1] for pair in pairs], dtype=float)
    design = np.column_stack([np.ones(len(x_value)), x_value])
    return np.linalg.lstsq(design, y_value, rcond=None)[0]


def regression_predict(beta: Optional[np.ndarray], value: Any) -> Optional[float]:
    x_value = safe_float(value)
    if beta is None or x_value is None:
        return None
    return float(np.clip(beta[0] + beta[1] * x_value, 0.10, 3.50))


def error_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = [
        (safe_float(row.get("ground_truth_speed_mps")), safe_float(row.get("predicted_speed_mps")))
        for row in rows
    ]
    pairs = [(truth, pred) for truth, pred in pairs if truth is not None and pred is not None]
    if not pairs:
        return {
            "count": 0,
            "mae_mps": None,
            "rmse_mps": None,
            "bias_mps": None,
            "median_absolute_error_mps": None,
            "within_0_25_mps": None,
            "within_0_50_mps": None,
            "r2": None,
        }
    truth = np.asarray([pair[0] for pair in pairs], dtype=float)
    pred = np.asarray([pair[1] for pair in pairs], dtype=float)
    error = pred - truth
    absolute = np.abs(error)
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(truth - np.mean(truth))))
    return {
        "count": len(pairs),
        "mae_mps": float(np.mean(absolute)),
        "rmse_mps": float(math.sqrt(np.mean(np.square(error)))),
        "bias_mps": float(np.mean(error)),
        "median_absolute_error_mps": float(np.median(absolute)),
        "within_0_25_mps": float(np.mean(absolute <= 0.25)),
        "within_0_50_mps": float(np.mean(absolute <= 0.50)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0
        position = end
    return ranks


def correlation_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = [
        (safe_float(row.get("relative_motion_index")), safe_float(row.get("ground_truth_speed_mps")))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return {"pearson": None, "spearman": None, "within_source_order_accuracy": None}
    x_value = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_value = np.asarray([pair[1] for pair in pairs], dtype=float)
    pearson_value = float(np.corrcoef(x_value, y_value)[0, 1])
    spearman_value = float(np.corrcoef(_rank(x_value), _rank(y_value))[0, 1])
    pearson = pearson_value if math.isfinite(pearson_value) else None
    spearman = spearman_value if math.isfinite(spearman_value) else None
    by_source: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_id", ""))].append(row)
    correct = 0
    total = 0
    for source_rows in by_source.values():
        for first_index in range(len(source_rows)):
            for second_index in range(first_index + 1, len(source_rows)):
                first = source_rows[first_index]
                second = source_rows[second_index]
                first_x = safe_float(first.get("relative_motion_index"))
                second_x = safe_float(second.get("relative_motion_index"))
                first_y = safe_float(first.get("ground_truth_speed_mps"))
                second_y = safe_float(second.get("ground_truth_speed_mps"))
                if None in (first_x, second_x, first_y, second_y):
                    continue
                if abs(first_x - second_x) < 1e-9 or abs(first_y - second_y) < 1e-9:
                    continue
                total += 1
                correct += int((first_x - second_x) * (first_y - second_y) > 0.0)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "within_source_order_accuracy": correct / total if total else None,
        "within_source_pairs": total,
    }


def _confirmed_crossings(index_path: Optional[Path]) -> Optional[int]:
    if index_path is None or not index_path.is_file():
        return None
    total = 0
    for row in read_dict_csv(index_path):
        value = safe_int(row.get("ground_truth_tracks"))
        if value is not None:
            total += value
    return total


def analyse_waymo(
    repository_root: Path,
    output_root: Path,
    explicit_roots: Optional[Sequence[os.PathLike[str] | str]],
    force: bool,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    search_roots = [
        Path(value).expanduser().resolve()
        for value in (explicit_roots or [])
        if Path(value).expanduser().exists()
    ]
    calibration_reports = [
        candidate
        for root in search_roots
        for version in (
            "calibration_v32",
            "calibration_v31",
            "calibration_v30",
            "calibration_v29",
            "calibration_v28",
        )
        for candidate in root.rglob(
            f"{version}/crossing_calibration_report.json"
        )
    ]
    if calibration_reports:
        report_path = max(calibration_reports, key=lambda path: path.stat().st_mtime)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evaluation_path = report_path.parent / "speed_validation" / "evaluation_tracks.csv"
        evaluation_rows = read_dict_csv(evaluation_path) if evaluation_path.is_file() else []
        validation_tracks = []
        for row in evaluation_rows:
            prediction = (
                row.get("model_speed_before_reliability_gate_mps")
                or row.get("estimated_speed_mps")
            )
            validation_tracks.append({**row, "predicted_speed_mps": prediction})
        summary = {
            "status": "complete",
            "strategy": "CROWD algorithm selects crossings; Waymo supplies matched speed only",
            "waymo_crossing_label_used_for_speed_selection": False,
            "roles": {
                "training": report.get("training_speed_selection", {}),
                "validation": report.get("validation_speed_selection", {}),
            },
            "independent_validation": report.get("speed_validation_metrics", {}),
            "absolute_speed_model_qualified": bool(
                report.get("metric_speed_qualified_for_crowd")
            ),
            "calibration_report": str(report_path),
        }
        write_json(output_root / "waymo_crossing_speed_summary.json", summary)
        return {
            "summary": summary,
            "tracks": validation_tracks,
            "validation_tracks": validation_tracks,
        }

    files = discover_waymo_files(repository_root, explicit_roots)
    manifests: Dict[str, Path] = files["manifests"]
    if not manifests:
        summary = {
            "status": "skipped",
            "reason": "processed Waymo manifests were not found",
            "searched_roots": [str(path) for path in files["roots"]],
        }
        write_json(output_root / "waymo_crossing_speed_summary.json", summary)
        log("Waymo analysis skipped: processed manifests were not found")
        return {"summary": summary, "tracks": []}

    relative_map = _relative_row_map(files)
    _ensure_waymo_relative_rows(
        manifests, files, relative_map, output_root, force, log
    )
    role_tracks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    role_summary: Dict[str, Dict[str, Any]] = {}
    for role, manifest_path in manifests.items():
        manifest_rows = read_dict_csv(manifest_path)
        eligible_rows = [row for row in manifest_rows if truthy(row.get("include"))]
        joined: List[Dict[str, Any]] = []
        for row in eligible_rows:
            key = (
                str(row.get("source_id", "")).strip(),
                normalise_id(row.get("prediction_track_id")),
            )
            relative = relative_map.get(key)
            if relative is None or str(relative.get("relative_motion_status", "")) != "valid":
                continue
            joined.append({**row, **relative, "dataset_role": role})
        role_tracks[role] = joined
        excluded = [row for row in manifest_rows if not truthy(row.get("include"))]
        reason_counts = Counter(str(row.get("exclusion_reason", "")) or "unknown" for row in excluded)
        association_failed = sum(
            count for reason, count in reason_counts.items() if reason in ASSOCIATION_REJECTION_REASONS
        )
        confirmed_crossings = _confirmed_crossings(files["indices"].get(role))
        matched_count = len(manifest_rows)
        role_summary[role] = {
            "manifest": str(manifest_path),
            "confirmed_crossing_pedestrians": confirmed_crossings,
            "matched_with_botsort": matched_count,
            "eligible_after_matching_and_motion_checks": len(eligible_rows),
            "analysed_with_valid_relative_motion": len(joined),
            "eligible_but_unanalysable": len(eligible_rows) - len(joined),
            "failed_checks": len(excluded),
            "failed_association_checks": association_failed,
            "failed_motion_quality_checks": len(excluded) - association_failed,
            "not_detected_or_not_matched": (
                max(0, confirmed_crossings - matched_count)
                if confirmed_crossings is not None
                else None
            ),
            "exclusion_reasons": dict(reason_counts),
        }

    training = role_tracks.get("training", [])
    validation = role_tracks.get("validation", [])
    training_predictions: List[Dict[str, Any]] = []
    for source_id in sorted({str(row.get("source_id", "")) for row in training}):
        train_fold = [row for row in training if str(row.get("source_id", "")) != source_id]
        beta = regression_fit(train_fold)
        for row in training:
            if str(row.get("source_id", "")) != source_id:
                continue
            prediction = regression_predict(beta, row.get("relative_motion_index"))
            training_predictions.append({**row, "predicted_speed_mps": prediction})
    beta = regression_fit(training)
    validation_predictions = [
        {
            **row,
            "predicted_speed_mps": regression_predict(beta, row.get("relative_motion_index")),
        }
        for row in validation
    ]
    training_truth = _numeric_values(training, "ground_truth_speed_mps")
    baseline_value = float(np.median(training_truth)) if training_truth else None
    validation_baseline = [
        {**row, "predicted_speed_mps": baseline_value} for row in validation
    ]
    for row in training_predictions + validation_predictions:
        truth = safe_float(row.get("ground_truth_speed_mps"))
        prediction = safe_float(row.get("predicted_speed_mps"))
        row["error_mps"] = prediction - truth if None not in (truth, prediction) else None
        row["absolute_error_mps"] = abs(prediction - truth) if None not in (truth, prediction) else None

    all_tracks = training_predictions + validation_predictions
    write_dict_csv(output_root / "waymo_crossing_speed_evaluation_tracks.csv", all_tracks)
    validation_metrics = error_metrics(validation_predictions)
    baseline_metrics = error_metrics(validation_baseline)
    validation_count = int(validation_metrics.get("count", 0))
    validation_mae = safe_float(validation_metrics.get("mae_mps"))
    baseline_mae = safe_float(baseline_metrics.get("mae_mps"))
    validation_r2 = safe_float(validation_metrics.get("r2"))
    qualification_reasons: List[str] = []
    if validation_count < 20:
        qualification_reasons.append("too_few_independent_validation_tracks")
    if (
        validation_mae is None
        or baseline_mae is None
        or validation_mae >= 0.90 * baseline_mae
    ):
        qualification_reasons.append("no_material_improvement_over_constant_baseline")
    if validation_r2 is None or validation_r2 <= 0.0:
        qualification_reasons.append("non_positive_validation_r2")

    summary = {
        "status": "complete",
        "roles": role_summary,
        "check_definitions": {
            "not_detected_or_not_matched": (
                "a confirmed Waymo crossing without a one to one BoT SORT association"
            ),
            "failed_association_checks": (
                "a matched pair rejected for frame count, IoU, prediction coverage or ground truth coverage"
            ),
            "failed_motion_quality_checks": (
                "a matched pair rejected for short, fragmented, small, unstable or insufficiently lateral motion"
            ),
            "eligible_but_unanalysable": (
                "a manifest eligible pair for which the current relative motion feature extraction was not valid"
            ),
        },
        "training_source_grouped_cross_validation": error_metrics(training_predictions),
        "independent_validation": validation_metrics,
        "independent_validation_constant_baseline": baseline_metrics,
        "independent_validation_correlations": correlation_metrics(validation),
        "absolute_speed_model_qualified": not qualification_reasons,
        "qualification_reasons": qualification_reasons,
        "model": {
            "form": "clipped linear regression of ground truth speed on within video relative motion index",
            "intercept": float(beta[0]) if beta is not None else None,
            "relative_motion_coefficient": float(beta[1]) if beta is not None else None,
            "clip_mps": [0.10, 3.50],
            "deployment_warning": (
                "This is an evaluation conversion only. It must not be applied to CROWD as a validated "
                "absolute speed model unless it improves materially over the constant baseline on an "
                "untouched external dataset."
            ),
        },
    }
    write_json(output_root / "waymo_crossing_speed_summary.json", summary)
    return {
        "summary": summary,
        "tracks": all_tracks,
        "validation_tracks": validation_predictions,
    }


def write_figures(
    output_root: Path,
    crowd_result: Mapping[str, Any],
    waymo_result: Mapping[str, Any],
    log: Callable[[str], None],
) -> List[str]:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    city_rows = sorted(
        crowd_result.get("city_rows", []),
        key=lambda row: int(row.get("detected_crossing_tracks", 0)),
        reverse=True,
    )[:30]
    validation = waymo_result.get("validation_tracks", [])
    validation_pairs = [
        (float(truth_value), float(prediction_value))
        for row in validation
        if (truth_value := safe_float(row.get("ground_truth_speed_mps"))) is not None
        if (prediction_value := safe_float(row.get("predicted_speed_mps"))) is not None
    ]
    truth = [pair[0] for pair in validation_pairs]
    prediction = [pair[1] for pair in validation_pairs]
    try:
        import plotly.graph_objects as go

        def write_static_formats(
            figure: Any,
            file_stem: str,
            width: int,
            height: int,
        ) -> None:
            for suffix, scale in (("png", 2), ("eps", 1)):
                try:
                    image_path = figure_root / f"{file_stem}.{suffix}"
                    figure.write_image(
                        str(image_path),
                        width=width,
                        height=height,
                        scale=scale,
                    )
                    written.append(str(image_path))
                except Exception as error:
                    log(
                        f"{suffix.upper()} export skipped for "
                        f"{file_stem}: {error}"
                    )

        if city_rows:
            fig = go.Figure()
            fig.add_bar(
                name="Detected crossings",
                x=[str(row.get("locality", "Unknown")) for row in city_rows],
                y=[int(row.get("detected_crossing_tracks", 0)) for row in city_rows],
            )
            fig.add_bar(
                name="Valid relative motion",
                x=[str(row.get("locality", "Unknown")) for row in city_rows],
                y=[int(row.get("analysed_crossing_tracks", 0)) for row in city_rows],
            )
            fig.update_layout(
                barmode="group",
                template="plotly_white",
                title="CROWD crossing tracks by city (top 30)",
                xaxis_title="City",
                yaxis_title="Tracks",
            )
            html_path = figure_root / "crowd_city_crossing_coverage.html"
            fig.write_html(
                str(html_path),
                include_plotlyjs="cdn",
                auto_open=True,
            )
            written.append(str(html_path))
            write_static_formats(
                fig,
                "crowd_city_crossing_coverage",
                1800,
                900,
            )
        if validation_pairs:
            fig = go.Figure()
            fig.add_scatter(x=truth, y=prediction, mode="markers", name="Track")
            lower = min(truth + prediction)
            upper = max(truth + prediction)
            fig.add_scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line={"dash": "dash", "color": "grey"},
                name="Ideal",
            )
            fig.update_layout(
                template="plotly_white",
                title="Waymo independent validation",
                xaxis_title="Ground truth speed (m/s)",
                yaxis_title="Predicted speed (m/s)",
            )
            html_path = figure_root / "waymo_validation_observed_vs_predicted.html"
            fig.write_html(
                str(html_path),
                include_plotlyjs="cdn",
                auto_open=True,
            )
            written.append(str(html_path))
            write_static_formats(
                fig,
                "waymo_validation_observed_vs_predicted",
                1000,
                900,
            )
    except ImportError:
        log("Plotly is unavailable; HTML, PNG and EPS figures were skipped")
    return written


def run_integrated_speed_report(
    crowd_sources: Sequence[Mapping[str, Any]],
    repository_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    waymo_roots: Optional[Sequence[os.PathLike[str] | str]] = None,
    force: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run CROWD reports and, when present, processed Waymo evaluation.

    ``crowd_sources`` must contain source_id, bbox_csv, fps, locality, state,
    country, and crossing_track_ids.  No command line parser is used; the
    calling analysis script supplies these values directly.
    """
    logger = log or _default_log
    root = Path(repository_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger(f"Integrated crossing speed report: {output}")
    crowd_result = analyse_crowd_sources(crowd_sources, output, force, logger)
    waymo_result = analyse_waymo(root, output, waymo_roots, force, logger)
    figures = write_figures(output, crowd_result, waymo_result, logger)
    summary = {
        "report_build_id": REPORT_BUILD_ID,
        "scientific_scope": {
            "crowd": (
                "Waymo qualified bbox crossing speed from YOLO plus BoT SORT CSV"
                if _SPEED_MODEL
                else "relative apparent crossing motion from YOLO plus BoT SORT bbox CSV"
            ),
            "waymo": (
                "CROWD algorithm selected crossings matched to Waymo planar speed; "
                "training only speed fitting followed by frozen validation evaluation"
            ),
            "absolute_crowd_speed_mps": bool(_SPEED_MODEL),
            "reason": (
                "the frozen bbox model passed its Waymo validation gates"
                if _SPEED_MODEL
                else "the frozen bbox model did not pass all Waymo validation gates"
            ),
        },
        "crowd": crowd_result["summary"],
        "waymo": waymo_result["summary"],
        "figures": figures,
    }
    write_json(output / "integrated_crossing_speed_summary.json", summary)
    logger("Integrated crossing speed report complete")
    return summary
