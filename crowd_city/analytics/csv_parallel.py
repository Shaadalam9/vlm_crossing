"""
Parallel detection-file worker for crowd-city analysis.

Detection files can be CSV or Parquet. Parquet is preferred by analysis.py
when both formats are available. Each worker owns its own Polars DataFrame and
returns only compact Python objects to the parent process. Shared aggregation
remains in analysis.py.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import polars as pl

import common
from crowd_city.crossing.detection import Detection
from crowd_city.crossing.metrics import Metrics
import crowd_city.crossing.metrics as crossing_metrics_module


_WORKER_MAPPING: Optional[pl.DataFrame] = None
_WORKER_CROSSING_PARAMETERS: Dict[str, Any] = {}
_WORKER_MIN_CONFIDENCE: float = 0.0
_WORKER_BOUNDARY_LEFT: float = 0.45
_WORKER_BOUNDARY_RIGHT: float = 0.55
_WORKER_PROCESSING_FPS: Optional[float] = None
# "detector" or "road_crossing"; see utils/crossing/road_crossing.py.
_WORKER_CROSSING_RULE: str = "detector"

_DETECTION = Detection()
_METRICS = Metrics()

YOLO_PERSON = 0
YOLO_BICYCLE = 1
YOLO_CAR = 2
YOLO_MOTORCYCLE = 3
YOLO_BUS = 5
YOLO_TRUCK = 7
YOLO_TRAFFIC_SIGN_IDS = (9, 11)
YOLO_CELLPHONE = 67

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove",
    "skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
    "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
]


def initialise_csv_worker(
    mapping: pl.DataFrame,
    crossing_parameters: Dict[str, Any],
    min_confidence: float,
    boundary_left: float,
    boundary_right: float,
    pipeline_model: Dict[str, Any],
    speed_model: Dict[str, Any],
) -> None:
    """Initialise process local state once per worker."""
    global _WORKER_MAPPING
    global _WORKER_CROSSING_PARAMETERS
    global _WORKER_MIN_CONFIDENCE
    global _WORKER_BOUNDARY_LEFT
    global _WORKER_BOUNDARY_RIGHT
    global _WORKER_PROCESSING_FPS
    global _WORKER_CROSSING_RULE

    _WORKER_MAPPING = mapping
    _WORKER_CROSSING_PARAMETERS = dict(crossing_parameters or {})
    _WORKER_MIN_CONFIDENCE = float(min_confidence)
    _WORKER_BOUNDARY_LEFT = float(boundary_left)
    _WORKER_BOUNDARY_RIGHT = float(boundary_right)
    _WORKER_PROCESSING_FPS = _normalise_processing_fps(
        common.get_configs("processing_fps")
    )
    _WORKER_CROSSING_RULE = str(common.get_configs("crossing_rule"))

    # The Waymo speed model is loaded in the parent before the worker pool
    # starts. Copy the already loaded model state into each spawned process so
    # speed calculations remain identical to the single process path.
    crossing_metrics_module._PIPELINE_MODEL = dict(pipeline_model or {})
    crossing_metrics_module._SPEED_MODEL = dict(speed_model or {})
    os.environ["CROWD_CROSSING_SPEED_UNIT"] = (
        "m/s" if crossing_metrics_module._SPEED_MODEL else "relative"
    )


def _normalise_processing_fps(value: object) -> Optional[float]:
    """Return the configured target analysis FPS, or None for native FPS."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        raise ValueError("processing_fps must be a positive number or null")

    try:
        fps = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "processing_fps must be a positive number or null"
        ) from error

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            "processing_fps must be a positive finite number or null"
        )

    return fps


def _resample_detection_fps(
    df: pl.DataFrame,
    source_fps: float,
    target_fps: Optional[float],
) -> tuple[pl.DataFrame, float]:
    """Downsample detections to a common analysis FPS and reindex frames.

    The detection files retain their original frame numbers. When a lower
    processing FPS is configured, this function keeps frames on a deterministic
    sampling grid and rewrites ``frame-count`` to the corresponding processing
    frame index. Downstream code can therefore use the returned effective FPS
    directly for timing and frame-scaled thresholds.

    Sources below the configured target are never upsampled. In that case all
    available frames are retained and the native source FPS remains effective.
    """
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source FPS: {source_fps}")

    if target_fps is None or target_fps >= source_fps:
        return df, float(source_fps)

    if df.height == 0 or "frame-count" not in df.columns:
        return df, float(target_fps)

    frame_expr = pl.col("frame-count").cast(pl.Float64, strict=False)
    finite_frames = (
        df
        .select(frame_expr.alias("_source_frame"))
        .filter(pl.col("_source_frame").is_not_null())
        .filter(pl.col("_source_frame").is_finite())
    )
    if finite_frames.height == 0:
        return df.clear(), float(target_fps)

    first_frame = float(finite_frames.get_column("_source_frame").min())
    ratio = float(target_fps) / float(source_fps)
    inverse_ratio = float(source_fps) / float(target_fps)

    sampled = (
        df
        .with_columns(frame_expr.alias("_source_frame"))
        .filter(pl.col("_source_frame").is_not_null())
        .filter(pl.col("_source_frame").is_finite())
        .with_columns(
            ((pl.col("_source_frame") - first_frame) * ratio)
            .round(0)
            .cast(pl.Int64)
            .alias("_processing_frame")
        )
        .with_columns(
            (
                first_frame
                + pl.col("_processing_frame").cast(pl.Float64) * inverse_ratio
            )
            .round(0)
            .alias("_expected_source_frame")
        )
        .filter(pl.col("_source_frame") == pl.col("_expected_source_frame"))
        .drop("frame-count", "_source_frame", "_expected_source_frame")
        .rename({"_processing_frame": "frame-count"})
    )

    return sampled, float(target_fps)


def _empty_metric_counts() -> Dict[str, int]:
    return {
        "persons": 0,
        "cellphones": 0,
        "traffic_signs": 0,
        "vehicles": 0,
        "bicycles": 0,
        "cars": 0,
        "motorcycles": 0,
        "buses": 0,
        "trucks": 0,
    }


def _metric_counts_from_confidence_filtered(df: pl.DataFrame) -> Dict[str, int]:
    """
    Match MetricsCache's first pass count semantics.

    This intentionally runs before removing null or -1 tracker IDs because the
    existing MetricsCache only applies the confidence threshold.
    """
    required = {"yolo-id", "unique-id"}
    if df.height == 0 or not required.issubset(set(df.columns)):
        return _empty_metric_counts()

    working = df.with_columns(
        pl.col("yolo-id").cast(pl.Int64, strict=False).alias("_metric_yolo_id")
    )
    uid = pl.col("unique-id")
    yolo = pl.col("_metric_yolo_id")

    result = working.select([
        uid.filter(yolo == YOLO_PERSON).n_unique().alias("persons"),
        uid.filter(yolo == YOLO_CELLPHONE).n_unique().alias("cellphones"),
        uid.filter(yolo.is_in(list(YOLO_TRAFFIC_SIGN_IDS))).n_unique().alias("traffic_signs"),
        uid.filter(yolo.is_in([YOLO_CAR, YOLO_MOTORCYCLE, YOLO_BUS, YOLO_TRUCK])).n_unique().alias("vehicles"),
        uid.filter(yolo == YOLO_BICYCLE).n_unique().alias("bicycles"),
        uid.filter(yolo == YOLO_CAR).n_unique().alias("cars"),
        uid.filter(yolo == YOLO_MOTORCYCLE).n_unique().alias("motorcycles"),
        uid.filter(yolo == YOLO_BUS).n_unique().alias("buses"),
        uid.filter(yolo == YOLO_TRUCK).n_unique().alias("trucks"),
    ])

    return {
        name: int(result.item(0, name) or 0)
        for name in _empty_metric_counts()
    }


TrackIndex = Dict[Any, pl.DataFrame]


def _build_track_index(df: pl.DataFrame) -> TrackIndex:
    """Build reusable tracks only for IDs that contain a person detection.

    Timing historically selected rows by ``unique-id`` alone, so if a tracker
    ID also contains another class those rows are intentionally retained.
    IDs that never represent a person are excluded because crossing code never
    queries them.
    """
    required = {"yolo-id", "unique-id", "frame-count"}
    if df.height == 0 or not required.issubset(set(df.columns)):
        return {}

    person_ids = (
        df
        .filter(pl.col("yolo-id") == YOLO_PERSON)
        .select("unique-id")
        .unique()
        .get_column("unique-id")
    )
    if len(person_ids) == 0:
        return {}

    ordered = (
        df
        .filter(pl.col("unique-id").is_in(person_ids))
        .sort(["unique-id", "frame-count"])
    )
    track_index: TrackIndex = {}

    for track in ordered.partition_by(
        "unique-id",
        maintain_order=True,
    ):
        if track.height == 0:
            continue
        unique_id = track.get_column("unique-id")[0]
        track_index[unique_id] = track

    return track_index


def _track(
    track_index: TrackIndex,
    track_id: Any,
) -> Optional[pl.DataFrame]:
    """Return the cached rows for one tracker ID."""
    return track_index.get(track_id)


def _time_to_cross_from_track_index(
    track_index: TrackIndex,
    ids: list,
    fps: float,
    id_bounds: Optional[Dict[Any, Tuple[int, int]]] = None,
) -> Dict[Any, float]:
    """Match Metrics.time_to_cross without filtering and sorting per ID.

    A tracker id can be reused for an unrelated object much later in the same
    video, so ``track_index[track_id]`` may hold rows from more than one
    appearance. When ``id_bounds`` gives the accepted crossing's own
    ``(start_frame, end_frame)``, that span is the duration directly; the full
    track is only used as a fallback when no bound is known for the id.
    """
    if not ids or not math.isfinite(float(fps)) or float(fps) <= 0:
        return {}

    output: Dict[Any, float] = {}
    for track_id in ids:
        bounds = (id_bounds or {}).get(track_id)
        if bounds is not None:
            start_frame, end_frame = bounds
            duration = (float(end_frame) - float(start_frame)) / float(fps)
            if duration > 0:
                output[track_id] = float(duration)
            continue

        track = _track(track_index, track_id)
        if track is None or track.height < 2 or "frame-count" not in track.columns:
            continue

        frames = []
        for value in track.get_column("frame-count").to_list():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                frames.append(numeric)

        if len(frames) < 2:
            continue

        duration = (max(frames) - min(frames)) / float(fps)
        if duration > 0:
            output[track_id] = float(duration)

    return output


def _time_to_start_from_track_index(
    mapping: pl.DataFrame,
    track_index: TrackIndex,
    track_ids: list,
    source_id: str,
    fps: float,
    id_bounds: Optional[Dict[Any, Tuple[int, int]]] = None,
):
    """Match waiting time calculation while reusing already sorted tracks.

    Restricts each track to its accepted crossing's own frames when
    ``id_bounds`` has them, so a tracker id reused later in the same video for
    an unrelated object cannot be mistaken for more of the same wait.
    """
    if not track_ids or not math.isfinite(float(fps)) or float(fps) <= 0:
        return None

    checks_per_second = float(_METRICS._get_config("check_per_sec_time"))
    if checks_per_second is None or checks_per_second <= 0:
        return None

    step = max(1, int(round(float(fps) / checks_per_second)))
    durations: Dict[Any, float] = {}

    for track_id in track_ids:
        track = _track(track_index, track_id)
        if track is None or track.height <= step:
            continue
        if not {"x-center", "height"}.issubset(set(track.columns)):
            continue

        bounds = (id_bounds or {}).get(track_id)
        if bounds is not None and "frame-count" in track.columns:
            start_frame, end_frame = bounds
            track = track.filter(
                pl.col("frame-count").cast(pl.Int64, strict=False).is_between(
                    start_frame, end_frame,
                )
            )
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
        frames = (
            track.get_column("frame-count").cast(pl.Float64, strict=False).to_numpy()
            if "frame-count" in track.columns
            else None
        )
        wait = crossing_metrics_module.leading_wait_check_units(
            x_values, frames, heights.to_numpy(), float(fps), checks_per_second,
        )
        if wait is not None:
            durations[track_id] = wait

    if not durations:
        return None

    return crossing_metrics_module.grouping_class.locality_country_wrapper(
        input_dict={source_id: durations},
        mapping=mapping,
    )


def _object_counts(df: pl.DataFrame) -> Dict[str, int]:
    """Match the legacy per CSV COCO object counting path."""
    if df.height == 0:
        return {}

    required = {"yolo-id", "unique-id"}
    if not required.issubset(set(df.columns)):
        return {}

    counts_df = (
        df.unique(subset=["yolo-id", "unique-id"])
        .group_by("yolo-id")
        .len()
        .rename({"len": "count"})
    )

    id_to_count = {
        int(row["yolo-id"]): int(row["count"])
        for row in counts_df.to_dicts()
        if row.get("yolo-id") is not None
    }
    return {
        class_name: int(id_to_count.get(index, 0))
        for index, class_name in enumerate(COCO_CLASSES)
        if int(id_to_count.get(index, 0)) != 0
    }


def _normalise_detection_schema(df: pl.DataFrame) -> pl.DataFrame:
    # Normalise detection column types across CSV and Parquet sources.
    if df.height == 0:
        return df

    expressions = []

    if "yolo-id" in df.columns:
        expressions.append(
            pl.col("yolo-id").cast(pl.Int64, strict=False).alias("yolo-id")
        )

    if "unique-id" in df.columns:
        expressions.append(
            pl.col("unique-id").cast(pl.Utf8, strict=False).alias("unique-id")
        )

    if "frame-count" in df.columns:
        expressions.append(
            pl.col("frame-count").cast(pl.Int64, strict=False).alias("frame-count")
        )

    for column in (
        "x-center",
        "y-center",
        "width",
        "height",
        "confidence",
    ):
        if column in df.columns:
            expressions.append(
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
            )

    return df.with_columns(expressions) if expressions else df


def _read_confidence_filtered(file_path: str) -> pl.DataFrame:
    """Read one detection file and apply the confidence filter as early as possible.

    Parquet uses the lazy scanner so Polars can apply predicate pushdown and
    use Parquet statistics before materialising the DataFrame. CSV keeps the
    existing eager read path for backwards compatibility.
    """
    suffix = os.path.splitext(file_path)[1].lower()

    if suffix == ".parquet":
        lazy_df = pl.scan_parquet(file_path, use_statistics=True)
        schema_names = set(lazy_df.collect_schema().names())
        if "confidence" not in schema_names:
            raise ValueError("confidence column is missing")

        filtered = (
            lazy_df
            .filter(
                pl.col("confidence").cast(pl.Float64, strict=False)
                >= float(_WORKER_MIN_CONFIDENCE)
            )
            .collect()
        )
        return _normalise_detection_schema(filtered)

    if suffix == ".csv":
        raw_df = pl.read_csv(file_path)
        if "confidence" not in raw_df.columns:
            raise ValueError("confidence column is missing")

        filtered = raw_df.filter(
            pl.col("confidence").cast(pl.Float64, strict=False)
            >= float(_WORKER_MIN_CONFIDENCE)
        )
        return _normalise_detection_schema(filtered)

    raise ValueError(f"Unsupported detection file format: {suffix or '<none>'}")


def _limit_detection_duration(
    df: pl.DataFrame,
    duration_seconds: float,
    fps: float,
) -> pl.DataFrame:
    """Keep only detection frames inside the mapped analysis duration.

    ``duration_seconds`` comes from MetaData.segment_lookup(). For the one
    segment that reaches a city's footage cap it may be shorter than the
    original segment. Using the first observed frame as the origin works for
    both relative and absolute frame-count conventions.
    """
    if (
        df.height == 0
        or duration_seconds <= 0
        or not math.isfinite(duration_seconds)
        or fps <= 0
        or not math.isfinite(fps)
        or "frame-count" not in df.columns
    ):
        return df

    frame_values = (
        df
        .get_column("frame-count")
        .cast(pl.Float64, strict=False)
        .drop_nulls()
    )
    if len(frame_values) == 0:
        return df

    first_frame = float(frame_values.min())
    last_allowed_frame = first_frame + duration_seconds * fps

    return df.filter(
        pl.col("frame-count")
        .cast(pl.Float64, strict=False)
        <= last_allowed_frame
    )


def _road_crossing_candidates(
    mapping: pl.DataFrame,
    df: pl.DataFrame,
    track_index: TrackIndex,
    source_id: str,
    fps: float,
) -> Dict[str, Any]:
    """Find rule-D candidates and their metrics, before segmentation.

    Rule D (utils/crossing/road_crossing.py) is decided after segmentation in
    the parent process, but only for these candidates, so their crossing
    duration, speed and waiting time are computed here with exactly the code
    and frozen speed model the detector's picks use.
    """
    from crowd_city.crossing.road_crossing import box_only_candidates, longest_track_run

    rows, scene_profile = crossing_metrics_module.rows_and_scene_profile(df, float(fps))
    person_tracks = crossing_metrics_module.group_tracks(
        row for row in rows if row.class_id == crossing_metrics_module.PERSON_CLASS_ID
    )
    trimmed = {}
    for track_id, track_rows in person_tracks.items():
        run = longest_track_run([row.frame for row in track_rows], fps)
        if run is not None:
            trimmed[track_id] = crossing_metrics_module.trim_rows_to_frame_range(track_rows, *run)
    aspect_ratio = crossing_metrics_module.safe_float(os.environ.get("CROWD_BBOX_ASPECT_RATIO"))
    if aspect_ratio is None or aspect_ratio <= 0.0:
        aspect_ratio = crossing_metrics_module.DEFAULT_ASPECT_RATIO
    features = crossing_metrics_module.contextual_track_features(
        trimmed,
        float(fps),
        source_id,
        float(aspect_ratio),
        scene_profile,
    )
    ids, bounds, size_rates = box_only_candidates(
        track_index,
        features,
        fps,
        float(_WORKER_CROSSING_PARAMETERS["min_crossing_x_range"]),
    )
    temp_data = _time_to_cross_from_track_index(track_index, ids, fps, bounds)
    return {
        "ids": ids,
        "id_bounds": bounds,
        "size_rates": size_rates,
        "temp_data": temp_data,
        "speed_value": _METRICS.calculate_speed_of_crossing(
            mapping, df, {source_id: temp_data}, id_bounds=bounds,
        ),
        "time_value": _time_to_start_from_track_index(
            mapping, track_index, list(temp_data.keys()), source_id, fps, bounds,
        ),
    }


def process_csv_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Read and analyse one detection file, returning compact mergeable results."""
    mapping = _WORKER_MAPPING
    if mapping is None:
        return {
            "status": "error",
            "file_name": str(task.get("file_name", "")),
            "message": "Detection worker was not initialised.",
        }

    file_path = os.fspath(task["file_path"])
    file_name = str(task["file_name"])
    filename_no_ext = str(task["filename_no_ext"])
    source_fps = float(task["fps"])
    locality_id = int(task["video_locality_id"])
    is_bbox_stream = bool(task["is_bbox_stream"])

    try:
        confidence_filtered = _read_confidence_filtered(file_path)

        # MetaData.segment_lookup supplies the original mapped duration for
        # ordinary segments and the remaining budget for the final capped
        # segment. Trim before every count and crossing calculation so no
        # detections outside the requested per-city footage are analysed.
        time_video = float(task.get("time_video", 0.0) or 0.0)
        confidence_filtered = _limit_detection_duration(
            confidence_filtered,
            time_video,
            source_fps,
        )

        # Apply a common processing frame rate after duration clipping. For
        # example, a 30 FPS source analysed with processing_fps=5 keeps roughly
        # every sixth source frame and rewrites frame-count onto a 5 FPS clock.
        # A 3 FPS source remains at 3 FPS because missing frames are not
        # synthesised.
        confidence_filtered, fps = _resample_detection_fps(
            confidence_filtered,
            source_fps,
            _WORKER_PROCESSING_FPS,
        )

        metric_counts = _metric_counts_from_confidence_filtered(
            confidence_filtered
        )

        required = {"unique-id", "yolo-id"}
        if not required.issubset(set(confidence_filtered.columns)):
            return {
                "status": "error",
                "file_name": file_name,
                "message": f"{file_name}: required detection columns are missing.",
            }

        uid_text = (
            pl.col("unique-id")
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.to_lowercase()
        )
        df = confidence_filtered.filter(
            uid_text.is_not_null()
            & ~uid_text.is_in(["", "-1", "-1.0", "nan", "none", "null"])
        )

        track_index = _build_track_index(df) if is_bbox_stream else {}
        object_counts = _object_counts(df)

        ids = []
        all_ids = []
        id_bounds: Dict[Any, Tuple[int, int]] = {}
        temp_data: Dict[Any, Any] = {}
        speed_value = None
        time_value = None
        road_candidates = None

        if is_bbox_stream:
            crossing_parameters = dict(_WORKER_CROSSING_PARAMETERS)
            boundary_left = float(
                crossing_parameters.pop(
                    "boundary_left",
                    _WORKER_BOUNDARY_LEFT,
                )
            )
            boundary_right = float(
                crossing_parameters.pop(
                    "boundary_right",
                    _WORKER_BOUNDARY_RIGHT,
                )
            )

            ids, all_ids, id_bounds = _DETECTION.pedestrian_crossing(
                df,
                filename_no_ext,
                mapping,
                boundary_left,
                boundary_right,
                person_id=0,
                fps=fps,
                track_index=track_index,
                **crossing_parameters,
            )

            temp_data = _time_to_cross_from_track_index(
                track_index,
                ids,
                fps,
                id_bounds,
            )

            speed_value = _METRICS.calculate_speed_of_crossing(
                mapping,
                df,
                {filename_no_ext: temp_data},
                id_bounds=id_bounds,
            )

            time_value = _time_to_start_from_track_index(
                mapping,
                track_index,
                list(temp_data.keys()),
                filename_no_ext,
                fps,
                id_bounds,
            )

            if _WORKER_CROSSING_RULE == "road_crossing":
                road_candidates = _road_crossing_candidates(
                    mapping, df, track_index, filename_no_ext, fps,
                )

        return {
            "status": "ok",
            "file_name": file_name,
            "filename_no_ext": filename_no_ext,
            "video_locality_id": locality_id,
            "is_bbox_stream": is_bbox_stream,
            "ids": ids,
            "all_ids": all_ids,
            "id_bounds": id_bounds,
            "temp_data": temp_data,
            "speed_value": speed_value,
            "time_value": time_value,
            "road_candidates": road_candidates,
            "object_counts": object_counts,
            "metric_counts": metric_counts,
            "time_video": float(time_video or 0),
            "source_fps": float(source_fps),
            "processing_fps": float(fps),
        }

    except Exception as exc:
        return {
            "status": "error",
            "file_name": file_name,
            "filename_no_ext": filename_no_ext,
            "message": f"{file_name}: {exc}",
        }
