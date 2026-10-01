"""Run road-surface segmentation over the detected crossings of one analysis.

This runs as a single sequential pass in the parent process, after the parallel
detection pass has decided which tracks are crossings. Segmentation is a GPU
workload with a large model, so loading it once here is far cheaper than
loading it inside every detection worker, and the pass only ever touches
segments that actually contain a crossing.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
import crowd_city.analytics.csv_parallel as csv_parallel
import crowd_city.crossing.metrics as crossing_metrics
from crowd_city.analytics.csv_parallel import (
    _limit_detection_duration,
    _read_confidence_filtered,
    _resample_detection_fps,
    initialise_csv_worker,
)
from crowd_city.core.grouping import Grouping
from crowd_city.crossing.road_metrics import (
    hesitation_seconds,
    road_intervals_for_tracks,
    road_restricted_speed,
)
from crowd_city.segmentation.frames import RemoteCredentials
from crowd_city.segmentation.pipeline import SegmentationPipeline, SegmentRequest
from crowd_city.segmentation.segformer import (
    IsolatedSurfaceSegmenter,
    segmentation_is_available,
)
from crowd_city.segmentation.store import (
    SegmentationSettings,
    SurfaceStore,
    configured_segmentation_root,
    crossing_fingerprint,
)
from crowd_city.segmentation.surface import (
    DEFAULT_FOOTPOINT_BAND_FRACTION,
    DEFAULT_FOOTPOINT_WIDTH_FRACTION,
)


logger = CustomLogger(__name__)
# Largest share of segments whose road surface may fail to be read before
# the road-crossing selection stops. Failed segments would simply lose their
# crossings, biasing the cities they belong to, so beyond this the run stops.
MAXIMUM_FAILED_SEGMENT_SHARE = 0.05
grouping = Grouping()


def _config(name: str) -> Any:
    # Always the value in config: no hard-coded fallback.
    return common.get_configs(name)


def _secret(name: str) -> Optional[str]:
    try:
        value = common.get_secrets(name)
    except Exception:
        return None
    text = str(value or "").strip()
    return text or None


def _crossing_tracks(
    df: pl.DataFrame,
    track_ids: Sequence[Any],
    id_bounds: Optional[Mapping[Any, Tuple[int, int]]] = None,
) -> Dict[str, pl.DataFrame]:
    """Return the detection rows of each crossing track, keyed by track id.

    A tracker id can be reused for an unrelated object elsewhere in the same
    video, so ``id_bounds`` (the accepted segment's own frame range, from
    ``Detection.pedestrian_crossing``) restricts each track to the frames it
    actually crossed on when known, instead of every row sharing that id.
    """
    if df.height == 0 or not track_ids:
        return {}
    wanted = {str(value) for value in track_ids}
    bounds = {str(key): value for key, value in (id_bounds or {}).items()}
    tracks: Dict[str, pl.DataFrame] = {}
    for track in df.partition_by("unique-id", maintain_order=True):
        if track.height == 0:
            continue
        track_id = str(track.get_column("unique-id")[0])
        if track_id not in wanted:
            continue
        track = track.sort("frame-count")
        track_bounds = bounds.get(track_id)
        if track_bounds is not None and "frame-count" in track.columns:
            start_frame, end_frame = track_bounds
            restricted = track.filter(
                pl.col("frame-count").cast(pl.Int64, strict=False).is_between(
                    start_frame, end_frame,
                )
            )
            if restricted.height > 0:
                track = restricted
        tracks[track_id] = track
    return tracks


def _segmentation_pipeline(
    df_mapping: pl.DataFrame,
    crossing_parameters: Mapping[str, Any],
) -> Optional[SegmentationPipeline]:
    """Build the CROWD segmentation pipeline from config, or return None.

    Shared by the road-crossing selection and the segmentation metric pass so
    both read the same store with the same settings.
    """
    if not bool(_config("use_segmentation")):
        logger.info("Segmentation-based crossing metrics are disabled in the configuration.")
        return None

    if not segmentation_is_available():
        logger.error(
            "use_segmentation is enabled but torch/transformers are unavailable; "
            "skipping the segmentation pass."
        )
        return None

    root = configured_segmentation_root()
    if not root:
        logger.error(
            "use_segmentation is enabled but 'seg_data' is not configured; "
            "skipping the segmentation pass."
        )
        return None

    store = SurfaceStore(root)
    if not store.available:
        logger.error(f"Segmentation store {root} could not be opened; skipping the pass.")
        return None

    coarse_hz = float(_config("segmentation_coarse_hz"))
    refine_hz = float(_config("segmentation_refine_hz"))
    segmenter = IsolatedSurfaceSegmenter(
        model_name=str(_config("segmentation_model")),
        device=str(_config("segmentation_device")),
        batch_size=int(_config("segmentation_batch_size")),
        input_width=int(_config("segmentation_input_width")),
        input_height=int(_config("segmentation_input_height")),
    )

    minimum_confidence = float(
        _config("segmentation_min_confidence")
    )
    settings = SegmentationSettings(
        model_identifier=segmenter.model_identifier,
        coarse_hz=coarse_hz,
        refine_hz=refine_hz,
        footpoint_band_fraction=DEFAULT_FOOTPOINT_BAND_FRACTION,
        footpoint_width_fraction=DEFAULT_FOOTPOINT_WIDTH_FRACTION,
        minimum_confidence=minimum_confidence,
        crossing_fingerprint=crossing_fingerprint(
            {
                "crossing_parameters": dict(crossing_parameters or {}),
                "min_confidence": _config("min_confidence"),
                "boundary_left": _config("boundary_left"),
                "boundary_right": _config("boundary_right"),
                "processing_fps": _config("processing_fps"),
                "crossing_rule": _config("crossing_rule"),
            }
        ),
    )

    # The detection-file readers below are the ones the parallel detection
    # workers use, and they read process-local state that only
    # initialise_csv_worker sets. Without this the parent would read every
    # detection file at confidence 0.0 while the detection pass used the
    # configured threshold, so the rows behind the crossing tracks would not
    # be the rows the crossings were decided from.
    initialise_csv_worker(
        df_mapping,
        dict(crossing_parameters or {}),
        float(_config("min_confidence")),
        float(_config("boundary_left")),
        float(_config("boundary_right")),
        dict(getattr(crossing_metrics, "_PIPELINE_MODEL", {}) or {}),
        dict(getattr(crossing_metrics, "_SPEED_MODEL", {}) or {}),
    )

    credentials = RemoteCredentials(
        base_url=str(_config("ftp_base_url")),
        username=_secret("ftp_username"),
        password=_secret("ftp_password"),
        token=_secret("ftp_token"),
    )

    return SegmentationPipeline(
        store=store,
        segmenter=segmenter,
        settings=settings,
        credentials=credentials,
        coarse_hz=coarse_hz,
        refine_hz=refine_hz,
        minimum_confidence=minimum_confidence,
    )


def run_segmentation_pass(
    df_mapping: pl.DataFrame,
    detection_tasks: Sequence[Mapping[str, Any]],
    crossing_ids: Mapping[str, Mapping[str, Any]],
    crossing_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return road-restricted speed and hesitation time, wrapped by locality.

    Both outputs use the same nested ``{locality_condition: {video: {track:
    value}}}`` shape as the baseline metrics, so the existing aggregation and
    plotting paths can consume them unchanged. The hesitation values are in
    seconds rather than sample counts.
    """
    empty: Dict[str, Any] = {
        "seg_speed": {},
        "seg_time": {},
        "diagnostics": Counter(),
        "enabled": False,
    }
    pipeline = _segmentation_pipeline(df_mapping, crossing_parameters)
    if pipeline is None:
        return empty
    coarse_hz, refine_hz = pipeline.coarse_hz, pipeline.refine_hz
    segmenter = pipeline.segmenter

    tasks_by_stem = {
        str(task["filename_no_ext"]): task
        for task in detection_tasks
    }
    pending = [
        stem
        for stem, payload in (crossing_ids or {}).items()
        if (payload or {}).get("ids") and stem in tasks_by_stem
    ]
    if not pending:
        logger.warning("No detected crossings available for the segmentation pass.")
        pipeline.close()
        return empty

    logger.info(
        f"Segmenting {len(pending)} detection segment(s) with crossings: "
        f"a {coarse_hz:g} Hz pass to find each road entry and exit, then "
        f"{refine_hz:g} Hz only inside those transitions. "
        f"Model {segmenter.model_identifier} on {segmenter.device}."
    )

    checks_per_second = float(_config("check_per_sec_time"))
    raw_speed: Dict[str, Dict[str, float]] = {}
    raw_time: Dict[str, Dict[str, float]] = {}
    diagnostics: Counter = Counter()

    def measure_one(stem: str) -> Any:
        try:
            return _process_segment(
                df_mapping=df_mapping,
                pipeline=pipeline,
                task=tasks_by_stem[stem],
                stem=stem,
                track_ids=list(crossing_ids[stem]["ids"]),
                checks_per_second=checks_per_second,
                id_bounds=crossing_ids[stem].get("id_bounds") or {},
            )
        except Exception as error:
            _raise_if_gpu_failure(error)
            logger.warning(f"Segmentation failed for {stem}: {error}")
            return "segment_exception"

    try:
        for stem, result in _parallel_segments(measure_one, sorted(pending), "Segmenting crossing windows"):
            if result == "segment_exception":
                diagnostics["segment_exception"] += 1
                continue
            if result is None:
                continue

            speed_values, time_values, segment_diagnostics = result
            diagnostics.update(segment_diagnostics)
            if speed_values:
                raw_speed[stem] = speed_values
            if time_values:
                raw_time[stem] = time_values
    finally:
        pipeline.close()

    diagnostics.update(pipeline.statistics)

    seg_speed = grouping.locality_country_wrapper(raw_speed, df_mapping) if raw_speed else {}
    seg_time = grouping.locality_country_wrapper(raw_time, df_mapping) if raw_time else {}

    logger.info(
        "Segmentation pass complete: "
        f"{pipeline.statistics['segments_reused']} segment(s) reused from the store, "
        f"{pipeline.statistics['segments_segmented']} segmented, "
        f"{pipeline.statistics['segments_failed']} failed, "
        f"{pipeline.statistics['frames_segmented']} frame(s) through the model "
        f"({pipeline.statistics['frames_coarse']} locating transitions, "
        f"{pipeline.statistics['frames_refine']} refining them)."
    )
    logger.info(
        f"Road-restricted speed produced for {sum(len(v) for v in raw_speed.values())} track(s); "
        f"hesitation time for {sum(len(v) for v in raw_time.values())} track(s)."
    )
    for reason, count in sorted(diagnostics.items(), key=lambda item: -item[1]):
        logger.info(f"Segmentation diagnostic: {reason}={count}.")

    return {
        "seg_speed": seg_speed,
        "seg_time": seg_time,
        "diagnostics": diagnostics,
        "enabled": True,
    }


class GpuFailure(RuntimeError):
    """The GPU stopped working, so no further segment can be segmented."""


def _raise_if_gpu_failure(error: BaseException) -> None:
    """Turn a fatal CUDA error into GpuFailure instead of a per-segment failure.

    Once a kernel times out or the device faults (for example an NVIDIA Xid 8
    watchdog timeout), the CUDA context stays broken for the rest of the
    process, so every later segment fails in milliseconds. Counting those as
    ordinary per-segment failures burned through a whole 2,511-segment run and
    then stopped on the failure share, losing the run. Stopping at the first
    one keeps everything segmented so far in the store for the next run.
    """
    message = str(error)
    if "CUDA error" in message or "CUDA-capable device" in message or "cudaError" in message:
        raise GpuFailure(
            f"The GPU stopped working during segmentation ({message.splitlines()[0]}). Every segment "
            "segmented so far is saved in the segmentation store and is reused when analysis.py is "
            "started again; check the GPU (nvidia-smi, kernel log) and rerun."
        ) from error


def _parallel_segments(
    function: Callable[[str], Any],
    stems: Sequence[str],
    description: str,
) -> Iterator[Tuple[str, Any]]:
    """Run ``function`` on every stem, ``segmentation_parallel_videos`` at a time.

    Reading a window is almost entirely waiting on the file server, so several
    segments are read at once; the pipeline serialises the GPU model itself.
    Results are yielded in completion order with a progress bar.
    """
    workers = max(1, int(_config("segmentation_parallel_videos")))
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(function, stem): stem for stem in stems}
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            yield futures[future], future.result()
    finally:
        # On an error (GpuFailure above all) do not start the segments still
        # queued; only the few already running are waited for.
        executor.shutdown(wait=True, cancel_futures=True)


def _prepared_detections(task: Mapping[str, Any]) -> Tuple[pl.DataFrame, float, int]:
    """Read one detection file exactly as the detection worker prepared it.

    Otherwise the frame numbers the crossing ids refer to would not line up
    with the video timestamps derived from them. Returns the detections, the
    effective frame rate and the first source frame, which resampling
    renumbers relative to and which is needed to put frames back on video time.
    """
    detections = _read_confidence_filtered(str(task["file_path"]))
    detections = _limit_detection_duration(
        detections,
        float(task.get("time_video", 0.0) or 0.0),
        float(task["fps"]),
    )
    first_source_frame = (
        detections.get_column("frame-count").cast(pl.Float64, strict=False).min()
        if detections.height and "frame-count" in detections.columns
        else None
    )
    detections, effective_fps = _resample_detection_fps(
        detections,
        float(task["fps"]),
        csv_parallel._WORKER_PROCESSING_FPS,
    )
    return detections, float(effective_fps), int(first_source_frame or 0)


def _segment_request(
    task: Mapping[str, Any],
    stem: str,
    effective_fps: float,
    first_source_frame: int,
    tracks: Dict[str, pl.DataFrame],
) -> SegmentRequest:
    return SegmentRequest(
        stem=stem,
        video_id=str(task["video_id"]),
        start_seconds=float(task["start_index"]),
        detection_fps=float(effective_fps),
        tracks=tracks,
        source_fps=float(task["fps"]),
        first_source_frame=int(first_source_frame),
    )


def check_failure_share(failed: int, attempted: int) -> None:
    """Stop when the road surface could not be read for too many segments."""
    if attempted and failed / attempted > MAXIMUM_FAILED_SEGMENT_SHARE:
        raise RuntimeError(
            f"The road surface could not be read for {failed} of {attempted} detection "
            f"segments (more than {MAXIMUM_FAILED_SEGMENT_SHARE:.0%}), e.g. because the video "
            "file server is unreachable. Stopping rather than undercounting crossings."
        )


def select_road_crossings(
    df_mapping: pl.DataFrame,
    detection_tasks: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    crossing_parameters: Mapping[str, Any],
    failure_totals: Optional[List[int]] = None,
) -> Dict[str, List[Any]]:
    """Return, per detection segment, the candidates that satisfy rule D.

    ``failure_totals``, when given, is a ``[failed, attempted]`` accumulator:
    failures are added to it instead of being checked here, so a caller that
    selects in several small batches can apply the failure limit to the whole
    run with check_failure_share.

    ``candidates`` is ``{stem: {"ids", "id_bounds", "size_rates", ...}}`` from
    the detection workers (see utils/crossing/road_crossing.py). Every
    candidate is segmented through the same store as the metric pass, which
    later reuses these labels. Rule D needs to know where the feet are, so if
    segmentation cannot run this raises rather than silently counting
    crossings another way.
    """
    from crowd_city.crossing.road_crossing import road_crossing_flags

    pipeline = _segmentation_pipeline(df_mapping, crossing_parameters)
    if pipeline is None:
        raise RuntimeError(
            "crossing_rule is 'road_crossing', which needs road-surface segmentation, "
            "but the segmentation pipeline could not be set up (see the messages above)."
        )
    tasks_by_stem = {str(task["filename_no_ext"]): task for task in detection_tasks}
    pending = sorted(
        stem for stem, payload in candidates.items()
        if (payload or {}).get("ids") and stem in tasks_by_stem
    )
    minimum_x_range = float(crossing_parameters["min_crossing_x_range"])
    logger.info(
        f"Selecting road crossings: segmenting {sum(len(candidates[stem]['ids']) for stem in pending)} "
        f"candidate track(s) in {len(pending)} detection segment(s)."
    )

    def select_one(stem: str) -> Tuple[str, Any]:
        payload = candidates[stem]
        task = tasks_by_stem[stem]
        try:
            detections, effective_fps, first_source_frame = _prepared_detections(task)
            tracks = _crossing_tracks(detections, list(payload["ids"]), payload.get("id_bounds") or {})
            if not tracks:
                return "diagnostic", "candidate_tracks_missing"
            timelines = pipeline.timelines(
                _segment_request(task, stem, effective_fps, first_source_frame, tracks)
            )
        except Exception as error:
            _raise_if_gpu_failure(error)
            logger.warning(f"Road-crossing selection failed for {stem}: {error}")
            return "diagnostic", "segment_exception"
        if not timelines:
            return "diagnostic", "no_surface_timeline"
        intervals = road_intervals_for_tracks(timelines)
        rates = {str(key): value for key, value in (payload.get("size_rates") or {}).items()}
        return "selected", [
            track_id
            for track_id in payload["ids"]
            if str(track_id) in tracks
            and road_crossing_flags(
                tracks[str(track_id)], intervals.get(str(track_id)), rates.get(str(track_id)), minimum_x_range,
            )["road_crossing"]
        ]

    selected: Dict[str, List[Any]] = {}
    diagnostics: Counter = Counter()
    try:
        for stem, (kind, value) in _parallel_segments(select_one, pending, "Selecting road crossings"):
            if kind == "diagnostic":
                diagnostics[value] += 1
            elif value:
                selected[stem] = value
    finally:
        pipeline.close()
    diagnostics.update(pipeline.statistics)
    failed = diagnostics["segment_exception"] + diagnostics["no_surface_timeline"]
    if failure_totals is not None:
        failure_totals[0] += failed
        failure_totals[1] += len(pending)
    else:
        check_failure_share(failed, len(pending))
    logger.info(
        f"Road crossings selected: {sum(len(v) for v in selected.values())} track(s) in "
        f"{len(selected)} segment(s), from {sum(len(candidates[stem]['ids']) for stem in pending)} candidates."
    )
    for reason, count in sorted(diagnostics.items(), key=lambda item: -item[1]):
        if count:
            logger.info(f"Road-crossing selection diagnostic: {reason}={count}.")
    return selected


def _process_segment(
    df_mapping: pl.DataFrame,
    pipeline: SegmentationPipeline,
    task: Mapping[str, Any],
    stem: str,
    track_ids: List[Any],
    checks_per_second: float,
    id_bounds: Optional[Mapping[Any, Tuple[int, int]]] = None,
) -> Optional[Tuple[Dict[str, float], Dict[str, float], Counter]]:
    """Segment one detection segment and derive both metrics from it."""
    diagnostics: Counter = Counter()

    detections, effective_fps, first_source_frame = _prepared_detections(task)
    if detections.height == 0:
        diagnostics["empty_detection_file"] += 1
        return None

    tracks = _crossing_tracks(detections, track_ids, id_bounds)
    if not tracks:
        diagnostics["crossing_tracks_missing"] += 1
        return None

    timelines = pipeline.timelines(
        _segment_request(task, stem, effective_fps, first_source_frame, tracks)
    )
    if not timelines:
        diagnostics["no_surface_timeline"] += 1
        return None

    intervals = road_intervals_for_tracks(timelines)
    diagnostics["tracks_without_road_interval"] += len(tracks) - len(intervals)
    if not intervals:
        return None

    time_values: Dict[str, float] = {}
    for track_id, interval in intervals.items():
        wait = hesitation_seconds(
            tracks[track_id],
            interval.entry_frame,
            float(effective_fps),
            checks_per_second,
        )
        if wait is None:
            # The track began with the pedestrian already on the road, so the
            # wait was never in view. Counted, so the share of crossings that
            # cannot contribute a hesitation time stays visible in the log.
            diagnostics["hesitation_unobservable"] += 1
            continue
        if wait <= 0.0:
            # The pedestrian did not stand still before stepping onto the
            # road. Like the bounding-box metric, which only records a track
            # once it has a stationary run, such a crossing has no hesitation
            # time rather than a hesitation time of zero, so it is left out
            # of the averages instead of pulling them towards zero.
            diagnostics["hesitation_absent"] += 1
            continue
        time_values[track_id] = float(wait)

    speed_values, speed_diagnostics = road_restricted_speed(
        df_mapping,
        detections,
        stem,
        float(effective_fps),
        intervals,
    )
    diagnostics.update(speed_diagnostics)

    return speed_values, time_values, diagnostics


# ---------------------------------------------------------------------
# Aggregation
#
# The baseline hesitation metric stores sample counts and divides by
# check_per_sec_time when averaging. The segmentation metric stores seconds
# directly, so it needs its own aggregation rather than the baseline helpers.
# ---------------------------------------------------------------------

def average_by_locality(
    nested: Mapping[str, Mapping[str, Mapping[str, float]]],
    minimum: float = 0.0,
    maximum: float = float("inf"),
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Average per-track values by locality and condition."""
    averages: Dict[str, float] = {}
    complete: Dict[str, List[float]] = {}
    for locality, videos in (nested or {}).items():
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


def average_by_country(
    df_mapping: pl.DataFrame,
    nested: Mapping[str, Mapping[str, Mapping[str, float]]],
    minimum: float = 0.0,
    maximum: float = float("inf"),
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Average per-track values by country and condition."""
    from crowd_city.core.metadata import MetaData

    metadata = MetaData()
    grouped: Dict[str, List[float]] = {}
    for videos in (nested or {}).values():
        for video_id, tracks in videos.items():
            result = metadata.find_values_with_video_id(df_mapping, video_id)
            if result is None:
                continue
            key = f"{result[8]}_{result[3]}"
            for value in tracks.values():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if minimum <= numeric <= maximum:
                    grouped.setdefault(key, []).append(numeric)
    averages = {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }
    return averages, grouped
