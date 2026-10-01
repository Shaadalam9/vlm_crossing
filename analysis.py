"""
Crossing speed and hesitation time per city, computed with the crowd-city code.

Reads <output>/videos.csv written by main.py and runs, for every tracking CSV,
the crowd-city analysis in crowd_city/ (an unchanged copy, see
crowd_city/__init__.py) in the same order as crowd-city's analysis.py:

1. csv_parallel.process_csv_task: confidence filter, crossing detection, and
   the bounding-box crossing time, speed and initiation time, plus the
   road-crossing (rule D) candidates.
2. With crossing_rule = "road_crossing": crossing_pass.select_road_crossings
   segments the candidates with SegFormer and keeps those that satisfy rule D,
   and the counted crossings and their bounding-box values are rebuilt from
   that selection as in crowd-city's _apply_road_crossing_rule.
3. With use_segmentation: crossing_pass.run_segmentation_pass measures the
   road-restricted speed and the hesitation time ending at road entry.

No Waymo speed model is used, so crowd-city reports speed as its relative
motion index: a pedestrian's speed divided by the median of the comparable
pedestrians in the same video (1.0 = typical; at least three are needed).
Hesitation times are in seconds. segmentation_is_primary picks which of the
two derivations is reported in speed_index / hesitation_s; both are always
kept. Outputs:

- <output>/analysis/<city>/<csv>.json   per-video cache (see utils/cache.py)
- <output>/segmentation/                crowd-city's surface-label store
- <output>/crossings.parquet and .csv   one row per crossing
- <output>/city_summary.csv             one row per city
- <output>/figures/                     plotly figures

Only videos whose tracking CSV, video, relevant settings or crowd-city code
changed are recomputed; run with --force to recompute all.
"""

import argparse
import contextlib
import glob
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from logmod import logs
from utils import cache
from utils.plotting.crossings import Crossings
from utils.summary import city_summary

import crowd_city.analytics.csv_parallel as csv_parallel
import crowd_city.crossing.metrics as crossing_metrics
import crowd_city.segmentation.crossing_pass as crossing_pass
from crowd_city.core.grouping import ALL
from crowd_city.crossing.detection import Detection
from crowd_city.segmentation.store import SurfaceStore, configured_segmentation_root

# Suppress a specific FutureWarning emitted by plotly.
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

VIDEOS_CSV = "videos.csv"
# Settings that change the per-video crowd-city results. Limits, plots and
# segmentation_is_primary only affect the aggregation and are not cached.
RESULT_CONFIG_KEYS = (
    "min_confidence", "boundary_left", "boundary_right", "check_per_sec_time", "processing_fps",
    "crossing_rule", "use_segmentation", "segmentation_model", "segmentation_input_width",
    "segmentation_input_height", "segmentation_min_confidence", "segmentation_coarse_hz",
    "segmentation_refine_hz",
)
CROSSINGS_SCHEMA = {
    "city": pl.Utf8,
    "video": pl.Utf8,
    "track_id": pl.Utf8,
    "start_frame": pl.Int64,
    "end_frame": pl.Int64,
    "start_time_s": pl.Float64,
    "crossing_time_s": pl.Float64,
    "speed_bbox_index": pl.Float64,
    "hesitation_bbox_s": pl.Float64,
    "speed_seg_index": pl.Float64,
    "hesitation_seg_s": pl.Float64,
    "speed_index": pl.Float64,
    "hesitation_s": pl.Float64,
}


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def load_videos() -> pl.DataFrame:
    """Return the videos of videos.csv whose tracking CSV exists, limited to cities_analyse."""
    path = os.path.join(common.get_output_dir(), VIDEOS_CSV)
    if not os.path.exists(path):
        logger.error(f"{path} not found. Run main.py first.")
        return pl.DataFrame()
    try:
        videos = pl.read_csv(path, schema_overrides={"city": pl.Utf8, "video": pl.Utf8, "csv": pl.Utf8})
    except pl.exceptions.NoDataError:
        return pl.DataFrame()
    cities = [c.lower() for c in common.get_configs("cities_analyse")]
    if cities:
        videos = videos.filter(pl.col("city").str.to_lowercase().is_in(cities))
    exists = [os.path.exists(p) for p in videos["csv"].to_list()]
    missing = len(exists) - sum(exists)
    if missing:
        logger.warning(f"{missing} videos have no tracking CSV yet and are skipped.")
    return videos.filter(pl.Series(exists))


def crossing_parameters() -> Dict[str, Any]:
    """crowd-city's fixed original CROWD crossing parameters, with the configured road strip."""
    parameters = Detection.crossing_parameter_defaults()
    parameters["boundary_left"] = float(common.get_configs("boundary_left"))
    parameters["boundary_right"] = float(common.get_configs("boundary_right"))
    return parameters


@contextlib.contextmanager
def nothing():
    yield None


def crowd_city_fingerprint() -> str:
    """Fingerprint of every crowd_city module and of this driver."""
    files = sorted(glob.glob(os.path.join(common.root_dir, "crowd_city", "**", "*.py"), recursive=True))
    return cache.make_key(**{os.path.relpath(f, common.root_dir): cache.file_fingerprint(f)
                             for f in files + [os.path.abspath(__file__)]})


# ---------------------------------------------------------------------
# crowd-city tasks
# ---------------------------------------------------------------------

def make_task(video: Dict[str, Any]) -> Dict[str, Any]:
    """Describe one video the way crowd-city's analysis.py describes a detection file.

    crowd-city names detection files <video_id>_<start_seconds>_<fps> and reads
    the fps back from that name, so the stem follows the same pattern.
    """
    fps = int(round(float(video["fps"])))
    video_id = f"{video['city']}__{video['video']}"
    stem = f"{video_id}_0_{fps}"
    return {
        "file_path": video["csv"],
        "file_name": os.path.basename(video["csv"]),
        "filename_no_ext": stem,
        "fps": fps,
        "video_locality_id": 0,
        "is_bbox_stream": True,
        "time_video": 0.0,  # analyse the whole video
        "video_id": video_id,
        "start_index": 0,
        "video_path": video["video_path"],
    }


def nested_values(nested: Optional[dict], stem: str) -> Dict[str, float]:
    """Values of one stem from a {locality: {stem: {track: value}}} structure, keyed by normalised id."""
    values = ((nested or {}).get(ALL) or {}).get(stem) or {}
    return {crossing_metrics.normalise_id(k): float(v) for k, v in values.items() if v is not None}


def run_crowd_city(tasks: List[Dict[str, Any]], crossing_parameters: Dict[str, Any]) -> Dict[str, List[dict]]:
    """Run crowd-city's analysis on these tasks; return the crossing rows of each stem."""
    mapping = pl.DataFrame()
    init_args = (
        mapping,
        dict(crossing_parameters),
        float(common.get_configs("min_confidence")),
        float(common.get_configs("boundary_left")),
        float(common.get_configs("boundary_right")),
        dict(crossing_metrics._PIPELINE_MODEL),
        dict(crossing_metrics._SPEED_MODEL),
    )

    # 1. Detection and bounding-box metrics, per detection file.
    results: Dict[str, dict] = {}
    workers = max(1, min(int(common.get_configs("cpu_worker")), len(tasks)))
    if workers == 1:
        csv_parallel.initialise_csv_worker(*init_args)
        executor_context = nothing()
    else:
        # spawn, not fork, as in crowd-city: forking after polars has started its thread pool can deadlock.
        executor_context = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                               initializer=csv_parallel.initialise_csv_worker, initargs=init_args)
    with executor_context as executor:
        mapped = map(csv_parallel.process_csv_task, tasks) if executor is None \
            else executor.map(csv_parallel.process_csv_task, tasks, chunksize=1)
        for result in tqdm(mapped, total=len(tasks), desc="Analysing detection files"):
            if result.get("status") != "ok":
                logger.error(result.get("message", "detection worker failed"))
                continue
            results[result["filename_no_ext"]] = result
    # The parent needs the same worker state for the segmentation steps.
    csv_parallel.initialise_csv_worker(*init_args)

    # Point the segmentation store at the local video files, so crowd-city's
    # pipeline reads them instead of resolving video ids on its file server.
    root = configured_segmentation_root()
    if root:
        store = SurfaceStore(root)
        store.ensure_directories()
        for task in tasks:
            store.store_url(task["video_id"], os.path.abspath(task["video_path"]))

    # 2. Which crossings are counted.
    crossings: Dict[str, dict] = {}
    if str(common.get_configs("crossing_rule")) == "road_crossing":
        road_candidates = {stem: r["road_candidates"] for stem, r in results.items() if r.get("road_candidates")}
        selected = crossing_pass.select_road_crossings(mapping, tasks, road_candidates, crossing_parameters)
        # As crowd-city's analysis._apply_road_crossing_rule: rule D's crossings replace the detector's.
        for stem in results:
            payload = road_candidates.get(stem) or {}
            chosen = list(selected.get(stem, []))
            keep = {crossing_metrics.normalise_id(track_id) for track_id in chosen}
            bounds = payload.get("id_bounds") or {}
            crossings[stem] = {
                "ids": chosen,
                "id_bounds": {track_id: bounds[track_id] for track_id in chosen if track_id in bounds},
                "temp_data": {k: v for k, v in (payload.get("temp_data") or {}).items()
                              if crossing_metrics.normalise_id(k) in keep},
                "speed_value": payload.get("speed_value"),
                "time_value": payload.get("time_value"),
            }
    else:
        for stem, r in results.items():
            keys = ("ids", "id_bounds", "temp_data", "speed_value", "time_value")
            crossings[stem] = {key: r.get(key) for key in keys}

    # 3. Road-surface metrics for the counted crossings.
    seg_speed: dict = {}
    seg_time: dict = {}
    if bool(common.get_configs("use_segmentation")):
        output = crossing_pass.run_segmentation_pass(
            df_mapping=mapping,
            detection_tasks=tasks,
            crossing_ids={stem: {"ids": c["ids"], "id_bounds": c["id_bounds"]} for stem, c in crossings.items()},
            crossing_parameters=crossing_parameters,
        )
        if not output.get("enabled"):
            raise SystemExit("use_segmentation is enabled but crowd-city's segmentation pass could not run.")
        seg_speed, seg_time = output["seg_speed"], output["seg_time"]

    # 4. One row per counted crossing. crowd-city stores the bounding-box
    # initiation time in check_per_sec_time units and divides when averaging.
    checks = float(common.get_configs("check_per_sec_time"))
    rows_by_stem: Dict[str, List[dict]] = {}
    for stem, c in crossings.items():
        fps = float(stem.rsplit("_", 1)[1])
        speed_bbox = nested_values(c.get("speed_value"), stem)
        time_bbox = nested_values(c.get("time_value"), stem)
        speed_seg = nested_values(seg_speed, stem)
        time_seg = nested_values(seg_time, stem)
        duration = {crossing_metrics.normalise_id(k): float(v) for k, v in (c.get("temp_data") or {}).items()}
        bounds = c.get("id_bounds") or {}
        rows = []
        for track_id in c.get("ids") or []:
            tid = crossing_metrics.normalise_id(track_id)
            start, end = bounds.get(track_id, (None, None))
            rows.append({
                "track_id": tid,
                "start_frame": start,
                "end_frame": end,
                "start_time_s": None if start is None else float(start) / fps,
                "crossing_time_s": duration.get(tid),
                "speed_bbox_index": speed_bbox.get(tid),
                "hesitation_bbox_s": None if tid not in time_bbox else time_bbox[tid] / checks,
                "speed_seg_index": speed_seg.get(tid),
                "hesitation_seg_s": time_seg.get(tid),
            })
        rows_by_stem[stem] = rows
    return rows_by_stem


# ---------------------------------------------------------------------
# Cache and aggregation
# ---------------------------------------------------------------------

def cache_path(video: Dict[str, Any]) -> str:
    name = os.path.splitext(os.path.basename(video["csv"]))[0]
    return os.path.join(common.get_output_dir(), "analysis", video["city"], name + ".json")


def result_key(video: Dict[str, Any], code: str) -> str:
    return cache.make_key(
        code=code,
        csv=cache.input_fingerprint(video["csv"]),
        video=cache.input_fingerprint(video["video_path"]),
        fps=float(video["fps"]),
        config={key: common.get_configs(key) for key in RESULT_CONFIG_KEYS},
    )


def reported(crossings: pl.DataFrame) -> pl.DataFrame:
    """Pick the reported derivation and apply crowd-city's speed and waiting-time limits."""
    lo_s, hi_s = float(common.get_configs("min_speed_limit")), float(common.get_configs("max_speed_limit"))
    lo_w, hi_w = float(common.get_configs("min_waiting_time")), float(common.get_configs("max_waiting_time"))

    def within(column: str, lo: float, hi: float) -> pl.Expr:
        return pl.when(pl.col(column).is_between(lo, hi)).then(pl.col(column)).alias(column)

    crossings = crossings.with_columns(
        within("speed_bbox_index", lo_s, hi_s), within("speed_seg_index", lo_s, hi_s),
        within("hesitation_bbox_s", lo_w, hi_w), within("hesitation_seg_s", lo_w, hi_w),
    )
    primary = bool(common.get_configs("use_segmentation")) and bool(common.get_configs("segmentation_is_primary"))
    suffix = "seg" if primary else "bbox"
    return crossings.with_columns(
        pl.col(f"speed_{suffix}_index").alias("speed_index"),
        pl.col(f"hesitation_{suffix}_s").alias("hesitation_s"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--force", action="store_true", help="ignore cached results and analyse every video again")
    args = parser.parse_args()
    # crowd-city resolves seg_data and model names relative to the working directory.
    os.chdir(common.root_dir)

    videos = load_videos()
    if videos.height == 0:
        logger.error("No tracked videos to analyse.")
        raise SystemExit(1)
    logger.info(f"Analysing {videos.height} videos from {videos['city'].n_unique()} cities "
                f"with crowd-city {open(os.path.join(common.root_dir, 'crowd_city', 'SOURCE_COMMIT')).read().strip()}.")

    parameters = crossing_parameters()
    code_fp = crowd_city_fingerprint()

    video_rows = videos.to_dicts()
    cached_rows: Dict[str, List[dict]] = {}
    todo = []
    for video in video_rows:
        key = result_key(video, code_fp)
        stored = {} if args.force else cache.load(cache_path(video))
        hit = cache.stage(stored, "crowd_city", key)
        if hit is None:
            todo.append((video, key))
        else:
            cached_rows[video["csv"]] = hit["rows"]
    logger.info(f"Videos: {len(cached_rows)} from cache, {len(todo)} to analyse.")

    if todo:
        tasks = [make_task(video) for video, _ in todo]
        rows_by_stem = run_crowd_city(tasks, parameters)
        for (video, key), task in zip(todo, tasks):
            rows = rows_by_stem.get(task["filename_no_ext"])
            if rows is None:
                logger.error(f"No crowd-city result for {video['csv']}; not cached.")
                continue
            cache.save(cache_path(video), {"city": video["city"], "video": video["video"], "csv": video["csv"],
                                           "crowd_city": {"key": key, "rows": rows}})
            cached_rows[video["csv"]] = rows

    crossing_rows = [{"city": v["city"], "video": v["video"], **row}
                     for v in video_rows for row in cached_rows.get(v["csv"], [])]
    output = common.get_output_dir()
    crossings = pl.DataFrame(crossing_rows, schema={k: v for k, v in CROSSINGS_SCHEMA.items()
                                                    if k not in ("speed_index", "hesitation_s")}, strict=False) \
        if crossing_rows else pl.DataFrame(schema={k: v for k, v in CROSSINGS_SCHEMA.items()
                                                   if k not in ("speed_index", "hesitation_s")})
    crossings = reported(crossings).select(list(CROSSINGS_SCHEMA)).sort(["city", "video", "start_frame"])
    crossings.write_parquet(os.path.join(output, "crossings.parquet"))
    crossings.write_csv(os.path.join(output, "crossings.csv"))
    logger.info(f"Counted {crossings.height} crossings; wrote {os.path.join(output, 'crossings.parquet')} and .csv.")

    summary = city_summary(crossings, videos)
    summary.write_csv(os.path.join(output, "city_summary.csv"))
    logger.info(f"Wrote {os.path.join(output, 'city_summary.csv')}.")
    with pl.Config(tbl_rows=50, tbl_cols=12, fmt_str_lengths=30):
        logger.info("\n{}", summary.select("city", "crossings", "speed_index_mean", "crossings_with_speed",
                                           "hesitation_mean_s", "crossings_with_hesitation"))

    Crossings().plot_all(crossings, summary)
