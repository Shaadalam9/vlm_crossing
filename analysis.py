"""
Crossing speed and hesitation time per city from the tracking CSVs.

Reads <output>/videos.csv written by main.py, detects road crossings in every
tracking CSV (utils/crossing/detection.py), measures each crossing
(utils/crossing/metrics.py) and writes:

- <output>/analysis/<city>/<csv>.json   per-video cache (see utils/cache.py)
- <output>/crossings.parquet and .csv   one row per detected crossing
- <output>/city_summary.csv             one row per city
- <output>/figures/                     plots per city

Only videos whose tracking CSV, relevant settings or analysis code changed are
recomputed; run with --force to recompute everything.

Adapted from analysis.py in crowd-dataset/crowd-city.
"""

import argparse
import multiprocessing
import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List, Tuple

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from logmod import logs
from utils import cache
from utils.crossing import detection as detection_module
from utils.crossing import metrics as metrics_module
from utils.crossing.detection import Detection
from utils.crossing.metrics import Metrics
from utils.plotting.crossings import Crossings

# Suppress a specific FutureWarning emitted by plotly.
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

detection = Detection()
metrics = Metrics()
crossings_plots = Crossings()

VIDEOS_CSV = "videos.csv"
CROSSINGS_SCHEMA = {
    "city": pl.Utf8,
    "video": pl.Utf8,
    "track_id": pl.Int64,
    "start_frame": pl.Int64,
    "end_frame": pl.Int64,
    "start_time_s": pl.Float64,
    "n_frames": pl.Int64,
    "direction": pl.Utf8,
    "crossing_time_s": pl.Float64,
    "speed_mps": pl.Float64,
    "hesitation_s": pl.Float64,
    "hesitated": pl.Boolean,
    "median_height_norm": pl.Float64,
    "person_height_m": pl.Float64,
}
BBOX_SCHEMA = {
    "yolo-id": pl.Int64,
    "x-center": pl.Float64,
    "y-center": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
    "unique-id": pl.Int64,
    "confidence": pl.Float64,
    "frame-count": pl.Int64,
}


def load_bbox_csv(path: str) -> pl.DataFrame:
    """Read a tracking CSV. CROWD CSVs may store ids as floats, so ids are read as floats and cast."""
    df = pl.read_csv(path, schema_overrides={k: pl.Float64 for k in BBOX_SCHEMA})
    return df.with_columns([pl.col(k).cast(v) for k, v in BBOX_SCHEMA.items() if k in df.columns])


def cache_path(video: Dict[str, Any]) -> str:
    """<output>/analysis/<city>/<csv name>.json; the CSV name carries the fps, so it is unique per city."""
    name = os.path.splitext(os.path.basename(video["csv"]))[0]
    return os.path.join(common.get_output_dir(), "analysis", video["city"], name + ".json")


def detection_key(video: Dict[str, Any], csv_fp: Dict[str, int], min_x: float, max_x: float) -> str:
    return cache.make_key(
        code=cache.file_fingerprint(detection_module.__file__),
        csv=csv_fp, fps=float(video["fps"]), boundary=[min_x, max_x],
    )


def metrics_key(video: Dict[str, Any], csv_fp: Dict[str, int], crossings: List[List[int]],
                min_x: float, max_x: float) -> str:
    return cache.make_key(
        code=cache.file_fingerprint(metrics_module.__file__),
        dedup=cache.object_fingerprint(Detection._dedup_per_frame),
        csv=csv_fp, crossings=crossings,
        fps=float(video["fps"]), size=[int(video["width"]), int(video["height"])],
        boundary=[min_x, max_x],
        stature_m=metrics.person_height_m(video["city"]),
        check_per_sec_time=common.get_configs("check_per_sec_time"),
        hesitation_reference=common.get_configs("hesitation_reference"),
    )


def process_video(video: Dict[str, Any], force: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    """
    Detect and measure every crossing in one video, reusing cached stages whose inputs did not change.

    Args:
        video (dict): Row of videos.csv.
        force (bool): Ignore the cache.

    Returns:
        tuple: (one dict per crossing, raw metrics without limits applied; status). status is
        "cached", "detection" (detection recomputed, same crossings so metrics reused),
        "metrics" (only metrics recomputed), "full" or "error".
    """
    path = cache_path(video)
    stored = {} if force else cache.load(path)
    min_x = float(common.get_configs("boundary_left"))
    max_x = float(common.get_configs("boundary_right"))
    fps = float(video["fps"])

    try:
        csv_fp = cache.input_fingerprint(video["csv"])
    except OSError as e:
        logger.error(f"Could not read {video['csv']}: {e}")
        return [], "error"

    det_key = detection_key(video, csv_fp, min_x, max_x)
    det = cache.stage(stored, "detection", det_key)
    df = None

    def frame() -> pl.DataFrame:
        nonlocal df
        if df is None:
            df = load_bbox_csv(video["csv"])
        return df

    detected = measured = False
    try:
        if det is None:
            detected = True
            pedestrian_ids, candidate_ids, bounds = (
                detection.pedestrian_crossing(frame(), fps, min_x, max_x) if frame().height else ([], [], {}))
            det = {
                "key": det_key,
                "crossings": [[int(i), int(bounds[i][0]), int(bounds[i][1])] for i in pedestrian_ids],
                "candidate_ids": [int(i) for i in candidate_ids],
            }

        met_key = metrics_key(video, csv_fp, det["crossings"], min_x, max_x)
        met = cache.stage(stored, "metrics", met_key)
        if met is None:
            measured = True
            aspect = float(video["width"]) / float(video["height"])
            rows = []
            for track_id, start_frame, end_frame in det["crossings"]:
                result = metrics.crossing_metrics(frame(), track_id, (start_frame, end_frame), fps, aspect,
                                                  video["city"], min_x, max_x)
                if result is not None:
                    rows.append({**result, "start_time_s": result["start_frame"] / fps})
            met = {"key": met_key, "rows": rows}
    except Exception as e:
        logger.error(f"Analysis failed for {video['csv']}: {e!r}")
        return [], "error"

    status = {(False, False): "cached", (True, False): "detection",
              (False, True): "metrics", (True, True): "full"}[(detected, measured)]
    if status != "cached":
        cache.save(path, {
            "city": video["city"], "video": video["video"], "csv": video["csv"],
            "detection": det, "metrics": met,
        })
    return [{"city": video["city"], "video": video["video"], **row} for row in met["rows"]], status


def apply_limits(crossings: pl.DataFrame) -> pl.DataFrame:
    """Null out speeds and hesitation times outside the configured limits (cheap, so never cached)."""
    lo_s, hi_s = float(common.get_configs("min_speed_limit")), float(common.get_configs("max_speed_limit"))
    lo_w, hi_w = float(common.get_configs("min_waiting_time")), float(common.get_configs("max_waiting_time"))
    wait_ok = pl.col("hesitation_s").is_between(lo_w, hi_w)
    return crossings.with_columns(
        pl.when(pl.col("speed_mps").is_between(lo_s, hi_s)).then(pl.col("speed_mps")).alias("speed_mps"),
        pl.when(wait_ok).then(pl.col("hesitated")).alias("hesitated"),
        pl.when(wait_ok).then(pl.col("hesitation_s")).alias("hesitation_s"),
    )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--force", action="store_true", help="ignore cached results and analyse every video again")
    args = parser.parse_args()

    videos = load_videos()
    if videos.height == 0:
        logger.error("No tracked videos to analyse.")
        raise SystemExit(1)
    logger.info(f"Analysing {videos.height} videos from {videos['city'].n_unique()} cities.")

    video_rows = videos.to_dicts()
    workers = max(1, int(common.get_configs("cpu_worker")))
    run = partial(process_video, force=args.force)
    crossing_rows: List[Dict[str, Any]] = []
    statuses: Counter = Counter()
    if workers == 1:
        results = map(run, video_rows)
    else:
        # spawn, not fork: forking after polars has started its thread pool can deadlock the workers.
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        results = executor.map(run, video_rows)
    for rows, status in tqdm(results, total=len(video_rows), unit="video"):
        crossing_rows.extend(rows)
        statuses[status] += 1
    if workers > 1:
        executor.shutdown()
    logger.info(f"Videos: {statuses['cached']} from cache, {statuses['detection']} detection only, "
                f"{statuses['metrics']} metrics only, {statuses['full']} fully analysed, {statuses['error']} failed.")

    output = common.get_output_dir()
    crossings = pl.DataFrame(crossing_rows, schema=CROSSINGS_SCHEMA, strict=False) if crossing_rows \
        else pl.DataFrame(schema=CROSSINGS_SCHEMA)
    crossings = apply_limits(crossings).sort(["city", "video", "start_frame"])
    crossings.write_parquet(os.path.join(output, "crossings.parquet"))
    crossings.write_csv(os.path.join(output, "crossings.csv"))
    logger.info(f"Detected {crossings.height} crossings; wrote {os.path.join(output, 'crossings.parquet')} and .csv.")

    summary = metrics.city_summary(crossings, videos)
    summary.write_csv(os.path.join(output, "city_summary.csv"))
    logger.info(f"Wrote {os.path.join(output, 'city_summary.csv')}.")
    with pl.Config(tbl_rows=50, tbl_cols=12, fmt_str_lengths=30):
        logger.info("\n{}", summary.select([c for c in (
            "city", "videos", "footage_h", "crossings", "speed_mean_mps", "speed_median_mps",
            "hesitation_mean_s", "share_hesitated") if c in summary.columns]))

    crossings_plots.plot_all(crossings, summary)
