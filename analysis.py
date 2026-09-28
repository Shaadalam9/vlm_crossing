"""
Crossing speed and hesitation time per city from the tracking CSVs.

Reads <output>/videos.csv written by main.py, detects road crossings in every
tracking CSV (utils/crossing/detection.py), measures each crossing
(utils/crossing/metrics.py) and writes:

- <output>/crossings.csv      one row per detected crossing
- <output>/city_summary.csv   one row per city
- <output>/figures/           plots per city

Adapted from analysis.py in crowd-dataset/crowd-city.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from logmod import logs
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


def process_video(video: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect and measure every crossing in one video.

    Args:
        video (dict): Row of videos.csv.

    Returns:
        list: One dict per crossing.
    """
    try:
        df = load_bbox_csv(video["csv"])
    except Exception as e:
        logger.error(f"Could not read {video['csv']}: {e}")
        return []
    if df.height == 0:
        return []

    fps = float(video["fps"])
    aspect = float(video["width"]) / float(video["height"])
    min_x = float(common.get_configs("boundary_left"))
    max_x = float(common.get_configs("boundary_right"))

    pedestrian_ids, _, bounds = detection.pedestrian_crossing(df, fps, min_x, max_x)

    rows = []
    for track_id in pedestrian_ids:
        result = metrics.crossing_metrics(df, track_id, bounds[track_id], fps, aspect, video["city"], min_x, max_x)
        if result is None:
            continue
        if not metrics.within_limits(result["speed_mps"], "min_speed_limit", "max_speed_limit"):
            result["speed_mps"] = None
        if not metrics.within_limits(result["hesitation_s"], "min_waiting_time", "max_waiting_time"):
            result["hesitation_s"] = None
            result["hesitated"] = None
        result["start_time_s"] = result["start_frame"] / fps
        rows.append({"city": video["city"], "video": video["video"], **result})
    return rows


def load_videos() -> pl.DataFrame:
    """Return the videos of videos.csv whose tracking CSV exists, limited to cities_analyse."""
    path = os.path.join(common.get_output_dir(), VIDEOS_CSV)
    if not os.path.exists(path):
        logger.error(f"{path} not found. Run main.py first.")
        return pl.DataFrame()
    videos = pl.read_csv(path)
    cities = [c.lower() for c in common.get_configs("cities_analyse")]
    if cities:
        videos = videos.filter(pl.col("city").str.to_lowercase().is_in(cities))
    exists = [os.path.exists(p) for p in videos["csv"].to_list()]
    missing = len(exists) - sum(exists)
    if missing:
        logger.warning(f"{missing} videos have no tracking CSV yet and are skipped.")
    return videos.filter(pl.Series(exists))


if __name__ == "__main__":
    videos = load_videos()
    if videos.height == 0:
        logger.error("No tracked videos to analyse.")
        raise SystemExit(1)
    logger.info(f"Analysing {videos.height} videos from {videos['city'].n_unique()} cities.")

    video_rows = videos.to_dicts()
    workers = max(1, int(common.get_configs("cpu_worker")))
    crossing_rows: List[Dict[str, Any]] = []
    if workers == 1:
        for row in tqdm(video_rows, unit="video"):
            crossing_rows.extend(process_video(row))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for rows in tqdm(executor.map(process_video, video_rows), total=len(video_rows), unit="video"):
                crossing_rows.extend(rows)

    output = common.get_output_dir()
    crossings = pl.DataFrame(crossing_rows, infer_schema_length=None) if crossing_rows else pl.DataFrame(
        schema={"city": pl.Utf8, "video": pl.Utf8, "speed_mps": pl.Float64, "hesitation_s": pl.Float64,
                "hesitated": pl.Boolean, "crossing_time_s": pl.Float64})
    crossings.write_csv(os.path.join(output, "crossings.csv"))
    logger.info(f"Detected {crossings.height} crossings; wrote {os.path.join(output, 'crossings.csv')}.")

    summary = metrics.city_summary(crossings, videos)
    summary.write_csv(os.path.join(output, "city_summary.csv"))
    logger.info(f"Wrote {os.path.join(output, 'city_summary.csv')}.")
    with pl.Config(tbl_rows=50, tbl_cols=12, fmt_str_lengths=30):
        logger.info("\n{}", summary.select([c for c in (
            "city", "videos", "footage_h", "crossings", "speed_mean_mps", "speed_median_mps",
            "hesitation_mean_s", "share_hesitated") if c in summary.columns]))

    crossings_plots.plot_all(crossings, summary)
