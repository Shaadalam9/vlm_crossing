# -----------------------------------------------------------------------------
# Pipeline overview (tracking stage), adapted from main.py in crowd-dataset/crowd:
# - Finds dashcam videos laid out as <data>/<city>/<video>.
# - For each video, runs YOLOv11x detection with BoT-SORT tracking unless its CSV
#   already exists (or always_analyse is true), writing
#   <output>/bbox/<city>/<video>_<fps>.csv in the CROWD column format.
# - Writes <output>/videos.csv, the mapping of every video to its city, fps,
#   size, duration and CSV. analysis.py reads it to compute crossing metrics.
# - Optionally emails a summary or crash report.
# -----------------------------------------------------------------------------

import os
import traceback

import polars as pl

import common                                    # configs, secrets, email, git utils
from custom_logger import CustomLogger           # structured logging
from helper_script import Video_Helper           # discovery and tracking utilities
from logmod import logs                          # log level/color setup

# Configure logging based on config file (verbosity & ANSI colors)
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

VIDEOS_CSV = "videos.csv"


def bbox_dir() -> str:
    """Return the folder that holds the per-video tracking CSVs."""
    return os.path.join(common.get_output_dir(), "bbox")


def previous_mapping(path: str) -> dict:
    """Rows of an existing videos.csv keyed by video path, to skip re-reading unchanged videos."""
    if not os.path.exists(path):
        return {}
    try:
        old = pl.read_csv(path, schema_overrides={"city": pl.Utf8, "video": pl.Utf8})
    except Exception:
        return {}
    if not {"video_path", "file_size", "file_mtime_ns"}.issubset(old.columns):
        return {}
    return {row["video_path"]: row for row in old.iter_rows(named=True)}


def build_mapping(helper: Video_Helper, videos, previous: dict) -> pl.DataFrame:
    """
    Read metadata of every video and return the video mapping.

    Args:
        helper (Video_Helper): Helper instance.
        videos (list): (city, video_path) tuples.
        previous (dict): Output of previous_mapping; unchanged videos are not opened again.

    Returns:
        pl.DataFrame: One row per readable video.
    """
    rows = []
    seen = {}
    prop_keys = ("fps", "frames", "width", "height", "duration_s")
    for city, path in videos:
        stamp = helper.file_stamp(path)
        old = previous.get(os.path.abspath(path))
        if old is not None and all(old.get(k) == v for k, v in stamp.items()):
            props = {k: old[k] for k in prop_keys}
        else:
            props = helper.get_video_properties(path)
        if props is None:
            continue
        csv_name = helper.csv_name(path, props["fps"])
        key = (city, csv_name)
        if key in seen:
            logger.warning(f"{path} and {seen[key]} map to the same CSV {csv_name}; skipping {path}.")
            continue
        seen[key] = path
        rows.append({
            "city": city,
            "video": os.path.splitext(os.path.basename(path))[0],
            "video_path": os.path.abspath(path),
            "csv": os.path.join(bbox_dir(), city, csv_name),
            **props,
            **stamp,
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def track_videos(helper: Video_Helper, mapping: pl.DataFrame) -> dict:
    """
    Run tracking for every video in the mapping that has no CSV yet.

    Args:
        helper (Video_Helper): Helper instance.
        mapping (pl.DataFrame): Output of build_mapping.

    Returns:
        dict: Counts of tracked, skipped and failed videos.
    """
    always_analyse = common.get_configs("always_analyse")
    save_annotated_video = common.get_configs("save_annotated_video")
    runs_root = common.resolve_path(common.get_configs("runs_root"))
    counts = {"tracked": 0, "skipped": 0, "failed": 0}
    settings = helper.tracking_settings()
    outdated = []

    for i, row in enumerate(mapping.iter_rows(named=True), start=1):
        csv_path = row["csv"]
        if os.path.exists(csv_path) and not always_analyse:
            logger.debug(f"CSV for {row['video_path']} exists; skipping.")
            stored = helper.stored_tracking_settings(csv_path)
            if stored is not None and stored != settings:
                outdated.append(csv_path)
            counts["skipped"] += 1
            continue

        logger.info(f"[{i}/{mapping.height}] Tracking {row['city']}/{row['video']} "
                    f"({row['fps']} fps, {row['duration_s'] / 60:.1f} min) on {helper.device}.")
        annotated = None
        if save_annotated_video:
            annotated = os.path.join(common.get_output_dir(), "annotated", row["city"], row["video"] + ".mp4")
        try:
            ok = helper.tracking_mode(
                input_video_path=row["video_path"],
                output_csv=csv_path,
                video_fps=row["fps"],
                run_root=os.path.join(runs_root, row["city"], row["video"]),
                annotated_video_path=annotated,
            )
        except Exception as e:
            logger.error(f"Tracking failed for {row['video_path']}: {e}")
            ok = False
        counts["tracked" if ok else "failed"] += 1

    if outdated:
        logger.warning(f"{len(outdated)} tracking CSVs were made with different tracking settings than the current "
                       f"config, e.g. {outdated[0]}. They are kept; delete them, or set always_analyse to true, "
                       f"to track them again.")
    if common.get_configs("delete_runs_files"):
        helper.delete_folder(runs_root)
    return counts


if __name__ == "__main__":
    try:
        if common.get_configs("git_pull"):
            common.git_pull()

        helper = Video_Helper()
        data_dir = common.resolve_path(common.get_configs("data"))
        videos = helper.discover_videos(data_dir, common.get_configs("cities_analyse"))
        cities = sorted({city for city, _ in videos})
        logger.info(f"Found {len(videos)} videos in {len(cities)} cities under {data_dir}.")
        if not videos:
            raise SystemExit(0)

        mapping_path = os.path.join(common.get_output_dir(), VIDEOS_CSV)
        mapping = build_mapping(helper, videos, previous_mapping(mapping_path))
        mapping.write_csv(mapping_path)
        logger.info(f"Wrote video mapping to {mapping_path}.")

        counts = track_videos(helper, mapping)
        summary = (f"Tracking finished: {counts['tracked']} tracked, {counts['skipped']} already done, "
                   f"{counts['failed']} failed.")
        logger.info(summary)

        if common.get_configs("email_send"):
            common.send_email(subject="vlm_crossing: tracking finished", content=summary,
                              sender=common.get_configs("email_sender"),
                              recipients=common.get_configs("email_recipients"))
    except Exception as e:
        logger.error(f"Tracking crashed: {e}")
        if common.get_configs("email_send"):
            common.send_email(subject="vlm_crossing: tracking crashed", content=traceback.format_exc(),
                              sender=common.get_configs("email_sender"),
                              recipients=common.get_configs("email_recipients"))
        raise
