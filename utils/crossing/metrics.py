"""Crossing time, crossing speed and hesitation time for detected crossings.

Adapted from utils/crossing/metrics.py and utils/crossing/road_metrics.py in
crowd-dataset/crowd-city.

Position in metres
------------------
crowd-city converts box motion to m/s with a model calibrated on the Waymo Open
Dataset. That model is not available here, so this module uses the pinhole
relation that the calibration also builds on: a person of stature H metres
whose box is h pixels tall is at a depth where one pixel spans H / h metres.
The lateral position of the pedestrian relative to the optical axis is then

    X = (x_px - cx) * H / h_px = (x - 0.5) * aspect * H / h

with x and h the normalised box centre and height and aspect = width / height
of the frame. X stays constant for a standing person while the dashcam drives
towards them, so forward ego-motion cancels. Camera yaw (turning) does not
cancel; the detector's camera-motion filters remove the worst of those tracks.
Speeds are therefore estimates whose accuracy depends on the assumed stature
and on the whole body being visible in the box.

Hesitation time
---------------
Following crowd-city's time_to_start_cross, the track is sampled
check_per_sec_time times per second. Samples are taken on a grid of frame
numbers, not of array indices, so frames the detector missed do not shorten or
stretch the measured time; positions between detections are interpolated
(segments are already split at gaps longer than a second). A sample is
stationary when the lateral position moved at most 10% of the stature (0.17 m
for 1.70 m) since the previous sample. The hesitation time is the first run of
at least three stationary samples. With hesitation_reference = "road_entry" it
is instead the stationary run that ends when the pedestrian enters the road strip, walking
backwards as crowd-city's road_metrics.hesitation_seconds does. Tracks with no
such run get 0.0 and hesitated = False. Tracks that start already on the road
have no observable wait and get None for "road_entry".
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl

import common
from utils.crossing.detection import PERSON_CLASS_ID, Detection

# Stationarity margin, as a fraction of the stature. Identical to crowd-city.
STATIONARY_MARGIN_FRACTION = 0.10
# Consecutive stationary samples required before a wait is reported at all.
MINIMUM_STATIONARY_SAMPLES = 3
# Smoothing window for box height and position, in seconds.
SMOOTHING_SECONDS = 0.3
# Shortest moving interval a speed is fitted on, in seconds.
MINIMUM_SPEED_SECONDS = 0.5
# Largest number of points used by the Theil-Sen fit (pairwise slopes are O(n^2)).
MAXIMUM_FIT_POINTS = 300


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median with edge windows shrunk to fit."""
    if window <= 1 or values.size < 3:
        return values.astype(float)
    half = window // 2
    return np.array([np.nanmedian(values[max(0, i - half):i + half + 1]) for i in range(values.size)])


def theil_sen_slope(t: np.ndarray, x: np.ndarray) -> Optional[float]:
    """Median of pairwise slopes: a line fit that ignores up to ~29% outliers."""
    if t.size < 2:
        return None
    if t.size > MAXIMUM_FIT_POINTS:
        idx = np.linspace(0, t.size - 1, MAXIMUM_FIT_POINTS).astype(int)
        t, x = t[idx], x[idx]
    dt = t[None, :] - t[:, None]
    dx = x[None, :] - x[:, None]
    mask = np.triu(np.ones_like(dt, dtype=bool), k=1) & (dt > 0)
    if not mask.any():
        return None
    return float(np.median(dx[mask] / dt[mask]))


class Metrics:
    """Per-crossing metrics used by analysis.py."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def person_height_m(city: str) -> float:
        """Return the assumed stature for a city (city_person_height_m, else person_height_m)."""
        per_city = common.get_configs("city_person_height_m") or {}
        for name, value in per_city.items():
            if str(name).lower() == str(city).lower():
                return float(value)
        return float(common.get_configs("person_height_m"))

    @staticmethod
    def crossing_track(df: pl.DataFrame, track_id: Any, bounds: Tuple[int, int]) -> pl.DataFrame:
        """Return the rows of one person track inside its crossing segment, one row per frame."""
        track = df.filter(
            (pl.col("yolo-id") == PERSON_CLASS_ID)
            & (pl.col("unique-id") == track_id)
            & pl.col("frame-count").is_between(bounds[0], bounds[1])
        )
        return Detection._dedup_per_frame(track).sort("frame-count")

    @staticmethod
    def lateral_position_m(track: pl.DataFrame, fps: float, aspect: float, stature_m: float) -> np.ndarray:
        """Lateral position of the pedestrian in metres relative to the optical axis."""
        window = max(1, int(round(SMOOTHING_SECONDS * fps)) | 1)
        x = rolling_median(track.get_column("x-center").cast(pl.Float64).to_numpy(), window)
        h = rolling_median(track.get_column("height").cast(pl.Float64).to_numpy(), window)
        h = np.clip(h, 1e-6, None)
        return (x - 0.5) * aspect * stature_m / h

    @staticmethod
    def time_to_cross(track: pl.DataFrame, fps: float) -> Optional[float]:
        """Return the observed duration of the crossing segment in seconds."""
        if track.height < 2 or fps <= 0:
            return None
        frames = track.get_column("frame-count")
        duration = (float(frames.max()) - float(frames.min())) / fps
        return duration if duration > 0 else None

    @staticmethod
    def time_to_start_cross(frames: np.ndarray, position_m: np.ndarray, fps: float, stature_m: float,
                            checks_per_second: float) -> Tuple[Optional[float], Optional[int]]:
        """
        Estimate the stationary interval before the pedestrian starts to move (crowd-city baseline).

        Returns:
            tuple: (hesitation seconds, frame at which walking starts). (0.0, None) when no
            stationary run of MINIMUM_STATIONARY_SAMPLES samples exists.
        """
        step = max(1, int(round(fps / checks_per_second)))
        if frames.size < 2 or int(frames[-1]) - int(frames[0]) < step:
            return None, None
        # Sample on frame numbers so missed detections do not distort the duration.
        targets = np.arange(int(frames[0]), int(frames[-1]) + 1, step)
        position = np.interp(targets, frames, position_m)
        margin = STATIONARY_MARGIN_FRACTION * stature_m

        stable_samples = 0
        end_index = None
        for index in range(targets.size - 1):
            if abs(float(position[index + 1]) - float(position[index])) <= margin:
                stable_samples += 1
            elif stable_samples >= MINIMUM_STATIONARY_SAMPLES:
                end_index = index
                break
            else:
                stable_samples = 0
        else:
            if stable_samples >= MINIMUM_STATIONARY_SAMPLES:
                # Stationary until the end of the track: no walking to measure afterwards.
                end_index = targets.size - 1

        if stable_samples < MINIMUM_STATIONARY_SAMPLES:
            return 0.0, None
        return float(stable_samples * step) / float(fps), int(targets[end_index])

    @staticmethod
    def hesitation_before_entry(frames: np.ndarray, position_m: np.ndarray, entry_frame: int, fps: float,
                                stature_m: float, checks_per_second: float) -> Optional[float]:
        """
        Return the stationary interval ending when the pedestrian enters the road strip.

        Walks backwards from the entry frame (crowd-city road_metrics.hesitation_seconds).
        None when the track starts less than one sampling interval before the entry.
        """
        step = max(1, int(round(fps / checks_per_second)))
        entry_frame = int(entry_frame)
        if entry_frame - int(frames[0]) < step:
            return None
        targets = np.arange(entry_frame, int(frames[0]) - 1, -step)
        position = np.interp(targets, frames, position_m)

        margin = STATIONARY_MARGIN_FRACTION * stature_m
        stable_samples = 0
        for index in range(targets.size - 1):
            if abs(float(position[index]) - float(position[index + 1])) > margin:
                break
            stable_samples += 1

        if stable_samples < MINIMUM_STATIONARY_SAMPLES:
            return 0.0
        return float(stable_samples * step) / float(fps)

    @staticmethod
    def speed_of_crossing(frames: np.ndarray, position_m: np.ndarray, fps: float,
                          start_frame: Optional[int] = None) -> Optional[float]:
        """
        Return the walking speed in m/s as the Theil-Sen slope of lateral position over time.

        Args:
            frames (np.ndarray): Frame numbers of the track.
            position_m (np.ndarray): Lateral position in metres per frame.
            fps (float): Frame rate.
            start_frame (int, optional): Ignore frames before this one (the hesitation interval).
        """
        mask = np.isfinite(position_m)
        if start_frame is not None:
            mask &= frames >= start_frame
        t = frames[mask] / fps
        if t.size < 2 or (t.max() - t.min()) < MINIMUM_SPEED_SECONDS:
            return None
        slope = theil_sen_slope(t, position_m[mask])
        return abs(slope) if slope is not None and math.isfinite(slope) else None

    def crossing_metrics(self, df: pl.DataFrame, track_id: Any, bounds: Tuple[int, int], fps: float,
                         aspect: float, city: str, min_x: float, max_x: float) -> Optional[Dict[str, Any]]:
        """
        Compute every metric of one detected crossing.

        Args:
            df (pl.DataFrame): Tracking CSV of the video.
            track_id: Tracker id of the pedestrian.
            bounds (tuple): (start_frame, end_frame) of the crossing segment.
            fps (float): Frame rate.
            aspect (float): Frame width / height.
            city (str): City name, for the assumed stature.
            min_x (float), max_x (float): Road strip.

        Returns:
            dict or None: Metrics of the crossing, None if the track is too short.
        """
        track = self.crossing_track(df, track_id, bounds)
        if track.height < 2:
            return None

        stature_m = self.person_height_m(city)
        checks = float(common.get_configs("check_per_sec_time"))
        frames = track.get_column("frame-count").cast(pl.Int64).to_numpy()
        x = track.get_column("x-center").cast(pl.Float64).to_numpy()
        position_m = self.lateral_position_m(track, fps, aspect, stature_m)

        walk_start_frame = None
        if common.get_configs("hesitation_reference") == "road_entry":
            on_road = np.nonzero((x >= min_x) & (x <= max_x))[0]
            entry_frame = int(frames[on_road[0]]) if on_road.size else int(frames[0])
            hesitation = self.hesitation_before_entry(frames, position_m, entry_frame, fps, stature_m, checks)
            if hesitation:
                walk_start_frame = entry_frame
        else:
            hesitation, walk_start_frame = self.time_to_start_cross(frames, position_m, fps, stature_m, checks)

        speed = self.speed_of_crossing(frames, position_m, fps, start_frame=walk_start_frame)

        return {
            "track_id": track_id,
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "n_frames": int(track.height),
            "direction": "left_to_right" if x[-1] > x[0] else "right_to_left",
            "crossing_time_s": self.time_to_cross(track, fps),
            "speed_mps": speed,
            "hesitation_s": hesitation,
            "hesitated": bool(hesitation) if hesitation is not None else None,
            "median_height_norm": float(np.median(track.get_column("height").to_numpy())),
            "person_height_m": stature_m,
        }

    def city_summary(self, crossings: pl.DataFrame, videos: pl.DataFrame) -> pl.DataFrame:
        """
        Aggregate crossings per city.

        Args:
            crossings (pl.DataFrame): One row per crossing (analysis.py output).
            videos (pl.DataFrame): One row per analysed video with city and duration_s.

        Returns:
            pl.DataFrame: One row per city.
        """
        footage = videos.group_by("city").agg(
            pl.len().alias("videos"),
            (pl.col("duration_s").sum() / 3600.0).alias("footage_h"),
        )
        # No early return for zero crossings: the per-city columns must exist
        # (as nulls) so the plots and the printed summary can rely on them.

        speed = pl.col("speed_mps")
        hes = pl.col("hesitation_s")
        # As in crowd-city, a pedestrian who never stood still has no hesitation
        # time rather than one of zero: zeros stay in crossings.csv but are left
        # out of the hesitation statistics, which would otherwise be pulled
        # towards zero by every crossing without a wait. share_hesitated says how
        # many waited, and hesitation_mean_all_s keeps the mean with the zeros.
        waited = hes.filter(hes > 0)
        per_city = crossings.group_by("city").agg(
            pl.len().alias("crossings"),
            speed.drop_nulls().len().alias("crossings_with_speed"),
            speed.mean().alias("speed_mean_mps"),
            speed.median().alias("speed_median_mps"),
            speed.std().alias("speed_std_mps"),
            hes.drop_nulls().len().alias("crossings_hesitation_observed"),
            waited.len().alias("crossings_hesitated"),
            waited.mean().alias("hesitation_mean_s"),
            waited.median().alias("hesitation_median_s"),
            waited.std().alias("hesitation_std_s"),
            pl.col("hesitated").mean().alias("share_hesitated"),
            hes.mean().alias("hesitation_mean_all_s"),
            pl.col("crossing_time_s").mean().alias("crossing_time_mean_s"),
        )
        summary = footage.join(per_city, on="city", how="left").with_columns(
            pl.col("crossings").fill_null(0),
        ).with_columns(
            (pl.col("crossings") / pl.col("footage_h")).alias("crossings_per_hour"),
        )
        return summary.sort("city")
