"""Road-crossing detection from YOLO + BoT-SORT bounding boxes.

Adapted from utils/crossing/detection.py in crowd-dataset/crowd-city. A person
track counts as crossing when it moves from one side of the image, through the
central strip [boundary_left, boundary_right] (the road ahead of the dashcam),
to the other side, and survives the same geometric false-positive filters as
CROWD. Frame thresholds are calibrated at 30 fps and scaled to the video fps.

Differences from crowd-city:
- fps and the city's mean stature come from arguments instead of mapping.csv.
- The rider filter is a simpler co-location test against bicycles and
  motorcycles instead of the pooled rider classifier.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

PERSON_CLASS_ID = 0
TWO_WHEELER_CLASS_IDS = (1, 3)  # bicycle, motorcycle
# Objects that do not move in the world: traffic light, fire hydrant, stop sign, parking meter, bench.
STATIC_CLASS_IDS = (9, 10, 11, 12, 13)

CROSSING_PARAMETER_DEFAULTS = {
    "tol": 0.00,
    "min_track_frames": 10,
    "min_road_frames": 3,
    "max_track_gap_frames": 30,
    "min_crossing_x_range": 0.14,
    "weak_crossing_x_range": 0.64,
    "low_x_range": 0.30,
    "low_x_min_road_frames": 20,
    "tiny_long_track_x_range": 0.36,
    "tiny_long_track_height": 0.12,
    "tiny_long_track_road_frames": 50,
    "slender_track_width": 0.05,
    "slender_track_height": 0.26,
    "slender_track_min_road_frames": 5,
    "slender_track_max_road_frames": 49,
    "large_lateral_x_range": 0.56,
    "large_lateral_tiny_height": 0.105,
    "camera_static_sx": 0.25,
    "camera_static_ratio": 0.60,
    "camera_static_relx": 0.18,
    "camera_static_height": 0.15,
    "camera_static_tiny_relx": 0.12,
    "camera_static_tiny_relx_height": 0.19,
    "weak_y_jitter_x_range": 0.50,
    "weak_y_jitter_motion": 0.30,
    "weak_y_jitter_height": 0.22,
    "no_static_slender_height": 0.24,
    "no_static_slender_max_road_frames": 20,
    "tiny_no_static_height": 0.12,
    "tiny_no_static_width": 0.026,
    "tiny_no_static_min_road_frames": 10,
    "no_static_tiny_min_road_frames": 5,
    "no_static_tiny_fast_speed": 0.006,
    "slender_static_relx_min": 0.13,
    "camera_tiny_height": 0.15,
    "min_static_shared_frames": 8,
    "long_weak_road_frames": 90,
    "jitter_road_frames": 40,
    "camera_min_road_frames": 5,
    "base_fps": 30.0,
}


class Detection:

    def __init__(self) -> None:
        pass

    @staticmethod
    def crossing_parameter_defaults() -> dict:
        """Return an independent copy of the fixed original CROWD defaults."""
        return dict(CROSSING_PARAMETER_DEFAULTS)

    @staticmethod
    def _scale_frames(value: int, fps: float, base_fps: float = 30.0, minimum: int = 1) -> int:
        """Scale a 30-fps calibrated frame threshold to the current FPS.

        Example: 10 frames at 30 fps becomes 20 frames at 60 fps and 5 frames at 15 fps.
        """
        try:
            scaled = int(round(float(value) * float(fps) / float(base_fps)))
        except Exception:
            scaled = int(value)
        return max(int(minimum), int(scaled))

    @staticmethod
    def _dedup_per_frame(df: pl.DataFrame) -> pl.DataFrame:
        """Keep highest-confidence detection per (yolo-id, unique-id, frame-count)."""
        if "confidence" not in df.columns:
            return df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        return (
            df.sort(
                ["yolo-id", "unique-id", "frame-count", "confidence"],
                descending=[False, False, False, True],
            )
            .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
        )

    def pedestrian_crossing(self, dataframe: pl.DataFrame, fps: float, min_x: float, max_x: float,
                            params: Optional[Dict[str, float]] = None,
                            ) -> Tuple[List[Any], List[Any], Dict[Any, Tuple[int, int]]]:
        """
        Identify pedestrian tracks that cross the road and filter false positives.

        - Splits reused tracker IDs into temporal segments when frame gaps are large.
        - A segment is a candidate when it goes left -> road strip -> right, or the reverse.
        - Candidates are then rejected by rider association, weak geometry and camera motion.

        Args:
            dataframe (pl.DataFrame): Tracking CSV of one video.
            fps (float): Video frame rate.
            min_x (float): Left edge of the road strip, normalised.
            max_x (float): Right edge of the road strip, normalised.
            params (dict, optional): Overrides for CROSSING_PARAMETER_DEFAULTS.

        Returns:
            tuple: (pedestrian_ids, crossed_ids, pedestrian_bounds). pedestrian_ids passed every
            filter; crossed_ids are all geometric candidates; pedestrian_bounds maps each accepted
            id to the (start_frame, end_frame) of the segment that qualified, because a tracker
            id can be reused later in the video by an unrelated object.
        """
        p = self.crossing_parameter_defaults()
        p.update(params or {})

        fps_value = float(fps) if fps and fps > 0 else float(p["base_fps"])
        base_fps_value = max(float(p["base_fps"]), 1e-9)

        def scaled(key: str, minimum: int = 1) -> int:
            return Detection._scale_frames(p[key], fps_value, base_fps_value, minimum=minimum)

        min_track_frames_s = scaled("min_track_frames")
        min_road_frames_s = scaled("min_road_frames")
        max_track_gap_frames_s = scaled("max_track_gap_frames", minimum=0)
        low_x_min_road_frames_s = scaled("low_x_min_road_frames")
        tiny_long_track_road_frames_s = scaled("tiny_long_track_road_frames")
        slender_track_min_road_frames_s = scaled("slender_track_min_road_frames")
        slender_track_max_road_frames_s = scaled("slender_track_max_road_frames")
        no_static_slender_max_road_frames_s = scaled("no_static_slender_max_road_frames")
        tiny_no_static_min_road_frames_s = scaled("tiny_no_static_min_road_frames")
        no_static_tiny_min_road_frames_s = scaled("no_static_tiny_min_road_frames")
        min_static_shared_frames_s = scaled("min_static_shared_frames")
        long_weak_road_frames_s = scaled("long_weak_road_frames")
        jitter_road_frames_s = scaled("jitter_road_frames")
        camera_min_road_frames_s = scaled("camera_min_road_frames")
        rider_min_shared_frames_s = Detection._scale_frames(4, fps_value, base_fps_value)

        persons = dataframe.filter((pl.col("yolo-id") == PERSON_CLASS_ID) & (pl.col("unique-id") >= 0))
        if persons.height == 0:
            return [], [], {}

        tracks = (
            Detection._dedup_per_frame(persons)
            .select(["unique-id", "frame-count", "x-center", "y-center", "width", "height"])
            .sort(["unique-id", "frame-count"])
        )
        track_partitions = tracks.partition_by("unique-id", maintain_order=True)

        tol = float(p["tol"])
        left_hard = float(min_x) - tol
        left_soft = float(min_x) + tol
        right_soft = float(max_x) - tol
        right_hard = float(max_x) + tol

        def split_segments(track: pl.DataFrame) -> List[pl.DataFrame]:
            """Split one tracker id into near-continuous temporal segments."""
            frames = track.get_column("frame-count").cast(pl.Int64, strict=False).to_list()
            if not frames:
                return []

            segments: List[pl.DataFrame] = []
            start_idx = 0
            prev_frame = int(frames[0])
            max_gap = max(int(max_track_gap_frames_s), 0)

            for idx in range(1, len(frames)):
                frame = int(frames[idx])
                if frame - prev_frame > max_gap:
                    segments.append(track.slice(start_idx, idx - start_idx))
                    start_idx = idx
                prev_frame = frame

            segments.append(track.slice(start_idx, len(frames) - start_idx))
            return segments

        def build_states(x: np.ndarray) -> np.ndarray:
            """0 = left of the strip, 1 = on the road strip, 2 = right of it, with hysteresis."""
            states = np.empty(x.size, dtype=np.int8)
            if x.size == 0:
                return states

            x0 = float(x[0])
            if x0 < float(min_x):
                s = 0
            elif x0 > float(max_x):
                s = 2
            else:
                s = 1
            states[0] = s

            for i in range(1, x.size):
                xi = float(x[i])
                if xi <= left_hard:
                    s = 0
                elif xi >= right_hard:
                    s = 2
                elif left_soft <= xi <= right_soft:
                    s = 1
                states[i] = s

            return states

        def segment_is_candidate(states: np.ndarray) -> bool:
            is_left = states == 0
            is_road = states == 1
            is_right = states == 2

            if int(is_road.sum()) < int(min_road_frames_s):
                return False

            left_before = np.maximum.accumulate(is_left)
            right_before = np.maximum.accumulate(is_right)
            left_after = np.maximum.accumulate(is_left[::-1])[::-1]
            right_after = np.maximum.accumulate(is_right[::-1])[::-1]

            crossing_mask = is_road & ((left_before & right_after) | (right_before & left_after))
            return bool(crossing_mask.any())

        candidate_segments = []
        crossed_ids: List[Any] = []

        for tr in track_partitions:
            if tr.height < int(min_track_frames_s):
                continue
            uid = tr.get_column("unique-id")[0]

            for seg in split_segments(tr):
                if seg.height < int(min_track_frames_s):
                    continue
                x = seg.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
                states = build_states(x)
                if not segment_is_candidate(states):
                    continue

                frames = seg.get_column("frame-count").cast(pl.Int64, strict=False).to_numpy()
                start_frame = int(frames.min())
                end_frame = int(frames.max())
                duration = max(1, end_frame - start_frame + 1)
                x_range = float(np.nanmax(x) - np.nanmin(x))
                y = seg.get_column("y-center").cast(pl.Float64, strict=False).to_numpy()

                candidate_segments.append({
                    "uid": uid,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "x_range": x_range,
                    "x_speed": x_range / duration,
                    "road_frames": int((states == 1).sum()),
                    "median_height": float(np.nanmedian(seg.get_column("height").to_numpy())),
                    "median_width": float(np.nanmedian(seg.get_column("width").to_numpy())),
                    "y_gross_motion": float(np.nansum(np.abs(np.diff(y)))) if y.size > 1 else 0.0,
                })
                if uid not in crossed_ids:
                    crossed_ids.append(uid)

        pedestrian_ids: List[Any] = []
        pedestrian_bounds: Dict[Any, Tuple[int, int]] = {}

        for c in candidate_segments:
            uid = c["uid"]
            if uid in pedestrian_bounds:
                continue
            x_range, x_speed, road_frames = c["x_range"], c["x_speed"], c["road_frames"]
            median_height, median_width, y_gross_motion = c["median_height"], c["median_width"], c["y_gross_motion"]

            segment_df = dataframe.filter(pl.col("frame-count").is_between(c["start_frame"], c["end_frame"]))

            if Detection.is_rider_id(segment_df, uid, min_shared_frames=rider_min_shared_frames_s):
                continue

            static_stats = Detection.static_reference_motion_stats(
                segment_df, uid, MIN_SHARED_FRAMES=min_static_shared_frames_s,
            )
            static_shared = int(static_stats["shared_frames"])
            static_sx_range = float(static_stats["static_x_range"])
            static_relx_range = float(static_stats["relative_x_range"])
            static_ratio = float(static_stats["static_to_person_ratio"])

            if x_range < p["min_crossing_x_range"]:
                continue

            if x_range < p["low_x_range"] and road_frames < low_x_min_road_frames_s:
                continue

            if x_range < p["weak_crossing_x_range"] and road_frames > long_weak_road_frames_s:
                continue

            if x_range < 0.56 and road_frames > jitter_road_frames_s and y_gross_motion > 0.30:
                continue

            if (
                x_range < p["weak_y_jitter_x_range"]
                and y_gross_motion > p["weak_y_jitter_motion"]
                and median_height < p["weak_y_jitter_height"]
            ):
                continue

            if (
                x_range < p["tiny_long_track_x_range"]
                and median_height < p["tiny_long_track_height"]
                and road_frames >= tiny_long_track_road_frames_s
            ):
                continue

            if (
                static_shared < min_static_shared_frames_s
                and median_height <= p["tiny_no_static_height"]
                and median_width <= p["tiny_no_static_width"]
                and (
                    road_frames >= tiny_no_static_min_road_frames_s
                    or road_frames >= no_static_tiny_min_road_frames_s
                    or x_speed >= p["no_static_tiny_fast_speed"]
                )
            ):
                continue

            if static_sx_range >= p["camera_static_sx"] and static_ratio >= p["camera_static_ratio"]:
                if median_height <= p["camera_tiny_height"] and road_frames >= camera_min_road_frames_s:
                    continue
                if (
                    static_relx_range <= p["camera_static_tiny_relx"]
                    and median_height <= p["camera_static_tiny_relx_height"]
                ):
                    continue
                if static_relx_range <= p["camera_static_relx"] and median_height <= p["camera_static_height"]:
                    continue

            if (
                median_width <= p["slender_track_width"]
                and median_height < p["slender_track_height"]
                and slender_track_min_road_frames_s <= road_frames <= slender_track_max_road_frames_s
            ):
                if static_shared < min_static_shared_frames_s:
                    if (
                        median_height < p["no_static_slender_height"]
                        and road_frames <= no_static_slender_max_road_frames_s
                    ):
                        continue
                else:
                    if static_relx_range < p["slender_static_relx_min"]:
                        continue
                    if (
                        static_sx_range >= p["camera_static_sx"]
                        and static_ratio >= p["camera_static_ratio"]
                        and median_height <= p["camera_tiny_height"]
                    ):
                        continue

            if (
                x_range > p["large_lateral_x_range"]
                and median_height < p["large_lateral_tiny_height"]
                and road_frames >= camera_min_road_frames_s
                and static_sx_range >= p["camera_static_sx"]
                and static_ratio >= p["camera_static_ratio"]
                and static_relx_range < 0.20
            ):
                continue

            if not Detection.is_valid_crossing(segment_df, uid, MIN_SHARED_FRAMES=min_static_shared_frames_s):
                continue

            pedestrian_ids.append(uid)
            pedestrian_bounds[uid] = (int(c["start_frame"]), int(c["end_frame"]))

        return pedestrian_ids, crossed_ids, pedestrian_bounds

    @staticmethod
    def is_rider_id(df: pl.DataFrame, person_id, min_shared_frames: int = 4, coloc_req: float = 0.7,
                    alpha_x: float = 0.75, beta_y: float = 0.25) -> bool:
        """
        Return True when the person rides (or pushes) a bicycle or motorcycle.

        A person is a rider when, for at least coloc_req of the frames shared with some
        two-wheeler track, the person's box is horizontally centred on the vehicle
        (|dx| <= alpha_x * max width) and the person's feet fall inside the vehicle's
        vertical extent (down to beta_y person-heights below it).
        """
        person = (
            df.filter((pl.col("yolo-id") == PERSON_CLASS_ID) & (pl.col("unique-id") == person_id))
            .unique(subset=["frame-count"], keep="first")
        )
        vehicles = df.filter(pl.col("yolo-id").is_in(list(TWO_WHEELER_CLASS_IDS)) & (pl.col("unique-id") >= 0))
        if person.height == 0 or vehicles.height == 0:
            return False

        for v_track in vehicles.partition_by("unique-id", maintain_order=True):
            v_track = v_track.unique(subset=["frame-count"], keep="first")
            joined = person.join(v_track, on="frame-count", how="inner", suffix="_v")
            if joined.height < int(min_shared_frames):
                continue
            px = joined.get_column("x-center").to_numpy()
            pw = joined.get_column("width").to_numpy()
            p_bottom = joined.get_column("y-center").to_numpy() + joined.get_column("height").to_numpy() / 2.0
            ph = joined.get_column("height").to_numpy()
            vx = joined.get_column("x-center_v").to_numpy()
            vw = joined.get_column("width_v").to_numpy()
            v_top = joined.get_column("y-center_v").to_numpy() - joined.get_column("height_v").to_numpy() / 2.0
            v_bottom = joined.get_column("y-center_v").to_numpy() + joined.get_column("height_v").to_numpy() / 2.0

            aligned_x = np.abs(px - vx) <= alpha_x * np.maximum(pw, vw)
            aligned_y = (p_bottom >= v_top) & (p_bottom <= v_bottom + beta_y * ph)
            if float(np.mean(aligned_x & aligned_y)) >= coloc_req:
                return True
        return False

    @staticmethod
    def _static_reference_candidates(df: pl.DataFrame, person_id, static_class_ids, min_shared_frames: int):
        """Yield the person track joined with each static-object track sharing enough frames."""
        df = Detection._dedup_per_frame(df)
        person = (
            df.filter((pl.col("yolo-id") == PERSON_CLASS_ID) & (pl.col("unique-id") == person_id))
            .sort("frame-count")
            .unique(subset=["frame-count"], keep="first")
        )
        if person.height == 0:
            return
        first_frame = int(person.get_column("frame-count").min())
        last_frame = int(person.get_column("frame-count").max())

        refs = df.filter(
            pl.col("frame-count").is_between(first_frame, last_frame)
            & pl.col("yolo-id").is_in(list(static_class_ids))
            & (pl.col("unique-id") >= 0)
        )
        if refs.height == 0:
            return

        for r in refs.partition_by("unique-id", maintain_order=True):
            r = r.sort("frame-count").unique(subset=["frame-count"], keep="first")
            joined = person.join(r, on="frame-count", how="inner", suffix="_ref").sort("frame-count")
            if joined.height >= int(min_shared_frames):
                yield r, joined

    @staticmethod
    def _robust_range(values, q: float = 0.05) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        return max(0.0, float(np.quantile(arr, 1.0 - q)) - float(np.quantile(arr, q)))

    @staticmethod
    def static_reference_motion_stats(df, person_id, STATIC_CLASS_IDS=STATIC_CLASS_IDS,
                                      MIN_SHARED_FRAMES=8, Q=0.05, EPS=1e-9):
        """Return motion statistics between a person track and the best static reference."""
        best = {
            "has_reference": False,
            "static_id": None,
            "shared_frames": 0,
            "person_x_range": 0.0,
            "static_x_range": 0.0,
            "relative_x_range": 0.0,
            "static_to_person_ratio": 0.0,
        }

        for r, joined in Detection._static_reference_candidates(df, person_id, STATIC_CLASS_IDS, MIN_SHARED_FRAMES):
            person_x = joined.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
            ref_x = joined.get_column("x-center_ref").cast(pl.Float64, strict=False).to_numpy()
            person_x_range = Detection._robust_range(person_x, Q)
            static_x_range = Detection._robust_range(ref_x, Q)
            cand = {
                "has_reference": True,
                "static_id": r.get_column("unique-id")[0],
                "shared_frames": int(joined.height),
                "person_x_range": person_x_range,
                "static_x_range": static_x_range,
                "relative_x_range": Detection._robust_range(person_x - ref_x, Q),
                "static_to_person_ratio": static_x_range / max(person_x_range, float(EPS)),
            }
            if (cand["shared_frames"], cand["static_x_range"]) > (best["shared_frames"], best["static_x_range"]):
                best = cand

        return best

    @staticmethod
    def is_valid_crossing(df, person_id, ratio_thresh=0.6, STATIC_CLASS_IDS=STATIC_CLASS_IDS,
                          MIN_SHARED_FRAMES=8, RELX_MIN=0.01, Q=0.05, EPS=1e-9):
        """Check whether an apparent crossing is independent of camera motion.

        When a static object moves across the image as much as the person does, the
        apparent crossing is caused by the dashcam turning, not by the pedestrian.
        """
        stats = Detection.static_reference_motion_stats(df, person_id, STATIC_CLASS_IDS, MIN_SHARED_FRAMES, Q, EPS)
        if not stats["has_reference"]:
            return True

        if stats["relative_x_range"] < RELX_MIN:
            return False

        if stats["static_to_person_ratio"] >= float(ratio_thresh) and stats["relative_x_range"] < (2.0 * RELX_MIN):
            return False

        return True
