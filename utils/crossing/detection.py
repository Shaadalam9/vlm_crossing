"""Road-crossing detection from YOLO + BoT-SORT bounding boxes.

Adapted from utils/crossing/detection.py in crowd-dataset/crowd-city. A person
track counts as crossing when it moves from one side of the image, through the
central strip [boundary_left, boundary_right] (the road ahead of the dashcam),
to the other side, and survives the same geometric false-positive filters as
CROWD. Frame thresholds are calibrated at 30 fps and scaled to the video fps.

Differences from crowd-city:
- fps comes from an argument instead of mapping.csv.
- Detections with unique-id -1 (YOLO boxes the tracker did not assign) are
  ignored, because they would otherwise be pooled as one object.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

PERSON_CLASS_ID = 0
BICYCLE_CLASS_ID = 1
CAR_CLASS_ID = 2
MOTORCYCLE_CLASS_ID = 3
BUS_CLASS_ID = 5
TRUCK_CLASS_ID = 7
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
    # Reject candidates whose mean lateral speed (normalised x per frame) exceeds this; None disables it.
    "max_crossing_speed_per_frame": None,
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
    def _longest_frame_run(frames, *, gap_allow: int = 2) -> int:
        """Return the longest near-continuous run of frame numbers."""
        try:
            values = sorted({int(f) for f in frames})
        except Exception:
            return 0

        if not values:
            return 0

        max_run = 1
        cur_run = 1
        max_gap = max(int(gap_allow), 0) + 1
        prev = values[0]

        for frame in values[1:]:
            if int(frame) - int(prev) <= max_gap:
                cur_run += 1
            else:
                max_run = max(max_run, cur_run)
                cur_run = 1
            prev = frame

        return max(max_run, cur_run)

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
        rider_min_continuous_shared_frames_s = Detection._scale_frames(12, fps_value, base_fps_value)
        rider_shared_run_gap_allow_s = Detection._scale_frames(2, fps_value, base_fps_value, minimum=0)
        rider_min_motion_steps_s = Detection._scale_frames(3, fps_value, base_fps_value)
        rider_short_shared_frames_s = Detection._scale_frames(8, fps_value, base_fps_value)

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

        # Sort once by frame. Each candidate window is then located with two
        # binary searches and extracted with slice(), avoiding a full DataFrame
        # filter for every candidate.
        frame_sorted_df = dataframe.filter(pl.col("frame-count").is_not_null()).sort("frame-count")
        frame_values = frame_sorted_df.get_column("frame-count").cast(pl.Int64, strict=False).to_numpy()

        for c in candidate_segments:
            uid = c["uid"]
            if uid in pedestrian_bounds:
                continue
            x_range, x_speed, road_frames = c["x_range"], c["x_speed"], c["road_frames"]
            median_height, median_width, y_gross_motion = c["median_height"], c["median_width"], c["y_gross_motion"]

            left_idx = int(np.searchsorted(frame_values, int(c["start_frame"]), side="left"))
            right_idx = int(np.searchsorted(frame_values, int(c["end_frame"]), side="right"))
            segment_df = frame_sorted_df.slice(left_idx, max(0, right_idx - left_idx))

            if Detection.is_rider_id(
                segment_df,
                uid,
                min_shared_frames=rider_min_shared_frames_s,
                min_continuous_shared_frames=rider_min_continuous_shared_frames_s,
                shared_run_gap_allow=rider_shared_run_gap_allow_s,
                min_motion_steps=rider_min_motion_steps_s,
                short_shared_frames=rider_short_shared_frames_s,
            ):
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

            max_speed = p["max_crossing_speed_per_frame"]
            if max_speed is not None and x_speed > float(max_speed):
                continue

            if not Detection.is_valid_crossing(segment_df, uid, MIN_SHARED_FRAMES=min_static_shared_frames_s):
                continue

            pedestrian_ids.append(uid)
            pedestrian_bounds[uid] = (int(c["start_frame"]), int(c["end_frame"]))

        return pedestrian_ids, crossed_ids, pedestrian_bounds

    @staticmethod
    def classify_rider_type(
        df: pl.DataFrame,
        person_id,
        *,
        min_shared_frames: int = 4,
        min_continuous_shared_frames: int = 12,
        shared_run_gap_allow: int = 2,
        min_vehicle_width_ratio: float = 0.50,
        min_vehicle_width_ratio_frames: float = 0.65,
        dist_rel_thresh: float = 0.8,
        prox_req: float = 0.7,
        alpha_x: float = 0.75,
        beta_y: float = 0.08,
        gamma_y: float = 1.4,
        coloc_req: float = 0.7,
        sim_thresh: float = 0.4,
        sim_req: float = 0.5,
        min_motion_steps: int = 3,
        motion_coloc_min: float = 0.5,
        short_shared_frames: int = 8,
        short_sim_req: float = 0.8,
        short_disp_req: float = 0.12,
        eps: float = 1e-9,
        include_large_vehicle_passengers: bool = False,
        pooled_min_coverage: float = 0.40,
        pooled_min_person_motion: float = 0.30,
        pooled_max_motion_mismatch: float = 0.50,
        pooled_max_offset_std: float = 0.25,
        pooled_min_seated_share: float = 0.35,
    ) -> dict:
        """
        Classify whether a person track is associated with a vehicle (crowd-city).

        For every two-wheeler track sharing a continuous run of frames with the person,
        the person is a rider when the vehicle stays close, sits below the torso and is
        at least half as wide in most shared frames, or when it moves with the person.
        When no single vehicle id qualifies, detections are pooled across ids
        (see _pooled_two_wheeler_association).
        """
        not_rider = {
            "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
            "score": 0.0, "shared_frames": 0, "longest_shared_run": 0,
        }
        df = Detection._dedup_per_frame(df)

        p = (
            df.filter((pl.col("yolo-id") == PERSON_CLASS_ID) & (pl.col("unique-id") == person_id))
            .sort("frame-count")
        )
        if p.height == 0:
            return not_rider

        p_frames = p.get_column("frame-count").to_numpy()
        if p_frames.size < min_shared_frames:
            return not_rider

        first_frame = int(p_frames.min())
        last_frame = int(p_frames.max())

        supported_vehicle_classes = [BICYCLE_CLASS_ID, MOTORCYCLE_CLASS_ID]
        if include_large_vehicle_passengers:
            supported_vehicle_classes.extend([CAR_CLASS_ID, BUS_CLASS_ID, TRUCK_CLASS_ID])

        vehicles = df.filter(
            pl.col("frame-count").is_between(first_frame, last_frame)
            & pl.col("yolo-id").is_in(supported_vehicle_classes)
            & (pl.col("unique-id") >= 0)
        )
        if vehicles.height == 0:
            return not_rider

        p1 = p.unique(subset=["frame-count"], keep="first")
        best = None

        for v in vehicles.partition_by("unique-id", maintain_order=True):
            v = v.sort("frame-count")
            vid = v.get_column("unique-id")[0]
            v_class = int(v.get_column("yolo-id")[0])
            vtype = {
                BICYCLE_CLASS_ID: "bicycle",
                MOTORCYCLE_CLASS_ID: "motorcycle",
                CAR_CLASS_ID: "car",
                BUS_CLASS_ID: "bus",
                TRUCK_CLASS_ID: "truck",
            }.get(v_class)
            if vtype is None:
                continue

            role = "rider" if v_class in (BICYCLE_CLASS_ID, MOTORCYCLE_CLASS_ID) else "passenger"

            v1 = v.unique(subset=["frame-count"], keep="first")
            j = p1.join(v1, on="frame-count", how="inner", suffix="_v")
            shared = j.height
            if shared < min_shared_frames:
                continue

            longest_shared_run = Detection._longest_frame_run(
                j.get_column("frame-count").to_list(),
                gap_allow=shared_run_gap_allow,
            )
            if role == "rider" and longest_shared_run < int(min_continuous_shared_frames):
                continue

            p_xy = j.select(["x-center", "y-center"]).to_numpy()
            v_xy = j.select(["x-center_v", "y-center_v"]).to_numpy()

            p_w = j.get_column("width").to_numpy()
            p_h = j.get_column("height").to_numpy()
            v_w = j.get_column("width_v").to_numpy()
            v_h = j.get_column("height_v").to_numpy()

            if role == "rider":
                vehicle_width_ratio_arr = v_w / np.maximum(p_w, eps)
                vehicle_width_ratio = float(np.median(vehicle_width_ratio_arr))
                vehicle_width_ratio_pass_ratio = float(
                    (vehicle_width_ratio_arr >= float(min_vehicle_width_ratio)).mean()
                )
                if vehicle_width_ratio_pass_ratio < float(min_vehicle_width_ratio_frames):
                    continue
            else:
                vehicle_width_ratio = 0.0
                vehicle_width_ratio_pass_ratio = 0.0

            dist = np.linalg.norm(p_xy - v_xy, axis=1)
            if role == "rider":
                dist_rel = dist / np.maximum(p_h, eps)
            else:
                dist_rel = dist / np.maximum(v_h, eps)

            prox = dist_rel < dist_rel_thresh
            prox_ratio = float(prox.mean())
            if prox_ratio < prox_req:
                continue

            relx = v_xy[:, 0] - p_xy[:, 0]
            rely = v_xy[:, 1] - p_xy[:, 1]

            if role == "rider":
                spatial = (np.abs(relx) < alpha_x * p_w) & (rely > beta_y * p_h) & (rely < gamma_y * p_h)
            else:
                spatial = (np.abs(relx) <= 0.5 * v_w) & (np.abs(rely) <= 0.5 * v_h)

            coloc = prox & spatial
            coloc_ratio = float(coloc.mean())

            p_mov = np.diff(p_xy, axis=0)
            v_mov = np.diff(v_xy, axis=0)

            sim_ratio = 0.0
            if p_mov.shape[0] > 0:
                na = np.linalg.norm(p_mov, axis=1)
                nb = np.linalg.norm(v_mov, axis=1)
                move_mask = (na > eps) & (nb > eps)

                cos = np.zeros_like(na, dtype=float)
                cos[move_mask] = (p_mov[move_mask] * v_mov[move_mask]).sum(axis=1) / (na[move_mask] * nb[move_mask])

                prox_steps = prox[1:]
                m = min(len(prox_steps), len(cos), len(move_mask))
                prox_steps = prox_steps[:m]
                cos = cos[:m]
                move_mask = move_mask[:m]

                denom_mask = prox_steps & move_mask
                denom = int(denom_mask.sum())
                if denom >= min_motion_steps:
                    sim_ratio = float(((cos > sim_thresh) & denom_mask).sum() / denom)

            if shared < short_shared_frames:
                if shared > 1:
                    p_disp = float(np.linalg.norm(p_xy[-1] - p_xy[0]))
                    p_disp_rel = p_disp / float(np.maximum(np.mean(p_h), eps))
                else:
                    p_disp_rel = 0.0

                if not (sim_ratio >= short_sim_req or p_disp_rel >= short_disp_req):
                    continue

            ok = (coloc_ratio >= coloc_req) or (sim_ratio >= sim_req and coloc_ratio >= motion_coloc_min)
            if not ok:
                continue

            score = 0.7 * coloc_ratio + 0.2 * prox_ratio + 0.1 * float(sim_ratio)
            cand = {
                "is_rider": True,
                "rider_type": vtype,
                "role": role,
                "vehicle_id": vid,
                "score": float(score),
                "shared_frames": int(shared),
                "longest_shared_run": int(longest_shared_run),
                "vehicle_width_ratio": float(vehicle_width_ratio),
                "vehicle_width_ratio_pass_ratio": float(vehicle_width_ratio_pass_ratio),
                "prox_ratio": prox_ratio,
                "coloc_ratio": coloc_ratio,
                "sim_ratio": float(sim_ratio),
            }

            if best is None or cand["score"] > best["score"]:
                best = cand

        if best is None:
            # The per-id test above needs one two-wheeler track to stay under
            # the person for a continuous run. In practice the rider's own body
            # hides the vehicle, so YOLO detects it only intermittently and the
            # tracker splits it across several ids; no single id then survives
            # the run requirement even though every detection that exists sits
            # exactly where a ridden vehicle would. Pool them instead.
            best = Detection._pooled_two_wheeler_association(
                p1,
                vehicles.filter(pl.col("yolo-id").is_in([BICYCLE_CLASS_ID, MOTORCYCLE_CLASS_ID])),
                min_shared_frames=min_shared_frames,
                min_continuous_shared_frames=min_continuous_shared_frames,
                min_vehicle_width_ratio=min_vehicle_width_ratio,
                dist_rel_thresh=dist_rel_thresh,
                alpha_x=alpha_x,
                beta_y=beta_y,
                gamma_y=gamma_y,
                min_coverage=pooled_min_coverage,
                min_person_motion=pooled_min_person_motion,
                max_motion_mismatch=pooled_max_motion_mismatch,
                max_offset_std=pooled_max_offset_std,
                min_seated_share=pooled_min_seated_share,
                eps=eps,
            )

        return best if best is not None else not_rider

    @staticmethod
    def _pooled_two_wheeler_association(
        person: pl.DataFrame,
        two_wheelers: pl.DataFrame,
        *,
        min_shared_frames: int,
        min_continuous_shared_frames: int,
        min_vehicle_width_ratio: float,
        dist_rel_thresh: float,
        alpha_x: float,
        beta_y: float,
        gamma_y: float,
        min_coverage: float,
        min_person_motion: float,
        max_motion_mismatch: float,
        max_offset_std: float,
        min_seated_share: float,
        eps: float,
    ) -> Optional[dict]:
        """Detect a rider from two-wheeler detections pooled across tracker ids.

        ``person`` holds one row per frame. In every frame the two-wheeler
        nearest the person's centre is kept if it sits in the rider position
        (below the torso, horizontally overlapping, at least half as wide);
        the frames that qualify are then judged together:

        - they must number at least ``min_shared_frames`` and span at least
          ``min_coverage`` of the person's track, so a few coincidental
          frames at one end of a crossing are not enough;
        - if the person moved noticeably over that span, the vehicle must have
          moved with them, which separates a ridden vehicle from a pedestrian
          walking past a parked one;
        - if the person barely moved, co-movement cannot be judged, so the
          vehicle must instead be present for ``min_continuous_shared_frames``
          frames in total;
        - the vehicle's horizontal offset from the person must stay steady
          (standard deviation at most ``max_offset_std`` person widths), and
          of the frames with any two-wheeler near the person, at least
          ``min_seated_share`` must have it in the rider position. A cyclist
          overtaking a pedestrian passes through the rider position briefly
          while sweeping across the pedestrian's box; a ridden vehicle stays
          put beneath its rider.
        """
        if person.height == 0 or two_wheelers.height == 0:
            return None

        joined = person.select(
            ["frame-count", "x-center", "y-center", "width", "height"]
        ).join(
            two_wheelers.select(
                ["frame-count", "unique-id", "yolo-id", "x-center", "y-center", "width", "height"]
            ),
            on="frame-count",
            how="inner",
            suffix="_v",
        )
        if joined.height == 0:
            return None

        joined = joined.with_columns(
            (pl.col("x-center_v") - pl.col("x-center")).alias("_relx"),
            (pl.col("y-center_v") - pl.col("y-center")).alias("_rely"),
        ).with_columns(
            (
                (pl.col("_relx") ** 2 + pl.col("_rely") ** 2).sqrt()
                / pl.max_horizontal(pl.col("height"), pl.lit(eps))
            ).alias("_dist_rel"),
        )
        seated = joined.filter(
            (pl.col("_dist_rel") < dist_rel_thresh)
            & (pl.col("_relx").abs() < alpha_x * pl.col("width"))
            & (pl.col("_rely") > beta_y * pl.col("height"))
            & (pl.col("_rely") < gamma_y * pl.col("height"))
            & (pl.col("width_v") >= min_vehicle_width_ratio * pl.col("width"))
        )
        if seated.height == 0:
            return None

        near_frames = joined.filter(
            pl.col("_dist_rel") < 1.25 * dist_rel_thresh
        ).get_column("frame-count").n_unique()

        seated = (
            seated.sort(["frame-count", "_dist_rel"])
            .unique(subset=["frame-count"], keep="first")
            .sort("frame-count")
        )
        shared = seated.height
        if shared < int(min_shared_frames):
            return None

        seated_share = float(shared) / float(max(near_frames, 1))
        if seated_share < float(min_seated_share):
            return None

        offset_std = float(
            (seated.get_column("_relx") / seated.get_column("width").clip(lower_bound=eps)).std() or 0.0
        )
        if offset_std > float(max_offset_std):
            return None

        person_frames = person.get_column("frame-count").to_numpy()
        person_span = float(person_frames.max() - person_frames.min() + 1)
        seated_frames = seated.get_column("frame-count").to_numpy()
        coverage = float(seated_frames.max() - seated_frames.min() + 1) / max(person_span, 1.0)
        if coverage < float(min_coverage):
            return None

        first = seated.row(0, named=True)
        last = seated.row(-1, named=True)
        person_disp = np.array(
            [last["x-center"] - first["x-center"], last["y-center"] - first["y-center"]]
        )
        vehicle_disp = np.array(
            [last["x-center_v"] - first["x-center_v"], last["y-center_v"] - first["y-center_v"]]
        )
        median_height = float(max(seated.get_column("height").median() or 0.0, eps))
        person_motion = float(np.linalg.norm(person_disp)) / median_height

        if person_motion >= float(min_person_motion):
            mismatch = float(np.linalg.norm(vehicle_disp - person_disp)) / max(
                float(np.linalg.norm(person_disp)), eps
            )
            if mismatch > float(max_motion_mismatch):
                return None
        else:
            mismatch = float("nan")
            if shared < int(min_continuous_shared_frames):
                return None

        vehicle_classes = seated.get_column("yolo-id").to_list()
        majority_class = max(set(vehicle_classes), key=vehicle_classes.count)
        vehicle_ids = seated.get_column("unique-id").unique().to_list()
        return {
            "is_rider": True,
            "rider_type": "bicycle" if int(majority_class) == BICYCLE_CLASS_ID else "motorcycle",
            "role": "rider",
            "vehicle_id": vehicle_ids[0] if len(vehicle_ids) == 1 else vehicle_ids,
            "score": float(min(1.0, coverage)),
            "shared_frames": int(shared),
            "longest_shared_run": int(Detection._longest_frame_run(seated_frames.tolist(), gap_allow=0)),
            "pooled": True,
            "pooled_vehicle_ids": len(vehicle_ids),
            "coverage": coverage,
            "person_motion": person_motion,
            "motion_mismatch": mismatch,
            "seated_share": seated_share,
            "offset_std": offset_std,
        }

    @staticmethod
    def is_rider_id(df: pl.DataFrame, person_id, **kwargs) -> bool:
        """Return True when the person rides a bicycle or motorcycle. kwargs go to classify_rider_type."""
        return bool(Detection.classify_rider_type(df, person_id, **kwargs).get("is_rider"))

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
    def _robust_range(values, q: float = 0.05, method: str = "linear") -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        return max(0.0, float(np.quantile(arr, 1.0 - q, method=method)) - float(np.quantile(arr, q, method=method)))

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
        As in crowd-city, ranges here use nearest-rank quantiles.
        """
        best = None
        for _, joined in Detection._static_reference_candidates(df, person_id, STATIC_CLASS_IDS, MIN_SHARED_FRAMES):
            px = joined.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
            sx = joined.get_column("x-center_ref").cast(pl.Float64, strict=False).to_numpy()
            px_rng = Detection._robust_range(px, Q, method="nearest")
            sx_rng = Detection._robust_range(sx, Q, method="nearest")
            cand = {
                "shared": int(joined.height),
                "sx_rng": sx_rng,
                "relx_rng": Detection._robust_range(px - sx, Q, method="nearest"),
                "ratio": sx_rng / max(px_rng, float(EPS)),
            }
            if best is None or (cand["shared"], cand["sx_rng"]) > (best["shared"], best["sx_rng"]):
                best = cand

        if best is None:
            return True

        if best["relx_rng"] < RELX_MIN:
            return False

        if best["ratio"] >= float(ratio_thresh) and best["relx_rng"] < (2.0 * RELX_MIN):
            return False

        return True
