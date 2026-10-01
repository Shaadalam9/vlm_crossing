import numpy as np
import polars as pl
from crowd_city.core.metadata import MetaData
from crowd_city.core.helper import Youtube_Helper
from typing import Tuple, List, Any, Optional, Dict

metadata = MetaData()
helper = Youtube_Helper()

CROSSING_PARAMETER_DEFAULTS = {
    "tol": 0.00,
    "min_track_frames": 10,
    "min_road_frames": 3,
    "max_track_gap_frames": 30,
    "min_crossing_x_range": 0.14,
    "weak_crossing_x_range": 0.64,
    "low_x_range": 0.30,
    "low_x_min_road_frames": 20,
    "weak_y_jitter_motion": 0.30,
    "long_weak_road_frames": 90,
}


class Detection:

    def __init__(self) -> None:
        pass

    @staticmethod
    def crossing_parameter_defaults() -> dict:
        """Return an independent copy of the fixed original CROWD defaults."""
        return dict(CROSSING_PARAMETER_DEFAULTS)

    @staticmethod
    def _first_float(value, default: Optional[float] = None) -> Optional[float]:
        """Return the first positive float found in a scalar, list-like value, or string."""
        if value is None:
            return default

        if isinstance(value, (list, tuple)):
            for item in value:
                found = Detection._first_float(item, None)
                if found is not None and found > 0:
                    return found
            return default

        try:
            found = float(value)
            if found > 0:
                return found
        except Exception:
            pass

        text = str(value)
        for ch in "[](){}'\"":
            text = text.replace(ch, " ")
        text = text.replace(",", " ")
        for token in text.split():
            try:
                found = float(token)
                if found > 0:
                    return found
            except Exception:
                continue
        return default

    @staticmethod
    def _fps_from_video_key(video_id: str, default: Optional[float] = None) -> Optional[float]:
        """Extract FPS from keys like videoId_startSeconds_fps, e.g. abc_31_30."""
        try:
            parts = str(video_id).rsplit("_", 2)
            if len(parts) == 3:
                return Detection._first_float(parts[-1], default)
        except Exception:
            pass
        return default

    @staticmethod
    def _resolve_fps(fps: Optional[float], df_mapping, video_id: str, default: float = 30.0) -> float:
        """Resolve FPS from explicit argument, mapping metadata, or video key suffix."""
        value = Detection._first_float(fps, None)
        if value is not None and value > 0:
            return float(value)

        value = Detection._fps_from_video_key(video_id, None)
        if value is not None and value > 0:
            return float(value)

        try:
            result = metadata.find_values_with_video_id(df_mapping, video_id)
            if result is not None and len(result) > 17:
                value = Detection._first_float(result[17], None)
                if value is not None and value > 0:
                    return float(value)
        except Exception:
            pass

        return float(default)

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

    def pedestrian_crossing(self, dataframe: pl.DataFrame, video_id: str, df_mapping, min_x: float, max_x: float,
                            person_id, tol: float = 0.00, min_track_frames: int = 10, min_road_frames: int = 3,
                            max_track_gap_frames: int = 30, min_crossing_x_range: float = 0.14,
                            max_crossing_speed_per_frame: Optional[float] = None,
                            weak_crossing_x_range: float = 0.64,
                            low_x_range: float = 0.30,
                            low_x_min_road_frames: int = 20,
                            tiny_long_track_x_range: float = 0.36,
                            tiny_long_track_height: float = 0.12,
                            tiny_long_track_road_frames: int = 50,
                            slender_track_width: float = 0.05,
                            slender_track_height: float = 0.26,
                            slender_track_min_road_frames: int = 5,
                            slender_track_max_road_frames: int = 49,
                            strong_static_relx: float = 0.155,
                            heavy_camera_static_frames: int = 80,
                            heavy_camera_static_sx: float = 0.02,
                            large_lateral_x_range: float = 0.56,
                            large_lateral_tiny_height: float = 0.105,
                            camera_static_sx: float = 0.25,
                            camera_static_ratio: float = 0.60,
                            camera_static_relx: float = 0.18,
                            camera_static_height: float = 0.15,
                            camera_static_tiny_relx: float = 0.12,
                            camera_static_tiny_relx_height: float = 0.19,
                            weak_y_jitter_x_range: float = 0.50,
                            weak_y_jitter_motion: float = 0.30,
                            weak_y_jitter_height: float = 0.22,
                            no_static_slender_height: float = 0.24,
                            no_static_slender_max_road_frames: int = 20,
                            tiny_no_static_height: float = 0.12,
                            tiny_no_static_width: float = 0.026,
                            tiny_no_static_min_road_frames: int = 10,
                            no_static_tiny_min_road_frames: int = 5,
                            no_static_tiny_fast_speed: float = 0.006,
                            slender_static_relx_min: float = 0.13,
                            camera_tiny_height: float = 0.15,
                            fps: Optional[float] = None,
                            base_fps: float = 30.0,
                            min_static_shared_frames: int = 8,
                            long_weak_road_frames: int = 90,
                            jitter_road_frames: int = 40,
                            camera_min_road_frames: int = 5,
                            track_index: Optional[dict[Any, pl.DataFrame]] = None
                            ) -> Tuple[List[Any], List[Any], Dict[Any, Tuple[int, int]]]:
        """
        Identifies pedestrian tracks that satisfy a road-crossing criterion and filters false positives.

        Tuned validation behaviour:
        - Splits reused tracker IDs into temporal segments when frame gaps are large.
        - Keeps the candidate stage broad, then rejects weak geometry cases after rider filtering.
        - Relaxed long-road rejection so slow true crossings are not removed.
        - Applies rider filtering on the segment window, not on the whole video, to avoid ID-reuse artefacts.
        - Scales all frame-count thresholds by fps/base_fps so 30 fps behaviour stays unchanged.

        Returns a ``(pedestrian_ids, crossed_ids, pedestrian_bounds)`` tuple. A
        tracker id can be reused for an unrelated object much later in the same
        video, so ``pedestrian_ids`` alone is not enough to recover the frames
        that actually crossed: a caller that filters the source dataframe by id
        would pick up that unrelated reappearance too. ``pedestrian_bounds``
        maps each accepted id to the ``(start_frame, end_frame)`` of the
        specific segment that qualified, so callers that need the crossing's
        own frames (crossing duration, waiting time, speed features) can
        restrict to that range instead of everything sharing the id.
        """
        fps_value = Detection._resolve_fps(fps, df_mapping, video_id, default=float(base_fps))
        base_fps_value = max(float(base_fps), 1e-9)

        def scaled(frames: int, minimum: int) -> int:
            return Detection._scale_frames(frames, fps_value, base_fps_value, minimum=minimum)

        min_track_frames_s = scaled(min_track_frames, minimum=1)
        min_road_frames_s = scaled(min_road_frames, minimum=1)
        max_track_gap_frames_s = scaled(max_track_gap_frames, minimum=0)
        low_x_min_road_frames_s = scaled(low_x_min_road_frames, minimum=1)
        tiny_long_track_road_frames_s = scaled(tiny_long_track_road_frames, minimum=1)
        slender_track_min_road_frames_s = scaled(slender_track_min_road_frames, minimum=1)
        slender_track_max_road_frames_s = scaled(slender_track_max_road_frames, minimum=1)
        no_static_slender_max_road_frames_s = scaled(no_static_slender_max_road_frames, minimum=1)
        tiny_no_static_min_road_frames_s = scaled(tiny_no_static_min_road_frames, minimum=1)
        no_static_tiny_min_road_frames_s = scaled(no_static_tiny_min_road_frames, minimum=1)
        min_static_shared_frames_s = scaled(min_static_shared_frames, minimum=1)
        long_weak_road_frames_s = scaled(long_weak_road_frames, minimum=1)
        jitter_road_frames_s = scaled(jitter_road_frames, minimum=1)
        camera_min_road_frames_s = scaled(camera_min_road_frames, minimum=1)

        rider_min_shared_frames_s = scaled(4, minimum=1)
        rider_min_continuous_shared_frames_s = scaled(12, minimum=1)
        rider_shared_run_gap_allow_s = scaled(2, minimum=0)
        rider_min_motion_steps_s = scaled(3, minimum=1)
        rider_short_shared_frames_s = scaled(8, minimum=1)

        track_partitions: List[pl.DataFrame] = []

        if track_index is not None:
            # The worker has already partitioned the complete CSV once. Reuse
            # those person tracks here instead of filtering, sorting and
            # partitioning the full person table again.
            for raw_track in track_index.values():
                if raw_track.height == 0:
                    continue
                person_track = raw_track.filter(
                    pl.col("yolo-id") == int(person_id)
                )
                if person_track.height == 0:
                    continue
                prepared = Detection._dedup_per_frame(person_track)
                prepared = (
                    prepared
                    .select([
                        "unique-id",
                        "frame-count",
                        "x-center",
                        "y-center",
                        "width",
                        "height",
                    ])
                    .sort("frame-count")
                )
                if prepared.height > 0:
                    track_partitions.append(prepared)
        else:
            crossed_df = dataframe.filter(pl.col("yolo-id") == 0)
            if crossed_df.height == 0:
                return [], [], {}

            crossed_df = Detection._dedup_per_frame(crossed_df)

            tracks = (
                crossed_df
                .select([
                    "unique-id",
                    "frame-count",
                    "x-center",
                    "y-center",
                    "width",
                    "height",
                ])
                .sort(["unique-id", "frame-count"])
            )
            if tracks.height == 0:
                return [], [], {}

            track_partitions = tracks.partition_by(
                "unique-id",
                maintain_order=True,
            )

        if not track_partitions:
            return [], [], {}

        left_hard = float(min_x) - float(tol)
        left_soft = float(min_x) + float(tol)
        right_soft = float(max_x) - float(tol)
        right_hard = float(max_x) + float(tol)

        def split_segments(track: pl.DataFrame) -> List[pl.DataFrame]:
            """Split one tracker id into near-continuous temporal segments."""
            if track.height == 0:
                return []

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
                else:
                    s = s

                states[i] = s

            return states

        def segment_is_candidate(seg: pl.DataFrame) -> bool:
            if seg.height < int(min_track_frames_s):
                return False

            x = seg.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
            if x.size == 0:
                return False

            states = build_states(x)

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

        candidate_segments: List[Tuple[Any, int, int, float, float, int, float, float, float]] = []
        crossed_ids: List[Any] = []
        crossed_ids_seen = set()

        for tr in track_partitions:
            if tr.height == 0:
                continue

            uid = tr.get_column("unique-id")[0]
            if tr.height < int(min_track_frames_s):
                continue

            for seg in split_segments(tr):
                if not segment_is_candidate(seg):
                    continue

                frames = seg.get_column("frame-count").cast(pl.Int64, strict=False).to_numpy()
                x = seg.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
                if frames.size == 0 or x.size == 0:
                    continue

                start_frame = int(frames.min())
                end_frame = int(frames.max())
                duration = max(1, end_frame - start_frame + 1)
                x_range = float(np.nanmax(x) - np.nanmin(x))
                x_speed = float(x_range / duration)

                states = build_states(x)
                road_frames = int((states == 1).sum())

                if "height" in seg.columns:
                    height = seg.get_column("height").cast(pl.Float64, strict=False).to_numpy()
                    median_height = float(np.nanmedian(height)) if height.size > 0 else 0.0
                else:
                    median_height = 0.0

                if "width" in seg.columns:
                    width = seg.get_column("width").cast(pl.Float64, strict=False).to_numpy()
                    median_width = float(np.nanmedian(width)) if width.size > 0 else 0.0
                else:
                    median_width = 0.0

                if "y-center" in seg.columns:
                    y = seg.get_column("y-center").cast(pl.Float64, strict=False).to_numpy()
                    y_gross_motion = float(np.nansum(np.abs(np.diff(y)))) if y.size > 1 else 0.0
                else:
                    y_gross_motion = 0.0

                candidate_segments.append(
                    (
                        uid, start_frame, end_frame, x_range, x_speed, road_frames,
                        median_height, median_width, y_gross_motion,
                    )
                )
                if uid not in crossed_ids_seen:
                    crossed_ids.append(uid)
                    crossed_ids_seen.add(uid)

        avg_height = None
        result = metadata.find_values_with_video_id(df_mapping, video_id)
        if result is not None:
            avg_height = result[15]

        pedestrian_ids: List[Any] = []
        pedestrian_ids_seen = set()
        pedestrian_bounds: Dict[Any, Tuple[int, int]] = {}

        # Sort once by frame. Each candidate window is then located with two
        # binary searches and extracted with slice(), avoiding a full DataFrame
        # filter for every candidate.
        frame_sorted_df = (
            dataframe
            .filter(pl.col("frame-count").is_not_null())
            .sort("frame-count")
        )
        frame_values = (
            frame_sorted_df
            .get_column("frame-count")
            .cast(pl.Int64, strict=False)
            .to_numpy()
        )

        for (
            uid, start_frame, end_frame, x_range, x_speed, road_frames,
            median_height, median_width, y_gross_motion,
        ) in candidate_segments:
            left_idx = int(np.searchsorted(frame_values, int(start_frame), side="left"))
            right_idx = int(np.searchsorted(frame_values, int(end_frame), side="right"))
            segment_df = frame_sorted_df.slice(
                left_idx,
                max(0, right_idx - left_idx),
            )

            if Detection.is_rider_id(
                segment_df,
                uid,
                avg_height,
                min_shared_frames=rider_min_shared_frames_s,
                min_continuous_shared_frames=rider_min_continuous_shared_frames_s,
                shared_run_gap_allow=rider_shared_run_gap_allow_s,
                min_motion_steps=rider_min_motion_steps_s,
                short_shared_frames=rider_short_shared_frames_s,
            ):
                continue

            static_stats = Detection.static_reference_motion_stats(
                segment_df,
                uid,
                MIN_SHARED_FRAMES=min_static_shared_frames_s,
            )
            static_shared = int(static_stats.get("shared_frames", 0) or 0)
            static_sx_range = float(static_stats.get("static_x_range", 0.0) or 0.0)
            static_relx_range = float(static_stats.get("relative_x_range", 0.0) or 0.0)
            static_ratio = float(static_stats.get("static_to_person_ratio", 0.0) or 0.0)

            if float(x_range) < float(min_crossing_x_range):
                continue

            if float(x_range) < float(low_x_range) and int(road_frames) < int(low_x_min_road_frames_s):
                continue

            if float(x_range) < float(weak_crossing_x_range) and int(road_frames) > int(long_weak_road_frames_s):
                continue

            if float(x_range) < 0.56 and int(road_frames) > int(jitter_road_frames_s) and float(y_gross_motion) > 0.30:
                continue

            if (
                float(x_range) < float(weak_y_jitter_x_range)
                and float(y_gross_motion) > float(weak_y_jitter_motion)
                and float(median_height) < float(weak_y_jitter_height)
            ):
                continue

            if (
                float(x_range) < float(tiny_long_track_x_range)
                and float(median_height) < float(tiny_long_track_height)
                and int(road_frames) >= int(tiny_long_track_road_frames_s)
            ):
                continue

            if (
                static_shared < int(min_static_shared_frames_s)
                and float(median_height) <= float(tiny_no_static_height)
                and float(median_width) <= float(tiny_no_static_width)
                and (
                    int(road_frames) >= int(tiny_no_static_min_road_frames_s)
                    or int(road_frames) >= int(no_static_tiny_min_road_frames_s)
                    or float(x_speed) >= float(no_static_tiny_fast_speed)
                )
            ):
                continue

            if static_sx_range >= float(camera_static_sx) and static_ratio >= float(camera_static_ratio):
                if (
                    float(median_height) <= float(camera_tiny_height)
                    and int(road_frames) >= int(camera_min_road_frames_s)
                ):
                    continue
                if (
                    static_relx_range <= float(camera_static_tiny_relx)
                    and float(median_height) <= float(camera_static_tiny_relx_height)
                ):
                    continue
                if (
                    static_relx_range <= float(camera_static_relx)
                    and float(median_height) <= float(camera_static_height)
                ):
                    continue

            if (
                float(median_width) <= float(slender_track_width)
                and float(median_height) < float(slender_track_height)
                and int(slender_track_min_road_frames_s) <= int(road_frames) <= int(slender_track_max_road_frames_s)
            ):
                if static_shared < int(min_static_shared_frames_s):
                    if (
                        float(median_height) < float(no_static_slender_height)
                        and int(road_frames) <= int(no_static_slender_max_road_frames_s)
                    ):
                        continue
                else:
                    if static_relx_range < float(slender_static_relx_min):
                        continue
                    if (
                        static_sx_range >= float(camera_static_sx)
                        and static_ratio >= float(camera_static_ratio)
                        and float(median_height) <= float(camera_tiny_height)
                    ):
                        continue

            if (
                float(x_range) > float(large_lateral_x_range)
                and float(median_height) < float(large_lateral_tiny_height)
                and int(road_frames) >= int(camera_min_road_frames_s)
                and static_sx_range >= float(camera_static_sx)
                and static_ratio >= float(camera_static_ratio)
                and static_relx_range < 0.20
            ):
                continue

            if max_crossing_speed_per_frame is not None:
                if float(x_speed) > float(max_crossing_speed_per_frame):
                    continue

            if not Detection.is_valid_crossing(
                segment_df,
                uid,
                MIN_SHARED_FRAMES=min_static_shared_frames_s,
            ):
                continue

            if uid not in pedestrian_ids_seen:
                pedestrian_ids.append(uid)
                pedestrian_ids_seen.add(uid)
                pedestrian_bounds[uid] = (int(start_frame), int(end_frame))

        return pedestrian_ids, crossed_ids, pedestrian_bounds

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
    def classify_rider_type(
        df: pl.DataFrame,
        person_id,
        *,
        avg_height: Optional[float] = None,
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
        person_class: int = 0,
        bicycle_class: int = 1,
        motorcycle_class: int = 3,
        car_class: int = 2,
        bus_class: int = 5,
        truck_class: int = 7,
        include_large_vehicle_passengers: bool = False,
        pooled_min_coverage: float = 0.40,
        pooled_min_person_motion: float = 0.30,
        pooled_max_motion_mismatch: float = 0.50,
        pooled_max_offset_std: float = 0.25,
        pooled_min_seated_share: float = 0.35,
    ) -> dict:
        """Classify whether a person track is associated with a vehicle."""
        if avg_height is not None:
            try:
                if float(avg_height) <= 0.0:
                    return {
                        "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                        "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
                    }
            except Exception:
                return {
                    "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                    "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
                }

        df = Detection._dedup_per_frame(df)

        p = (
            df.filter((pl.col("yolo-id") == person_class) & (pl.col("unique-id") == person_id))
              .sort("frame-count")
        )
        if p.height == 0:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        p_frames = p.get_column("frame-count").to_numpy()
        if p_frames.size < min_shared_frames:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        first_frame = int(p_frames.min())
        last_frame = int(p_frames.max())

        supported_vehicle_classes = [bicycle_class, motorcycle_class]
        if include_large_vehicle_passengers:
            supported_vehicle_classes.extend([car_class, bus_class, truck_class])

        vehicles = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in(supported_vehicle_classes))
        )
        if vehicles.height == 0:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        p1 = p.unique(subset=["frame-count"], keep="first")
        best = None

        # Partition once instead of filtering vehicles for every tracker ID.
        vehicle_partitions = vehicles.partition_by(
            "unique-id",
            maintain_order=True,
        )

        for v in vehicle_partitions:
            if v.height == 0:
                continue

            v = v.sort("frame-count")
            vid = v.get_column("unique-id")[0]
            v_class = int(v.get_column("yolo-id")[0])
            vtype = (
                "bicycle" if v_class == bicycle_class else
                "motorcycle" if v_class == motorcycle_class else
                "car" if v_class == car_class else
                "bus" if v_class == bus_class else
                "truck" if v_class == truck_class else
                None
            )
            if vtype is None:
                continue

            role = "rider" if v_class in (bicycle_class, motorcycle_class) else "passenger"

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
                inside = (np.abs(relx) <= 0.5 * v_w) & (np.abs(rely) <= 0.5 * v_h)
                spatial = inside

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
                vehicles.filter(pl.col("yolo-id").is_in([bicycle_class, motorcycle_class])),
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
                bicycle_class=bicycle_class,
                eps=eps,
            )

        if best is None:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        return best

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
        bicycle_class: int,
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
            "rider_type": "bicycle" if int(majority_class) == int(bicycle_class) else "motorcycle",
            "role": "rider",
            "vehicle_id": vehicle_ids[0] if len(vehicle_ids) == 1 else vehicle_ids,
            "score": float(min(1.0, coverage)),
            "shared_frames": int(shared),
            "longest_shared_run": int(
                Detection._longest_frame_run(seated_frames.tolist(), gap_allow=0)
            ),
            "pooled": True,
            "pooled_vehicle_ids": len(vehicle_ids),
            "coverage": coverage,
            "person_motion": person_motion,
            "motion_mismatch": mismatch,
            "seated_share": seated_share,
            "offset_std": offset_std,
        }

    @staticmethod
    def is_rider_id(
        df: pl.DataFrame,
        id,
        avg_height: Optional[float] = None,
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
    ) -> bool:
        """Return True when the person is associated with a vehicle."""
        res = Detection.classify_rider_type(
            df,
            id,
            avg_height=avg_height,
            min_shared_frames=min_shared_frames,
            min_continuous_shared_frames=min_continuous_shared_frames,
            shared_run_gap_allow=shared_run_gap_allow,
            min_vehicle_width_ratio=min_vehicle_width_ratio,
            min_vehicle_width_ratio_frames=min_vehicle_width_ratio_frames,
            dist_rel_thresh=dist_rel_thresh,
            prox_req=prox_req,
            alpha_x=alpha_x,
            beta_y=beta_y,
            gamma_y=gamma_y,
            coloc_req=coloc_req,
            sim_thresh=sim_thresh,
            sim_req=sim_req,
            min_motion_steps=min_motion_steps,
            motion_coloc_min=motion_coloc_min,
            short_shared_frames=short_shared_frames,
            short_sim_req=short_sim_req,
            short_disp_req=short_disp_req,
            eps=eps,
            include_large_vehicle_passengers=include_large_vehicle_passengers,
            pooled_min_coverage=pooled_min_coverage,
            pooled_min_person_motion=pooled_min_person_motion,
            pooled_max_motion_mismatch=pooled_max_motion_mismatch,
            pooled_max_offset_std=pooled_max_offset_std,
            pooled_min_seated_share=pooled_min_seated_share,
        )
        return bool(res.get("is_rider"))

    @staticmethod
    def static_reference_motion_stats(df, person_id, STATIC_CLASS_IDS=(9, 10, 11, 12, 13),
                                      MIN_SHARED_FRAMES=8, Q=0.05, EPS=1e-9):
        """Return motion statistics between a person track and the best static reference."""
        empty = {
            "has_reference": False,
            "static_id": None,
            "static_class": None,
            "shared_frames": 0,
            "person_x_range": 0.0,
            "static_x_range": 0.0,
            "relative_x_range": 0.0,
            "static_to_person_ratio": 0.0,
        }

        if df.height == 0:
            return empty

        if "confidence" in df.columns:
            df = Detection._dedup_per_frame(df)

        p = (
            df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == person_id))
              .sort("frame-count")
        )
        if p.height == 0:
            return empty

        p = p.unique(subset=["frame-count"], keep="first")
        first_frame = int(p.get_column("frame-count").min())
        last_frame = int(p.get_column("frame-count").max())

        refs = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in(STATIC_CLASS_IDS))
        )
        if refs.height == 0:
            return empty

        def robust_range(values) -> float:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 0.0
            lo = float(np.quantile(arr, float(Q)))
            hi = float(np.quantile(arr, 1.0 - float(Q)))
            return max(0.0, hi - lo)

        best = None

        # Partition once rather than filtering refs for every static object ID.
        reference_partitions = refs.partition_by(
            "unique-id",
            maintain_order=True,
        )

        for r in reference_partitions:
            if r.height == 0:
                continue

            r = r.sort("frame-count")
            ref_id = r.get_column("unique-id")[0]
            r = r.unique(subset=["frame-count"], keep="first")
            joined = p.join(r, on="frame-count", how="inner", suffix="_ref")
            shared = joined.height
            if shared < int(MIN_SHARED_FRAMES):
                continue

            person_x = joined.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
            ref_x = joined.get_column("x-center_ref").cast(pl.Float64, strict=False).to_numpy()

            person_x_range = robust_range(person_x)
            static_x_range = robust_range(ref_x)
            relative_x_range = robust_range(person_x - ref_x)
            ratio = static_x_range / max(person_x_range, float(EPS))

            ref_class = None
            try:
                ref_class = int(r.get_column("yolo-id")[0])
            except Exception:
                ref_class = None

            cand = {
                "has_reference": True,
                "static_id": ref_id,
                "static_class": ref_class,
                "shared_frames": int(shared),
                "person_x_range": float(person_x_range),
                "static_x_range": float(static_x_range),
                "relative_x_range": float(relative_x_range),
                "static_to_person_ratio": float(ratio),
            }

            if best is None or (cand["shared_frames"], cand["static_x_range"]) > (
                best["shared_frames"], best["static_x_range"]
            ):
                best = cand

        return best if best is not None else empty

    @staticmethod
    def is_valid_crossing(df, person_id, ratio_thresh=0.6, STATIC_CLASS_IDS=(9, 10, 11, 12, 13),
                          MIN_SHARED_FRAMES=8, RELX_MIN=0.01, Q=0.05, EPS=1e-9):
        """Check whether an apparent crossing is independent of camera motion."""
        if "confidence" in df.columns:
            df = (
                df.sort(
                    ["yolo-id", "unique-id", "frame-count", "confidence"],
                    descending=[False, False, False, True],
                )
                .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
            )
        else:
            df = df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        person_track = (
            df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == person_id))
            .sort("frame-count")
            .unique(subset=["frame-count"], keep="first")
        )
        if person_track.height == 0:
            return False

        frames = person_track.get_column("frame-count").to_numpy()
        first_frame = int(frames.min())
        last_frame = int(frames.max())

        static_objs = (
            df.filter(
                (pl.col("frame-count") >= first_frame)
                & (pl.col("frame-count") <= last_frame)
                & (pl.col("yolo-id").is_in(list(STATIC_CLASS_IDS)))
            )
            .sort("frame-count")
        )

        if static_objs.height == 0:
            return True

        def robust_range(series: pl.Series) -> float:
            try:
                lo = series.quantile(Q, "nearest")
                hi = series.quantile(1.0 - Q, "nearest")
                return float(hi - lo)
            except Exception:
                return 0.0

        best = None

        # Partition once rather than filtering static_objs for every tracker ID.
        static_partitions = static_objs.partition_by(
            "unique-id",
            maintain_order=True,
        )

        for s_track in static_partitions:
            if s_track.height == 0:
                continue

            s_track = (
                s_track
                .sort("frame-count")
                .unique(subset=["frame-count"], keep="first")
            )

            joined = person_track.join(s_track, on="frame-count", how="inner", suffix="_s")

            if joined.height < MIN_SHARED_FRAMES:
                continue

            px = joined.get_column("x-center")
            sx = joined.get_column("x-center_s")
            relx = px - sx

            px_rng = robust_range(px)
            sx_rng = robust_range(sx)
            relx_rng = robust_range(relx)

            ratio = float(sx_rng / max(px_rng, EPS))

            cand = {
                "shared": int(joined.height),
                "px_rng": float(px_rng),
                "sx_rng": float(sx_rng),
                "relx_rng": float(relx_rng),
                "ratio": float(ratio),
            }

            if best is None:
                best = cand
            elif (cand["shared"], cand["sx_rng"]) > (best["shared"], best["sx_rng"]):
                best = cand

        if best is None:
            return True

        if best["relx_rng"] < RELX_MIN:
            return False

        if best["ratio"] >= float(ratio_thresh) and best["relx_rng"] < (2.0 * RELX_MIN):
            return False

        return True
