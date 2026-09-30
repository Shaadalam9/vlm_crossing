"""Video discovery and YOLO + BoT-SORT tracking for local dashcam videos.

Adapted from helper_script.py (Youtube_Helper.tracking_mode) in
crowd-dataset/crowd. The YouTube/FTP download logic is not needed here because
videos are read from the local data folder, laid out as data/<city>/<video>.

The CSV written per video has the same columns as CROWD:
yolo-id, x-center, y-center, width, height, unique-id, confidence, frame-count
with box coordinates normalised to the frame size and frame-count starting at 1.
"""

import json
import logging
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
import ultralytics
from ultralytics import YOLO  # type: ignore

import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger
logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Show only errors

# Consts
LINE_THICKNESS = 1
TRACK_TRAIL_LENGTH = 30
CSV_COLUMNS = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]


class Video_Helper:
    """Helper for finding dashcam videos and running detection and tracking on them.

    Attributes:
        tracking_model (str): YOLO weights, e.g. yolo11x.pt (downloaded by ultralytics if missing).
        bbox_tracker (str): Tracker YAML (BoT-SORT).
        confidence (float): Minimum detection confidence.
    """

    def __init__(self):
        self.tracking_model = common.get_configs("tracking_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.confidence = common.get_configs("min_confidence")
        self.imgsz = common.get_configs("yolo_imgsz")
        self.half = common.get_configs("half_precision")
        self.display_frame_tracking = common.get_configs("display_frame_tracking")
        self.video_extensions = tuple(ext.lower() for ext in common.get_configs("video_extensions"))
        self.device = self.detect_device(common.get_configs("device"))

    # ------------------------------------------------------------------
    # Discovery and metadata
    # ------------------------------------------------------------------

    def discover_videos(self, data_dir: str, cities: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Find videos laid out as <data_dir>/<city>/<video>.

        Args:
            data_dir (str): Root data folder.
            cities (list, optional): Only return these city folders. Empty or None means all.

        Returns:
            list: (city, video_path) tuples sorted by city and file name.
        """
        if not os.path.isdir(data_dir):
            logger.error(f"Data folder {data_dir} does not exist.")
            return []

        wanted = {c.lower() for c in cities} if cities else None
        videos = []
        for city in sorted(os.listdir(data_dir)):
            city_dir = os.path.join(data_dir, city)
            if not os.path.isdir(city_dir) or city.startswith("."):
                continue
            if wanted is not None and city.lower() not in wanted:
                continue
            for root, _, files in os.walk(city_dir):
                for name in sorted(files):
                    if name.startswith(".") or not name.lower().endswith(self.video_extensions):
                        continue
                    videos.append((city, os.path.join(root, name)))
        return videos

    @staticmethod
    def get_video_properties(video_path: str) -> Optional[Dict[str, float]]:
        """
        Read fps, frame count, size and duration of a video with OpenCV.

        Args:
            video_path (str): Video file.

        Returns:
            dict or None: fps, frames, width, height, duration_s; None if the file cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}.")
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if fps <= 0 or width <= 0 or height <= 0:
            logger.error(f"Video {video_path} has invalid metadata (fps={fps}, size={width}x{height}).")
            return None
        return {
            "fps": round(fps, 3),
            "frames": frames,
            "width": width,
            "height": height,
            "duration_s": frames / fps if frames > 0 else 0.0,
        }

    @staticmethod
    def file_stamp(path: str) -> Dict[str, int]:
        """Size and modification time of a file, used to reuse metadata of unchanged videos."""
        stat = os.stat(path)
        return {"file_size": int(stat.st_size), "file_mtime_ns": int(stat.st_mtime_ns)}

    @staticmethod
    def csv_name(video_path: str, fps: float) -> str:
        """Return the CROWD-style CSV name <video>_<fps>.csv for a video."""
        stem = os.path.splitext(os.path.basename(video_path))[0]
        return f"{stem}_{int(round(fps))}.csv"

    @staticmethod
    def detect_device(requested: str = "auto") -> str:
        """
        Pick the torch device for YOLO.

        Args:
            requested (str): "auto", "cpu", "mps", "cuda" or "cuda:<n>".

        Returns:
            str: Device string.
        """
        if requested and requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def tracking_settings(self) -> Dict[str, object]:
        """Settings that change the tracking CSV. Stored next to each CSV as <csv>.json."""
        tracker = common.resolve_path(self.bbox_tracker)
        tracker_config = self.bbox_tracker
        if os.path.exists(tracker):
            with open(tracker) as f:
                tracker_config = yaml.safe_load(f)
            tracker_config.pop("track_buffer", None)  # replaced by track_buffer_sec
        return {
            "tracking_model": self.tracking_model,
            "min_confidence": self.confidence,
            "yolo_imgsz": self.imgsz,
            "half_precision": self.half,
            "track_buffer_sec": common.get_configs("track_buffer_sec"),
            "tracker": tracker_config,
            "ultralytics": ultralytics.__version__,
        }

    @staticmethod
    def settings_path(output_csv: str) -> str:
        return os.path.splitext(output_csv)[0] + ".json"

    def stored_tracking_settings(self, output_csv: str) -> Optional[Dict[str, object]]:
        """Settings a CSV was tracked with, or None for CSVs written before settings were stored."""
        try:
            with open(self.settings_path(output_csv)) as f:
                return json.load(f).get("settings")
        except (OSError, ValueError):
            return None

    def update_track_buffer_in_yaml(self, yaml_path: str, video_fps: float) -> None:
        """Set track_buffer so lost tracks are kept for track_buffer_sec seconds at this fps."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        config['track_buffer'] = int(round(common.get_configs("track_buffer_sec") * video_fps))

        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def _prepare_tracker_yaml(self, run_root: str, video_fps: float) -> str:
        """Copy the tracker YAML into the run folder and scale its track buffer to the fps."""
        src = common.resolve_path(self.bbox_tracker)
        if not os.path.exists(src):
            # Built-in ultralytics tracker names such as botsort.yaml.
            return self.bbox_tracker
        os.makedirs(run_root, exist_ok=True)
        dst = os.path.join(run_root, os.path.basename(src))
        try:
            shutil.copyfile(src, dst)
            self.update_track_buffer_in_yaml(dst, video_fps)
            return dst
        except Exception as e:
            logger.warning(f"Failed to prepare tracker YAML copy: {e!r}. Using original tracker path.")
            return src

    def tracking_mode(self, input_video_path: str, output_csv: str, video_fps: float,
                      run_root: str = "runs", annotated_video_path: Optional[str] = None) -> bool:
        """
        Run YOLO detection with BoT-SORT tracking on every frame and write one CSV per video.

        Rows are written to <output_csv>.partial and renamed only when the whole video
        has been processed, so an interrupted run is redone next time.

        Args:
            input_video_path (str): Video file.
            output_csv (str): CSV to write.
            video_fps (float): Frame rate of the video.
            run_root (str): Scratch folder for the tracker YAML copy.
            annotated_video_path (str, optional): Also write an annotated mp4 here.

        Returns:
            bool: True if the CSV was written.
        """
        tracker_path = self._prepare_tracker_yaml(run_root, video_fps)
        # A fresh model per video also gives a fresh tracker, so ids never carry over.
        model = YOLO(self.tracking_model)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {input_video_path}.")
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        video_writer = None
        if annotated_video_path:
            os.makedirs(os.path.dirname(annotated_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            video_writer = cv2.VideoWriter(annotated_video_path, fourcc, video_fps, (frame_width, frame_height))
        draw = video_writer is not None or self.display_frame_tracking

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        partial_csv = output_csv + ".partial"
        track_history = defaultdict(list)
        frame_count = 0
        failed_frames = 0
        stopped_by_user = False
        progress_bar = tqdm(total=total_frames, unit="frames", dynamic_ncols=True,
                            desc=os.path.basename(input_video_path))

        with open(partial_csv, "w") as csv_file:
            csv_file.write(",".join(CSV_COLUMNS) + "\n")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame_count += 1
                progress_bar.update(1)

                try:
                    results = model.track(
                        frame,
                        tracker=tracker_path,
                        persist=True,
                        conf=self.confidence,
                        imgsz=self.imgsz,
                        half=self.half,
                        save=False,
                        verbose=False,
                        device=self.device,
                    )
                except Exception as e:
                    logger.error(f"[Frame {frame_count}] Tracking failed: {e}.")
                    failed_frames += 1
                    continue

                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    if draw:
                        self._show_and_write(frame, video_writer)
                    continue

                cls = boxes.cls.int().cpu().tolist()
                xywhn = boxes.xywhn.cpu().tolist()
                confs = boxes.conf.float().cpu().tolist()
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(cls)

                lines = []
                for cls_i, (x, y, w, h), tid_i, conf_i in zip(cls, xywhn, ids, confs):
                    lines.append(f"{cls_i},{x:.6f},{y:.6f},{w:.6f},{h:.6f},{tid_i},{conf_i:.6f},{frame_count}\n")
                csv_file.writelines(lines)

                if draw:
                    annotated = results[0].plot(line_width=LINE_THICKNESS)
                    xywh = boxes.xywh.cpu().tolist()
                    for (x, y, _, _), tid_i in zip(xywh, ids):
                        if tid_i < 0:
                            continue
                        track = track_history[tid_i]
                        track.append((float(x), float(y)))
                        if len(track) > TRACK_TRAIL_LENGTH:
                            track.pop(0)
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated, [points], isClosed=False,
                                      color=(230, 230, 230), thickness=LINE_THICKNESS * 5)
                    if self._show_and_write(annotated, video_writer):
                        stopped_by_user = True
                        break

        cap.release()
        progress_bar.close()
        if video_writer is not None:
            video_writer.release()
        if self.display_frame_tracking:
            cv2.destroyAllWindows()

        if frame_count == 0 or stopped_by_user:
            if stopped_by_user:
                logger.warning(f"Tracking of {input_video_path} stopped by user; CSV not kept.")
            else:
                logger.error(f"No frames could be read from {input_video_path}.")
            os.remove(partial_csv)
            return False
        if failed_frames:
            logger.warning(f"{failed_frames} of {frame_count} frames failed in {input_video_path}.")

        os.replace(partial_csv, output_csv)
        with open(self.settings_path(output_csv), "w") as f:
            json.dump({"video": os.path.abspath(input_video_path), "frames": frame_count,
                       "failed_frames": failed_frames, "settings": self.tracking_settings()}, f, indent=1)
        logger.info(f"Wrote {output_csv} ({frame_count} frames).")
        return True

    def _show_and_write(self, frame, video_writer) -> bool:
        """Write/display a frame. Returns True when the user pressed 'q' in the display window."""
        if video_writer is not None:
            video_writer.write(frame)
        if self.display_frame_tracking:
            cv2.imshow("YOLOv11 Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True
        return False

    @staticmethod
    def delete_folder(folder_path: str) -> bool:
        """
        Delete a folder and all its contents.

        Args:
            folder_path (str): Folder to delete.

        Returns:
            bool: True if the folder was deleted.
        """
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                return True
            except Exception as e:
                logger.error(f"Failed to delete folder {folder_path}: {e}")
        return False
