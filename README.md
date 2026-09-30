# Pedestrian crossing speed and hesitation time from dashcam videos

Detects and tracks road users in dashcam videos with YOLOv11x and BoT-SORT, then
measures, for every pedestrian who crosses the road in front of the car:

- **crossing speed** (m/s)
- **hesitation time** (s): how long the pedestrian stood still before starting to cross
- **crossing time** (s): how long the crossing was observed

The pipeline follows [crowd-dataset/crowd](https://github.com/crowd-dataset/crowd)
(tracking) and [crowd-dataset/crowd-city](https://github.com/crowd-dataset/crowd-city)
(crossing detection and metrics), and uses the same project structure: `common.py`,
`config` / `default.config`, `logmod.py`, `custom_logger.py`, `helper_script.py`,
`main.py`, `analysis.py` and `utils/`.

## Data layout

```
data/
├── Amsterdam/
│   ├── drive_01.mp4
│   └── drive_02.mp4
└── Delhi/
    └── morning.mov
```

Every sub-folder of `data/` is a city, and every video in it (sub-folders included) is analysed.

## Setup

```bash
uv sync
cp default.config config
```

`config` must contain every key of `default.config` (the run stops otherwise). `default.secret` → `secret` is only needed for email notifications.

## Running

1. **Tracking** writes one CSV per video to `_output/bbox/<city>/<video>_<fps>.csv`, a `<video>_<fps>.json` next to it with the tracking settings used, and `_output/videos.csv` (city, fps, size and duration of every video). Videos that already have a CSV are skipped unless `always_analyse` is true, and an interrupted video is redone on the next run. If the tracking settings in `config` differ from those stored with a CSV, the run warns but keeps the CSV: delete it (or set `always_analyse`) to track it again.

   ```bash
   uv run python main.py
   ```

   The first run downloads `yolo11x.pt` through ultralytics. The device is picked automatically (CUDA, Apple MPS, then CPU).

2. **Analysis** writes `_output/crossings.parquet` and `_output/crossings.csv` (one row per crossing), `_output/city_summary.csv` (one row per city) and plots in `_output/figures/`.

   ```bash
   uv run python analysis.py
   ```

   Add `--force` to ignore the cache and analyse every video again.

### What is rerun

Both stages only process what changed, so adding videos or editing the analysis code is cheap.

| Change | Rerun |
|---|---|
| New video in `data/` | Tracking and analysis of that video only |
| Nothing | Nothing (cached results are combined again) |
| Code in `utils/crossing/metrics.py`, or `check_per_sec_time`, `hesitation_reference`, stature | Metrics of every video; detection reused, tracking CSVs not read for detection |
| Code in `utils/crossing/detection.py`, or `boundary_left` / `boundary_right` | Detection of every video; metrics reused for videos whose crossings are unchanged |
| A tracking CSV is rewritten | Analysis of that video |
| Comments, docstrings or formatting in the analysis code | Nothing |
| Speed or waiting-time limits, plots, summary code | Nothing (always rebuilt from the cached values) |

Per-video results are cached in `_output/analysis/<city>/<video>_<fps>.json`: the detected crossings (track id and frame bounds) and one row of raw metrics per crossing, each stored with a key built from its inputs (see `utils/cache.py`). Code is fingerprinted from its syntax tree, so only real code changes count. Video metadata in `videos.csv` is also reused for files whose size and modification time are unchanged.

The tracking CSV columns are the same as CROWD's (`yolo-id, x-center, y-center, width, height, unique-id, confidence, frame-count`, boxes normalised to the frame), so `analysis.py` also works on CROWD CSVs listed in a `videos.csv`.

## Method

**Crossing detection** (`utils/crossing/detection.py`) is ported from crowd-city. A person track is a crossing when it moves from one side of the image, through the central strip `[boundary_left, boundary_right]`, to the other side. Candidates are then rejected by CROWD's geometric filters: too little lateral movement, jitter, tiny or slender boxes, and camera motion measured against static objects. People riding bicycles or motorcycles are also rejected with crowd-city's rider classifier: a two-wheeler that stays under the person, is wide enough and moves with them, including detections pooled across fragmented vehicle track ids.

**Crossing speed** (`utils/crossing/metrics.py`). crowd-city converts box motion to m/s with a model calibrated on the Waymo Open Dataset, which is not included here. Instead, the pinhole relation places the pedestrian in metres: a person of stature *H* whose box is *h* tall is at lateral position `X = (x − 0.5) · aspect · H / h`. Forward motion of the car does not change *X*, so it cancels out. The speed is the Theil–Sen slope of *X* over time, fitted after the hesitation interval. The stature is `person_height_m` (1.70 m), and `city_person_height_m` can override it per city, e.g. `{"Amsterdam": 1.78}`.

**Hesitation time** follows crowd-city's `time_to_start_cross`. The track is sampled `check_per_sec_time` times per second, on frame numbers rather than detections, so missed frames do not change the measured time. A sample is stationary when the pedestrian moved at most 10% of their stature since the previous one. The hesitation time is the first run of at least three stationary samples. With `"hesitation_reference": "road_entry"` it is instead the stationary run ending when the pedestrian enters the road strip, as in crowd-city's `road_metrics.hesitation_seconds`. Crossings without such a run get `0.0` and `hesitated = false` in `crossings.csv`. As in crowd-city (commit `46ff6d6`), such a pedestrian has no hesitation time rather than one of zero, so the hesitation mean, median, standard deviation and plots in `city_summary.csv` and `figures/` use only the crossings with a wait (`crossings_hesitated`). `share_hesitated` gives the fraction that waited, and `hesitation_mean_all_s` the mean including the zeros.

### Limitations

- Speeds assume the whole body is inside the box and an average stature. Occlusion by the bonnet or other cars shortens the box and inflates the speed.
- Camera yaw (the car turning) is not compensated in the speed. Crossing detection only rejects the tracks it affects most.
- The road is assumed to be the central image strip, as in CROWD without segmentation.

## Configuration

| Key | Meaning |
|---|---|
| `data`, `output` | Input and output folders, relative to the repository or absolute |
| `cities_analyse` | Only these city folders; empty means all |
| `tracking_model`, `bbox_tracker` | YOLO weights and BoT-SORT YAML |
| `track_buffer_sec` | Seconds a lost track is kept; scaled to frames per video |
| `min_confidence`, `yolo_imgsz`, `device`, `half_precision` | Detection settings |
| `save_annotated_video`, `display_frame_tracking` | Write `_output/annotated/<city>/<video>.mp4` / show frames live |
| `boundary_left`, `boundary_right` | Road strip in normalised image x |
| `check_per_sec_time`, `hesitation_reference` | Hesitation sampling rate and reference (`track_start` or `road_entry`) |
| `min/max_speed_limit`, `min/max_waiting_time` | Values outside these ranges are dropped |
| `cpu_worker` | Processes used by `analysis.py` |
| `save_images` | Also export PNG/EPS figures (needs kaleido) |
