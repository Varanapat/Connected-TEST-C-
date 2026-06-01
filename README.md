# 🎯 Stationary Period Analysis with YOLO

> **Automatically detect and measure how long each person stands still in a video — using AI, frame by frame.**
> [Colab](https://colab.research.google.com/drive/1QZSGvSxLA4XN4v22NWO4hItb2d72ZhvJ?usp=sharing)
> Youtube

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=pytorch&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

Modern spaces are full of cameras — but most of the time, the footage just sits there unused. We watch it for security, but rarely ask the deeper question: **where do people choose to stop?**

**Stationary Period Analysis with YOLO** is a computer vision pipeline that answers exactly that. It processes any video file and automatically identifies:

- Which people are stationary vs. moving at every moment
- How long each individual person remains still (in seconds)
- When and where stationary periods begin and end across the full clip

Instead of relying on manual observation or expensive sensor installations, this system extracts behavioral insights directly from video — using only a standard camera and open-source AI.

### Who is this for?

| Use Case | Example |
|---|---|
| 🏪 Retail & hospitality | Identify where customers linger longest |
| 🏛️ Public space design | Understand foot traffic patterns and dwell zones |
| 📚 Library / co-working | Detect seat occupancy duration without hardware |
| 🚀 Research & prototyping | Baseline system for behavior analysis pipelines |
| 🏆 Hackathon projects | End-to-end AI pipeline with visual output and CSV export |

---

## ✨ Features

- **🤖 AI-Powered Person Detection** — Uses YOLOv8 (COCO-trained) to detect people in each frame with configurable confidence thresholds. Filters non-person classes automatically.

- **🏷️ Persistent Identity Tracking** — Integrates ByteTrack and BotSort multi-object trackers to assign consistent IDs to each person across the entire video, even when they are briefly occluded or leave and re-enter the frame.

- **📐 Frame-to-Frame Displacement Analysis** — Computes Euclidean distance between bounding box centers on consecutive frames to measure movement speed in pixels/second. A configurable threshold determines whether a person is stationary or moving.

- **⏱️ Stationary Period Detection** — Records the exact start time, end time, and duration (in seconds) of every stationary event per tracking ID. Applies a minimum duration filter to eliminate noise from brief pauses.

- **🎬 Annotated Video Output** — Generates an annotated MP4 with color-coded bounding boxes (🟢 green = still, 🔴 red = moving), live speed readout, and tracking ID labels on every frame. Optional slow-motion playback factor for detailed review.

- **📊 Timeline Speed Graph** — Plots speed over time for each tracking ID with the stationary threshold line and shaded regions marking detected stationary periods. Auto-saved as a PNG at 160 DPI.

- **📁 CSV Data Export** — Exports three structured datasets: frame-by-frame tracking records, per-frame detection summary, and stationary period records — ready for downstream analysis in any tool.

- **🔍 Full-Clip ID Audit** — Separate audit pass that produces a full-clip video and summary CSV solely for verifying tracking ID consistency, useful for tuning tracker parameters before the main analysis run.

- **🛠️ Tunable Parameters** — All key values (confidence, speed threshold, minimum stationary duration, slow factor, max frames for testing) are centralized in a single configuration cell for easy iteration.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│              Video file (.mov / .mp4 / .avi)                │
│           FPS · Resolution · Total frame count              │
└──────────────────────────┬──────────────────────────────────┘
                           │ cv2.VideoCapture (frame-by-frame)
┌──────────────────────────▼──────────────────────────────────┐
│                     DETECTION LAYER                         │
│                    YOLOv8 (yolov8n.pt)                      │
│   Class filter: person only (COCO class 0)                  │
│   Confidence threshold: configurable (default 0.35)         │
│   Output: bounding boxes [x1, y1, x2, y2] + confidence      │
└──────────────────────────┬──────────────────────────────────┘
                           │ model.track(persist=True)
┌──────────────────────────▼──────────────────────────────────┐
│                      TRACKING LAYER                         │
│           ByteTrack / BotSort multi-object tracker          │
│   Assigns and maintains unique track_id per person          │
│   Handles occlusion, re-entry, and crowded scenes           │
│   Output: track_id → bounding box mapping per frame         │
└──────────────────────────┬──────────────────────────────────┘
                           │ center displacement per frame
┌──────────────────────────▼──────────────────────────────────┐
│                    ANALYSIS LAYER                           │
│   calculate_center(): bbox → (cx, cy)                       │
│   calculate_speed(): Euclidean distance / Δtime             │
│   Speed unit: pixels/second                                 │
│   Stationary: speed ≤ STATIONARY_SPEED_THRESHOLD            │
│   Tracks stationary_start_time per ID                       │
│   Closes period when movement resumes or ID disappears       │
└──────────────────────────┬──────────────────────────────────┘
                           │ stationary_periods list
┌──────────────────────────▼──────────────────────────────────┐
│                      OUTPUT LAYER                           │
│  ┌──────────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ Annotated video  │  │  CSV export  │  │  Speed graph  │ │
│  │ (color-coded)    │  │  (3 files)   │  │  (PNG, 160DPI)│ │
│  └──────────────────┘  └──────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Workflow

### Step-by-step pipeline

**Step 1 — Load & inspect video**
OpenCV reads the video file and extracts metadata: FPS, resolution, total frame count, and duration. A sample frame is displayed for sanity check.

**Step 2 — Single-frame detection preview**
YOLO runs on one frame to verify detection quality before committing to a full-clip run. Bounding boxes and confidence scores are visualized.

**Step 3 — ID Audit pass (optional but recommended)**
A separate full-clip pass assigns tracking IDs and exports an annotated video + CSV summary. Use this to confirm that the same person keeps the same ID throughout, especially in crowded or occluded scenes. Adjust tracker or confidence if IDs flip unexpectedly.

**Step 4 — Main analysis pass**
Frame-by-frame loop:
- YOLO detects and tracks all persons
- For each active ID: compute center → compute speed → classify as STILL or MOVING
- If transitioning from moving → still: record `stationary_start_time`
- If transitioning from still → moving: close the stationary period and log it
- If ID disappears for > `MAX_LOST_SECONDS`: force-close any open period

**Step 5 — Write annotated video**
Each frame is annotated with colored bounding boxes, status labels, and speed values, then written to an MP4 at the configured slow-motion factor.

**Step 6 — Export data**
Three CSV files are saved: full frame-by-frame tracking, detection summary per frame, and stationary period records.

**Step 7 — Visualize**
Speed-over-time graphs are generated per tracking ID, showing the threshold line and shaded stationary windows. Graphs are saved as PNG and displayed inline.

---

## 🧰 Technology Stack

### AI / Computer Vision

| Library | Role |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Person detection and multi-object tracking |
| ByteTrack | Default tracker for stationary analysis |
| BotSort | Alternative tracker (better for re-identification) |
| OpenCV (`cv2`) | Video I/O, frame processing, annotation rendering |

### Data & Analysis

| Library | Role |
|---|---|
| NumPy | Array operations for bounding box math |
| pandas | DataFrame construction and CSV export |
| Matplotlib | Speed timeline graphs with stationary overlays |
| Python `math` | Euclidean distance via `math.hypot` |

### Environment

| Tool | Role |
|---|---|
| Jupyter Notebook | Step-by-step interactive execution |
| Python 3.8+ | Runtime |
| pathlib | Cross-platform file path handling |

---

## 📁 Project Structure

```
stationary_analysis/
├── stationary_yolo_step_by_step.ipynb   # Main analysis notebook (18 cells)
├── entrance.mov                          # Input video (example footage)
├── yolov8n.pt                            # YOLOv8 nano model weights
└── outputs_notebook/                     # All generated outputs
    ├── id_tracking_full_clip.mp4         # Audit video: bounding boxes + IDs only
    ├── id_tracking_audit_frame_by_frame.csv   # Audit: bbox + ID per frame
    ├── id_tracking_audit_id_summary.csv       # Audit: first/last frame per ID
    ├── annotated_tracking_full_clip_slow.mp4  # Main output: color-coded analysis video
    ├── tracking_frame_by_frame.csv            # Full tracking data: speed + status per frame
    ├── frame_detection_summary.csv            # Per-frame person count + active IDs
    ├── stationary_periods.csv                 # All stationary events with duration
    └── stationary_speed_timeline.png          # Speed graph per tracking ID
```

### Key files explained

| File | Description |
|---|---|
| `stationary_yolo_step_by_step.ipynb` | The complete analysis pipeline, split into 18 inspectable cells |
| `yolov8n.pt` | Lightweight YOLOv8 model. Replace with `yolov8s.pt` for higher accuracy |
| `tracking_frame_by_frame.csv` | Raw tracking data; contains `track_id`, `center_x/y`, `speed_px_per_sec`, `is_stationary` |
| `stationary_periods.csv` | One row per stationary event: `track_id`, `start_time_sec`, `end_time_sec`, `duration_sec` |
| `stationary_speed_timeline.png` | Visual summary of movement behavior per person |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stationary-yolo-analysis.git
cd stationary-yolo-analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install ultralytics opencv-python matplotlib pandas numpy
```

> **Note:** `ultralytics` automatically installs PyTorch. If you have a GPU, install the appropriate CUDA-compatible PyTorch version first for significantly faster processing.

### 4. Download or provide a YOLOv8 model

The notebook uses `yolov8n.pt` (nano — fast and lightweight). It will auto-download on first run via `ultralytics`. To manually place it:

```bash
# The model will auto-download when YOLO() is called.
# Or download manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook stationary_yolo_step_by_step.ipynb
```

---

## 📋 Usage

### Basic run

1. Open `stationary_yolo_step_by_step.ipynb` in Jupyter
2. In **Cell 3**, set your video path:
   ```python
   VIDEO_PATH = find_existing_path(["your_video.mp4"])
   ```
3. Adjust parameters as needed (see table below)
4. Run all cells in order (Cell 1 → Cell 18)
5. Find outputs in the `outputs_notebook/` folder

### Configuration parameters (Cell 3)

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE` | `0.35` | Minimum detection confidence (0–1). Increase to reduce false positives. |
| `STATIONARY_SPEED_THRESHOLD` | `100.0` | Speed in px/sec below which a person is classified as stationary. |
| `MIN_STATIONARY_SECONDS` | `0.5` | Minimum duration to record a stationary period (filters micro-pauses). |
| `MAX_LOST_SECONDS` | `0.5` | Seconds an ID can be missing before its open period is force-closed. |
| `SLOW_FACTOR` | `1.0` | Output video playback factor. `2.0` = half speed; `1.0` = real time. |
| `TRACKING_AUDIT_TRACKER` | `botsort.yaml` | Tracker used in the audit pass. Try `bytetrack.yaml` if IDs flip frequently. |
| `MAX_FRAMES_FOR_TEST` | `None` | Limit frames for quick testing (e.g. `300`). Set to `None` for full clip. |

### Quick test run

To validate the pipeline on a short segment before committing to a full-clip run:

```python
# In Cell 3, change:
MAX_FRAMES_FOR_TEST = 300  # Process only first ~10 seconds at 30 FPS
```

Set back to `None` for full analysis.

---

## 🤖 AI Model Details

### Detection model

| Property | Value |
|---|---|
| Model | YOLOv8 nano (`yolov8n.pt`) |
| Training dataset | COCO (80 classes) |
| Target class | `person` (class ID 0) |
| Input | Individual video frames (BGR) |
| Output | Bounding boxes `[x1, y1, x2, y2]` + confidence scores |
| Inference mode | `model.track()` with `persist=True` |

### Tracking algorithm

| Property | Value |
|---|---|
| Primary tracker | ByteTrack (`bytetrack.yaml`) |
| Alternative tracker | BotSort (`botsort.yaml`) |
| Track state | Maintained across frames via `persist=True` |
| ID assignment | Unique integer per person, consistent for the video duration |

### Stationary classification

```
speed = Euclidean_distance(center_t, center_{t-1}) / Δtime

if speed <= STATIONARY_SPEED_THRESHOLD:
    → STILL  🟢
else:
    → MOVING 🔴
```

The center is computed from the midpoint of the bounding box (`(x1+x2)/2`, `(y1+y2)/2`), which is more stable than using raw corner coordinates.

---

## 📊 Output Data Schema

### `stationary_periods.csv`

The primary output. One row per detected stationary event.

| Column | Type | Description |
|---|---|---|
| `track_id` | int | Unique person identifier |
| `start_time_sec` | float | Time when stationary period began |
| `end_time_sec` | float | Time when stationary period ended |
| `duration_sec` | float | Total duration in seconds |

### `tracking_frame_by_frame.csv`

Full tracking data for every frame where a person was detected.

| Column | Type | Description |
|---|---|---|
| `frame_index` | int | Frame number (0-indexed) |
| `time_sec` | float | Timestamp in seconds |
| `track_id` | int | Person identifier |
| `center_x` | float | Bounding box center X (pixels) |
| `center_y` | float | Bounding box center Y (pixels) |
| `speed_px_per_sec` | float | Movement speed in pixels/second |
| `is_stationary` | bool | True if speed ≤ threshold |

### `frame_detection_summary.csv`

Per-frame detection summary for audit and debugging.

| Column | Type | Description |
|---|---|---|
| `frame_index` | int | Frame number |
| `time_sec` | float | Timestamp in seconds |
| `detected_person_count` | int | Number of persons detected |
| `track_ids` | str | Comma-separated active tracking IDs |

---

## 🖼️ Example Results

### Annotated video frame

```
┌────────────────────────────────────────────────────┐
│ Frame 142/890 | Time 4.73s                         │
│                                                    │
│  ┌─────────────────┐      ┌──────────────────┐     │
│  │ ID 1 | STILL    │      │ ID 2 | MOVING    │     │
│  │ 12.3 px/s       │      │ 284.7 px/s       │     │
│  │  (GREEN BOX)    │      │   (RED BOX)      │     │
│  └─────────────────┘      └──────────────────┘     │
└────────────────────────────────────────────────────┘
```

### Speed timeline graph

```
Speed (px/s)
400 │                    ████
300 │              ██████    ████
200 │─ ─ ─ ─ ─ threshold ─ ─ ─ ─ ─ ─
100 │  ░░░░░░░░░░              ░░░░░░   ← stationary periods
  0 └──────────────────────────────────► Time (sec)
        0     5    10    15    20
```

### Stationary summary example

| track_id | total_stationary_duration_sec |
|---|---|
| 1 | 14.32 |
| 2 | 3.07 |
| 3 | 0.83 |

---

## ⚠️ Challenges & Limitations

**Bounding box jitter** — Even when a person is visibly still, YOLO bounding boxes fluctuate slightly between frames due to subtle pose changes, lighting, and compression artifacts. This causes non-zero speed readings for stationary people. The `STATIONARY_SPEED_THRESHOLD` parameter compensates for this, but requires tuning per video.

**ID switching in crowded scenes** — When multiple people are close together or occlude each other, the tracker may swap IDs. This creates artificially short stationary periods. Mitigation: use BotSort, increase confidence, or use a larger YOLO model.

**Reappearing persons** — If a person leaves the frame and returns, they typically receive a new tracking ID, fragmenting their behavioral record. This is a fundamental limitation of online tracking without re-identification (ReID).

**Fixed camera assumption** — The system assumes a static camera. Camera movement or shake will produce false speed readings for all tracked persons. Future work could incorporate background subtraction or homography compensation.

**Pixels-per-second as a proxy for real-world distance** — Speed is measured in pixels/second, not meters/second. The relationship between pixel displacement and real movement depends on camera angle, focal length, and scene depth. Calibration data would be needed for metric measurements.

---

## 🔧 Troubleshooting

**IDs keep switching between persons**
- Increase `CONFIDENCE` (e.g. 0.45 or 0.50)
- Switch `TRACKING_AUDIT_TRACKER` from `bytetrack.yaml` to `botsort.yaml`
- Use a larger model: replace `yolov8n.pt` with `yolov8s.pt` or `yolov8m.pt`

**Person is visually still but system reports MOVING**
- Increase `STATIONARY_SPEED_THRESHOLD` (try 30, 50, 80)
- The bounding box is likely jittering due to motion blur or compression

**Processing is too slow**
- Set `MAX_FRAMES_FOR_TEST = 150` to test on a short segment first
- Use `yolov8n.pt` (nano model) for fastest inference
- Run on GPU — install CUDA-compatible PyTorch before installing ultralytics

**Output MP4 cannot be opened**
- Do not interrupt the notebook while cells 9 or 10 are writing video
- The `video_writer.release()` call in the `finally` block must complete to finalize MP4 metadata

**No stationary periods found**
- Lower `STATIONARY_SPEED_THRESHOLD` (e.g. start with 100, try 150 or 200)
- Check `frame_detection_summary.csv` to confirm people are being detected
- Run cell 13 to visually inspect frames with zero detections

---

## 🚀 Future Improvements

- **GPU acceleration** — Add automatic CUDA/MPS device detection for 5–10× speedup on compatible hardware
- **Region of Interest (ROI) filtering** — Allow analysis to focus on a specific zone of the frame (e.g. a product shelf, a checkout counter)
- **Real-time streaming** — Adapt the pipeline to accept RTSP camera streams for live monitoring
- **Re-identification (ReID)** — Integrate a ReID model to handle persons who leave and re-enter the frame
- **Heatmap generation** — Visualize cumulative stationary duration as a spatial heatmap overlaid on the video frame
- **Multi-camera support** — Stitch together analysis across multiple synchronized camera feeds
- **Web dashboard** — Build a Streamlit or Gradio interface for no-code operation
- **Metric calibration** — Add camera calibration to convert pixel speed to real-world meters/second
- **Notification system** — Trigger alerts when a person has been stationary for longer than a configured threshold

---

## 🏆 Hackathon Context

### Problem Statement

Existing video surveillance infrastructure captures enormous amounts of behavioral data that is almost never analyzed. Operators can answer "is there a person?" but cannot answer "how long did that person stay?" without manual review or expensive hardware.

### Solution

A lightweight, open-source pipeline that runs on any laptop, requires only a video file as input, and produces structured behavioral data (CSV + annotated video + graphs) as output. No additional hardware, no cloud dependency, no database setup.

### Innovation

The key insight is treating **stationary duration** as a first-class metric derived from computer vision rather than from dedicated presence sensors. By combining:
1. Real-time person detection (YOLO)
2. Persistent identity tracking (ByteTrack)
3. Frame-to-frame displacement analysis

...the system extracts a behavioral signal that previously required purpose-built infrastructure.

### Business Impact

- Retail: Understand which product zones generate the longest dwell time
- Hospitality: Optimize seating layout based on where guests naturally linger
- Operations: Detect bottlenecks where people bunch up and stop moving

### Social Impact

- Enables evidence-based public space design without invasive hardware
- Fully offline — no personal data leaves the analysis machine
- Open source — accessible to researchers, municipalities, and small businesses alike

---

## 👥 Contributors

| Name | Role |
|---|---|
| *(your name here)* | Computer Vision Engineer |
| *(teammate)* | Data Analyst |
| *(teammate)* | Project Lead |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 framework and pre-trained weights
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the multi-object tracking algorithm
- [OpenCV](https://opencv.org/) for the video I/O and annotation primitives
- COCO dataset contributors for enabling open-vocabulary person detection

---

*Built with 💡 curiosity and ⏱️ a lot of frame-by-frame analysis.*
