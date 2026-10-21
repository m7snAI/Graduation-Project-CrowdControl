# Graduation-Project-CrowdControl

AI-based real-time crowd monitoring and congestion prediction, built as a graduation project at Taibah University (College of Computer Science and Engineering). The long-term target setting is the Prophet's Mosque (Al-Masjid an-Nabawi) in Madinah; development and evaluation use public crowd datasets, because no public footage of the mosque exists.

The system turns crowd video into zone-level statistics (how many people are in each area, how fast they move, where they flow) and will use a graph neural network to predict congestion before it happens.

## Pipeline

```mermaid
flowchart LR
    A[Video frames] --> B[Detection<br/>YOLO]
    B --> C[Tracking<br/>ByteTrack / BoT-SORT]
    C --> D[Trajectories<br/>foot points]
    D --> E[Zone table<br/>per zone, per second]
    E --> F[Zone graph<br/>zones = nodes]
    F --> G[GNN forecast]
    G --> H[Alerts and dashboard]
```

| Stage | Status |
|---|---|
| Detection | Done on MOT20 |
| Tracking | Done on MOT20 |
| Trajectories | Done on MOT20 (pixel coordinates) |
| Zone statistics and crowding levels | Done on MOT20 (3 × 4 grid) |
| Zone graph | Planned (zone-to-zone transition table already produced) |
| GNN forecasting | Planned |
| Dashboard | Planned |

An interactive walkthrough of every stage, with the models compared, the datasets used and the graph structure for each dataset, is in [`docs/pipeline-explorer.html`](docs/pipeline-explorer.html). Download the file and open it in a browser (GitHub shows HTML files as source code).

## Repository contents

| Path | What it is |
|---|---|
| `Code_Graduation-Project-CrowdControl.ipynb` | Colab notebook, version 4: detection, tracking, trajectories and zone analysis on MOT20 |
| `docs/pipeline-explorer.html` | Interactive stage-by-stage overview of the full pipeline |

The newer rebuild described in the 21 October 2026 progress report (YOLOv8m with tuned BoT-SORT, crowd alerts and a batched tracker reaching 22.49 FPS end to end) lives in a separate notebook that is not in this repository yet.

## Notebook overview

The notebook is organised as numbered cells that run top to bottom in Google Colab.

| Cells | Step |
|---|---|
| 0 | Configuration: paths, data split, model list, zone settings |
| 1–3 | Setup, MOT20 to YOLO format conversion, data leakage check |
| 4–7 | Train YOLOv8n/s/m and YOLO11n, compare them, measure real FPS, difficulty analysis |
| 8–12 | Cache detections, tune ByteTrack and DeepSORT on VAL, final scoring on TEST with TrackEval |
| 13–14 | Trajectory CSV (centre and foot point), tracking and trajectory videos, report figures |
| 15–22 | Zone analysis: zone assignment, per-second statistics, accuracy against ground truth, figures, zone video |

### How to run

1. Download MOT20 and its labels from [motchallenge.net/data/MOT20](https://motchallenge.net/data/MOT20/). The dataset is too large for this repository.
2. Upload `MOT20.zip` and `MOT20Labels.zip` to `MyDrive/Crowd_AI_Project/datasets/` in Google Drive.
3. Open the notebook in Colab with a GPU runtime and use **Runtime → Run all**.

Every expensive step saves its results to `MyDrive/Crowd_AI_Project/runs_v4/` and is skipped on later runs, so running everything again is safe. To recompute one step on purpose, add its name to `FORCE` in the configuration cell (for example `FORCE = {"bytetrack"}`). Once all GPU steps are saved, the rest runs without a GPU.

## Data split

Only MOT20-01, 02, 03 and 05 have public ground truth.

| Split | Data | Used for |
|---|---|---|
| TRAIN | MOT20-01, 02, 03 | Fine-tuning the detectors |
| VAL | MOT20-05, frames 1–1657 | Choosing the detector, tuning trackers, learning zone thresholds |
| TEST | MOT20-05, frames 1658–3315 | Final scores only |

## Results in this notebook (MOT20-05)

### Detection (TEST half)

| Model | Params (M) | mAP50 | mAP50-95 | Recall | FPS (batch 1) |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 3.01 | 0.873 | 0.533 | 0.859 | 36.9 |
| YOLOv8s (used for tracking) | 11.14 | 0.871 | 0.541 | 0.865 | 37.2 |
| YOLOv8m | 25.86 | 0.879 | 0.533 | 0.879 | 34.6 |
| YOLO11n | 2.59 | 0.872 | 0.510 | 0.846 | 34.0 |

Images at 1280 px, up to 50 epochs, seed 42. The detector was selected on VAL mAP50-95.

### Tracking (TEST half, official TrackEval)

| Tracker | HOTA | MOTA | IDF1 | ID switches | End-to-end FPS |
|---|---:|---:|---:|---:|---:|
| ByteTrack, default | 51.0 | 63.2 | 57.6 | 3,105 | 15.6 |
| ByteTrack, tuned | 60.5 | 82.0 | 74.4 | 759 | 20.0 |
| DeepSORT, tuned | 53.5 | 79.1 | 64.7 | 1,969 | 1.2 |

Tuned ByteTrack settings (chosen on VAL): high 0.6, low 0.2, new track 0.7, buffer 30, match 0.7.

### Zone analysis

- 12 zones (3 × 4 grid), statistics every second (133 steps over the video).
- Per zone: count, density, floor coverage, inflow and outflow, speed, direction, waiting time, and a zone-to-zone transition table for the graph stage.
- Two crowding levels (LOW / MEDIUM / HIGH): one against each zone's own history, one for physical crowding that is comparable across zones. All thresholds are learned on VAL only.

## Planned datasets

| Dataset | Role |
|---|---|
| MOT20 | Detector and tracker training and scoring (in use) |
| CrowdTraj (2026) | Dense CCTV-style scenes with tracks and real-world coordinates; end-to-end demo and seconds-ahead forecasting |
| HAJJv2 + HAJJv2-CrowdCount | Hajj footage with per-second counts; tests how the detector holds up on Hajj-style crowds |
| ATC (Osaka) | 92 days of pedestrian trajectories in metres; forecasting 1, 3 and 5 minutes ahead |

Each dataset keeps its own zone graph; datasets are not merged into one graph.

## Privacy

The pipeline detects anonymous body boxes only. It does not process faces, track identities beyond the time a person is in view, or store personal data. Outputs are zone-level counts. Any use of real footage would need permission from the responsible authority and anonymisation.
