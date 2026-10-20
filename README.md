# Graduation-Project-CrowdControl

**AI-Based Crowd Monitoring Using YOLOv8 and Optimized BoT-SORT**

**Project Status:** Research and development — Completed through CELL 94.

## 1. Project Overview

CrowdControl is an AI-based crowd monitoring system designed to detect, track, and analyze pedestrian movement in densely crowded environments.

The project focuses on handling occlusion, maintaining tracking identities, analyzing crowd density, monitoring pedestrian flow, and detecting potentially crowded areas.

### Current Pipeline

```text
Video Input
    ↓
YOLOv8m — Person Detection
    ↓
Optimized Batched BoT-SORT — Multi-Object Tracking
    ↓
Pedestrian Trajectories
    ↓
12-Zone Crowd Analysis
    ↓
Occupancy, Density, Inflow and Outflow
    ↓
Crowd-Level Alerts
```

**Note:** GNN-based crowd prediction is planned for a future development stage and has not yet been implemented.

## 2. Dataset – MOT20

This project uses the MOT20 dataset from MOTChallenge for crowded pedestrian detection and multi-object tracking.

**Download Dataset:**  
https://motchallenge.net/data/MOT20/

Due to its large size, the dataset is not included in this GitHub repository.

Download the dataset separately and follow the project setup instructions.

### Dataset Split

| Split | Dataset | Purpose |
|---|---|---|
| TRAIN | MOT20-01, MOT20-02, MOT20-03 | Detector training |
| VAL | MOT20-05, frames 1–1657 | Model selection, optimization, calibration and validation |
| TEST | MOT20-05, frames 1658–3315 | Reserved for final evaluation |

The latest optimization experiments (CELLS 92–94) use VAL only and do not access TEST.

**Evaluation limitation:** TEST frames were examined during earlier development. Therefore, although subsequent development keeps them excluded, they are not a completely unseen independent test set.

## 3. Person Detection — YOLOv8m

The current selected detector is YOLOv8m, using project-trained weights.

Current inference configuration:

- Model: YOLOv8m
- Input resolution: 1280 pixels
- Target class: Person
- Confidence threshold: 0.01
- IoU threshold: 0.70

The detector generates bounding boxes and confidence scores that are passed to the tracking system.

## 4. Multi-Object Tracking — Optimized BoT-SORT

The project uses an optimized implementation of BoT-SORT.

### Current Configuration

- Tracker: BoT-SORT
- Global Motion Compensation (GMC): Disabled
- Re-Identification (ReID): Disabled
- Kalman Filter: Optimized using batched updates
- Tracking IDs and bounding boxes are maintained across video frames

### Optimization

A batched Kalman Filter implementation was developed to reduce tracking overhead while preserving the original tracking results.

The optimized tracker was verified against the original implementation across the complete validation sequence.

Tracking outputs were identical in the verification experiments.

## 5. Trajectory Analysis

Each tracked pedestrian is represented by a unique tracking ID and a sequence of bounding boxes over time.

The bottom-center point of each bounding box is used to represent the pedestrian's position on the ground.

These positions are used for zone assignment and movement analysis.

## 6. Zone-Based Crowd Monitoring

The camera view is divided into a **3 × 4 grid**, creating 12 zones.

The monitoring system calculates:

- Pedestrian occupancy
- Crowd density
- Zone inflow and outflow
- Stable transitions between zones
- Crowd-level alerts

### Stable Zone Transitions

A pedestrian must remain in a new zone for **5 consecutive observed frames** before a zone transition is confirmed.

This reduces false transitions caused by tracking fluctuations near zone boundaries.

### Crowd Alerts

Crowd thresholds are calibrated using validation data.

The current alert outputs are:

- `NORMAL`
- `HIGH_CROWD`

Crowd alerts are based on zone occupancy and estimated floor coverage.

## 7. Latest Verified Results — CELL 94

The full monitoring pipeline was benchmarked on MOT20-05 validation frames 1–1657.

### Performance

| Metric | Result |
|---|---:|
| Frames processed | 1,657 |
| Integrated processing speed | **22.49 FPS** |
| Total processing time | 73.66 seconds |
| Average image decoding | 10.64 ms/frame |
| Average YOLO inference | 16.79 ms/frame |
| Average tracking | 15.61 ms/frame |
| Average monitoring | 0.83 ms/frame |

The benchmark includes image decoding, live YOLO inference, optimized BoT-SORT tracking, zone analysis, flow, and alerts.

It does not include video rendering or GNN processing.

### Monitoring Results

| Metric | Result |
|---|---:|
| Tracked observations | 348,720 |
| Confirmed zone transitions | 1,319 |
| Zone-window records | 804 |
| NORMAL alerts | 674 |
| HIGH_CROWD alerts | 130 |

### Validation

The following checks passed:

- Complete validation sequence processed
- No duplicate tracking IDs within individual frames
- Correct number of zone-window records
- All 12 zones represented per window
- Correct final partial window
- Total zone inflow equals total zone outflow
- Tracking outputs identical to CELL 93
- Monitoring outputs identical to CELL 93

**CELL 94 Status: PASS**

These checks establish consistency with the previous validated implementation, not independent ground-truth accuracy.

## 8. Current Optimizations

The following improvements have been implemented and verified:

1. Disabled unnecessary Global Motion Compensation.
2. Disabled ReID for the current tracking configuration.
3. Implemented batched Kalman Filter updates.
4. Saved the optimized tracker as a reusable Python module.
5. Integrated optimized tracking with the monitoring pipeline.
6. Verified identical tracking and monitoring outputs.
7. Benchmarked the complete pipeline on validation data.

## 9. Current Limitations

- Integrated processing speed is 22.49 FPS, below the target of 25 FPS.
- Dense crowds and occlusions remain challenging for tracking.
- Crowd monitoring currently uses predefined camera-view zones.
- The GNN prediction stage has not yet been implemented.
- The pipeline has not yet been packaged as a production deployment.
- A separate unseen evaluation sequence would provide stronger independent evidence.

## 10. Next Development Steps

The next planned work begins at **CELL 95**.

Planned tasks:

1. Investigate CPU image prefetching to reduce decoding overhead.
2. Improve integrated processing speed toward 25 FPS.
3. Preserve identical tracking and monitoring outputs during optimization.
4. Continue validation before moving to later project stages.

## 11. Development Environment

The project is developed using:

- Python
- Google Colab
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Google Drive for dataset and experiment storage

The research notebook contains the training, tracking, monitoring, and benchmarking workflow.

Dataset files, trained weights, and large generated experiment outputs are stored separately from this repository.

## 12. Project Status

**Completed:** Person detection, optimized multi-object tracking, trajectory extraction, zone-based crowd monitoring, alert generation, and integrated validation benchmarking.

**In progress:** Performance optimization.

**Future work:** GNN-based crowd prediction and further system integration.

---

**Graduation Project — CrowdControl**

*Current documented milestone: CELL 94.*
