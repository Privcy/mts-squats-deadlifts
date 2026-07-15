# Multivariate Time-Series Classification for Squat Variations and Stiff-Leg Deadlift Form Analysis using Pose Estimation

This repository contains the official implementation, experimental architecture, and data preprocessing configurations for the BodyMTS classification pipeline presented by Marichella Salve M. Mendoza (Mapua University, 2026). 

This research introduces a framework designed to automate human exercise form assessment by converting visual joint data into structured temporal signs. With the fusion of 2D Human Pose Estimation and Multivariate Time Series Classification (MTSC), the pipeline accurately evaluates strength and conditioning (S&C) performance, discerning proper executions from specific, injury-inducing biomechanical deviations.

This study evaluates **three 2D pose estimation models** — OpenPose, MediaPipe, and RTMPose — paired with a **ROCKET-based multivariate time-series classifier** to analyze form correctness across five compound exercise variations: **Back Squat, Front Squat, Dumbbell Squat, Goblet Squat, and Stiff-Leg Deadlift**.
 
Gym-related injuries are a growing concern. One study reported a 29.9% injury rate among gym members in Saudi Arabia, with squats and deadlifts among the highest-risk compound movements when performed incorrectly. Existing pose-estimation-based feedback systems typically rely on frame-by-frame analysis, which fails to capture the temporal dynamics of a full repetition. BodyMTS addresses this gap by converting joint keypoint sequences into structured multivariate time series and classifying full repetitions with ROCKET + RidgeClassifierCV, rather than judging isolated frames.
 
**Key findings:**
- **RTMPose** achieved the highest overall test accuracy (**93.89%**), outperforming OpenPose (87.59%) and MediaPipe (84.84%)
- RTMPose showed superior robustness to severe self-occlusion during the Stiff-Leg Deadlift (**89.19%** accuracy), thanks to its top-down pipeline and SimCC architecture
- Spatial Z-score normalization successfully decoupled classification from camera distance/setup
- The full pipeline runs with **sub-75ms inference latency**, supporting real-time use on standard edge devices
---
## Table of Contents
 
- [Overview](#overview)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Live XAI Dashboard](#live-ai-xai-dashboard)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
- [Dataset](#dataset)
- [Results](#results)
- [Citation](#citation)
- [Author](#author)
- [License](#license)
---
 
## Overview
 
Improper exercise form during resistance training is a leading cause of preventable musculoskeletal injury, yet manual form assessment by a coach or trainer isn't always available. **BodyMTS** addresses this by treating exercise form classification as a **multivariate time-series problem**: joint coordinates extracted frame-by-frame from video are treated as multiple parallel signals evolving over time, then classified using time-series-native models rather than generic image classifiers.
 
Key components of the approach:
 
- **2D pose estimation** (OpenPose / MediaPipe / RTMPose) to extract (x, y) joint keypoints from raw video, with irrelevant background information removed via semantic noise reduction
- **Three-stage normalization** — spatial (torso-length scaling), temporal (interpolation/padding to a uniform sequence length), and Z-score standardization — to decouple movement shape from participant size and camera distance
- **Conversion to multivariate time series** using `sktime`-compatible formats
- **ROCKET** (RandOm Convolutional KErnel Transform) for high-dimensional temporal feature extraction
- **RidgeClassifierCV** (L2-regularized) for final classification
- An **explainable AI (XAI) dashboard** that visualizes which joints and frames drove a given classification decision
The pipeline is evaluated across five exercise variations:
 
- **Back Squat (BS)**
- **Front Squat (FS)**
- **Dumbbell Squat (DS)**
- **Goblet Squat (GS)**
- **Stiff-Leg Deadlift (SD)** — including detection of restricted hip extension and spinal rounding under severe self-occlusion
---
 
## System Architecture & Workflow
 
![BodyMTS Conceptual Framework](figs/cframework.png)
 
**Fig. 1 — Conceptual Framework of the implemented BodyMTS processing pipeline.** The architecture maps the full sequential flow: raw dynamic video capture → localized 2D pose estimation (OpenPose/MediaPipe/RTMPose) → `sktime` conversion → 80:10:10 train/validation/test partitioning → high-dimensional ROCKET feature transform → RidgeClassifierCV evaluation.
 
## Live AI (XAI) Dashboard
 
![Live XAI Dashboard - Incorrect Stiff-Leg Deadlift Form Error](figs/livexai_isd.png)
 
**Fig. 2 — Live XAI Dashboard capturing an incorrect Stiff-Leg Dumbbell Deadlift (SD) execution**, using the MediaPipe feature extractor. The lower graph plots vertical displacement trajectory (Y-axis) across frames. A restricted hip extension curve (Features 26 and 27) reveals structural hinge constraints and spinal rounding, allowing the ROCKET engine to correctly flag the movement as an **INCORRECT (Form Error)** execution.
 
---
 
## Installation
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```
**Core dependencies** (confirm against your `requirements.txt`):
- `sktime`
- `scikit-learn`
- `mediapipe`, `openpose`, and/or `rtmpose` (depending on the extractor used)
- `numpy`, `pandas`
- `matplotlib` / `plotly` (for the XAI dashboard)

## Usage
 
The classification pipeline is run as a Python module via `tsc.rocket`, configured with the `tsc/rocket_config` file. Exercise variations covered: **BS** (Back Squat), **DS** (Dumbbell Squat), **FS** (Front Squat), **GS** (Goblet Squat), **SD** (Stiff-Leg Deadlift).
 
```bash
# Run the ROCKET + RidgeClassifierCV pipeline
python -m tsc.rocket --rocket_config tsc/rocket_config
```
 
To save the run output to a timestamped log file (PowerShell):
 
```powershell
python -m tsc.rocket --rocket_config tsc/rocket_config 2>&1 | tee "results_$((Get-Date).ToString('yyyy-MM-dd_HH-mm')).txt"
```
 
> Note: `tee` here is the PowerShell cmdlet (`Tee-Object` alias), not the Unix `tee`. On macOS/Linux, use the standard `tee` with a `date`-based filename instead, e.g.:
> ```bash
> python -m tsc.rocket --rocket_config tsc/rocket_config 2>&1 | tee "results_$(date +%Y-%m-%d_%H-%M).txt"
> ```
 

## Pipeline Details
 
| Stage | Description |
|---|---|
| **Video Segmentation** | Raw footage segmented into individual repetition clips using LosslessCut |
| **Pose Estimation** | Frame-level 2D keypoint (x, y) extraction via OpenPose, MediaPipe, or RTMPose; background/non-skeletal data discarded |
| **Spatial Normalization** | Keypoints divided by torso length (Euclidean distance between mid-shoulder and mid-hip) for person-invariance |
| **Temporal Normalization** | Long sequences down-sampled via linear interpolation; short sequences receive final-value padding to a uniform length |
| **Z-Score Standardization** | Feature-wise standardization ( Z = (x − μ) / σ ) to decouple movement shape from camera distance |
| **Time-Series Conversion** | Normalized joint coordinates combined into a continuous multivariate time-series structure (`sktime`-compatible) |
| **Data Split** | 80% train / 10% validation / 10% test |
| **Feature Transform** | ROCKET random convolutional kernels extract high-dimensional temporal features from each channel |
| **Classification** | RidgeClassifierCV (L2-regularized) on ROCKET-transformed features |
| **Explainability** | Live dashboard visualizes per-feature trajectories (e.g., hip/knee Y-coordinates) driving each prediction |

## Dataset
 
- **1,905 repetition videos** collected from **14 participants** (11 male, 3 female)
- Exercises: Back Squat, Front Squat, Dumbbell Squat, Goblet Squat, and Stiff-Leg Deadlift
- Split **80% train / 10% validation / 10% test**
- Ground-truth correct/incorrect form labels were independently verified by **two licensed physical therapists**, yielding a **Cohen's Kappa of 0.694** (85% raw agreement) — indicating substantial inter-annotator reliability beyond chance

## Results
 
### Test Accuracy per Model and Exercise Variation
 
| Exercise Variation | OpenPose | MediaPipe | RTMPose |
|---|---|---|---|
| Back Squat (BS) | 91.30% | 82.86% | **100%** |
| Dumbbell Squat (DS) | 90.00% | 90.00% | **95.00%** |
| Front Squat (FS) | 77.50% | 80.00% | **95.00%** |
| Goblet Squat (GS) | **92.68%** | 90.24% | 90.24% |
| Stiff-Leg Deadlift (SD) | 86.49% | 81.08% | **89.19%** |
| **Average** | 87.59% | 84.84% | **93.89%** |
 
### Classification Inference Latency (ms)
 
| Exercise Variation | OpenPose | MediaPipe | RTMPose |
|---|---|---|---|
| Back Squat (BS) | 16.11ms | 20.15ms | 24.24ms |
| Dumbbell Squat (DS) | 72.50ms | 73.34ms | 73.86ms |
| Front Squat (FS) | 21.03ms | 23.85ms | 23.08ms |
| Goblet Squat (GS) | 21.64ms | 21.38ms | 21.47ms |
| Stiff-Leg Deadlift (SD) | 28.89ms | 28.76ms | 30.80ms |
 
**Discussion:** RTMPose achieved the highest overall test accuracy (93.89%), outperforming OpenPose (87.59%) and MediaPipe (84.84%). The Stiff-Leg Deadlift was the most difficult exercise to track overall, since the arms occlude the knee and thigh regions during the bottom phase of the movement — MediaPipe in particular exhibited significant tracking instability during occlusion frames (dropping to 81.08% accuracy). RTMPose's top-down pipeline and SimCC-based keypoint localization made it notably more robust to this occlusion, maintaining 89.19% accuracy on the SD exercise.
 
The full ROCKET + RidgeClassifierCV pipeline maintained **sub-75ms inference latency** across every model–exercise combination (16.11ms–73.86ms), confirming its viability for real-time feedback on standard recording setups.
 
## Future Work
 
- Deploy the RTMPose–ROCKET pipeline to mobile/edge devices via **ONNX** or **TensorFlow Lite** for live feedback
- Incorporate additional kinematic features such as joint velocity and acceleration
- Extend the framework to unilateral exercises (e.g., lunges, single-leg deadlifts)
## Citation
 
If you use this work, please cite:
 
```bibtex
@article{mendoza2026bodymts,
  author  = {Mendoza, Marichella Salve M. and Tomas, John Paul Q.},
  title   = {Multivariate Time-Series Classification for Squat Variations and Stiff-Leg Deadlift Form Analysis using Pose Estimation},
  school  = {Mapúa University},
  address = {Makati City, Philippines},
  note    = {BodyMTS pipeline}
}
```
## Author
 
**Marichella Salve M. Mendoza** — marichellamendoza@mymail.mapua.edu.ph
**John Paul Q. Tomas** — jpqtomas@mapua.edu.ph
School of Information Technology, Mapúa University, Makati City, Philippines
