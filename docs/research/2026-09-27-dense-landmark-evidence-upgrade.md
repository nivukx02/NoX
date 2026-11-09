# Dense Landmark Evidence Upgrade

Date: 2026-09-27

## Research basis

Current markerless multi-view research increasingly treats dense surface evidence as a first-class object instead of collapsing perception immediately to a sparse joint list.

### MAMMA, CVPR 2026

MAMMA uses dense 2D surface landmarks together with explicit uncertainty, visibility and contact predictions. These signals improve person correspondence under occlusion and close interaction before calibrated multi-view body fitting.

Source: https://openaccess.thecvf.com/content/CVPR2026/html/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.html

### ReGen4D / complex 4D markerless capture evaluation

The 2026 evaluation work emphasizes synchronized multi-view capture, calibration, realistic multi-person occlusions, identity exchange and temporal coherence, showing that realistic interaction exposes weaknesses in current markerless systems.

Source: https://openaccess.thecvf.com/content/CVPR2026W/ReGen4D/html/Park_A_Dataset_and_Evaluation_for_Complex_4D_Markerless_Human_Motion_CVPRW_2026_paper.html

### Dense interleaved spatiotemporal fusion

DenseWarper-style multi-view methods combine temporal information with sparse/interleaved view observations rather than treating every frame as an isolated spatial solve.

Source: https://arxiv.org/html/2605.14525

## Implemented changes

### DenseLandmarkEvidenceFusionEngine

Added a bounded production evidence layer with:

- per-landmark and per-camera timestamp validation;
- NaN/Inf fail-closed handling;
- stale evidence rejection;
- duplicate same-camera suppression;
- robust median and Huber reweighting;
- confidence, uncertainty and visibility fusion;
- segmentation consistency and rolling-shutter quality weighting;
- cross-view agreement using residual percentiles;
- bounded short-horizon temporal prediction;
- temporal stability scoring;
- contact probability state with hysteresis;
- per-camera telemetry;
- bounded landmark and camera state memory;
- bounded diagnostic history;
- explicit authoritative and hard-failure semantics.

The engine does not claim any physical millimeter accuracy by itself. It creates the evidence contract required by downstream calibrated reconstruction and precision gates.

### PerceptionPipeline integration

Pose observations are now converted into dense landmark evidence in the existing perception pipeline without removing or replacing streaming pose inference, accurate pose correction, face tracking, eye tracking, hand tracking, object tracking, subject association, temporal motion fusion, face/eye fusion, or runtime governance.

The perception result exposes dense landmark evidence for downstream production quality logic.

### Operator telemetry integration

The existing ViewModel and Studio contract already exposes dense-landmark fields. The engine result now also exposes cross-view evidence availability, preventing a single-camera node from being misclassified as having poor cross-view agreement when cross-view evidence is simply not present yet.


### Uncertainty-aware spatiotemporal bundle adjustment

The temporal calibration stage now consumes dense evidence quality as part of its robust observation weighting. Each dynamic observation can carry uncertainty, visibility, temporal stability, cross-view agreement, contact support, rolling-shutter fraction and temporal-alignment confidence. Poor evidence contributes less to offset estimation while retaining the existing robust reprojection objective.

This prevents low-quality views from disproportionately shaping the camera-time solution and keeps the temporal calibration result tied to the same evidence quality model used by dense landmark fusion.

## Accuracy qualification policy

The dense evidence layer improves robustness and observability, but practical sub-2-mm qualification still requires a controlled physical benchmark with explicit reference uncertainty, calibrated geometry, synchronization measurement, region and visibility cohorts, P95/P99 statistics, and failure/dropout accounting.

Architecture alone is not an empirical accuracy certificate.

## Build policy for this tranche

No Android application build was started.
No Python simulation was run.
Source-level contract verification was performed by fetching the affected files from main and checking the integrated symbols and data flow.