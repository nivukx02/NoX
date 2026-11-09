# Reconstruction Capacity and Evidence Contract

## Purpose

NoX processes multi-camera evidence through bounded composition and specialist engines. Bounded input is a safety property: it controls worst-case work and memory. It must not be mistaken for proof that the complete source scene was processed, or that a physical device can sustain a requested camera or actor count.

## Current composition limits

- The frontier reconstruction composition root forwards at most 128 camera entries into its federated motion-fusion stage.

When distinct eligible landmark observations exceed the frame budget, NoX reserves per-camera quotas in descending camera-authority order and then reallocates unused quota. Normal-sized frames retain the legacy first-seen ordering. This prevents a camera-major input collection from consuming the full landmark budget before other views can contribute.
- It forwards at most 64 actor entries into the same stage.
- Landmark and surface-observation limits are configurable through `FrontierReconstructionCoordinator.Config`; the defaults are 65,536 landmarks and 24,000 surface observations per frame.
- The distributed rig coordinator separately ranks camera nodes and selects an authoritative subset for expensive reconstruction. Its selected subset may be smaller than the supplied federation.

Camera and actor capacity is measured by unique identifiers using a bounded cap-plus-one selector, not by raw collection size. Duplicate IDs therefore do not create a false capacity failure. When a source exceeds one of the composition limits, `ReconstructionFrameCapacityGuard` adds a stream-specific warning and the frame reports `MeasurementFailure.HIGH_UNCERTAINTY` unless a more specific failure already has precedence. This prevents capacity truncation from being silently represented as an unqualified success. Camera and actor overflow counts are exact; for landmark and surface streams the guard stops as soon as it can prove that at least one eligible item exceeds the cap.

## Memory and frame-budget behavior

- Landmark admission uses stable first-seen de-duplication and stops after retaining the configured maximum plus enough evidence to prove overflow. It avoids materializing a complete filtered copy or a complete distinct copy of the incoming frame.
- Surface admission keeps the top configured number of eligible observations by confidence × visibility in a bounded priority queue. Selection retains O(k) observations for a configured limit k, rather than sorting and allocating a second full-frame list. Equal-score candidates preserve original input order.
- Actor detail requests are de-duplicated lazily and capped at 64 entries; distinct-camera summaries stop at 128 entries. These are retained-work limits, not claims that excess source inputs become processed evidence.
- Runtime configuration now bounds frame history to 4,096 records, landmark and surface capacities to 250,000 entries per frame, and surface prediction horizon to 120 frames. Default per-frame capacities remain 65,536 landmarks and 24,000 surface observations.
- Overflow counts saturate at cap + 1 because the guard requires only proof of overflow, avoiding a second full scan solely to count excess items. Overflow is visible as a warning and prevents an unqualified-success state.


## Scaling 50 actors and 100+ cameras

A 100-camera scene should be represented as a federation of independently timestamped camera nodes, not as a claim that one Android device can open 100 simultaneous physical camera streams. Android concurrent-camera availability is device-dependent. The app's actual camera sessions, transport bandwidth, timestamp alignment, GPU/NPU throughput, memory pressure, and thermal behavior must be measured on the target hardware.

The existing 64-actor contract includes the requested 50-actor target. The 128-camera input bound can represent a 100-camera federation, subject to the downstream coordinator's authoritative view selection and each camera's evidence quality. A submitted count alone does not establish that all cameras are connected, synchronized, calibrated, or contributing useful geometry.

## Accuracy and reference comparison

No source-code change proves sub-millimetre physical accuracy, universal superiority over infrared optical systems, or a fixed performance level across phones, lenses, environments, and actor densities. A qualified `REFERENCE_LOWER_ERROR` result from the active-optical comparison must keep the health authority blocked even when legacy telemetry aliases such as `REFERENCE_WORSE` are also supported. Those claims require measured, independently referenced results under matched capture conditions.

The implementation should continue to prioritize:

1. Camera calibration and held-out reprojection residuals, rather than visual confidence alone.
2. Exposure-time synchronization evidence and explicit timing uncertainty.
3. Multi-view triangulation with observation uncertainty and degenerate-baseline rejection.
4. Cross-camera identity continuity and occlusion-aware association.
5. Runtime capacity, memory, transport, and thermal headroom without hiding stream truncation.
6. P95/P99 physical error against an independent reference, with reference uncertainty and repeat sessions documented.

## Engineering references

- [OpenCV camera calibration tutorial](https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html) describes intrinsic/extrinsic calibration and reprojection error as a calibration diagnostic.
- [OpenCV calibration and 3D reconstruction API](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html) documents triangulation in homogeneous coordinates and the underlying geometric operations.
- [Jiang et al., *Probabilistic Triangulation for Uncalibrated Multi-View 3D Human Pose Estimation*, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html) motivates treating camera-pose uncertainty probabilistically instead of assuming calibration is exact.
- [Usman et al., *MetaPose: Fast 3D Pose From Multiple Views Without 3D Supervision*, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Usman_MetaPose_Fast_3D_Pose_From_Multiple_Views_Without_3D_Supervision_CVPR_2022_paper.pdf) discusses multi-view uncertainty and assumes synchronized test-time views.
- [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose) and its [3D body-keypoint dataset documentation](https://mmpose.readthedocs.io/en/latest/dataset_zoo/3d_body_keypoint.html) provide established model/dataset baselines such as Human3.6M and CMU Panoptic for comparative evaluation.

These references support the engineering principles above; they do not validate NoX's physical accuracy or throughput. No app build, device benchmark, or unit-test run is implied by this document.


Surface-fusion configuration rejects non-finite scales and decay values before they enter the spatial index. At patch-capacity saturation, eviction is batched at 1% of the current patch count to amortize the global utility sort while preserving the configured hard maximum.
