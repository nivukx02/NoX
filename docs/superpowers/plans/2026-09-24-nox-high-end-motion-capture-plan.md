# NoX High-End Motion Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade NoX toward a production-grade sensor-fused motion-capture platform while preserving all existing behavior and adding real geometric, temporal, calibration, session, performance, and UI capabilities.

**Architecture:** Keep the existing module boundaries. Strengthen contracts between perception, timing, calibration, reconstruction, and UI rather than collapsing them into giant files.

**Tech Stack:** Kotlin, Android Camera2, Jetpack Compose, existing NoX core modules, Figma design workflow, Exa research, Scite literature validation, GitHub source control.

## Iteration 01 — 2026-09-24

- Fixed a source-integrity failure in `UltraMotionStudio.kt`: the file contained three copies of its helper declaration set plus generated-text corruption. One complete implementation remains; no functional UI section was intentionally removed.
- Added capture timing provenance to `LandmarkObservation`, including frame dimensions, rolling-shutter skew, frame duration, exposure duration, and capture-timing confidence.
- Propagated Camera2 timing metadata through `PoseDetectionEngine` and `PerceptionPipeline` into both recording and calibration paths.
- Added `RollingShutterEvidenceEngine`, a bounded timing-evidence model that converts measured skew and row position into signed row offset, frame-relative rolling-shutter fraction, confidence scaling, and warnings.
- Integrated rolling-shutter evidence into multi-view observation weighting and reprojection consistency.
- Extended the eye-tracking path with a bounded timing-quality scale so readout skew, exposure duration, and capture timing confidence reduce gaze trust and increase stabilization when capture timing is degraded.
- Added the same timing provenance and bounded quality attenuation to tracked face landmarks and dense 468-point face mesh geometry.
- Replaced the motion-engine `rollingShutterRisk = 0.0` stub with a measured risk function driven by Camera2 skew/frame-duration ratio and timing confidence, while retaining existing user-controlled compensation tuning.
- Corrected detector-to-sensor row handling by using the existing `ImageCoordinateTransform`, so row-time compensation remains valid under 0°, 90°, 180°, and 270° rotations.
- Added a real MediaPipe multi-person Pose Landmarker front end using the documented `numPoses` capability and world-landmark-capable result contract; configured capacity now matches the project's 32-subject customization ceiling.
- Added `MultiActorPoseAssociationEngine` with global assignment, detector-ID evidence, IoU/center/velocity/skeleton continuity, occlusion prediction, stable actor IDs, and nanosecond-timestamped subpixel micro-motion traces.
- Added a real MediaPipe multi-person Pose Landmarker front end using `numPoses`, `ImageProcessingOptions.setRotationDegrees`, normalized landmark visibility/presence, and asynchronous executor isolation; the model is provisioned as a pinned build input rather than silently assuming the asset exists.
- Connected multi-actor association to the perception graph and exposed runtime actor capacity, identity strictness, occlusion recovery, cross-camera association, micro-motion sensitivity/retention, and sub-frame timing controls in the operator UI.
- Kept the Android application build intentionally unstarted; static source integrity remains the validation boundary for this iteration.
- Integrated multi-actor pose and actor tracks into `PerceptionPipeline`, including lifecycle cleanup and operator-facing subject-count controls.
- Promoted signed row-time offsets into multi-view observations and added the offset to velocity-based reprojection timing, converting rolling-shutter handling from confidence-only attenuation into an actual spatiotemporal projection correction.
- Added regression tests for UI source integrity, timing metadata propagation, and rolling-shutter evidence behavior.
- Performed standalone Kotlin compilation and executable assertions for the new rolling-shutter evidence engine. Android app build/run was intentionally not started.
- Research basis includes Android Camera2 rolling-shutter metadata, Android adaptive Compose layout guidance, Apple Liquid Glass hierarchy guidance, and published work on spatiotemporal bundle adjustment and rolling-shutter-aware dynamic reconstruction.
- Ruling: code volume is not used as a proxy for production quality. Artificially expanding files to 10k+ lines would violate the project's maintainability and production-readiness constraints; capability, testability, observability, and integration are the governing metrics.


**Spec:** `docs/superpowers/specs/2026-09-24-nox-high-end-motion-capture-design.md`

## Iteration 02 — Multi-evidence 3D reconstruction and operator controls

- Added `MultiEvidence3DReconstructionEngine` with robust consensus optimization across primary, multi-view, monocular-depth, temporal, inertial and kinematic evidence families.
- Added correlation-group caps so correlated estimates do not masquerade as independent measurements.
- Added uncertainty inflation from hypothesis disagreement, explicit outlier counts, source-diversity diagnostics, deterministic contributor attribution, and bounded acceptance rules.
- Integrated the fusion layer into `SkeletonReconstructionEngine` after kinematic constraints and bundle refinement without removing the existing triangulation, temporal filtering, Kalman filtering, or bundle-refinement stages.
- Added runtime evidence-fusion tuning and persisted operator controls for fusion strength, disagreement tolerance, and minimum source diversity.
- Added regression tests covering outlier suppression, disagreement-driven uncertainty, primary-only fallback, and tuning validation.
- Added the same controls to the professional Reconstruction settings surface.
- Figma composition was attempted against the existing MotionField Fusion Preview design file but remains blocked by the connected Starter-plan MCP call limit; no false completion claim is recorded.
- Android app build/run was intentionally not started.

## Iteration 03 — Dense surface normals and mesh quality

- Added `MeshNormalEstimator` with area-weighted vertex normals, degenerate-triangle rejection, index safety, and deterministic zero-normal fallback.
- Extended `ReconstructionMesh` with optional per-vertex normals without breaking existing callers.
- Extended `DenseMeshQuality` with validated normal coverage telemetry.
- Integrated normal estimation into both `SparseTsdfFusionEngine` and `VoxelSurfaceMesher`, so dense surface confidence now reflects geometric support for normals instead of only vertex/triangle counts.
- Added regression tests for planar consistency, invalid triangles, and finite fallback normals.
- Android build/run remains intentionally unstarted.

## Iteration 04 — Adaptive motion regimes and diagnostic completeness

- Added `MotionRegimeAdaptiveController` to classify steady, micro-detail, precision, normal, fast-action, extreme-action, camera-dominant and occluded-recovery regimes.
- Integrated bounded regime multipliers into `MotionCaptureEngine` for innovation gates, prediction horizon, velocity/acceleration response and jitter suppression.
- Fixed a diagnostics propagation defect where common-mode motion, differential motion and camera-compensated motion were calculated but not stored in `Quality`.
- Added regression tests covering fast action, camera-dominant motion and occlusion recovery.
- No existing motion estimator stage was removed; adaptation sits on top of the existing temporal estimator and user tuning.

## Iteration 05 — Adaptive-regime control and dense-mesh telemetry

- Added an operator-controlled adaptive motion regime strength to `MotionCaptureTuning`.
- Wired persistence and professional settings for adaptive-regime strength so automatic motion behavior can be softened or disabled without removing the controller.
- Integrated adaptive policy strength into innovation gating, prediction horizon, velocity/acceleration response, jitter suppression, and reacquisition.
- Exposed dense-mesh normal coverage as runtime telemetry after TSDF/voxel surface extraction.
- Build/run remains intentionally unstarted; source-level structural verification was performed on the changed files.

## Global Constraints

- Work on `main`.
- Do not start the Android app build in this pass.
- Preserve existing motion-only product scope.
- Add production functionality, not repeated placeholder code.
- Use TDD for new behavior and regression fixes.
- Verify before claiming completion.

## Review Focus

- Timestamp regression and discontinuity.
- Unsynchronized multi-camera observations.
- Rolling-shutter correction under fast motion.
- Calibration drift and thermal/vibration degradation.
- Occlusion and subject identity changes.
- UI state divergence between ViewModel and operator surface.

### Task 1: Spatiotemporal observation contract

**Files:**
- Create: `core/model/src/main/java/com/nivukx/nox/core/model/SpatiotemporalObservation.kt`
- Create: `core/model/src/test/java/com/nivukx/nox/core/model/SpatiotemporalObservationTest.kt`

**Purpose:** Represent a camera observation with frame timestamp, row timing, temporal offset, sensor confidence, rolling-shutter confidence, and observation provenance without changing existing `CameraObservation` semantics.

- [x] Write failing tests for timestamp ordering, bounded confidence, and deterministic derived sample time.
- [x] Verify the tests fail for the missing type.
- [x] Implement immutable validated data structures.
- [x] Verify targeted tests pass in isolated Kotlin compilation.
- [x] Commit `feat(model): add spatiotemporal observation contract`.

### Task 2: Sub-frame motion trajectory alignment

**Files:**
- Modify: `core/sync/src/main/java/com/nivukx/nox/core/sync/MultiCameraTemporalAlignmentEngine.kt`
- Create: `core/sync/src/main/java/com/nivukx/nox/core/sync/SubframeMotionAlignmentEngine.kt`
- Create: `core/sync/src/test/java/com/nivukx/nox/core/sync/SubframeMotionAlignmentEngineTest.kt`

**Purpose:** Extend the existing multi-camera temporal alignment stack with a bounded normalized cross-correlation solver that estimates signed sub-frame offset from corresponding motion trajectories. The solver remains a calibration-stage tool; it does not overwrite authoritative sensor timestamps.

- [x] Write failing tests for zero offset, signed offset, noisy correspondence, and insufficient evidence.
- [x] Verify the tests fail because the new solver is absent.
- [x] Implement bounded correlation, quadratic peak refinement, confidence, and a minimum-evidence gate.
- [x] Propagate accepted alignment evidence into the existing temporal-alignment telemetry contract.
- [x] Verify green with isolated Kotlin compilation after correcting the synthetic signal fixture.
- [x] Commit `feat(sync): add sub-frame motion trajectory alignment`.

### Task 3: Rolling-shutter-aware reconstruction bridge

**Files:**
- Modify: `core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/RollingShutterCompensator.kt`
- Modify: `core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/TriangulationEngine.kt`
- Create: `core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/RollingShutterCompensatorTest.kt`

**Purpose:** Carry temporal alignment and row timing into triangulation confidence rather than applying an isolated pixel correction.

- [ ] Add failing tests for row-zero/row-one timing and confidence degradation under excessive extrapolation.
- [ ] Verify red.
- [ ] Implement bounded corrected observations and propagate temporal uncertainty into effective weights.
- [ ] Verify green.
- [ ] Commit `feat(reconstruction): propagate temporal and rolling-shutter confidence`.

### Task 4: Spatiotemporal bundle-adjustment refinement

**Files:**
- Modify: `core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/BundleAdjustmentEngine.kt`
- Create: `core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/SpatiotemporalBundleAdjustmentEngine.kt`
- Create: `core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/SpatiotemporalBundleAdjustmentEngineTest.kt`

**Purpose:** Add a calibration/reconstruction-stage optimizer for camera time offsets and dynamic trajectory parameters while retaining the reference camera and existing bundle-adjustment engine.

- [x] Write regression tests for known synthetic temporal offset recovery and unobservable input.
- [x] Verify the missing-type Red condition in an isolated harness.
- [x] Implement bounded offset search, interpolation, robust reprojection cost, peak refinement, confidence, and multi-view integration.
- [ ] Verify against the full project source tree when build/test execution is permitted.
- [ ] Commit `feat(reconstruction): add spatiotemporal bundle adjustment`.

### Task 5: Calibration contract integration

**Files:**
- Modify: `core/calibration/src/main/java/com/nivukx/nox/core/calibration/CameraCalibrationWorkflowEngine.kt`
- Modify: `core/calibration/src/main/java/com/nivukx/nox/core/calibration/MultiCameraCalibrationDriftEngine.kt`
- Create: `core/calibration/src/test/java/com/nivukx/nox/core/calibration/TemporalCalibrationGateTest.kt`

**Purpose:** Make synchronization uncertainty and rolling-shutter health explicit calibration gates.

- [x] Write regression coverage for degraded temporal alignment confidence and excessive sub-frame offset.
- [x] Verify the new contract is intentionally absent from the pre-change production signature.
- [x] Integrate temporal confidence and sub-frame offset into the existing SYNC gate without weakening geometry thresholds.
- [ ] Verify against the full project source tree when build/test execution is permitted.
- [ ] Commit `feat(calibration): gate rig readiness on temporal health`.

### Task 5b: Camera-to-IMU temporal alignment

**Files:**
- Create: `core/sync/src/main/java/com/nivukx/nox/core/sync/ImuVisualMotionAlignmentEngine.kt`
- Create: `core/sync/src/main/java/com/nivukx/nox/core/sync/CameraImuSynchronizationEngine.kt`
- Extend: `core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/InertialFusionEngine.kt`

- [x] Add failing tests for signed IMU/visual offset recovery and low-energy rejection.
- [x] Implement bounded normalized correlation, sub-frame peak refinement, confidence, EWMA stabilization, and lifecycle reset.
- [x] Expose the estimate through inertial diagnostics.
- [ ] Verify against the full project source tree when build/test execution is permitted.

### Task 5c: Long-session temporal calibration drift

**Files:**
- Extend: `core/calibration/src/main/java/com/nivukx/nox/core/calibration/MultiCameraCalibrationDriftEngine.kt`
- Extend: `core/calibration/src/test/java/com/nivukx/nox/core/calibration/MultiCameraCalibrationDriftEngineTest.kt`

- [x] Add temporal-alignment confidence and sub-frame offset to per-camera drift evidence.
- [x] Fold temporal trust into rig trust and recalibration decisions.
- [x] Add regression coverage for temporal degradation without geometric drift.
- [ ] Verify against the full project source tree when build/test execution is permitted.

### Task 6a: Multi-scale temporal motion diagnostics

**Files:**
- Create: `core/ai/src/main/java/com/nivukx/nox/core/ai/TemporalMotionSpectrumAnalyzer.kt`
- Extend: `core/ai/src/main/java/com/nivukx/nox/core/ai/MotionCaptureEngine.kt`

- [x] Add spectral regression tests for periodic motion, high-frequency vibration, and insufficient history.
- [x] Implement a bounded frequency bank with periodicity, dominant cadence, and vibration-risk diagnostics.
- [x] Integrate bounded history into `MotionCaptureEngine.Quality` and capture readiness without replacing kinematic truth.
- [ ] Verify against the full project source tree when build/test execution is permitted.

### Task 6: Motion engine telemetry contract

**Files:**
- Modify: `core/ai/src/main/java/com/nivukx/nox/core/ai/MotionCaptureEngine.kt`
- Modify: `app/src/main/java/com/nivukx/nox/NoXViewModel.kt`
- Modify: `app/src/main/java/com/nivukx/nox/MainActivity.kt`
- Modify: `app/src/main/java/com/nivukx/nox/ui/UltraMotionStudio.kt`
- Extend: `core/ai/src/test/java/com/nivukx/nox/core/ai/MotionCaptureEngineTest.kt`

**Purpose:** Preserve the existing camera/common-mode/differential motion work while exposing a stable diagnostic view model and operator presentation.

- [ ] Add failing contract tests for telemetry bounds and reset behavior.
- [ ] Verify red.
- [ ] Implement only missing contract wiring.
- [ ] Verify green.
- [ ] Commit `feat(ui): formalize motion decomposition telemetry`.

### Task 7: Face/eye temporal robustness

**Files:**
- Modify: `core/ai/src/main/java/com/nivukx/nox/core/ai/FaceEyeConsistencyEngine.kt`
- Modify: `core/ai/src/main/java/com/nivukx/nox/core/ai/FaceMeshGeometryEngine.kt`
- Modify: `core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/Eye3DTemporalConsistencyEngine.kt`
- Create focused regression tests beside each modified subsystem.

**Purpose:** Preserve identity across temporary mesh loss, eye geometry dropouts, and camera topology changes.

- [x] Add a prolonged face-tracking gap regression.
- [x] Add explicit maximum track-gap reset and bounded history pruning.
- [ ] Verify against the full project source tree when build/test execution is permitted.
- [ ] Commit `feat(face): harden dense face and eye continuity`.

### Task 8: Operator customization expansion

**Files:**
- Modify: `core/ai/src/main/java/com/nivukx/nox/core/ai/MotionCaptureTuning.kt`
- Modify: `app/src/main/java/com/nivukx/nox/ui/ProfessionalMotionSettings.kt`
- Modify: `app/src/main/java/com/nivukx/nox/NoXViewModel.kt`

**Purpose:** Add high-end motion-capture presets for handheld, locked-rig, fast-action, facial-detail, hand-detail, and creature motion without bypassing safety/uncertainty gates.

- [x] Extend customization tests for sub-frame alignment strictness and rolling-shutter compensation bounds.
- [x] Add bounded settings, persistence, Figma controls, and connect sub-frame strictness to reconstruction acceptance.
- [ ] Run full project verification when build/test execution is permitted.
- [ ] Commit `feat(settings): expand production capture customization`.

### Task 9: Figma operator-console design system

**Files:**
- Figma file: `NoX Motion Studio - High End UI System`

**Purpose:** Define actual visual tokens and operator surfaces for Capture, Calibration, Motion, Face/Eye, Geometry, Multi-view, Integrity, and Output.

- [ ] Create design tokens and component primitives.
- [ ] Create the primary Motion Studio screen at compact and expanded sizes.
- [ ] Map visual hierarchy to existing Compose components.
- [ ] Review layout and accessibility in Figma.
- [ ] Produce implementation-ready design context.

### Task 10: Compose UI implementation

**Files:**
- Modify: `app/src/main/java/com/nivukx/nox/ui/UltraMotionStudio.kt`
- Modify: `app/src/main/java/com/nivukx/nox/ui/GlassUi.kt`
- Modify: `app/src/main/java/com/nivukx/nox/ui/HighEndOperatorRail.kt`
- Modify: `app/src/main/java/com/nivukx/nox/ui/ProfessionalMotionSettings.kt`

**Purpose:** Implement the Figma operator system using project-native Compose components.

- [ ] Write UI state/semantics tests.
- [ ] Implement compact/medium/expanded layouts.
- [ ] Implement restrained liquid-glass control surfaces.
- [ ] Verify component contracts and visual structure statically.
- [ ] Commit `feat(ui): implement high-end motion studio system`.

### Task 11: Session/replay robustness

**Files:**
- Inspect and extend existing capture/network/model session classes.
- Create deterministic replay tests for timestamp ordering, partial chunk recovery, and schema compatibility.

- [x] Add regression coverage for preserving a complete newline-less NDJSON tail.
- [x] Implement JSON-aware tail validation so complete records are preserved instead of truncated.
- [ ] Verify against the full project source tree when build/test execution is permitted.
- [ ] Commit `feat(session): harden deterministic capture replay`.

### Task 12: Observability and engineering diagnostics

**Files:**
- Add bounded diagnostic exporters under existing diagnostics package.
- Integrate stage timings and quality trends into existing runtime telemetry without adding network dependencies to the capture path.

- [ ] Define diagnostic event schema.
- [ ] Add structured runtime counters and degradation reasons.
- [ ] Verify serialization and lifecycle cleanup.
- [ ] Commit `feat(diagnostics): add production capture telemetry`.

### Audit Note: Settings adapter mapping

The advanced settings adapter was explicitly audited after adding sub-frame and rolling-shutter controls. Missing title-to-field mappings were found and repaired so the UI sliders mutate the persisted `MotionCaptureCustomization` fields instead of remaining visual-only.

### Audit Note: Rolling-shutter regression repair

A source-level verification pass detected and repaired a misplaced function boundary introduced while adding correction strength. The compensator was re-audited for balanced delimiters and now exposes a bounded correction-strength policy with regression coverage.

### Task 13: Whole-system review

- [ ] Review all changed files against the spec.
- [ ] Run all available verification that does not start the Android app build.
- [ ] Inspect GitHub diff and status.
- [ ] Review Figma implementation correspondence.
- [ ] Record remaining unverified items honestly.
- [ ] Do not claim production readiness until device/build/hardware validation is available.
