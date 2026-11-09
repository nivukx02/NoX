# NoX High-End Native Motion Capture System Design

**Status:** Approved architecture entering specification review
**Target branch:** `main`
**Application:** NoX
**Package:** `com.nivukx.nox`
**Primary runtime:** Android
**Native engine:** C++20 + Android NDK + CMake
**Managed orchestration:** Kotlin
**UI:** Jetpack Compose + Material 3 / liquid-glass visual system
**Capture contract:** Motion-only canonical output; camera pixels may be transient perception inputs but are not the canonical exported artifact

## 1. Objective

Upgrade NoX from its existing production-oriented motion-capture architecture into a high-end native motion-capture and 3D reconstruction system without deleting or downgrading any existing engine, capture path, reconstruction path, synchronization feature, UI capability, or session contract.

The upgrade must add real engineering depth through native C++ numerical engines, explicit geometry and uncertainty models, deterministic temporal state estimation, multi-camera calibration, synchronization, robust reconstruction, kinematic optimization, face/eye/hand geometry integration, scene reconstruction, lighting telemetry, GPU-aware scheduling, production data contracts, regression fixtures, and operator tooling.

The target is a 300k+ line production-class repository over the full implementation program. Code volume is an outcome of complete capabilities and testable implementation, not a substitute for functionality. Existing code is preserved unless a focused extraction is required to reduce coupling without changing behavior.

No new AI model or AI-dependent core capability is introduced by this upgrade. The authoritative high-end computation path is deterministic/native C++20 numerical and geometric processing, with Kotlin responsible for lifecycle, orchestration, persistence, UI state and platform APIs. Existing optional perception integrations and their engines remain available and are not deleted merely because this upgrade adds the native path.

## 2. Non-Negotiable Constraints

1. Work on `main`.
2. Do not delete existing production engines.
3. Do not silently remove or weaken existing features.
4. Do not replace real implementations with placeholders.
5. Do not inflate files with duplicate or dead code merely to reach a line count.
6. No new AI model dependency is permitted for the authoritative capture/reconstruction path.
7. Native C++20 is authoritative for hot-path numerical geometry, filtering, triangulation, optimization, and reconstruction.
8. Kotlin remains authoritative for Android lifecycle, orchestration, persistence, permissions, UI state, and platform service integration.
9. Existing Kotlin engines remain connected through explicit interfaces and are progressively wrapped by stable native contracts where appropriate.
10. All runtime fallbacks must be explicit, observable, and evidence-producing.
11. No fake metric scale, fake synchronization, fake depth certainty, fake reconstruction confidence, or fake frame-rate claims.
12. Motion-only canonical export remains preserved: no ordinary video recording feature is introduced into NoX.
13. Camera frames may exist transiently in RAM for perception/reconstruction only and must not become the canonical recorded artifact.
14. Every feature added to production code requires tests before implementation.
15. Completion claims require fresh verification evidence.

## 3. Current Repository Baseline

Current repository organization:

- `app`
- `core:model`
- `core:sync`
- `core:network`
- `core:capture`
- `core:calibration`
- `core:reconstruction`
- `core:ai`

The repository currently contains approximately 315 code files and about 3.4 MB of code. Existing high-value engines include capture, multi-camera synchronization, calibration drift monitoring, motion capture, face tracking, eye tracking, hand tracking, temporal state estimation, monocular depth fusion, multi-view fusion, triangulation, reprojection consistency, bundle adjustment, kinematic constraints, world-space motion consistency, occlusion recovery, dense facial geometry, lighting telemetry, and a liquid-glass-oriented Motion Studio UI.

Those systems are inputs to this design, not disposable scaffolding.

## 4. Native Engine Architecture

Add a new module:

`core/native-engine`

Expected structure:

```text
core/native-engine/
  build.gradle.kts
  CMakeLists.txt
  src/main/cpp/
    core/
      math/
      geometry/
      geometry2d/
      geometry3d/
      rotations/
      interpolation/
      robust/
      filtering/
      statistics/
    calibration/
    synchronization/
    perception/
    tracking/
    triangulation/
    reconstruction/
    optimization/
    kinematics/
    face/
    eyes/
    hands/
    lighting/
    scene/
    runtime/
    memory/
    diagnostics/
    export/
    ffi/
  src/main/java/com/nivukx/nox/core/nativeengine/
    NativeMotionEngine.kt
    NativeEngineConfig.kt
    NativeEngineDiagnostics.kt
```

Native implementation requirements:

- C++20.
- RAII ownership.
- Explicit thread-safety contracts.
- Deterministic numerical kernels where practical.
- SIMD-friendly contiguous structures.
- SoA/AoS choice documented for every hot-path structure.
- Memory pools for high-frequency observations.
- Lock-free or bounded queues where a queue is actually required.
- No per-frame uncontrolled heap allocation in the hot path.
- Stable JNI boundaries carrying compact typed buffers rather than object graphs.
- Versioned ABI contract.
- Native diagnostics exposed to Kotlin.
- Debug validation builds with assertions and invariant checks.
- Release builds with equivalent logical safeguards but optimized numerical paths.

## 5. Engine Integration Model

The native engine is not a replacement application layer.

The production data flow is:

```text
Android Camera2 / Sensor APIs
       |
       v
existing CaptureController
       |
       +---- existing Kotlin evidence engines
       |
       +---- NativeMotionEngine JNI boundary
                    |
                    +-- calibration
                    +-- synchronization
                    +-- temporal filtering
                    +-- robust geometry
                    +-- reconstruction
                    +-- optimization
                    +-- diagnostics
       |
       v
existing session/reconstruction contracts
       |
       v
NoX Motion Studio
       |
       v
motion-only export
```

Every native result must include provenance, timestamp, confidence, uncertainty and processing epoch where relevant.

## 6. Motion Observation Model

The canonical native observation model supports:

- camera identifier
- frame identifier
- monotonic timestamp
- sensor timestamp
- exposure interval
- rolling-shutter row timing
- 2D joint position
- 2D confidence
- 3D estimate
- 3D covariance or bounded uncertainty
- velocity
- acceleration
- jerk
- orientation
- angular velocity
- angular acceleration
- occlusion state
- visibility state
- cross-view correspondence confidence
- reprojection residual
- kinematic residual
- camera-motion penalty
- synchronization penalty
- contact evidence
- motion regime

The model is extensible and versioned.

## 7. Feature Program

### Feature 01: Native runtime foundation

Add the native engine module, JNI contract, ABI/version handshake, native diagnostics, deterministic configuration, allocator policy, worker scheduler and typed observation buffers.

Preserve all existing module APIs.

### Feature 02: High-rate capture ingestion

Upgrade the capture bridge to support bounded high-rate ingestion of camera metadata, IMU packets, timestamps and transient image-derived observations.

Add timestamp normalization, monotonic ordering, duplicate rejection, gap detection and capture-session epochs.

### Feature 03: Calibration engine

Implement high-end calibration support for:

- camera intrinsics
- focal lengths
- principal point
- radial distortion
- tangential distortion
- Brown-Conrady coefficients
- extrinsic rotation
- extrinsic translation
- camera rig graph
- calibration confidence
- calibration residuals
- drift estimates
- calibration validity intervals

Support assisted target calibration and natural-feature/SfM-style calibration without making specialist calibration hardware mandatory.

### Feature 04: Multi-camera synchronization

Implement a native synchronization estimator supporting:

- monotonic-clock offset
- sensor-clock offset
- latency
- jitter
- clock drift
- audio cross-correlation input
- visual event correlation
- sub-frame alignment
- per-camera temporal confidence

Synchronization becomes a reconstruction input rather than a networking-only concern.

### Feature 05: Body motion tracking

Preserve the existing body perception engines and connect them to the native temporal tracker.

Track joint positions, velocity, acceleration, jerk, confidence and identity with bounded prediction and explicit occlusion state.

### Feature 06: Face and facial-surface reconstruction

Preserve the existing face detector, contour and dense face mesh paths.

Add native geometric fusion for:

- head pose
- facial surface landmarks
- temporal stabilization
- face shape alignment
- multi-view triangulation
- metric face geometry
- facial motion residuals

Do not treat image-space face-mesh Z as metric depth.

### Feature 07: Eye and gaze reconstruction

Preserve existing eye engines.

Add native binocular geometry:

- eye centers
- iris centers
- gaze rays
- head-relative gaze
- world-relative gaze
- eyelid state
- gaze confidence
- binocular consistency
- temporal gaze stabilization

### Feature 08: Hand and finger reconstruction

Preserve existing hand-tracking engines.

Add native kinematic and multi-view fusion for:

- wrist pose
- palm geometry
- finger joint locations
- finger rotations
- hand-body association
- occlusion recovery
- temporal consistency

### Feature 09: Temporal motion state engine

Replace ad-hoc smoothing with a formal state estimator that preserves micro-motion while controlling detector noise.

State variables include position, velocity, acceleration, jerk, orientation, angular velocity and uncertainty.

Use bounded robust innovation gating and covariance-aware prediction.

### Feature 10: IMU/vision fusion

Fuse gyroscope and accelerometer evidence with visual motion observations.

Requirements:

- bias estimation
- gravity alignment
- orientation propagation
- angular-motion residual
- visual/IMU confidence weighting
- dropout handling
- drift detection
- reset/recovery semantics

Magnetometer data remains optional and never becomes a mandatory dependency.

### Feature 11: Multi-camera subject association

Associate detections across cameras using:

- geometric compatibility
- temporal proximity
- appearance-neutral geometry constraints
- motion continuity
- identity tracks
- camera visibility graphs
- confidence-weighted assignment

The authoritative association path must not require an AI identity model.

### Feature 12: Robust 3D triangulation

Implement weighted multi-view triangulation using:

- calibrated projection matrices
- distortion correction
- confidence weights
- robust loss functions
- cheirality checks
- baseline/conditioning checks
- reprojection residuals
- uncertainty propagation

Reject degenerate geometry explicitly.

### Feature 13: Metric-scale recovery

Metric scale can be established using measured rig geometry, known scene dimensions, stable camera baselines, supported depth evidence or body-shape constraints.

Every scale source is recorded with provenance.

No relative-depth estimate is relabeled as metric.

### Feature 14: Dense scene reconstruction

Add native scene-point and surface reconstruction capable of:

- depth fusion
- point-map accumulation
- temporal consistency
- static/dynamic separation
- surface confidence
- normal estimation
- mesh generation
- topology stability
- hole recovery

The human reconstruction and static scene reconstruction remain separable but share a world coordinate system.

### Feature 15: Kinematic optimization

Implement native optimization over:

- bone-length consistency
- joint limits
- parent-child rotations
- root trajectory
- contact constraints
- temporal smoothness
- reprojection error
- multi-view agreement
- face/head coupling
- hand-wrist coupling

Use robust objective functions and explicit convergence diagnostics.

### Feature 16: Occlusion and missing-data recovery

Implement bounded recovery for partially or temporarily invisible joints.

Recovery is limited by uncertainty budgets and temporal state. Long unsupported gaps must remain uncertain rather than hallucinated.

### Feature 17: Lighting/environment capture

Preserve existing lighting telemetry.

Extend the native representation with:

- temporal illumination statistics
- directional cues
- scene exposure consistency
- light-change events
- environment confidence

The output supports downstream VFX/avatar workflows without claiming physically exhaustive light-field capture.

### Feature 18: Performance/runtime governor

Add native performance governance for:

- frame budget
- inference budget
- reconstruction budget
- memory pressure
- thermal state
- queue depth
- dropped observation rate
- native worker utilization
- synchronization health

The governor changes internal scheduling and workload allocation but must not silently change the semantic contract.

### Feature 19: Professional Motion Studio UX

Preserve existing UI.

Upgrade the studio into clearly separated surfaces:

- Capture
- Rig
- Calibration
- Sync
- Live Reconstruction
- Face/Eye
- Hands
- Scene
- Timeline
- Diagnostics
- Export
- Advanced settings

Liquid glass is applied to navigation, controls and telemetry surfaces. The reconstruction viewport remains visually clear and is not buried under decorative translucent layers.

### Feature 20: Evidence, replay and release validation

Every session can optionally store a deterministic motion-evidence trace containing:

- configuration
- calibration hash
- synchronization result
- engine versions
- native ABI version
- frame/joint statistics
- reprojection residual statistics
- dropped data
- thermal events
- uncertainty statistics
- solver convergence
- export validation

Add deterministic replay fixtures so the native pipeline can be regression-tested from recorded structured observations without requiring a live camera.

## 8. Data and Export Contracts

Canonical output remains motion-only.

Required export capabilities:

- structured session archive
- skeleton trajectory
- joint rotations
- root motion
- face landmarks/mesh references
- eye/gaze state
- hand trajectories
- confidence
- uncertainty
- timestamps
- camera calibration bundle
- synchronization bundle
- scene geometry reference
- lighting telemetry
- engine provenance
- schema version

Supported downstream target formats may include internal binary, JSON diagnostic views, glTF-compatible skeletal representations and production interchange adapters, provided the existing motion-only contract remains intact.

## 9. Testing Strategy

Tests are written before implementation for every feature.

Native tests cover:

- vector/matrix/quaternion operations
- robust statistics
- clock offset estimation
- calibration
- distortion correction
- projection
- triangulation
- reprojection
- covariance propagation
- temporal state estimation
- IMU integration
- multi-view association
- kinematic constraints
- optimization convergence
- occlusion recovery
- mesh consistency
- export determinism

Android tests cover JNI boundary correctness, lifecycle behavior, configuration propagation, UI state, capture-session transitions and regression compatibility.

Replay suites cover:

- single-camera
- two-camera
- four-camera
- six-camera
- moving-camera
- partial occlusion
- rapid motion
- low-light telemetry
- clock drift
- dropped frames
- sensor gaps
- calibration perturbations

## 10. Performance Targets

Targets are measured, not assumed.

The native architecture must support:

- bounded hot-path allocations
- deterministic frame budgeting
- concurrent capture and reconstruction stages
- parallel multi-camera processing
- SIMD-friendly math
- optional GPU compute integration points
- native ring buffers
- predictable memory pressure behavior

Device-specific capability limits must be represented explicitly.

## 11. Failure Policy

The system must never silently convert a low-confidence result into a high-confidence result.

Examples:

- poor baseline -> triangulation confidence falls
- camera timing drift -> temporal confidence falls
- missing camera -> multi-view contribution is removed explicitly
- occlusion -> uncertainty expands
- failed calibration -> session cannot claim calibrated multi-view quality
- native engine error -> controlled session error and recoverable state
- memory pressure -> bounded queue shedding with evidence

## 12. UI/UX Quality Rules

The operator must be able to answer these questions without opening a deep settings panel:

1. Are all cameras connected?
2. Are timestamps synchronized?
3. Is calibration valid?
4. Is the subject tracked?
5. Is 3D reconstruction healthy?
6. What is the current reconstruction confidence?
7. Are observations being dropped?
8. Is the system thermally constrained?
9. What will be exported?

Advanced controls expose full numerical customization without forcing casual users through engineering telemetry.

## 13. Dependency Policy

No new AI model dependency is introduced.

Existing perception dependencies remain intact where the current application already relies on them. The native high-end engine is authoritative for numerical motion/reconstruction computation and does not depend on an AI inference model for correctness.

Native dependencies should remain limited to Android NDK/CMake/platform capabilities unless a dependency has a concrete, tested production need.

## 14. Compatibility and Preservation

The following are explicitly preserved:

- existing Camera2 capture flow
- existing multi-camera networking
- existing synchronization engines
- existing calibration engines
- existing body tracking
- existing face tracking
- existing eye tracking
- existing hand tracking
- existing dense face mesh
- existing temporal motion estimation
- existing monocular depth fusion
- existing multi-view fusion
- existing triangulation
- existing bundle adjustment
- existing reprojection consistency
- existing kinematic constraints
- existing world-space motion consistency
- existing motion-only session contract
- existing Motion Studio UI
- existing liquid-glass UI components
- existing tests and build structure

The upgrade adds and integrates capability. It does not downgrade the existing architecture to accommodate the native engine.

## 15. Implementation Order

The implementation follows the approved 20-stage sequence:

1. Native runtime foundation
2. Capture ingestion
3. Calibration
4. Synchronization
5. Body motion tracking
6. Face reconstruction
7. Eye/gaze reconstruction
8. Hand reconstruction
9. Temporal motion state engine
10. IMU/vision fusion
11. Multi-camera association
12. Robust 3D triangulation
13. Metric scale
14. Dense scene reconstruction
15. Kinematic optimization
16. Occlusion recovery
17. Lighting/environment capture
18. Runtime governor
19. Motion Studio UX
20. Evidence/replay/release validation

Every stage is independently testable and committed.

## 16. Definition of Production-Ready

A feature is considered production-ready only when:

- implementation exists
- tests exist
- failing test was observed before implementation
- tests pass after implementation
- relevant existing tests still pass
- performance is measured where applicable
- failure paths are explicit
- native/Kotlin boundaries are versioned
- diagnostics exist for non-trivial runtime behavior
- no placeholder implementation remains
- no undocumented silent downgrade exists
- exported data validates against its schema

## 17. Success Criteria

The complete upgrade must provide a real native high-end motion-capture engine that can:

- coordinate multiple consumer cameras
- synchronize them at sub-frame precision where evidence supports it
- calibrate camera geometry and lens distortion
- reconstruct 3D body motion
- reconstruct facial geometry
- reconstruct eyes and gaze
- reconstruct hands and fingers
- combine vision and IMU evidence
- recover motion through bounded occlusion
- estimate metric scene/body geometry where evidence permits
- reconstruct a temporally stable 3D representation
- preserve confidence and uncertainty
- provide professional operator diagnostics
- export production motion data
- remain motion-only at the canonical session level
- integrate with all existing NoX engines without deleting them

The target is a continuously extensible 300k+ LOC production architecture built from real functionality, native geometry, mathematics, test coverage, diagnostics and operator tooling rather than artificial code inflation.
