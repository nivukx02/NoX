# NoX High-End Native Motion Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Upgrade NoX into a production-grade native C++20 motion-capture and 3D reconstruction system by adding a deterministic native computation layer, integrating it with all existing engines, and preserving the motion-only session contract.

**Architecture:** core:native-engine becomes the authoritative numerical hot path while existing Kotlin modules remain responsible for Android lifecycle, platform capture, orchestration, and existing perception engines. Native results cross the JNI boundary through versioned compact typed buffers, carry timestamp/confidence/uncertainty/provenance, and flow into the existing session/reconstruction/UI contracts. The 20-stage sequence is dependency-aware: runtime foundations first, then capture/calibration/sync, perception and temporal estimation, reconstruction/optimization, runtime governance, UX, and finally deterministic evidence/replay validation.

**Tech Stack:** Android SDK 37, minSdk 30, Java 17, Kotlin/Compose, existing NoX modules, Android NDK, C++20, CMake, JNI, GoogleTest/CTest for native host tests, JUnit for Kotlin/JNI tests, existing Camera2/sensor APIs, existing perception integrations, existing Material 3/Liquid Glass UI system.

**Spec:** docs/superpowers/specs/2026-09-25-nox-high-end-native-motion-capture-design.md

## Global Constraints

- Work on main.
- Do not delete existing production engines.
- Do not silently remove or weaken existing features.
- Do not replace real implementations with placeholders.
- Do not inflate files with duplicate or dead code merely to reach a line count.
- No new AI model dependency is permitted for the authoritative capture/reconstruction path.
- Native C++20 is authoritative for hot-path numerical geometry, filtering, triangulation, optimization, and reconstruction.
- Kotlin remains authoritative for Android lifecycle, orchestration, persistence, permissions, UI state, and platform service integration.
- Existing Kotlin engines remain connected through explicit interfaces and are progressively wrapped by stable native contracts where appropriate.
- All runtime fallbacks must be explicit, observable, and evidence-producing.
- No fake metric scale, fake synchronization, fake depth certainty, fake reconstruction confidence, or fake frame-rate claims.
- Motion-only canonical export remains preserved: no ordinary video recording feature is introduced into NoX.
- Camera frames may exist transiently in RAM for perception/reconstruction only and must not become the canonical recorded artifact.
- Every feature added to production code requires tests before implementation.
- Completion claims require fresh verification evidence.
- Never run an app assemble/install/build workflow merely to demonstrate completion; validation uses native tests, targeted module unit tests, static checks, and replay fixtures unless the user explicitly changes this constraint.
- Every stage ends with a focused commit on main.
- The 300k+ LOC target is a consequence of real capability and test coverage, never generated padding.

## Review Focus

1. Uncalibrated or partially calibrated rigs: the system must refuse to claim calibrated multi-view reconstruction and must expose degraded confidence rather than invent geometry. Tests live in Task 03 and Task 20.
2. Clock drift and dropped frames: synchronization confidence must degrade, observations remain timestamped, and stale-frame reuse is forbidden. Tests live in Task 04 and Task 20.
3. Long or rapid occlusions: prediction may bridge bounded gaps while uncertainty expands; unsupported long gaps remain uncertain. Tests live in Task 16 and Task 20.
4. JNI lifetime and backpressure failures: native buffers remain bounded, ownership explicit, and queue shedding observable. Tests live in Task 01, Task 02, and Task 18.
5. Mixed legacy/native results: existing engines and the native path coexist without semantic drift in the motion-only contract. Tests live in Task 05, Task 12, Task 19, and Task 20.

---

## Task 01: Native Runtime Foundation

**Files**
- Create: core/native-engine/build.gradle.kts
- Create: core/native-engine/CMakeLists.txt
- Create: core/native-engine/src/main/cpp/ffi/native_engine_abi.h
- Create: core/native-engine/src/main/cpp/ffi/native_engine_abi.cpp
- Create: core/native-engine/src/main/cpp/runtime/native_engine_runtime.h
- Create: core/native-engine/src/main/cpp/runtime/native_engine_runtime.cpp
- Create: core/native-engine/src/main/cpp/memory/native_buffer_pool.h
- Create: core/native-engine/src/main/cpp/memory/native_buffer_pool.cpp
- Create: core/native-engine/src/main/cpp/diagnostics/native_engine_diagnostics.h
- Create: core/native-engine/src/main/cpp/diagnostics/native_engine_diagnostics.cpp
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/NativeEngineConfig.kt
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/NativeEngineDiagnostics.kt
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/NativeMotionEngine.kt
- Create: core/native-engine/src/test/java/com/nivukx/nox/core/nativeengine/NativeEngineConfigTest.kt
- Create: core/native-engine/src/main/cpp/tests/native_engine_runtime_test.cpp
- Modify: settings.gradle.kts
- Modify: root build.gradle.kts

**Interfaces**
- ABI function: uint32_t native_engine_abi_version().
- Runtime lifecycle: NativeRuntimeHandle native_runtime_create(const NativeRuntimeConfig*), bool native_runtime_start(NativeRuntimeHandle), void native_runtime_stop(NativeRuntimeHandle).
- Diagnostics snapshot: NativeDiagnostics native_runtime_diagnostics(NativeRuntimeHandle).
- Kotlin bridge: NativeMotionEngine.create(config), start(), stop(), diagnostics().
- NativeEngineConfig carries worker count, queue capacity, deterministic seed, validation level, and frame budget.

- [ ] Step 1: Write the failing native test for a stable ABI version, creation, start, stop, and destroy lifecycle.
- [ ] Step 2: Run the native test before implementation and capture the expected missing-symbol/module failure.
- [ ] Step 3: Implement the C++20 module with RAII ownership, deterministic configuration normalization, fixed-capacity pools, and explicit lifecycle state.
- [ ] Step 4: Implement JNI with typed arrays/POD structures only; no native object graph crosses the boundary.
- [ ] Step 5: Run CTest and the new Kotlin unit tests; expected result is green.
- [ ] Step 6: Commit: feat: add native motion engine runtime foundation.

---

## Task 02: High-Rate Capture Ingestion

**Files**
- Create: core/native-engine/src/main/cpp/core/observation/motion_observation.h
- Create: core/native-engine/src/main/cpp/core/observation/motion_observation.cpp
- Create: core/native-engine/src/main/cpp/capture/native_capture_ingestor.h
- Create: core/native-engine/src/main/cpp/capture/native_capture_ingestor.cpp
- Create: core/native-engine/src/main/cpp/capture/timestamp_normalizer.h
- Create: core/native-engine/src/main/cpp/capture/timestamp_normalizer.cpp
- Create: core/native-engine/src/main/cpp/capture/capture_gap_tracker.h
- Create: core/native-engine/src/main/cpp/capture/capture_gap_tracker.cpp
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/NativeCapturePacket.kt
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/NativeObservationBuffer.kt
- Create: core/native-engine/src/test/java/com/nivukx/nox/core/nativeengine/NativeCapturePacketTest.kt
- Create: core/native-engine/src/main/cpp/tests/capture_ingestion_test.cpp
- Modify: existing core:capture controller integration to forward normalized metadata/observations without removing its current capture paths.

**Interfaces**
- MotionObservation includes cameraId, frameId, monotonic timestamp, sensor timestamp, exposure interval, rolling-shutter row timing, 2D keypoints, confidence, visibility, and provenance.
- TimestampNormalizer::normalize(cameraTimestamp, sensorTimestamp) returns NormalizedTimestamp with monotonicNs, sensorNs, offsetNs, and epoch.
- CaptureIngestor::push(packet) returns PushResult with accepted, duplicate, gap, queueDepth, and epoch.

- [ ] Step 1: Write failing tests for ordering, duplicate rejection, epoch changes, and gap detection.
- [ ] Step 2: Run the focused CTest target and record failure.
- [ ] Step 3: Implement monotonic ordering, duplicate rejection, timestamp normalization, bounded queues, and session epochs.
- [ ] Step 4: Add JNI copy-on-submit ownership so Kotlin buffers are never retained past the bridge call.
- [ ] Step 5: Run focused native and Kotlin tests and verify dropped-observation diagnostics.
- [ ] Step 6: Commit: feat: add bounded native capture ingestion.

---

## Task 03: Native Camera Calibration

**Files**
- Create: core/native-engine/src/main/cpp/calibration/camera_intrinsics.h
- Create: core/native-engine/src/main/cpp/calibration/camera_intrinsics.cpp
- Create: core/native-engine/src/main/cpp/calibration/brown_conrady_distortion.h
- Create: core/native-engine/src/main/cpp/calibration/brown_conrady_distortion.cpp
- Create: core/native-engine/src/main/cpp/calibration/rig_graph.h
- Create: core/native-engine/src/main/cpp/calibration/rig_graph.cpp
- Create: core/native-engine/src/main/cpp/calibration/calibration_solver.h
- Create: core/native-engine/src/main/cpp/calibration/calibration_solver.cpp
- Create: core/native-engine/src/main/cpp/calibration/calibration_health.h
- Create: core/native-engine/src/main/cpp/calibration/calibration_health.cpp
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/calibration/NativeCalibrationConfig.kt
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/calibration/NativeCalibrationResult.kt
- Create: core/native-engine/src/main/cpp/tests/calibration_solver_test.cpp
- Modify: core:calibration adapter so the existing calibration engine remains available and becomes an evidence/input provider for the native numerical stage.

**Interfaces**
- BrownConradyDistortion::undistort(pixel, intrinsics, coefficients).
- CalibrationSolver::solve(observations, initialRig) returns CalibrationResult with intrinsics, extrinsics, residual RMS, covariance bounds, validity interval, and confidence.
- CalibrationHealth::evaluate(result) returns Valid, Degraded, or Invalid.

- [ ] Step 1: Write failing tests for zero distortion, known synthetic projection, rigid transform recovery, and invalid baseline/conditioning.
- [ ] Step 2: Run those tests and observe failure.
- [ ] Step 3: Implement deterministic projection/undistortion and robust residual weighting.
- [ ] Step 4: Implement rig-graph propagation, residual evaluation, and validity intervals; invalid geometry can never be reported as calibrated.
- [ ] Step 5: Adapt existing Kotlin calibration results into native inputs and preserve their current output path.
- [ ] Step 6: Run focused native tests plus the existing core:calibration unit tests and commit: feat: add native camera calibration engine.

---

## Task 04: Multi-Camera Synchronization

**Files**
- Create: core/native-engine/src/main/cpp/synchronization/clock_model.h
- Create: core/native-engine/src/main/cpp/synchronization/clock_model.cpp
- Create: core/native-engine/src/main/cpp/synchronization/event_correlation.h
- Create: core/native-engine/src/main/cpp/synchronization/event_correlation.cpp
- Create: core/native-engine/src/main/cpp/synchronization/subframe_alignment.h
- Create: core/native-engine/src/main/cpp/synchronization/subframe_alignment.cpp
- Create: core/native-engine/src/main/cpp/synchronization/synchronization_estimator.h
- Create: core/native-engine/src/main/cpp/synchronization/synchronization_estimator.cpp
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/synchronization/NativeSyncResult.kt
- Create: core/native-engine/src/main/cpp/tests/synchronization_estimator_test.cpp
- Modify: core:sync integration without removing the current network/session synchronization engine.

**Interfaces**
- ClockModel::predict(remoteNs) returns local estimate and uncertainty.
- SynchronizationEstimator::estimate(cameraSeries) returns offset, drift ppm, jitter ns, latency ns, subframe delta, and temporal confidence.
- Confidence is deterministic and bounded in [0, 1].

- [ ] Step 1: Write failing synthetic tests for clock offset, clock drift, jitter, latency, visual event correlation, and missing event series.
- [ ] Step 2: Run the tests and verify failure.
- [ ] Step 3: Implement the affine clock model and robust event correlation.
- [ ] Step 4: Implement bounded sub-frame interpolation; never invent an unsupported timestamp.
- [ ] Step 5: Surface synchronization evidence through the existing core:sync layer and native diagnostics.
- [ ] Step 6: Run focused tests and commit: feat: add native multi-camera synchronization.

---

## Task 05: Body Motion Tracking Integration

**Files**
- Create: core/native-engine/src/main/cpp/tracking/body_track_state.h
- Create: core/native-engine/src/main/cpp/tracking/body_tracker.h
- Create: core/native-engine/src/main/cpp/tracking/body_tracker.cpp
- Create: core/native-engine/src/main/cpp/tracking/joint_track.h
- Create: core/native-engine/src/main/cpp/tracking/joint_track.cpp
- Create: core/native-engine/src/main/cpp/tracking/identity_track.h
- Create: core/native-engine/src/main/cpp/tracking/identity_track.cpp
- Create: core/native-engine/src/main/cpp/tests/body_tracker_test.cpp
- Modify: existing core:ai motion-capture bridge so current detectors and trackers feed native temporal state without deleting them.
- Modify: core:reconstruction integration boundary to accept native body states.

**Interfaces**
- BodyTracker::update(jointObservations, timestamp) returns BodyTrackState.
- JointTrackState contains position, velocity, acceleration, jerk, orientation, angular velocity, confidence, uncertainty, visibility, and occlusion state.

- [ ] Step 1: Write failing tests for constant velocity, abrupt motion, missing joint, identity continuity, and confidence degradation.
- [ ] Step 2: Run focused tests and capture failure.
- [ ] Step 3: Implement bounded prediction and state accumulation from the canonical observation model.
- [ ] Step 4: Integrate existing body observations through an explicit adapter and preserve the original engine lifecycle.
- [ ] Step 5: Verify the motion-only session output remains semantically compatible.
- [ ] Step 6: Run native and relevant reconstruction tests and commit: feat: integrate native body motion tracking.

---

## Task 06: Face and Facial-Surface Reconstruction

**Files**
- Create: core/native-engine/src/main/cpp/face/face_track_state.h
- Create: core/native-engine/src/main/cpp/face/face_geometry_fusion.h
- Create: core/native-engine/src/main/cpp/face/face_geometry_fusion.cpp
- Create: core/native-engine/src/main/cpp/face/face_surface_alignment.h
- Create: core/native-engine/src/main/cpp/face/face_surface_alignment.cpp
- Create: core/native-engine/src/main/cpp/tests/face_geometry_fusion_test.cpp
- Modify: existing face detector, contour, and dense-mesh adapters.
- Modify: existing facial reconstruction result mapping.

**Interfaces**
- FaceGeometryFusion::update(faceObservations, calibration, synchronization) returns FaceGeometryState.
- FaceGeometryState carries head pose, metric landmark positions, surface confidence, temporal residual, multi-view residual, and provenance.
- Image-space face-mesh Z is never treated as metric depth without calibrated evidence.

- [ ] Step 1: Write failing synthetic multi-view face tests and a non-metric-Z rejection test.
- [ ] Step 2: Run tests and observe failure.
- [ ] Step 3: Implement calibrated geometric fusion, surface alignment, and temporal stabilization.
- [ ] Step 4: Connect the existing face engines to native fusion and preserve all existing face outputs.
- [ ] Step 5: Run native face tests and existing face/reconstruction regression tests.
- [ ] Step 6: Commit: feat: add native facial geometry fusion.

---

## Task 07: Eye and Gaze Reconstruction

**Files**
- Create: core/native-engine/src/main/cpp/eyes/eye_geometry.h
- Create: core/native-engine/src/main/cpp/eyes/eye_geometry.cpp
- Create: core/native-engine/src/main/cpp/eyes/gaze_fusion.h
- Create: core/native-engine/src/main/cpp/eyes/gaze_fusion.cpp
- Create: core/native-engine/src/main/cpp/eyes/eyelid_state.h
- Create: core/native-engine/src/main/cpp/eyes/eyelid_state.cpp
- Create: core/native-engine/src/main/cpp/tests/gaze_fusion_test.cpp
- Modify: existing eye-tracking bridge.

**Interfaces**
- EyeGeometry::solve(left, right, headPose) returns eye centers, iris centers, and normalized gaze rays.
- GazeFusion::update(observations, worldPose) returns head-relative and world-relative gaze plus confidence and temporal residual.
- No new AI identity or gaze model dependency is introduced.

- [ ] Step 1: Write failing tests for binocular consistency, ray normalization, head-motion compensation, and low-confidence disagreement.
- [ ] Step 2: Run tests and capture failure.
- [ ] Step 3: Implement deterministic ray geometry and confidence-weighted temporal fusion.
- [ ] Step 4: Integrate existing eye observations and preserve current eyelid/gaze outputs.
- [ ] Step 5: Verify confidence decreases when binocular geometry disagrees beyond the configured tolerance.
- [ ] Step 6: Commit: feat: add native eye and gaze reconstruction.

---

## Task 08: Hand and Finger Reconstruction

**Files**
- Create: core/native-engine/src/main/cpp/hands/hand_track_state.h
- Create: core/native-engine/src/main/cpp/hands/hand_geometry_fusion.h
- Create: core/native-engine/src/main/cpp/hands/hand_geometry_fusion.cpp
- Create: core/native-engine/src/main/cpp/hands/finger_kinematics.h
- Create: core/native-engine/src/main/cpp/hands/finger_kinematics.cpp
- Create: core/native-engine/src/main/cpp/tests/hand_geometry_fusion_test.cpp
- Modify: existing hand-tracking integration.

**Interfaces**
- HandGeometryFusion::update(cameraObservations, calibration, bodyState) returns wrist/palm/finger geometry and confidence.
- FingerKinematics::solve(joints, priorState) returns joint rotations, constraint status, and uncertainty.

- [ ] Step 1: Write failing tests for joint-length consistency, handedness, wrist-body association, and partial-finger occlusion.
- [ ] Step 2: Run tests and capture failure.
- [ ] Step 3: Implement multi-view fusion and hand kinematic constraints.
- [ ] Step 4: Integrate current hand observations without deleting detector or landmark code.
- [ ] Step 5: Verify confidence decreases for unsupported finger gaps and hand state remains coupled to the body track.
- [ ] Step 6: Commit: feat: add native hand and finger reconstruction.

---

## Task 09: Temporal Motion State Engine

**Files**
- Create: core/native-engine/src/main/cpp/filtering/robust_state_vector.h
- Create: core/native-engine/src/main/cpp/filtering/robust_state_vector.cpp
- Create: core/native-engine/src/main/cpp/filtering/innovation_gate.h
- Create: core/native-engine/src/main/cpp/filtering/innovation_gate.cpp
- Create: core/native-engine/src/main/cpp/tracking/temporal_motion_state.h
- Create: core/native-engine/src/main/cpp/tracking/temporal_motion_state.cpp
- Create: core/native-engine/src/main/cpp/tests/temporal_motion_state_test.cpp
- Modify: existing temporal smoothing integration to feed native state rather than deleting the existing engine.

**Interfaces**
- TemporalMotionState::predict(dt).
- TemporalMotionState::correct(observation, covariance).
- TemporalMotionState::state() returns position, velocity, acceleration, jerk, orientation, angular velocity, angular acceleration, and covariance.
- InnovationGate::accept(innovation, covariance) returns accept/reject and normalized residual.

- [ ] Step 1: Write failing tests for constant velocity, micro-motion preservation, noisy observations, outliers, and variable timestep.
- [ ] Step 2: Run the focused tests and verify failure.
- [ ] Step 3: Implement bounded robust prediction/correction with covariance-aware gating and quaternion-safe orientation.
- [ ] Step 4: Connect the native state to body, face, eye, and hand tracks through explicit interfaces.
- [ ] Step 5: Run all native temporal tests and dependent feature tests.
- [ ] Step 6: Commit: feat: add covariance-aware temporal motion state.

---

## Task 10: IMU/Vision Fusion

**Files**
- Create: core/native-engine/src/main/cpp/geometry3d/imu_state.h
- Create: core/native-engine/src/main/cpp/geometry3d/imu_state.cpp
- Create: core/native-engine/src/main/cpp/tracking/imu_vision_fusion.h
- Create: core/native-engine/src/main/cpp/tracking/imu_vision_fusion.cpp
- Create: core/native-engine/src/main/cpp/tracking/imu_bias_estimator.h
- Create: core/native-engine/src/main/cpp/tracking/imu_bias_estimator.cpp
- Create: core/native-engine/src/main/cpp/tests/imu_vision_fusion_test.cpp
- Modify: Android sensor integration so gyroscope and accelerometer packets are forwarded to the native observation path.

**Interfaces**
- ImuVisionFusion::propagate(gyro, accel, dt).
- ImuVisionFusion::correct(visualPose, visualCovariance).
- ImuVisionFusion::reset(reason).
- ImuVisionFusion::status() returns orientation, biases, gravity alignment, visual residual, IMU residual, and drift confidence.

- [ ] Step 1: Write failing tests for stationary gravity alignment, angular velocity propagation, visual correction, sensor dropout, and bias convergence.
- [ ] Step 2: Run tests and confirm failure.
- [ ] Step 3: Implement quaternion propagation, bias state, gravity alignment, covariance weighting, dropout, and explicit reset/recovery.
- [ ] Step 4: Connect timestamps from Tasks 02 and 04.
- [ ] Step 5: Verify magnetometer remains optional.
- [ ] Step 6: Commit: feat: add deterministic imu vision fusion.

---

## Task 11: Multi-Camera Subject Association

**Files**
- Create: core/native-engine/src/main/cpp/tracking/camera_visibility_graph.h
- Create: core/native-engine/src/main/cpp/tracking/camera_visibility_graph.cpp
- Create: core/native-engine/src/main/cpp/tracking/multi_camera_assignment.h
- Create: core/native-engine/src/main/cpp/tracking/multi_camera_assignment.cpp
- Create: core/native-engine/src/main/cpp/tracking/geometric_identity_track.h
- Create: core/native-engine/src/main/cpp/tracking/geometric_identity_track.cpp
- Create: core/native-engine/src/main/cpp/tests/multi_camera_assignment_test.cpp

**Interfaces**
- CameraVisibilityGraph::update(cameraId, visibleSubjectIds, timestamp).
- MultiCameraAssignment::solve(tracks, detections, geometry, synchronization) returns assignments with cost, confidence, and ambiguity margin.
- Assignment evidence uses geometric compatibility, temporal proximity, motion continuity, visibility graph, and confidence-weighted costs.

- [ ] Step 1: Write failing tests for two-camera identity consistency, crossing subjects, disappearing/reappearing subjects, and ambiguous assignments.
- [ ] Step 2: Run and confirm failure.
- [ ] Step 3: Implement deterministic assignment costs and bounded ambiguity handling.
- [ ] Step 4: Feed calibration and synchronization confidence so poor geometry increases assignment uncertainty.
- [ ] Step 5: Run all association and preceding tracking tests.
- [ ] Step 6: Commit: feat: add geometry-first multi-camera association.

---

## Task 12: Robust 3D Triangulation

**Files**
- Create: core/native-engine/src/main/cpp/triangulation/weighted_triangulator.h
- Create: core/native-engine/src/main/cpp/triangulation/weighted_triangulator.cpp
- Create: core/native-engine/src/main/cpp/triangulation/cheirality_gate.h
- Create: core/native-engine/src/main/cpp/triangulation/cheirality_gate.cpp
- Create: core/native-engine/src/main/cpp/triangulation/reprojection_evaluator.h
- Create: core/native-engine/src/main/cpp/triangulation/reprojection_evaluator.cpp
- Create: core/native-engine/src/main/cpp/triangulation/covariance_propagator.h
- Create: core/native-engine/src/main/cpp/triangulation/covariance_propagator.cpp
- Create: core/native-engine/src/main/cpp/tests/triangulation_test.cpp
- Modify: existing triangulation integration through an additive adapter.

**Interfaces**
- WeightedTriangulator::solve(points2d, cameras, weights) returns point, covariance, reprojectionRms, conditioning, confidence, and valid.
- ReprojectionEvaluator::evaluate(point, cameras, measurements) returns per-view residuals and a robust aggregate.

- [ ] Step 1: Write failing synthetic tests for exact triangulation, noise, outlier views, cheirality failure, and near-zero baseline.
- [ ] Step 2: Run failures.
- [ ] Step 3: Implement normalized linear initialization, robust weighted refinement, cheirality validation, conditioning checks, and covariance propagation.
- [ ] Step 4: Feed calibration/sync confidence into observation weights and reject degenerate geometry.
- [ ] Step 5: Compare residual evidence with the existing Kotlin triangulation path without removing it.
- [ ] Step 6: Commit: feat: add robust native multi-view triangulation.

---

## Task 13: Metric-Scale Recovery

**Files**
- Create: core/native-engine/src/main/cpp/reconstruction/scale_evidence.h
- Create: core/native-engine/src/main/cpp/reconstruction/scale_evidence.cpp
- Create: core/native-engine/src/main/cpp/reconstruction/metric_scale_solver.h
- Create: core/native-engine/src/main/cpp/reconstruction/metric_scale_solver.cpp
- Create: core/native-engine/src/main/cpp/tests/metric_scale_solver_test.cpp
- Modify: reconstruction result model to carry scale provenance and confidence.

**Interfaces**
- ScaleEvidence contains sourceType, numericScale, uncertainty, and provenanceHash.
- MetricScaleSolver::solve(evidenceSet) returns scale, uncertainty, source, confidence, and valid.
- Supported sources are measuredRig, knownSceneDimension, stableBaseline, supportedDepth, and bodyConstraint.

- [ ] Step 1: Write failing tests for exact rig scale, conflicting scale sources, insufficient evidence, and relative-depth-only input.
- [ ] Step 2: Run failures.
- [ ] Step 3: Implement robust source weighting and conflict detection.
- [ ] Step 4: Reject relative-depth-only input from being relabeled as metric.
- [ ] Step 5: Verify provenance is serialized into reconstruction/session results.
- [ ] Step 6: Commit: feat: add provenance-backed metric scale recovery.

---

## Task 14: Dense Scene Reconstruction

**Files**
- Create: core/native-engine/src/main/cpp/scene/scene_point_map.h
- Create: core/native-engine/src/main/cpp/scene/scene_point_map.cpp
- Create: core/native-engine/src/main/cpp/scene/depth_fusion.h
- Create: core/native-engine/src/main/cpp/scene/depth_fusion.cpp
- Create: core/native-engine/src/main/cpp/scene/static_dynamic_separator.h
- Create: core/native-engine/src/main/cpp/scene/static_dynamic_separator.cpp
- Create: core/native-engine/src/main/cpp/scene/normal_estimator.h
- Create: core/native-engine/src/main/cpp/scene/normal_estimator.cpp
- Create: core/native-engine/src/main/cpp/scene/mesh_builder.h
- Create: core/native-engine/src/main/cpp/scene/mesh_builder.cpp
- Create: core/native-engine/src/main/cpp/tests/scene_reconstruction_test.cpp
- Modify: existing monocular depth and multi-view reconstruction adapters.

**Interfaces**
- DepthFusion::integrate(depthObservation, cameraPose, timestamp).
- ScenePointMap::snapshot().
- StaticDynamicSeparator::classify(pointHistory).
- MeshBuilder::build(pointMap, normals, confidence) returns world-coordinate mesh plus topology confidence.

- [ ] Step 1: Write failing deterministic tests for a plane, static/dynamic separation, hole recovery, and normal estimation.
- [ ] Step 2: Run failures.
- [ ] Step 3: Implement confidence-weighted point-map accumulation, temporal depth fusion, static/dynamic separation, normals, and mesh generation.
- [ ] Step 4: Keep human reconstruction separate from static scene reconstruction while sharing calibration/world coordinates.
- [ ] Step 5: Preserve existing monocular and multi-view engines as evidence providers.
- [ ] Step 6: Run tests and commit: feat: add native dense scene reconstruction.

---

## Task 15: Kinematic Optimization

**Files**
- Create: core/native-engine/src/main/cpp/kinematics/skeleton_model.h
- Create: core/native-engine/src/main/cpp/kinematics/skeleton_model.cpp
- Create: core/native-engine/src/main/cpp/optimization/objective_terms.h
- Create: core/native-engine/src/main/cpp/optimization/objective_terms.cpp
- Create: core/native-engine/src/main/cpp/optimization/robust_solver.h
- Create: core/native-engine/src/main/cpp/optimization/robust_solver.cpp
- Create: core/native-engine/src/main/cpp/kinematics/contact_constraints.h
- Create: core/native-engine/src/main/cpp/kinematics/contact_constraints.cpp
- Create: core/native-engine/src/main/cpp/tests/kinematic_optimizer_test.cpp
- Modify: existing kinematic-constraint and bundle-adjustment integrations.

**Interfaces**
- SkeletonModel::forwardKinematics(root, localRotations, boneLengths).
- ObjectiveTerms::evaluate(candidate, observations, calibration, synchronization).
- RobustSolver::solve(initialState, objective) returns final state, objective value, iterations, convergence flag, and residual breakdown.

Objective terms cover bone-length consistency, joint limits, parent-child rotations, root trajectory, contact constraints, temporal smoothness, reprojection error, multi-view agreement, face/head coupling, and hand/wrist coupling.

- [ ] Step 1: Write failing tests for bone-length preservation, joint limits, contact stabilization, multi-view agreement, and non-convergence reporting.
- [ ] Step 2: Run failures.
- [ ] Step 3: Implement deterministic robust optimization with explicit convergence and stopping diagnostics.
- [ ] Step 4: Use existing bundle-adjustment and kinematic-constraint engines as evidence/initialization components rather than replacing them.
- [ ] Step 5: Verify convergence is only reported when configured residual and iteration criteria are satisfied.
- [ ] Step 6: Commit: feat: add native kinematic motion optimization.

---

## Task 16: Occlusion and Missing-Data Recovery

**Files**
- Create: core/native-engine/src/main/cpp/tracking/occlusion_state.h
- Create: core/native-engine/src/main/cpp/tracking/occlusion_state.cpp
- Create: core/native-engine/src/main/cpp/reconstruction/missing_data_recovery.h
- Create: core/native-engine/src/main/cpp/reconstruction/missing_data_recovery.cpp
- Create: core/native-engine/src/main/cpp/tests/missing_data_recovery_test.cpp

**Interfaces**
- MissingDataRecovery::predict(track, gapDuration, constraints) returns a bounded state plus expanded uncertainty.
- OcclusionState values are visible, partiallyOccluded, predicted, unsupported.
- Recovery has a maximum prediction horizon and cannot promote unsupported estimates to observed confidence.

- [ ] Step 1: Write failing tests for short occlusion, long occlusion, reappearance, impossible kinematic jumps, and uncertainty growth.
- [ ] Step 2: Run failures.
- [ ] Step 3: Implement bounded prediction from temporal state plus skeleton constraints and covariance inflation with unsupported duration.
- [ ] Step 4: Integrate explicit occlusion provenance into body, face, eye, and hand tracks.
- [ ] Step 5: Run synthetic gaps and verify confidence never increases merely because a gap persists.
- [ ] Step 6: Commit: feat: add bounded occlusion recovery.

---

## Task 17: Lighting and Environment Capture

**Files**
- Create: core/native-engine/src/main/cpp/lighting/illumination_statistics.h
- Create: core/native-engine/src/main/cpp/lighting/illumination_statistics.cpp
- Create: core/native-engine/src/main/cpp/lighting/directional_evidence.h
- Create: core/native-engine/src/main/cpp/lighting/directional_evidence.cpp
- Create: core/native-engine/src/main/cpp/lighting/light_event_detector.h
- Create: core/native-engine/src/main/cpp/lighting/light_event_detector.cpp
- Create: core/native-engine/src/main/cpp/tests/lighting_telemetry_test.cpp
- Modify: existing lighting telemetry integration.

**Interfaces**
- IlluminationStatistics::update(frameTelemetry) returns rolling exposure/illumination statistics.
- DirectionalEvidence::estimate(samples, sceneGeometry) returns measurable directional cues and confidence.
- LightEventDetector::update(stats) emits timestamped light-change events.

- [ ] Step 1: Write failing tests for stable illumination, exposure changes, rapid light events, and missing telemetry.
- [ ] Step 2: Run failure.
- [ ] Step 3: Implement temporal statistics, directional evidence where supported by measurements, and event detection.
- [ ] Step 4: Preserve existing telemetry fields and map them into the native representation.
- [ ] Step 5: Verify the output describes environment evidence without claiming exhaustive light-field capture.
- [ ] Step 6: Commit: feat: extend lighting environment telemetry.

---

## Task 18: Performance and Runtime Governor

**Files**
- Create: core/native-engine/src/main/cpp/runtime/work_scheduler.h
- Create: core/native-engine/src/main/cpp/runtime/work_scheduler.cpp
- Create: core/native-engine/src/main/cpp/runtime/runtime_budget.h
- Create: core/native-engine/src/main/cpp/runtime/runtime_budget.cpp
- Create: core/native-engine/src/main/cpp/runtime/queue_pressure.h
- Create: core/native-engine/src/main/cpp/runtime/queue_pressure.cpp
- Create: core/native-engine/src/main/cpp/runtime/thermal_policy.h
- Create: core/native-engine/src/main/cpp/runtime/thermal_policy.cpp
- Create: core/native-engine/src/main/cpp/tests/runtime_governor_test.cpp
- Modify: NativeEngineDiagnostics.kt.

**Interfaces**
- RuntimeBudget::beginFrame(timestamp), remainingNs(), endFrame(stats).
- WorkScheduler::submit(priority, workItem) returns accepted/rejected plus queue evidence.
- ThermalPolicy::evaluate(state) returns a workload policy that adjusts internal scheduling without changing semantic result definitions.

- [ ] Step 1: Write failing tests for queue overflow, thermal transitions, deadline pressure, fair camera scheduling, and bounded shedding.
- [ ] Step 2: Run failures.
- [ ] Step 3: Implement bounded worker queues, frame budgets, priority classes, memory checks, and thermal policy.
- [ ] Step 4: Increment evidence counters for every deferred or shed workload item; never silently rewrite confidence.
- [ ] Step 5: Run native stress tests for bounded allocation, queue depth, and deterministic scheduler decisions.
- [ ] Step 6: Commit: feat: add native runtime governor and diagnostics.

---

## Task 19: Professional Motion Studio UX

**Files**
- Create: app/src/main/java/com/nivukx/nox/ui/studio/CaptureSurface.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/RigSurface.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/CalibrationSurface.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/SyncSurface.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/ReconstructionSurface.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/DiagnosticsSurface.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/ExportSurface.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/StudioTelemetryState.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/StudioUiState.kt
- Create: app/src/test/java/com/nivukx/nox/ui/studio/StudioUiStateTest.kt
- Modify: app/src/main/java/com/nivukx/nox/ui/UltraMotionStudio.kt
- Modify: app/src/main/java/com/nivukx/nox/NoXViewModel.kt
- Modify: app/src/main/java/com/nivukx/nox/MainActivity.kt

**Interfaces**
- StudioTelemetryState contains camera connectivity, sync health, calibration health, subject tracking, reconstruction health, confidence, drops, thermal state, export state, and native ABI/runtime status.
- StudioUiState separates Capture, Rig, Calibration, Sync, Reconstruction, Face/Eye, Hands, Scene, Timeline, Diagnostics, Export, and Advanced Settings.
- UI surfaces consume state snapshots and never raw native pointers.

- [ ] Step 1: Write failing state-reduction tests for connected/uncalibrated, synchronized/calibrated, thermal-constrained, native-error, and export-invalid states.
- [ ] Step 2: Run tests and capture failure.
- [ ] Step 3: Implement explicit mapping from native diagnostics and existing engines to StudioTelemetryState.
- [ ] Step 4: Split the existing large studio surface into focused Compose surfaces while preserving all current functionality and motion-only semantics.
- [ ] Step 5: Keep Liquid Glass styling for navigation, controls, and telemetry while leaving the reconstruction viewport visually clear.
- [ ] Step 6: Run targeted UI state tests and existing app unit tests; commit: feat: upgrade motion studio operator surfaces.

---

## Task 20: Evidence, Replay, Export Validation, and Release Gates

**Files**
- Create: core/native-engine/src/main/cpp/diagnostics/evidence_trace.h
- Create: core/native-engine/src/main/cpp/diagnostics/evidence_trace.cpp
- Create: core/native-engine/src/main/cpp/export/motion_export_schema.h
- Create: core/native-engine/src/main/cpp/export/motion_export_schema.cpp
- Create: core/native-engine/src/main/cpp/export/export_validator.h
- Create: core/native-engine/src/main/cpp/export/export_validator.cpp
- Create: core/native-engine/src/main/cpp/replay/replay_reader.h
- Create: core/native-engine/src/main/cpp/replay/replay_reader.cpp
- Create: core/native-engine/src/main/cpp/replay/replay_runner.h
- Create: core/native-engine/src/main/cpp/replay/replay_runner.cpp
- Create: core/native-engine/src/main/cpp/tests/replay_validation_test.cpp
- Create: core/native-engine/src/main/java/com/nivukx/nox/core/nativeengine/replay/ReplayManifest.kt
- Create: core/native-engine/src/test/java/com/nivukx/nox/core/nativeengine/replay/ReplayManifestTest.kt
- Create fixtures under core/native-engine/src/test/resources/replay:
  - single_camera_motion.json
  - dual_camera_clock_drift.json
  - four_camera_occlusion.json
  - six_camera_rapid_motion.json
  - moving_camera.json
  - low_light_telemetry.json
  - dropped_frames.json
  - calibration_perturbation.json
- Modify: existing session/export contracts only through additive adapters.

**Interfaces**
- EvidenceTraceBuilder::record(event) and finalize() produce a deterministic structured trace.
- ReplayRunner::run(manifest, fixture) returns output hashes and stage diagnostics.
- ExportValidator::validate(session) returns schema/version checks, motion-only contract checks, provenance checks, and deterministic hash checks.
- Canonical export rejects ordinary video, raw camera-frame archives, and depth-video streams.

- [ ] Step 1: Write failing tests for deterministic replay, fixture checksum stability, schema validation, motion-only export, and confidence-inflation detection.

~~~cpp
TEST(ReplayValidation, SameFixtureProducesSameDigest) {
    const auto first = run_fixture("dual_camera_clock_drift.json");
    const auto second = run_fixture("dual_camera_clock_drift.json");
    EXPECT_EQ(first.output_digest, second.output_digest);
}

TEST(ExportValidation, RejectsCanonicalVideoArtifact) {
    auto session = make_motion_only_session();
    session.artifacts.push_back(ArtifactKind::CanonicalVideo);
    EXPECT_FALSE(ExportValidator::validate(session).valid);
}
~~~

- [ ] Step 2: Run the full native suite and confirm the new replay/export tests fail before implementation.
- [ ] Step 3: Implement deterministic evidence traces, replay loading, replay execution, export validation, provenance serialization, and fixture manifests.
- [ ] Step 4: Execute replay suites for single, two, four, and six camera inputs, moving camera, occlusion, rapid motion, low-light telemetry, clock drift, dropped frames, and calibration perturbations.
- [ ] Step 5: Run the complete verification matrix without launching/installing the Android application: native CTest suite, affected Gradle unit-test tasks, static checks for changed modules, repository-wide placeholder scan, and Git diff review against the preservation list.
- [ ] Step 6: Commit: test: add deterministic motion evidence and release validation.

---

# Cross-Stage Integration Checks

After Task 04, every observation carries normalized timestamp and synchronization metadata.

After Tasks 06-08, face, eye, and hand states use the same world-coordinate and confidence conventions as body tracks.

After Task 09, temporal subsystems expose covariance/uncertainty and no opaque smoothing constant may replace the formal state estimator.

After Task 12, every reconstructed 3D point exposes reprojection residual and conditioning evidence.

After Task 13, every metric result records scale source, uncertainty, and provenance.

After Task 15, every optimized skeleton state exposes convergence status and residual breakdown.

After Task 16, every predicted interval retains explicit predicted/unsupported state.

After Task 18, every shed or deferred workload item is observable through diagnostics.

After Task 19, all operator-critical health fields are visible without entering a deep advanced-settings surface.

After Task 20, release claims are tied to deterministic fixture results or measured runtime evidence.

# Required Verification Matrix

Use these commands during implementation. They are validation commands, not application launch/install commands:

~~~bash
cmake -S core/native-engine -B core/native-engine/build/test -DNOX_NATIVE_HOST_TESTS=ON
cmake --build core/native-engine/build/test --target nox_native_tests -j2
ctest --test-dir core/native-engine/build/test --output-on-failure

./gradlew :core:native-engine:testDebugUnitTest
./gradlew :core:model:testDebugUnitTest
./gradlew :core:sync:testDebugUnitTest
./gradlew :core:capture:testDebugUnitTest
./gradlew :core:calibration:testDebugUnitTest
./gradlew :core:reconstruction:testDebugUnitTest
./gradlew :app:testDebugUnitTest
~~~

Do not run assembleDebug, assembleRelease, installDebug, emulator launch, or an application session unless the user explicitly changes the current no-app-build constraint.

# Repository Preservation Checks

Before every commit:

~~~bash
git status --short
git diff --check
git diff --stat
~~~

Before final release validation:

~~~bash
git grep -n -E "stub-patterns" -- . ':!docs/superpowers'
git diff --name-status <baseline-sha>..HEAD
git log --oneline --decorate -25
~~~

The production-code placeholder scan must return no matches. Documentation may describe intentional research statements; implementation files may not contain stub implementations.

# Commit Sequence

1. feat: add native motion engine runtime foundation
2. feat: add bounded native capture ingestion
3. feat: add native camera calibration engine
4. feat: add native multi-camera synchronization
5. feat: integrate native body motion tracking
6. feat: add native facial geometry fusion
7. feat: add native eye and gaze reconstruction
8. feat: add native hand and finger reconstruction
9. feat: add covariance-aware temporal motion state
10. feat: add deterministic imu vision fusion
11. feat: add geometry-first multi-camera association
12. feat: add robust native multi-view triangulation
13. feat: add provenance-backed metric scale recovery
14. feat: add native dense scene reconstruction
15. feat: add native kinematic motion optimization
16. feat: add bounded occlusion recovery
17. feat: extend lighting environment telemetry
18. feat: add native runtime governor and diagnostics
19. feat: upgrade motion studio operator surfaces
20. test: add deterministic motion evidence and release validation

# Implementation Discipline

Every task uses the TDD sequence:

1. Write the smallest focused failing test.
2. Run exactly that test and observe the failure.
3. Implement the smallest real production behavior satisfying the test.
4. Run the focused test until green.
5. Run the task regression set.
6. Review the diff for preservation and unintended coupling.
7. Commit the task.
8. Move forward only after the current task is green.

Native hot-path code uses focused translation units and stable typed structures instead of giant multi-purpose source files. Existing oversized Kotlin files may be split only where the split is behavior-preserving and improves ownership boundaries. No file is required to reach an arbitrary line count; complexity corresponds to real production behavior, tests, diagnostics, and interfaces.

# End-State Acceptance

The implementation is ready for final release review only when:

- all 20 stage commits exist on main;
- all native host tests pass;
- all affected Kotlin unit tests pass;
- the existing motion-only contract is still enforced;
- all existing engine paths named in the approved spec remain present and callable;
- calibration, synchronization, triangulation, scale, temporal state, optimization, occlusion, and runtime health expose measurable evidence;
- replay fixtures produce deterministic outputs;
- export validation rejects ordinary canonical video artifacts;
- no production placeholder implementation remains;
- no silent downgrade or fabricated confidence/scale/synchronization claim is present;
- performance behavior is measured for supported fixtures and device configurations;
- the final branch review is completed using fresh verification evidence.
