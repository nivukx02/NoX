# NoX High-End Motion Capture Architecture Design

## Goal

Evolve NoX into a production-grade mobile motion-capture platform on the `main` branch, centered on sensor-fused motion inference, calibrated multi-view 3D reconstruction, dense facial/eye capture, robust multi-camera synchronization, session integrity, performance governance, and a professional adaptive operator UI.

## Product Boundary

NoX is motion-capture software. It does not become a general HDR/HLG/SDR video recorder. Visual frames are treated as perception inputs and synchronization evidence; the product output is motion, geometry, metadata, calibration, lighting observations, and derived reconstruction data.

## Current Architecture

The repository already contains separate modules for AI/perception, calibration, capture, model, network, reconstruction, and synchronization. The current reconstruction layer already includes bundle adjustment, triangulation, rolling-shutter compensation, inertial fusion, adaptive multi-view weighting, uncertainty propagation, kinematic constraints, face/eye reconstruction, dense mesh temporal consistency, and temporal stabilization.

The current motion estimator has a timestamped temporal state, robust innovation gating, camera-motion context, common-mode/differential motion diagnostics, and timestamp-regression reset behavior. The UI has an adaptive Motion Studio with a glass control layer and high-density telemetry.

## Research-Derived Principles

1. Ordinary multi-view triangulation assumes synchronized observations. Dynamic reconstruction therefore needs explicit temporal alignment and motion priors; spatiotemporal bundle adjustment jointly optimizing camera parameters, temporal alignment, static points, and dynamic trajectories is the architectural reference.
2. Rolling-shutter timing is a first-class geometric variable. Per-row timing and IMU data can be used to estimate camera motion during readout and separate camera-induced motion from independently moving content.
3. Physically plausible motion requires kinematic and environmental constraints, not only frame-to-frame smoothing.
4. Multi-camera calibration is a temporal contract. Baseline drift, reprojection error, thermal stress, vibration, and synchronization confidence must influence the trust of downstream geometry.
5. Operator UI should be adaptive to window size, preserve a clear content/control hierarchy, and use translucent surfaces sparingly around controls instead of making the entire viewport visually noisy.

## Architecture

### Sensor and Timing
- Camera timestamp validation and monotonicity enforcement.
- IMU capture and sensor confidence.
- Rolling-shutter model with row timing.
- Camera-to-IMU extrinsics and time offset.
- Timing health and uncertainty propagated into perception and reconstruction.

### Motion Intelligence
- Detector-independent temporal state estimation.
- Robust innovation gating.
- Camera/common-mode motion decomposition.
- Differential/articulated motion scoring.
- Micro-motion preservation.
- Occlusion prediction and bounded reacquisition.
- Measurement uncertainty.
- Camera-motion and rolling-shutter confidence modulation.
- Per-subject motion state isolation.

### Body, Face, Eyes, Hands
- Multi-subject identity preservation.
- Body pose and articulation.
- Dense face mesh as auxiliary close-range geometry.
- Head pose and gaze fusion.
- Eye temporal consistency.
- Hand geometry and temporal consistency.
- Face/eye identity association across cameras.

### 3D Reconstruction
- Calibrated observation formation.
- Rolling-shutter-aware corrections.
- Robust triangulation with parallax and cheirality gates.
- Temporal trajectory reconstruction.
- Spatiotemporal alignment.
- Bundle adjustment for calibration and reconstruction refinement.
- Uncertainty propagation.
- Kinematic constraint solving.
- Ground-contact and root-motion consistency.
- Avatar retargeting guards.

### Multi-Camera Rig
- Explicit camera topology.
- Clock offset and skew estimation.
- Calibration quality gates.
- Drift detection without silently mutating the reference.
- Adaptive view evidence weighting.
- Hold/recalibrate decisions.

### Session and Transport
- Deterministic timestamps.
- Chunked session recording.
- Binary schema versioning.
- Dense mesh transport with topology-aware keyframes.
- Integrity validation.
- Recovery and replay-ready metadata.
- Stable schema migration rules.

### Performance
- GPU/NPU-aware scheduling.
- Thermal and memory governance.
- Stage-level telemetry.
- Bounded inference degradation.
- Compose stability and recomposition discipline.
- Future benchmark hooks without requiring a production build in this phase.

### UI/UX
- Adaptive width tiers.
- Operator-first capture surface.
- Capture readiness and integrity as explicit states.
- Motion customization presets plus advanced tuning.
- Reconstruction controls grouped by responsibility.
- Dense but readable telemetry.
- Restrained translucent/liquid-glass-inspired control chrome.
- Accessible contrast, touch targets, semantics, and error states.

## Non-Goals

- Artificial line-count inflation.
- Prototype-only throwaway implementations.
- Silent downgrade of existing capabilities.
- Launching the Android app during this implementation pass.
- Claiming production certification without build, device, and hardware verification.

## Success Criteria

- Existing motion/3D/face/eye functionality remains intact while new telemetry and algorithms integrate through stable interfaces.
- Every new behavior has regression coverage.
- Multi-view timing and rolling-shutter uncertainty are represented in the reconstruction contract.
- UI exposes meaningful reconstruction and timing state without overwhelming the capture workflow.
- All repository writes remain on `main`.
