# NoX high-end capture upgrade research record: 2026-09-29

## Scope

This pass upgrades the motion-only NoX pipeline without introducing video encoding, HDR/HLG/SDR recording, or cinematic color processing. The focus is metric capture integrity: concurrent-camera admission, provenance-rich observations, continuous-time timing, reconstruction evidence, runtime safety, and operator usability.

## External findings

### Android concurrent cameras

Android documents that parallel camera streams can have significant processing cost and that memory use can grow with additional pipelines. Camera2 also exposes platform-declared concurrent camera combinations and runtime session-configuration checks.

Implementation consequence:

- Treat concurrent-camera support as a topology constraint, not a boolean feature flag.
- Budget stream memory, frame throughput, compute pressure and thermal headroom before capture.
- Keep the camera set bounded and reject unsupported combinations before opening a session.
- Preserve per-camera evidence so a later reconstruction can identify degraded views.

References:

- https://developer.android.com/media/camera/camera2/multiple-camera-streams-simultaneously
- https://developer.android.com/reference/kotlin/android/hardware/camera2/CameraManager
- https://developer.android.com/media/camera/camera2/multi-camera

### Dense markerless evidence

MAMMA at CVPR 2026 uses dense 2D landmarks with uncertainty, visibility and contact estimates and combines these with segmentation to handle difficult close interactions. HUM4D likewise emphasizes synchronized multi-view capture, accurate calibration and external reference data for difficult occlusion regimes.

Implementation consequence:

- A landmark is now represented conceptually as evidence plus provenance, not only x/y.
- Visibility, contact, occlusion, identity and semantic mask agreement are carried separately.
- Robust agreement weighting happens before geometry admission.
- The fusion layer does not claim physical accuracy by itself.

References:

- https://openaccess.thecvf.com/content/CVPR2026/papers/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.pdf
- https://arxiv.org/abs/2608.01509

### Rolling shutter and continuous time

Recent rolling-shutter self-calibration work models row-dependent image timing and continuous camera trajectories, then uses dual projection constraints to improve temporal and geometric consistency.

Implementation consequence:

- Treat each image row as a time sample.
- Use a continuous-time state prior instead of a frame-wide timing penalty alone.
- Down-weight model/measurement disagreement with robust residual handling.
- Hard-limit extrapolation and expose the event as explicit evidence.

Reference:

- https://arxiv.org/abs/2608.01509

### Precision and reference qualification

Professional motion-capture guidance uses reconstruction residuals, ray contribution, stable calibration and rigid camera mounting as operational indicators. These are evidence metrics, not universal guarantees of accuracy for every sensor arrangement.

Implementation consequence:

- Keep live precision admission distinct from measured qualification.
- Continue requiring independent reference measurements before NoX presents a sub-millimetre result as a physical accuracy claim.
- Use camera independence, geometric diversity, synchronization and calibrated reference error as separate signals.

Reference:

- https://docs.optitrack.com/v3.2/quick-start-guides/quick-start-guide-precision-capture

### Operator UI and Liquid Glass

Apple's Liquid Glass guidance emphasizes material hierarchy, adaptive appearance, legibility, content priority, predictable controls and restraint. That is especially relevant to a mocap console where telemetry must not obscure the reconstruction viewport.

Implementation consequence:

- Keep the 3D reconstruction and motion timeline visually dominant.
- Use glass as a semantic layer for navigation and telemetry rather than decoration.
- Adapt opacity and contrast to the underlying imagery.
- Preserve a reduced-transparency path and avoid unnecessary always-on motion.

References:

- https://developer.apple.com/documentation/technologyoverviews/liquid-glass
- https://developer.apple.com/documentation/TechnologyOverviews/adopting-liquid-glass

## New production modules

1. `ConcurrentCaptureCapabilityEngine`
   - deterministic concurrent-camera admission;
   - memory and throughput budgeting;
   - thermal and synchronization gates;
   - calibration and view-diversity evidence;
   - platform-declared concurrent-set enforcement.

2. `DenseLandmarkEvidenceFusionEngine`
   - uncertainty/visibility/contact provenance;
   - occlusion and identity weighting;
   - semantic-mask agreement;
   - robust cross-view consensus;
   - geometry admission blockers.

3. `ContinuousTimeObservationRefiner`
   - row-aware observation timestamps;
   - cubic Hermite continuous-time state interpolation;
   - velocity/acceleration uncertainty;
   - bounded extrapolation;
   - robust measurement disagreement handling.

## Accuracy contract

No amount of source code can prove a universal '<1 mm in practice' result on arbitrary phones or arbitrary capture volumes. The physically meaningful contract requires independent reference instrumentation, controlled calibration, dynamic trials, occlusion trials and repeat sessions.

NoX therefore continues to distinguish:

- **live sub-millimetre capable**: the current operational uncertainty/evidence budget meets the configured gate;
- **measured sub-millimetre qualified**: an external reference cohort meets the configured statistical and session requirements.

That distinction is intentional and is preserved by the existing precision authority and reference-metrology engine.


## Additional production upgrades in the same pass

4. ThermalExtrinsicDriftEngine
   - tracks translation and rotation drift against calibration poses;
   - estimates drift coefficients per degree Celsius;
   - measures thermal correlation;
   - detects mount-integrity degradation;
   - provides stable/watch/recalibrate/blocked states.

5. AdaptiveReconstructionDetailEngine
   - converts uncertainty, geometry, thermal, memory, GPU, dynamics and view evidence into a bounded reconstruction-detail budget;
   - adapts voxel size, mesh budget, surface frequency, face/eye/hand rates and bundle-adjustment iterations;
   - can hard-hold metric output when evidence becomes unsafe.

6. FaceEyeMetricEvidenceEngine
   - separates dense face coverage from eye geometry;
   - scores binocular consistency, eye uncertainty, temporal stability, calibration drift and gaze calibration;
   - promotes metric face/eye output only when the complete evidence chain reaches its gates.

7. ContactAwareInteractionFusionEngine
   - models hand/object proximity, relative speed, contact persistence, inertial confidence and temporal consistency;
   - distinguishes kinematic, inertial and fused-contact reasoning;
   - prevents a transient contact detection from becoming a rigid constraint immediately.

8. Camera2 rolling-shutter evidence
   - RollingShutterEvidenceEngine now scales reported sensor skew to the active output viewport when active-array/crop metadata are available, matching Android Camera2 guidance.

9. Production-policy integration
   - the adaptive reconstruction authority now feeds HighEndReconstructionQualityEngine;
   - the production quality policy records adaptive-detail evidence and can hard-hold output;
   - capture preflight accepts concurrent-rig admission, thermal-drift and face/eye evidence as explicit policy channels.

## 2026-10-01 continuation: hardware timing, crowd fairness, calibration-aware precision

10. CameraSessionNegotiationEngine
   - validates the exact Camera2 SessionConfiguration topology on API 35+ when CameraDeviceSetup is available;
   - verifies concurrent session topology through the runtime CameraManager validator on API 30+;
   - keeps UNKNOWN distinct from REJECTED so legacy devices retain an explicit fallback state.

11. Camera2CaptureController hardware readout channel
   - captures API 34+ onReadoutStarted timestamps by camera frame number;
   - preserves exposure-start sensor timestamps as the canonical source clock;
   - records readout timing separately for continuous-time reconstruction.

12. MonotonicTimestampSequencer and temporal multi-person inference
   - multi-person MediaPipe inference uses VIDEO mode on one serialized worker;
   - camera timestamps are converted to strictly increasing millisecond timestamps with the smallest necessary adjustment;
   - capture-sequence reset prevents identities and temporal state leaking between takes.

13. MaximumWeightBipartiteAssignmentEngine
   - global one-to-one Hungarian assignment replaces greedy cross-camera identity pairing;
   - deterministic tie breaking;
   - bounded matrix size prevents crowd complexity from becoming unbounded;
   - weak matches can remain unmatched through explicit dummy assignments.

14. MultiPersonRuntimeCapacityGovernor fairness
   - retains the existing 50-subject target and 64-subject hard ceiling;
   - adds bounded aging so repeatedly deferred identities receive a controlled scheduling boost;
   - optional face/eye/hand/dense-surface detail is throttled before core body tracking under pressure.

15. HardwareReadoutTimingEvidenceEngine
   - evaluates readout delay, P95 jitter, baseline drift, coverage and callback ordering;
   - distinguishes stable/watch/degraded/blocked readout evidence;
   - persists the evidence independently in session telemetry.

16. CalibrationUncertaintyProfile and covariance propagation
   - calibration snapshots can now carry per-camera translational, rotational and intrinsic uncertainty envelopes;
   - version-2 calibration wire format preserves these profiles while decoding version-1 snapshots;
   - triangulation covariance and first-order sub-millimetre feasibility now account for correlated calibration uncertainty rather than image residuals alone.

17. PrecisionQualificationProtocolEngine
   - formalizes practical qualification as an external-reference protocol;
   - requires independent sessions, multiple subjects/joints, dynamic and static cohorts, occluded trials, dropout limits and bounded reference uncertainty;
   - exposes protocol status directly in PhysicalPrecisionCapabilityEngine.Decision.

18. Liquid Glass operator evidence
   - session verification, hardware readout availability, readout timing and metric face/eye state are exposed to the studio telemetry model;
   - the interface keeps diagnostic evidence separate from the dominant reconstruction viewport.

## Accuracy and certification note

The repository now contains stronger measurement and gating infrastructure for sub-millimetre qualification, but the code still does not constitute a universal guarantee of <1 mm physical accuracy across arbitrary devices or capture volumes. That requires physical measurement with independent reference instrumentation, controlled calibration, repeated sessions, dynamic/occlusion coverage and acceptance criteria. Android explicitly documents separate exposure-start and readout-start timestamps, while Apple’s current Liquid Glass guidance emphasizes hierarchy, legibility, restrained glass usage and accessibility adaptation.

