# NivukxNoX high-end crowd and 3D evidence upgrade: 2026-10-01

## Upgrade scope

This revision strengthens the existing production-oriented NoX motion-capture architecture without deleting the live path, calibrated multi-view path, face/eye path, hand path, dense surface path, timing path, or operator UI.

The implementation objective is evidence quality rather than source-file inflation. New complexity is introduced as independently bounded engines and explicit data contracts so the system remains reviewable and diagnosable.

## External research used for this revision

### MAMMA: Markerless Accurate Multi-person Motion Acquisition

CVPR 2026 MAMMA demonstrates a multi-view markerless pipeline that predicts dense 2D surface landmarks with confidence, uncertainty, visibility and contact information, then uses calibrated multi-view geometry and SMPL-X fitting for close multi-person interaction.

The work reports a held-out marker comparison against a marker-based Vicon pipeline with a 0.862 mm difference under its specific matched evaluation. This is evidence that modern markerless reconstruction can approach reference-grade performance in a particular experimental setting, not a universal guarantee for phones or arbitrary capture volumes.

Source:
https://openaccess.thecvf.com/content/CVPR2026/papers/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.pdf

### MediaPipe Pose Landmarker

MediaPipe Pose Landmarker exposes normalized image landmarks and world landmarks, and its video mode requires monotonically increasing timestamps. NoX now preserves that body-space world evidence as a secondary geometric prior and keeps timestamp normalization explicit.

Sources:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/java
https://developers.google.com/ml-kit/vision/pose-detection

### Android concurrent camera processing

Android documents that concurrent camera streams have additional processing and memory costs and that supported combinations are device-specific. NoX therefore continues to treat topology admission, queue capacity, runtime pressure and thermal headroom as evidence rather than assuming that a larger camera count automatically improves capture.

Sources:
https://developer.android.com/media/camera/camera2/multiple-camera-streams-simultaneously
https://developer.android.com/reference/kotlin/android/hardware/camera2/CameraManager

### Liquid Glass and operator UI

Apple's Liquid Glass guidance emphasizes hierarchy, adaptive material, content legibility and using glass as a functional layer. NoX keeps dense evidence surfaces readable and adds the new crowd intelligence surface inside the existing Liquid Glass operator system.

Sources:
https://developer.apple.com/documentation/technologyoverviews/liquid-glass
https://developer.apple.com/documentation/TechnologyOverviews/adopting-liquid-glass

## Implemented production upgrades

### 1. Multi-actor 3D body-space evidence

MultiActorPoseLandmarkerEngine now maps MediaPipe body-space world landmarks into a provider-neutral WorldJoint contract.

The contract stores:
- landmark index;
- x/y/z body-space coordinates in meters as provided by the learned model;
- per-landmark visibility/presence-derived confidence.

The data is explicitly not promoted to calibrated rig-world coordinates. Calibrated triangulation remains authoritative for metric scene geometry.

### 2. Evidence-ranked 50-actor admission

MultiActorPoseAssociationEngine no longer uses detector order as the only admission criterion when the runtime actor ceiling is reached.

Admission is:
1. finite-value validation;
2. geometric sanitization;
3. confidence-ranked deterministic ordering;
4. bounded actor admission;
5. global assignment;
6. adaptive temporal prediction.

This prevents lower-quality detections from displacing stronger actors merely because they appeared earlier in a detector result list.

### 3. Body-space shape prior for cross-camera identity

CrossCameraActorIdentityFusionEngine now carries world landmarks in reference tracks and includes a body-space bone-length signature in identity scoring.

The world-shape score is based on translation/orientation-invariant bone lengths with robust median/MAD treatment.

This is intentionally a secondary identity cue because learned body-space coordinates are not a substitute for calibrated camera geometry.

### 4. Multi-actor transport v3

MultiActorPoseWireCodec is now version 3 while retaining version 1 and version 2 decode compatibility.

The v3 actor extension preserves:
- bounded body joints;
- bounded body-space world joints;
- prediction confidence;
- motion innovation;
- acceleration-state indication;
- global-assignment boundedness.

The format keeps the current 32 KiB Nearby transport safety contract and uses 33 body/world joints, with bounded fragmentation for oversized 50+ actor evidence.

### 5. Crowd-motion evidence authority

ProductionCrowdMotionEvidenceEngine provides a bounded per-frame evidence layer for dense scenes.

It evaluates:
- actor visibility;
- temporal stability;
- camera-compensated motion;
- crowd pressure;
- interaction and contact/occlusion pressure;
- identity risk;
- prediction confidence;
- fairness aging;
- body-space landmark coverage;
- body-space bone consistency;
- detail scheduling capacity.

The engine emits per-actor detail budgets and a global bottleneck so optional dense work can be reduced without removing canonical body identity continuity.

### 6. 50-actor dense-landmark scaling

The global dense-landmark evidence configuration is now sized for 50 actors while remaining bounded:
- 32 camera evidence slots;
- 50 subject slots;
- 1,024 landmark slots per subject;
- 64,000 retained fused landmarks;
- 65,536 observations per frame.

The limits remain explicit to prevent dense-scene memory growth from becoming unbounded.

### 7. Identity-exchange scaling

The production identity-exchange risk path is aligned to the 50-actor target. The separate qualification profile remains more restrictive by design because benchmark qualification and live capture capacity are different contracts.

### 8. Operator telemetry

The high-end studio now shows:
- crowd identity safety;
- interaction pressure;
- camera-compensated differential motion;
- optional-detail scheduling capacity;
- predicted actor count;
- held actor count;
- body-space 3D landmark coverage;
- body-space bone consistency;
- 3D-prior reliability;
- active bottleneck and warnings.

The operator can therefore see whether the 50-person system is limited by identity ambiguity, visibility, timing, runtime pressure, or interaction density instead of receiving a single opaque confidence number.

### 9. Lifecycle hardening

The stateful multi-person MediaPipe worker now shuts down in an ordered manner:
1. stop accepting new closure work;
2. reset the monotonic timestamp sequencer;
3. request executor shutdown;
4. wait for in-flight work for a bounded period;
5. escalate to immediate shutdown only if needed;
6. reset identity state;
7. release the landmarker.

This reduces lifecycle races and avoids replacing or closing the stateful landmarker while another worker is actively executing it.

## 3D reconstruction authority

The current reconstruction stack remains:

Camera2 and peer evidence
-> temporal normalization
-> body/face/hand perception
-> global multi-person identity
-> calibrated 2D observation assembly
-> robust triangulation
-> directional uncertainty estimation
-> Kalman/temporal filtering
-> kinematic and physics constraints
-> bundle refinement
-> multi-evidence fusion
-> global 4D consistency
-> dense surface / TSDF / Gaussian evidence
-> production quality gates
-> structured motion output.

The new body-space world landmark path enters as an additional prior and identity signal. It does not bypass calibration or uncertainty gates.

## Precision statement

A requested target such as <1 mm remains a qualification target.

NoX may expose modeled uncertainty, feasibility, or evidence quality, but a practical sub-millimetre result must be established against an independent reference system under matched capture conditions with:
- calibrated camera geometry;
- synchronized timing;
- defined reference uncertainty;
- repeat measurements;
- region-wise statistics;
- P95/P99 error;
- temporal latency;
- identity continuity;
- dropout rate;
- occlusion cohorts.

Likewise, NoX does not claim to be universally better than infrared or marker-based optical motion capture from architecture alone.

## File-size and source-quality rule

The project does not pad files with repeated boilerplate to satisfy arbitrary line counts. Large source files already exist where a domain genuinely requires substantial logic, but production quality is controlled by:
- explicit contracts;
- finite state machines;
- bounded memory;
- deterministic scheduling;
- evidence provenance;
- uncertainty propagation;
- failure attribution;
- testable modules;
- real runtime integration.

For that reason, this revision adds real engines and data paths rather than artificial source volume.

## Validation boundary for this revision

No Android/Gradle build, runtime instrumentation test, or Python simulation was executed in this revision.

The change set was reviewed through repository-level static inspection and contract tracing only, in accordance with the requested validation constraint.

## Upgrade acceptance criteria for the next physical qualification phase

The implementation is ready for a real-device qualification phase only when the following evidence is available:
1. a documented multi-camera rig geometry;
2. synchronized camera capture with measured timing residuals;
3. a reference motion-capture system;
4. repeated multi-person trials, including close interaction;
5. controlled occlusion trials;
6. dynamic motion trials;
7. 50-actor load trials with recorded thermal/memory/latency telemetry;
8. region-wise body/face/eye/hand error statistics;
9. P95/P99 reconstruction error;
10. identity-switch/dropout statistics;
11. surface reconstruction coverage and temporal topology statistics.

The engineering architecture is designed to collect those measurements without converting architectural confidence into an unsupported physical-accuracy claim.
### 10. Aligned body-space prior in calibrated 3D reconstruction

BodySpacePriorAlignmentEngine fits a bounded robust similarity transform between learned body-space world landmarks and jointly observed calibrated metric joints.

The transform uses:
- weighted anatomical anchors;
- robust Huber reweighting;
- quaternion-based proper rotation estimation;
- bounded scale;
- a limited coordinate-handness hypothesis;
- residual and fit-quality rejection.

Aligned priors are capped below authoritative multi-view confidence and carry a hard uncertainty floor. Priors can stabilize or fill weak joints but cannot become a substitute for calibrated metric geometry.

MultiEvidence3DReconstructionEngine now treats this source as BODY_SPACE_PRIOR with lower source weight and its own correlation group, preventing duplicate confidence inflation.

### 11. Native runtime load balancing

NativeEngineRuntime now keeps camera-shard preference but adds queue-depth-aware work stealing. A worker first tries its preferred camera shard and then can consume other non-empty camera queues when its own shard is idle.

This preserves deterministic worker ownership where possible while allowing a hot camera to use otherwise idle CPU capacity. Camera queue depth remains bounded, and worker lifecycle remains fail-safe through joined shutdown and processor recreation on worker faults.
## Correlation-aware metric uncertainty
Triangulation uncertainty now consumes an explicit cross-view-independence factor. Redundant cameras that share common-mode timing, calibration, crop, or inference errors no longer receive full independent information credit. The resulting view-independence score is exposed alongside anisotropic covariance and emits degraded/hard-risk warnings when the apparent view count is statistically misleading.
