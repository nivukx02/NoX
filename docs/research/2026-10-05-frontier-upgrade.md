# NivukxNoX Frontier Upgrade Record — 2026-10-05

## Objective

This continuation upgrades the existing NoX production motion-capture architecture without removing established capability. The focus is physical-evidence integrity, multi-camera scalability, multi-actor fairness, dense 3D reconstruction robustness, memory/crash containment, and operator-grade precision UX.

The implementation intentionally does not turn source-code targets into unsupported physical claims. A requested sub-millimeter target, an empirically measured sub-millimeter result, and measured superiority against an optical/infrared reference remain separate states.

## Implemented architecture upgrades

### 1. Exposure-center timing authority

ExposureMidpointTimingEngine models the temporal center of an exposure using sensor timestamp semantics, exposure duration and rolling-shutter skew. It also estimates row timing burden, blur burden, uncertainty and confidence reduction.

RollingShutterEvidenceEngine now consumes this timing authority, so rolling-shutter evidence and downstream calibrated geometry are penalized when acquisition timing is materially uncertain.

### 2. Fair 50-actor detail scheduling

ActorWorkloadScheduler adds deterministic, bounded scheduling across the existing multi-actor reconstruction path.

Priority combines motion information, occlusion/identity risk, confidence, requested detail tier, primary-actor priority, continuity and starvation aging. Actors can be held from the most expensive detail tier without losing their identity or base reconstruction path.

### 3. Authoritative 32-view plus 96-witness camera federation

The existing 128-camera distributed envelope remains intact. Dense reconstruction remains bounded by an authoritative view budget, while MultiViewCameraContributionPlanner now exposes a bounded witness-camera set from excluded views.

Witness cameras are therefore still represented in the rig decision surface instead of disappearing just because they are not selected for the most expensive reconstruction pass.

### 4. Stronger robust triangulation

TriangulationEngine now uses deeper nonlinear refinement defaults, an additional consensus refinement pass, larger bounded probabilistic observation-mode support, finite-safe Huber/Cauchy/Tukey-style residual weighting, and a robust objective blending quadratic and logarithmic tail penalties.

### 5. Independent frontier precision qualification

FrontierPrecisionQualificationEngine is integrated into MultiViewFusionEngine alongside the existing UltraPrecisionEvidenceEngine.

The second gate distinguishes disabled qualification, insufficient evidence, evidence-ready but unmeasured, measured target attainment, matched-reference qualification, and hard failure.

No reference superiority state can be produced without matched external-reference evidence. Explicit empirical sample counts are required; observation volume is not substituted for reference-trial samples.

### 6. Native memory admission containment

The hardware-buffer lease registry now has bounded active-lease and resident-memory admission limits. The registry rejects new acquisitions before they can exceed its hard memory budget, and it releases native AHardwareBuffer references on all failure paths.

This complements existing memory-lease auditing, stale-lease reclamation and ordered native shutdown.

### 7. Native diagnostics latency containment

The native diagnostics path is bounded to a short worker-progress observation window so operator telemetry cannot create a multi-second synchronous stall while waiting for worker progress.

## Research-derived design inputs

### Camera timing

Android Camera2 exposes sensor-timestamp and rolling-shutter-skew metadata with defined timing semantics. NoX therefore treats sensor timing as measured evidence and propagates its uncertainty rather than assigning a single ideal timestamp to every pixel.

Primary documentation:
- https://developer.android.com/reference/android/hardware/camera2/CaptureResult
- https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics

### Multi-camera systems

Android supports concurrent operation for multiple cameras on compatible devices, but the platform does not imply that arbitrary external cameras become a single perfectly synchronized sensor. NoX keeps camera admission, timestamp evidence, calibration and rig authority as separate layers.

Primary documentation:
- https://developer.android.com/media/camera/camerax/configuration

### Multi-view 3D human tracking

CVPR 2026 LAMP uses camera calibration and 6-DoF poses to lift multi-camera observations into a unified metric 3D world frame before temporal human-motion fitting. NoX already follows this world-space evidence direction and now strengthens the timing, robust geometry and actor-scheduling layers around it.

Paper:
- https://openaccess.thecvf.com/content/CVPR2026/html/Yang_LAMP_Localization_Aware_Multi-camera_People_Tracking_in_Metric_3D_World_CVPR_2026_paper.html

### Multi-person motion acquisition

CVPR 2026 MAMMA reports a markerless multi-person acquisition framework that explicitly uses dense body-surface landmarks, uncertainty, visibility and contact evidence, with segmentation conditioning for close interaction and cross-view identity matching using geometric consistency. NoX already exposes contact, visibility, uncertainty, cross-view and identity evidence; the current continuation strengthens those channels with fair actor-detail scheduling, pre-update identity ambiguity, and dense reconstruction gating.

Primary source:
- https://openaccess.thecvf.com/content/CVPR2026/papers/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.pdf

### Global 4D reconstruction

CVPR 2026 TROPHIES couples humans, scenes and cameras with global alignment, scale consistency, contact priors and cross-view temporal coherence. NoX's existing global-4D, world-space, contact and Gaussian consistency stack is retained; this continuation makes the supporting evidence and qualification boundary more explicit.

Paper:
- https://openaccess.thecvf.com/content/CVPR2026/html/Liu_TROPHIES_Temporal_Reconstruction_of_Places_Humans_and_Cameras_from_Multi-view_CVPR_2026_paper.html

### Operator UI

Apple's current Liquid Glass guidance emphasizes hierarchy, adaptive materials, legibility, predictable controls and restraint in applying effects. NoX keeps Liquid Glass limited to operator control/surface layers while preserving direct access to precision-critical telemetry.

Primary documentation:
- https://developer.apple.com/documentation/TechnologyOverviews/liquid-glass
- https://developer.apple.com/documentation/TechnologyOverviews/adopting-liquid-glass
- https://developer.apple.com/design/human-interface-guidelines/layout

## Accuracy boundary

No source-code change can legitimately establish universal less-than-1-mm real-world positional error across arbitrary consumer cameras, lighting, distances, baselines, lens states, thermal conditions, occlusion patterns and motion speeds.

NoX therefore keeps the following states separate:

1. requested target;
2. modeled uncertainty;
3. measured capture error;
4. matched-reference comparison;
5. repeatability across the operating envelope.

A statement that NoX is better than an infrared optical reference must remain an empirical benchmark result under matched conditions, not a compiler-time property.

## Validation boundary for this continuation

This continuation was performed by source inspection, architecture tracing, code modification and repository-state verification.

No Android/Gradle build was started.

No Python simulation was run.

No claim of measured sub-millimeter performance was produced from source inspection alone.


## Continuation pass: 2026-10-05 18:34 IST

Additional production hardening completed on main:

- **Stable identity:** re-identification now uses bounded global Hungarian assignment rather than greedy candidate consumption. Configuration validation rejects invalid match weights and geometry gates.
- **Rig identity accounting:** truncated current components are reported as unmatched, and global identity confidence now includes coverage of unmatched current components.
- **Cross-camera scalability:** subject graph construction no longer scans every observation pair. Calibrated views use bounded spatial buckets; uncalibrated fallback uses bounded temporal neighborhoods; candidate edges are retained through bounded top-K priority queues.
- **Metric trust boundary:** uncalibrated cross-camera associations remain useful for identity continuity but are capped below calibrated metric-association confidence.
- **Temporal synchronization:** spatiotemporal bundle adjustment can seed its existing robust offset refinement with bounded dynamic-time-warp alignment derived from measured motion magnitudes. The DTW seed is advisory; the existing reprojection optimizer remains authoritative.
- **Face/eye timing trust:** live face/eye fusion now down-weights evidence using measured capture timing quality, including rolling-shutter and exposure timing confidence.
- **Eye identity robustness:** fallback eye-track keys now include the full quantized face bounds and center geometry, reducing collisions between close unlabeled faces.
- **Native triangulation conditioning:** 3x3 solves use scale-aware pivot floors, and robust median computation is unbiased for even sample counts.
- **Native calibration conditioning:** calibration 3x3 solves use the same scale-aware pivot strategy.
- **Dense precision qualification:** readiness now requires explicit evidence sufficiency before entering READY; configurable full-evidence sample count is part of the contract.
- **Global 4D consistency:** root acceleration now has an explicit production envelope, and duplicate contact warnings were removed.
- **Gaussian surface fusion:** visibility and photometric evidence are propagated through the Kotlin/JNI/native batch contract and influence fusion weight and opacity; normal consistency is part of primitive association.
- **Large-rig capacity:** 64 actor identity/fusion ceilings remain aligned with the production 50-actor target, while the camera federation continues to use a bounded 128-camera envelope.

### Research-to-implementation notes

The current continuation pass uses research-backed principles rather than claims of magical physical accuracy:

1. Dynamic temporal alignment is treated as a bounded initialization problem before robust reprojection refinement.
2. Dense surface updates require observed visibility/photometric support and do not let invisible primitives accumulate full posterior influence.
3. Operator UI remains progressive enhancement: material effects must degrade without destroying hierarchy or readability.
4. Physical superiority over infrared/optical reference systems is still an empirical qualification question. Source code alone does not establish a sub-millimeter or better-than-reference measurement result.

### Validation boundary

No Android/Gradle build was started in this continuation pass.
No Python simulation was executed.
No runtime test suite was executed.
The repository was source-audited and focused regression tests were added/updated without executing them.


## Continuation pass: temporal authority, scheduler hysteresis and depth-aware rig evidence

This pass extends the existing frontier architecture with four concrete production controls.

### 8. Dense reconstruction temporal idempotence

Exact duplicate dense-frame timestamps are now treated as replayed evidence rather than a fresh measurement. The previous authoritative result is returned without reinjecting position, velocity, acceleration or precision evidence. Older timestamps remain hard rejected.

Dense precision qualification is also stricter at the point and frame levels: tail-adjusted feasibility, tail-adjusted depth error, reprojection-tail bounds, independent-view count, view diversity and a qualifying-point ratio now participate in the precision authority decision.

### 9. Stateful pressure hysteresis

ProductionCaptureStageScheduler now consumes its existing confirmation/recovery settings as actual stateful hysteresis. Severe pressure can shed optional work immediately, while non-severe pressure must persist for the configured confirmation window before optional fidelity is reduced. Recovery likewise ramps optional stages through reduced quality until the configured healthy-frame window completes.

This prevents pressure flapping from repeatedly reallocating dense reconstruction, face/eye, hand and global-4D work every frame.

### 10. Depth-aware multi-camera identity graph

When calibrated depth is available, cross-camera candidate generation now uses bounded three-dimensional spatial cells instead of only planar x/y buckets. Metric association also uses the depth dimension when both observations carry depth.

The temporal fallback remains bounded for partially calibrated rigs, preserving continuity without allowing incompatible metric depth to masquerade as a valid cross-view geometric match.

### 11. Memory-budget authority

MotionRuntimeResourceGovernor now derives an effective memory-pressure signal from allocated lease bytes versus its configured byte budget, then combines that signal with caller telemetry. This prevents an under-reported pressure metric from allowing the governor to claim nominal operation while its own hard allocation envelope is nearly exhausted.

### 12. Liquid Glass readability hierarchy

The operator surface now uses an explicit adaptive readability plate between the glass shell and content. Plate opacity reacts to measured content contrast, transparency settings and operator focus. This keeps telemetry legible on high-detail reconstruction imagery without introducing expensive blur as a dependency.

### Validation boundary

No Android/Gradle build was started.
No Python simulation was run.
No runtime/unit test suite was executed.
Focused regression tests were added for dense timestamp idempotence, scheduler pressure hysteresis, global-4D timestamp idempotence, depth-aware cross-camera separation, memory-budget pressure and Liquid Glass source integrity.



## Continuation pass: 2026-10-05 large-rig fairness and modern 4D evidence alignment

The implementation was cross-checked against additional CVPR 2026 work after the previous continuation:

- **LAMP** reinforces the value of lifting multi-camera observations into a shared metric 3D world before temporal fitting. NoX's calibrated world-space association, timing authority and robust metric gates follow the same evidence direction. citeturn436476search0
- **MAMMA** emphasizes dense visibility/contact-aware surface evidence and evaluates markerless reconstruction against a commercial optical reference. NoX consequently keeps visibility/contact/uncertainty as first-class inputs while reserving physical superiority for matched empirical qualification. citeturn436476search35
- **TROPHIES** couples people, scenes and cameras with global alignment, scale consistency, contact priors and cross-view temporal coherence. NoX's global-4D authority, rig evidence and timestamp monotonicity are hardened around these same failure boundaries. citeturn817416search0
- **SparseCam4D / 4C4D / Any4D** show that high-fidelity 4D reconstruction is increasingly organized around spatio-temporal consistency, scalable representations and explicit geometry-vs-appearance handling. NoX retains a calibrated authoritative path while allowing broader witness/federation evidence without forcing every camera through the most expensive solver. citeturn817416search1turn817416search4turn817416search9
- **Apple's Liquid Glass guidance** explicitly separates the functional glass layer from the content layer and prioritizes legibility and hierarchy. The latest NoX operator-surface change therefore adds an adaptive readability plate below live telemetry rather than multiplying decorative glass effects across the reconstruction viewport. citeturn436476search1turn436476search4

### Newly verified production controls

- Dense reconstruction now treats duplicate timestamps as idempotent replay and tightens point/frame precision authority using tail-adjusted metrics, reprojection limits, independent-view count, view diversity and qualifying-point ratio.
- Capture-stage pressure uses confirmation/recovery hysteresis to prevent resource-driven oscillation while retaining immediate severe-pressure shedding and deferred-stage starvation limits.
- Global 4D ignores duplicate and older timestamped samples without mutating temporal kinematics.
- Cross-camera association uses calibrated depth in both candidate indexing and metric distance checks.
- Runtime memory pressure is derived from the governor's actual lease budget as well as externally reported pressure.
- A 50-actor fairness/risk-priority regression and a 128-camera authoritative/witness envelope regression were added without executing the test suite.

No build or Python simulation was run.
## Continuation pass: live timing authority, stale-detector containment, and crowd identity contradiction gating

This continuation closed three production-path gaps discovered by direct repository tracing.

### 13. Live face/eye timing propagation

PerceptionPipeline already measured frame duration, exposure duration, rolling-shutter skew and capture timing confidence and supplied them to pose, face and face-mesh processing. The face-eye fusion call was previously dropping those measurements. The live call now forwards the same timing evidence into FaceEyeFusionEngine so gaze/head confidence is reduced when the actual capture timing is poor.

### 14. Detector freshness authority

A detector fallback can be represented as a successful task result even when the value is actually the previous frame. The perception boundary now tracks faceEvidenceFresh separately from task success. Face fallback therefore cannot be reclassified as fresh merely because the task continuation completed successfully.

### 15. Crowd identity contradiction gate

MultiActorPoseAssociationEngine already combines appearance, geometry, skeleton, velocity, uncertainty and detector identity evidence. The live identity contract now adds a hard contradiction gate: when high-quality appearance descriptors are available but strongly disagree, that candidate assignment is rejected rather than being allowed through because of incidental geometric overlap.

This is particularly important in close interaction and actor-crossing cases, where geometry alone is underdetermined.

### Research alignment

The latest literature sweep reinforces the architecture:
- Recent gaze uncertainty research emphasizes post-hoc empirical calibration and actionable selective reliability under domain shift.
- Modern asynchronous/rolling-shutter multi-camera estimators model per-row timing and freeze temporal calibration under degenerate motion instead of letting offsets drift on noise.
- Multi-camera dynamic reconstruction work continues to rely on shared metric world coordinates, explicit temporal coherence, visibility/contact evidence and separated geometry/appearance signals.

No build was started, no Python simulation was run, and no runtime/unit test suite was executed.
## Continuation pass: reference qualification integrity

The production benchmark gate now requires a reference comparison to be both condition-matched and statistically separated before `gatePassed` can become true. A reference result that is present but inconclusive can no longer certify a benchmark.

This follows the current validation direction in markerless MoCap research: internal stability and temporal agreement are useful for configuration screening, but they do not establish measurement accuracy against a professional reference system. External validation must remain explicit and matched to the evaluated capture conditions.

The benchmark regression suite now covers the failure case where a matched reference comparison exists but uncertainty is too large to establish statistical separation.

## Continuation pass: calibrated-geometry uncertainty provenance

SelfCalibrationEngine now computes every camera uncertainty profile from the exact intrinsics and extrinsics that will be published in the final CalibrationSnapshot. When native intrinsic refinement changes the calibration after bundle adjustment, uncertainty is no longer derived from the pre-refinement camera state or pre-refinement reprojection RMS.

This preserves an evidence invariant across the reconstruction boundary: calibration geometry, reported reprojection error, and calibration uncertainty must describe one coherent snapshot. Downstream multi-view triangulation, face mesh fusion, cross-camera identity fusion, and session recording consume these profiles directly.


## Continuation pass: P95-tail uncertainty and empirical qualification floor

The production benchmark authority received two metrology hardening changes.

### 16. Quantile-specific uncertainty

Reference superiority now estimates uncertainty for the **P95 tail statistic itself** using a deterministic, bounded bootstrap over benchmark errors. The previous robust standard error described dispersion of the raw error samples, which is not the same statistical object as uncertainty of a tail quantile. The combined separation test now therefore combines P95 bootstrap uncertainty with the supplied reference uncertainty.

The bootstrap is deterministic for reproducible qualification reports and is confined to benchmark completion rather than the real-time frame path.

### 17. Minimum empirical qualification evidence

Benchmark `gatePassed` now requires at least 600 empirical samples by default, matching the existing frontier precision qualification evidence floor. A small or cherry-picked benchmark cannot certify a production capture claim merely because its observed P95 happens to be favorable.

The focused regression suite now covers both variable-tail P95 uncertainty and rejection of a 599-sample benchmark. No runtime test execution was performed.

## 2026-10-05 frontier continuation: geometry, surface and ocular authority

This continuation hardens three previously bounded frontier seams without removing the existing perception/reconstruction stack.

### Global human-scene 4D refinement

GlobalHumanScene4DOptimizationEngine now accepts temporal observations, per-camera world-space observations, fixed-camera gauge declarations and surface constraints. Refinement uses bounded robust IRLS evidence weighting, temporal continuity regularization, kinematic diagnostics for velocity/acceleration/jerk envelopes, and conservative camera-translation correction. Camera rotation authority remains outside this layer, so the optimizer cannot silently rewrite calibrated orientation state.

The optimizer exposes point-level residual diagnostics, trajectory envelope scores, rejected-observation counts and optimized camera-correction counts. A worsening solution is rolled back to the previous accepted state.

FrontierReconstructionOrchestrator now passes explicit global surface constraints into this solver instead of dropping them at the orchestration boundary.

### Frontier identity assignment

MultiActorHypothesisAssociationEngine previously resolved primary identities in descending local candidate order. The frontier path now uses a deterministic bounded Hungarian assignment across the complete observation/actor candidate matrix. Geometry, metric depth, appearance, detector confidence and detector identity hints remain separate evidence terms, with hard geometry/depth/appearance contradiction gates. Alternate hypotheses remain bounded and explicit rather than silently replacing the globally optimized primary assignment.

### Sparse-view human reconstruction

SparseViewHumanReconstructionEngine now performs robust multi-view position consensus and carries per-view confidence, visibility, photometric consistency, normal consistency, declared uncertainty, temporal support and learned-prior support. Evidence provenance remains explicit:

- OBSERVED: strong measured multi-view geometry;
- GEOMETRICALLY_CONSTRAINED: sufficient geometry with weaker evidence;
- TEMPORALLY_PROPAGATED: temporal recovery without enough current geometry;
- LEARNED: bounded learned completion;
- INSUFFICIENT_EVIDENCE: fail-closed state.

Learned geometry is only allowed to move a geometrically supported estimate by a bounded amount and cannot promote an insufficient observation into an observed state.

### Dynamic Gaussian surface refinement

DynamicGaussianSurfaceRefinementEngine now adds deterministic spatial/semantic deduplication, robust local neighborhood consensus for surface position and normals, uncertainty-aware authority, provenance-aware confidence, temporal stability, silhouette consistency and motion-boundary evidence. Inferred Gaussian covariance remains inflated so completion cannot masquerade as measured geometry.

### World-space gaze

WorldSpaceGazeReconstructionEngine now uses uncertainty/freshness/occlusion weighted spherical consensus with repeated robust refinement, cross-eye consistency weighting, weighted P95 angular uncertainty, explicit outlier counts and binocular-disagreement warnings. Blink and weak eyelid observations remain excluded from the world-space estimate.

### Native resource contract hardening

hardware_buffer_lease_registry.h now declares the same maximum-budget accessor methods already implemented by hardware_buffer_lease_registry.cpp, closing a concrete C++ declaration/definition interface mismatch in the native memory safety layer.

### Qualification boundary

These changes improve the measurement architecture and failure containment. They do not establish a physical guarantee of less than 1 mm accuracy, nano-scale reconstruction accuracy, or superiority to infrared/active optical reference systems. Those claims require matched-reference calibration, synchronized ground truth, repeatability testing and statistically valid qualification under the exact operating regime. The repository already contains reference-qualification infrastructure; this continuation keeps that boundary intact.

No Python simulation was used, and no Android application build was manually launched during this continuation.

### Current CVPR/UI alignment after this continuation

CVPR 2026 MAMMA reinforces the value of visibility/contact-aware dense surface evidence, person-specific correspondences and robust multi-person capture under close interaction. NoX now carries these ideas through its explicit sparse-region evidence, provenance and global identity contracts rather than treating one detector score as scene truth.
HumanBA (CVPR 2026) demonstrates that reliable human motion can be informative camera evidence instead of simply being masked away. NoX's global 4D layer now consumes bounded camera-linked human geometry while preserving camera gauge and rollback protections.
SV-GS (CVPR 2026) and RetimeGS (CVPR 2026) reinforce skeleton-guided sparse-view deformation and continuous-time 4D Gaussian reasoning. NoX already contains continuous-time surface state; the current frontier continuation adds stronger neighborhood, uncertainty and provenance arbitration to the Gaussian refinement stage.
Apple's current Liquid Glass guidance emphasizes functional glass layers for controls/navigation, clear hierarchy, restrained use of glass and preservation of readability/accessibility. The new evidence ladder therefore uses the existing adaptive glass surface as an operator instrument rather than making every content element translucent.

### New continuation integration points

- Sparse reconstructed regions automatically become bounded global 4D surface constraints when actor/point identities match.
- World-space gaze now exposes a metric binocular target, target uncertainty and vergence angle when both eyes provide geometrically useful rays.
- Face/eye authority consumes the new gaze confidence and binocular agreement terms.
- Precision qualification now distinguishes an actual measured target exceedance from missing measurement evidence.
- The native hardware-buffer registry header now matches its implementation budget-accessor contract.

All additions remain bounded, deterministic where practical, provenance-aware, and additive to the existing NoX capture modes.