# NoX High-End Motion Capture Upgrade, 2026-09-27

## Research-driven upgrade decisions

Recent CVPR 2026 work reinforces that high-quality 4D reconstruction is constrained by sparse-camera geometry and by spatial/temporal inconsistency. SparseCam4D explicitly models spatio-temporal distortion in sparse-camera reconstruction, while 4C4D demonstrates high-fidelity 4D reconstruction from only four portable cameras. These results support keeping geometry, synchronization, temporal continuity, and view selection as first-class NoX signals rather than relying on a single pose-confidence number.

Recent benchmark work on complex markerless motion capture also emphasizes severe occlusion, identity exchange, realistic multi-person interaction, and synchronized ground truth. NoX therefore continues its existing uncertainty, identity, calibration, and temporal evidence pipelines and adds stronger bounded evidence aggregation and failure stratification.

## New production modules in this cycle

### 1. AdaptiveMotionEvidenceEngine
Separates common-mode camera motion from differential subject motion and tracks motion at position, velocity and acceleration scales. It also exposes micro-motion, vibration, occlusion, timing, and temporal-confidence signals.

### 2. AnisotropicReconstructionUncertaintyEngine
Stops treating 3D uncertainty as isotropic. It explicitly penalizes shallow parallax and weak angular diversity, producing separate lateral/depth uncertainty budgets and an anisotropy ratio.

### 3. CaptureRuntimeResilienceEngine
Adds rate-limited recovery decisions for long sessions. Critical memory/thermal pressure blocks capture; queue backpressure drains or reduces optional stages before escalating to native-session restart.

## Benchmark principle

NoX should not claim superiority over infrared/marker optical motion capture from architecture alone. A credible superiority claim requires a controlled comparison with matched capture conditions, documented reference-system uncertainty, region-wise error statistics, dropout, temporal lag, identity continuity, and sufficient samples.

## UI/UX direction

Apple's current Liquid Glass guidance places the material primarily in the functional/navigation layer and warns against crowding or stacking glass layers. Android's current adaptive guidance emphasizes responsive layouts, navigation rails on large displays, keyboard/mouse support, and at least 48dp touch targets. NoX's existing Liquid Glass chrome and 48dp controls are therefore preserved and extended through additional operator settings rather than glazing the reconstruction content itself.

## Quality constraints

- No Android build was started in this cycle.
- New runtime state is explicitly bounded.
- Recovery actions are rate-limited.
- Uncertainty and degradation are surfaced rather than hidden.
- Existing motion, face/eye, hand, mesh, calibration, synchronization, and 4D paths remain additive.


## Reference-grade metrology implemented

This cycle adds MetricPrecisionBudgetEngine, a conservative uncertainty budget for a requested 2 mm P95 envelope. It explicitly models image reprojection, stereo baseline/parallax, depth uncertainty, camera-pose uncertainty, temporal skew, rolling shutter, calibration prior, metric scale, and reconstruction prior uncertainty.

The model is deliberately not presented as proof of practical 2 mm accuracy. Reference-grade mode is evidence-gated. With quality lock enabled, NoX blocks the mode when the modeled P95 exceeds 2 mm or when calibration, synchronization, multi-view diversity, cross-view agreement, or observation-count requirements are not met.

The operator UI exposes the contract and the telemetry surface can carry modeled P95/P99, feasibility, required baseline, recommended baseline, required focal length, and minimum parallax guidance.

## Camera-rig integration

MotionCaptureCustomizationEngine.resolveWithRig(...) now evaluates active camera health and geometry with CaptureRigObservabilityEngine, selects a deterministic subset, feeds the resulting view diversity and cross-view agreement into the production quality planner, and propagates selected camera IDs and rig warnings into the resolved plan.

This makes camera count an evidence input rather than a vanity number: adding redundant cameras does not automatically increase the quality budget.


## This cycle: bounded temporal fusion and empirical precision provenance

### 4. HighEndTemporalFusionEngine
Adds a bounded multi-camera temporal state estimator with robust cross-view fusion, Huber-style residual weighting, duplicate-camera rejection, constant-acceleration prediction, confidence half-life decay, explicit WARMING/TRACKING/OCCLUDED/LOST states, finite velocity/acceleration limits, deterministic capacity eviction, and reset semantics. Prediction is time-bounded so prolonged occlusion cannot silently become authoritative fabricated motion.

### 5. ReferenceGradeEvidenceGateEngine
Separates modeled 2 mm feasibility from measured qualification. Reference-grade mode is blocked until matched capture conditions, a passed empirical benchmark, sufficient observations, measured P95/P99, documented reference provenance, and reference-system uncertainty are available. The uncertainty-adjusted target is conservative rather than allowing the reference system's own uncertainty to disappear from the budget.

### 6. Runtime integration and operator truth
MotionCaptureCustomizationEngine now carries empirical reference evidence and temporal-fusion summaries into its production plan. Timestamp-regression recovery resets temporal state rather than crashing the capture policy. The Liquid Glass operator surface now exposes modeled versus measured precision provenance and explicitly shows when empirical evidence is still required.

### 7. Accuracy claim discipline
The target remains an engineering contract, not a guaranteed field result. A real claim of <2 mm or superiority to an infrared/marker reference still requires controlled measurement against a documented reference under matched conditions. The implementation intentionally refuses to turn modeled covariance or architectural sophistication into fabricated accuracy.


## Latest research pass: 2026-09-27

### Occlusion, interaction and identity
CVPR 2026 MAMMA reinforces the need for visibility-aware and contact-aware dense landmarks under heavy multi-person occlusion. HUM4D/ReGen4D further shows that realistic interaction, identity exchange, temporal inconsistency and synchronization expose large generalization gaps in markerless systems. NoX responds with contact-preserving occlusion evidence, graph-consistent cross-camera association, bounded temporal recovery and explicit identity-merge suppression.

Sources:
- https://openaccess.thecvf.com/content/CVPR2026/html/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.html
- https://openaccess.thecvf.com/content/CVPR2026W/ReGen4D/html/Park_A_Dataset_and_Evaluation_for_Complex_4D_Markerless_Human_Motion_CVPRW_2026_paper.html

### Calibration-light mobile capture
Kineo demonstrates a practical direction for calibration-free metric capture from unsynchronized, uncalibrated consumer RGB cameras using confidence-driven sampling, simultaneous camera recovery and graph optimization. NoX therefore keeps relative-only, prior-scaled metric and explicitly calibrated metric modes distinct rather than silently treating all RGB reconstruction as metric.

Source:
- https://arxiv.org/html/2510.24464

### Face and eye tracking
Recent 2026 gaze work emphasizes temporal modeling, personalized calibration, uncertainty tied to observable input factors, and joint 3D eye reconstruction. PicoEyes combines 3D eye parameters, segmentation, depth, gaze calibration and forecasting; ST-Gaze reports that preserving spatial structure before temporal recurrence improves video gaze estimation; FIUD explicitly ties uncertainty to interpretable factors such as illumination and sharpness.

Sources:
- https://arxiv.org/html/2605.07188v2
- https://arxiv.org/abs/2608.31033
- https://arxiv.org/html/2607.20072v1

NoX now schedules face/eye inference using sharpness, eye visibility, uncertainty, lighting and temporal freshness, and rejects stale face/eye temporal state rather than allowing old measurements to rewrite current gaze.

### Operator reliability and self-healing
Professional mocap products expose camera health, automated calibration recovery, system-health telemetry and deterministic multi-camera operator workflows. NoX now separates KEEP, HEAL_CANDIDATE, RECALIBRATION_REQUIRED and HOLD states, retains diagnostics for excluded cameras, and bounds long-session replay, resource and peer-state memory.

Source:
- https://www.vicon.com/software/evoke/

### Mobile memory and scheduling
The runtime path now adds hard byte-budget admission control, concurrent allocation peak telemetry, bounded replay-peer state, quality-aware camera-view selection and stale-state protection across face, eye, hand, surface and world-space reconstruction. These controls are intended to make long sessions fail conservatively rather than drift into unbounded memory growth or stale evidence reuse.

## Current implementation principle

High-end quality is defined by measurable geometry, uncertainty, temporal integrity, recovery behavior and reproducible evidence. File line count is not used as a proxy for production quality. NoX continues to add independent, testable production modules rather than padding individual files to arbitrary sizes.


### Temporal integrity and long-session resilience

The latest implementation pass hardened temporal provenance across the global quality policy, world-space reconstruction, canonical surface history, bilateral eye fusion, face stabilization, hand uncertainty calibration, subject-track lifecycle, and native temporal correction. Each stateful layer now rejects non-increasing evidence timestamps instead of coercing them into a normal frame interval.

Runtime resilience was also strengthened with hard byte-budget admission control, bounded replay-peer state, and native outstanding/peak allocation telemetry. The allocation telemetry distinguishes cumulative calls from concurrent live allocations so peak memory analysis is meaningful.

These choices align with the broader production principle that stale asynchronous evidence must be discarded at the state boundary rather than “smoothed” into the current frame.


### Latest implementation pass: interaction physics and synchronization evidence

Current 2026 research on 4D hand-object capture emphasizes factorized spatiotemporal multi-view evidence, volumetric/ARAP constraints, collision handling, and contact-aware refinement. NoX's native interaction path now preserves contact hysteresis while additionally tracking contact age, peak approach rate, impact score, interaction confidence, and stale-evidence rejection. The existing kinematic contact constraints remain the geometry projection authority.

The synchronization path now exposes measured residual P95, outlier ratio, sample count, and monotonic-input integrity. The production gate can warn, block, and contract-check on these measured synchronization diagnostics rather than relying on a single scalar synchronization confidence.

Operator settings expose the maximum synchronization residual and outlier ratio alongside existing timing and reference-grade controls.

The current Liquid Glass UI direction remains intentionally layered: navigation and controls use glass, while dense evidence content retains opaque/regular material where legibility matters. This follows Apple's current guidance to treat Liquid Glass as a functional layer rather than coating every content surface.


### Latest implementation pass: temporal warping, surface deformation and runtime containment

RocSync-style synchronization research reinforces that heterogeneous camera systems need explicit exposure-window timing, offset/drift estimation and rolling-shutter-aware temporal correction rather than a single device timestamp. NoX now extends its spatiotemporal calibration evidence with an explicit per-sample correction object containing clock, scanline and exposure-window terms plus residual uncertainty, while keeping a hard measured residual/outlier contract. The correction API is intended to feed reconstruction timing directly instead of leaving calibration as a UI-only metric.

Dense surface reconstruction now has a dedicated temporal deformation evidence layer. It tracks corresponding world-space vertices, edge-length strain, normal angular change, curvature change, topology continuity, visible coverage and confidence. Hard violations block the resolved capture plan. Operator contracts expose maximum strain P95, minimum temporal stability and minimum topology continuity.

GPU lifetime safety is now represented as a generation-checked lease state machine and is integrated into the runtime resource governor. Zero-copy paths require generation-tracked lease ownership, GPU completion before release, bounded in-flight capacity and watchdog expiry. Telemetry distinguishes active/in-flight leases, stale generations, invalid releases, watchdog aborts and admission rejection.

Runtime fault handling now uses a bounded circuit-breaker policy with fault classification, exponential backoff, recovery budgets and explicit actions for native crashes, GPU device loss, camera stream resets, memory pressure, synchronization failure and reconstruction failure. The policy does not perform hidden recovery itself; it produces deterministic lifecycle actions for the owning runtime.

The native diagnostics ABI was advanced to version 7 to expose GPU lease lifetime counters in the same snapshot as queue, worker and thermal telemetry.

These additions continue the same production rule: architectural sophistication is not treated as proof of field accuracy. Claims such as <2 mm practical precision or superiority to infrared/reference optical mocap remain empirical qualification targets that require a controlled matched-condition benchmark.


### Latest implementation pass: uncertainty-aware face/eye fusion and workload preservation

Recent multi-camera validation work reinforces that temporal errors become increasingly damaging when position estimates are differentiated into velocity and acceleration, while current complex 4D benchmarks continue to expose synchronization, occlusion, and temporal-coherence failures. NoX therefore now carries explicit face/eye uncertainty decomposition and a deterministic workload-preservation policy alongside the existing geometric quality gates.

The face/eye fusion path now separates aleatoric signal weakness, temporal instability, binocular disagreement, gaze innovation, stale evidence, model mismatch, and out-of-distribution risk. The downstream face/eye consistency gate applies that uncertainty to confidence and exposes uncertainty/OOD diagnostics instead of allowing a single high detector score to dominate.

The runtime now includes ProductionCaptureStageScheduler. It protects sensor ingestion, timestamp alignment, primary tracking, multi-view association, triangulation and temporal fusion while reducing or deferring optional dense-surface, face/eye, hand, contact, global-4D and export stages under sustained CPU/GPU/memory/thermal/queue pressure. Deferred work has a bounded retry budget so optional fidelity cannot starve indefinitely, and core motion cannot be silently dropped just to preserve decorative fidelity.

This follows Android Camera2's documented pipeline behavior: camera processing is staged, multiple requests may be in flight, and pipelines can drop frames when processing cannot keep up. The NoX policy therefore makes workload shedding explicit and evidence-visible rather than letting the camera pipeline fail implicitly.

The research ledger continues to treat sub-frame synchronization as a first-order requirement. RocSync reports that even single-digit millisecond offsets can produce reprojection and trajectory errors in geometry-driven multi-view applications and demonstrates a low-cost sub-frame synchronization approach. NoX's synchronization residual/outlier contracts, temporal calibration, and evidence gates remain aligned with that requirement.

The project still does not claim universal <2 mm field accuracy or superiority to marker/infrared reference capture. Those remain empirical benchmark qualifications requiring matched-condition reference measurements, uncertainty accounting, and sufficient repeated samples.


### Dense landmark evidence and multi-camera provenance

The latest implementation pass adds DenseLandmarkEvidenceFusionEngine as an explicit evidence layer between learned 2D perception and downstream reconstruction policy. It carries per-landmark confidence, detector-derived uncertainty, visibility, contact probability, segmentation consistency, residuals, source diversity and bounded temporal prediction. Robust consensus uses a median anchor, Huber weighting and a second hard geometric/depth gate so a high-confidence outlier cannot dominate the fused point.

The dense layer is deliberately bounded for mobile execution: frame-level observation admission is capped, per-landmark observations are capped, landmark/camera state is stale-evicted, and history stores lightweight digests instead of whole frame payloads. These limits are part of correctness because Camera2 pipelines can have multiple in-flight requests and can drop frames when downstream processing cannot keep up.

A critical provenance fix now passes the real runtime camera identifier into every perception frame. This prevents all cameras in a multi-camera session from being mislabeled as camera 0.

The dense engine exposes `crossViewEvidenceAvailable` separately from `crossViewAgreement`. NoX will not interpret unrelated cameras' 2D image coordinates as geometric agreement. Cross-view agreement remains unavailable until the calibrated reconstruction path supplies a common ray/world-space representation. This avoids manufacturing a precision metric from incompatible coordinate frames.

The dense evidence now contributes to both HighEndReconstructionQualityEngine and CaptureIntegrityEngine, with conservative hard gates, warnings and a dedicated DENSE_LANDMARKS bottleneck. Historical rejected/stale counters remain diagnostic only; current-frame hard failure is propagated independently so one bad frame cannot permanently poison later capture.

The detector-derived uncertainty currently used at the perception boundary is explicitly a conservative proxy derived from detector confidence. It is not claimed as calibrated metric uncertainty and cannot qualify a <2 mm physical-accuracy claim without a matched reference benchmark.