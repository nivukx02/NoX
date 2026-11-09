# NoX High-End Motion Capture Upgrade
Date: 2026-09-28

## Scope

This upgrade keeps NoX strictly a motion-capture product. Conventional video-recording modes are not introduced. The existing multi-camera RGB perception, calibration, synchronization, body/face/eye/hand tracking, dense reconstruction, temporal fusion, physics constraints, and Liquid Glass studio remain the foundation.

## Research inputs

### Calibration-free sparse RGB reconstruction

Kineo, "Calibration-Free Metric Motion Capture From Sparse RGB Cameras", describes a sparse consumer-RGB pipeline that estimates camera calibration, lens distortion, metric-scale 3D keypoints and dense scene point maps without requiring a pre-calibrated synchronized rig.

https://arxiv.org/html/2510.24464v2

Relevant design lessons adopted in NoX:
- treat calibration as an optimization/evidence problem rather than an assumed constant;
- weight multi-view observations by confidence;
- retain explicit metric-scale and reprojection evidence;
- use global spatiotemporal consistency rather than a single-frame pose result.

### Dense multi-person reconstruction

MAMMA, CVPR 2026, uses multi-view video, segmentation-aware dense landmarks, visibility/contact information and SMPL-X reconstruction to improve markerless reconstruction under occlusion and close interaction.

https://openaccess.thecvf.com/content/CVPR2026/papers/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.pdf

NoX already contains dense-landmark, contact, identity, surface, kinematic and multi-person subsystems. The new precision policy increases the emphasis on their evidence agreement instead of treating any single network output as ground truth.

### Liquid Glass and operator hierarchy

Apple's current Human Interface Guidelines state that Liquid Glass works best as a distinct functional layer for controls and navigation, while the content layer should avoid being covered with unnecessary glass effects. The guidance also emphasizes legibility, hierarchy, adaptability and consistent feedback.

https://developer.apple.com/design/human-interface-guidelines/materials

https://developer.apple.com/documentation/technologyoverviews/liquid-glass

NoX therefore keeps Liquid Glass concentrated around interactive controls, diagnostics and navigation while leaving the reconstructed subject and capture evidence visually dominant.

### Jetpack Compose performance

Android's current Compose guidance emphasizes stable data, limiting unnecessary recomposition, derived-state techniques for rapidly changing state when appropriate, deferred state reads, lazy layout keys/content types, and Baseline Profiles for startup/runtime performance.

https://developer.android.com/develop/ui/compose/performance

https://developer.android.com/develop/ui/compose/performance/bestpractices

https://developer.android.com/develop/ui/compose/performance/baseline-profiles

The NoX UI upgrade follows the same principle: visual sophistication must not force high-frequency capture telemetry to trigger unnecessary composition work.

## Precision policy

The new UltraPrecisionEvidenceEngine does not claim that an ordinary RGB phone can universally outperform optical or infrared systems.

It separates four states:

1. Modeled geometry and uncertainty.
2. Empirical evidence collected from the real implementation.
3. Matched-reference comparison with explicit reference uncertainty.
4. Qualification only when the measured conditions actually satisfy the configured error contract.

The default ultra-precision qualification target is P95 <= 0.75 mm with P99 <= 1.75 mm and at least 600 matched evidence samples. These values are engineering acceptance targets, not claims of achieved accuracy.

A future result can only support a statement that NoX measured lower error than a reference optical/infrared system when the reference conditions are matched and the measured distributions are statistically separated after uncertainty is included.

## Geometry model

The precision engine includes contributions from:

- triangulation geometry;
- effective focal length;
- camera baseline;
- observed parallax;
- 2D localization/reprojection uncertainty;
- camera pose uncertainty;
- temporal skew;
- rolling-shutter contribution;
- metric-scale uncertainty;
- occlusion and cross-view agreement;
- temporal surface stability;
- thermal and memory risk.

This is intentionally more conservative than reporting a single triangulation equation as if every other error source were zero.

## Motion engine upgrade

HighOrderMotionRefinementEngine adds robust temporal evidence over motion score, speed, acceleration, jerk, jitter, confidence, occlusion and vibration.

It uses bounded history, median/MAD-style robust weighting and explicit camera-motion suppression evidence. It adjusts confidence and temporal evidence while leaving detector coordinates untouched, preserving the separation between perception and downstream metric reconstruction.

## Structural correction

MultiViewFusionEngine contained a forward reference where denseIdentityGateScale was used before its local declaration. The value is now computed before the combined reconstruction score, preserving the intended identity-gating behavior.

## Production principles

NoX should continue to prefer:
- measured evidence over marketing claims;
- bounded queues and explicit ownership over unbounded allocations;
- confidence-aware fusion over blind averaging;
- empirical qualification over modelled accuracy claims;
- adaptive runtime policy over silent quality downgrade;
- clear operator feedback over decorative UI.

Android builds and Python simulations were intentionally not started during this upgrade pass.


## Second upgrade pass

### Cross-view reprojection consensus

The Kotlin triangulator now performs a bounded consensus refinement pass after its robust Gauss-Newton stage. Each view receives a consensus weight derived from pairwise agreement between its reprojection residual and the other calibrated views. Views that disagree strongly are downweighted rather than allowed to distort the metric point.

The dense body/surface path now consumes the resulting consensus score in geometric confidence, and the dense face mesh path preserves it in each facial surface point and surface-level confidence.

### Rig-wide calibration evidence

Multi-camera calibration quality now reports and gates:
- pairwise redundancy;
- baseline diversity;
- angular diversity;
- low-confidence pair concentration.

A four-camera rig with only one useful stereo pair can therefore be identified as structurally weaker than a genuinely redundant rig even when its average reprojection error looks acceptable.

### Robust eye calibration

Eye gaze calibration now uses bounded IRLS refinement with robust median/MAD residual weighting rather than a single least-squares fit. A bad fixation sample can be suppressed without rewriting the public calibration profile format.

### Adaptive Liquid Glass

Liquid Glass now accepts an effect-strength budget. The main motion-capture studio derives that budget from measured GPU latency, session health and capture-integrity state. When capture pressure rises, blur-like visual intensity, specular alpha and elevation are reduced while layout and interaction remain unchanged.

This follows the core production principle that visual polish must not compete with the real-time motion pipeline for GPU budget.


### Native geometry parity

The native weighted triangulator now consumes the same reprojection-consensus evidence model. Consensus confidence is exposed in the native triangulation result, contributes to the native confidence score, and can block an otherwise numerically valid point when cross-view residual evidence is inconsistent.

This keeps the Kotlin and native geometry paths aligned rather than allowing one path to accept a point that the other would regard as low-evidence.


## Third upgrade pass

### Temporal metric-scale stability

Metric recovery now has a bounded temporal scale-stability gate. A per-frame metric scale estimate is insufficient for authoritative capture if it drifts during a long take, so NoX now tracks scale jitter, drift per minute, sample warmup and confidence. The result can remain explicitly non-authoritative until the scale is temporally stable.

### Sub-millisecond synchronization contract

The multi-camera synchronization quality engine now supports an opt-in precision contract with:
- maximum 1 ms absolute clock offset;
- maximum 750 microseconds peer spread;
- maximum 500 microseconds mean sub-frame offset;
- minimum 0.90 sub-frame confidence;
- minimum 0.90 aggregate synchronization confidence.

Normal synchronization mode keeps its existing envelope. Precision mode adds a hard qualification layer rather than silently tightening every capture.

### Ultra-precision benchmark profile

CapturePrecisionBenchmarkEngine now exposes an explicit ultra-precision profile:
- P50 <= 0.50 mm
- P95 <= 0.75 mm
- P99 <= 1.75 mm
- jitter <= 0.50 mm
- scale drift <= 0.25 %/min
- minimum 600 samples
- tighter temporal reference gap and lag search

The profile is an acceptance contract, not an assertion that these numbers are already achieved on arbitrary hardware.

### Operator-health corrections

Studio telemetry no longer double-declares the GPU stale-generation field. Health signal totals are derived from the actual signal set instead of a hard-coded count. Precision health now considers scale drift and precision-feasibility evidence.

### Regression coverage

Added or corrected repository tests for:
- temporal metric-scale stability;
- sub-millisecond synchronization qualification;
- ultra-precision benchmark configuration;
- synchronization test helper correctness;
- current triangulation result-field naming.

These tests were added but not executed in this pass.

## Fourth upgrade pass

### Robust metric-scale trend qualification
The metric-scale temporal gate now uses a bounded confidence-weighted robust trend fit rather than relying on a single endpoint slope. It also exposes maximum step jump, trend confidence and outlier density. Authoritative scale publication now requires a stable trend, sufficient trend confidence and bounded instantaneous jumps.

### High-order temporal motion reconstruction
Dense reconstruction now owns a bounded per-point kinematic state for high-order motion evidence. The new KinematicStabilityEngine uses short-window cubic regression with Huber reweighting to estimate velocity, acceleration and jerk, then supplies bounded prediction for brief observation loss. Kinematic tracks are capped and explicitly released with stale point state to control memory growth.

### First-order sub-millimetre geometry evidence
Triangulation now calculates effective baseline, depth sensitivity and a first-order depth error estimate using the stereo/SfM relation dZ ~= Z^2/(fB) * dd. The output is a feasibility score and warnings, not a claim of achieved physical accuracy. Dense-surface confidence incorporates this evidence.

### Precision-aware operator telemetry
Studio telemetry now exposes dense motion stability, kinematic innovation, jerk, first-order depth error and sub-millimetre feasibility. Studio health can warn or hard-block on severe precision evidence degradation while preserving explicit UNKNOWN/DEGRADED semantics.

### Precision synchronization stability
The synchronization quality engine now estimates per-peer clock drift in ppm from bounded timed-offset history. Precision qualification additionally requires a configurable number of consecutive clean evaluations, preventing a single good synchronization sample from authorizing an otherwise drifting rig. The default precision contract adds an 80 ppm clock-drift ceiling, a three-evaluation stability window and a separate warning envelope.

### Operator UI
The Motion Studio operator deck now surfaces first-order precision feasibility in the Liquid Glass evidence matrix. Existing adaptive glass behavior remains intact.

### Regression coverage
Added regression coverage for:
- robust temporal kinematic reconstruction;
- bounded high-order prediction and stale behavior;
- synchronization precision stability windows;
- clock-drift evidence;
- triangulation first-order precision evidence.

Tests were added but not executed in this pass, per the project constraint.
## Fifth upgrade pass

### Research-driven interaction robustness
Current 2026 benchmarks reinforce that realistic markerless MoCap fails primarily around severe occlusion, close interaction, identity exchange and weak temporal modeling rather than simple single-person poses. NoX therefore keeps cross-view identity gating, visibility/contact evidence, temporal continuity and high-order kinematics as first-class production signals. HUM4D explicitly evaluates these difficult conditions; MAMMA likewise uses dense visibility/contact-aware landmarks and person-specific correspondence for close interactions. These findings informed the existing identity and dense-surface evidence path rather than being treated as cosmetic research references.

### Dense rolling-shutter timing
Dense landmarks now accept optional sensor-row metadata, frame duration, exposure time, rolling-shutter skew and capture-timing confidence. The physical rolling-shutter evidence model converts these into row-time offset, readout ratio, temporal burden, per-observation confidence and signed frame-relative timing. The dense triangulation path now carries the resulting rolling-shutter fraction and timing confidence instead of silently assuming a global exposure instant.

### High-order facial temporal reconstruction
Dense face points now use bounded high-order temporal dynamics in addition to face-specific triangulation and reprojection consensus. Per-point kinematic state is capped, expired state is released, and temporal stability contributes to the final face-surface confidence. This keeps rapid facial movement and short detector dropouts from being handled by an unrelated temporal model.

### Explicit dense precision qualification
Dense surface results now distinguish ordinary geometric trust from explicit precision qualification evidence. Qualification requires sub-millimetre feasibility evidence, first-order depth error at or below 1 mm, and sufficient temporal motion stability. A good-looking reconstruction therefore cannot silently masquerade as a sub-millimetre result.

### Ultra-precision contract integration
The ultra-precision evidence engine now consumes dense geometry feasibility, first-order depth error, dense motion stability and dense kinematic innovation. High-end qualification can therefore block even when camera-level geometry looks strong but the actual dense reconstruction evidence is weak.

### Rolling-shutter API consistency
The rolling-shutter evidence input now supports an optional normalized sensor-row value, preserving compatibility with pixel-derived rows while matching the existing multi-view fusion call path.

### Operator surface
The Liquid Glass Motion Studio precision cell can explicitly show QUALIFIED, HOLD, a measured feasibility percentage, or PENDING when evidence is unavailable. Precision health telemetry is propagated independently from generic reconstruction confidence.

### Verification policy
Source-level regression tests were added or extended for kinematics, face precision qualification, rolling-shutter normalized rows, synchronization precision stability, triangulation precision evidence and ultra-precision dense evidence. Tests and the Android build remain intentionally unexecuted in this pass.

### Native IMU translational evidence
The native IMU/vision fusion path now keeps second-order inertial orientation propagation plus bounded translational evidence: world-frame linear acceleration, velocity, displacement, stationary velocity damping, velocity/displacement caps and translation confidence. The translation state is auxiliary evidence and is intentionally not treated as authoritative world position without an external metric reference.

### Matched-reference precision science
Added a standalone matched-reference benchmark engine for paired NoX versus external-reference measurements. It reports P50/P95/P99/mean/RMS/max error, paired deltas, MAD, tail separation, reference uncertainty, matched fraction, outlier fraction and a bounded qualification score. The engine explicitly retains high-uncertainty reference samples so the uncertainty cannot be hidden by preprocessing. This is the appropriate evidence layer for any future claim comparing NoX against an infrared/marker-based reference under matched conditions.


## Sixth upgrade pass

### Tail-aware triangulation precision
Triangulation now preserves both RMS reprojection evidence and a configurable high-quantile residual tail. The precision layer exposes P95-style residual evidence, a tail-adjusted first-order depth error and a separate tail-adjusted sub-millimetre feasibility score. This does not replace the existing score or declare physical accuracy; it prevents a small high-error tail from disappearing inside an average.

### Precision policy plumbing
The dense-landmark strictness controls now tune the triangulation precision tail, residual inflation and minimum pixel-noise floor through the existing `MotionCaptureCustomization` path. The precision policy is therefore shared between high-end settings and the reconstruction runtime rather than living as a disconnected research parameter.

### Dense-surface tail telemetry
Dense reconstruction now retains triangulation tail precision on each point and exposes aggregate tail-adjusted depth/feasibility diagnostics. Ordinary geometry qualification remains backward-compatible, while the tail is visible as additional evidence and warnings.

### Native inertial translation correction
The IMU/vision fusion path now subtracts the accelerometer bias estimate before rotating acceleration into world coordinates. Previously, gyro bias was applied but the already-estimated accelerometer bias was not, allowing a persistent DC acceleration component to contaminate velocity and displacement. The correction is bounded and remains auxiliary translational evidence rather than absolute metric position.

### Operator workflow hierarchy
Motion Studio now has a compact Liquid Glass `Capture intent` shelf above the detailed evidence matrix. It reuses the existing QUALITY, BALANCED and LOW_LATENCY runtime presets and presents synchronization, multi-view geometry, dense precision, timing and overall health together. This applies progressive disclosure instead of increasing the settings surface indefinitely.

### Verification policy
Source-level changes were committed without running the Android build or Python simulation. The new precision and UI paths remain subject to the existing regression suite and physical matched-reference qualification process before any sub-millimetre or infrared-comparison claim is made.


### Bounded inertial zero-velocity correction
Native IMU propagation now includes a confidence-gated, bounded stationary velocity correction. It acts only when stationary confidence, bias confidence and acceleration residual are jointly strong, reducing accumulated low-frequency velocity drift without treating every low-speed sample as zero motion. This is complementary to the existing velocity cap, stationary damping and vision orientation correction.


## Seventh upgrade pass

### Face precision parity
The face mesh solver now carries triangulation P95 residual and tail-adjusted first-order depth evidence per surface point, plus aggregate surface-level tail error and feasibility. This keeps facial reconstruction from using a weaker precision standard than the dense body surface path.

### Qualification evidence propagation
Dense tail-adjusted precision now crosses the multi-view fusion boundary and is provided to the ultra-precision evidence engine. The qualification engine treats the tail as an additional warning/evidence signal rather than silently replacing the established metric, preserving compatibility while increasing observability.

### UI performance hierarchy
The new Capture Intent Liquid Glass shelf inherits the existing adaptive glass scale. Under GPU or runtime pressure, the shelf can reduce material complexity together with the rest of the operator surface rather than becoming an independent rendering hotspot.

### Research alignment
Recent markerless MoCap work reinforces the engineering priorities used here: Kineo reports calibration-free metric capture with confidence-driven spatiotemporal sampling and pairwise reprojection consensus, while MAMMA emphasizes dense visibility/contact-aware landmarks and person-specific correspondence for heavy occlusion and interaction. HUM4D highlights severe occlusion, identity exchange and realistic multi-person dynamics as important failure cases. Primary sources: Kineo (arXiv 2510.24464), MAMMA (CVPR 2026 Open Access), and HUM4D (CVPR Workshops 2026).

### UI performance basis
Android's current Compose guidance recommends controlling recomposition, caching expensive calculations, deferring state reads and using Baseline Profiles for critical execution paths. The project remains intentionally unbuilt in this pass, so these recommendations are being applied conservatively at source level rather than being claimed as benchmarked runtime gains. Primary source: Android Developers, Jetpack Compose Performance and Baseline Profiles documentation.


## Eighth upgrade pass

### Persistent-state memory hardening
Adaptive camera/joint evidence state now has explicit camera and joint-state caps plus stale pruning and deterministic eviction. Temporal reconstruction now enforces its joint-track limit across the persistent state map, not just the per-frame observation list. Face reconstruction applies the same protection to its temporal point map. These changes target long-running capture sessions where changing identities or detector point sets could otherwise create unbounded retained state.

### Multi-view camera diversity
The camera-contribution planner now exposes minimum pairwise geometric diversity and a camera-starvation risk score. The fusion engine uses diversity as an additional quality scale and refuses authoritative acceptance when selected views collapse below a minimum geometric-diversity envelope. The operator health layer and Liquid Glass workflow shelf now expose the same evidence.

### Analytic triangulation Jacobian
The triangulation hot path now uses an exact Jacobian of the calibrated Brown-Conrady projection model for Gauss-Newton refinement, with finite-difference derivatives retained as a defensive fallback for degenerate camera depth or non-finite intermediate values. This removes three projection perturbation evaluations per observation per iteration in the normal path while also avoiding finite-difference truncation noise.

### Synchronization chronology hardening
Clock-drift history now ignores out-of-order timestamp samples rather than contaminating temporal drift estimation. Such samples remain visible as explicit peer warnings so transport/replay ordering defects cannot silently become synchronization confidence.

### Operator evidence
The studio workflow surface now reports camera diversity alongside synchronization, dense precision and timing. High-risk dependence on a single or weakly diverse camera set is surfaced as an operator health warning instead of being hidden behind the aggregate reconstruction score.

### Research basis
The design priorities remain aligned with current research: Kineo emphasizes confidence-aware metric multi-view reconstruction and reprojection consensus; MAMMA focuses on dense visibility/contact-aware correspondence in heavy occlusion and interactions; HUM4D demonstrates that severe occlusion, identity exchange, temporal coherence and multi-view evidence remain hard failure modes in current markerless systems. Current Android Compose guidance likewise emphasizes phase-aware state reads, stable inputs, and benchmark-driven performance profiling for production UI.


## Ninth upgrade cycle: evidence independence, identity exchange, operator gating and selection stability

### Correlation-aware evidence fusion
NoX now distinguishes dependent quality signals from genuinely independent evidence. Geometry, temporal, uncertainty, rig, physics, global, identity and dense-surface signals are fused by dependency family with bounded redundancy influence and explicit common-mode risk. This prevents multiple metrics derived from the same pixels or clocks from artificially stacking into an overconfident score.

The fusion contract now exposes effective evidence coverage and can reject a state when evidence coverage is below the configured contract or common-mode risk is above the configured envelope. Qualification mode uses stricter thresholds than normal production mode.

### Identity-exchange protection
A bounded MultiPersonIdentityExchangeRiskEngine now evaluates subject visibility, cross-view agreement, temporal identity confidence, interaction overlap and pair ambiguity. It identifies CLEAR, GUARDED, HIGH_RISK and HOLD states. Dense reconstruction propagates this evidence into the existing identity gate, operator health and reconstruction acceptance path. Subject history is explicitly bounded and reset with the dense reconstruction session.

This is specifically aimed at realistic multi-person failure cases highlighted by current markerless MoCap research: severe occlusion, close interaction and rapid identity exchange.

### Camera-selection temporal hysteresis
Multi-view camera selection now has bounded temporal hysteresis. Small frame-to-frame trust oscillations do not immediately replace otherwise stable camera contributions. Held cameras can still leave selection immediately, and selection history decays when a view disappears. This reduces view-set churn without hiding real camera failures.

### Operator workflow
The Liquid Glass operator surface now exposes correlated evidence and multi-person identity status alongside synchronization, geometry, dense precision, timing and camera diversity. Health warnings distinguish correlated evidence rejection, common-mode risk and identity HOLD/HIGH_RISK states.

### Configuration
High-end settings now expose:
- correlated-evidence enable/qualification mode;
- minimum evidence coverage;
- maximum common-mode risk;
- identity-exchange enable/qualification mode;
- identity high-risk and hold thresholds;
- bounded identity history length.

All new state remains measurable and contract-driven rather than being represented as decorative controls.

### Verification policy
Regression tests were added for correlated-evidence behavior, acceptance contracts, identity-exchange risk, subject churn bounds and camera-selection hysteresis. The Android build, device execution and Python simulation remain intentionally unexecuted for this upgrade cycle, per the project constraint.

NoX still treats sub-millimetre accuracy and any comparison against infrared or marker-based capture as physical qualification targets. Those claims require matched hardware, synchronized reference measurements, uncertainty accounting and a sufficient empirical sample set; architecture alone does not establish them.


## Tenth upgrade cycle: uncertainty correlation, display budgeting, teardown safety and native parity

### Correlation-aware uncertainty
A new CorrelatedUncertaintyBudgetEngine models dependent metric error sources with a bounded covariance-like budget. Reprojection, triangulation, temporal timing, calibration, rolling shutter, occlusion and scale uncertainty are no longer required to behave as statistically independent contributors. The anisotropic reconstruction engine now consumes this budget separately for lateral and depth envelopes.

This is deliberately an engineering uncertainty model. It is not a substitute for matched ground-truth measurement.

### Native uncertainty correction
The native depth uncertainty fusion path previously used an inverse correlation factor in its variance expression, which allowed higher positive correlation to reduce the reported variance. That is the wrong direction for dependent evidence. The native fusion now uses explicit correlation inflation and bases robust variance reduction on effective sample count rather than raw accepted-sample count.

A native regression test now checks that high-correlation depth evidence cannot report lower variance than the corresponding low-correlation pair solely because the measurements are more dependent.

### Adaptive studio rendering budget
StudioVisualBudgetEngine centralizes UI visual-quality decisions using GPU latency, session health and capture-integrity pressure. Under stress it reduces material complexity, shadow/animation intensity and diagnostic refresh rate while explicitly preserving the motion viewport's detail.

The policy is now the source of truth for Liquid Glass effect scaling in Motion Studio instead of a duplicated chain of local thresholds.

### Session teardown hardening
NoXViewModel cleanup now separates reset-style session cleanup from critical resource closure. High-cost resources such as IMU capture, recorder, perception, GPU acceleration, orientation, camera, native motion and global dense-evidence resources are closed through failure-isolated cleanup actions. A single subsystem cleanup exception therefore cannot prevent later resources from being released.

### Research basis
Recent research continues to support uncertainty-aware and confidence-aware multi-view design. ActiveMoCap demonstrates that view selection can be guided by predicted 3D pose uncertainty; Kineo emphasizes confidence-driven sampling and explicit pairwise reprojection confidence; current 2026 dynamic-surface work such as 4DSurf emphasizes temporal surface consistency and scalable long-sequence deformation modeling. The NoX architecture applies these ideas as measurable evidence and gating contracts rather than claiming reproduction of any paper's benchmark results.

The Android project already pins the 2026.09 Compose BOM. Current Compose 1.12 guidance includes Wide Color Gamut/HDR support and emphasizes phase-aware rendering and performance control. The new visual-budget layer follows that direction by reducing nonessential material work under capture pressure rather than sacrificing reconstruction content.

### Verification
Added source-level regression coverage for:
- correlation-aware uncertainty fusion;
- adaptive studio visual budgets;
- native correlation-dependent depth uncertainty;
- existing identity exchange and camera selection protections from the previous cycle.

The Android build, device execution and Python simulation remain intentionally unexecuted.


## Eleventh upgrade cycle: anisotropic view selection and motion-aware recovery

### Anisotropic view contribution
NoX now adds an AnisotropicViewContributionEngine on top of existing multi-view observability. It constructs a ray-space information matrix and extracts the weakest principal information axis. Candidate camera views are rewarded when their rays constrain that weak direction, in addition to detector confidence, temporal quality, appearance evidence and view novelty.

This is an information-direction selector, not a claim of globally optimal Bayesian active perception. The goal is to avoid selecting a group of individually confident views that collectively remain depth- or axis-degenerate.

Dense reconstruction now reports:
- anisotropic weak-axis constraint;
- anisotropic view condition score;
- warnings for weak-axis underconstraint and near-rank-deficient camera geometry.

Studio health surfaces these metrics and can block severely rank-deficient dense geometry.

### Motion-aware recovery
A bounded MotionAwareRecoveryEngine now coordinates per-track AdaptiveKalman3DFilter instances for short-gap motion recovery. It uses:
- confidence-aware measurement updates;
- robust innovation weighting inherited from the adaptive filter;
- prediction confidence decay;
- deterministic track eviction;
- timestamp reversal recovery;
- bounded short-occlusion prediction.

The observed calibrated triangulated point remains authoritative. The new layer is used primarily for short occlusion bridging, so temporal prediction does not overwrite good measured geometry.

This follows the current research direction represented by CVPR 2026 RAM: motion-aware identity/recovery, adaptive Kalman state estimation, temporal memory and prediction for occlusion robustness. NoX applies the principle with its existing detector-neutral metric point architecture rather than copying RAM's neural modules or claiming its benchmark results.

### Dense-engine correctness hardening
During source review, the dense reconstruction path contained references to pointConfidence, kinematic and stateBand without local declarations. These are now explicitly derived from the calibrated reconstruction result, robust kinematic engine and production confidence thresholds. This also makes the point state transition into TRUSTED, OBSERVED or LOW_CONFIDENCE explicit.

### Tests
Added source-level tests for:
- anisotropic view contribution and deterministic ranking;
- bounded motion-recovery tracks;
- short occlusion prediction;
- stale-track expiration;
- timestamp-reversal safety;
- dense anisotropic telemetry.

The Android build, device execution and Python simulation remain intentionally unexecuted.


## Twelfth upgrade cycle: true state projection, conditional measurement gating, temporal memory and geometry UX

### True forward state projection
The adaptive 3D Kalman layer now exposes a non-mutating forecast operation. Occlusion recovery can project an active track forward in time without advancing or corrupting the authoritative filter timestamp. The forecast propagates position with the current velocity and grows uncertainty using process and velocity variance.

This closes a correctness gap where a recovery request could previously return the last observed position while labeling it as a prediction.

### Conditional measurement gating
MotionAwareRecoveryEngine now holds out measurements below the configured minimum confidence when a reliable prior state exists. The prior state is forecast instead, with explicit warnings and confidence decay. This prevents low-quality detections from overwriting stable motion during severe blur, self-occlusion or cross-view ambiguity.

### Temporal-memory confidence
HighEndMotionStateEstimator now maintains a bounded temporal-memory confidence per joint. The memory score combines:
- directional consistency between the observed displacement and prior velocity;
- normalized innovation against the current uncertainty envelope;
- exponential temporal memory.

Prediction reliance is attenuated when temporal memory is weak. The aggregate motion-quality score now includes temporal-memory confidence, and the main capture pipeline feeds that evidence into both multi-view and monocular reconstruction quality gates.

This implements the useful part of recent 2026 motion-recovery research as a measurable temporal prior rather than pretending a large neural network is present when it is not.

### Calibration drift source repair
MultiCameraCalibrationDriftEngine received a source-level correctness repair in its per-camera evaluation path. Vibration risk and absolute sub-frame offset are now evaluated as separate signals instead of being merged by malformed assignment logic. Existing tests already cover temporal alignment and sub-frame drift degradation.

### Studio operator UX
The Liquid Glass operator deck now exposes two additional dense-geometry signals:
- Weak Axis: how strongly the selected views constrain the weakest 3D information direction;
- Axis Condition: the conditioning score of the current ray-information configuration.

These indicators make geometric failure visible to an operator before it silently appears as unstable depth.

### Verification policy
Added regression coverage for:
- non-mutating Kalman forward forecasts;
- low-confidence measurement holdout;
- temporal-memory continuity scoring.

The Android build, device execution and Python simulation remain intentionally unexecuted. No claim of sub-millimetre physical accuracy has been promoted by these code changes; the existing qualification contract remains empirical.


## Thirteenth upgrade cycle: capture-session generation fencing

### Late-result protection
NoXViewModel now assigns a monotonically increasing capture generation to each recording session. Long-lived frame, depth and IMU collectors capture the generation that started them and immediately discard data after the session generation changes.

The generation is advanced when a recording session begins and when capture finalization starts. This prevents late asynchronous emissions from a prior session from mutating the state of a new session after cancellation, reducing a class of race-driven state corruption and hard-to-reproduce cross-session crashes.

The Android build and device execution remain intentionally unexecuted for this cycle.


## Fourteenth upgrade cycle: temporal-memory operator tuning

The existing capture customization surface now controls temporal-memory behavior indirectly through the production temporal estimator:
- temporal smoothing controls memory adaptation rate;
- capture-integrity strictness controls the minimum trust placed on remembered motion.

This keeps the new temporal prior aligned with the existing high-end capture profiles instead of adding an isolated settings island.


## Fifteenth upgrade cycle: probabilistic multi-view evidence, strict P99 qualification, and binocular ray conditioning

### Multimodal 2D probability evidence
The triangulation boundary now accepts up to eight bounded Gaussian spatial modes per 2D observation, plus a distribution-calibration score. Point-only observations remain fully compatible.

The robust solve:
- uses distribution-aware observation weighting;
- evaluates 3D reprojection against preserved probability mass;
- reports probabilistic evidence score/warnings;
- inflates spatial uncertainty when probability mass disagrees with the recovered 3D point.

This implements the practical part of recent 2026 probabilistic multi-view work: ambiguous heatmaps should not be collapsed blindly to one peak. MEOM explicitly studies this failure mode and uses probability mass plus calibration diagnostics rather than peak-only decoding.

### Dense reconstruction propagation
Probabilistic evidence is now included in dense geometric confidence, point diagnostics, frame telemetry, ViewModel state, the Liquid Glass studio, and health policy. Very weak probability/reprojection agreement can place the dense stage into an operational hold and severe mismatch can hard-block it.

### Precision qualification
MeasuredPrecisionQualificationEngine now uncertainty-adjusts both P95 and P99 targets against external reference uncertainty. The qualification system continues to distinguish modeled feasibility from measured physical performance.

### Binocular gaze conditioning
HighEndBinocularGazeGeometryEngine now exposes a ray-geometry condition derived from the two-ray intersection denominator. Near-parallel gaze rays are explicitly warned as weakly conditioned, and the reconstruction intelligence panel shows the conditioning percentage separately from angular agreement and IPD consistency.

### Research basis
The probability-distribution design is informed by MEOM: Xiong, Shi and Forssén, “Multi-View Expected-OKS Maximization for Human Pose Triangulation,” submitted 31 August 2026, which argues that multimodal heatmap mass should be retained under ambiguous/occluded views and that calibration of that mass matters. citeturn793330academia24

The temporal-memory/recovery direction from the preceding cycles remains informed by RAM: Jia et al., CVPR 2026, combining motion-aware tracking, adaptive Kalman filtering, memory-augmented temporal reconstruction and prediction for occlusion-resilient multi-person motion recovery. citeturn793330search25

### Verification policy
Dedicated regression coverage was added for probabilistic triangulation, uncertainty inflation, P99 qualification, and binocular ray conditioning.

The Android build, device execution, and Python simulation remain intentionally unexecuted. No claim that NoX currently achieves physical sub-millimetre accuracy or exceeds infrared/marker-based systems has been added; those remain empirical qualification targets.
