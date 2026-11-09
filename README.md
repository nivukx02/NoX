# NivukxNoX

**NivukxNoX** is the canonical technology name for the production-oriented, motion-only Android distributed motion-capture stack. **NoX** remains the repository and application shorthand.

- Technology: NivukxNoX
- Application shorthand: NoX
- Package: com.nivukx.nox
- Release: 1.2.0
- Minimum Android: API 30
- Target/compile API: 37
- Java: 17

## Scope

NivukxNoX is strictly a motion-capture technology. It does not define conventional video/HDR/HLG/SDR recording as a product feature. Camera imagery may exist transiently in memory for perception, calibration and reconstruction, while the canonical capture artifact is structured motion/state evidence.

## Production architecture

The stack combines:

1. synchronized multi-device capture;
2. calibration and temporal evidence;
3. RGB and multi-view perception;
4. body, hand, face and eye tracking;
5. robust triangulation and uncertainty gating;
6. interleaved spatiotemporal fusion;
7. global 4D consistency and body-scale continuity;
8. dynamic scene/subject separation;
9. temporal gaze and hand kinematics;
10. dense mesh temporal consistency;
11. bounded native queues and buffer pools;
12. thermal, memory and workload-aware execution;
13. operator-facing precision telemetry and Liquid Glass controls;
14. controlled accuracy benchmarking against external references.

## Accuracy policy

NivukxNoX does not claim to be "better than infrared" from architecture alone. Professional optical/infrared systems and markerless systems have different operating envelopes. Any superiority claim must come from matched-condition measurements with explicit reference uncertainty, region/visibility cohorts, temporal latency, dropout, identity continuity and tail-error statistics.

The local stress harness used during development therefore includes controlled, occluded, sparse-view, high-load, timestamp-corruption, NaN/Inf and memory-churn scenarios. Synthetic simulation is a validation tool, not a substitute for a physical capture benchmark.

## Current validation status

The latest local synthetic stress run identified:

- sustained occlusion + naive constant-velocity prediction can create extreme P99/max errors;
- bounded prediction reduces catastrophic tail excursions but does not solve sparse-view reconstruction;
- sparse 3-camera conditions are substantially harder than well-covered multi-view conditions;
- synthetic memory churn returned to a near-baseline allocation state after repeated capture cycles;
- timestamp reversal and non-finite observation injection are detectable failure conditions and must remain fail-closed.

The Android application build is intentionally not started as part of this validation pass.

## 2026-10-01 high-end continuation

The current main branch extends the existing architecture with a production crowd-intelligence layer and richer 3D evidence provenance:

- 50-actor dense-landmark evidence admission with explicit bounded storage;
- distributed 128-camera evidence federation with shard-aware authority and quarantine;
- bounded authoritative reconstruction selection so 100+ physical cameras can participate without forcing a single device to open every sensor;
- MediaPipe body-space world landmarks retained as a secondary 3D body prior;
- confidence-ranked deterministic actor admission;
- cross-camera identity scoring augmented by robust body-space bone-length signatures;
- multi-actor wire protocol v3 with v1/v2 decode compatibility and bounded 3D/prediction evidence;
- per-actor crowd scheduling for body, face, eye, hand and dense-surface workloads;
- interaction/occlusion pressure and identity-risk telemetry;
- operator-visible body-space coverage, bone consistency and 3D-prior reliability;
- ordered shutdown of the stateful multi-person inference worker;
- production identity-exchange analysis aligned with the 50-actor runtime target.

The implementation continues to keep architectural evidence separate from physical metrology claims. A modeled <1 mm target, a measured <1 mm result, and superiority against an external optical/infrared reference are three different statements and are not interchangeable.

The full research and implementation record is maintained in docs/research/2026-10-01-high-end-crowd-3d-evidence-upgrade.md.

### Current upgrade validation boundary

This 2026-10-01 continuation was reviewed by repository inspection and contract tracing. No Android/Gradle build, runtime instrumentation test, or Python simulation was executed in this pass.

### Current high-end upgrade

The production capture stack now carries a 128-camera distributed evidence envelope while keeping dense reconstruction bounded to a selected authoritative view set. The crowd path is designed around a 50-actor target with a 64-actor hard ceiling, and cross-modal identity fusion is bounded to 64 tracks / 128 observations.

Recent hardening also adds higher-order multi-view actor hypothesis consensus, fail-closed calibrated geometry checks, bounded face/eye device history, robust iterative binocular gaze consensus, and post-capture enforcement of distributed camera authority. These controls improve evidence quality and lifecycle safety without pretending that source-code sophistication alone certifies physical sub-millimetre accuracy.

This continuation further hardens the production stack with configurable distributed-rig authority (128 global / 32 authoritative views), stricter multi-view requirements for ultra/reference workflows, higher-order crowd identity consensus, robust binocular gaze fusion, bounded face/eye/cross-camera history, fair actor-detail aging, assignment lookup optimization, 128-depth jitter tolerance, and innovation-aware native Gaussian uncertainty. Reference and infrared comparisons remain empirical qualification tasks, not source-code claims.


### Latest high-end upgrade layer
- Hardware-timing authority now combines measured readout timing, calibrated logical-camera synchronization, visual timing-beacon calibration, peer clock fitting, and rolling-shutter risk instead of treating timestamps as exact.
- Reference-grade qualification now requires explicit temporal authority and residual limits, with measured comparison support against an independent reference system such as an infrared optical rig.
- Dense 3D Gaussian surface state now retains bounded full symmetric XYZ covariance, so cross-axis reconstruction uncertainty is preserved during multi-view fusion.
- Dense landmark and multi-actor pose transport now support lossless bounded fragmentation and reassembly under the existing 32 KiB packet safety limit, preserving 50-actor scenes without silent truncation.
- High-speed acquisition profiles now prefer max available resolution, short manual exposure, stable sensor timestamps, RAW/depth sidecars, and disabled electronic/optical stabilization for precision workflows when the hardware supports them.
- Operator Liquid Glass controls now expose professional camera acquisition settings directly instead of hiding precision-critical choices inside configuration objects.


### 2026-10-05 frontier architecture continuation

The latest main-branch continuation adds exposure-midpoint timing authority for rolling-shutter capture, fairness-aware 50-actor detail scheduling, a 32-authoritative/96-witness camera federation layer within the existing 128-camera envelope, stronger redescending robust triangulation, independent frontier precision qualification, hard hardware-buffer resident-memory admission limits, and bounded native diagnostics latency.

The precision path now explicitly separates modeled feasibility, measured error, target attainment, and matched-reference qualification. A source-code target of <1 mm is never promoted to a physical performance claim without empirical reference-aligned measurements. The camera path likewise separates global camera participation from the bounded view set used for expensive reconstruction.

A detailed research and implementation record is maintained in docs/research/2026-10-05-frontier-upgrade.md.

### Latest precision and distributed transport hardening

The current main branch additionally carries per-camera calibration-focus state through self-calibration, calibration snapshots, the version-3 calibration wire contract, and session provenance fingerprints. Reference-grade quality gating can therefore distinguish matched lens state from focus drift instead of assuming mobile optics remain invariant.

Distributed dense-landmark, multi-actor, and face-mesh evidence is route-budget aware under the fixed 32 KiB packet ceiling. Fragment reassembly is bounded by both assembly count and aggregate payload bytes, and routed evidence is dispatched back into typed reassembly rather than being reduced to opaque raw payloads. Face-mesh reassembly uses one bounded allocation instead of repeated whole-buffer copies.

Thermal extrinsic authority uses hysteretic state transitions, while higher-order actor association incorporates calibrated depth consistency when depth is available. These are evidence-integrity upgrades, not claims of universal sub-millimeter accuracy or superiority to active optical reference systems.


### 18:34 continuation hardening

- Global stable-ID re-identification is one-to-one through bounded Hungarian assignment.
- Rig identity confidence now accounts for current-component truncation and unmatched coverage.
- Cross-camera association uses bounded spatial/temporal candidate generation instead of quadratic all-pairs graph construction.
- Uncalibrated cross-camera association remains identity evidence but cannot reach calibrated metric-association confidence.
- Spatiotemporal bundle adjustment can use bounded DTW motion alignment as a seed before robust reprojection refinement.
- Face/eye fusion down-weights poor capture timing, and fallback eye-track keys use denser face geometry.
- Native triangulation and calibration linear solves use scale-aware pivot thresholds.
- Dense landmark precision requires actual evidence sufficiency before READY/authority.
- Global 4D consistency includes an explicit root-acceleration envelope.
- Gaussian surface fusion consumes visibility, photometric consistency, and normal-consistency evidence.
- Focused regression tests cover clock outlier rejection, identity crossing, actor-detail fairness, dense evidence sufficiency, and large-rig identity truncation.

These changes improve robustness and qualification discipline; they do not constitute a measured sub-millimeter or better-than-infrared result without matched physical benchmark data.

### Latest temporal, scalability and operator hardening

The current main-branch pass further hardens production authority without reducing any existing feature path:

- Dense reconstruction rejects exact duplicate timestamps idempotently, preventing duplicated evidence from mutating kinematics or inflating precision authority.
- Point-level dense precision requires tail-adjusted feasibility, tail-adjusted depth evidence, reprojection-tail bounds, independent-view count and view diversity before a point can be precision-qualified.
- Frame-level dense precision now requires a sufficient ratio of precision-qualified points rather than treating aggregate averages as sufficient authority.
- Capture-stage scheduling now uses explicit pressure confirmation and recovery hysteresis. Transient pressure no longer thrashes optional fidelity stages, while severe pressure still sheds optional work immediately and deferred stages retain starvation protection.
- Global 4D temporal state is idempotent for duplicate and older timestamps. Older samples are surfaced as ignored evidence instead of being reinjected into root velocity/acceleration estimation.
- Cross-camera subject association is now depth-aware when calibrated depth is available, with bounded 3D spatial candidate cells plus the existing temporal fallback. Overlapping image positions at incompatible metric depths cannot silently merge through the graph path.
- Runtime memory safety now derives effective pressure from the governor's own byte-budget occupancy as well as caller-reported pressure, preventing low telemetry from masking a nearly exhausted allocation envelope.
- Liquid Glass operator surfaces now include an adaptive readability plate beneath live content. The material hierarchy remains shell → plate → content, with capture/accessibility priority preserved over decorative transparency.

These changes are evidence-integrity and production-stability upgrades. They do not create a physical guarantee of sub-millimetre accuracy without matched empirical measurement and reference qualification.



### Latest live-path hardening continuation

The latest main-branch continuation closes live-path authority gaps discovered through production call tracing:

- PerceptionPipeline now forwards measured capture timing into FaceEyeFusionEngine instead of dropping it between detector and fusion layers.
- Face detector fallback carries an explicit freshness flag, preventing a cached face frame from masquerading as newly observed evidence.
- MultiActorPoseAssociationEngine rejects high-quality, strongly contradictory appearance evidence before global assignment, strengthening close-interaction identity continuity.
- MotionCaptureEngine treats duplicate timestamps as idempotent replay and keeps the previous authoritative result instead of reinjecting temporal state.

These changes preserve all existing capture modes while tightening evidence provenance at the boundaries where stale, contradictory or replayed measurements can otherwise leak into downstream authority.


## Frontier A+B+C upgrade — 2026-10-05

The current main branch extends the existing production stack through a heterogeneous A+B+C architecture:

- A: native/geometric authority with exposure-midpoint timing, rolling-shutter uncertainty, hard native memory-domain admission, deterministic replay qualification, and accelerator-aware workload planning;
- B: calibrated world-space learned proposals, hierarchical temporal motion inference, dense world-space correspondence, bounded actor identity hypotheses, and fairness-aware 50-actor detail scheduling;
- C: sparse-view human reconstruction, dynamic Gaussian/surface-flow refinement, global human-scene 4D optimization, dense face-surface authority, and robust world-space gaze reconstruction.

### Capacity and evidence contracts

- 50 actors remains the runtime target;
- 64 actors remains the hard engineering ceiling;
- 128 cameras remains the distributed federation ceiling;
- 32 cameras are selected as the bounded authoritative reconstruction set;
- 96 cameras may remain witness evidence.

Authoritative reconstruction now exposes provenance that separates observed geometry, geometry-constrained state, temporal propagation, learned priors, and generative completion.

### Precision boundary

The project continues to distinguish modeled feasibility from measured physical accuracy. A configured <1 mm target does not certify a measured <1 mm result, and source-code capability does not establish superiority over professional infrared/optical reference systems.

### Verification boundary

No Python simulation tests are part of the frontier upgrade. Verification is performed with repository-native Kotlin/JVM, Android/Gradle, native C++ and deterministic replay contracts. The full build remains governed by GitHub Actions.
