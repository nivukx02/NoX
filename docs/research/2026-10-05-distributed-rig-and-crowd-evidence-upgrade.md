# NoX high-end continuation: distributed camera and crowd evidence research

## Evidence reviewed

The implementation direction was cross-checked against current multi-view reconstruction, synchronization, multi-person capture, and operator-interface research.

### Multi-view reconstruction

UnstructuredFusion demonstrates that practical realtime 4D reconstruction benefits from online camera alignment, skeleton-aware non-rigid tracking, temporal fusion, TSDF geometry, and dynamic texture accumulation. NoX already contains analogous calibrated reconstruction, temporal fusion, sparse TSDF and Gaussian surface components; the continuation therefore strengthens evidence arbitration rather than replacing those systems.

Source:
https://suzhuo.github.io/Files/Unfusion.pdf

Kineo demonstrates the value of confidence-driven keypoint sampling, graph-based global optimization, metric reconstruction and explicit pairwise reprojection reliability for uncalibrated/unsynchronized RGB-camera capture. NoX adopts the principle that reconstruction confidence must be attached to geometry evidence rather than inferred from detector confidence alone.

Source:
https://arxiv.org/html/2510.24464v2

### Temporal synchronization

RocSync reports millisecond-level heterogeneous-camera synchronization using visual timing evidence and validates downstream improvements to pose estimation and 3D reconstruction. The engineering implication for NoX is that camera timestamps, exposure/readout timing, clock drift and residual error must remain explicit authority signals.

Source:
https://pmc.ncbi.nlm.nih.gov/articles/PMC12900090/

NoX therefore keeps its existing clock fitting, sub-frame correction, rolling-shutter evidence and synchronization quality gates and extends the physical deployment model to distributed camera nodes.

### Multi-person reconstruction

Current 2026 markerless-capture research continues to emphasize severe occlusion, close interaction, identity exchange, temporal continuity and dense visibility/contact evidence as the difficult cases. NoX therefore preserves its 50-actor target and bounded 64-actor engineering ceiling while increasing cross-camera association and contact evidence capacity so crowd processing does not collapse at the first serious interaction.

### Operator interface

Current Liquid Glass guidance emphasizes content-first hierarchy, adaptive opacity, restrained use of translucent materials, responsive motion, accessibility, and keeping controls/navigation distinct from primary content. NoX's reconstruction content remains readable while Liquid Glass is used for the operator layer.

Source:
https://www.apple.com/newsroom/2025/06/apple-unveils-a-delightful-and-elegant-new-software-design/

## Architecture decisions

1. Global rig capacity is distinct from per-device simultaneous camera capability.
2. 128 camera identities are accepted into the distributed evidence envelope.
3. Only a bounded authoritative subset enters expensive dense reconstruction.
4. Camera timing/calibration violations are quarantined instead of silently down-weighted into authoritative geometry.
5. Selection uses spatial diversity, baseline, temporal trust, information gain and hysteresis.
6. Crowd identity capacity is expanded to a 64-actor engineering ceiling while retaining the 50-actor operational target.
7. Artificial line-count expansion is explicitly rejected as a quality metric. Production quality is measured by deterministic behavior, bounded memory, observability, failure containment, benchmarkability and integration.

## Physical-accuracy contract

No architecture-only implementation is allowed to claim <1 mm physical accuracy or superiority over optical/infrared reference systems. Those claims require matched-condition empirical qualification with reference uncertainty, calibration quality, temporal residuals, occlusion cohorts, identity continuity and tail-error statistics.

The system may maintain a <1 mm qualification target and refuse promotion when measured evidence does not satisfy it.


## Continuation: higher-order identity, eye robustness, and lifecycle hardening

The next production pass adds three evidence layers without replacing the existing calibrated reconstruction stack.

### Higher-order actor hypotheses

Recent multi-view reconstruction work emphasizes that pairwise view links can produce false chains when occlusion and repeated poses make two-camera associations ambiguous. NoX now evaluates bounded three-view-and-higher-order actor hypotheses after pairwise graph generation. Candidate views are confidence-ranked, triplets are bounded, and consensus is based on pair agreement, temporal consistency, calibrated geometry consistency, scale consistency, and detector confidence. Calibrated geometry failures fail closed rather than falling back to an uncalibrated prior.

This is intentionally a soft authority layer. It reduces confidence and raises ambiguity warnings when higher-order evidence does not converge instead of fabricating an identity.

### Eye and gaze fusion

The 3D eye solver now uses iterative robust spherical consensus for world-space gaze directions. Per-view eye confidence is combined with eyelid openness and blink state, then Cauchy-like angular reweighting suppresses high-angle outliers. Gaze inlier ratio is exposed as evidence alongside angular spread, triangulation uncertainty, view diversity, parallax, and temporal stability.

### Long-session resource behavior

Face contour, dense face mesh, and eye reconstruction histories now enforce a 128-device resident-cardinality ceiling in addition to per-device frame history bounds. Oldest device state is evicted when a distributed rig experiences device churn, preventing identity/history maps from growing without bound.

Cross-modal identity fusion now supports a bounded 64-track / 128-observation envelope for the 50-actor production target.

### Source references

- MEOM, multimodal multi-view pose reasoning and probability-distribution preservation: https://arxiv.org/abs/2608.30521
- COMPOSE, higher-order cross-view human pose association: https://arxiv.org/abs/2601.09698
- Kineo, multi-camera kinematic and temporal reconstruction context: https://arxiv.org/html/2510.24464v2
- RocSync, practical distributed synchronization context: https://pmc.ncbi.nlm.nih.gov/articles/PMC12900090/


## Continuation: uncertainty integrity and failure containment

The native temporal Gaussian surface now retains an innovation-derived covariance component during repeated fusion. Pure inverse-variance accumulation can otherwise collapse posterior variance toward an unrealistically small value when systematic calibration error, correspondence mismatch, motion, or correlated sensor noise is present. The new bounded inflation term keeps observed disagreement visible in the uncertainty budget.

The native TSDF memory-trim path also clears the cached extracted high-detail mesh, not only the voxel map. This makes memory-pressure recovery actually release the largest derived surface allocation.

The hand detector path now recovers from timestamp regressions and detector exceptions by resetting stateful VIDEO-mode inference. It returns the last verified observation as explicitly dropped/backpressure-marked when available instead of taking down the entire multimodal perception graph. No synthetic hand geometry is generated.


## Continuation: configurable authority, temporal depth, and crowd fairness

The distributed camera authority budget is now part of the high-end settings path rather than being a fixed hidden constant. Operators can independently tune the global camera envelope, authoritative dense-view budget, shard count, minimum authority health/synchronization, and temporal residual/outlier contracts. The coordinator can be reconfigured safely between planning decisions and clears hysteresis state when its policy changes.

The native jitter buffer default is aligned to the 128-depth runtime camera queue, increasing burst/reorder tolerance for distributed nodes.

The bounded maximum-weight identity assignment now pre-indexes candidate pairs, removing a repeated full candidate-list scan for each accepted match.

The actor-detail scheduler now uses bounded aging fairness. High-priority subjects still win when evidence differs materially, but actors that have been repeatedly skipped for optional face/surface detail accumulate a capped priority boost. Canonical body identity tracking remains independent from optional detail budgets.

These changes preserve the central high-end design rule: more capacity must not turn into uncontrolled quadratic memory or compute growth. Global capture scale is broad; expensive reconstruction authority stays bounded and evidence-gated.


## Follow-on precision upgrades
1. Timing authority was promoted from telemetry to production admission. NoX now supports detector-neutral visual timing-beacon calibration with exposure-center and row-time correction, robust affine device-clock fitting, explicit source agreement, and reference-grade residual gating.
2. Multi-view temporal optimization now reports joint camera-offset consensus, accepted-camera ratio, and improvement agreement so one camera cannot establish a false global temporal solution.
3. Dense surface reconstruction now retains full symmetric XYZ covariance internally, preserving cross-axis correlation from repeated multi-view innovations with positive-definiteness safeguards and diagonal fallback on degeneracy.
4. Dense evidence transport is lossless at the semantic frame level: large landmark and multi-actor frames are fragmented under the existing packet ceiling, CRC protected, reassembled out of order, expired, and globally bounded to prevent memory growth.
5. High-speed camera acquisition is now motion-clarity aware. Manual exposure is clamped to real Camera2 exposure limits and compared with a frame-rate-derived blur budget; precision presets request short exposure, max resolution, sensor timestamps, and no stabilization unless explicitly overridden.
6. Independent reference comparison is now a first-class validation artifact. It reports measured NoX/reference P95/P99, dropout, temporal lag, jitter, relative advantage, repeatability confidence, and an indeterminate state whenever the cohort is too weak to support a claim.


## Continuation: routed evidence integrity, optical-state conditioning, and memory ceilings

The distributed transport path now treats route overhead as part of packet budgeting rather than assuming every fragment can consume the direct-link payload envelope. Dense landmarks, multi-actor pose, and face-mesh evidence all expose route-aware fragment budgets, while the relay path dispatches routed inner packets into the typed reassembly paths instead of reducing them to opaque raw bytes.

Fragment reassembly is bounded by both assembly count and aggregate resident payload bytes. This prevents a 100+ camera burst of partial frames from converting a bounded object-count policy into an unbounded byte allocation. Face-mesh reassembly also uses one bounded destination allocation instead of repeated full-buffer concatenation.

Calibration artifacts now preserve per-camera lens-focus distance at calibration time. A focus-conditioning engine compares current focus against the calibration state and exposes a conservative geometry-conditioning score. Reference-grade admission requires measured focus compatibility rather than assuming that an autofocus lens preserves identical intrinsics throughout a capture.

The calibration wire contract advances to version 3 while retaining version 1 and version 2 decode compatibility. Version 2 continues to carry uncertainty profiles; version 3 adds optional per-camera focus metadata. The session configuration fingerprint includes the stored focus state so a capture cannot silently reuse a provenance identity after optical calibration conditions change.

Long-session thermal extrinsic authority now uses hysteretic WATCH/RECALIBRATE transitions with immediate hard blocking for explicit failures and delayed recovery for noisy threshold crossings. This reduces state chatter during thermal ramps without hiding genuine calibration failure.

Higher-order actor identity consensus now incorporates calibrated depth disagreement in addition to 2D geometry, temporal consistency, scale and confidence. When depth evidence exists, triplet geometry uses a 3D distance term and reports low-depth-consistency warnings instead of allowing a visually plausible but metrically inconsistent actor chain to remain highly confident.

### Cross-camera photometric consistency
The latest pass adds `CrossCameraPhotometricCalibrationEngine`, a bounded deterministic radiometric alignment layer. It normalizes black/white levels and relative exposure*ISO, then fits constrained per-channel affine transforms with Huber reweighting and explicit saturation/low-signal rejection. The result reports gain/offset, RMSE, rejection ratio, white-balance shift, and consistency without claiming absolute photometric accuracy. Dense landmark evidence now accepts an optional photometric-consistency factor and uses it only as a bounded observation weight.

This direction follows current multi-view reconstruction work showing that ISP and exposure inconsistency can degrade reconstruction. CVPR 2026 PPISP argues for physically interpretable compensation instead of unconstrained per-frame color latents; NoX adopts the deterministic, auditable subset appropriate for an online mobile capture system.

## 2026-10-05 spatial and acquisition continuation

The distributed camera coordinator now measures volumetric camera coverage in addition to pairwise angular separation and baseline. The score uses the normalized spread of calibrated camera positions across X/Y/Z and penalizes near-coplanar camera walls without imposing a universal hard gate. This is intended to improve conditioning for limbs, hands, feet, and dense surfaces when cameras can be placed at different heights and depths.

The camera acquisition profile now carries a capability-clamped dynamic-range mode. High-end calibration, precision, and reference-qualification presets prefer 10-bit capture and the Camera2 HLG10 profile when the device advertises it; otherwise the policy falls back to the standard path and emits an explicit warning. Session fingerprints record the operator's dynamic-range intent.

Reference benchmarking now tracks capture-run identity and run-to-run P95 spread. A single favorable capture cannot qualify a reference comparison; independent runs must meet the repeatability contract.

