# NoX High-End Hybrid Motion Capture Architecture

## Purpose

NoX now exposes three acquisition paths without removing the existing live motion path:

1. LIVE_MOTION_ONLY: low-latency structured motion. Body, multi-actor association, face, eye, hand and temporal engines remain active. No visual evidence file is created.
2. RECORD_AND_POSTPROCESS: captures the structured motion artifact and an encoded camera evidence stream. The visual stream can be decoded after capture and re-enter the same multimodal perception graph. Calibrated 3D reconstruction is attempted through the canonical MultiViewFusionEngine.
3. HYBRID_LIVE_AND_POST: exposes the live solution to the operator while retaining the recorded visual evidence. Post-capture refinement can therefore spend more computation than the real-time path.

## Real-Time Dataflow

```text
Camera2
  |
  +--> analysis ImageReader
  |       |
  |       +--> pose
  |       +--> multi-person pose
  |       +--> identity association
  |       +--> face / face mesh
  |       +--> eye
  |       +--> hand
  |       +--> temporal / uncertainty / motion fusion
  |
  +--> optional encoder Surface
          |
          +--> HEVC / AVC / AV1 evidence
```

The encoder path uses a camera-fed input Surface and an independent MediaCodec drain worker. This avoids forcing the visual recording path through a CPU-side bitmap/YUV conversion before encoding.

## Distributed Evidence Transport
## Nearby Transport Capacity Policy
The 128-camera figure is a distributed federation envelope, not a promise that one Android host should maintain 128 saturated high-bandwidth links. Nearby Connections byte payloads are constrained by the platform API; NoX therefore keeps byte packets below 32 KiB and uses the existing distributed route/shard planner to avoid a monolithic star for dense evidence. Large recording payloads should use stream-oriented transport rather than inflating control packets. Connection admission remains separately bounded from the global camera envelope.

NoX keeps the existing 32 KiB packet safety boundary instead of weakening it to support large multi-actor evidence. Dense landmark frames may exceed a single packet once a full 50-actor scene is retained, so DenseLandmarkChunkedWireCodec splits the complete evidence set into bounded deterministic chunks. DenseLandmarkFragmentWireCodec wraps each chunk in the versioned NoX wire protocol with CRC protection, and DenseLandmarkFragmentReassembler enforces exact chunk counts, duplicate rejection, expiry, and a global in-flight assembly ceiling. This makes packet loss and out-of-order delivery a bounded quality failure rather than a memory-growth or silent-truncation path.

## Distributed Camera Capacity

NoX now treats very large camera rigs as a distributed evidence federation rather than assuming a single
Android process can own every physical sensor. The global rig boundary is 128 camera identities, while
the reconstruction solver receives a bounded authoritative subset.

The distributed coordinator uses deterministic spatial/sector sharding, per-camera calibration and
timing authority, local shard ranking, global diversity-aware selection, quarantine for timing or
calibration contract breaches, and selection hysteresis. This keeps 100+ camera deployments feasible
as a networked rig without turning the dense reconstruction hot path into an unbounded all-camera
solve.

The Android platform remains capability-driven: a device's own concurrent-camera support is queried
at runtime, and additional cameras are represented by independent capture nodes. The architecture
therefore distinguishes **global rig capacity** from **per-device simultaneous camera capability**.

## Multi-Person Capacity

The runtime target is 50 actors with a bounded 64-actor engineering ceiling.

The capacity governor is not a claim that every Android phone can run full-resolution dense body, face, eye, hand and surface inference for 50 people simultaneously. Instead:

- actor state is bounded;
- body inference is retained ahead of optional expensive modalities;
- face/eye/hand/surface cadence is adapted using CPU, GPU, NPU, thermal and memory headroom;
- actors above the hard ceiling are held outside the active reconstruction graph;
- warnings are persisted instead of silently dropping evidence.

This is a more honest architecture for high-density capture than changing a UI maximum while keeping allocation unbounded.

## Offline Re-Entry

```text
MediaExtractor
    |
MediaCodec decoder
    |
ImageReader YUV_420_888
    |
bounded Image queue
    |
PerceptionPipeline
    |
multi-person association
    |
PostCaptureReconstructionBridge
    |
MultiViewFusionEngine
    |
triangulation + uncertainty + kinematics + physics + temporal + global 4D
```

The queue is bounded and every Image has explicit ownership. A frame that cannot be admitted under backpressure is closed immediately.

The post-capture pass is sequence-safe: the perception graph is reset between independent solves without closing the underlying model instances.

## Multi-Camera Bundle Contract

HybridEvidenceBundleManifest and HybridCameraEvidenceRef provide the contract for aggregating camera recordings.

Bundle admission checks include:

- common session identity;
- duplicate-device rejection;
- common configuration fingerprint;
- temporal overlap;
- calibration coverage;
- hard bounded camera count.

This is deliberately metadata-first. Expensive reconstruction begins only after the evidence relationship is structurally valid.

## Precision Qualification

NoX must not convert a requested error tolerance into a claimed measured result.

A target such as <1 mm is represented as a qualification target, not a guaranteed runtime output. A physically qualified result requires:

- a defined reference system;
- calibrated geometry;
- synchronized measurements;
- matched population and capture conditions;
- sufficient independent evidence;
- explicit uncertainty reporting.

This prevents the UI, logs or model names from making a stronger metrology claim than the captured evidence supports.

## Research-Informed Direction

The architecture draws from several current research directions:

- Apple ML CoMotion (2025) demonstrates online multi-person 3D motion tracking with temporal modeling and explicit handling of occlusion continuity.
- MAMMA (CVPR 2026) demonstrates multi-view markerless multi-person motion acquisition with dense 2D evidence, visibility/uncertainty/contact information and SMPL-X-oriented reconstruction.
- SAM 3D Body (Meta, 2025) demonstrates detailed full-body reconstruction and optimization from image evidence.
- Android Camera2 / MediaCodec surface-input patterns support hardware-facing acquisition and encoding without forcing every stage through CPU image conversion.

NoX uses these as architectural evidence, not as a promise that one mobile camera reproduces the accuracy of a calibrated active optical system under all conditions.

## Production Safety Rules

### Memory

- Bound ImageReader depth and software queues.
- Release every decoder and camera-owned Image deterministically.
- Keep evidence encoding on its own drain worker.
- Fail closed when configured evidence size limits are exceeded.

### Lifecycle

```text
IDLE
 -> PREPARING
 -> ARMED
 -> LIVE | RECORDING | HYBRID
 -> DRAINING
 -> POSTPROCESS_QUEUED
 -> PROCESSING
 -> COMPLETE | DEGRADED | FAILED
```

Camera shutdown precedes encoder EOS processing. ViewModel cleanup closes the post-capture processor, reconstruction bridge, hybrid coordinator, structured recorder, perception graph, GPU resources and camera.

### Determinism

SessionConfigurationFingerprint is schema version 3 and includes evidence mode, recorded video profile, post-capture policy, bounded multi-person capacity policy, and the existing motion/capture configuration.

A capture therefore remains traceable to the configuration that produced it.

## Current Scope

The current implementation upgrades the repository architecture and wires the core live/hybrid/offline boundaries. It does not claim universal 50-person full-density inference on every device, universal sub-millimeter accuracy, or superiority over infrared/active optical systems without measured matched-reference evidence.

The arbitrary file-length requirement is intentionally not implemented by padding source files with repeated or meaningless code. Production complexity is represented through independent, reviewable engines and explicit contracts instead.

## Latest High-End Extensions

The current architecture additionally includes:

- geometry-aware cross-camera actor identity fusion using calibrated reprojection and stable-joint evidence;
- compact binary multi-actor transport so the peer graph carries the complete admitted actor set rather than a single primary pose;
- per-actor face and hand association and identity-preserving dense evidence slots;
- rotating detailed 3D reconstruction across the crowd while retaining continuous standard reconstruction for every admitted actor;
- robust device clock fitting with offset and oscillator-drift estimation;
- sub-frame timing compensation combining exposure-center timing, rolling-shutter risk and timestamp jitter;
- sampled optical evidence quality and sparse photometric motion fields used as temporal camera-motion evidence;
- RAW sensor and depth sidecar retention in the hybrid evidence manifest;
- cryptographic SHA-256 evidence sealing with direct file re-verification;
- matched-reference benchmark contracts with P95/P99, identity, jitter, coverage, latency, thermal and dropped-frame metrics.

These upgrades preserve the central engineering principle: markerless reconstruction should become more accurate by increasing the quality, diversity, timing fidelity and provenance of measured evidence, not by turning confidence labels into physical-distance claims.


## Routed evidence and optical-state authority

Distributed evidence transport is route-budget aware. The global 32 KiB wire safety boundary is never weakened; each routed fragment receives a payload budget derived from the complete hop list, route node identifiers, route envelope and inner fragment header. Dense landmarks, multi-actor pose and face mesh all remain lossless at the semantic-frame level under this nested constraint.

Each fragment reassembler enforces both maximum in-flight assemblies and a maximum aggregate payload byte budget. This prevents partial multi-camera frames from accumulating into an uncontrolled resident-memory spike.

Calibration snapshots preserve per-camera focus distance at calibration time. Focus-conditioned quality is treated as an optical calibration authority signal because focus changes can alter effective intrinsics on real mobile lenses. Reference-grade capture therefore requires matched measured focus state, or an empirically validated focus-to-intrinsics model.

Thermal extrinsic state transitions are hysteretic. Temporary threshold crossings do not immediately revoke a stable camera, while persistent degradation escalates to recalibration and explicit hard failures remain blocking.

Higher-order actor association uses calibrated 3D depth disagreement when available, reducing identity chains that look plausible in 2D but are inconsistent with metric geometry.
