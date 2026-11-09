# NoX Frontier Fusion Upgrade Research — 2026-10-06

## Scope

This document records the external research used to guide the latest feature-additive upgrade of NoX. The goal is a production-oriented RGB/multi-camera motion-capture and reconstruction fabric that improves robustness, scalability, semantics, and uncertainty handling without making unsupported metrology claims.

## Research-derived architecture

### 1. Dense multi-person markerless capture

Recent multi-view markerless capture work demonstrates that dense 2D surface landmarks, visibility, contact probability and uncertainty are valuable complements to sparse body joints. The implementation direction used in NoX therefore keeps dense observations as first-class evidence and makes occlusion/contact signals available to downstream reconstruction rather than treating them as detector-only diagnostics.

Reference:
- MAMMA, CVPR 2026, project and paper materials: https://github.com/cuevhv/mamma

### 2. World-space multi-camera motion reasoning

Modern multi-camera systems increasingly lift calibrated 2D observations into a shared 3D/world frame before applying temporal reasoning. NoX's WorldSpaceMotionProposalEngine follows the same architectural separation: calibrated geometry establishes the base solution, robust temporal fitting suppresses inconsistent rays, and learned proposals remain bounded priors rather than authority.

Reference:
- LAMP, CVPR 2026 project materials: https://github.com/jin-s13/lamp

### 3. Probabilistic bundle adjustment and uncertainty

Probabilistic reconstruction is useful because a single point estimate can hide ill-conditioning, weak baselines and correlated evidence. NoX therefore records covariance-like uncertainty, timing uncertainty and calibration uncertainty separately and combines them only through explicit bounded policies.

Reference:
- ProBA, probabilistic bundle adjustment project/paper materials: https://github.com/cvg/ProBA

### 4. Camera calibration as a graph

Recent calibration research treats cameras as a graph with confidence-weighted pairwise relationships, path consistency and global optimization rather than a flat set of independent calibration records. NoX now contains UltraMultiCameraCalibrationAuthorityEngine, which derives a bounded camera graph, evaluates bidirectional connectivity, loop residuals and uncertainty, and produces active calibration targets.

Reference:
- Multi-Camera Pairwise Calibration Through Joint Optimization, 2026 paper/project materials.

### 5. Continuous-time synchronization

Fast motion makes timestamp mismatch a geometric error, not merely a logging problem. NoX already exposes exposure timing, rolling-shutter behavior and synchronization residuals. The new frontier fusion layer propagates temporal uncertainty into metric uncertainty through an explicit scene-speed bound instead of inventing a distance-based conversion.

Reference directions:
- Universal Online Temporal Calibration, 2026 research direction.
- eKalibr-Inertial, continuous-time calibration and factor-graph research direction.

### 6. Dynamic 3D reconstruction

Dynamic human/scene reconstruction research increasingly combines explicit geometry with learned or implicit temporal deformation. NoX already combines sparse-view reconstruction, TSDF, Gaussian surface refinement, temporal surface flow and global 4D optimization. The new ProbabilisticDynamicSurfaceFusionEngine adds bounded patch state, robust temporal prediction and explicit observed/geometric/temporal/learned/completed provenance.

### 7. Operator interface

Apple's Liquid Glass guidance emphasizes a distinct functional navigation/control layer, restrained transparency, readability support, and accessibility-aware reduction of transparency, motion and contrast. NoX keeps LiquidGlassSurface around controls rather than putting decorative glass across the reconstruction content itself.

References:
- Apple Human Interface Guidelines — Materials: https://developer.apple.com/design/human-interface-guidelines/materials
- Adopting Liquid Glass: https://developer.apple.com/documentation/technologyoverviews/adopting-liquid-glass
- WWDC25 Meet Liquid Glass: https://developer.apple.com/videos/play/wwdc2025/219/

## NoX implementation changes from this research pass

- Added FrontierMeasurementContracts for explicit measurement authority, failure reason and provenance.
- Added UltraFederatedMotionFusionEngine with 128-camera federation support, 32 authority views, 96 witnesses, robust ray fusion and 64-actor bounded scheduling.
- Added ProbabilisticDynamicSurfaceFusionEngine for uncertainty-aware temporal surface state.
- Added FrontierReconstructionCoordinator and exposed it through FrontierReconstructionOrchestrator and FrontierCaptureSessionCoordinator.
- Added UltraMultiCameraCalibrationAuthorityEngine with graph connectivity, loop consistency, uncertainty weighting and active-calibration targets.
- Upgraded WorldSpaceMotionProposalEngine with bounded robust iterative reweighting and learned-prior uncertainty scaling.
- Added frontier runtime configuration controls to HighEndMotionCaptureSettings.
- Added FrontierHighEndConfigurationAdapter to map operator settings into engine-level policies.
- Added FrontierCaptureCapacityCommandCenter and integrated it into the operator surface.
- Hardened native Gaussian eviction and spatial-index invariants with a regression test.
- Upgraded the CI runner image from Ubuntu 24.04 to Ubuntu 26.04 to move away from a failing runner-assignment pool.

## Accuracy boundary

NoX must not claim universal sub-millimetre performance or unconditional superiority over active optical/infrared systems from source code alone. Active optical systems deliberately improve observability with identifiable markers, calibrated capture volumes and specialized cameras. NoX's differentiation should instead be measured through matched trials covering identity continuity, occlusion, contact, dense surface reconstruction, semantics, latency, scalability, and precision tails.

A real sub-millimetre qualification requires independent reference measurements, synchronized coordinate/time systems, calibration uncertainty, P50/P95/P99 errors, repeatability, thermal characterization, camera-layout variation, motion-speed variation, occlusion cohorts, identity-switch accounting, and held-out qualification data.

## Production rule

Line count is not used as a proxy for engineering quality. Files should grow because they contain additional real behavior, interfaces, validation, diagnostics and tests. Padding source files to arbitrary line counts would make the system larger without making it more capable or reliable.

## Final extension pass

Additional production layers added after the initial research record:

- **UltraMultiActorCameraLoadBalancer** distributes per-actor authority camera work across the federation, with per-camera load ceilings, hysteresis, starvation repair and deterministic assignment.
- **AdaptiveActorReconstructionDetailAllocator** allocates body/detail/Gaussian/surface/face/eye/hand budgets by actor uncertainty and scene priority under memory/GPU/NPU pressure.
- **MultiHypothesisOcclusionRecoveryEngine** maintains four bounded motion hypotheses for short occlusion gaps and explicitly labels ambiguous or predicted state as non-authoritative.
- **ReconstructionRepresentationAuthorityEngine** arbitrates TSDF, Gaussian, probabilistic surface and learned completion using measured coverage, uncertainty, topology, temporal stability and runtime cost.
- **AdaptiveCameraPlacementPlanner** emits physically interpretable candidate camera placements using sector coverage, baseline, depth conditioning and occlusion hotspots.
- **Modality-specific actor uncertainty** now propagates face/eye/hand/surface/identity/occlusion confidence into detail allocation.
- **Native Gaussian state** now rejects stale timestamps, sanitizes non-finite velocities and bounds velocity magnitude.

The resulting architecture keeps three separate concerns explicit:
1. measurement acquisition and calibrated geometry;
2. temporal/learned inference and completion;
3. empirical qualification against an independent reference.

This separation is intentional. It prevents visually strong AI reconstruction from being misreported as metrologically verified accuracy.
