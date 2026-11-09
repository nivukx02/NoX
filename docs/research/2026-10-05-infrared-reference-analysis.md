# NoX high-end optical-reference analysis — 2026-10-05

## Purpose

This record translates current high-end optical motion-capture characteristics into concrete NoX engineering requirements. It does not claim that NoX has experimentally exceeded any commercial optical reference.

## What the reference systems actually optimize

Vicon documents multi-camera calibration as a measurement-volume problem: cameras are calibrated relative to one another and to a physical reference, and the calibration must cover the capture volume. Vicon also explicitly reports different performance for static and dynamic objects, demonstrating that motion changes measurement variability.

OptiTrack describes sub-millimeter marker tracking as a mature hardware/software stack with camera-side processing, high-speed capture, calibration and real-time 3D solving. Its current platform also separates marker-based high-precision tracking from markerless and mixed modes.

The engineering implication is that NoX should treat the following as first-class measurement state:

1. camera calibration geometry and uncertainty;
2. sensor timing and rolling-shutter behavior;
3. baseline and angular observability;
4. cross-view correspondence and identity consistency;
5. dynamic motion compensation;
6. occlusion/contact evidence;
7. uncertainty propagation into every reconstruction representation;
8. independent matched-reference qualification.

## NoX implementation response

### Geometry

NoX retains analytic calibrated triangulation, robust reprojection rejection, parallax checks, camera contribution planning, global 4D refinement and uncertainty-aware sparse/dense reconstruction.

The current continuation adds calibrated world-ray consistency to live dense landmarks and motion-aware rolling-shutter compensation to triangulation. Facial reconstruction receives the same row-time timing authority.

### Identity

The live multi-actor pose engine already uses global assignment, appearance, geometry, velocity continuity, skeleton shape, identity risk and world-joint continuity. The frontier hypothesis engine was strengthened with deterministic Hungarian assignment and an additional velocity-evidence term.

This matters because close actor crossings are underdetermined if identity is inferred from geometry alone.

### Dynamic surface reconstruction

NoX combines sparse-view reconstruction, continuous-time Gaussian surface state, SDF flow regularization, TSDF fusion and mesh extraction. The continuation adds robust local Gaussian surface consensus, uncertainty/provenance weighting, spatially indexed neighborhoods, and native TSDF triangle quality rejection.

### Runtime scalability

The runtime target remains 50 actors with a bounded 64-actor engineering ceiling and a 128-camera distributed federation. The authoritative reconstruction view budget remains bounded so all cameras can contribute evidence without forcing every expensive algorithm to run against every frame.

The distributed transport layer now reserves bandwidth by evidence class: timing, identity, geometry, surface and diagnostics. This is designed to preserve measurement-critical data when a 128-camera federation is bandwidth constrained.

### Memory safety

Camera2 image channels close undelivered image buffers on channel teardown. Native hardware-buffer leases remain bounded by active-lease and resident-byte ceilings, and the JNI diagnostics now expose those ceilings. Sparse TSDF memory remains bounded by both voxel count and estimated resident bytes.

### Face and eye reconstruction

Face contours now preserve capture timing, rolling-shutter skew, exposure duration and signed readout direction. Distributed face reconstruction rejects stale/low-timing-confidence input.

World-space gaze now performs robust spherical consensus and exposes metric binocular target geometry when measured eye origins are available. The existing binocular authority layer also propagates eye-center uncertainty into gaze confidence and point uncertainty.

### Precision qualification

NoX intentionally distinguishes:
- modeled precision feasibility;
- measured benchmark attainment;
- matched-reference qualification.

The qualification system requires empirical reference-aligned measurements before a physical sub-millimetre claim can be considered qualified. A measurement that exceeds a target is now represented as a target-exceedance failure rather than missing evidence.

## Hard engineering conclusion

High-end infrared/active optical systems retain a hardware observability advantage because they deliberately create and observe identifiable markers with calibrated cameras at controlled acquisition conditions. NoX's RGB markerless approach can become highly capable in cases where marker systems are inconvenient, fragile, or insufficiently semantic, but a claim that it is universally more accurate requires matched empirical trials.

The correct NoX objective is therefore not to remove the reference system from engineering reality. It is to build a hybrid-capable software architecture whose RGB measurement quality, uncertainty accounting, occlusion recovery, semantic reconstruction and scalability can be evaluated directly against that reference.

## Qualification requirements for a genuine <1 mm claim

A real <1 mm qualification should require:
- the exact camera/lens/lighting configuration under evaluation;
- calibrated intrinsics and extrinsics with documented uncertainty;
- synchronization and rolling-shutter uncertainty;
- a traceable reference artifact or reference motion system;
- at least the repository's empirical sample floor;
- P50/P95/P99 error distributions;
- repeatability across motion speed, direction, occlusion and camera layouts;
- thermal drift characterization;
- identity-switch and dropout accounting;
- dense-surface error separately from joint-center error;
- confidence/coverage calibration;
- hold-out trials that are not used to tune the model.

The repository should never convert a visual confidence score into a metrology claim.

## Sources

- Vicon accuracy FAQ and dynamic/static accuracy discussion: https://www.vicon.com/support/faqs/how-accurate-precise-are-your-systems/
- Vicon calibration guidance: https://www.vicon.com/resources/blog/how-to-calibrate-a-motion-capture-system-for-accurate-biomechanical-motion-capture/
- OptiTrack motion capture platform: https://www.optitrack.com/
- OptiTrack precision camera information: https://www.optitrack.com/cameras/primex-120
