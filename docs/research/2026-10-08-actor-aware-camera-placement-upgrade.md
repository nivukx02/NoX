# NoX frontier upgrade: actor-aware metric camera authority

Date: 2026-10-08

## Upgrade objective

NoX already exposes distributed camera authority, temporal synchronization, calibration uncertainty, dense reconstruction, interaction/contact evidence and bounded multi-actor scheduling.

This upgrade makes measured camera placement a first-class input to actor-level authority allocation.

## Engineering changes

- CameraEvidence can retain an optional metric camera position and optical forward vector.
- FrontierRigEvidenceCoordinator forwards measured camera pose into authority evidence.
- Actor allocation accepts optional actor world position and occlusion risk.
- CameraPlacementInformationEngine scores optical-axis facing, preferred working distance, angular novelty and baseline conditioning.
- UltraMultiActorCameraLoadBalancer incorporates metric geometry and information novelty without removing the previous visibility, authority, load-balance or hysteresis terms.
- Missing metric anchors are explicit degraded states. The allocator never invents coordinates.

## Production boundary

The 128-camera federation, 32-camera expensive authority budget, 96-camera witness envelope, 50-actor operational target and 64-actor engineering ceiling are preserved.

The arbitrary requirement to pad every file to a fixed number of lines is intentionally not implemented. Repeated source text does not increase reconstruction accuracy, timing authority, calibration quality or numerical stability. The added capability is isolated in typed contracts and a bounded engine instead.

## Precision boundary

Source code cannot establish universal sub-millimeter performance or categorical superiority over infrared or active optical capture. Physical results depend on optics, calibration stability, synchronization, exposure/readout timing, scene texture, occlusion, baseline, target motion and the reference system used for qualification.

The less-than-one-millimeter objective is therefore retained as an empirical qualification target. The production engine must report uncertainty honestly rather than manufacture a sub-millimeter claim.
