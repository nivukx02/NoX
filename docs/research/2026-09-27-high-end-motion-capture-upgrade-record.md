# NoX 2026-09-27 high-end motion-capture upgrade record

## Engineering evidence adopted

The 2026 HUM4D/ReGen4D benchmark uses synchronized multi-view RGB-D, accurate calibration and Vicon ground truth, and deliberately stresses inter-person occlusion, identity exchange, close contact, motion blur and depth variation. NoX therefore treats synchronized multiview evidence, occlusion-stratified error, identity continuity and temporal consistency as core production signals instead of optional diagnostics.

Professional optical/infrared systems are useful reference systems, but performance varies with camera geometry, calibration, marker characteristics, motion, volume, environment and sampling configuration. NoX therefore does not assert superiority from architecture alone. A controlled benchmark must supply matched conditions and an uncertainty budget for the reference.

## Upgrade units added

1. Production motion-capture quality authority with explicit contracts for P95/P99 spatial error, lag, dropout, synchronization, calibration, tracking, identity continuity, mesh integrity, face/eye/hand confidence, thermal pressure and memory pressure.
2. Bounded benchmark memory with region-aware statistics and separate low-visibility/occlusion analysis.
3. A reference-comparison gate that leaves optical/infrared comparisons inconclusive when conditions are unmatched or reference uncertainty is missing.
4. Deterministic camera-rig observability and camera selection using baseline, angular diversity, sector coverage, role diversity, camera health and redundancy.
5. Operator UI exposure of production evidence without replacing existing reconstruction controls.
6. No expansion into HDR/HLG/SDR recording. NoX remains motion-capture focused.

## Code-volume policy

NoX already contains large production-oriented files. Artificially padding every file to 10k lines would increase compile time, maintenance cost and defect surface without increasing capability. The upgrade therefore adds real algorithms, contracts, diagnostics and tests rather than filler.

## Next high-end targets

- Reference-calibrated multi-camera benchmark capture.
- Per-joint and per-region residual distributions, not just aggregate averages.
- Stronger sub-frame temporal alignment and rolling-shutter-aware observation timing.
- Memory and lifecycle pressure telemetry at native boundaries.
- Zero-copy GPU/CPU interop only when device capabilities and measured profiling evidence support it.
- Interaction-aware occlusion recovery and contact-constrained reconstruction.
- Rig-level camera admission and deterministic replay of camera selection decisions.


## Latest upgrade tranche: dense landmark and dense metric surface reconstruction

MAMMA (CVPR 2026) reports that dense 2D surface landmarks become substantially more useful when each landmark carries uncertainty, visibility and contact information, especially for close interaction and heavy occlusion. NoX now has that evidence representation in the perception layer and keeps it separate from physical benchmark qualification.

The follow-on reconstruction layer is now DenseMultiViewSurfaceReconstructionEngine. It consumes calibrated multi-camera triangulation observations per dense point and adds:

- robust per-point metric triangulation using the existing TriangulationEngine;
- explicit view diversity and independent-camera accounting;
- directional uncertainty and uncertainty-tail reporting;
- bounded constant-acceleration occlusion bridging;
- temporal point confidence and kinematic continuity;
- topology-edge strain measurement and temporal topology stability;
- contact-support propagation;
- bounded point/observation/history memory and stale eviction;
- fail-closed handling for timestamp reversal, weak parallax, weak cheirality and excessive reprojection error.

The engine is exposed through MultiViewFusionEngine without replacing the established skeleton reconstruction path. It is intentionally optional until the capture transport supplies calibrated dense observations from multiple devices. This prevents a false appearance of multi-camera dense reconstruction when the runtime currently has only a single local view.

Research sources:
- https://openaccess.thecvf.com/content/CVPR2026/html/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.html
- https://openaccess.thecvf.com/content/CVPR2026W/ReGen4D/html/Park_A_Dataset_and_Evaluation_for_Complex_4D_Markerless_Human_Motion_CVPRW_2026_paper.html

The engineering target remains empirical: dense reconstruction architecture may improve robustness, but it does not itself establish sub-millimeter or infrared-superior physical accuracy.

## Latest upgrade tranche: dense evidence authority and production gating

The dense landmark layer now fuses per-view position, confidence, uncertainty, visibility, contact probability, segmentation consistency, rolling-shutter compensation state and reprojection/depth residuals. Robust median initialization plus Huber down-weighting suppresses isolated view outliers before weighted fusion.

The temporal branch keeps bounded constant-acceleration state per landmark and permits only short, explicitly non-authoritative prediction bridges during occlusion. Prediction increases uncertainty and cannot silently satisfy an authoritative cross-view contract.

The production capture quality gate now treats dense coverage, visible coverage, uncertainty P95, cross-view agreement, temporal stability, contact coverage and prediction reliance as first-class evidence. Dense hard failure, severe uncertainty, insufficient coverage and measured cross-view disagreement can block authoritative capture. Operator telemetry exposes the same signals so the UI and capture gate use the same evidence model.

Camera identity is preserved from the runtime perception call into compact dense-fusion camera slots, while multi-actor identity receives bounded subject slots. This prevents accidental collapse of independent camera evidence into a single source when distributed capture is active.

Source-level integrity checks cover brace balance, file growth, TODO/FIXME markers and end-to-end symbol presence. Android/Gradle/CTest execution remains intentionally deferred per the current project constraint; no Python simulation was used.

Research basis:
- MAMMA, CVPR 2026: dense 2D surface landmarks with uncertainty, visibility and contact for occlusion-heavy multi-person markerless capture.
- HUM4D/ReGen4D, CVPR 2026 Workshop: synchronized multi-view RGB-D, calibration, Vicon references and difficult occlusion/identity-exchange scenarios.

## 2026-09-27 refinement and semantic dense-evidence extension

### Subpixel landmark refinement

Added `SubpixelLandmarkRefinementEngine` between pose detection and dense reconstruction. The engine keeps raw detector observations immutable while adding bounded temporal prediction, uncertainty-adaptive gain, robust innovation weighting, rolling-shutter row-time correction, motion-regime process noise, micro-motion preservation, finite-value rejection, stale-state pruning, and bounded state memory. The output is explicitly marked with uncertainty and refinement state instead of being presented as a physical accuracy guarantee.

### Semantic dense evidence expansion

The dense evidence path now receives the refined body landmarks plus additional face mesh points, hand landmarks, left/right eye centers, and multi-actor joints. Each semantic group uses a collision-resistant landmark namespace within the per-subject dense key space. Camera identity remains derived from the real Camera2 camera identifier rather than collapsing evidence to camera 0.

### Production boundary

These changes improve evidence quality, temporal stability, and reconstruction detail, but they do not establish sub-millimeter physical accuracy. A claim below 1 mm still requires matched-condition physical benchmarking against a calibrated reference system with explicit reference uncertainty, synchronization residuals, visibility cohorts, dropout, and tail-error statistics. CVPR 2026 MAMMA provides a useful research precedent for dense landmarks carrying uncertainty, visibility, and contact into multi-view body fitting. citeturn131909search24

### UI direction

The existing Studio uses Liquid Glass primarily for the navigation/control layer while keeping dense production evidence on clearer surfaces. This matches Apple’s current Liquid Glass guidance, which recommends using the material sparingly for controls/navigation and avoiding it as a blanket content-layer treatment. citeturn131909search0turn131909search1
