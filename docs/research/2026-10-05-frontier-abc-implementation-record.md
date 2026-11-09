# NivukxNoX Frontier A+B+C Implementation Record

## 2026-10-05 research-to-code mapping

### World-space motion
- LAMP (CVPR 2026) uses calibrated camera pose plus 2D keypoints lifted into a shared 3D world frame, followed by a spatio-temporal Transformer. NoX now mirrors the lift-then-fit separation through WorldSpaceMotionProposalEngine and HierarchicalMotionInferenceEngine while keeping calibrated geometry authoritative.
- MAMMA (CVPR 2026 Oral) uses dense 2D surface landmarks with uncertainty, visibility and contact probabilities for multi-person motion acquisition. NoX retains these semantics across DenseWorldPointCorrespondenceEngine, existing dense landmark/contact systems, and the frontier actor scheduler.

### Global 4D reconstruction
- TROPHIES (CVPR 2026) jointly reasons about humans, scenes and cameras with global alignment, scale consistency, contact priors and temporal coherence. NoX's GlobalHumanScene4DOptimizationEngine, FrontierReconstructionOrchestrator and existing Global4DConsistencyEngine form a bounded version of the same architectural direction.
- CVPR 2026 best paper D4RT demonstrates a unified Transformer that estimates depth, spatio-temporal correspondence and camera parameters for dynamic 4D scenes. NoX keeps this as a learned proposal direction rather than replacing the calibrated native geometry spine.

### Sparse-view and continuous-time surfaces
- SV-GS (CVPR 2026) studies sparse-view 4D reconstruction with skeleton-driven deformation. NoX adds SparseViewHumanReconstructionEngine with explicit observed/constrained/inferred state and a learned-prior boundary.
- RetimeGS (CVPR 2026) targets continuous-time 4D Gaussian reconstruction to reduce temporal aliasing. NoX adds ContinuousTimeGaussianSurfaceEngine with bounded histories, interpolation, bounded extrapolation and covariance growth.

- CVPR 2026 HumanBA introduces human-aware bundle adjustment by treating reliable human motion as informative camera constraints instead of masking humans out; NoX adopts the safer production subset as motion-aware reliability weighting around its existing analytic BA.
- CVPR 2026 4DSurf regularizes large dynamic surface deformations with SDF-flow constraints and overlapping temporal segments; NoX preserves metric surface authority and adds continuous-time bounded surface state and Gaussian flow evidence.
- CVPR 2026 RetimeGS addresses temporal aliasing in arbitrary-time 4D Gaussian rendering; NoX exposes continuous-time interpolation/extrapolation with explicit uncertainty inflation.
- CVPR 2026 Complet4R and MotionScale reinforce geometric completion, scalable 4D Gaussian motion fields, and long-sequence propagation as useful high-end directions; NoX keeps inferred completion explicitly lower authority than measured geometry.

### Production interpretation
- Research prototypes are not copied blindly. NoX requires finite-value checks, bounded state, evidence provenance, uncertainty propagation, deterministic tie-breaking and fail-closed authority.
- Learned output is never allowed to silently override calibrated geometry.
- Observed geometry is always higher authority than temporal propagation, learned priors or generative completion.

## Accuracy contract

Sub-millimetre values remain qualification targets. The code cannot prove physical <1 mm performance without independent reference trials. Existing FrontierPrecisionQualificationEngine is the authority for physical target attainment and matched-reference comparison.

## Runtime contract

- 50 actors is the runtime target.
- 64 actors is the hard engineering ceiling.
- 128 cameras is the distributed federation envelope.
- 32 cameras is the bounded authoritative reconstruction set.
- Native frontier memory admission is hard-bounded at the configured domain limit.
- Python simulation tests are not part of the frontier verification strategy.

## Current verification status

GitHub Actions is being observed through the connected workflow interface. Several intermediate commits failed or were cancelled while the connected branch was being reconciled. The final state must not be described as fully green until the latest main-branch checks complete successfully.

## Non-claims

No source file claims that NoX is physically superior to infrared motion capture, achieves <1 mm in all conditions, has nanometre physical resolution, or is literally a century ahead of current technology. Those claims require measured evidence, not architecture prose.

## Relevant primary references
- CVPR 2026 LAMP: https://openaccess.thecvf.com/content/CVPR2026/html/Yang_LAMP_Localization_Aware_Multi-camera_People_Tracking_in_Metric_3D_World_CVPR_2026_paper.html
- CVPR 2026 MAMMA: https://mamma.is.tuebingen.mpg.de/
- CVPR 2026 TROPHIES: https://openaccess.thecvf.com/content/CVPR2026/html/Liu_TROPHIES_Temporal_Reconstruction_of_Places_Humans_and_Cameras_from_Multi-view_CVPR_2026_paper.html
- CVPR 2026 SV-GS: https://openaccess.thecvf.com/content/CVPR2026/html/Chao_SV-GS_Sparse_View_4D_Reconstruction_with_Skeleton-Driven_Gaussian_Splatting_CVPR_2026_paper.html
- CVPR 2026 RetimeGS: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RetimeGS_Continuous-Time_Reconstruction_of_4D_Gaussian_Splatting_CVPR_2026_paper.html
- CVPR 2026 D4RT best paper summary: https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers

### Latest production extensions
- Spatially localized reconstruction detail field: adaptive voxel/mesh allocation now has an optional region-level authority map using uncertainty, multi-view observability, temporal dynamics, semantic importance, curvature and actor presence. This is an allocation layer, not a second geometry truth.
- Face-eye causal skew gate: face and eye evidence must remain within a bounded cross-modal timestamp skew before the pair can become authoritative.
- Native lease identity hardening: memory-domain resets preserve monotonic lease identity so stale pre-reset leases cannot collide with newly acquired leases.
- CI infrastructure finding: runs 1207-1213 fail in the `runner-smoke` job as well as build/test jobs, while the GitHub Actions API exposes no step records or logs for those failures. No manual build or Python simulation was used to diagnose this; source verification remains separate from the infrastructure failure.
