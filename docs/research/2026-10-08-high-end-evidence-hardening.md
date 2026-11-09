# NoX high-end evidence hardening: 2026-10-08

## Scope

This continuation hardens three production boundaries without removing existing functionality:

1. temporal motion estimation now consumes measured exposure-integration blur as a first-class trust signal;
2. multi-view reconstruction view ordering now accounts for calibration uncertainty, timing uncertainty, rolling-shutter risk and optical blur in addition to geometry and detector quality;
3. the Liquid Glass operator surface exposes accessibility semantics while preserving the existing adaptive transparency and motion-reduction controls.

## Research basis

Recent multi-person markerless capture research continues to emphasize dense landmarks, explicit uncertainty, visibility, contact and calibrated multi-view correspondence under close interaction and occlusion. MAMMA (CVPR 2026) reports uncertainty and visibility alongside dense 2D landmark locations and builds cycle-consistent multi-view correspondence before reconstruction.

Apple's current Liquid Glass guidance places the material in the functional navigation/control layer, recommends restrained use, and explicitly ties appearance to legibility and accessibility-related system settings. Android Compose guidance likewise recommends semantic labels, roles, state and actions for custom composables.

Primary references:

- MAMMA, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/papers/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.pdf
- Apple Human Interface Guidelines: Materials: https://developer.apple.com/design/human-interface-guidelines/materials
- Apple Liquid Glass overview: https://developer.apple.com/documentation/technologyoverviews/liquid-glass
- Android Compose accessibility: https://developer.android.com/guide/topics/ui/accessibility/composables

## Engineering rules preserved

- Existing capture and reconstruction paths remain additive.
- No accuracy guarantee is promoted from a modeled target to a physical claim.
- 50 actors remains the runtime target and 128 cameras remains the distributed federation envelope already established by the repository.
- No Python simulation testing is introduced.
- This slice is source-level verified only; the Android/Gradle build was intentionally not started in this pass.
- Artificial line-count expansion is not used as a correctness metric; implementation quality is measured by real behavior, boundedness, evidence provenance, and testable contracts.

## Next priority

Promote the same evidence discipline into dense surface reconstruction: carry anisotropic uncertainty, camera information gain and temporal uncertainty into the final surface confidence and mesh extraction budget, then qualify the failure paths for sparse, occluded and high-blur views.
