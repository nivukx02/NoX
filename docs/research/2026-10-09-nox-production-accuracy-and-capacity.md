# NoX production accuracy, capacity, and capture-UX guardrails

Date: 2026-10-09

## Research findings that affect implementation

- Multi-camera accuracy begins with measured camera intrinsics/extrinsics, lens distortion, stable device identity, calibration coverage and a consistent world scale. OptiTrack recommends periodic recalibration or refinement when camera placement can change: https://docs.optitrack.com/motive/calibration
- Vicon's Shogun Markerless setup likewise treats calibration-volume coverage, wand data, masking/reflections, scale and a fixed origin as explicit preparation stages: https://help.vicon.com/download/attachments/340985765/Getting%20started%20with%20Vicon%20Shogun%20Markerless.pdf
- The CVPR 2026 workshop paper *A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture* reports degraded state-of-the-art performance under severe occlusion, interactions, similar-looking subjects and temporal misalignment, and uses synchronized multi-view RGB-D plus Vicon ground truth to evaluate the result: https://openaccess.thecvf.com/content/CVPR2026W/ReGen4D/papers/Park_A_Dataset_and_Evaluation_for_Complex_4D_Markerless_Human_Motion_CVPRW_2026_paper.pdf
- Android's Material 3 and Compose accessibility guidance supports meaningful selection state, clear hierarchy, adaptive surfaces, scalable content and semantic feedback. Interactive targets should meet at least 48 dp: https://developer.android.com/develop/ui/compose/designsystems/material3 and https://developer.android.com/develop/ui/compose/accessibility/api-defaults

## Consequences for NoX

1. Keep the actor/camera counts in the UI explicitly labelled as configured budgets. A 50-actor target or 128-camera software envelope is not proof that the connected devices can sustain synchronization, inference, transport or reconstruction at those rates.
2. Keep profile-selection state truthful. When a user edits a defining setting, the UI must show **Custom** instead of leaving a preset falsely highlighted.
3. Never present a requested sub-millimetre P95/P99 threshold as achieved accuracy. Only report measured error after evaluating against a matched, independent reference with documented units, time alignment, calibration, volume, actor motions and environmental conditions.
4. Preserve full-body identity continuity even when expensive detail work is budgeted selectively. Track identity swaps, occlusion duration, re-acquisition time and per-actor missing-evidence time as first-class crowd metrics.
5. For production qualification, report at minimum:
   - 3D joint position error distributions (median, P95 and P99), in millimetres, against independent ground truth.
   - Surface error (point-to-surface distance or Chamfer-style metric), missing-surface ratio and temporal surface stability.
   - Actor identity switches, fragmentation, occlusion recovery latency and per-actor coverage.
   - Camera calibration/reprojection residuals, camera-to-camera time skew, dropped-frame rate, and usable-view diversity.
   - End-to-end latency, sustained FPS, thermal throttling, peak memory and recovery after camera/network loss.
6. Qualify 50 actors and 100+ cameras as separate scaling gates before combining them. Test dense interactions, overlapping limbs, similar clothing, subjects leaving/re-entering views, heterogeneous phones, uneven network paths and long-duration thermal steady state. The benchmark must use recorded or live real-camera evidence; synthetic-only results cannot certify real-world accuracy.

## Product UX change in this revision

The Frontier Control panel now marks the selected profile, switches to **Custom** when a profile-defining value is edited, summarizes configured actor/camera/inference budgets, and explains the difference between settings and measured quality. The profile matcher is covered by targeted unit tests. These changes are additive; the app build and test suites were not run in this revision.
