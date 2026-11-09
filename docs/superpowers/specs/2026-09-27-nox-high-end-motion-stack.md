# NivukxNoX High-End Motion Stack Specification
Date: 2026-09-27
Branch: main

## Objective
Upgrade the existing NivukxNoX motion-capture stack without removing established capabilities. The immediate slice strengthens temporal motion inference, empirical reference-grade precision gating, fail-closed runtime policy, and operator evidence presentation.

## Non-negotiable engineering constraints
- Motion-capture scope only. No HDR/HLG/SDR video-recording feature work.
- Existing body, hand, face, eye, calibration, synchronization, reconstruction, lighting telemetry, 4D consistency, and Liquid Glass capabilities remain additive.
- Reference-grade accuracy is an evidence contract, not a marketing claim.
- A 2 mm target is only operationally claimable when an empirical benchmark exists under matched conditions and the external reference uncertainty is documented.
- Quality-lock mode must never silently lower fidelity.
- Adaptive quality may reduce optional workload only when quality-lock is disabled.
- Occlusion recovery must be bounded and must never output unbounded hallucinated trajectories.
- All hot-path state is bounded by explicit capacity/time budgets.
- No artificial line-count padding. Production quality comes from independent, testable modules and evidence-bearing code.
- This pass does not launch the Android application build. CI-triggering commits use [skip ci].

## Research-derived architecture decisions
1. Use visibility-aware and contact-aware temporal evidence because current markerless multi-view research shows severe degradation under heavy occlusion and interaction.
2. Treat view diversity, reprojection agreement, synchronization, and temporal consistency as first-class signals.
3. Keep metric uncertainty anisotropic; shallow parallax must increase depth uncertainty instead of hiding it inside a scalar confidence.
4. Separate modeled precision feasibility from measured benchmark evidence.
5. Keep Liquid Glass in the operator/navigation layer and preserve reconstruction content readability. Use adaptive layouts and large touch targets.

## Acceptance criteria for this slice
- A bounded multi-camera temporal fusion engine exists with robust outlier handling, occlusion recovery, prediction limits, confidence decay, and deterministic lifecycle transitions.
- An empirical reference-grade evidence gate exists and blocks 2 mm reference mode without benchmark evidence.
- MotionCaptureCustomizationEngine consumes temporal fusion evidence and reference-grade empirical evidence without weakening existing gates.
- Operator telemetry can expose whether a precision target is modeled, measured, reference-qualified, or blocked.
- Tests cover out-of-order timestamps, duplicate camera evidence, hard outliers, bounded prediction, occlusion decay, recovery, and empirical reference gating.
