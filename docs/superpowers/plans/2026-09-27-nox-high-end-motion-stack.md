# Plan: NivukxNoX High-End Motion Stack
Date: 2026-09-27

## Task 1 - Robust temporal fusion engine
Files:
- core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/HighEndTemporalFusionEngine.kt
- core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/HighEndTemporalFusionEngineTest.kt

Implement a bounded multi-camera temporal estimator with robust Huber-style weighting, duplicate-view fusion, velocity/acceleration state, deterministic outlier rejection, bounded prediction during occlusion, confidence decay, lifecycle transitions, and reset/close semantics.

RED:
- Write tests first for out-of-order observations, hard outliers, duplicate-view fusion, bounded prediction, confidence decay, recovery, and resource bounds.
- Verify the isolated test fails before the engine exists.

GREEN:
- Implement the minimal production engine that satisfies those tests.
- Run an isolated Kotlin/JVM harness against the engine and tests without invoking the Android build.

Expected:
- Deterministic summaries.
- No unbounded state growth.
- No negative or NaN confidence/state values.

## Task 2 - Empirical reference-grade gate
Files:
- core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/ReferenceGradeEvidenceGateEngine.kt
- core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/ReferenceGradeEvidenceGateEngineTest.kt

Implement a strict evidence gate that distinguishes modeled feasibility from measured qualification. Require explicit matched-condition benchmark status, target P95/P99 evidence, sample sufficiency, and documented reference uncertainty before a 2 mm reference-grade mode can be declared ready.

RED:
- Write tests for missing benchmark evidence, measured P95 above target, undocumented reference uncertainty, insufficient samples, and qualified evidence.
GREEN:
- Implement fail-closed gate semantics.

## Task 3 - Integrate into runtime policy
Files:
- core/ai/src/main/java/com/nivukx/nox/core/ai/MotionCaptureCustomizationEngine.kt
- core/ai/src/test/java/com/nivukx/nox/core/ai/MotionCaptureCustomizationEngineTest.kt
- core/model/src/main/java/com/nivukx/nox/core/model/HighEndMotionCaptureSettings.kt
- core/model/src/test/java/com/nivukx/nox/core/model/HighEndMotionCaptureSettingsTest.kt

Add empirical reference evidence to RuntimeEvidence and integrate the temporal fusion confidence and evidence gate into resolve/resolveWithMotionEvidence. Preserve existing quality locks and reason codes.

RED:
- Add regression tests proving reference mode blocks without empirical evidence and temporal fusion can lower confidence without corrupting the plan.
GREEN:
- Integrate without removing existing paths.

## Task 4 - Operator evidence surface
Files:
- app/src/main/java/com/nivukx/nox/ui/studio/ReferenceGradeEvidenceCard.kt
- app/src/main/java/com/nivukx/nox/ui/studio/StudioTelemetryState.kt
- app/src/main/java/com/nivukx/nox/ui/studio/MotionCaptureCustomizationPanel.kt

Expose modeled-versus-measured precision provenance, benchmark status, reference uncertainty, and explicit block reasons. Keep Liquid Glass only in the functional/operator layer and retain responsive 48dp controls.

RED:
- Add source-integrity tests for the new UI contract and state mapping.
GREEN:
- Integrate the panel without changing reconstruction rendering.

## Verification
- Run isolated JVM tests for Tasks 1-2 locally with kotlinc.
- Inspect GitHub diffs and file contents after every commit.
- Do not run Android assemble/build in this pass.
- Use [skip ci] commit messages so the existing push-triggered Android and native workflows are not started.

## Review focus
- Numerical stability and deterministic state transitions.
- Bounds on memory, tracks, observations, and prediction horizons.
- No accidental accuracy claims.
- No quality downgrade under locked contracts.
- UI readability and evidence provenance.
