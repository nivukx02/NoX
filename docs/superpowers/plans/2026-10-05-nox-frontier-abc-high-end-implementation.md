# NivukxNoX Frontier A+B+C High-End Upgrade Implementation Plan

> REQUIRED SUB-SKILL: Use superpowers:executing-plans because no subagent tool is available in this harness. Tasks use checkbox syntax for tracking.

**Goal:** Integrate A+B+C heterogeneous world-space motion reasoning, reconstruction-heavy dynamic geometry, distributed 128-camera/50-actor runtime control, precision provenance, and high-end Liquid Glass operator surfaces without reducing existing NoX capabilities.

**Architecture:** A is the authoritative native/geometric/runtime spine. B adds learned world-space hypotheses and dense evidence without allowing model output to bypass geometry or uncertainty gates. C adds sparse-view, dynamic-surface, Gaussian, implicit, and 4D refinement as bounded reconstruction stages, with observed geometry always outranking inferred completion.

**Tech Stack:** Kotlin 2.4.10, Android API 30-37, Jetpack Compose Material 3, the existing native NDK/C++ toolchain, coroutines, deterministic replay fixtures, and GitHub Actions verification.

**Spec:** docs/superpowers/specs/2026-10-05-nox-frontier-abc-high-end-design.md

## Global Constraints

- Work on main because the user explicitly authorized main-branch execution.
- Do not remove existing capture modes, body/hand/face/eye tracking, reconstruction, federation, memory controls, diagnostics, or precision qualification.
- Do not add Python simulation tests.
- Do not inflate files with filler solely to satisfy line-count targets.
- Preserve the 50-actor target, 64-actor hard ceiling, 128-camera federation ceiling, and bounded authoritative reconstruction set.
- Keep modeled <1 mm precision targets separate from measured reference-qualified results.
- Keep 'better than infrared' as an empirical qualification hypothesis, never a source-code claim.
- All native allocations, Kotlin histories, queues, diagnostics, fragment reassembly, and reconstruction state remain bounded.
- Every authoritative result carries uncertainty and provenance.

## Review Focus

1. Learned world-space proposals bypassing geometry/uncertainty gates. Test NaN/Inf, stale timestamps, and contradictory camera geometry.
2. Dense reconstruction consuming unbounded inferred surface state. Test hard point/subject/observation budgets and repeated reset/replay.
3. 50-actor scheduling starving difficult/occluded actors. Test fairness under CPU, thermal, and memory pressure.
4. Precision authority promoted by aggregate averages. Test tail residuals, insufficient independent views, insufficient empirical samples, stale timing, and calibration.
5. Liquid Glass reducing readability under hostile scene luminance. Test adaptation and reduced transparency/motion behavior.

---

### Task 1: Shared frontier evidence contracts

**Files**
- Create: core/model/src/main/java/com/nivukx/nox/core/model/FrontierEvidenceModels.kt
- Create: core/model/src/test/java/com/nivukx/nox/core/model/FrontierEvidenceModelsTest.kt

**Interfaces**
- Produces CaptureTimingEvidence, ObservationProvenance, EvidenceOrigin, WorldSpaceProposal, UncertaintyEnvelope, ReconstructionRegionProvenance, and PrecisionAuthoritySnapshot.
- Later tasks consume these immutable value objects.

- [ ] Write failing tests for finite-value rejection, origin ordering, timing midpoint calculation, covariance invariants, and provenance identity.
- [ ] Run the focused model test and verify expected failures.
- [ ] Implement immutable contracts with hard numeric validation.
- [ ] Run the focused model test and verify all assertions pass.
- [ ] Commit: feat: add frontier evidence contracts.

### Task 2: Exposure-midpoint and timing authority

**Files**
- Create: core/sync/src/main/java/com/nivukx/nox/core/sync/ExposureMidpointTimingAuthorityEngine.kt
- Create: core/sync/src/test/java/com/nivukx/nox/core/sync/ExposureMidpointTimingAuthorityEngineTest.kt
- Modify: core/capture/src/main/java/com/nivukx/nox/core/capture/Camera2CaptureController.kt

**Interfaces**
- Consumes Camera2 timing/readout metadata.
- Produces CaptureTimingEvidence with midpoint, row timing, uncertainty, and authority state.

- [ ] Write failing tests for midpoint, rolling-shutter offset, stale-result rejection, duplicate timestamp idempotence, and non-finite metadata.
- [ ] Run targeted tests and observe failures.
- [ ] Implement timing authority and wire it into Camera2 completion without changing capture outputs.
- [ ] Run targeted sync/capture tests and verify pass.
- [ ] Commit: feat: promote exposure midpoint to timing authority.

### Task 3: Learned world-space motion proposal layer

**Files**
- Create: core/ai/src/main/java/com/nivukx/nox/core/ai/WorldSpaceMotionProposalEngine.kt
- Create: core/ai/src/test/java/com/nivukx/nox/core/ai/WorldSpaceMotionProposalEngineTest.kt

**Interfaces**
- Consumes calibrated 2D observations, timing evidence, camera states, and temporal features.
- Produces bounded WorldSpaceProposal candidates with confidence and uncertainty.
- Provides deterministic geometric fallback when learned proposal input is unavailable.

- [ ] Write failing tests for world-space lifting, viewpoint-conditioned weighting, invalid model output rejection, and deterministic fallback.
- [ ] Run focused tests and observe failures.
- [ ] Implement model-agnostic proposal interfaces plus deterministic geometry proposals and confidence fusion.
- [ ] Run focused tests and verify pass.
- [ ] Commit: feat: add learned world-space motion proposals.

### Task 4: Hierarchical motion inference

**Files**
- Create: core/ai/src/main/java/com/nivukx/nox/core/ai/HierarchicalMotionInferenceEngine.kt
- Create: core/ai/src/test/java/com/nivukx/nox/core/ai/HierarchicalMotionInferenceEngineTest.kt
- Modify: core/ai/src/main/java/com/nivukx/nox/core/ai/MotionCaptureEngine.kt
- Modify: core/ai/src/main/java/com/nivukx/nox/core/ai/PerceptionPipeline.kt

**Interfaces**
- Consumes WorldSpaceProposal, existing motion state, timing evidence, and uncertainty.
- Produces authoritative hierarchical motion state with observed/inferred/predicted origin, acceleration/jerk, and bounded multi-hypothesis recovery.

- [ ] Write failing tests for regime changes, acceleration/jerk bounds, multi-hypothesis occlusion recovery, timestamp replay, and uncertainty-aware proposal fusion.
- [ ] Run focused tests and observe failures.
- [ ] Implement hierarchical state estimator and integrate it after current conditioning without removing public behavior.
- [ ] Run focused AI tests and verify pass.
- [ ] Commit: feat: upgrade motion inference to hierarchical world-space fusion.

### Task 5: Dense world-space point correspondence

**Files**
- Create: core/ai/src/main/java/com/nivukx/nox/core/ai/DenseWorldPointCorrespondenceEngine.kt
- Create: core/ai/src/test/java/com/nivukx/nox/core/ai/DenseWorldPointCorrespondenceEngineTest.kt
- Modify: core/ai/src/main/java/com/nivukx/nox/core/ai/DenseLandmarkEvidenceFusionEngine.kt

**Interfaces**
- Consumes dense 2D landmarks, visibility/contact/uncertainty, calibration, and actor identity.
- Produces persistent world-space point IDs, cross-view correspondences, and region provenance.

- [ ] Write failing tests for cross-view matching, occlusion handling, identity continuity, bounded state, and out-of-order timestamps.
- [ ] Run targeted tests and observe failures.
- [ ] Implement geometry-first correspondence with appearance as secondary evidence and strict caps.
- [ ] Run targeted tests and verify pass.
- [ ] Commit: feat: add dense world-space point correspondence.

### Task 6: Multi-actor hypothesis association

**Files**
- Create: core/ai/src/main/java/com/nivukx/nox/core/ai/MultiActorHypothesisAssociationEngine.kt
- Create: core/ai/src/test/java/com/nivukx/nox/core/ai/MultiActorHypothesisAssociationEngineTest.kt
- Modify: core/ai/src/main/java/com/nivukx/nox/core/ai/MultiActorPoseAssociationEngine.kt

**Interfaces**
- Consumes current actor observations and dense world-space correspondence output.
- Produces deterministic one-to-one actor assignments plus bounded alternate hypotheses.

- [ ] Write failing tests for crossing actors, contradictory appearance, ambiguous geometry, depth-compatible candidates, and hypothesis expiry.
- [ ] Run targeted tests and observe failures.
- [ ] Implement staged candidate pruning followed by deterministic assignment and bounded alternates.
- [ ] Run targeted AI tests and verify pass.
- [ ] Commit: feat: harden actor identity with bounded hypotheses.

### Task 7: Dense face surface reconstruction

**Files**
- Create: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/DenseFaceSurfaceReconstructionEngine.kt
- Create: core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/DenseFaceSurfaceReconstructionEngineTest.kt
- Modify: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/FaceEyeEvidenceFusionEngine.kt

**Interfaces**
- Consumes dense face landmarks, camera geometry, timing evidence, freshness, and eye state.
- Produces world-space face surface state with per-region uncertainty and observed/inferred provenance.

- [ ] Write failing tests for multi-view face fusion, stale fallback, occlusion, temporal flow, and covariance.
- [ ] Run focused tests and observe failures.
- [ ] Implement dense face surface fusion and wire it into existing face/eye authority without removing outputs.
- [ ] Run focused reconstruction tests and verify pass.
- [ ] Commit: feat: add dense multi-view face surface authority.

### Task 8: High-end eye and gaze geometry

**Files**
- Create: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/WorldSpaceGazeReconstructionEngine.kt
- Create: core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/WorldSpaceGazeReconstructionEngineTest.kt
- Modify: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/FaceEyeEvidenceFusionEngine.kt
- Modify: core/native-engine/src/main/cpp/eyes/gaze_fusion.cpp

**Interfaces**
- Consumes per-eye rays, eyelid/blink confidence, head pose, timing, and IPD calibration.
- Produces head-relative and world-relative gaze, saccade/fixation state, angular uncertainty, and provenance.

- [ ] Write failing tests for binocular consensus, outlier rejection, blink suppression, timing uncertainty, and frame transforms.
- [ ] Run Kotlin/native focused tests and observe failures.
- [ ] Implement robust spherical consensus and native acceleration while preserving current JNI contracts.
- [ ] Run focused Kotlin/native tests and verify pass.
- [ ] Commit: feat: upgrade world-space gaze reconstruction.

### Task 9: Sparse-view human reconstruction

**Files**
- Create: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/SparseViewHumanReconstructionEngine.kt
- Create: core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/SparseViewHumanReconstructionEngineTest.kt
- Modify: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/MultiViewFusionEngine.kt

**Interfaces**
- Consumes calibrated observations, actor identity, dense correspondence, body priors, and timing.
- Produces geometry with explicit observed/constrained/inferred regions and sparse-view diagnostics.

- [ ] Write failing tests for two-view/three-view degeneracy, low-overlap placement, learned-prior fallback, and inferred-region labeling.
- [ ] Run focused tests and observe failures.
- [ ] Implement decoupled sparse-view human/background strategy and adaptive view selection.
- [ ] Run focused tests and verify pass.
- [ ] Commit: feat: add sparse-view geometry-first human reconstruction.

### Task 10: Dynamic Gaussian and surface-flow refinement

**Files**
- Create: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/DynamicGaussianSurfaceRefinementEngine.kt
- Create: core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/DynamicGaussianSurfaceRefinementEngineTest.kt
- Modify: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/DenseMultiViewSurfaceReconstructionEngine.kt
- Modify: core/native-engine/src/main/cpp/scene/gaussian_deformation_graph.cpp

**Interfaces**
- Consumes surface state, normals, covariance, temporal flow, sparse-view uncertainty, and actor motion.
- Produces bounded Gaussian surface refinements and mesh-ready states.

- [ ] Write failing tests for bounded Gaussian count, full covariance propagation, temporal surface-flow regularization, and observed-over-inferred authority.
- [ ] Run focused tests and observe failures.
- [ ] Implement deformation graph refinement, covariance-aware fusion, and adaptive detail tiers.
- [ ] Run focused Kotlin/native reconstruction tests and verify pass.
- [ ] Commit: feat: add dynamic Gaussian surface refinement.

### Task 11: Global 4D optimizer

**Files**
- Create: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/GlobalHumanScene4DOptimizationEngine.kt
- Create: core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/GlobalHumanScene4DOptimizationEngineTest.kt
- Modify: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/Global4DConsistencyEngine.kt

**Interfaces**
- Consumes camera trajectories, actor trajectories, surface states, contact constraints, and provenance.
- Produces bounded global refinement, residuals, and failure attribution.

- [ ] Write failing tests for camera/actor/surface coupling, iteration budget, divergence detection, contact rejection, and deterministic reruns.
- [ ] Run focused tests and observe failures.
- [ ] Implement bounded global optimization using existing native solver facilities where practical.
- [ ] Run focused reconstruction/native tests and verify pass.
- [ ] Commit: feat: add bounded global human-scene 4D optimization.

### Task 12: 50-actor fairness scheduler

**Files**
- Create: core/ai/src/main/java/com/nivukx/nox/core/ai/FrontierActorDetailScheduler.kt
- Create: core/ai/src/test/java/com/nivukx/nox/core/ai/FrontierActorDetailSchedulerTest.kt
- Modify: core/ai/src/main/java/com/nivukx/nox/core/ai/MultiPersonRuntimeCapacityGovernor.kt
- Modify: core/ai/src/main/java/com/nivukx/nox/core/ai/PerceptionPipeline.kt

**Interfaces**
- Consumes actor uncertainty, visibility, motion complexity, interaction risk, system headroom, and stage costs.
- Produces per-actor body/face/eye/hand/dense/surface cadence allocations.

- [ ] Write failing tests for fairness, starvation prevention, severe-pressure body priority, recovery hysteresis, and the 64-actor ceiling.
- [ ] Run focused tests and observe failures.
- [ ] Implement priority allocation with aging/fairness and bounded stage activation.
- [ ] Run focused tests and verify pass.
- [ ] Commit: feat: add fairness-aware actor detail scheduling.

### Task 13: 128-camera authority federation

**Files**
- Create: core/sync/src/main/java/com/nivukx/nox/core/sync/FrontierCameraAuthorityFederationEngine.kt
- Create: core/sync/src/test/java/com/nivukx/nox/core/sync/FrontierCameraAuthorityFederationEngineTest.kt
- Modify: core/sync/src/main/java/com/nivukx/nox/core/sync/MultiCameraSynchronizationQualityEngine.kt

**Interfaces**
- Consumes camera health, timing, calibration, geometric coverage, actor visibility, thermal state, and network state.
- Produces bounded 32-authoritative + 96-witness selection with quarantine and hysteresis.

- [ ] Write failing tests for 128-camera admission, authority selection, sector diversity, timing/calibration quarantine, recovery hysteresis, and actor visibility.
- [ ] Run focused tests and observe failures.
- [ ] Implement deterministic federation selection without requiring one Android process to own every remote sensor.
- [ ] Run focused sync/network tests and verify pass.
- [ ] Commit: feat: strengthen 128-camera evidence federation.

### Task 14: Heterogeneous runtime execution planner

**Files**
- Create: core/native-engine/src/main/cpp/runtime/heterogeneous_execution_planner.cpp
- Create: core/native-engine/src/main/cpp/runtime/heterogeneous_execution_planner.h
- Create: core/native-engine/src/main/cpp/tests/heterogeneous_execution_planner_test.cpp
- Modify: core/native-engine/src/main/cpp/runtime/native_engine_runtime.cpp

**Interfaces**
- Consumes stage cost, backend capability, thermal state, memory headroom, and precision sensitivity.
- Produces deterministic backend selection and fallback decisions.

- [ ] Write failing native tests for backend selection, unavailable-backend fallback, thermal shedding, precision-sensitive routing, and memory admission.
- [ ] Run the focused native test and observe failure.
- [ ] Implement planner and integrate it without changing the public ABI.
- [ ] Run focused native tests and verify pass.
- [ ] Commit: feat: add heterogeneous execution planning.

### Task 15: Hard native memory admission

**Files**
- Create: core/native-engine/src/main/cpp/memory/frontier_memory_domain.cpp
- Create: core/native-engine/src/main/cpp/memory/frontier_memory_domain.h
- Create: core/native-engine/src/main/cpp/tests/frontier_memory_domain_test.cpp
- Modify: core/native-engine/src/main/cpp/runtime/hardware_buffer_lease_registry.cpp
- Modify: core/native-engine/src/main/cpp/memory/native_buffer_pool.cpp

**Interfaces**
- Produces per-domain hard byte budgets and admission/release accounting consumed by runtime and reconstruction.

- [ ] Write failing native tests for budget admission, double release, domain isolation, repeated acquire/release, and pressure recovery.
- [ ] Run focused native test and observe failure.
- [ ] Implement bounded domain accounting and connect it to existing lease/pool structures.
- [ ] Run focused native memory tests and verify pass.
- [ ] Commit: feat: enforce frontier native memory domains.

### Task 16: Reconstruction provenance and precision authority

**Files**
- Create: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/ReconstructionProvenanceEngine.kt
- Create: core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/ReconstructionProvenanceEngineTest.kt
- Modify: core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/HighEndReconstructionQualityEngine.kt
- Modify: core/ai/src/main/java/com/nivukx/nox/core/ai/ProductionMotionCaptureQualityGate.kt

**Interfaces**
- Consumes per-stage evidence and quality metrics.
- Produces provenance fingerprints and precision authority states.

- [ ] Write failing tests for observed/inferred separation, focus/calibration mismatch, insufficient independent views, tail residual rejection, and empirical sample floors.
- [ ] Run focused tests and observe failures.
- [ ] Implement provenance aggregation and make it authoritative for precision promotion.
- [ ] Run targeted quality tests and verify pass.
- [ ] Commit: feat: harden reconstruction provenance and precision authority.

### Task 17: High-end Liquid Glass operator deck

**Files**
- Create: app/src/main/java/com/nivukx/nox/ui/studio/FrontierOperatorDeck.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/CameraFederationInspector.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/ActorEvidenceMatrix.kt
- Create: app/src/main/java/com/nivukx/nox/ui/studio/ReconstructionProvenanceInspector.kt
- Create: app/src/test/java/com/nivukx/nox/ui/studio/FrontierOperatorDeckTest.kt
- Modify: app/src/main/java/com/nivukx/nox/ui/studio/LiquidGlassSurface.kt
- Modify: app/src/main/java/com/nivukx/nox/ui/studio/StudioGlassTokens.kt

**Interfaces**
- Consumes studio UI state, precision authority, camera federation, actor scheduler, and memory/thermal telemetry.
- Produces adaptive Material 3/Liquid Glass operator surfaces with readable plates and explicit operational states.

- [ ] Write failing Compose/model tests for state hierarchy, readability adaptation, reduced transparency, reduced motion, and critical-state persistence.
- [ ] Run focused UI tests and observe failures.
- [ ] Implement operator deck and inspectors without deleting current surfaces.
- [ ] Run focused UI tests and verify pass.
- [ ] Commit: feat: add frontier Liquid Glass operator deck.

### Task 18: Advanced configuration and provenance export

**Files**
- Create: core/model/src/main/java/com/nivukx/nox/core/model/FrontierCaptureProfile.kt
- Create: core/model/src/test/java/com/nivukx/nox/core/model/FrontierCaptureProfileTest.kt
- Modify: core/model/src/main/java/com/nivukx/nox/core/model/HighEndMotionCaptureSettings.kt
- Modify: app/src/main/java/com/nivukx/nox/NoXViewModel.kt

**Interfaces**
- Provides immutable advanced capture configuration and serialization-ready profile fingerprint.
- ViewModel exposes configuration without embedding solver logic.

- [ ] Write failing settings tests for bounded ranges, compatibility defaults, fingerprint stability, and invalid combinations.
- [ ] Run focused tests and observe failures.
- [ ] Implement profile model and adapter into existing settings.
- [ ] Run focused model/app tests and verify pass.
- [ ] Commit: feat: expose frontier capture configuration.

### Task 19: Deterministic replay contract

**Files**
- Create: core/native-engine/src/main/cpp/replay/frontier_replay_contract.cpp
- Create: core/native-engine/src/main/cpp/replay/frontier_replay_contract.h
- Create: core/native-engine/src/main/cpp/tests/frontier_replay_contract_test.cpp
- Modify: core/native-engine/src/main/cpp/replay/replay_runner.cpp

- [ ] Write failing replay tests for timing corruption, identity crossing, sparse-view reconstruction, memory pressure, and deterministic reruns.
- [ ] Run focused replay test and observe failures.
- [ ] Implement contract validation and deterministic output digesting.
- [ ] Run focused replay tests and verify pass.
- [ ] Commit: test: add frontier deterministic replay contract.

### Task 20: Repository verification and documentation

**Files**
- Modify: README.md
- Modify: docs/architecture/HIGH_END_HYBRID_MOTION_CAPTURE.md
- Create: docs/research/2026-10-05-frontier-abc-implementation-record.md

- [ ] Document A+B+C integration, evidence boundaries, physical-accuracy boundaries, and qualification requirements.
- [ ] Run repository-native Kotlin and native C++ test suites and collect exact counts.
- [ ] Run static/compile verification that does not invoke Python simulation.
- [ ] Review git diff against the spec and plan and fix Critical/Important findings in one pass.
- [ ] Commit: docs: record frontier A+B+C upgrade qualification.

### Task 21: Fresh whole-branch review

- [ ] Review the complete main diff against the spec and plan as a separate authoring pass.
- [ ] Re-grade findings by user-facing impact.
- [ ] Critical/Important findings require a failing regression test, root-cause fix, and fresh relevant-suite verification.
- [ ] Record all rulings and deferred minor findings.
- [ ] Verify final repository state before any completion claim.