# NoX MotionField Fusion Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved MotionField Fusion Preview as a production-grade GPU camera/motion compositor with timestamp synchronization, multi-actor overlays, runtime GPU policy, and graceful fallback.

**Architecture:** Keep Camera2CaptureController authoritative. Introduce a GPU-facing preview boundary where Camera2 writes into a SurfaceTexture-backed camera input surface, the renderer samples the external texture, and authoritative NoX actor/reconstruction state is composited in GPU geometry. Use GLES 3.1+ as the implemented backend, with capability-driven backend selection reserved for a validated Vulkan path rather than a fake implementation.

**Tech Stack:** Android Kotlin, OpenGL ES 3.1, SurfaceTexture/Surface, Compose, existing NoX core-model/core-ai/core-reconstruction modules.

**Spec:** `docs/superpowers/specs/2026-09-24-nox-motionfield-fusion-preview-design.md`

## Global Constraints

- Always modify `main`.
- Do not remove or downgrade existing perception/reconstruction/capture functionality.
- Do not artificially inflate source files with repetitive code.
- The Android app build/run remains unstarted during this pass.
- Sensor timestamps are the preview time authority.
- Predicted motion must remain visually and semantically distinguishable from measured motion.
- GPU acceleration must be capability-driven and must never bypass capture-integrity or thermal protection.
- Renderer failure must not terminate camera capture.
- Maximum actor capacity remains 32.
- Existing liquid-glass UI architecture remains intact.

## Review Focus

- Camera SurfaceTexture lifecycle and race-free release under surface recreation.
- Monotonic timestamp selection when inference is slower than display refresh.
- Prediction horizon clamping and stale-data rendering.
- GPU capability fallback when OpenGL ES 3.1 is unavailable.
- Renderer failure isolation so capture survives shader/context errors.

---

### Task 1: Motion preview model contract

**Files:**
- Modify: `core/model/src/main/java/com/nivukx/nox/core/model/NoXModels.kt`
- Create: `core/model/src/test/java/com/nivukx/nox/core/model/MotionFieldPreviewModelTest.kt`

**Interfaces:**
- Produces `MotionFieldPreviewFrame`, `MotionFieldActor`, `MotionFieldJoint`, `MotionFieldGpuState`, and `MotionFieldCustomization` for later tasks.

- [ ] **Step 1: Write the failing test**

```kotlin
@Test
fun preview_frame_rejects_invalid_timestamp_and_confidence() {
    assertFailsWith<IllegalArgumentException> {
        MotionFieldPreviewFrame(
            timestampNanos = -1L,
            cameraWidth = 1920,
            cameraHeight = 1080,
            displayTimestampNanos = 0L,
            actors = emptyList(),
            reconstructionConfidence = 0.8,
            gpu = MotionFieldGpuState(),
        )
    }
}
```

- [ ] **Step 2: Run the pure Kotlin test and verify it fails because the type does not exist.**
- [ ] **Step 3: Add the immutable preview records with strict validation, bounded confidence fields, actor/joint confidence, prediction state, micro-motion metadata, GPU telemetry, and customization flags.**
- [ ] **Step 4: Run the test and verify it passes.**
- [ ] **Step 5: Add tests for bounded actor count, normalized joint coordinates, and stale/prediction flags.**
- [x] **Step 6: Commit `feat(model): add MotionField preview contracts`.**

### Task 2: Timestamp-aware MotionField synchronizer

**Files:**
- Create: `core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/MotionFieldFrameSynchronizer.kt`
- Create: `core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/MotionFieldFrameSynchronizerTest.kt`

**Interfaces:**
- Consumes `MotionFieldPreviewFrame` candidates and sensor/display timestamps.
- Produces a `Selection` containing source timestamp, target timestamp, interpolation/extrapolation factor, stale flag, and bounded prediction horizon.

- [ ] **Step 1: Write failing tests for nearest-frame selection, stale rejection, and bounded extrapolation.**
- [ ] **Step 2: Run the pure Kotlin tests and observe the expected missing-type failure.**
- [ ] **Step 3: Implement timestamp ordering, nearest candidate selection, bounded extrapolation, and stale-state classification.**
- [ ] **Step 4: Run tests and verify they pass.**
- [ ] **Step 5: Add a regression test proving out-of-order frames cannot move the synchronizer backward in time.**
- [x] **Step 6: Commit `feat(reconstruction): synchronize live motion preview by sensor time`.**

### Task 3: GPU capability and detail policy

**Files:**
- Create: `core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/MotionFieldGpuPolicyEngine.kt`
- Create: `core/reconstruction/src/test/java/com/nivukx/nox/core/reconstruction/MotionFieldGpuPolicyEngineTest.kt`

**Interfaces:**
- Consumes backend capabilities, thermal scale, memory scale, user GPU mode, actor count, and preview detail.
- Produces a deterministic `Decision` with backend, geometry budget, update budget, and warnings.

- [ ] **Step 1: Write failing tests for maximum-GPU selection, GLES fallback, and thermal protection.**
- [ ] **Step 2: Verify the tests fail before implementation.**
- [ ] **Step 3: Implement capability-driven policy.**
- [ ] **Step 4: Run tests and verify they pass.**
- [ ] **Step 5: Add tests proving policy never selects an unavailable backend or violates the configured thermal floor.**
- [x] **Step 6: Commit `feat(reconstruction): add MotionField GPU policy engine`.**

### Task 4: Production GLES MotionField renderer

**Files:**
- Create: `app/src/main/java/com/nivukx/nox/graphics/MotionFieldGlesRenderer.kt`
- Create: `app/src/main/java/com/nivukx/nox/graphics/MotionFieldPreviewSurface.kt`
- Create: `app/src/main/java/com/nivukx/nox/graphics/MotionFieldShaders.kt`
- Create: `app/src/test/java/com/nivukx/nox/graphics/MotionFieldRendererContractTest.kt`

**Interfaces:**
- `MotionFieldPreviewSurface` exposes `setCameraSurfaceListener((Surface?) -> Unit)` and `submitFrame(MotionFieldPreviewFrame)`.
- `MotionFieldGlesRenderer` owns EGL/GL resources, camera SurfaceTexture, external OES texture, shader programs, geometry buffers, and frame rendering state.

- [ ] **Step 1: Write renderer contract tests for initialization state, frame submission before initialization, and cleanup idempotency.**
- [ ] **Step 2: Verify the tests fail because the renderer types do not exist.**
- [ ] **Step 3: Implement the surface lifecycle: create OES texture, create SurfaceTexture, wrap it in Surface, publish the camera input surface, and tear both down deterministically.**
- [ ] **Step 4: Implement camera texture sampling with `GL_TEXTURE_EXTERNAL_OES` and the SurfaceTexture transform matrix.**
- [ ] **Step 5: Implement GPU joint/edge geometry generation from normalized actor landmarks with distinct observed/predicted states.**
- [ ] **Step 6: Implement confidence, uncertainty, and micro-motion visual layers without blocking camera presentation.**
- [ ] **Step 7: Verify renderer contract tests and run source-level shader/resource validation.**
- [x] **Step 8: Commit `feat(app): add GPU MotionField live preview renderer`.**

### Task 5: Integrate MotionField with the existing perception and camera pipeline

**Files:**
- Modify: `app/src/main/java/com/nivukx/nox/NoXViewModel.kt`
- Modify: `app/src/main/java/com/nivukx/nox/MainActivity.kt`
- Create: `app/src/test/java/com/nivukx/nox/MotionFieldPreviewIntegrationContractTest.kt`

**Interfaces:**
- ViewModel publishes `StateFlow<MotionFieldPreviewFrame?>`.
- MainActivity's preview surface submits the latest frame and sends the renderer-owned camera `Surface` back into Camera2CaptureController through the existing `setPreviewSurface` boundary.

- [ ] **Step 1: Write a failing integration contract test requiring preview publication from `PerceptionPipeline.Result.actorTracks`.**
- [ ] **Step 2: Verify it fails because no preview state exists.**
- [ ] **Step 3: Map actor tracks and current reconstructed skeleton quality into `MotionFieldPreviewFrame`.**
- [ ] **Step 4: Publish the state only when the live preview setting is enabled; preserve capture when disabled.**
- [ ] **Step 5: Replace the direct `SurfaceView` camera preview with `MotionFieldPreviewSurface`, keeping fallback to the current direct preview if renderer creation fails.**
- [ ] **Step 6: Feed real sensor timestamps, reconstruction state, actor micro-motion, and GPU telemetry into the renderer.**
- [ ] **Step 7: Run the integration contract tests and static lifecycle audit.**
- [x] **Step 8: Commit `feat(app): integrate live MotionField preview with capture`.**

### Task 6: High-end operator controls and telemetry

**Files:**
- Modify: `core/model/src/main/java/com/nivukx/nox/core/model/UltraMotionModels.kt`
- Modify: `app/src/main/java/com/nivukx/nox/NoXViewModel.kt`
- Modify: `app/src/main/java/com/nivukx/nox/ui/ProfessionalMotionSettings.kt`
- Modify: `app/src/main/java/com/nivukx/nox/ui/UltraMotionStudio.kt`

- [ ] **Step 1: Add failing persistence/validation tests for Live Motion Preview, GPU mode, overlay layers, prediction horizon, and uncertainty visualization.**
- [ ] **Step 2: Implement `MotionCaptureCustomization` fields and validation.**
- [ ] **Step 3: Persist and restore the new controls in NoXViewModel.**
- [ ] **Step 4: Add liquid-glass settings controls and a live telemetry strip for GPU backend, render latency, stale state, actor count, and preview confidence.**
- [ ] **Step 5: Preserve responsive/adaptive layout behavior and existing studio controls.**
- [ ] **Step 6: Run static UI/source-integrity checks.**
- [x] **Step 7: Commit `feat(ui): add MotionField operator controls and telemetry`.**

### Task 7: Documentation, audit, and verification

**Files:**
- Modify: `docs/superpowers/plans/2026-09-24-nox-high-end-motion-capture-plan.md`
- Modify: `docs/superpowers/specs/2026-09-24-nox-motionfield-fusion-preview-design.md`

- [ ] **Step 1: Record implementation decisions and any rulings.**
- [ ] **Step 2: Audit all changed Kotlin files for balanced braces, duplicate declarations, and forbidden hardcoded GPU fallbacks.**
- [ ] **Step 3: Run every feasible pure Kotlin test in the changed test set without starting the Android application build.**
- [ ] **Step 4: Verify main branch head and changed-file inventory.**
- [ ] **Step 5: Commit `docs: record MotionField implementation and verification`.**

---

## Completion Contract

Before declaring this iteration complete:
- All feasible unit/contract tests named above have been executed and their output read.
- No Android application build/run has been started.
- Main branch contains the implementation commits.
- Renderer failure isolation is verified by source-level lifecycle checks.
- Existing multi-actor, face, eye, reconstruction, timing, and calibration code remains present.
- No artificial repetitive source inflation has been introduced.


## Verification Notes — 2026-09-24

- Pure-Kotlin validation of MotionField synchronizer, bounded display prediction, and GPU policy passed.
- Source audit found balanced braces across all changed MotionField/core integration files.
- Android application build/run was intentionally not started.
- Vulkan remains capability-gated and not falsely reported as active; the implemented live compositor is OpenGL ES rendering with a persistent GPU geometry buffer.
- Figma design-system inspection succeeded, but the actual composed Figma write was blocked by the connected Starter-plan MCP tool-call limit, so no claim is made that the screen was written to the file.


### Final implementation notes
- Added bounded display prediction using actor directional velocity vectors. Prediction is applied only to the display copy after sensor-time synchronization; recorded sensor measurements remain authoritative.
- Hardened GLES renderer lifecycle for partial initialization failures and immediate Camera2 surface release during view detachment.
- Renderer telemetry now reports actual GL render latency and a clearly labeled render-budget occupancy estimate rather than pretending to expose driver GPU utilization.
- Persistent dynamic GPU geometry buffering replaces per-frame GL buffer creation for the overlay path.
