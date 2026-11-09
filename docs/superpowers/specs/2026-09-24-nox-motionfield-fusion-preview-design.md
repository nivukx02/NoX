# NoX MotionField Fusion Preview Design

## Goal
Add a production-grade live motion-capture preview that renders the authoritative NoX camera frame together with synchronized actor motion evidence, using a GPU-resident render path and explicit capability/quality controls.

## Scope
- Live camera texture presentation.
- Stable multi-actor body overlay from the existing multi-actor perception pipeline.
- Timestamp-aware preview synchronization using measured sensor timestamps.
- Confidence, prediction, uncertainty, and micro-motion visualization states.
- GPU backend selection with a GLES 3.1+ implementation and a capability-driven Vulkan-ready abstraction.
- Runtime maximum-GPU policy without bypassing thermal, memory, or capture-integrity protections.
- User setting to enable/disable live reconstruction preview.
- Liquid-glass operator controls for preview layers and GPU policy.

## Non-goals
- Replacing Camera2 capture.
- Pretending predicted positions are measured observations.
- Claiming nanometer spatial accuracy from commodity phone cameras.
- Requiring Vulkan on devices that do not advertise compatible support.
- Starting the Android app build or device run during this implementation pass.

## Architecture
The camera remains owned by Camera2CaptureController. The new MotionFieldPreviewView owns a GPU presentation surface and an internal camera-input SurfaceTexture. Camera2 writes preview frames into the SurfaceTexture-backed Surface; the renderer samples the external camera texture and composites motion geometry in the same GPU pass before presenting to the view.

MotionFieldFrameSynchronizer receives camera sensor timestamp plus the latest actor/reconstruction observations. It selects the closest authoritative motion state, computes bounded interpolation/extrapolation metadata, and labels predicted content separately from measured content.

MotionFieldGpuBackend is an abstraction. Backend selection is capability-driven: Vulkan is selected only when its Android external-memory/interoperability requirements are actually available and the implementation is present; otherwise GLES 3.1 compute/render is selected. CPU is retained as a diagnostics fallback, never as the normal live-render path on GPU-capable devices.

## Data contract
Live preview state contains timestampNanos, actor IDs, normalized joint positions, joint confidence, predicted flag, missed frames, micro-motion score, optional 3D skeleton joints, reconstruction confidence, uncertainty, GPU backend, render latency, and warnings.

## Rendering
The GLES path uses an external OES camera texture, a transform matrix from SurfaceTexture, a camera quad, and GPU line/point geometry for actor joints. Actor IDs and status badges are rendered through Compose around the GPU surface rather than baked into the camera pixels. Predicted joints use a separate geometry state so the UI cannot confuse prediction with observation.

## Synchronization
Sensor timestamps are the authoritative time base. Preview presentation uses the display cadence as the target and only permits a bounded extrapolation horizon. A stale motion frame is never silently treated as current.

## GPU policy
MAXIMUM selects the highest validated render path and highest detail budget compatible with capture-integrity and thermal floors. HIGH and BALANCED reduce overlay detail and update frequency but do not disable core capture. Thermal or memory protection can lower presentation detail while preserving recorded motion data.

## Failure handling
- Camera SurfaceTexture creation failure: fall back to the existing direct SurfaceView preview and publish a diagnostic warning.
- GPU shader compile/link failure: disable overlay rendering but keep camera preview alive.
- Stale motion state: render camera-only plus stale-state indicator.
- Unsupported backend: select next validated backend.
- Renderer teardown: release SurfaceTexture, Surface, GL objects, buffers, and executors deterministically.

## Testing
Pure unit tests cover timestamp selection, prediction-window clamping, backend capability decisions, detail-budget selection, and preview-state validation. Android renderer lifecycle remains a device/instrumentation concern and is guarded by strict state invariants in the production class.

## Success criteria
- Enabling Live Motion Preview produces a real camera preview with real tracked actor geometry.
- Disabling it returns to camera-only presentation.
- Multiple actors preserve their stable identities in the overlay.
- Preview timing uses sensor timestamps rather than callback arrival time.
- GPU backend/latency state is visible to the operator.
- Renderer failure does not terminate capture.
- Existing reconstruction, face, eye, calibration, synchronization, and capture paths remain intact.