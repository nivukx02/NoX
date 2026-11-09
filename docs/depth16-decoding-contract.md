# Android DEPTH16 Decoding Contract

The Camera2 DEPTH16 sample is a packed 16-bit value, not a 16-bit integer distance. Android defines the range in bits 0 through 12 and confidence in bits 13 through 15. Range is expressed in millimetres when produced by a camera. Confidence code 0 denotes 100% confidence; codes 1 through 7 encode 0 through 6/7 respectively. See the [Android ImageFormat reference](https://developer.android.com/reference/android/graphics/ImageFormat.html#DEPTH16).

NoX now separates the range and confidence signals. A zero range stays an invalid zero-depth sample, while valid samples retain the per-pixel confidence array for reconstruction weighting.

Before allocating arrays, the decoder validates positive dimensions, a 4,194,304-pixel safety ceiling, row/pixel strides, long-arithmetic byte offsets, and the complete last-pixel boundary. Truncated or oversized camera buffers are rejected rather than partially integrated. This allocation ceiling is a safety guard, not a negotiated output-resolution target.

This change does not claim sub-millimetre practical accuracy; sensor calibration, device-specific depth error, exposure/sync timing and physical reference measurements remain authoritative.


## Per-pixel confidence in reconstruction

The DEPTH16 confidence map follows depth through both normal and adaptive native TSDF integration. Confidence scales each sample's observation weight; invalid or zero-confidence samples are rejected. The legacy scalar frame confidence remains multiplicative, and callers without a per-pixel map retain the prior scalar-only behavior. JNI checks array lengths and releases both pinned arrays on success and exception paths.

## DepthFrame allocation and telemetry guards

`DepthFrame` computes expected sample count in 64-bit arithmetic and rejects counts larger than JVM array capacity before comparing depth/confidence buffers. Descriptor telemetry clamps finite confidence into `[0, 1]` and treats malformed confidence as zero; the real-time constructor does not scan every pixel, avoiding an extra full-frame pass.

The public Kotlin TSDF API and both JNI entry points compute `width × height` using checked arithmetic before comparing array lengths, preventing dimension multiplication overflow on 32-bit devices.


## Live DEPTH16 stream ownership

`Camera2CaptureController` exposes its bounded depth channel independently of persistent sensor sidecars. `MultiCameraCaptureFleetCoordinator` accepts an optional per-camera `depthFrameSinkFactory`; on READY it owns one collector for that entry generation, rejects stale frames using entry identity/generation/state checks, isolates sink exceptions, and cancels collectors on camera removal, fleet reset, or fleet close. The default sink factory is null, so existing callers retain their prior behavior until a reconstruction consumer is connected.


## Calibrated native depth ingestion

The live fleet can be wired to `NativeDepthFrameReconstructionSink` through `depthFrameSinkFactory`, for example `depthFrameSinkFactory = { cameraId -> { frame -> depthSink.submit(cameraId, frame) } }`. The sink serializes integration and mesh extraction against one shared native TSDF volume, requires a timestamp-specific calibration result, rejects mismatched dimensions or unresolved poses, requires the depth image to be explicitly marked rectified for the pinhole intrinsics, and rejects out-of-order frames per camera. No identity pose or guessed intrinsics are substituted when calibration is absent. Integrating frames is synchronous and bounded by the upstream recent-frame channel; telemetry exposes accepted, stale, uncalibrated, invalid and failed counts.


## App session depth input

`FrontierCaptureSessionCoordinator.Input` accepts up to 128 timestamped `DepthFrameEvidence` records, at most one per camera per session step. The session returns a submission outcome for every frame; absent sink, absent or invalid calibration, closed sink, and integration failure are explicit warnings and mark the session `RECONSTRUCTION_LIMITED`. Stale frames are surfaced separately. The native sink is injectable and optional to avoid loading the native TSDF library on devices without a calibrated depth stream.

Depth calibration observation weights must be finite and strictly positive (up to 1.0). A zero weight is rejected rather than counted as an integrated frame when it contributes no TSDF evidence.


The native depth sink exposes a backend interface for deterministic integration tests and platform-specific implementations. Its default backend delegates to the existing native bridge, lazily initialized only when the sink actually integrates, extracts, or resets a native volume. Tests can therefore exercise timestamp admission, capacity pressure, retry after integration failure, and cleanup without loading JNI.


## Live stream negotiation

When the camera fleet's per-camera depth sink factory returns a consumer, it now resolves and reserves that consumer before opening the camera and passes an explicit live-depth flag into Camera2 session negotiation. That flag requests DEPTH16 even when the capture profile disables persistent depth sidecars. Disk writing remains independently gated by sidecar policy. Null or failing sink factories leave the live-depth output disabled; sink callbacks are not invoked under the fleet lock.


The live Camera2 decoder now also enforces the documented DEPTH16 plane layout: even width and height, a two-byte packed pixel stride, and row-byte alignment matching a 16-pixel horizontal stride. Unsupported/truncated plane layouts are rejected instead of partially decoded. This follows the Android [`ImageFormat.DEPTH16`](https://developer.android.com/reference/android/graphics/ImageFormat.html#DEPTH16) contract.


## App capture integration

The app's `NoXViewModel.beginRecording()` previously scheduled two Camera2 opens for the same capture generation. The second open could invalidate the first and was missing the persistent-sidecar policy. The path now computes live-depth enablement from the presence of local-camera intrinsics and extrinsics, installs frame collectors, and performs exactly one final camera open carrying the resolved sidecar and live-depth settings. If Camera2 negotiates away the requested depth output, the app records an explicit runtime reason instead of silently treating depth fusion as active.


## Native observation-confidence bound

Frame-level DEPTH16 confidence is combined with the adaptive TSDF scale through `NativeDepthConfidencePolicy`. The policy clamps the product to `[0, 1]` and converts non-finite values to zero before invoking the strict Kotlin/JNI API, while individual sample confidence remains an independent native weight.


## Kotlin sparse TSDF sample confidence

The Kotlin sparse TSDF path now receives the same per-pixel DEPTH16 confidence map as the native bridge. Its weighted fusion rejects zero and non-finite samples, multiplies each accepted sample's evidence weight by its bounded confidence, and validates dimensions with checked 64-bit arithmetic. Callers without a confidence map retain the prior behavior.
