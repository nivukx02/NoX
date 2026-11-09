# Multi-Camera Capture Lifecycle and Callback Fencing

## Failure mode

Camera2 open and configuration callbacks are asynchronous. A completion can arrive after the operator closes a camera, closes the fleet, or removes and re-admits the same camera ID. Without an ownership fence, the old callback may mutate a stale entry or report a status that belongs to a superseded request.

## Lifecycle contract

- Every open request reserves a monotonically increasing generation on the current Entry.
- The reservation rechecks state inside the same critical section that transitions to OPENING, so concurrent callers cannot both dispatch opens for one ADMITTED/FAILED entry.
- Before dispatching Camera2 work, the coordinator confirms that the fleet is open and that the exact Entry is still the current map value for that camera ID.
- Success, failure, and synchronous dispatch-error paths verify fleet state, Entry identity, generation, and OPENING state under the fleet lock.
- A stale callback neither mutates current state nor invokes the operator status callback.
- Valid status snapshots are created from the checked Entry while the lock is held.
- HAL close work remains outside the fleet lock to avoid holding a global monitor during potentially slow device shutdown.

## Platform constraints

The fleet is a logical orchestration layer, not a guarantee that the phone can open all admitted cameras concurrently. Use the live capabilities reported by Android rather than the configured fleet maximum.

- [Android CameraDevice API](https://developer.android.com/reference/android/hardware/camera2/CameraDevice) describes asynchronous Camera2 device lifecycle callbacks and explicit closing.
- [Android CameraManager API](https://developer.android.com/reference/android/hardware/camera2/CameraManager) provides getConcurrentCameraIds() and isConcurrentSessionConfigurationSupported(...) for device-specific concurrency support.
- [Android multiple camera streams guide](https://developer.android.com/media/camera/camera2/multiple-camera-streams-simultaneously) explains that supported stream combinations and attainable performance vary with hardware capabilities and configuration.

## Verification boundary

The added source-integrity tests preserve the callback-fencing invariants. They do not replace instrumented tests using delayed callbacks or validation against physical Camera2 hardware. The APK build and unit tests are not run as part of this change.
