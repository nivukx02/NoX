package com.nivukx.nox.core.model

import kotlin.test.Test
import kotlin.test.assertFailsWith

class CalibrationSnapshotContractTest {

    @Test
    fun mismatchedIntrinsicAndExtrinsicCameraIdsAreRejected() {
        val intrinsicId = DeviceId("intrinsic")
        val extrinsicId = DeviceId("extrinsic")

        assertFailsWith<IllegalArgumentException> {
            CalibrationSnapshot(
                intrinsics =
                    mapOf(
                        intrinsicId to CameraIntrinsics(
                            fx = 1000.0,
                            fy = 1000.0,
                            cx = 500.0,
                            cy = 400.0,
                        ),
                    ),
                extrinsics =
                    mapOf(
                        extrinsicId to CameraExtrinsics(
                            pose = Pose(),
                            referenceDevice = extrinsicId,
                        ),
                    ),
                reprojectionErrorPx = 0.4,
                calibratedAtNanos = 100L,
            )
        }
    }

    @Test
    fun focusMetadataCannotReferenceUnknownCamera() {
        val cameraId = DeviceId("camera")

        assertFailsWith<IllegalArgumentException> {
            CalibrationSnapshot(
                intrinsics =
                    mapOf(
                        cameraId to CameraIntrinsics(
                            fx = 1000.0,
                            fy = 1000.0,
                            cx = 500.0,
                            cy = 400.0,
                        ),
                    ),
                extrinsics =
                    mapOf(
                        cameraId to CameraExtrinsics(
                            pose = Pose(),
                            referenceDevice = cameraId,
                        ),
                    ),
                reprojectionErrorPx = 0.4,
                calibratedAtNanos = 100L,
                calibrationFocusDistanceDiopters =
                    mapOf(
                        DeviceId("unknown") to 2.0,
                    ),
            )
        }
    }
}
