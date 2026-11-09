package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.FrontierLandmarkMeasurement
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class BoundedMultiCameraLandmarkSelectorTest {
    private fun observation(landmark: Int, cameraId: String, timestamp: Long = 1_000L) =
        FrontierLandmarkMeasurement(
            landmarkId = landmark,
            cameraId = cameraId,
            point = Vec2(0.25 + landmark * 0.001, 0.40),
            confidence = 0.9,
            intrinsics = CameraIntrinsics(
                fx = 1_000.0,
                fy = 1_000.0,
                cx = 640.0,
                cy = 360.0,
                imageWidth = 1_280,
                imageHeight = 720,
            ),
            cameraPose = Pose(),
            observedAtNanos = timestamp,
        )

    @Test
    fun framesWithinCapacityPreserveFirstSeenOrdering() {
        val cameraB = observation(1, "camera-b")
        val cameraA = observation(2, "camera-a")
        val result = BoundedMultiCameraLandmarkSelector.select(
            items = listOf(cameraB, cameraA),
            timestampNanos = 1_000L,
            maximumItems = 4,
            prioritizedCameraIds = listOf("camera-a", "camera-b"),
        )
        assertEquals(listOf(cameraB, cameraA), result.items)
        assertEquals(2, result.eligibleCountCappedAtOverflow)
    }

    @Test
    fun overloadedCameraMajorInputKeepsEveryCameraRepresented() {
        val items = buildList {
            for (cameraId in listOf("camera-a", "camera-b", "camera-c")) {
                for (landmark in 0 until 10) add(observation(landmark, cameraId))
            }
        }
        val result = BoundedMultiCameraLandmarkSelector.select(
            items = items,
            timestampNanos = 1_000L,
            maximumItems = 6,
            prioritizedCameraIds = listOf("camera-a", "camera-b", "camera-c"),
        )
        assertEquals(6, result.items.size)
        assertEquals(7, result.eligibleCountCappedAtOverflow)
        assertEquals(
            mapOf("camera-a" to 2, "camera-b" to 2, "camera-c" to 2),
            result.items.groupingBy { it.cameraId }.eachCount(),
        )
    }

    @Test
    fun unusedCameraQuotaIsReassignedToAvailableEvidence() {
        val items = buildList {
            add(observation(0, "camera-a"))
            for (landmark in 0 until 8) add(observation(landmark + 1, "camera-b"))
            for (landmark in 0 until 8) add(observation(landmark + 20, "camera-c"))
        }
        val result = BoundedMultiCameraLandmarkSelector.select(
            items = items,
            timestampNanos = 1_000L,
            maximumItems = 6,
            prioritizedCameraIds = listOf("camera-a", "camera-b", "camera-c"),
        )
        assertEquals(6, result.items.size)
        assertEquals(1, result.items.count { it.cameraId == "camera-a" })
        assertEquals(3, result.items.count { it.cameraId == "camera-b" })
        assertEquals(2, result.items.count { it.cameraId == "camera-c" })
    }

    @Test
    fun futureObservationsAreExcludedBeforeCapacityAdmission() {
        val valid = observation(1, "camera-a")
        val future = observation(2, "camera-b", timestamp = 1_001L)
        val result = BoundedMultiCameraLandmarkSelector.select(
            items = listOf(valid, future),
            timestampNanos = 1_000L,
            maximumItems = 1,
            prioritizedCameraIds = listOf("camera-a", "camera-b"),
        )
        assertFalse(result.items.any { it.observedAtNanos > 1_000L })
        assertEquals(listOf(valid), result.items)
    }
}
