package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.DepthSource
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.nativeengine.DepthSampleConvention
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class NativeDepthFrameReconstructionSinkTest {
    private fun frame(
        timestamp: Long = 10_000L,
        width: Int = 4,
        height: Int = 3,
    ) = DepthFrame(
        timestampNanos = timestamp,
        source = DepthSource.DEPTH16,
        width = width,
        height = height,
        depthMeters = FloatArray(width * height) { 1.0f },
        confidence = FloatArray(width * height) { 1.0f },
    )

    private class RecordingBackend : NativeDepthFrameReconstructionSink.Backend {
        val integratedTimestamps = mutableListOf<Pair<String, Long>>()
        var failNextIntegration = false
        var resetCount = 0
        var closed = false

        override fun integrateDepthFrame(
            depthFrame: DepthFrame,
            intrinsics: CameraIntrinsics,
            pose: Pose,
            stride: Int,
            confidence: Double,
            depthConvention: DepthSampleConvention,
        ) {
            if (failNextIntegration) {
                failNextIntegration = false
                throw IllegalStateException("injected native integration failure")
            }
            integratedTimestamps += "camera" to depthFrame.timestampNanos
        }

        override fun extract(
            timestampNanos: Long,
            isoLevel: Float,
        ): NativeSparseTsdfReconstructionBridge.ReconstructionMeshWithQuality? = null

        override fun reset() {
            resetCount += 1
            integratedTimestamps.clear()
        }

        override fun close() {
            closed = true
        }
    }

    private fun calibration(
        timestamp: Long = 10_000L,
        width: Int = 4,
        height: Int = 3,
        rectified: Boolean = true,
        weight: Double = 0.8,
        pose: Pose = Pose(),
    ) =
        NativeDepthFrameReconstructionSink.CameraCalibration(
            resolvedTimestampNanos = timestamp,
            intrinsics = CameraIntrinsics(
                fx = 500.0,
                fy = 500.0,
                cx = width / 2.0,
                cy = height / 2.0,
                imageWidth = width,
                imageHeight = height,
            ),
            cameraToWorld = pose,
            observationWeight = weight,
            depthConvention = DepthSampleConvention.EUCLIDEAN_RAY,
            depthImageRectified = rectified,
        )

    @Test
    fun calibrationMustMatchFrameTimestampAndOutputDimensions() {
        val current = frame()
        assertTrue(calibration().isCompatibleWith(current))
        assertFalse(calibration(timestamp = current.timestampNanos + 1L).isCompatibleWith(current))
        assertFalse(calibration(width = current.width + 1).isCompatibleWith(current))
        assertFalse(calibration(rectified = false).isCompatibleWith(current))
    }

    @Test
    fun rejectsNonFinitePoseAndObservationWeight() {
        val invalidPose = Pose(
            position = Vec3(Double.NaN, 0.0, 0.0),
            rotation = Quat.IDENTITY,
        )
        assertFalse(calibration(pose = invalidPose).isCompatibleWith(frame()))
        assertFalse(calibration(weight = Double.NaN).isCompatibleWith(frame()))
        assertFalse(calibration(weight = 0.0).isCompatibleWith(frame()))
    }

    @Test
    fun rejectsDegenerateQuaternionAndNonFiniteIntrinsics() {
        val degenerate = Pose(
            position = Vec3(0.0, 0.0, 0.0),
            rotation = Quat(0.0, 0.0, 0.0, 0.0),
        )
        assertFalse(calibration(pose = degenerate).isCompatibleWith(frame()))
        val badIntrinsics = calibration().copy(
            intrinsics = calibration().intrinsics.copy(fx = Double.POSITIVE_INFINITY),
        )
        assertFalse(badIntrinsics.isCompatibleWith(frame()))
    }

    @Test
    fun observationWeightMustBePositiveToClaimAnIntegratedFrame() {
        assertFalse(calibration(weight = 0.0).isCompatibleWith(frame()))
        assertTrue(calibration(weight = 0.01).isCompatibleWith(frame()))
        assertTrue(calibration(weight = 1.0).isCompatibleWith(frame()))
        assertFalse(calibration(weight = 1.01).isCompatibleWith(frame()))
    }


    @Test
    fun cameraTimestampWatermarksRejectCapacityWithoutEvictingOrderingHistory() {
        val watermarks = CameraDepthTimestampWatermarks(maximumTrackedCameras = 2)

        assertTrue(
            watermarks.admission("cam-a", 100L) ==
                CameraDepthTimestampWatermarks.Admission.ACCEPT,
        )
        watermarks.markIntegrated("cam-a", 100L)
        assertTrue(
            watermarks.admission("cam-b", 200L) ==
                CameraDepthTimestampWatermarks.Admission.ACCEPT,
        )
        watermarks.markIntegrated("cam-b", 200L)

        assertTrue(
            watermarks.admission("cam-c", 300L) ==
                CameraDepthTimestampWatermarks.Admission.CAPACITY_EXCEEDED,
        )
        assertTrue(
            watermarks.admission("cam-a", 100L) ==
                CameraDepthTimestampWatermarks.Admission.STALE,
        )
        assertTrue(
            watermarks.admission("cam-a", 101L) ==
                CameraDepthTimestampWatermarks.Admission.ACCEPT,
        )
        assertTrue(watermarks.size == 2)
    }

    @Test
    fun timestampWatermarkCapacityIsReleasedOnlyOnSessionReset() {
        val watermarks = CameraDepthTimestampWatermarks(maximumTrackedCameras = 1)
        watermarks.markIntegrated("cam-a", 100L)
        assertTrue(
            watermarks.admission("cam-b", 200L) ==
                CameraDepthTimestampWatermarks.Admission.CAPACITY_EXCEEDED,
        )

        watermarks.clear()

        assertTrue(watermarks.size == 0)
        assertTrue(
            watermarks.admission("cam-b", 200L) ==
                CameraDepthTimestampWatermarks.Admission.ACCEPT,
        )
    }


    @Test
    fun submitIntegratesOnlyCalibratedMonotonicFramesAndEnforcesCameraCapacity() {
        val backend = RecordingBackend()
        val sink = NativeDepthFrameReconstructionSink(
            calibrationResolver = NativeDepthFrameReconstructionSink.CalibrationResolver { _, timestamp ->
                calibration(timestamp = timestamp)
            },
            config = NativeDepthFrameReconstructionSink.Config(
                maximumTrackedCameras = 2,
                sampleStride = 2,
            ),
            backendOverride = backend,
        )

        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATED,
            sink.submit("cam-a", frame(timestamp = 100L)),
        )
        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.STALE_FRAME,
            sink.submit("cam-a", frame(timestamp = 100L)),
        )
        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATED,
            sink.submit("cam-b", frame(timestamp = 200L)),
        )
        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.CAMERA_CAPACITY_EXCEEDED,
            sink.submit("cam-c", frame(timestamp = 300L)),
        )
        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.STALE_FRAME,
            sink.submit("cam-a", frame(timestamp = 99L)),
        )

        assertEquals(listOf(100L, 200L), backend.integratedTimestamps.map { it.second })
        assertEquals(2, sink.diagnostics().trackedCameras)
        assertEquals(2L, sink.diagnostics().staleFrames)
        assertEquals(1L, sink.diagnostics().cameraCapacityRejections)
        assertEquals(
            "maximum-tracked-depth-cameras-exceeded",
            sink.diagnostics().lastFailure,
        )
        sink.close()
        assertEquals(true, backend.closed)
    }

    @Test
    fun failedNativeIntegrationDoesNotConsumeTheFrameTimestampWatermark() {
        val backend = RecordingBackend().apply { failNextIntegration = true }
        val sink = NativeDepthFrameReconstructionSink(
            calibrationResolver = NativeDepthFrameReconstructionSink.CalibrationResolver { _, timestamp ->
                calibration(timestamp = timestamp)
            },
            backendOverride = backend,
        )
        val frame = frame(timestamp = 500L)

        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATION_FAILED,
            sink.submit("cam-a", frame),
        )
        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATED,
            sink.submit("cam-a", frame),
        )
        assertEquals(1, backend.integratedTimestamps.size)
        assertEquals(1L, sink.diagnostics().integrationFailures)
        sink.close()
    }

    @Test
    fun resetClearsWatermarksAndDelegatesToTheSharedBackend() {
        val backend = RecordingBackend()
        val sink = NativeDepthFrameReconstructionSink(
            calibrationResolver = NativeDepthFrameReconstructionSink.CalibrationResolver { _, timestamp ->
                calibration(timestamp = timestamp)
            },
            backendOverride = backend,
        )
        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATED,
            sink.submit("cam-a", frame(timestamp = 800L)),
        )

        sink.reset()

        assertEquals(1, backend.resetCount)
        assertEquals(0, sink.diagnostics().trackedCameras)
        assertEquals(
            NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATED,
            sink.submit("cam-a", frame(timestamp = 800L)),
        )
        sink.close()
    }

}
