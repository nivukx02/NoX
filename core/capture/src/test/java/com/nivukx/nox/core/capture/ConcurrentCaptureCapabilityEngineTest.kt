package com.nivukx.nox.core.capture

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConcurrentCaptureCapabilityEngineTest {
    private val engine = ConcurrentCaptureCapabilityEngine()

    private fun camera(
        id: String,
        diversity: Double,
        thermal: Double = 0.18,
        memory: Double = 0.18,
        sync: Double = 0.98,
        calibration: Double = 0.98,
        skew: Double = 0.8,
    ) = ConcurrentCaptureCapabilityEngine.CameraEndpoint(
        cameraId = id,
        maxWidth = 1920,
        maxHeight = 1080,
        maxFps = 120,
        supportsConcurrent = true,
        supportsRaw = true,
        rawWidth = 4000,
        rawHeight = 3000,
        supportsDepth = true,
        depthWidth = 1280,
        depthHeight = 720,
        thermalLoad = thermal,
        memoryPressure = memory,
        synchronizationConfidence = sync,
        calibrationConfidence = calibration,
        viewDiversity = diversity,
        pairSkewMillis = skew,
    )

    @Test
    fun admitsWellBoundedRig() {
        val cameras = listOf(
            camera("0", 0.10),
            camera("1", 0.35),
            camera("2", 0.65),
            camera("3", 0.90),
            camera("4", 0.50),
        )
        val plan = engine.plan(
            cameras = cameras,
            concurrentSets = listOf(
                ConcurrentCaptureCapabilityEngine.ConcurrentSet(
                    cameraIds = cameras.map { it.cameraId }.toSet(),
                ),
            ),
            requestedCameraCount = 4,
        )
        assertTrue(plan.selectedCameraIds.size == 4)
        assertTrue(plan.admissible)
        assertTrue(plan.totalMemoryMb > 0.0)
        assertTrue(plan.totalThroughputMbPerSecond > 0.0)
    }

    @Test
    fun blocksWhenConcurrentSetIsNotDeclared() {
        val cameras = listOf(camera("0", 0.2), camera("1", 0.8))
        val plan = engine.plan(
            cameras = cameras,
            concurrentSets = emptyList(),
            requestedCameraCount = 2,
        )
        assertFalse(plan.admissible)
        assertTrue(plan.blockers.contains("NO_CONCURRENT_CAMERA_SET"))
    }

    @Test
    fun blocksThermallyUnsafeRig() {
        val cameras = listOf(
            camera("0", 0.1, thermal = 0.95),
            camera("1", 0.4, thermal = 0.95),
            camera("2", 0.8, thermal = 0.95),
            camera("3", 0.9, thermal = 0.95),
        )
        val plan = engine.plan(
            cameras = cameras,
            concurrentSets = listOf(
                ConcurrentCaptureCapabilityEngine.ConcurrentSet(
                    cameraIds = cameras.map { it.cameraId }.toSet(),
                ),
            ),
            requestedCameraCount = 4,
        )
        assertFalse(plan.admissible)
        assertTrue(plan.blockers.any { it.contains("THERMAL") })
    }

    @Test
    fun pairEvidenceRejectsExcessiveSkew() {
        val result = engine.evaluatePair(
            camera("0", 0.1, skew = 0.2),
            camera("1", 0.9, skew = 0.2),
        )
        assertTrue(result.skew < 0.50)
        assertTrue(result.reasons.contains("TEMPORAL_SKEW"))
    }

    @Test
    fun streamThroughputScalesWithFrameRate() {
        val thirty = engine.estimateStreamCost(
            ConcurrentCaptureCapabilityEngine.StreamOption(
                ConcurrentCaptureCapabilityEngine.StreamKind.ANALYSIS_YUV,
                1920,
                1080,
                30,
            ),
        ).second
        val oneTwenty = engine.estimateStreamCost(
            ConcurrentCaptureCapabilityEngine.StreamOption(
                ConcurrentCaptureCapabilityEngine.StreamKind.ANALYSIS_YUV,
                1920,
                1080,
                120,
            ),
        ).second
        assertTrue(oneTwenty > thirty)
    }

    @Test
    fun usesLargestDeclaredConcurrentSetWhenPlanningFromFeatureMatrices() {
        val matrices = (0 until 4).map { index ->
            com.nivukx.nox.core.model.CameraFeatureMatrix(
                cameraId = index.toString(),
                supportsRaw = false,
                rawWidth = 0,
                rawHeight = 0,
                supportsDepth16 = false,
                depthWidth = 0,
                depthHeight = 0,
                supportsManualSensor = true,
                supportsOis = true,
                supportsPdaf = false,
                supportsConcurrentCamera = true,
                maxAnalysisWidth = 1920,
                maxAnalysisHeight = 1080,
                maxFps = 60,
                recommendedProfile =
                    com.nivukx.nox.core.model.CameraCaptureProfile(),
                concurrentCameraSets = listOf(
                    setOf("0", "1", "2", "3"),
                ),
            )
        }

        val plan =
            engine.planFromFeatureMatrices(
                matrices = matrices,
                synchronizationConfidence =
                    matrices.associate { it.cameraId to 0.98 },
                calibrationConfidence =
                    matrices.associate { it.cameraId to 0.98 },
                thermalLoad =
                    matrices.associate { it.cameraId to 0.10 },
                memoryPressure =
                    matrices.associate { it.cameraId to 0.10 },
                viewDiversity =
                    matrices.associate { it.cameraId to 0.90 },
                pairSkewMillis =
                    matrices.associate { it.cameraId to 0.5 },
                requestedCameraCount = 4,
            )

        assertTrue(plan.admissible)
        assertTrue(plan.selectedCameraIds.size == 4)
        assertTrue(plan.topologyScore > 0.50)
    }

}
