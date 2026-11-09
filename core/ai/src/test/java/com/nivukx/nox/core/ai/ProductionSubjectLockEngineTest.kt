package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ProductionSubjectLockEngineTest {

    @Test
    fun acquiresAndHoldsHumanLockAcrossSmallMotion() {
        val engine = ProductionSubjectLockEngine()

        val first = engine.update(
            timestampNanos = 0L,
            observations = listOf(
                human(
                    id = 7L,
                    x = 0.42,
                    y = 0.40,
                    confidence = 0.92,
                ),
            ),
        )

        val second = engine.update(
            timestampNanos = 120_000_000L,
            observations = listOf(
                human(
                    id = 7L,
                    x = 0.43,
                    y = 0.40,
                    confidence = 0.94,
                ),
            ),
        )

        val third = engine.update(
            timestampNanos = 260_000_000L,
            observations = listOf(
                human(
                    id = 7L,
                    x = 0.45,
                    y = 0.41,
                    confidence = 0.96,
                ),
            ),
        )

        assertTrue(first.globalConfidence > 0.0)
        assertTrue(second.tracks.isNotEmpty())
        assertNotNull(third.focus)
        assertTrue(
            third.lockedTrackId == 1L ||
                third.lockState == ProductionSubjectLockEngine.LockState.CANDIDATE ||
                third.lockState == ProductionSubjectLockEngine.LockState.LOCKED,
        )
        assertTrue(third.focus!!.confidence > 0.30)
    }

    @Test
    fun preservesIdentityWhenDetectorIdChanges() {
        val engine = ProductionSubjectLockEngine()

        engine.update(
            timestampNanos = 0L,
            observations = listOf(
                human(id = 10L, x = 0.50, y = 0.50, confidence = 0.90),
            ),
        )

        val result = engine.update(
            timestampNanos = 50_000_000L,
            observations = listOf(
                human(id = 99L, x = 0.51, y = 0.50, confidence = 0.91),
            ),
        )

        assertEquals(1, result.tracks.size)
        assertEquals(0.51, result.tracks.single().predictedX, 0.08)
        assertTrue(result.tracks.single().confidence > 0.70)
    }

    @Test
    fun humanEyesTakeFocusPriorityWhenEvidenceIsStrong() {
        val engine = ProductionSubjectLockEngine()

        val result = engine.update(
            timestampNanos = 100_000_000L,
            observations = listOf(
                ProductionSubjectLockEngine.Observation(
                    detectorId = 42L,
                    kind = ProductionSubjectLockEngine.SubjectKind.HUMAN,
                    left = 0.20,
                    top = 0.20,
                    right = 0.80,
                    bottom = 0.90,
                    confidence = 0.92,
                    saliency = 0.85,
                    leftEye = ProductionSubjectLockEngine.EyeObservation(
                        x = 0.41,
                        y = 0.46,
                        confidence = 0.98,
                        openness = 0.92,
                    ),
                    rightEye = ProductionSubjectLockEngine.EyeObservation(
                        x = 0.59,
                        y = 0.46,
                        confidence = 0.97,
                        openness = 0.93,
                    ),
                    gaze = ProductionSubjectLockEngine.GazeObservation(
                        x = 0.50,
                        y = 0.45,
                        confidence = 0.96,
                    ),
                ),
            ),
        )

        assertEquals(
            ProductionSubjectLockEngine.FocusSource.GAZE_FUSION,
            result.focus?.source,
        )
        assertTrue((result.focus?.confidence ?: 0.0) > 0.70)
    }

    @Test
    fun cameraMotionDoesNotDestroyTrackState() {
        val engine = ProductionSubjectLockEngine()

        engine.update(
            timestampNanos = 0L,
            observations = listOf(
                human(id = 1L, x = 0.45, y = 0.45, confidence = 0.90),
            ),
        )

        val result = engine.update(
            timestampNanos = 33_000_000L,
            observations = listOf(
                human(id = 1L, x = 0.48, y = 0.46, confidence = 0.91),
            ),
            cameraMotion = ProductionSubjectLockEngine.CameraMotion(
                vx = 0.50,
                vy = 0.35,
                confidence = 0.88,
            ),
        )

        assertEquals(1, result.tracks.size)
        assertTrue(result.cameraMotionRisk > 0.0)
        assertTrue(result.globalConfidence > 0.20)
    }

    private fun human(
        id: Long,
        x: Double,
        y: Double,
        confidence: Double,
    ): ProductionSubjectLockEngine.Observation =
        ProductionSubjectLockEngine.Observation(
            detectorId = id,
            kind = ProductionSubjectLockEngine.SubjectKind.HUMAN,
            left = (x - 0.08).coerceIn(0.0, 1.0),
            top = (y - 0.10).coerceIn(0.0, 1.0),
            right = (x + 0.08).coerceIn(0.0, 1.0),
            bottom = (y + 0.10).coerceIn(0.0, 1.0),
            confidence = confidence,
            saliency = confidence,
        )
    @Test
    fun globalLockAssociationScalesPastTwentyFourCandidates() {
        val engine =
            ProductionSubjectLockEngine()

        val first =
            engine.update(
                timestampNanos = 1_000_000_000L,
                observations =
                    (0 until 40).map { index ->
                        human(
                            id = 1000L + index,
                            x = 0.05 + (index % 8) * 0.11,
                            y = 0.12 + (index / 8) * 0.17,
                            confidence = 0.92,
                        )
                    },
            )

        val second =
            engine.update(
                timestampNanos = 1_033_333_333L,
                observations =
                    (0 until 40).map { index ->
                        human(
                            id = 5000L + index,
                            x = 0.051 + (index % 8) * 0.11,
                            y = 0.121 + (index / 8) * 0.17,
                            confidence = 0.93,
                        )
                    },
            )

        assertEquals(
            40,
            first.tracks.size,
        )
        assertTrue(
            second.tracks.size >= 32,
        )
    }

}
