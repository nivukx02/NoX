package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ReconstructionUncertaintyPropagationEngineTest {

    @Test
    fun stableMultiViewSkeletonIsAccepted() {
        val engine = ReconstructionUncertaintyPropagationEngine()
        val result = engine.evaluate(input(
            timestampNanos = 1_000_000_000L,
            uncertainty = 0.012,
            reprojection = 0.96,
            temporal = 0.94,
            observability = 0.95,
            drift = 0.97,
            sync = 0.96,
            timing = 0.96,
        ))

        assertTrue(result.accepted)
        assertEquals(
            ReconstructionUncertaintyPropagationEngine.State.STABLE,
            result.state,
        )
        assertTrue(result.trustedJointRatio > 0.80)
        assertTrue(result.p95UncertaintyMeters < 0.085)
    }

    @Test
    fun degradedGeometryInflatesUncertainty() {
        val engine = ReconstructionUncertaintyPropagationEngine()
        val result = engine.evaluate(input(
            timestampNanos = 2_000_000_000L,
            uncertainty = 0.035,
            reprojection = 0.50,
            temporal = 0.52,
            observability = 0.48,
            drift = 0.55,
            sync = 0.56,
            timing = 0.58,
            cameraMotion = 0.42,
            occlusion = 0.35,
        ))

        assertTrue(result.inflationFactor > 1.8)
        assertTrue(result.score < 0.70)
        assertTrue(result.warnings.isNotEmpty())
    }

    @Test
    fun persistentHardUncertaintyBlocks() {
        val engine = ReconstructionUncertaintyPropagationEngine(
            ReconstructionUncertaintyPropagationEngine.Config(
                persistentWarningFrames = 2,
                persistentBlockedFrames = 3,
            ),
        )

        var result: ReconstructionUncertaintyPropagationEngine.Result? = null
        repeat(5) { index ->
            result = engine.evaluate(input(
                timestampNanos = 3_000_000_000L +
                    index * 16_000_000L,
                uncertainty = 0.20,
                reprojection = 0.35,
                temporal = 0.35,
                observability = 0.35,
                drift = 0.35,
                sync = 0.40,
                timing = 0.40,
                cameraMotion = 0.50,
                occlusion = 0.45,
            ))
        }

        assertEquals(
            ReconstructionUncertaintyPropagationEngine.State.BLOCKED,
            result?.state,
        )
        assertTrue(result?.accepted == false)
    }

    private fun input(
        timestampNanos: Long,
        uncertainty: Double,
        reprojection: Double,
        temporal: Double,
        observability: Double,
        drift: Double,
        sync: Double,
        timing: Double,
        cameraMotion: Double = 0.05,
        occlusion: Double = 0.04,
    ) = ReconstructionUncertaintyPropagationEngine.Input(
        timestampNanos = timestampNanos,
        skeleton = skeleton(uncertainty),
        reprojectionQuality = reprojection,
        temporalQuality = temporal,
        geometricObservability = observability,
        rigGeometryDriftScore = drift,
        synchronizationConfidence = sync,
        captureTimingConfidence = timing,
        cameraMotionRisk = cameraMotion,
        occlusionRisk = occlusion,
    )

    private fun skeleton(
        uncertainty: Double,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = 1L,
            subjectType = SubjectType.HUMAN,
            trackId = 7L,
            joints = (0 until 24).map { id ->
                ReconstructionJoint(
                    landmarkId = id,
                    position = Vec3(
                        id * 0.01,
                        id * 0.006,
                        1.2 + id * 0.002,
                    ),
                    velocity = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.92,
                    reprojectionErrorPx = 1.8,
                    observationCount = 4,
                    positionUncertaintyMeters = uncertainty,
                )
            },
            globalConfidence = 0.94,
        )
}
