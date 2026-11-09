package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibratedMotionUncertaintyEngineTest {

    private fun skeleton(
        timestampNanos: Long,
        x: Double,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 23,
                    position = Vec3(x, 1.0, 0.0),
                    velocity = Vec3(0.30, 0.0, 0.0),
                    confidence = 0.94,
                    reprojectionErrorPx = 1.1,
                    observationCount = 4,
                    positionUncertaintyMeters = 0.012,
                ),
            ),
            globalConfidence = 0.94,
        )

    @Test
    fun warmupDoesNotInventOodRisk() {
        val engine = CalibratedMotionUncertaintyEngine()

        val result =
            engine.evaluate(
                skeleton = skeleton(
                    timestampNanos = 1_000_000_000L,
                    x = 0.0,
                ),
            )

        assertEquals(1, result.envelopeJointCount)
        assertTrue(result.outOfDistributionRisk < 0.50)
        assertTrue(result.calibrationProgress < 1.0)
    }


    @Test
    fun predictionOnlyJointDoesNotBecomeCalibrationEvidence() {
        val engine = CalibratedMotionUncertaintyEngine()

        val predictionOnly =
            SkeletonFrame(
                timestampNanos = 5_000_000_000L,
                subjectType = SubjectType.HUMAN,
                trackId = 2L,
                joints = listOf(
                    ReconstructionJoint(
                        landmarkId = 24,
                        position = Vec3(0.0, 1.0, 0.0),
                        velocity = Vec3(0.0, 0.0, 0.0),
                        confidence = 0.80,
                        reprojectionErrorPx = Double.POSITIVE_INFINITY,
                        observationCount = 0,
                        positionUncertaintyMeters = 0.09,
                    ),
                ),
                globalConfidence = 0.80,
            )

        val result = engine.evaluate(predictionOnly)

        assertEquals(0, result.measuredJointCount)
        assertEquals(1, result.predictionOnlyJointCount)
        assertEquals(1, result.envelopes.size)
        assertTrue(result.envelopes.single().predictionOnly)
    }

    @Test
    fun persistentNormalMotionBuildsCalibrationEnvelope() {
        val engine =
            CalibratedMotionUncertaintyEngine(
                CalibratedMotionUncertaintyEngine.Config(
                    minimumCalibrationSamples = 5,
                    maximumResidualSamples = 32,
                ),
            )

        var timestamp = 2_000_000_000L
        var x = 0.0

        repeat(12) {
            val result =
                engine.evaluate(
                    skeleton(
                        timestampNanos = timestamp,
                        x = x,
                    ),
                )

            assertTrue(result.meanRadiusMeters > 0.0)
            assertTrue(result.outOfDistributionRisk < 0.80)

            timestamp += 33_000_000L
            x += 0.010
        }

        val final =
            engine.evaluate(
                skeleton(
                    timestampNanos = timestamp,
                    x = x,
                ),
            )

        assertTrue(final.calibrationProgress > 0.50)
        assertTrue(final.empiricalCoverage >= 0.0)
        assertTrue(final.meanRadiusMeters < 0.60)
    }


    @Test
    fun extremeOutlierDoesNotWidenFutureCalibrationHistory() {
        val engine =
            CalibratedMotionUncertaintyEngine(
                CalibratedMotionUncertaintyEngine.Config(
                    minimumCalibrationSamples = 4,
                    maximumResidualSamples = 24,
                ),
            )

        var timestamp = 4_000_000_000L
        var x = 0.0

        repeat(10) {
            engine.evaluate(
                skeleton(
                    timestampNanos = timestamp,
                    x = x,
                ),
            )
            timestamp += 33_000_000L
            x += 0.010
        }

        val before =
            engine.evaluate(
                skeleton(
                    timestampNanos = timestamp,
                    x = x,
                ),
            )

        timestamp += 33_000_000L

        val outlier =
            engine.evaluate(
                skeleton(
                    timestampNanos = timestamp,
                    x = 4.0,
                ),
            )

        timestamp += 33_000_000L
        x += 0.010

        val after =
            engine.evaluate(
                skeleton(
                    timestampNanos = timestamp,
                    x = x,
                ),
            )

        assertTrue(outlier.outOfDistributionRisk > 0.0)
        assertTrue(
            after.meanRadiusMeters <
                before.meanRadiusMeters * 2.5,
        )
    }

    @Test
    fun abruptInnovationRaisesOodRisk() {
        val engine =
            CalibratedMotionUncertaintyEngine(
                CalibratedMotionUncertaintyEngine.Config(
                    minimumCalibrationSamples = 4,
                    maximumResidualSamples = 24,
                    highRiskThreshold = 0.55,
                ),
            )

        var timestamp = 3_000_000_000L
        repeat(8) {
            engine.evaluate(
                skeleton(
                    timestampNanos = timestamp,
                    x = it * 0.010,
                ),
            )
            timestamp += 33_000_000L
        }

        val result =
            engine.evaluate(
                skeleton(
                    timestampNanos = timestamp,
                    x = 2.0,
                ),
            )

        assertTrue(result.outOfDistributionRisk > 0.0)
        assertTrue(result.warnings.any {
            it.contains("out-of-distribution")
                || it.contains("outside-calibrated-envelope")
        })
    }
}
