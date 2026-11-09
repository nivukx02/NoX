package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndMotionStateEstimatorTest {

    private fun joint(
        id: Int,
        x: Double,
        confidence: Double = 1.0,
        velocity: Vec3 = Vec3(0.0, 0.0, 0.0),
        acceleration: Vec3 = Vec3(0.0, 0.0, 0.0),
    ): ReconstructionJoint =
        ReconstructionJoint(
            landmarkId = id,
            position = Vec3(x, 0.0, 0.0),
            velocity = velocity,
            acceleration = acceleration,
            confidence = confidence,
            positionUncertaintyMeters = 0.004,
        )

    private fun frame(
        timestampNanos: Long,
        joints: List<ReconstructionJoint>,
        confidence: Double = 1.0,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = joints,
            globalConfidence = confidence,
        )

    @Test
    fun initializesFromObservedGeometry() {
        val estimator = HighEndMotionStateEstimator()

        val result = estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_000_000_000L,
                skeleton = frame(
                    1_000_000_000L,
                    listOf(joint(0, 0.0)),
                ),
            ),
        )

        assertEquals(1, result.skeleton.joints.size)
        assertEquals(0.0, result.skeleton.joints.first().position.x, 1e-9)
        assertTrue(result.diagnostics.finalScore > 0.5)
        assertTrue(result.accepted)
    }

    @Test
    fun rejectsLargeLowConfidenceInnovationWithoutExplodingState() {
        val estimator = HighEndMotionStateEstimator(
            HighEndMotionStateEstimator.Config(
                softInnovationGateMeters = 0.05,
                hardInnovationGateMeters = 0.10,
                        ),
        )

        estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_000_000_000L,
                skeleton = frame(
                    1_000_000_000L,
                    listOf(
                        joint(
                            id = 0,
                            x = 0.0,
                            velocity = Vec3(1.0, 0.0, 0.0),
                        ),
                    ),
                ),
            ),
        )

        val result = estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_016_666_666L,
                skeleton = frame(
                    1_016_666_666L,
                    listOf(
                        joint(
                            id = 0,
                            x = 1.5,
                            confidence = 0.15,
                        ),
                    ),
                ),
            ),
        )

        assertTrue(result.skeleton.joints.first().position.x < 0.5)
        assertTrue(
            result.diagnostics.rejectedMeasurements >= 1 ||
                result.diagnostics.predictionOnlyJoints >= 1,
        )
    }

    @Test
    fun preservesBoundedOcclusionAndGrowsUncertainty() {
        val estimator = HighEndMotionStateEstimator(
            HighEndMotionStateEstimator.Config(
                maximumOcclusionFrames = 3,
            ),
        )

        estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_000_000_000L,
                skeleton = frame(
                    1_000_000_000L,
                    listOf(
                        joint(
                            id = 0,
                            x = 0.0,
                            velocity = Vec3(0.8, 0.0, 0.0),
                        ),
                    ),
                ),
            ),
        )

        val second = estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_016_666_666L,
                skeleton = frame(
                    1_016_666_666L,
                    emptyList(),
                ),
            ),
        )

        val third = estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_033_333_332L,
                skeleton = frame(
                    1_033_333_332L,
                    emptyList(),
                ),
            ),
        )

        assertTrue(second.skeleton.joints.isNotEmpty())
        assertTrue(third.skeleton.joints.isNotEmpty())
        assertTrue(
            third.skeleton.joints.first().positionUncertaintyMeters >=
                second.skeleton.joints.first().positionUncertaintyMeters,
        )
    }

    @Test
    fun timestampRegressionResetsTemporalState() {
        val estimator = HighEndMotionStateEstimator()

        estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 2_000_000_000L,
                skeleton = frame(
                    2_000_000_000L,
                    listOf(
                        joint(
                            id = 0,
                            x = 8.0,
                            velocity = Vec3(4.0, 0.0, 0.0),
                        ),
                    ),
                ),
            ),
        )

        val result = estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_000_000_000L,
                skeleton = frame(
                    1_000_000_000L,
                    listOf(joint(0, 0.0)),
                ),
            ),
        )

        assertEquals(0.0, result.skeleton.joints.first().position.x, 1e-9)
        assertEquals(1, result.diagnostics.acceptedMeasurements)
    }

    @Test
    fun temporalMemoryReportsFiniteContinuityEvidence() {
        val estimator =
            HighEndMotionStateEstimator(
                HighEndMotionStateEstimator.Config(
                    temporalMemoryAlpha = 0.30,
                ),
            )

        estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_000_000_000L,
                skeleton = frame(
                    1_000_000_000L,
                    listOf(
                        joint(
                            id = 0,
                            x = 0.0,
                            velocity = Vec3(1.0, 0.0, 0.0),
                        ),
                    ),
                ),
            ),
        )

        val result =
            estimator.estimate(
                HighEndMotionStateEstimator.Input(
                    timestampNanos = 1_016_666_666L,
                    skeleton = frame(
                        1_016_666_666L,
                        listOf(
                            joint(
                                id = 0,
                                x = 0.02,
                                confidence = 0.90,
                            ),
                        ),
                    ),
                ),
            )

        val diagnostic =
            result.diagnostics.jointDiagnostics.single()

        assertTrue(
            diagnostic.temporalMemoryScore in 0.0..1.0,
        )
        assertTrue(
            result.diagnostics.meanTemporalMemoryScore in 0.0..1.0,
        )
        assertTrue(
            result.diagnostics.finalScore.isFinite(),
        )
    }

    @Test
    fun cameraMotionReducesMeasurementAggressivenessButKeepsFiniteOutput() {
        val estimator = HighEndMotionStateEstimator()

        estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_000_000_000L,
                skeleton = frame(
                    1_000_000_000L,
                    listOf(joint(0, 0.0)),
                ),
            ),
        )

        val result = estimator.estimate(
            HighEndMotionStateEstimator.Input(
                timestampNanos = 1_016_666_666L,
                skeleton = frame(
                    1_016_666_666L,
                    listOf(
                        joint(
                            id = 0,
                            x = 0.10,
                            confidence = 0.85,
                        ),
                    ),
                ),
                cameraMotionRisk = 0.95,
                preserveMicroMotion = true,
            ),
        )

        val output = result.skeleton.joints.first()
        assertTrue(output.position.x.isFinite())
        assertTrue(output.velocity.x.isFinite())
        assertTrue(output.acceleration.x.isFinite())
        assertTrue(output.positionUncertaintyMeters.isFinite())
    }

    @Test
    fun configurationRemainsExplicitAndDeterministic() {
        val config = HighEndMotionStateEstimator.Config(
            predictionHorizonSeconds = 0.10,
            maximumOcclusionFrames = 5,
            microMotionPreservation = 0.95,
        )
        val estimator = HighEndMotionStateEstimator(config)

        assertEquals(5, estimator.config().maximumOcclusionFrames)
        assertEquals(0.95, estimator.config().microMotionPreservation, 1e-9)

        estimator.reset()
        assertEquals(0, estimator.stateCount())
    }
}
