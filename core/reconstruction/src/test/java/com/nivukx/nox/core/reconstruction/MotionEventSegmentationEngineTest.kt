package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionEventSegmentationEngineTest {

    @Test
    fun sustainedMotionProducesWalkOrRunEvent() {
        val engine = MotionEventSegmentationEngine()
        val result = engine.classify(
            MotionEventSegmentationEngine.Input(
                frame = frame(1_000_000_000L, 1.5),
                rootVelocityMetersPerSecond = 1.8,
                rootAccelerationMetersPerSecond2 = 1.2,
                leftFootStable = true,
                rightFootStable = false,
                footSkateRisk = 0.08,
                confidence = 0.92,
            ),
        )

        assertTrue(
            result.type ==
                MotionEventSegmentationEngine.EventType.WALK ||
                result.type ==
                MotionEventSegmentationEngine.EventType.RUN,
        )
        assertTrue(result.confidence >= 0.0)
    }

    @Test
    fun cameraMotionPreventsOverconfidentLocomotionClassification() {
        val engine = MotionEventSegmentationEngine()

        val result =
            engine.classify(
                MotionEventSegmentationEngine.Input(
                    frame = frame(
                        1_260_000_000L,
                        1.8,
                        uncertaintyMeters = 0.03,
                        globalConfidence = 0.94,
                    ),
                    rootVelocityMetersPerSecond = 2.4,
                    rootAccelerationMetersPerSecond2 = 1.1,
                    leftFootStable = true,
                    rightFootStable = true,
                    footSkateRisk = 0.72,
                    cameraMotionRisk = 0.86,
                    confidence = 0.92,
                ),
            )

        assertTrue(
            result.confidence < 0.70 ||
                result.type ==
                    MotionEventSegmentationEngine.EventType.UNKNOWN,
        )
        assertTrue(
            result.diagnostics.any {
                it.contains("camera-motion-risk")
            },
        )
    }

    @Test
    fun highUncertaintyAndFootSkateDoNotProduceOverconfidentSemanticEvent() {
        val engine = MotionEventSegmentationEngine()

        val result =
            engine.classify(
                MotionEventSegmentationEngine.Input(
                    frame = frame(
                        1_200_000_000L,
                        1.8,
                        uncertaintyMeters = 0.24,
                        globalConfidence = 0.42,
                    ),
                    rootVelocityMetersPerSecond = 2.2,
                    rootAccelerationMetersPerSecond2 = 3.2,
                    leftFootStable = false,
                    rightFootStable = false,
                    footSkateRisk = 0.82,
                    confidence = 0.48,
                ),
            )

        assertTrue(
            result.confidence < 0.65 ||
                result.type ==
                    MotionEventSegmentationEngine.EventType.UNKNOWN,
        )
        assertTrue(result.evidenceQuality < 0.65)
        assertTrue(result.diagnostics.any { it.contains("foot-skate") })
    }

    @Test
    fun stopAfterActiveMotionCanProduceExit() {
        val engine = MotionEventSegmentationEngine()

        repeat(4) { index ->
            engine.classify(
                MotionEventSegmentationEngine.Input(
                    frame = frame(
                        1_000_000_000L +
                            index * 16_666_666L,
                        1.0,
                    ),
                    rootVelocityMetersPerSecond = 1.5,
                    rootAccelerationMetersPerSecond2 = 0.8,
                    leftFootStable = false,
                    rightFootStable = false,
                    footSkateRisk = 0.05,
                    confidence = 0.92,
                ),
            )
        }

        val stopped = engine.classify(
            MotionEventSegmentationEngine.Input(
                frame = frame(1_100_000_000L, 0.0),
                rootVelocityMetersPerSecond = 0.0,
                rootAccelerationMetersPerSecond2 = 0.0,
                leftFootStable = true,
                rightFootStable = true,
                footSkateRisk = 0.0,
                confidence = 0.95,
            ),
        )

        assertTrue(
            stopped.phase ==
                MotionEventSegmentationEngine.Phase.EXIT ||
                stopped.type ==
                MotionEventSegmentationEngine.EventType.IDLE,
        )
    }

    private fun frame(
        timestamp: Long,
        rootX: Double,
        uncertaintyMeters: Double = 0.02,
        globalConfidence: Double = 0.91,
    ) = SkeletonFrame(
        timestampNanos = timestamp,
        subjectType = SubjectType.HUMAN,
        trackId = 3L,
        joints = listOf(
            ReconstructionJoint(
                landmarkId = 23,
                position = Vec3(rootX, 1.0, 0.0),
                velocity = Vec3(1.5, 0.0, 0.0),
                confidence = 0.94,
                reprojectionErrorPx = 1.0,
                observationCount = 4,
                positionUncertaintyMeters = uncertaintyMeters,
            ),
            ReconstructionJoint(
                landmarkId = 11,
                position = Vec3(rootX, 1.5, 0.0),
                velocity = Vec3(1.5, 0.0, 0.0),
                confidence = 0.90,
                reprojectionErrorPx = 1.1,
                observationCount = 4,
            ),
            ReconstructionJoint(
                landmarkId = 27,
                position = Vec3(rootX - 0.1, 0.0, 0.1),
                velocity = Vec3(0.0, 0.0, 0.0),
                confidence = 0.90,
                reprojectionErrorPx = 1.0,
                observationCount = 4,
            ),
            ReconstructionJoint(
                landmarkId = 28,
                position = Vec3(rootX + 0.1, 0.0, 0.1),
                velocity = Vec3(0.0, 0.0, 0.0),
                confidence = 0.90,
                reprojectionErrorPx = 1.0,
                observationCount = 4,
            ),
        ),
        globalConfidence = globalConfidence,
    )
}
