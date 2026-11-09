package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Hand3DFrame
import com.nivukx.nox.core.model.Hand3DLandmark
import com.nivukx.nox.core.model.Hand3DResult
import com.nivukx.nox.core.model.HandSide
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HandTemporalConsistencyEngineTest {

    @Test
    fun stableHandSequenceRemainsReliable() {
        val engine = HandTemporalConsistencyEngine()

        engine.evaluate(frame(1_000_000_000L, 0.92, Vec3(0.0, 0.0, 0.0)))

        val result = engine.evaluate(
            frame(
                1_016_666_667L,
                0.94,
                Vec3(0.005, 0.0, 0.002),
            ),
        )

        assertTrue(result.confidence > 0.55)
        assertTrue(result.temporalStability > 0.45)
        assertTrue(result.warnings.none { it.startsWith("fatal:") })
    }

    @Test
    fun extremeLowConfidenceJumpIsRejected() {
        val engine = HandTemporalConsistencyEngine()

        engine.evaluate(
            frame(
                2_000_000_000L,
                0.95,
                Vec3(0.0, 0.0, 0.0),
            ),
        )

        val result = engine.evaluate(
            frame(
                2_016_666_667L,
                0.22,
                Vec3(0.8, 0.4, 0.2),
            ),
        )

        assertTrue(result.rejectedLandmarks > 0)
        assertTrue(
            result.hands.any {
                it.rejectedLandmarks > 0
            },
        )
    }

    @Test
    fun uncertaintyAndArticulationChangesReduceConfidence() {
        val engine = HandTemporalConsistencyEngine()

        engine.evaluate(
            frame(
                3_000_000_000L,
                0.90,
                Vec3(0.0, 0.0, 0.0),
            ),
        )

        val result = engine.evaluate(
            frame(
                3_016_666_667L,
                0.42,
                Vec3(0.12, 0.05, 0.08),
                uncertaintyMeters = 0.14,
            ),
        )

        assertTrue(result.confidence < 0.90)
        assertTrue(result.articulationStability < 1.0)
    }
    @Test
    fun accelerated_finger_sequence_reports_bounded_acceleration() {
        val engine =
            HandTemporalConsistencyEngine(
                HandTemporalConsistencyEngine.Config(
                    accelerationScaleMetersPerSecond2 = 40.0,
                    minimumConfidence = 0.10,
                ),
            )

        engine.evaluate(
            frame(
                5_000_000_000L,
                0.96,
                Vec3(0.0, 0.0, 0.0),
            ),
        )
        engine.evaluate(
            frame(
                5_016_666_667L,
                0.96,
                Vec3(0.004, 0.0, 0.0),
            ),
        )

        val result =
            engine.evaluate(
                frame(
                    5_033_333_334L,
                    0.96,
                    Vec3(0.012, 0.0, 0.0),
                ),
            )

        val diagnostic = result.hands.single()
        assertTrue(
            diagnostic.meanAccelerationMetersPerSecond2.isFinite(),
        )
        assertTrue(
            diagnostic.meanAccelerationMetersPerSecond2 <=
                40.0 + 1e-6,
        )
        assertTrue(result.confidence > 0.0)
    }

    @Test
    fun strictnessChangesInnovationGate() {
        val engine = HandTemporalConsistencyEngine()
        engine.configureStrictness(1.0)

        engine.evaluate(
            frame(
                4_000_000_000L,
                0.96,
                Vec3(0.0, 0.0, 0.0),
            ),
        )

        val result = engine.evaluate(
            frame(
                4_016_666_667L,
                0.96,
                Vec3(0.12, 0.0, 0.0),
            ),
        )

        assertTrue(result.warnings.isNotEmpty())
        assertTrue(result.confidence < 0.90)
    }


    private fun frame(
        timestamp: Long,
        confidence: Double,
        offset: Vec3,
        uncertaintyMeters: Double = 0.015,
    ) =
        Hand3DFrame(
            timestampNanos = timestamp,
            hands = listOf(
                Hand3DResult(
                    handId = 101L,
                    side = HandSide.RIGHT,
                    timestampNanos = timestamp,
                    landmarks = (0 until 21).map { index ->
                        Hand3DLandmark(
                            index = index,
                            position = offset + Vec3(
                                index * 0.002 +
                                    if (
                                        uncertaintyMeters >= 0.10 &&
                                            index >= 10
                                    ) {
                                        (index - 9) * 0.0015
                                    } else {
                                        0.0
                                    },
                                index * 0.001,
                                index * 0.001,
                            ),
                            confidence = confidence,
                            uncertaintyMeters = uncertaintyMeters,
                            observationCount = 2,
                        )
                    },
                    confidence = confidence,
                    meanUncertaintyMeters = uncertaintyMeters,
                    reprojectionErrorPx = 2.0,
                ),
            ),
            confidence = confidence,
        )
}
