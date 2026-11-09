package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertTrue
import org.junit.Test

class BodyJointUncertaintyGateEngineTest {

    private val engine = BodyJointUncertaintyGateEngine()

    private fun input(
        x: Double = 0.5,
        y: Double = 0.5,
        confidence: Double = 0.95,
        timing: Double = 0.96,
        alignment: Double = 0.96,
        uncertaintyNanos: Long = 1_000_000L,
        rolling: Double = 0.95,
        rowOffsetNanos: Long = 100_000L,
        frameDurationNanos: Long = 16_666_667L,
        exposureTimeNanos: Long = 4_000_000L,
        predicted: Boolean = false,
    ) =
        BodyJointUncertaintyGateEngine.Input(
            normalizedX = x,
            normalizedY = y,
            observationConfidence = confidence,
            captureTimingConfidence = timing,
            temporalAlignmentConfidence = alignment,
            temporalUncertaintyNanos = uncertaintyNanos,
            rollingShutterFraction = rolling,
            rowTimeOffsetNanos = rowOffsetNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
            predicted = predicted,
            baseUncertaintyMeters = 0.015,
        )

    @Test
    fun strongObservationReceivesHighFusionWeight() {
        val result = engine.evaluate(input())

        assertTrue(result.accepted)
        assertTrue(result.viewExpertWeight > 0.45)
        assertTrue(result.uncertaintyMeters < 0.15)
    }

    @Test
    fun borderObservationIsRejected() {
        val result = engine.evaluate(
            input(x = 0.003, y = 0.5),
        )

        assertTrue(result.hardRejected)
        assertTrue(
            "image-boundary-risk" in result.reasons,
        )
    }

    @Test
    fun poorTimingAndRollingShutterReduceTrust() {
        val clean = engine.evaluate(input())
        val degraded = engine.evaluate(
            input(
                timing = 0.25,
                alignment = 0.30,
                uncertaintyNanos = 35_000_000L,
                rolling = 0.35,
                rowOffsetNanos = 7_500_000L,
                exposureTimeNanos = 14_000_000L,
            ),
        )

        assertTrue(
            degraded.viewExpertWeight <
                clean.viewExpertWeight,
        )
        assertTrue(
            degraded.uncertaintyMeters >=
                clean.uncertaintyMeters,
        )
        assertTrue(
            degraded.reasons.any {
                it.contains("risk") ||
                    it.contains("alignment") ||
                    it.contains("exposure")
            },
        )
    }

    @Test
    fun predictedObservationIsDownweighted() {
        val observed = engine.evaluate(input())
        val predicted = engine.evaluate(
            input(predicted = true),
        )

        assertTrue(
            predicted.viewExpertWeight <
                observed.viewExpertWeight,
        )
        assertTrue(
            "prediction-only-observation" in
                predicted.reasons,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun confidenceCannotEscapeProductionRange() {
        engine.evaluate(
            input(confidence = 1.2),
        )
    }
    @Test
    fun anatomicalStabilityAndLongSpanReduceReliabilityWithoutHardRejectingGoodEvidence() {
        val stable = BodyJointUncertaintyGateEngine().evaluate(
            BodyJointUncertaintyGateEngine.Input(
                normalizedX = 0.50,
                normalizedY = 0.50,
                observationConfidence = 0.95,
                captureTimingConfidence = 0.98,
                temporalAlignmentConfidence = 0.98,
                temporalUncertaintyNanos = 100_000L,
                rollingShutterFraction = 0.98,
                rowTimeOffsetNanos = 100_000L,
                frameDurationNanos = 16_666_666L,
                exposureTimeNanos = 4_000_000L,
                baseUncertaintyMeters = 0.004,
                anatomicalStability = 1.0,
                temporalSpanFrames = 1,
            ),
        )
        val unstable = BodyJointUncertaintyGateEngine().evaluate(
            BodyJointUncertaintyGateEngine.Input(
                normalizedX = 0.50,
                normalizedY = 0.50,
                observationConfidence = 0.95,
                captureTimingConfidence = 0.98,
                temporalAlignmentConfidence = 0.98,
                temporalUncertaintyNanos = 100_000L,
                rollingShutterFraction = 0.98,
                rowTimeOffsetNanos = 100_000L,
                frameDurationNanos = 16_666_666L,
                exposureTimeNanos = 4_000_000L,
                baseUncertaintyMeters = 0.004,
                anatomicalStability = 0.35,
                temporalSpanFrames = 40,
            ),
        )
        assertTrue(stable.viewExpertWeight > unstable.viewExpertWeight)
        assertTrue(unstable.reasons.any { it == "anatomically-unstable-landmark" })
        assertTrue(unstable.reasons.any { it == "long-temporal-span-reliability-reduced" })
    )
}