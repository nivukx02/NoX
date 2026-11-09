package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.PerceptionSource
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveEvidenceReliabilityTest {

    @Test
    fun healthyEvidenceRecoversTowardBaseReliability() {
        val engine = AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                baseReliability = 0.80,
                learningRate = 0.18,
                recoveryRate = 0.08,
            ),
        )

        var reliability = 0.0
        repeat(24) { index ->
            reliability = engine.observe(
                observation = AdaptiveEvidenceReliability.Observation(
                    source = PerceptionSource.MEDIAPIPE,
                    timestampNanos = index.toLong() * 16_666_667L,
                    reportedConfidence = 0.96,
                    residual = 0.02,
                    disagreement = 0.02,
                    stability = 0.96,
                    ageMillis = 0.0,
                ),
                consensusConfidence = 0.94,
            ).value
        }

        assertTrue("reliability should recover above 0.70", reliability > 0.70)
        assertTrue("reliability must remain bounded", reliability <= 1.0)
    }

    @Test
    fun rejectedEvidenceDegradesAndCanRecover() {
        val engine = AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                baseReliability = 0.82,
                learningRate = 0.30,
                recoveryRate = 0.10,
            ),
        )

        repeat(10) { index ->
            engine.observe(
                observation = AdaptiveEvidenceReliability.Observation(
                    source = PerceptionSource.ONNX,
                    timestampNanos = index.toLong() * 20_000_000L,
                    reportedConfidence = 0.20,
                    residual = 2.5,
                    disagreement = 0.90,
                    stability = 0.15,
                    ageMillis = 0.0,
                    accepted = false,
                ),
                consensusConfidence = 0.85,
            )
        }

        val degraded = engine.reliability(
            PerceptionSource.ONNX,
        )
        assertTrue("bad evidence should degrade", degraded.value < 0.65)
        assertTrue("degraded source should be marked", degraded.degraded)

        repeat(20) { index ->
            engine.observe(
                observation = AdaptiveEvidenceReliability.Observation(
                    source = PerceptionSource.ONNX,
                    timestampNanos =
                        (index + 100L) * 20_000_000L,
                    reportedConfidence = 0.93,
                    residual = 0.04,
                    disagreement = 0.03,
                    stability = 0.91,
                    ageMillis = 0.0,
                ),
                consensusConfidence = 0.91,
            )
        }

        val recovered = engine.reliability(
            PerceptionSource.ONNX,
        )
        assertTrue("source should recover after sustained good evidence",
            recovered.value > degraded.value)
    }

    @Test
    fun staleEvidenceApproachesZeroWeight() {
        val engine = AdaptiveEvidenceReliability()

        engine.observe(
            observation = AdaptiveEvidenceReliability.Observation(
                source = PerceptionSource.SENSOR,
                timestampNanos = 1_000_000L,
                reportedConfidence = 0.95,
                residual = 0.01,
                disagreement = 0.01,
                stability = 0.95,
            ),
            consensusConfidence = 0.95,
        )

        val fresh = engine.weight(
            source = PerceptionSource.SENSOR,
            ageMillis = 0.0,
            reportedConfidence = 0.95,
        )

        val stale = engine.weight(
            source = PerceptionSource.SENSOR,
            ageMillis = 2_000.0,
            reportedConfidence = 0.95,
        )

        assertTrue("fresh evidence should carry weight", fresh > 0.40)
        assertTrue("stale evidence should be strongly suppressed",
            stale < fresh * 0.15)
    }

    @Test
    fun snapshotReportsConsensusAndDominantSource() {
        val engine = AdaptiveEvidenceReliability()

        engine.observe(
            observation = AdaptiveEvidenceReliability.Observation(
                source = PerceptionSource.ML_KIT,
                timestampNanos = 10L,
                reportedConfidence = 0.92,
                residual = 0.03,
                disagreement = 0.03,
                stability = 0.90,
            ),
            consensusConfidence = 0.90,
        )

        engine.observe(
            observation = AdaptiveEvidenceReliability.Observation(
                source = PerceptionSource.MEDIAPIPE,
                timestampNanos = 11L,
                reportedConfidence = 0.88,
                residual = 0.04,
                disagreement = 0.04,
                stability = 0.88,
            ),
            consensusConfidence = 0.90,
        )

        val snapshot = engine.snapshot(11L)

        assertTrue(snapshot.effectiveConsensus > 0.50)
        assertTrue(snapshot.values.containsKey(PerceptionSource.ML_KIT))
        assertTrue(snapshot.values.containsKey(PerceptionSource.MEDIAPIPE))
        assertTrue(snapshot.highestReliabilitySource != null)
        assertTrue(snapshot.lowestReliabilitySource != null)
    }

    @Test
    fun nonFiniteInputIsNeverPromoted() {
        val engine = AdaptiveEvidenceReliability()

        val result = engine.observe(
            observation = AdaptiveEvidenceReliability.Observation(
                source = PerceptionSource.UNKNOWN,
                timestampNanos = 100L,
                reportedConfidence = Double.NaN,
                residual = Double.POSITIVE_INFINITY,
                disagreement = Double.NaN,
                stability = Double.NaN,
            ),
            consensusConfidence = 0.90,
        )

        assertTrue(
            "non-finite evidence must reduce reliability",
            result.value <
                AdaptiveEvidenceReliability.Config().baseReliability,
        )
        assertTrue(
            "reliability must remain bounded",
            result.value >=
                AdaptiveEvidenceReliability.Config().minimumReliability,
        )
    }
}
