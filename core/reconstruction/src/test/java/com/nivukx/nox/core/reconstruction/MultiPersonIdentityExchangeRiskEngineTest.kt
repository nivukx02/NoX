package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiPersonIdentityExchangeRiskEngineTest {

    @Test
    fun singleStableSubjectIsClear() {
        val engine = MultiPersonIdentityExchangeRiskEngine()
        val result = engine.evaluate(
            timestampNanos = 1_000_000L,
            subjects = listOf(
                subject(
                    subjectId = 1,
                    visibility = 0.96,
                    confidence = 0.95,
                    crossView = 0.96,
                    temporal = 0.97,
                    overlap = 0.02,
                ),
            ),
        )

        assertEquals(
            MultiPersonIdentityExchangeRiskEngine.State.CLEAR,
            result.state,
        )
        assertTrue(result.productionIdentityConfidence > 0.80)
        assertTrue(result.exchangeRisk < 0.30)
    }

    @Test
    fun occlusionAndInteractionRaiseExchangeRisk() {
        val engine = MultiPersonIdentityExchangeRiskEngine()
        val result = engine.evaluate(
            timestampNanos = 2_000_000L,
            subjects = listOf(
                subject(
                    subjectId = 1,
                    visibility = 0.30,
                    confidence = 0.45,
                    crossView = 0.40,
                    temporal = 0.42,
                    overlap = 0.82,
                ),
                subject(
                    subjectId = 2,
                    visibility = 0.31,
                    confidence = 0.44,
                    crossView = 0.39,
                    temporal = 0.41,
                    overlap = 0.81,
                ),
            ),
        )

        assertTrue(
            result.state ==
                MultiPersonIdentityExchangeRiskEngine.State.HIGH_RISK ||
                result.state ==
                MultiPersonIdentityExchangeRiskEngine.State.HOLD,
        )
        assertTrue(result.exchangeRisk > 0.50)
        assertTrue(result.warnings.any { it.contains("occlusion") })
        assertTrue(result.warnings.any { it.contains("ambiguity") })
    }

    @Test
    fun historyDetectsRapidIdentityQualityChange() {
        val engine = MultiPersonIdentityExchangeRiskEngine()
        repeat(6) { frame ->
            engine.evaluate(
                timestampNanos = frame * 33_333_333L,
                subjects = listOf(
                    subject(
                        subjectId = 7,
                        visibility =
                            if (frame < 4) 0.95 else 0.35,
                        confidence =
                            if (frame < 4) 0.94 else 0.38,
                        crossView =
                            if (frame < 4) 0.93 else 0.42,
                        temporal =
                            if (frame < 4) 0.95 else 0.40,
                        overlap =
                            if (frame < 4) 0.02 else 0.72,
                    ),
                ),
            )
        }

        val result = engine.evaluate(
            timestampNanos = 250_000_000L,
            subjects = listOf(
                subject(
                    subjectId = 7,
                    visibility = 0.34,
                    confidence = 0.36,
                    crossView = 0.40,
                    temporal = 0.35,
                    overlap = 0.75,
                ),
            ),
        )

        assertTrue(result.warnings.any { it.contains("temporal") })
        assertTrue(result.exchangeRisk > 0.50)
    }

    @Test
    fun stateIsBoundedUnderSubjectChurn() {
        val engine = MultiPersonIdentityExchangeRiskEngine(
            MultiPersonIdentityExchangeRiskEngine.Config(
                maximumSubjects = 4,
                historyFrames = 3,
            ),
        )

        repeat(20) { frame ->
            engine.evaluate(
                timestampNanos = frame * 10_000_000L,
                subjects =
                    (0 until 12).map { id ->
                        subject(
                            subjectId = id + frame * 100,
                            visibility = 0.80,
                            confidence = 0.82,
                            crossView = 0.84,
                            temporal = 0.85,
                            overlap = 0.05,
                        )
                    },
            )
        }

        val result = engine.evaluate(
            timestampNanos = 400_000_000L,
            subjects = listOf(
                subject(
                    subjectId = 1000,
                    visibility = 0.90,
                    confidence = 0.90,
                    crossView = 0.90,
                    temporal = 0.90,
                ),
            ),
        )

        assertTrue(result.evaluatedSubjects <= 4)
        assertTrue(result.warnings.contains("subject-cap-applied"))
    }

    private fun subject(
        subjectId: Int,
        visibility: Double,
        confidence: Double,
        crossView: Double,
        temporal: Double,
        overlap: Double = 0.0,
    ) =
        MultiPersonIdentityExchangeRiskEngine.SubjectEvidence(
            subjectId = subjectId,
            visibleLandmarks = (visibility * 100.0).toInt(),
            totalLandmarks = 100,
            meanConfidence = confidence,
            crossViewAgreement = crossView,
            temporalIdentityConfidence = temporal,
            contactOverlap = overlap,
            cameraCount = 4,
        )
}
