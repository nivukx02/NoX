package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReferenceMetrologyBenchmarkEngineTest {

    @Test
    fun preciseReferenceCohortCanQualify() {
        val engine =
            ReferenceMetrologyBenchmarkEngine(
                ReferenceMetrologyBenchmarkEngine.Config(
                    minimumValidSamples = 6,
                    minimumIndependentSessions = 3,
                    minimumRepeatabilitySamplesPerSession = 2,
                    maximumStoredSamples = 64,
                ),
            )

        repeat(3) { session ->
            repeat(2) { index ->
                val x =
                    0.500 +
                        session * 0.00001 +
                        index * 0.00001

                engine.add(
                    sample(
                        session = "S" + session,
                        reference = Vec3(x, 0.0, 0.0),
                        estimated = Vec3(x + 0.00020, 0.0, 0.0),
                    ),
                )
            }
        }

        val result = engine.evaluate()

        assertTrue(result.qualificationReady)
        assertTrue(result.global.p95Millimeters < 1.0)
        assertTrue(result.global.rmseMillimeters < 1.0)
        assertEquals(3, result.independentSessions)
        assertEquals(6, result.validSamples)
    }

    @Test
    fun poorTailAndDropoutRejectsQualification() {
        val engine =
            ReferenceMetrologyBenchmarkEngine(
                ReferenceMetrologyBenchmarkEngine.Config(
                    minimumValidSamples = 4,
                    minimumIndependentSessions = 2,
                    minimumRepeatabilitySamplesPerSession = 2,
                    maximumDropoutFraction = 0.10,
                    maximumStoredSamples = 64,
                ),
            )

        repeat(2) { session ->
            engine.add(
                sample(
                    session = "S" + session,
                    estimated = Vec3(0.0, 0.0, 0.0),
                    reference = Vec3(0.0, 0.0, 0.0),
                ),
            )
            engine.add(
                sample(
                    session = "S" + session,
                    estimated = Vec3(0.0, 0.0, 0.0),
                    reference = Vec3(0.0, 0.0, 0.0),
                ).copy(
                    estimated =
                        Vec3(
                            0.020,
                            0.0,
                            0.0,
                        ),
                ),
            )
        }

        engine.add(
            sample(
                session = "S0",
                validEstimate = false,
            ),
        )

        val result = engine.evaluate()

        assertFalse(result.qualificationReady)
        assertTrue(result.qualificationReasons.contains("GLOBAL_P95_EXCEEDS_TARGET"))
        assertTrue(result.dropoutFraction > 0.0)
    }

    @Test
    fun occludedRegimeIsSeparatedFromVisibleRegime() {
        val engine =
            ReferenceMetrologyBenchmarkEngine(
                ReferenceMetrologyBenchmarkEngine.Config(
                    minimumValidSamples = 2,
                    minimumIndependentSessions = 2,
                    minimumRepeatabilitySamplesPerSession = 1,
                    maximumStoredSamples = 32,
                    maximumOccludedP95Millimeters = 1.0,
                ),
            )

        engine.add(
            sample(
                session = "A",
                estimated = Vec3(0.0, 0.0, 0.0),
                reference = Vec3(0.0, 0.0, 0.0),
            ),
        )
        engine.add(
            sample(
                session = "B",
                estimated = Vec3(0.010, 0.0, 0.0),
                reference = Vec3(0.0, 0.0, 0.0),
                occluded = true,
            ),
        )

        val result = engine.evaluate()

        assertTrue(result.occluded.count == 1)
        assertTrue(
            result.occluded.p95Millimeters >
                result.visible.p95Millimeters,
        )
        assertFalse(result.qualificationReady)
    }

    @Test
    fun independentSessionsAreRequired() {
        val engine =
            ReferenceMetrologyBenchmarkEngine(
                ReferenceMetrologyBenchmarkEngine.Config(
                    minimumValidSamples = 2,
                    minimumIndependentSessions = 3,
                    minimumRepeatabilitySamplesPerSession = 1,
                    maximumStoredSamples = 32,
                ),
            )

        engine.add(
            sample(
                session = "only",
                estimated = Vec3(0.0, 0.0, 0.0),
            ),
        )
        engine.add(
            sample(
                session = "only",
                estimated = Vec3(0.0001, 0.0, 0.0),
            ),
        )

        val result = engine.evaluate()

        assertTrue(
            result.qualificationReasons.contains(
                "INSUFFICIENT_INDEPENDENT_SESSIONS",
            ),
        )
    }

    private fun sample(
        session: String,
        estimated: Vec3 = Vec3(0.0, 0.0, 0.0),
        reference: Vec3 = Vec3(0.0, 0.0, 0.0),
        occluded: Boolean = false,
        validEstimate: Boolean = true,
    ) =
        ReferenceMetrologyBenchmarkEngine.Sample(
            sessionId = session,
            timestampNanos = 1_000_000L,
            subjectId = "subject-1",
            jointId = 11,
            estimated = estimated,
            reference = reference,
            referenceUncertaintyMillimeters = 0.10,
            estimatedConfidence = 0.99,
            occluded = occluded,
            dynamic = true,
            validEstimate = validEstimate,
        )
}
