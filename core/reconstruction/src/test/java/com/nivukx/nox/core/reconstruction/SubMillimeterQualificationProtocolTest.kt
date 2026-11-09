package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SubMillimeterQualificationProtocolTest {
    @Test(expected = IllegalArgumentException::class)
    fun slow_p99_limit_cannot_be_below_slow_p95_limit() {
        SubMillimeterQualificationProtocol(
            SubMillimeterQualificationProtocol.Config(
                maximumSlowP95ErrorMm = 0.80,
                maximumSlowP99ErrorMm = 0.70,
            ),
        )
    }



    @Test
    fun strict_lab_profile_rejects_single_correlated_reference_session() {
        val protocol =
            SubMillimeterQualificationProtocol(
                SubMillimeterQualificationProtocol.Config.labQualification()
            )

        val samples =
            buildList {
                repeat(3) { regime ->
                    repeat(120) { index ->
                        val velocity =
                            when (regime) {
                                0 -> 0.0
                                1 -> 0.50
                                else -> 1.50
                            }
                        val reference =
                            Vec3(
                                (regime * 120 + index).toDouble() *
                                    velocity /
                                    60.0,
                                0.20,
                                1.50,
                            )
                        add(
                            SubMillimeterQualificationProtocol.Sample(
                                timestampNanos =
                                    (regime * 120L + index) *
                                        16_666_667L,
                                estimate =
                                    reference +
                                        Vec3(
                                            0.00010,
                                            -0.00005,
                                            0.00005,
                                        ),
                                reference = reference,
                                referenceUncertaintyMm = 0.10,
                                confidence = 0.99,
                                referenceSessionId = "single-session",
                                sourceVelocityMetersPerSecond = velocity,
                            ),
                        )
                    }
                }
            }

        val report =
            protocol.qualify(
                samples = samples,
                temporalLagMs = 2.0,
            )

        assertFalse(report.accepted)
        assertEquals(1, report.referenceSessionCount)
        assertTrue(
            report.reasons.contains(
                "INSUFFICIENT_INDEPENDENT_REFERENCE_SESSIONS",
            ),
        )
    }

    @Test
    fun three_independent_reference_sessions_satisfy_lab_session_gate() {
        val protocol =
            SubMillimeterQualificationProtocol(
                SubMillimeterQualificationProtocol.Config.labQualification()
            )

        val samples =
            buildList {
                repeat(3) { session ->
                    repeat(120) { index ->
                        val velocity =
                            when {
                                index < 40 -> 0.0
                                index < 80 -> 0.50
                                else -> 1.50
                            }
                        val reference =
                            Vec3(
                                (session * 120 + index).toDouble() *
                                    velocity /
                                    60.0,
                                0.20,
                                1.50,
                            )
                        add(
                            SubMillimeterQualificationProtocol.Sample(
                                timestampNanos =
                                    (session * 120L + index) *
                                        16_666_667L,
                                estimate =
                                    reference +
                                        Vec3(
                                            0.00010,
                                            -0.00005,
                                            0.00005,
                                        ),
                                reference = reference,
                                referenceUncertaintyMm = 0.10,
                                confidence = 0.99,
                                referenceSessionId = "session-$session",
                                sourceVelocityMetersPerSecond = velocity,
                            ),
                        )
                    }
                }
            }

        val report =
            protocol.qualify(
                samples = samples,
                temporalLagMs = 2.0,
            )

        assertEquals(3, report.referenceSessionCount)
        assertTrue(
            report.reasons.none {
                it == "INSUFFICIENT_INDEPENDENT_REFERENCE_SESSIONS"
            },
        )
    }

    @Test
    fun insufficient_regime_evidence_never_earns_submillimeter_status() {
        val protocol =
            SubMillimeterQualificationProtocol(
                SubMillimeterQualificationProtocol.Config(
                    minimumSamplesPerRegime = 4,
                ),
            )

        val report =
            protocol.qualify(
                samples =
                    List(12) { index ->
                        SubMillimeterQualificationProtocol.Sample(
                            timestampNanos =
                                index.toLong() * 16_666_667L,
                            estimate =
                                Vec3(
                                    index * 0.00001,
                                    0.0,
                                    1.0,
                                ),
                            reference =
                                Vec3(
                                    index * 0.00001,
                                    0.0,
                                    1.0,
                                ),
                            referenceUncertaintyMm = 0.10,
                            confidence = 0.99,
                            sourceVelocityMetersPerSecond = 0.0,
                        )
                    },
                temporalLagMs = 2.0,
            )

        assertFalse(report.accepted)
        assertTrue(
            report.reasons.any {
                it == "STATIC:INSUFFICIENT_REGIME_SAMPLES"
            },
        )
    }

    @Test
    fun clean_static_slow_and_dynamic_regimes_can_earn_submillimeter_status() {
        val protocol =
            SubMillimeterQualificationProtocol(
                SubMillimeterQualificationProtocol.Config(
                    minimumSamplesPerRegime = 120,
                    maximumReferenceUncertaintyMm = 0.25,
                    maximumLagMs = 5.0,
                ),
            )

        val samples =
            buildList {
                repeat(120) { index ->
                    add(
                        sample(
                            index = index,
                            velocity = 0.0,
                        ),
                    )
                }
                repeat(120) { index ->
                    add(
                        sample(
                            index = 120 + index,
                            velocity = 0.50,
                        ),
                    )
                }
                repeat(120) { index ->
                    add(
                        sample(
                            index = 240 + index,
                            velocity = 1.50,
                        ),
                    )
                }
            }

        val report =
            protocol.qualify(
                samples = samples,
                temporalLagMs = 2.5,
            )

        assertTrue(report.accepted)
        assertTrue(report.reasons.isEmpty())
        assertTrue(
            report.regimeMetrics
                .filter { it.sampleCount > 0 }
                .all { it.accepted },
        )
        assertTrue(
            report.regimeMetrics
                .first {
                    it.regime ==
                        SubMillimeterQualificationProtocol.MotionRegime.DYNAMIC
                }
                .p95ErrorMm!! < 1.0,
        )
    }

    private fun sample(
        index: Int,
        velocity: Double,
    ): SubMillimeterQualificationProtocol.Sample {
        val timestamp =
            index.toLong() * 16_666_667L
        val x =
            index.toDouble() *
                velocity /
                60.0

        val reference =
            Vec3(
                x,
                0.20,
                1.50,
            )

        return SubMillimeterQualificationProtocol.Sample(
            timestampNanos = timestamp,
            estimate =
                reference +
                    Vec3(
                        0.00020,
                        -0.00010,
                        0.00010,
                    ),
            reference = reference,
            referenceUncertaintyMm = 0.10,
            confidence = 0.98,
            sourceVelocityMetersPerSecond = velocity,
        )
    }
}
