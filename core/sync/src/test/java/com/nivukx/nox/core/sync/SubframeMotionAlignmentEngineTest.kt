package com.nivukx.nox.core.sync

class SubframeMotionAlignmentEngineTest {

    private val engine =
        SubframeMotionAlignmentEngine(
            SubframeMotionAlignmentEngine.Config(
                minimumSamples = 12,
                maximumOffsetNanos = 40_000_000L,
            ),
        )

    fun recovers_zero_offset() {
        val first = samples(0L)
        val second = samples(0L)

        val result = engine.estimate(first, second)

        check(result.accepted)
        check(kotlin.math.abs(result.offsetNanos) < 2_000_000.0)
    }

    fun recovers_signed_positive_delay() {
        val first = samples(0L)
        val second = samples(
            timestampShiftNanos = 8_000_000L,
            phaseShiftNanos = 0L,
        )

        val result = engine.estimate(first, second)

        check(result.accepted)
        check(result.offsetNanos > 5_000_000.0)
        check(result.offsetNanos < 11_000_000.0)
    }

    fun remains_stable_with_noise() {
        val first = samples(0L)
        val second =
            samples(
                timestampShiftNanos = -6_000_000L,
                phaseShiftNanos = 0L,
                noiseScale = 0.035,
            )

        val result = engine.estimate(first, second)

        check(result.accepted)
        check(result.confidence > 0.45)
        check(kotlin.math.abs(result.offsetNanos) > 2_000_000.0)
    }

    fun rejects_insufficient_evidence() {
        val first =
            samples(
                count = 5,
            )
        val second =
            samples(
                count = 5,
            )

        val result = engine.estimate(first, second)

        check(!result.accepted)
        check(result.confidence == 0.0)
    }

    private fun samples(
        timestampShiftNanos: Long = 0L,
        phaseShiftNanos: Long = 0L,
        count: Int = 30,
        noiseScale: Double = 0.0,
    ): List<SubframeMotionAlignmentEngine.Sample> {
        val baseTimestampNanos = 100_000_000L
        return (0 until count).map { index ->
            val timestamp =
                baseTimestampNanos +
                    index.toLong() * 16_666_667L +
                    timestampShiftNanos
            val phaseSeconds =
                (
                    index.toLong() * 16_666_667L +
                        phaseShiftNanos
                    ) / 1e9
            val noise =
                if (noiseScale == 0.0) {
                    0.0
                } else {
                    kotlin.math.sin(
                        index * 17.0,
                    ) * noiseScale
                }
            SubframeMotionAlignmentEngine.Sample(
                timestampNanos = timestamp,
                value =
                    kotlin.math.sin(
                        phaseSeconds * 10.0,
                    ) +
                        0.35 *
                        kotlin.math.sin(
                            phaseSeconds * 3.0,
                        ) +
                        noise,
            )
        }
    }
}

fun main() {
    val test = SubframeMotionAlignmentEngineTest()
    test.recovers_zero_offset()
    test.recovers_signed_positive_delay()
    test.remains_stable_with_noise()
    test.rejects_insufficient_evidence()
}
