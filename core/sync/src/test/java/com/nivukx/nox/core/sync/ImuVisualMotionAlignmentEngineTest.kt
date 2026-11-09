package com.nivukx.nox.core.sync

class ImuVisualMotionAlignmentEngineTest {

    private val engine =
        ImuVisualMotionAlignmentEngine(
            ImuVisualMotionAlignmentEngine.Config(
                minimumSamples = 16,
                maximumOffsetNanos = 30_000_000L,
                searchStepNanos = 1_000_000L,
            ),
        )

    fun recovers_zero_offset() {
        val imu = samples(0L)
        val visual = samples(0L, phaseScale = 1.0)

        val result = engine.estimate(imu, visual)

        check(result.accepted)
        check(kotlin.math.abs(result.offsetNanos) < 2_000_000.0)
        check(result.confidence > 0.55)
    }

    fun recovers_signed_positive_visual_delay() {
        val imu = samples(0L)
        val visual =
            samples(
                timestampShiftNanos = 9_000_000L,
                phaseScale = 1.0,
            )

        val result = engine.estimate(imu, visual)

        check(result.accepted)
        check(result.offsetNanos > 6_000_000.0)
        check(result.offsetNanos < 12_000_000.0)
    }

    fun rejects_flat_motion_without_observability() {
        val flat =
            (0 until 24).map { index ->
                ImuVisualMotionAlignmentEngine.Sample(
                    timestampNanos =
                        1_000_000_000L +
                            index * 10_000_000L,
                    angularSpeedRadiansPerSecond = 0.0,
                )
            }

        val result = engine.estimate(flat, flat)

        check(!result.accepted)
        check(result.confidence == 0.0)
        check(
            result.warnings.any {
                it.contains(
                    "energy",
                    ignoreCase = true,
                )
            },
        )
    }

    fun remains_robust_with_noise() {
        val imu =
            samples(
                0L,
                phaseScale = 1.0,
                noiseScale = 0.04,
            )
        val visual =
            samples(
                -6_000_000L,
                phaseScale = 1.0,
                noiseScale = 0.04,
            )

        val result = engine.estimate(imu, visual)

        check(result.accepted)
        check(result.confidence > 0.35)
        check(
            kotlin.math.abs(
                result.offsetNanos,
            ) > 2_000_000.0,
        )
    }

    private fun samples(
        timestampShiftNanos: Long,
        phaseScale: Double = 1.0,
        noiseScale: Double = 0.0,
        count: Int = 36,
    ): List<ImuVisualMotionAlignmentEngine.Sample> =
        (0 until count).map { index ->
            val t =
                1_000_000_000L +
                    index * 10_000_000L +
                    timestampShiftNanos
            val seconds =
                index * 0.010 * phaseScale
            val noise =
                kotlin.math.sin(index * 11.0) *
                    noiseScale
            ImuVisualMotionAlignmentEngine.Sample(
                timestampNanos = t,
                angularSpeedRadiansPerSecond =
                    0.6 +
                        1.2 *
                        kotlin.math.sin(
                            seconds * 5.5,
                        ) +
                        0.5 *
                        kotlin.math.sin(
                            seconds * 2.0,
                        ) +
                        noise,
            )
        }
}

fun main() {
    val test = ImuVisualMotionAlignmentEngineTest()
    test.recovers_zero_offset()
    test.recovers_signed_positive_visual_delay()
    test.rejects_flat_motion_without_observability()
    test.remains_robust_with_noise()
}
