package com.nivukx.nox.core.capture

import kotlin.math.abs
import kotlin.math.max

/**
 * Production-grade capture-clock integrity monitor.
 *
 * Camera2 exposes a sensor timestamp for each capture. NoX treats that timestamp as first-class
 * evidence rather than using wall-clock arrival time as the frame clock. This engine evaluates
 * cadence, monotonicity, jitter, gaps, duplicate timestamps, sensor/host drift, exposure overlap,
 * and rolling-shutter pressure without modifying the samples it observes.
 *
 * The class is platform-independent so it can be exercised with deterministic unit tests and reused
 * by offline session validation. It is stateful and synchronized because capture callbacks can be
 * delivered on a dedicated camera thread while diagnostics are read by the UI.
 */
class CaptureTimestampIntegrityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val expectedFrameIntervalNanos: Long = 33_333_333L,
        val minimumFrameIntervalNanos: Long = 2_000_000L,
        val maximumFrameIntervalNanos: Long = 500_000_000L,
        val historyWindow: Int = 120,
        val warmupSamples: Int = 8,
        val jitterTargetFraction: Double = 0.035,
        val jitterHardFraction: Double = 0.20,
        val gapWarningMultiplier: Double = 1.60,
        val gapCriticalMultiplier: Double = 3.50,
        val maximumClockDriftPpm: Double = 250.0,
        val maximumRollingShutterFraction: Double = 0.90,
        val maximumExposureDutyCycle: Double = 0.98,
        val maximumConsecutiveInvalidSamples: Int = 4,
    ) {
        init {
            require(expectedFrameIntervalNanos > 0L)
            require(minimumFrameIntervalNanos > 0L)
            require(maximumFrameIntervalNanos >= minimumFrameIntervalNanos)
            require(historyWindow >= 8)
            require(warmupSamples >= 2)
            require(jitterTargetFraction > 0.0)
            require(jitterHardFraction > jitterTargetFraction)
            require(gapWarningMultiplier > 1.0)
            require(gapCriticalMultiplier > gapWarningMultiplier)
            require(maximumClockDriftPpm > 0.0)
            require(maximumRollingShutterFraction > 0.0)
            require(maximumExposureDutyCycle in 0.0..1.0)
            require(maximumConsecutiveInvalidSamples in 1..64)
        }
    }

    enum class Verdict {
        WARMING,
        VALID,
        JITTER,
        GAP,
        DUPLICATE,
        OUT_OF_ORDER,
        HOST_CLOCK_DRIFT,
        ROLLING_SHUTTER_PRESSURE,
        EXPOSURE_OVERLAP,
        INVALID,
    }

    data class Sample(
        val sensorTimestampNanos: Long,
        val monotonicTimestampNanos: Long,
        val frameDurationNanos: Long,
        val exposureTimeNanos: Long,
        val rollingShutterSkewNanos: Long,
    ) {
        init {
            require(sensorTimestampNanos >= 0L)
            require(monotonicTimestampNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            require(rollingShutterSkewNanos >= 0L)
        }
    }

    data class Assessment(
        val sensorTimestampNanos: Long,
        val deltaNanos: Long?,
        val hostDeltaNanos: Long?,
        val intervalErrorFraction: Double,
        val jitterFraction: Double,
        val rollingShutterFraction: Double,
        val exposureDutyCycle: Double,
        val driftPpm: Double,
        val cadenceFps: Double,
        val verdict: Verdict,
        val warnings: List<String>,
        val confidence: Double,
        val acceptedForCadence: Boolean = false,
        val consecutiveInvalidSamples: Int = 0,
    ) {
        val healthy: Boolean
            get() = verdict == Verdict.VALID || verdict == Verdict.WARMING
    }

    data class Quality(
        val sampleCount: Int,
        val validSampleCount: Int,
        val duplicateCount: Int,
        val outOfOrderCount: Int,
        val hostOutOfOrderCount: Int,
        val invalidIntervalCount: Int,
        val gapCount: Int,
        val largestGapNanos: Long,
        val meanIntervalNanos: Double,
        val medianIntervalNanos: Double,
        val p95AbsoluteJitterNanos: Long,
        val meanAbsoluteJitterNanos: Double,
        val cadenceFps: Double,
        val hostCadenceFps: Double,
        val driftPpm: Double,
        val rollingShutterFraction: Double,
        val exposureDutyCycle: Double,
        val confidence: Double,
        val healthy: Boolean,
        val warnings: List<String>,
    )

    private var expectedFrameIntervalNanos = config.expectedFrameIntervalNanos

    private val sensorIntervals = ArrayDeque<Long>()
    private val hostIntervals = ArrayDeque<Long>()
    private val absoluteJitters = ArrayDeque<Long>()
    private val rollingFractions = ArrayDeque<Double>()
    private val exposureDuties = ArrayDeque<Double>()

    private var previous: Sample? = null
    private var firstSensorTimestamp = Long.MIN_VALUE
    private var lastSensorTimestamp = Long.MIN_VALUE
    private var firstHostTimestamp = Long.MIN_VALUE
    private var lastHostTimestamp = Long.MIN_VALUE

    private var sampleCount = 0
    private var validSampleCount = 0
    private var duplicateCount = 0
    private var outOfOrderCount = 0
    private var hostOutOfOrderCount = 0
    private var invalidIntervalCount = 0
    private var consecutiveInvalidSamples = 0
    private var gapCount = 0
    private var largestGapNanos = 0L

    @Synchronized
    fun observe(sample: Sample): Assessment {
        sampleCount += 1

        if (firstSensorTimestamp == Long.MIN_VALUE) {
            firstSensorTimestamp = sample.sensorTimestampNanos
            lastSensorTimestamp = sample.sensorTimestampNanos
            firstHostTimestamp = sample.monotonicTimestampNanos
            lastHostTimestamp = sample.monotonicTimestampNanos
            previous = sample

            return Assessment(
                sensorTimestampNanos = sample.sensorTimestampNanos,
                deltaNanos = null,
                hostDeltaNanos = null,
                intervalErrorFraction = 0.0,
                jitterFraction = 0.0,
                rollingShutterFraction = rollingFraction(sample),
                exposureDutyCycle = exposureDuty(sample),
                driftPpm = 0.0,
                cadenceFps = 0.0,
                verdict = Verdict.WARMING,
                warnings = listOf("capture timing monitor warming up"),
                confidence = 0.0,
                acceptedForCadence = false,
                consecutiveInvalidSamples = consecutiveInvalidSamples,
            )
        }

        val previousSample = previous
            ?: return invalidAssessment(
                sample,
                "capture timing state was unexpectedly empty",
            )

        val delta = sample.sensorTimestampNanos -
            previousSample.sensorTimestampNanos
        val hostDelta = sample.monotonicTimestampNanos -
            previousSample.monotonicTimestampNanos

        val rollingFraction = rollingFraction(sample)
        val exposureDuty = exposureDuty(sample)
        val intervalError = intervalErrorFraction(delta)
        val jitter = intervalJitterFraction(delta)
        val driftPpm = driftPpm(
            sensorDelta = delta,
            hostDelta = hostDelta,
        )

        var verdict = Verdict.VALID
        val warnings = ArrayList<String>(6)
        var acceptedForCadence = false
        var updatePrevious = true

        when {
            delta == 0L -> {
                duplicateCount += 1
                invalidIntervalCount += 1
                consecutiveInvalidSamples += 1
                verdict = Verdict.DUPLICATE
                warnings += "duplicate sensor timestamp"
                updatePrevious = false
            }

            delta < 0L -> {
                outOfOrderCount += 1
                invalidIntervalCount += 1
                consecutiveInvalidSamples += 1
                verdict = Verdict.OUT_OF_ORDER
                warnings += "sensor timestamps moved backwards"
                updatePrevious = false
            }

            hostDelta <= 0L -> {
                hostOutOfOrderCount += 1
                invalidIntervalCount += 1
                consecutiveInvalidSamples += 1
                verdict = Verdict.INVALID
                warnings += "host timestamps moved backwards"
                updatePrevious = false
            }

            delta < config.minimumFrameIntervalNanos -> {
                invalidIntervalCount += 1
                consecutiveInvalidSamples += 1
                verdict = Verdict.INVALID
                warnings += "frame interval below supported timing envelope"
            }

            delta > config.maximumFrameIntervalNanos -> {
                invalidIntervalCount += 1
                consecutiveInvalidSamples += 1
                gapCount += 1
                largestGapNanos = max(largestGapNanos, delta)
                verdict = Verdict.GAP
                warnings += "frame cadence gap exceeds maximum interval"
            }

            else -> {
                validSampleCount += 1
                acceptedForCadence = true
                consecutiveInvalidSamples = 0
                push(sensorIntervals, delta)
                push(
                    absoluteJitters,
                    abs(delta - expectedFrameIntervalNanos),
                )
                largestGapNanos = max(
                    largestGapNanos,
                    delta,
                )

                if (
                    delta.toDouble() >
                    expectedFrameIntervalNanos.toDouble() *
                        config.gapWarningMultiplier
                ) {
                    gapCount += 1
                    warnings += "frame cadence gap detected"
                    verdict = Verdict.GAP
                }
            }
        }

        if (hostDelta > 0L) {
            push(hostIntervals, hostDelta)
        }
        push(rollingFractions, rollingFraction)
        push(exposureDuties, exposureDuty)

        if (
            driftPpm.isFinite() &&
            abs(driftPpm) > config.maximumClockDriftPpm &&
            verdict == Verdict.VALID
        ) {
            verdict = Verdict.HOST_CLOCK_DRIFT
            warnings += "sensor and host capture clocks are diverging"
        }

        if (
            rollingFraction > config.maximumRollingShutterFraction &&
            verdict == Verdict.VALID
        ) {
            verdict = Verdict.ROLLING_SHUTTER_PRESSURE
            warnings += "rolling-shutter readout occupies most of the frame interval"
        }

        if (
            exposureDuty > config.maximumExposureDutyCycle &&
            verdict == Verdict.VALID
        ) {
            verdict = Verdict.EXPOSURE_OVERLAP
            warnings += "exposure duration approaches frame-duration limit"
        }

        if (
            jitter > 1.0 &&
            verdict == Verdict.VALID
        ) {
            verdict = Verdict.JITTER
            warnings += "capture cadence jitter exceeds production target"
        }

        if (
            consecutiveInvalidSamples >=
                config.maximumConsecutiveInvalidSamples
        ) {
            warnings += "consecutive timing integrity failures exceed runtime recovery budget"
        }

        if (
            sampleCount < config.warmupSamples &&
            verdict == Verdict.VALID
        ) {
            verdict = Verdict.WARMING
            warnings += "capture timing monitor warming up"
        }

        val cadence = fps(sensorIntervals)
        val hostCadence = fps(hostIntervals)
        val confidence = assessmentConfidence(
            intervalError = intervalError,
            jitter = jitter,
            driftPpm = driftPpm,
            rollingFraction = rollingFraction,
            exposureDuty = exposureDuty,
            verdict = verdict,
        )

        if (updatePrevious) {
            previous = sample
            lastSensorTimestamp = sample.sensorTimestampNanos
            lastHostTimestamp = sample.monotonicTimestampNanos
        }

        return Assessment(
            sensorTimestampNanos = sample.sensorTimestampNanos,
            deltaNanos = delta,
            hostDeltaNanos = hostDelta,
            intervalErrorFraction = intervalError,
            jitterFraction = jitter,
            rollingShutterFraction = rollingFraction,
            exposureDutyCycle = exposureDuty,
            driftPpm = driftPpm,
            cadenceFps = cadence,
            verdict = verdict,
            warnings = warnings.distinct(),
            confidence = confidence,
            acceptedForCadence = acceptedForCadence,
            consecutiveInvalidSamples = consecutiveInvalidSamples,
        )
    }

    fun configureFrameRate(fps: Int) {
        require(fps in 1..240)
        synchronized(this) {
            expectedFrameIntervalNanos =
                (1_000_000_000.0 / fps.toDouble()).toLong().coerceAtLeast(1L)
            reset()
        }
    }

    @Synchronized
    fun quality(): Quality {
        val sensorMean = meanLong(sensorIntervals)
        val hostMean = meanLong(hostIntervals)
        val meanJitter = meanLong(absoluteJitters)

        val drift = aggregateDriftPpm(
            sensorStart = firstSensorTimestamp,
            sensorEnd = lastSensorTimestamp,
            hostStart = firstHostTimestamp,
            hostEnd = lastHostTimestamp,
        )

        val rolling = meanDouble(rollingFractions)
        val exposure = meanDouble(exposureDuties)
        val jitterP95 = percentile(absoluteJitters, 0.95)

        val cadence = if (sensorMean > 0.0) {
            1_000_000_000.0 / sensorMean
        } else {
            0.0
        }

        val hostCadence = if (hostMean > 0.0) {
            1_000_000_000.0 / hostMean
        } else {
            0.0
        }

        val confidence = qualityConfidence(
            sampleCount = sampleCount,
            meanInterval = sensorMean,
            meanJitter = meanJitter,
            driftPpm = drift,
            rollingFraction = rolling,
            exposureDuty = exposure,
        )

        val warnings = buildList {
            if (sampleCount < config.warmupSamples) {
                add("timing sample count is below production warm-up")
            }
            if (duplicateCount > 0) {
                add("$duplicateCount duplicate sensor timestamp(s)")
            }
            if (outOfOrderCount > 0) {
                add("$outOfOrderCount out-of-order sensor timestamp(s)")
            }
            if (hostOutOfOrderCount > 0) {
                add("$hostOutOfOrderCount out-of-order host timestamp(s)")
            }
            if (gapCount > 0) {
                add("$gapCount timing gap(s) exceeded the warning threshold")
            }
            if (
                jitterP95 >
                    (expectedFrameIntervalNanos *
                        config.jitterHardFraction).toLong()
            ) {
                add("95th-percentile cadence jitter exceeds hard target")
            }
            if (abs(drift) > config.maximumClockDriftPpm) {
                add("sensor/host clock drift exceeds configured envelope")
            }
            if (rolling > config.maximumRollingShutterFraction) {
                add("rolling-shutter fraction is high")
            }
            if (exposure > config.maximumExposureDutyCycle) {
                add("mean exposure duty cycle is high")
            }
            if (consecutiveInvalidSamples >= config.maximumConsecutiveInvalidSamples) {
                add("consecutive timing integrity failures exceed runtime recovery budget")
            }
        }

        val healthy = sampleCount >= config.warmupSamples &&
            duplicateCount == 0 &&
            outOfOrderCount == 0 &&
            hostOutOfOrderCount == 0 &&
            invalidIntervalCount == 0 &&
            gapCount == 0 &&
            jitterP95 <=
                (expectedFrameIntervalNanos *
                    config.jitterHardFraction).toLong() &&
            abs(drift) <= config.maximumClockDriftPpm &&
            rolling <= config.maximumRollingShutterFraction &&
            exposure <= config.maximumExposureDutyCycle

        return Quality(
            sampleCount = sampleCount,
            validSampleCount = validSampleCount,
            duplicateCount = duplicateCount,
            outOfOrderCount = outOfOrderCount,
            hostOutOfOrderCount = hostOutOfOrderCount,
            invalidIntervalCount = invalidIntervalCount,
            gapCount = gapCount,
            largestGapNanos = largestGapNanos,
            meanIntervalNanos = sensorMean,
            medianIntervalNanos = median(sensorIntervals),
            p95AbsoluteJitterNanos = jitterP95,
            meanAbsoluteJitterNanos = meanJitter,
            cadenceFps = cadence,
            hostCadenceFps = hostCadence,
            driftPpm = drift,
            rollingShutterFraction = rolling,
            exposureDutyCycle = exposure,
            confidence = confidence,
            healthy = healthy,
            warnings = warnings,
        )
    }

    @Synchronized
    fun reset() {
        sensorIntervals.clear()
        hostIntervals.clear()
        absoluteJitters.clear()
        rollingFractions.clear()
        exposureDuties.clear()

        previous = null
        firstSensorTimestamp = Long.MIN_VALUE
        lastSensorTimestamp = Long.MIN_VALUE
        firstHostTimestamp = Long.MIN_VALUE
        lastHostTimestamp = Long.MIN_VALUE

        sampleCount = 0
        validSampleCount = 0
        duplicateCount = 0
        outOfOrderCount = 0
        hostOutOfOrderCount = 0
        invalidIntervalCount = 0
        consecutiveInvalidSamples = 0
        gapCount = 0
        largestGapNanos = 0L
    }

    private fun rollingFraction(sample: Sample): Double {
        if (sample.rollingShutterSkewNanos <= 0L) return 0.0

        val denominator = if (sample.frameDurationNanos > 0L) {
            sample.frameDurationNanos
        } else {
            expectedFrameIntervalNanos
        }

        return (
            sample.rollingShutterSkewNanos.toDouble() /
                denominator.toDouble()
            ).coerceIn(0.0, 2.0)
    }

    private fun exposureDuty(sample: Sample): Double {
        if (sample.exposureTimeNanos <= 0L) return 0.0

        val denominator = if (sample.frameDurationNanos > 0L) {
            sample.frameDurationNanos
        } else {
            expectedFrameIntervalNanos
        }

        return (
            sample.exposureTimeNanos.toDouble() /
                denominator.toDouble()
            ).coerceIn(0.0, 2.0)
    }

    private fun intervalErrorFraction(delta: Long): Double {
        if (delta <= 0L) return 1.0

        return abs(
            delta.toDouble() -
                expectedFrameIntervalNanos.toDouble(),
        ) /
            expectedFrameIntervalNanos.toDouble()
                .coerceAtLeast(1.0)
    }

    private fun intervalJitterFraction(delta: Long): Double {
        if (delta <= 0L) return 1.0

        val centered = abs(
            delta -
                expectedFrameIntervalNanos,
        )

        val target = (
            expectedFrameIntervalNanos.toDouble() *
                config.jitterTargetFraction
        ).coerceAtLeast(1.0)

        return (
            centered.toDouble() / target
        ).coerceIn(0.0, 8.0)
    }

    private fun driftPpm(
        sensorDelta: Long,
        hostDelta: Long,
    ): Double {
        if (sensorDelta <= 0L || hostDelta <= 0L) return 0.0

        return (
            sensorDelta.toDouble() -
                hostDelta.toDouble()
            ) /
                hostDelta.toDouble().coerceAtLeast(1.0) *
                1_000_000.0
    }

    private fun aggregateDriftPpm(
        sensorStart: Long,
        sensorEnd: Long,
        hostStart: Long,
        hostEnd: Long,
    ): Double {
        if (
            sensorStart == Long.MIN_VALUE ||
            sensorEnd == Long.MIN_VALUE ||
            hostStart == Long.MIN_VALUE ||
            hostEnd == Long.MIN_VALUE
        ) {
            return 0.0
        }

        val sensorSpan = sensorEnd - sensorStart
        val hostSpan = hostEnd - hostStart

        if (sensorSpan <= 0L || hostSpan <= 0L) return 0.0

        return (
            sensorSpan.toDouble() -
                hostSpan.toDouble()
            ) /
                hostSpan.toDouble() *
                1_000_000.0
    }

    private fun assessmentConfidence(
        intervalError: Double,
        jitter: Double,
        driftPpm: Double,
        rollingFraction: Double,
        exposureDuty: Double,
        verdict: Verdict,
    ): Double {
        val cadenceScore = (
            1.0 -
                intervalError.coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)

        val jitterScore = (
            1.0 -
                jitter.coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)

        val driftScore = (
            1.0 -
                abs(driftPpm) /
                    config.maximumClockDriftPpm
            ).coerceIn(0.0, 1.0)

        val rollingScore = (
            1.0 -
                rollingFraction /
                    config.maximumRollingShutterFraction
            ).coerceIn(0.0, 1.0)

        val exposureScore = (
            1.0 -
                exposureDuty /
                    config.maximumExposureDutyCycle.coerceAtLeast(0.01)
            ).coerceIn(0.0, 1.0)

        val verdictPenalty = when (verdict) {
            Verdict.VALID,
            Verdict.WARMING -> 1.0
            Verdict.JITTER -> 0.72
            Verdict.GAP -> 0.60
            Verdict.HOST_CLOCK_DRIFT -> 0.55
            Verdict.ROLLING_SHUTTER_PRESSURE -> 0.68
            Verdict.EXPOSURE_OVERLAP -> 0.66
            Verdict.DUPLICATE,
            Verdict.OUT_OF_ORDER,
            Verdict.INVALID -> 0.15
        }

        return (
            cadenceScore * 0.26 +
                jitterScore * 0.24 +
                driftScore * 0.16 +
                rollingScore * 0.16 +
                exposureScore * 0.18
            ).coerceIn(0.0, 1.0) *
            verdictPenalty
    }

    private fun qualityConfidence(
        sampleCount: Int,
        meanInterval: Double,
        meanJitter: Double,
        driftPpm: Double,
        rollingFraction: Double,
        exposureDuty: Double,
    ): Double {
        val warmup = (
            sampleCount.toDouble() /
                config.warmupSamples.toDouble()
            ).coerceIn(0.0, 1.0)

        val cadenceError = if (meanInterval <= 0.0) {
            1.0
        } else {
            abs(
                meanInterval -
                    expectedFrameIntervalNanos,
            ) /
                expectedFrameIntervalNanos.toDouble()
        }

        val cadenceScore = (
            1.0 -
                cadenceError
        ).coerceIn(0.0, 1.0)

        val jitterScore = (
            1.0 -
                meanJitter /
                    (
                        expectedFrameIntervalNanos.toDouble() *
                            config.jitterHardFraction
                    ).coerceAtLeast(1.0)
        ).coerceIn(0.0, 1.0)

        val driftScore = (
            1.0 -
                abs(driftPpm) /
                    config.maximumClockDriftPpm
        ).coerceIn(0.0, 1.0)

        val rollingScore = (
            1.0 -
                rollingFraction /
                    config.maximumRollingShutterFraction
        ).coerceIn(0.0, 1.0)

        val exposureScore = (
            1.0 -
                exposureDuty /
                    config.maximumExposureDutyCycle.coerceAtLeast(0.01)
        ).coerceIn(0.0, 1.0)

        return (
            warmup * 0.16 +
                cadenceScore * 0.20 +
                jitterScore * 0.24 +
                driftScore * 0.14 +
                rollingScore * 0.12 +
                exposureScore * 0.14
            ).coerceIn(0.0, 1.0)
    }

    private fun push(
        deque: ArrayDeque<Long>,
        value: Long,
    ) {
        deque.addLast(value)
        while (deque.size > config.historyWindow) {
            deque.removeFirst()
        }
    }

    private fun push(
        deque: ArrayDeque<Double>,
        value: Double,
    ) {
        deque.addLast(value)
        while (deque.size > config.historyWindow) {
            deque.removeFirst()
        }
    }

    private fun meanLong(values: ArrayDeque<Long>): Double =
        if (values.isEmpty()) 0.0
        else values.sumOf { it.toDouble() } / values.size.toDouble()

    private fun meanDouble(values: ArrayDeque<Double>): Double =
        if (values.isEmpty()) 0.0
        else values.sum() / values.size.toDouble()

    private fun median(values: ArrayDeque<Long>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        return if (sorted.size % 2 == 0) {
            (
                sorted[sorted.size / 2 - 1] +
                    sorted[sorted.size / 2]
            ) / 2.0
        } else {
            sorted[sorted.size / 2].toDouble()
        }
    }

    private fun percentile(
        values: ArrayDeque<Long>,
        fraction: Double,
    ): Long {
        if (values.isEmpty()) return 0L

        val sorted = values.sorted()
        val index = (
            (sorted.lastIndex * fraction)
                .coerceIn(0.0, sorted.lastIndex.toDouble())
            ).toInt()

        return sorted[index]
    }

    private fun fps(values: ArrayDeque<Long>): Double {
        val interval = meanLong(values)
        return if (interval <= 0.0) {
            0.0
        } else {
            1_000_000_000.0 / interval
        }
    }

    private fun invalidAssessment(
        sample: Sample,
        warning: String,
    ): Assessment =
        Assessment(
            sensorTimestampNanos = sample.sensorTimestampNanos,
            deltaNanos = null,
            hostDeltaNanos = null,
            intervalErrorFraction = 1.0,
            jitterFraction = 1.0,
            rollingShutterFraction = rollingFraction(sample),
            exposureDutyCycle = exposureDuty(sample),
            driftPpm = 0.0,
            cadenceFps = fps(sensorIntervals),
            verdict = Verdict.INVALID,
            warnings = listOf(warning),
            confidence = 0.0,
        )
}
