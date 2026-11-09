
package com.nivukx.nox.core.sync

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Scores synchronization quality for distributed motion capture.
 *
 * Motion reconstruction is highly sensitive to timestamp skew during fast movement. This engine
 * treats clock offset, round-trip time, packet loss, jitter and timestamp continuity as separate
 * evidence channels and exposes a single bounded synchronization score plus actionable remedies.
 */
class CameraSynchronizationQualityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val preferredSkewMillis: Double = 0.75,
        val maximumSkewMillis: Double = 8.0,
        val preferredRoundTripMillis: Double = 20.0,
        val maximumRoundTripMillis: Double = 120.0,
        val preferredPacketLoss: Double = 0.005,
        val maximumPacketLoss: Double = 0.08,
        val preferredJitterMillis: Double = 0.35,
        val maximumJitterMillis: Double = 5.0,
        val minimumSamples: Int = 8,
        val fastMotionPenalty: Double = 0.18,
        val timestampAuthorityWeight: Double = 0.10,
    ) {
        init {
            require(preferredSkewMillis > 0.0)
            require(maximumSkewMillis > preferredSkewMillis)
            require(preferredRoundTripMillis > 0.0)
            require(maximumRoundTripMillis > preferredRoundTripMillis)
            require(preferredPacketLoss >= 0.0)
            require(maximumPacketLoss > preferredPacketLoss)
            require(preferredJitterMillis > 0.0)
            require(maximumJitterMillis > preferredJitterMillis)
            require(minimumSamples >= 1)
            require(fastMotionPenalty in 0.0..1.0)
            require(timestampAuthorityWeight in 0.0..1.0)
        }
    }

    data class CameraSample(
        val cameraId: String,
        val timestampNanos: Long,
        val clockOffsetNanos: Long,
        val roundTripNanos: Long,
        val packetLoss: Double,
        val jitterNanos: Long,
        val timestampDomain: MultiCameraTemporalAlignmentEngine.TimestampDomain =
            MultiCameraTemporalAlignmentEngine.TimestampDomain.MONOTONIC,
        val hardwareSynchronized: Boolean = false,
        val sourceComparableAcrossDevices: Boolean = true,
        val hardwareReadoutTimingConfidence: Double = 0.0,
    ) {
        init {
            require(packetLoss in 0.0..1.0)
            require(roundTripNanos >= 0L)
            require(jitterNanos >= 0L)
            require(hardwareReadoutTimingConfidence in 0.0..1.0)
        }
    }

    data class CameraResult(
        val cameraId: String,
        val sampleCount: Int,
        val score: Double,
        val skewScore: Double,
        val roundTripScore: Double,
        val packetLossScore: Double,
        val jitterScore: Double,
        val timestampAuthorityScore: Double,
        val stable: Boolean,
        val diagnostics: List<String>,
    )

    data class Result(
        val ready: Boolean,
        val rigScore: Double,
        val worstCameraId: String?,
        val cameras: List<CameraResult>,
        val clockGraphConfidence: Double? = null,
        val clockGraphCoverage: Double? = null,
        val clockGraphCycleConsistency: Double? = null,
        val clockGraphAccepted: Boolean? = null,
        val recommendations: List<String>,
        val diagnostics: List<String>,
    )

    fun evaluate(
        samples: List<CameraSample>,
        fastMotion: Boolean = false,
        clockGraph: DistributedClockAuthorityEngine.Result? = null,
    ): Result {
        if (samples.isEmpty()) {
            return Result(
                ready = false,
                rigScore = 0.0,
                worstCameraId = null,
                cameras = emptyList(),
                recommendations = listOf(
                    "No synchronization samples available.",
                ),
                diagnostics = listOf(
                    "sampleCount=0",
                ),
            )
        }

        val grouped = samples.groupBy { it.cameraId }

        val cameras = grouped.map { (cameraId, cameraSamples) ->
            val skew = cameraSamples.map {
                abs(it.clockOffsetNanos).toDouble() / 1e6
            }.average()

            val roundTrip = cameraSamples
                .map {
                    it.roundTripNanos.toDouble() / 1e6
                }
                .average()

            val loss = cameraSamples
                .map { it.packetLoss.coerceIn(0.0, 1.0) }
                .average()

            val jitter = cameraSamples
                .map { it.jitterNanos.toDouble() / 1e6 }
                .average()

            val skewScore = quality(
                skew,
                config.preferredSkewMillis,
                config.maximumSkewMillis,
            )
            val roundTripScore = quality(
                roundTrip,
                config.preferredRoundTripMillis,
                config.maximumRoundTripMillis,
            )
            val packetLossScore = quality(
                loss,
                config.preferredPacketLoss,
                config.maximumPacketLoss,
            )
            val jitterScore = quality(
                jitter,
                config.preferredJitterMillis,
                config.maximumJitterMillis,
            )

            val timestampAuthorityScore =
                timestampAuthorityScore(
                    cameraSamples,
                )

            val sampleScore = (
                cameraSamples.size.toDouble() /
                    config.minimumSamples.toDouble()
                ).coerceIn(0.25, 1.0)

            val baseWeight =
                (
                    config.timestampAuthorityWeight
                ).coerceIn(0.0, 1.0)

            val transportWeight =
                (
                    1.0 -
                        baseWeight
                ).coerceAtLeast(0.0)

            val transportScore = (
                skewScore * 0.34 +
                    roundTripScore * 0.18 +
                    packetLossScore * 0.20 +
                    jitterScore * 0.18 +
                    sampleScore * 0.10
                ).coerceIn(0.0, 1.0)

            val score = (
                transportScore * transportWeight +
                    timestampAuthorityScore * baseWeight
                ).coerceIn(0.0, 1.0)

            val stable =
                cameraSamples.size >= config.minimumSamples &&
                    score >= 0.58

            CameraResult(
                cameraId = cameraId,
                sampleCount = cameraSamples.size,
                score = score,
                skewScore = skewScore,
                roundTripScore = roundTripScore,
                packetLossScore = packetLossScore,
                jitterScore = jitterScore,
                timestampAuthorityScore =
                    timestampAuthorityScore,
                stable = stable,
                diagnostics = listOf(
                    "skewMs=" + format(skew),
                    "roundTripMs=" + format(roundTrip),
                    "packetLoss=" + format(loss),
                    "jitterMs=" + format(jitter),
                    "timestampAuthority=" +
                        format(timestampAuthorityScore),
                    "sampleCount=" + cameraSamples.size,
                ),
            )
        }.sortedBy { it.cameraId }

        val rawRigScore = cameras
            .map { it.score }
            .average()

        val motionAdjustedRigScore = if (fastMotion) {
            (
                rawRigScore *
                    (1.0 - config.fastMotionPenalty *
                        cameras.count { !it.stable }
                            .toDouble() /
                        cameras.size.toDouble().coerceAtLeast(1.0))
            ).coerceIn(0.0, 1.0)
        } else {
            rawRigScore
        }

        val graphScore =
            clockGraph?.confidence?.coerceIn(0.0, 1.0)
                ?: motionAdjustedRigScore

        val rigScore =
            if (clockGraph != null) {
                (
                    motionAdjustedRigScore * 0.72 +
                        graphScore * 0.28
                ).coerceIn(0.0, 1.0)
            } else {
                motionAdjustedRigScore
            }

        val ready =
            cameras.size >= 2 &&
                cameras.all { it.stable } &&
                rigScore >= 0.65 &&
                (
                    clockGraph == null ||
                        (
                            clockGraph.accepted &&
                                clockGraph.cycleConsistency >= 0.85 &&
                                clockGraph.graphCoverage >= 0.90
                        )
                )

        val recommendations = buildList {
            if (cameras.size < 2) {
                add("Connect at least two synchronized capture devices.")
            }
            if (cameras.any { it.skewScore < 0.55 }) {
                add("Re-estimate peer clock offsets before fast-motion capture.")
            }
            if (cameras.any { it.jitterScore < 0.55 }) {
                add("Reduce network jitter or use a lower-latency transport path.")
            }
            if (cameras.any { it.packetLossScore < 0.55 }) {
                add("Resolve packet loss before depending on distributed geometry.")
            }
            if (fastMotion && !ready) {
                add("Fast-motion capture requires tighter synchronization than static capture.")
            }
            if (
                clockGraph != null &&
                !clockGraph.accepted
            ) {
                add(
                    "Multi-master clock graph is not reference-consistent; do not use its offsets as authoritative.",
                )
            }
        }

        return Result(
            ready = ready,
            rigScore = rigScore,
            worstCameraId = cameras.minByOrNull { it.score }?.cameraId,
            cameras = cameras,
            clockGraphConfidence = clockGraph?.confidence,
            clockGraphCoverage = clockGraph?.graphCoverage,
            clockGraphCycleConsistency =
                clockGraph?.cycleConsistency,
            clockGraphAccepted = clockGraph?.accepted,
            recommendations = recommendations.distinct(),
            diagnostics =
                buildList {
                    add("cameras=" + cameras.size)
                    add("rigScore=" + format(rigScore))
                    add("fastMotion=" + fastMotion)
                    add("ready=" + ready)
                    clockGraph?.let {
                        add(
                            "clockGraphConfidence=" +
                                format(it.confidence),
                        )
                        add(
                            "clockGraphCoverage=" +
                                format(it.graphCoverage),
                        )
                        add(
                            "clockGraphCycleConsistency=" +
                                format(it.cycleConsistency),
                        )
                        add(
                            "clockGraphAccepted=" +
                                it.accepted,
                        )
                    }
                },
        )
    }

    private fun timestampAuthorityScore(
        samples: List<CameraSample>,
    ): Double {
        if (samples.isEmpty()) return 0.0

        val domainScores =
            samples.map {
                when (it.timestampDomain) {
                    MultiCameraTemporalAlignmentEngine.TimestampDomain.SENSOR ->
                        if (it.hardwareSynchronized) 1.0 else 0.84
                    MultiCameraTemporalAlignmentEngine.TimestampDomain.REALTIME ->
                        0.72
                    MultiCameraTemporalAlignmentEngine.TimestampDomain.MONOTONIC ->
                        if (it.sourceComparableAcrossDevices) 0.78 else 0.52
                    MultiCameraTemporalAlignmentEngine.TimestampDomain.UNKNOWN ->
                        0.20
                }
            }.average()

        val hardwareScore =
            samples.map {
                if (it.hardwareSynchronized) {
                    1.0
                } else {
                    it.hardwareReadoutTimingConfidence
                        .coerceIn(0.0, 1.0)
                        .let { confidence ->
                            if (confidence > 0.0) {
                                0.70 + 0.30 * confidence
                            } else {
                                0.70
                            }
                        }
                }
            }.average()

        return (
            domainScores * 0.62 +
                hardwareScore * 0.38
            ).coerceIn(0.0, 1.0)
    }

    private fun quality(
        value: Double,
        preferred: Double,
        maximum: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= preferred) return 1.0
        if (value >= maximum) return 0.08

        val normalized = (
            value - preferred
        ) / (
            maximum - preferred
        ).coerceAtLeast(1e-6)

        return (
            1.0 -
                normalized * 0.92
            ).coerceIn(0.08, 1.0)
    }

    private fun format(value: Double): String =
        "%.4f".format(
            java.util.Locale.US,
            value,
        )
}
