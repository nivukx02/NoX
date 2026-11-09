package com.nivukx.nox.core.calibration

import kotlin.math.max

/**
 * State machine for operator-guided multi-camera calibration.
 *
 * Calibration is a workflow with measurable gates, not a single optimistic "calibrated" flag.
 */
class CameraCalibrationWorkflowEngine(
    private val config: Config = Config(),
) {

    enum class Phase {
        IDLE, CONNECT, COVERAGE, SYNC, INTRINSICS, EXTRINSICS, VERIFY, READY, FAILED,
    }

    data class Config(
        val minimumCameras: Int = 2,
        val minimumCoverage: Double = 0.72,
        val maximumClockSkewMillis: Double = 4.0,
        val minimumRigScore: Double = 0.60,
        val minimumReprojectionScore: Double = 0.55,
        val minimumTemporalAlignmentConfidence: Double = 0.75,
        val maximumSubframeOffsetMillis: Double = 5.0,
    ) {
        init {
            require(minimumTemporalAlignmentConfidence in 0.0..1.0)
            require(maximumSubframeOffsetMillis > 0.0)
        }
    }

    data class Input(
        val cameraCount: Int,
        val coverage: Double,
        val synchronized: Boolean,
        val clockSkewMillis: Double,
        val intrinsicsSolved: Boolean,
        val extrinsicsSolved: Boolean,
        val quality: MultiCameraCalibrationQualityEngine.Result?,
        val temporalAlignmentConfidence: Double = 1.0,
        val subframeOffsetMillis: Double = 0.0,
        val temporalWarnings: List<String> = emptyList(),
    ) {
        init {
            require(temporalAlignmentConfidence in 0.0..1.0)
            require(subframeOffsetMillis.isFinite())
        }
    }

    data class Status(
        val phase: Phase,
        val progress: Double,
        val ready: Boolean,
        val blocking: List<String>,
        val guidance: List<String>,
        val diagnostics: List<String>,
    )

    @Synchronized
    fun evaluate(input: Input): Status {
        val blocking = mutableListOf<String>()
        val guidance = mutableListOf<String>()
        val coverage = input.coverage.coerceIn(0.0, 1.0)
        val clockSkew = input.clockSkewMillis.coerceAtLeast(0.0)
        val quality = input.quality

        val phase = when {
            input.cameraCount < config.minimumCameras -> {
                blocking += "not enough cameras"
                guidance += "Connect and authorize at least " + config.minimumCameras + " cameras."
                Phase.CONNECT
            }
            coverage < config.minimumCoverage -> {
                blocking += "subject-space coverage is too low"
                guidance += "Move cameras until the capture volume is covered from multiple angles."
                Phase.COVERAGE
            }
            !input.synchronized ||
                clockSkew > config.maximumClockSkewMillis ||
                input.temporalAlignmentConfidence <
                    config.minimumTemporalAlignmentConfidence ||
                kotlin.math.abs(input.subframeOffsetMillis) >
                    config.maximumSubframeOffsetMillis -> {
                blocking +=
                    if (
                        input.temporalAlignmentConfidence <
                            config.minimumTemporalAlignmentConfidence ||
                        kotlin.math.abs(input.subframeOffsetMillis) >
                            config.maximumSubframeOffsetMillis
                    ) {
                        "sub-frame temporal alignment is not stable"
                    } else {
                        "camera timing is not stable"
                    }
                guidance +=
                    if (
                        input.temporalAlignmentConfidence <
                            config.minimumTemporalAlignmentConfidence ||
                        kotlin.math.abs(input.subframeOffsetMillis) >
                            config.maximumSubframeOffsetMillis
                    ) {
                        "Refine temporal alignment before solving multi-view geometry."
                    } else {
                        "Synchronize clocks before solving multi-view geometry."
                    }
                Phase.SYNC
            }
            !input.intrinsicsSolved -> {
                blocking += "intrinsics are not solved"
                guidance += "Capture a calibration target across the usable lens field."
                Phase.INTRINSICS
            }
            !input.extrinsicsSolved -> {
                blocking += "extrinsics are not solved"
                guidance += "Move the calibration target through the overlapping camera volume."
                Phase.EXTRINSICS
            }
            quality == null -> {
                blocking += "calibration quality has not been evaluated"
                guidance += "Run the geometry and synchronization verification pass."
                Phase.VERIFY
            }
            quality.rigScore < config.minimumRigScore ||
                quality.reprojectionScore < config.minimumReprojectionScore -> {
                blocking += "calibration quality is below the production threshold"
                guidance += quality.recommendations
                Phase.FAILED
            }
            else -> Phase.READY
        }

        val progress = when (phase) {
            Phase.IDLE -> 0.0
            Phase.CONNECT -> (input.cameraCount.toDouble() / config.minimumCameras.toDouble()).coerceIn(0.0, 1.0) * 0.12
            Phase.COVERAGE -> 0.12 + normalized(coverage, config.minimumCoverage) * 0.16
            Phase.SYNC -> 0.28 + syncProgress(clockSkew) * 0.16
            Phase.INTRINSICS -> 0.44 + if (input.intrinsicsSolved) 0.08 else 0.0
            Phase.EXTRINSICS -> 0.52 + if (input.extrinsicsSolved) 0.16 else 0.0
            Phase.VERIFY -> 0.68
            Phase.READY -> 1.0
            Phase.FAILED -> 0.75
        }

        return Status(
            phase = phase,
            progress = progress.coerceIn(0.0, 1.0),
            ready = phase == Phase.READY,
            blocking = blocking.distinct(),
            guidance = guidance.distinct(),
            diagnostics = listOf(
                "cameraCount=" + input.cameraCount,
                "coverage=" + format(coverage),
                "clockSkewMs=" + format(clockSkew),
                "temporalAlignmentConfidence=" +
                    format(input.temporalAlignmentConfidence),
                "subframeOffsetMs=" +
                    format(input.subframeOffsetMillis),
                "temporalWarnings=" +
                    input.temporalWarnings.joinToString("|"),
                "intrinsicsSolved=" + input.intrinsicsSolved,
                "extrinsicsSolved=" + input.extrinsicsSolved,
                "rigScore=" + format(quality?.rigScore ?: 0.0),
                "reprojectionScore=" + format(quality?.reprojectionScore ?: 0.0),
                "phase=" + phase.name,
            ),
        )
    }

    private fun normalized(value: Double, target: Double): Double =
        (value / target.coerceAtLeast(1e-6)).coerceIn(0.0, 1.0)

    private fun syncProgress(skewMillis: Double): Double =
        (1.0 - skewMillis / config.maximumClockSkewMillis).coerceIn(0.0, 1.0)

    private fun format(value: Double): String =
        "%.4f".format(java.util.Locale.US, value)
}