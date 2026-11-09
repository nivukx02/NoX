
package com.nivukx.nox.core.ai

import kotlin.math.max
import kotlin.math.min

/**
 * Stage-aware runtime performance governor for mobile motion capture.
 *
 * It consumes observed latency and device headroom and returns explicit stage scales. The governor
 * does not invent thermal information; callers provide measured signals from the platform.
 */
class MotionPerformanceGovernor(
    private val config: Config = Config(),
) {

    data class Config(
        val targetFrameBudgetMillis: Double = 16.67,
        val highQualityBudgetMillis: Double = 12.5,
        val degradedBudgetMillis: Double = 24.0,
        val minimumScale: Double = 0.20,
        val maximumScale: Double = 1.0,
        val recoveryRate: Double = 0.12,
        val degradationRate: Double = 0.24,
        val thermalFloor: Double = 0.22,
        val memoryFloor: Double = 0.20,
    )

    data class Telemetry(
        val frameTimeMillis: Double,
        val detectorMillis: Double,
        val trackerMillis: Double,
        val depthMillis: Double,
        val faceMillis: Double,
        val eyeMillis: Double,
        val handMillis: Double,
        val reconstructionMillis: Double,
        val meshMillis: Double,
        val thermalHeadroom: Double,
        val memoryHeadroom: Double,
        val gpuHeadroom: Double,
        val npuHeadroom: Double,
        val droppedFrames: Int = 0,
        val zeroCopyCameraInput: Boolean = false,
        val cameraInputHostCopies: Int = 0,
    ) {
        init {
            val durations = listOf(
                frameTimeMillis,
                detectorMillis,
                trackerMillis,
                depthMillis,
                faceMillis,
                eyeMillis,
                handMillis,
                reconstructionMillis,
                meshMillis,
            )
            require(durations.all { it.isFinite() && it >= 0.0 })
            val headroom = listOf(
                thermalHeadroom,
                memoryHeadroom,
                gpuHeadroom,
                npuHeadroom,
            )
            require(headroom.all { it.isFinite() && it in 0.0..1.0 })
            require(droppedFrames >= 0)
            require(cameraInputHostCopies >= 0)
        }
    }

    data class StageBudget(
        val detectorScale: Double,
        val trackerScale: Double,
        val depthScale: Double,
        val faceScale: Double,
        val eyeScale: Double,
        val handScale: Double,
        val reconstructionScale: Double,
        val meshScale: Double,
        val targetFrameTimeMillis: Double,
        val degraded: Boolean,
        val warnings: List<String>,
        val diagnostics: List<String>,
    )

    private var currentScale = 1.0

    @Synchronized
    fun update(
        telemetry: Telemetry,
        preferredQuality: Double = 0.85,
    ): StageBudget {
        val frameBudget = when {
            preferredQuality >= 0.90 ->
                config.highQualityBudgetMillis
            preferredQuality <= 0.35 ->
                config.degradedBudgetMillis
            else ->
                config.targetFrameBudgetMillis
        }

        val latencyPressure = (
            telemetry.frameTimeMillis /
                frameBudget.coerceAtLeast(1.0)
            ).coerceIn(0.05, 4.0)

        val devicePressure = max(
            max(
                1.0 - telemetry.thermalHeadroom.coerceIn(0.0, 1.0),
                1.0 - telemetry.memoryHeadroom.coerceIn(0.0, 1.0),
            ),
            1.0 - min(
                telemetry.gpuHeadroom.coerceIn(0.0, 1.0),
                telemetry.npuHeadroom.coerceIn(0.0, 1.0),
            ),
        )

        val droppedPressure =
            (telemetry.droppedFrames.toDouble() / 8.0)
                .coerceIn(0.0, 1.0)

        val cameraTransferPressure =
            (
                telemetry.cameraInputHostCopies.toDouble() /
                    2.0
                ).coerceIn(0.0, 1.0) *
                if (telemetry.zeroCopyCameraInput) 0.0 else 1.0

        val totalPressure = (
            (latencyPressure - 0.75).coerceIn(0.0, 2.5) * 0.58 +
                devicePressure * 0.22 +
                droppedPressure * 0.12 +
                cameraTransferPressure * 0.08
        ).coerceIn(0.0, 1.0)

        currentScale = if (totalPressure > 0.50) {
            currentScale - config.degradationRate * (
                totalPressure - 0.50
            )
        } else {
            currentScale + config.recoveryRate * (
                0.50 - totalPressure
            )
        }

        currentScale = currentScale.coerceIn(
            config.minimumScale,
            config.maximumScale,
        )

        val detectorScale = stageScale(
            current = telemetry.detectorMillis,
            total = telemetry.frameTimeMillis,
            base = currentScale,
            multiplier = 0.92,
        )
        val trackerScale = stageScale(
            current = telemetry.trackerMillis,
            total = telemetry.frameTimeMillis,
            base = currentScale,
            multiplier = 1.06,
        )
        val depthScale = stageScale(
            current = telemetry.depthMillis,
            total = telemetry.frameTimeMillis,
            base = currentScale,
            multiplier = 0.82,
        )
        val faceScale = stageScale(
            current = telemetry.faceMillis,
            total = telemetry.frameTimeMillis,
            base = currentScale,
            multiplier = 0.86,
        )
        val eyeScale = stageScale(
            current = telemetry.eyeMillis,
            total = telemetry.frameTimeMillis,
            base = currentScale,
            multiplier = 0.80,
        )
        val handScale = stageScale(
            current = telemetry.handMillis,
            total = telemetry.frameTimeMillis,
            base = currentScale,
            multiplier = 0.74,
        )
        val reconstructionScale = stageScale(
            current = telemetry.reconstructionMillis,
            total = telemetry.frameTimeMillis,
            base = currentScale,
            multiplier = 1.12,
        )
        val meshScale = stageScale(
            current = telemetry.meshMillis,
            total = telemetry.frameTimeMillis,
            base = currentScale,
            multiplier = 0.62,
        )

        val warnings = buildList {
            if (latencyPressure > 1.0) {
                add("frame budget exceeded")
            }
            if (telemetry.thermalHeadroom < config.thermalFloor) {
                add("thermal headroom below target")
            }
            if (telemetry.memoryHeadroom < config.memoryFloor) {
                add("memory headroom below target")
            }
            if (telemetry.droppedFrames > 0) {
                add("dropped frames detected")
            }
            if (!telemetry.zeroCopyCameraInput &&
                telemetry.cameraInputHostCopies > 0
            ) {
                add("camera host copies present")
            }
            if (currentScale < 0.50) {
                add("quality governor degraded")
            }
        }

        return StageBudget(
            detectorScale = detectorScale,
            trackerScale = trackerScale,
            depthScale = depthScale,
            faceScale = faceScale,
            eyeScale = eyeScale,
            handScale = handScale,
            reconstructionScale = reconstructionScale,
            meshScale = meshScale,
            targetFrameTimeMillis = frameBudget,
            degraded =
                currentScale < 0.72 ||
                    totalPressure > 0.70,
            warnings = warnings.distinct(),
            diagnostics = listOf(
                "frameTimeMs=" + format(telemetry.frameTimeMillis),
                "latencyPressure=" + format(latencyPressure),
                "devicePressure=" + format(devicePressure),
                "droppedPressure=" + format(droppedPressure),
                "cameraTransferPressure=" +
                    format(cameraTransferPressure),
                "zeroCopyCameraInput=" +
                    telemetry.zeroCopyCameraInput,
                "cameraInputHostCopies=" +
                    telemetry.cameraInputHostCopies,
                "totalPressure=" + format(totalPressure),
                "qualityScale=" + format(currentScale),
                "detectorScale=" + format(detectorScale),
                "depthScale=" + format(depthScale),
                "faceScale=" + format(faceScale),
                "eyeScale=" + format(eyeScale),
                "handScale=" + format(handScale),
                "reconstructionScale=" + format(reconstructionScale),
                "meshScale=" + format(meshScale),
            ),
        )
    }

    @Synchronized
    fun reset() {
        currentScale = 1.0
    }

    private fun stageScale(
        current: Double,
        total: Double,
        base: Double,
        multiplier: Double,
    ): Double {
        if (!current.isFinite() || !total.isFinite() || total <= 0.0) {
            return base.coerceIn(
                config.minimumScale,
                config.maximumScale,
            )
        }

        val share = (
            current / total
        ).coerceIn(0.0, 1.0)

        return (
            base *
                multiplier *
                (1.0 - share * 0.38)
        ).coerceIn(
            config.minimumScale,
            config.maximumScale,
        )
    }

    private fun format(value: Double): String =
        "%.4f".format(
            java.util.Locale.US,
            value,
        )
}
