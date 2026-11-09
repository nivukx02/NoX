package com.nivukx.nox.core.model

/**
 * Measurement contract for comparing NoX markerless capture with an external reference system.
 *
 * The protocol deliberately avoids declaring a winner. It defines matched conditions and metrics
 * that allow measured system differences to be reported reproducibly.
 */
enum class BenchmarkReferenceSystem {
    OPTICAL_INFRARED_MARKER,
    MARKER_BASED_OPTICAL,
    ROBOTIC_TRACKER,
    HIGH_PRECISION_CALIBRATION_RIG,
}

enum class BenchmarkSceneClass {
    STATIC_CALIBRATION,
    SINGLE_ACTOR,
    MULTI_ACTOR_NON_INTERACTING,
    MULTI_ACTOR_INTERACTION,
    FAST_MOTION,
    OCCLUSION_HEAVY,
}

data class BenchmarkMetricSample(
    val timestampNanos: Long,
    val actorId: Long,
    val jointId: Int,
    val referenceX: Double,
    val referenceY: Double,
    val referenceZ: Double,
    val estimatedX: Double,
    val estimatedY: Double,
    val estimatedZ: Double,
    val confidence: Double,
    val visible: Boolean,
    val predicted: Boolean,
    val identityCorrect: Boolean,
) {
    init {
        require(timestampNanos >= 0L)
        require(actorId >= 0L)
        require(jointId >= 0)
        listOf(
            referenceX,
            referenceY,
            referenceZ,
            estimatedX,
            estimatedY,
            estimatedZ,
            confidence,
        ).forEach {
            require(it.isFinite())
        }
        require(confidence in 0.0..1.0)
    }

    val errorMillimeters: Double
        get() {
            val dx = estimatedX - referenceX
            val dy = estimatedY - referenceY
            val dz = estimatedZ - referenceZ
            return kotlin.math.sqrt(
                dx * dx +
                    dy * dy +
                    dz * dz,
            ) * 1000.0
        }
}

data class MotionCaptureBenchmarkMetrics(
    val sampleCount: Int,
    val actorCount: Int,
    val meanErrorMillimeters: Double,
    val medianErrorMillimeters: Double,
    val p95ErrorMillimeters: Double,
    val p99ErrorMillimeters: Double,
    val maximumErrorMillimeters: Double,
    val identityAccuracy: Double,
    val temporalJitterMillimeters: Double,
    val predictedSampleRatio: Double,
    val visibleSampleRatio: Double,
    val reconstructionCoverage: Double,
    val frameLatencyMilliseconds: Double,
    val sustainedCaptureFps: Double,
    val thermalRiseCelsius: Double,
    val droppedFrameRatio: Double,
) {
    init {
        require(sampleCount >= 0)
        require(actorCount >= 0)
        listOf(
            meanErrorMillimeters,
            medianErrorMillimeters,
            p95ErrorMillimeters,
            p99ErrorMillimeters,
            maximumErrorMillimeters,
            identityAccuracy,
            temporalJitterMillimeters,
            predictedSampleRatio,
            visibleSampleRatio,
            reconstructionCoverage,
            frameLatencyMilliseconds,
            sustainedCaptureFps,
            thermalRiseCelsius,
            droppedFrameRatio,
        ).forEach { require(it.isFinite() && it >= 0.0) }
        require(identityAccuracy in 0.0..1.0)
        require(predictedSampleRatio in 0.0..1.0)
        require(visibleSampleRatio in 0.0..1.0)
        require(reconstructionCoverage in 0.0..1.0)
        require(droppedFrameRatio in 0.0..1.0)
    }
}

data class MotionCaptureBenchmarkProtocol(
    val referenceSystem: BenchmarkReferenceSystem,
    val sceneClass: BenchmarkSceneClass,
    val captureDurationSeconds: Int,
    val referenceFrameRate: Int,
    val noxFrameRate: Int,
    val synchronizedStartRequired: Boolean = true,
    val matchedCalibrationRequired: Boolean = true,
    val repeatedRuns: Int = 5,
    val minimumSamplesPerActorJoint: Int = 120,
    val evaluateIdentityContinuity: Boolean = true,
    val evaluateOcclusionRecovery: Boolean = true,
    val evaluateTailErrors: Boolean = true,
    val evaluateThermalSustainability: Boolean = true,
) {
    init {
        require(captureDurationSeconds >= 5)
        require(referenceFrameRate >= 15)
        require(noxFrameRate >= 15)
        require(repeatedRuns >= 1)
        require(minimumSamplesPerActorJoint >= 20)
    }
}

data class MotionCaptureBenchmarkReport(
    val protocol: MotionCaptureBenchmarkProtocol,
    val runs: List<MotionCaptureBenchmarkMetrics>,
    val aggregate: MotionCaptureBenchmarkMetrics,
    val uncertaintyQualification:
        PrecisionQualificationState,
    val caveats: List<String>,
) {
    init {
        require(runs.isNotEmpty())
        require(caveats.isNotEmpty())
    }

    val referenceComparisonAvailable: Boolean
        get() = protocol.referenceSystem !=
            BenchmarkReferenceSystem.ROBOTIC_TRACKER ||
            runs.isNotEmpty()
}
