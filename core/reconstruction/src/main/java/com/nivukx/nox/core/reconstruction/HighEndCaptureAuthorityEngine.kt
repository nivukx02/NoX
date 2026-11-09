package com.nivukx.nox.core.reconstruction

/**
 * Cross-engine evidence authority for production motion capture.
 *
 * This engine does not create measurements. It adjudicates whether measurements produced by
 * calibration, synchronization, optical quality, triangulation, identity, temporal fusion and
 * downstream body/face/hand reconstruction are strong enough to be consumed as authoritative data.
 *
 * Design goals:
 *  - weak channels cannot be hidden by a high average;
 *  - independent geometric evidence is separated from visual confidence;
 *  - precision claims require empirical evidence rather than modeled feasibility;
 *  - multi-camera timing and optical quality are treated as first-class geometry inputs;
 *  - every decision is deterministic and replayable.
 */
class HighEndCaptureAuthorityEngine(
    private val config: Config = Config(),
) {

    enum class Band {
        AUTHORITATIVE,
        HIGH_CONFIDENCE,
        GUARDED,
        OBSERVATION_ONLY,
        BLOCKED,
    }

    enum class BlockCode {
        NO_CAMERA_EVIDENCE,
        CALIBRATION_TOO_WEAK,
        TIMING_TOO_WEAK_FOR_MULTI_VIEW,
        EPIPOLAR_CONSISTENCY_TOO_WEAK,
        METRIC_EVIDENCE_TOO_WEAK,
        EXCESSIVE_OPTICAL_BLUR,
        EXCESSIVE_RUNTIME_PRESSURE,
        EMPIRICAL_PRECISION_EVIDENCE_MISSING,
        REFERENCE_CONDITIONS_UNMATCHED,
        REFERENCE_UNCERTAINTY_TOO_HIGH,
    }

    data class Config(
        val geometryWeight: Double = 0.24,
        val temporalWeight: Double = 0.20,
        val metricWeight: Double = 0.18,
        val opticalWeight: Double = 0.11,
        val identityWeight: Double = 0.10,
        val runtimeWeight: Double = 0.08,
        val coverageWeight: Double = 0.09,
        val minimumCalibration: Double = 0.55,
        val minimumTimingForMultiView: Double = 0.55,
        val minimumEpipolar: Double = 0.50,
        val minimumMetricEvidence: Double = 0.48,
        val maximumMotionBlurRisk: Double = 0.48,
        val maximumRuntimePressure: Double = 0.88,
        val hardCalibration: Double = 0.30,
        val hardTimingForMultiView: Double = 0.30,
        val hardEpipolar: Double = 0.18,
        val hardMetricEvidence: Double = 0.16,
        val minimumAuthorityScore: Double = 0.78,
        val minimumPrecisionScore: Double = 0.86,
        val minimumPrecisionEvidenceSamples: Int = 600,
        val maximumPrecisionP95Millimeters: Double = 1.0,
        val maximumReferenceUncertaintyFraction: Double = 0.35,
    ) {
        init {
            listOf(
                geometryWeight,
                temporalWeight,
                metricWeight,
                opticalWeight,
                identityWeight,
                runtimeWeight,
                coverageWeight,
                minimumCalibration,
                minimumTimingForMultiView,
                minimumEpipolar,
                minimumMetricEvidence,
                maximumMotionBlurRisk,
                maximumRuntimePressure,
                hardCalibration,
                hardTimingForMultiView,
                hardEpipolar,
                hardMetricEvidence,
                minimumAuthorityScore,
                minimumPrecisionScore,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            require(
                geometryWeight + temporalWeight + metricWeight + opticalWeight +
                    identityWeight + runtimeWeight + coverageWeight > 0.0,
            )
            require(minimumPrecisionEvidenceSamples > 0)
            require(maximumPrecisionP95Millimeters > 0.0)
            require(
                maximumReferenceUncertaintyFraction.isFinite() &&
                    maximumReferenceUncertaintyFraction in 0.05..0.75,
            )
        }
    }

    data class Input(
        val timestampNanos: Long,
        val connectedCameras: Int,
        val authoritativeCameras: Int = 0,
        val calibrationConfidence: Double = 1.0,
        val synchronizationConfidence: Double = 1.0,
        val timingAuthorityConfidence: Double = 1.0,
        val temporalConsistency: Double = 1.0,
        val viewDiversity: Double = 1.0,
        val crossViewAgreement: Double = 1.0,
        val epipolarConsistency: Double = 1.0,
        val bodyMetricEvidenceAuthority: Double = 1.0,
        val faceMetricEvidenceAuthority: Double? = null,
        val handMetricEvidenceAuthority: Double? = null,
        val faceEpipolarConsistency: Double? = null,
        val handEpipolarConsistency: Double? = null,
        val crossModalMetricAuthority: Double? = null,
        val crossModalEpipolarAuthority: Double? = null,
        val reconstructionUncertaintyMillimeters: Double = 0.0,
        val motionP95ErrorMillimeters: Double = 0.0,
        val motionP99ErrorMillimeters: Double = 0.0,
        val temporalLagMilliseconds: Double = 0.0,
        val rollingShutterRisk: Double = 0.0,
        val motionBlurRisk: Double = 0.0,
        val cameraMotionRisk: Double = 0.0,
        val identityContinuity: Double = 1.0,
        val occlusionRisk: Double = 0.0,
        val contactConsistency: Double = 1.0,
        val thermalHeadroom: Double = 1.0,
        val memoryHeadroom: Double = 1.0,
        val gpuHeadroom: Double = 1.0,
        val npuHeadroom: Double = 1.0,
        val cpuHeadroom: Double = 1.0,
        val transportHeadroom: Double = 1.0,
        val observedGeometryRatio: Double = 1.0,
        val inferredGeometryRatio: Double = 0.0,
        val metricScaleUncertaintyRatio: Double = 0.0,
        val empiricalBenchmarkAccepted: Boolean = false,
        val empiricalBenchmarkSampleCount: Int = 0,
        val empiricalP95Millimeters: Double? = null,
        val referenceUncertaintyMillimeters: Double? = null,
        val referenceConditionsMatched: Boolean = false,
        val externalReferenceDocumented: Boolean = false,
        val targetP95Millimeters: Double = 5.0,
        val targetRequiresIndependentReference: Boolean = true,
        val distributedPlan: DistributedCameraRigCoordinator.Plan? = null,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(connectedCameras >= 0)
            require(authoritativeCameras >= 0)
            listOf(
                calibrationConfidence,
                synchronizationConfidence,
                timingAuthorityConfidence,
                temporalConsistency,
                viewDiversity,
                crossViewAgreement,
                epipolarConsistency,
                bodyMetricEvidenceAuthority,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            listOfNotNull(
                faceMetricEvidenceAuthority,
                handMetricEvidenceAuthority,
                faceEpipolarConsistency,
                handEpipolarConsistency,
                crossModalMetricAuthority,
                crossModalEpipolarAuthority,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            listOf(
                identityContinuity,
                contactConsistency,
                thermalHeadroom,
                memoryHeadroom,
                gpuHeadroom,
                npuHeadroom,
                cpuHeadroom,
                transportHeadroom,
                observedGeometryRatio,
                inferredGeometryRatio,
                rollingShutterRisk,
                motionBlurRisk,
                cameraMotionRisk,
                occlusionRisk,
                metricScaleUncertaintyRatio,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            listOf(
                reconstructionUncertaintyMillimeters,
                motionP95ErrorMillimeters,
                motionP99ErrorMillimeters,
                temporalLagMilliseconds,
                targetP95Millimeters,
            ).forEach { require(it.isFinite() && it >= 0.0) }
            require(motionP99ErrorMillimeters >= motionP95ErrorMillimeters)
            require(empiricalBenchmarkSampleCount >= 0)
            empiricalP95Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            referenceUncertaintyMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            require(authoritativeCameras <= connectedCameras)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val band: Band,
        val score: Double,
        val authoritative: Boolean,
        val hardBlocked: Boolean,
        val precisionQualified: Boolean,
        val empiricalPrecisionSupported: Boolean,
        val referenceRequired: Boolean,
        val geometryScore: Double,
        val temporalScore: Double,
        val metricScore: Double,
        val opticalScore: Double,
        val identityScore: Double,
        val runtimeScore: Double,
        val coverageScore: Double,
        val distributedPlanApplied: Boolean = false,
        val distributedSelectedCameraCount: Int = 0,
        val distributedGlobalHealth: Double = 0.0,
        val weakestChannel: String,
        val bottleneckScore: Double,
        val blockCodes: List<BlockCode>,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun evaluate(input: Input): Result {
        val effectiveAuthoritativeCameras =
            if (input.authoritativeCameras > 0) {
                input.authoritativeCameras
            } else {
                minOf(1, input.connectedCameras)
            }

        val multiView = input.connectedCameras >= 2
        val distributedTemporal =
            input.distributedPlan?.globalTemporalHealth ?: input.synchronizationConfidence
        val distributedGeometry =
            input.distributedPlan?.globalGeometryScore ?: input.crossViewAgreement
        val distributedHealth =
            input.distributedPlan?.globalHealth ?: input.calibrationConfidence

        val geometryScore = weightedMean(
            input.calibrationConfidence to 0.30,
            input.viewDiversity to 0.16,
            minOf(input.crossViewAgreement, distributedGeometry) to 0.18,
            minOf(
                input.epipolarConsistency,
                input.faceEpipolarConsistency
                    ?: input.epipolarConsistency,
                input.handEpipolarConsistency
                    ?: input.epipolarConsistency,
            ) to 0.20,
            (input.crossModalEpipolarAuthority
                ?: 1.0) to 0.04,
            distributedHealth to 0.12,
        )

        val temporalScore = weightedMean(
            input.synchronizationConfidence to 0.30,
            input.timingAuthorityConfidence to 0.24,
            input.temporalConsistency to 0.22,
            (1.0 - input.rollingShutterRisk) to 0.12,
            (1.0 - input.cameraMotionRisk) to 0.12,
        ).let { minOf(it, distributedTemporal.coerceIn(0.0, 1.0)) }

        val metricEvidenceMedian =
            median(
                listOfNotNull(
                    input.bodyMetricEvidenceAuthority,
                    input.faceMetricEvidenceAuthority,
                    input.handMetricEvidenceAuthority,
                ).ifEmpty { listOf(0.0) },
            )
        val crossModalMetricEvidence =
            input.crossModalMetricAuthority
                ?.coerceIn(0.0, 1.0)
                ?: 1.0

        val uncertaintyScore = uncertaintyAuthority(
            uncertaintyMillimeters = input.reconstructionUncertaintyMillimeters,
            targetMillimeters = input.targetP95Millimeters,
        )
        val empiricalScore =
            if (input.targetRequiresIndependentReference) {
                empiricalAuthority(
                    accepted = input.empiricalBenchmarkAccepted,
                    samples = input.empiricalBenchmarkSampleCount,
                    p95Millimeters = input.empiricalP95Millimeters,
                    targetMillimeters = input.targetP95Millimeters,
                )
            } else {
                1.0
            }

        val metricScore = weightedMean(
            metricEvidenceMedian to 0.16,
            uncertaintyScore to 0.34,
            empiricalScore to 0.22,
            (1.0 - input.metricScaleUncertaintyRatio) to 0.18,
            crossModalMetricEvidence to 0.10,
        )

        val opticalScore = weightedMean(
            (1.0 - input.motionBlurRisk) to 0.52,
            (1.0 - input.rollingShutterRisk) to 0.20,
            (1.0 - input.cameraMotionRisk) to 0.16,
            input.transportHeadroom to 0.12,
        )

        val identityScore = weightedMean(
            input.identityContinuity to 0.42,
            (1.0 - input.occlusionRisk) to 0.24,
            input.contactConsistency to 0.18,
            input.crossViewAgreement to 0.16,
        )

        val runtimeScore = weightedMean(
            input.thermalHeadroom to 0.18,
            input.memoryHeadroom to 0.24,
            input.gpuHeadroom to 0.24,
            input.npuHeadroom to 0.14,
            input.cpuHeadroom to 0.20,
        )

        val coverageScore =
            weightedMean(
                (if (input.connectedCameras <= 1) 1.0 else input.viewDiversity) to 0.30,
                input.observedGeometryRatio to 0.30,
                (1.0 - input.inferredGeometryRatio * 0.40) to 0.12,
                (
                    effectiveAuthoritativeCameras.toDouble() /
                        input.connectedCameras.coerceAtLeast(1).toDouble()
                ).coerceIn(0.0, 1.0) to 0.28,
            )

        val channels = listOf(
            "geometry" to geometryScore,
            "temporal" to temporalScore,
            "metric" to metricScore,
            "optical" to opticalScore,
            "identity" to identityScore,
            "runtime" to runtimeScore,
            "coverage" to coverageScore,
        )

        val score =
            weightedGeometricMean(
                channels = channels,
                weights = mapOf(
                    "geometry" to config.geometryWeight,
                    "temporal" to config.temporalWeight,
                    "metric" to config.metricWeight,
                    "optical" to config.opticalWeight,
                    "identity" to config.identityWeight,
                    "runtime" to config.runtimeWeight,
                    "coverage" to config.coverageWeight,
                ),
            )

        val blocks = mutableListOf<BlockCode>()
        if (input.connectedCameras <= 0) {
            blocks += BlockCode.NO_CAMERA_EVIDENCE
        }
        if (input.calibrationConfidence < config.hardCalibration) {
            blocks += BlockCode.CALIBRATION_TOO_WEAK
        }
        if (multiView && input.timingAuthorityConfidence < config.hardTimingForMultiView) {
            blocks += BlockCode.TIMING_TOO_WEAK_FOR_MULTI_VIEW
        }
        if (minOf(
                input.epipolarConsistency,
                input.faceEpipolarConsistency
                    ?: input.epipolarConsistency,
                input.handEpipolarConsistency
                    ?: input.epipolarConsistency,
            ) < config.hardEpipolar
        ) {
            blocks += BlockCode.EPIPOLAR_CONSISTENCY_TOO_WEAK
        }
        if (metricEvidenceMedian < config.hardMetricEvidence) {
            blocks += BlockCode.METRIC_EVIDENCE_TOO_WEAK
        }
        if (input.motionBlurRisk > config.maximumMotionBlurRisk) {
            blocks += BlockCode.EXCESSIVE_OPTICAL_BLUR
        }
        if (1.0 - runtimeScore > config.maximumRuntimePressure) {
            blocks += BlockCode.EXCESSIVE_RUNTIME_PRESSURE
        }

        val referenceRequired =
            input.targetRequiresIndependentReference &&
                input.targetP95Millimeters <= config.maximumPrecisionP95Millimeters

        val referenceUncertaintyReady =
            input.referenceUncertaintyMillimeters != null &&
                input.referenceUncertaintyMillimeters <=
                    input.targetP95Millimeters *
                        config.maximumReferenceUncertaintyFraction

        val empiricalReady =
            input.empiricalBenchmarkAccepted &&
                input.empiricalBenchmarkSampleCount >=
                    config.minimumPrecisionEvidenceSamples &&
                input.empiricalP95Millimeters != null &&
                input.referenceUncertaintyMillimeters != null &&
                referenceUncertaintyReady &&
                input.externalReferenceDocumented &&
                (!referenceRequired || input.referenceConditionsMatched)

        if (referenceRequired && !empiricalReady) {
            if (!input.empiricalBenchmarkAccepted ||
                input.empiricalBenchmarkSampleCount <
                    config.minimumPrecisionEvidenceSamples ||
                input.empiricalP95Millimeters == null
            ) {
                blocks += BlockCode.EMPIRICAL_PRECISION_EVIDENCE_MISSING
            } else if (!referenceUncertaintyReady) {
                blocks += BlockCode.REFERENCE_UNCERTAINTY_TOO_HIGH
            } else if (!input.referenceConditionsMatched) {
                blocks += BlockCode.REFERENCE_CONDITIONS_UNMATCHED
            }
        }

        val minimumCrossEngineEpipolar =
            minOf(
                input.epipolarConsistency,
                input.faceEpipolarConsistency
                    ?: input.epipolarConsistency,
                input.handEpipolarConsistency
                    ?: input.epipolarConsistency,
                input.crossModalEpipolarAuthority ?: 1.0,
            )

        val criticalGate =
            input.calibrationConfidence >= config.minimumCalibration &&
                (!multiView || input.timingAuthorityConfidence >= config.minimumTimingForMultiView) &&
                minimumCrossEngineEpipolar >= config.minimumEpipolar &&
                metricEvidenceMedian >= config.minimumMetricEvidence &&
                crossModalMetricEvidence >= config.minimumMetricEvidence &&
                input.motionBlurRisk <= config.maximumMotionBlurRisk

        val authoritative =
            blocks.isEmpty() &&
                criticalGate &&
                score >= config.minimumAuthorityScore &&
                effectiveAuthoritativeCameras >= if (multiView) 2 else 1

        val precisionQualified =
            authoritative &&
                score >= config.minimumPrecisionScore &&
                empiricalReady &&
                input.motionP95ErrorMillimeters <= input.targetP95Millimeters &&
                input.motionP99ErrorMillimeters <= input.targetP95Millimeters * 2.4

        /*
         * Qualification failures such as missing external reference evidence intentionally do not
         * stop ordinary capture. Safety failures that indicate the live image/compute path is
         * unfit for authoritative geometry do stop capture admission.
         */
        val hardBlocked = blocks.any {
            it == BlockCode.NO_CAMERA_EVIDENCE ||
                it == BlockCode.CALIBRATION_TOO_WEAK ||
                it == BlockCode.TIMING_TOO_WEAK_FOR_MULTI_VIEW ||
                it == BlockCode.EPIPOLAR_CONSISTENCY_TOO_WEAK ||
                it == BlockCode.METRIC_EVIDENCE_TOO_WEAK ||
                it == BlockCode.EXCESSIVE_OPTICAL_BLUR ||
                it == BlockCode.EXCESSIVE_RUNTIME_PRESSURE
        }

        val band =
            when {
                hardBlocked -> Band.BLOCKED
                precisionQualified -> Band.AUTHORITATIVE
                authoritative -> Band.HIGH_CONFIDENCE
                criticalGate && score >= 0.58 -> Band.GUARDED
                input.connectedCameras > 0 -> Band.OBSERVATION_ONLY
                else -> Band.BLOCKED
            }

        val warnings = buildList {
            if (geometryScore < config.minimumCalibration) add("geometry-authority-limited")
            if (temporalScore < config.minimumTimingForMultiView && multiView) {
                add("temporal-authority-limited")
            }
            if (metricScore < config.minimumMetricEvidence) add("metric-authority-limited")
            if (opticalScore < 0.60) add("optical-quality-limited")
            if (identityScore < 0.60) add("identity-or-occlusion-limited")
            if (runtimeScore < 0.60) add("runtime-headroom-limited")
            if (input.inferredGeometryRatio > input.observedGeometryRatio) {
                add("inference-dominates-observed-geometry")
            }
            if (referenceRequired && !precisionQualified) {
                add("sub-millimeter-precision-not-empirically-qualified")
            }
        }

        val recommendations = buildList {
            when (weakest(channels).first) {
                "geometry" -> add("Increase independent calibrated view diversity and geometric parallax.")
                "temporal" -> add("Improve sensor timestamp authority and inter-camera synchronization.")
                "metric" -> add("Increase measured metric evidence and reduce reconstruction uncertainty.")
                "optical" -> add("Reduce motion blur, rolling-shutter exposure and camera-motion contamination.")
                "identity" -> add("Increase occlusion redundancy and strengthen cross-camera identity continuity.")
                "runtime" -> add("Raise compute, thermal and memory headroom before increasing capture load.")
                "coverage" -> add("Increase authoritative camera coverage before promoting reconstructed geometry.")
            }
            if (referenceRequired && !empiricalReady) {
                add("Run a matched empirical benchmark with documented reference uncertainty before claiming precision.")
            }
        }

        val bottleneck = weakest(channels)
        return Result(
            timestampNanos = input.timestampNanos,
            band = band,
            score = score,
            authoritative = authoritative,
            hardBlocked = hardBlocked,
            precisionQualified = precisionQualified,
            empiricalPrecisionSupported = empiricalReady,
            referenceRequired = referenceRequired,
            geometryScore = geometryScore,
            temporalScore = temporalScore,
            metricScore = metricScore,
            opticalScore = opticalScore,
            identityScore = identityScore,
            runtimeScore = runtimeScore,
            coverageScore = coverageScore,
            distributedPlanApplied =
                input.distributedPlan != null,
            distributedSelectedCameraCount =
                input.distributedPlan?.selectedCameraCount ?: 0,
            distributedGlobalHealth =
                input.distributedPlan?.globalHealth ?: 0.0,
            weakestChannel = bottleneck.first,
            bottleneckScore = bottleneck.second,
            blockCodes = blocks.distinct(),
            warnings = warnings.distinct(),
            recommendations = recommendations.distinct(),
        )
    }

    private fun weightedMean(
        vararg components: Pair<Double, Double>,
    ): Double {
        require(components.isNotEmpty())
        var weightedSum = 0.0
        var totalWeight = 0.0
        components.forEach { (rawValue, rawWeight) ->
            require(rawValue.isFinite())
            require(rawWeight.isFinite() && rawWeight >= 0.0)
            val value = rawValue.coerceIn(0.0, 1.0)
            weightedSum += value * rawWeight
            totalWeight += rawWeight
        }
        if (totalWeight <= 0.0) return 0.0
        return (weightedSum / totalWeight).coerceIn(0.0, 1.0)
    }

    private fun median(values: List<Double>): Double {
        require(values.isNotEmpty())
        val sorted = values.map { it.coerceIn(0.0, 1.0) }.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun uncertaintyAuthority(
        uncertaintyMillimeters: Double,
        targetMillimeters: Double,
    ): Double {
        if (uncertaintyMillimeters <= 0.0) return 1.0
        val target = targetMillimeters.coerceAtLeast(0.10)
        return (1.0 - uncertaintyMillimeters / (target * 4.0))
            .coerceIn(0.0, 1.0)
    }

    private fun empiricalAuthority(
        accepted: Boolean,
        samples: Int,
        p95Millimeters: Double?,
        targetMillimeters: Double,
    ): Double {
        if (!accepted || samples <= 0 || p95Millimeters == null) return 0.0
        val sampleScore =
            (samples.toDouble() / config.minimumPrecisionEvidenceSamples.toDouble())
                .coerceIn(0.0, 1.0)
        val errorScore =
            (1.0 - p95Millimeters / targetMillimeters.coerceAtLeast(0.10))
                .coerceIn(0.0, 1.0)
        return (0.35 * sampleScore + 0.65 * errorScore).coerceIn(0.0, 1.0)
    }

    private fun weightedGeometricMean(
        channels: List<Pair<String, Double>>,
        weights: Map<String, Double>,
    ): Double {
        var weightedLog = 0.0
        var totalWeight = 0.0
        channels.forEach { (name, rawScore) ->
            val weight = weights[name] ?: 0.0
            if (weight <= 0.0) return@forEach
            val score = rawScore.coerceIn(1.0e-6, 1.0)
            weightedLog += weight * kotlin.math.ln(score)
            totalWeight += weight
        }
        if (totalWeight <= 0.0) return 0.0
        return kotlin.math.exp(weightedLog / totalWeight).coerceIn(0.0, 1.0)
    }

    private fun weakest(
        channels: List<Pair<String, Double>>,
    ): Pair<String, Double> =
        channels.minByOrNull { it.second }
            ?: ("unknown" to 0.0)
}
