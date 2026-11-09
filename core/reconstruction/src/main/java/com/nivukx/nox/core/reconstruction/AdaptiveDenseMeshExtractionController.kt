package com.nivukx.nox.core.reconstruction

/**
 * Runtime scheduler for dense-surface extraction.
 *
 * Depth fusion and mesh extraction are different workloads. Fusion protects the incoming motion
 * signal, while extraction turns accumulated voxels into a presentation mesh and can become much
 * more expensive as coverage grows. This controller keeps those workloads decoupled and chooses
 * cadence/detail from measured scene evidence and device headroom.
 */
class AdaptiveDenseMeshExtractionController(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumIntervalNanos: Long = 40_000_000L,
        val stableIntervalNanos: Long = 180_000_000L,
        val maximumIntervalNanos: Long = 420_000_000L,
        val minimumCellBudget: Int = 48_000,
        // Matches the production native mesh coherence envelope without forcing the runtime to use it.
        val maximumCellBudget: Int = 480_000,
        val minimumVertexBudget: Int = 64_000,
        val maximumVertexBudget: Int = 500_000,
        val stablePressureThreshold: Double = 0.28,
        val highPriorityThreshold: Double = 0.68,
        val targetExtractionLatencyMillis: Double = 12.0,
        val maximumObservedLatencyMillis: Double = 48.0,
    ) {
        init {
            require(minimumIntervalNanos > 0L)
            require(stableIntervalNanos >= minimumIntervalNanos)
            require(maximumIntervalNanos >= stableIntervalNanos)
            require(minimumCellBudget > 0)
            require(maximumCellBudget >= minimumCellBudget)
            require(minimumVertexBudget > 0)
            require(maximumVertexBudget >= minimumVertexBudget)
            require(stablePressureThreshold in 0.0..1.0)
            require(highPriorityThreshold in 0.0..1.0)
            require(targetExtractionLatencyMillis > 0.0)
            require(maximumObservedLatencyMillis >= targetExtractionLatencyMillis)
        }
    }

    data class Evidence(
        val timestampNanos: Long,
        val motionEnergy: Double,
        val uncertainty: Double,
        val cameraCoverage: Double,
        val cameraInformationGain: Double = 1.0,
        val reconstructionConfidence: Double,
        val temporalStability: Double,
        val thermalHeadroom: Double,
        val gpuHeadroom: Double,
        val requestedDetail: Double,
        val voxelCount: Int,
        val maxVoxels: Int,
        val geometryConditioning: Double = 1.0,
        val viewDiversity: Double = 1.0,
        val metricUncertaintyP95Meters: Double = 0.0,
        val contactConfidence: Double = 1.0,
        val identitySafety: Double = 1.0,
        val calibrationConfidence: Double = 1.0,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(voxelCount >= 0)
            require(maxVoxels > 0)
            listOf(
                motionEnergy,
                uncertainty,
                cameraCoverage,
                cameraInformationGain,
                reconstructionConfidence,
                temporalStability,
                thermalHeadroom,
                gpuHeadroom,
                requestedDetail,
                geometryConditioning,
                viewDiversity,
                contactConfidence,
                identitySafety,
                calibrationConfidence,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }

            require(
                metricUncertaintyP95Meters.isFinite() &&
                    metricUncertaintyP95Meters >= 0.0,
            )
        }
    }

    data class Decision(
        val shouldExtract: Boolean,
        val minimumIntervalNanos: Long,
        val cellBudget: Int,
        val vertexBudget: Int,
        val isoBand: Double,
        val detailLevel: Double,
        val computePressure: Double,
        val priorityScore: Double,
        val observedExtractionLatencyMillis: Double,
        val reason: String,
    )

    private var lastExtractionTimestampNanos: Long = Long.MIN_VALUE
    private var extractionLatencyEmaMillis: Double = 0.0

    @Synchronized
    fun recordExtractionLatency(durationMillis: Double) {
        val sample = durationMillis
            .coerceAtLeast(0.0)
            .coerceAtMost(config.maximumObservedLatencyMillis * 2.0)

        extractionLatencyEmaMillis =
            if (extractionLatencyEmaMillis <= 0.0) {
                sample
            } else {
                extractionLatencyEmaMillis * 0.82 +
                    sample * 0.18
            }
    }

    @Synchronized
    fun evaluate(evidence: Evidence): Decision {
        val metricPrecisionTrust =
            (
                1.0 -
                    evidence.metricUncertaintyP95Meters / 0.010
            ).coerceIn(0.0, 1.0)

        val geometryEvidence =
            evidence.geometryConditioning * 0.55 +
                evidence.viewDiversity * 0.25 +
                evidence.calibrationConfidence * 0.20

        val physicalIntegrity =
            evidence.contactConfidence * 0.25 +
                evidence.identitySafety * 0.35 +
                metricPrecisionTrust * 0.25 +
                geometryEvidence * 0.15

        val qualityDemand = (
            evidence.requestedDetail * 0.20 +
                evidence.motionEnergy * 0.16 +
                evidence.uncertainty * 0.13 +
                evidence.cameraCoverage * 0.08 +
                evidence.cameraInformationGain * 0.07 +
                evidence.reconstructionConfidence * 0.10 +
                evidence.temporalStability * 0.09 +
                evidence.geometryConditioning * 0.07 +
                evidence.viewDiversity * 0.05 +
                physicalIntegrity * 0.05
            ).coerceIn(0.0, 1.0)

        val thermalPressure = (1.0 - evidence.thermalHeadroom)
            .coerceIn(0.0, 1.0)
        val gpuPressure = (1.0 - evidence.gpuHeadroom)
            .coerceIn(0.0, 1.0)
        val extractionLatencyPressure =
            if (extractionLatencyEmaMillis <= 0.0) {
                0.0
            } else {
                (
                    extractionLatencyEmaMillis /
                        config.maximumObservedLatencyMillis
                    ).coerceIn(0.0, 1.0)
            }

        val physicalPressure =
            (
                (1.0 - evidence.geometryConditioning) * 0.22 +
                    (1.0 - evidence.viewDiversity) * 0.14 +
                    (1.0 - evidence.contactConfidence) * 0.10 +
                    (1.0 - evidence.identitySafety) * 0.16 +
                    (1.0 - evidence.calibrationConfidence) * 0.14 +
                    (1.0 - metricPrecisionTrust) * 0.24
            ).coerceIn(0.0, 1.0)

        val basePressure = (
            thermalPressure * 0.24 +
                gpuPressure * 0.16 +
                extractionLatencyPressure * 0.14 +
                (1.0 - evidence.reconstructionConfidence) * 0.11 +
                evidence.uncertainty * 0.09 +
                (1.0 - evidence.cameraCoverage) * 0.04 +
                (1.0 - evidence.cameraInformationGain) * 0.04 +
                (1.0 - evidence.temporalStability) * 0.04 +
                physicalPressure * 0.14
            ).coerceIn(0.0, 1.0)

        val safetyPressureFloor = when {
            evidence.thermalHeadroom < 0.45 -> 0.52
            evidence.gpuHeadroom < 0.50 -> 0.50
            extractionLatencyEmaMillis >= config.targetExtractionLatencyMillis * 3.0 -> 0.52
            physicalPressure >= 0.50 -> 0.28
            else -> 0.0
        }

        val pressure = maxOf(basePressure, safetyPressureFloor)
            .coerceIn(0.0, 1.0)

        val geometryUrgency =
            (
                (1.0 - evidence.geometryConditioning) * 0.30 +
                    (1.0 - evidence.viewDiversity) * 0.20 +
                    (1.0 - metricPrecisionTrust) * 0.25 +
                    (1.0 - evidence.identitySafety) * 0.15 +
                    (1.0 - evidence.contactConfidence) * 0.10
            ).coerceIn(0.0, 1.0)

        val urgency = (
            evidence.motionEnergy * 0.26 +
                evidence.uncertainty * 0.22 +
                (1.0 - evidence.temporalStability) * 0.13 +
                (1.0 - evidence.reconstructionConfidence) * 0.10 +
                (1.0 - evidence.cameraCoverage) * 0.05 +
                (1.0 - evidence.cameraInformationGain) * 0.04 +
                geometryUrgency * 0.20
            ).coerceIn(0.0, 1.0)

        val priority = (
            urgency * 0.58 +
                qualityDemand * 0.24 +
                evidence.requestedDetail * 0.18
            ).coerceIn(0.0, 1.0)

        val interval = interpolate(
            maximum = config.maximumIntervalNanos.toDouble(),
            minimum = config.minimumIntervalNanos.toDouble(),
            factor = priority,
        ).toLong().coerceIn(
            config.minimumIntervalNanos,
            config.maximumIntervalNanos,
        )

        val stableScene = (
            pressure <= config.stablePressureThreshold &&
                urgency < 0.34 &&
                evidence.temporalStability >= 0.82 &&
                evidence.motionEnergy < 0.34
            )

        val selectedInterval = when {
            stableScene -> config.stableIntervalNanos
            pressure >= 0.50 || priority >= config.highPriorityThreshold ->
                config.minimumIntervalNanos
            else -> interval
        }

        val elapsed = if (lastExtractionTimestampNanos == Long.MIN_VALUE) {
            Long.MAX_VALUE
        } else {
            (evidence.timestampNanos - lastExtractionTimestampNanos)
                .coerceAtLeast(0L)
        }

        val shouldExtract = evidence.voxelCount >= 16 && (
            lastExtractionTimestampNanos == Long.MIN_VALUE ||
                elapsed >= selectedInterval ||
                priority >= config.highPriorityThreshold
            )

        val thermalScale = evidence.thermalHeadroom.coerceIn(0.20, 1.0)
        val gpuScale = evidence.gpuHeadroom.coerceIn(0.20, 1.0)
        val computeScale = minOf(
            1.0,
            0.52 + qualityDemand * 0.64,
            0.56 + thermalScale * 0.44,
            0.58 + gpuScale * 0.42,
        )

        val occupancy = (
            evidence.voxelCount.toDouble() /
                evidence.maxVoxels.toDouble().coerceAtLeast(1.0)
            ).coerceIn(0.0, 1.0)

        val detailLevel = (
            qualityDemand * 0.68 +
                priority * 0.18 +
                evidence.requestedDetail * 0.14
            ).coerceIn(0.0, 1.0)

        val budgetScale = (
            computeScale * (0.80 + detailLevel * 0.32) *
                (0.92 - occupancy * 0.14)
            ).coerceIn(0.30, 1.0)

        val cellBudget = (
            config.minimumCellBudget +
                (config.maximumCellBudget - config.minimumCellBudget) *
                    budgetScale
            ).toInt().coerceIn(
                config.minimumCellBudget,
                config.maximumCellBudget,
            )

        val vertexBudget = (
            config.minimumVertexBudget +
                (config.maximumVertexBudget - config.minimumVertexBudget) *
                    budgetScale
            ).toInt().coerceIn(
                config.minimumVertexBudget,
                config.maximumVertexBudget,
            )

        val isoBand = (
            0.20 -
                detailLevel * 0.08 +
                pressure * 0.035
            ).coerceIn(0.10, 0.22)

        val reason = when {
            evidence.thermalHeadroom < 0.45 -> "thermal_pressure"
            evidence.gpuHeadroom < 0.50 -> "gpu_pressure"
            evidence.motionEnergy >= 0.78 -> "motion_priority"
            evidence.uncertainty >= 0.72 -> "uncertainty_priority"
            evidence.cameraInformationGain < 0.42 ->
                "view_information_limited"
            evidence.cameraCoverage < 0.45 -> "coverage_limited"
            evidence.identitySafety < 0.55 -> "identity_safety_limited"
            evidence.contactConfidence < 0.55 -> "contact_evidence_limited"
            evidence.calibrationConfidence < 0.55 -> "calibration_limited"
            metricPrecisionTrust < 0.55 -> "metric_uncertainty_limited"
            evidence.geometryConditioning < 0.55 -> "geometry_conditioning_limited"
            evidence.viewDiversity < 0.55 -> "view_diversity_limited"
            stableScene -> "stable_scene_defer"
            priority >= config.highPriorityThreshold -> "high_priority"
            else -> "balanced"
        }

        if (shouldExtract) {
            lastExtractionTimestampNanos = evidence.timestampNanos
        }

        return Decision(
            shouldExtract = shouldExtract,
            minimumIntervalNanos = selectedInterval,
            cellBudget = cellBudget,
            vertexBudget = vertexBudget,
            isoBand = isoBand,
            detailLevel = detailLevel,
            computePressure = pressure,
            priorityScore = priority,
            observedExtractionLatencyMillis = extractionLatencyEmaMillis,
            reason = reason,
        )
    }

    @Synchronized
    fun reset() {
        lastExtractionTimestampNanos = Long.MIN_VALUE
        extractionLatencyEmaMillis = 0.0
    }

    private fun interpolate(
        maximum: Double,
        minimum: Double,
        factor: Double,
    ): Double =
        maximum + (minimum - maximum) * factor
}
