package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Production uncertainty budget for metric 3D reconstruction.
 *
 * It preserves anisotropy: triangulation depth uncertainty can be dramatically worse than lateral
 * uncertainty at shallow parallax. A single scalar error hides this failure mode and is therefore
 * not sufficient for high-end reconstruction gating.
 */
class AnisotropicReconstructionUncertaintyEngine(
    private val maximumJointStates: Int = 4096,
    private val maximumSamples: Int = 8192,
    private val correlatedBudgetEngine: CorrelatedUncertaintyBudgetEngine =
        CorrelatedUncertaintyBudgetEngine(),
) : AutoCloseable {

    data class JointInput(
        val jointId: Int,
        val reprojectionRmsPx: Double,
        val triangulationUncertaintyMeters: Double,
        val parallaxDegrees: Double,
        val angularDiversity: Double,
        val inlierRatio: Double,
        val cheiralityConfidence: Double,
        val temporalUncertaintyMeters: Double,
        val calibrationDriftRisk: Double,
        val rollingShutterRisk: Double,
        val depthConfidence: Double = 1.0,
        val scaleUncertaintyMeters: Double = 0.0,
        val supportingViews: Int = 2,
        val sceneDepthMeters: Double = 2.0,
    ) {
        init {
            require(jointId >= 0)
            require(reprojectionRmsPx.isFinite() && reprojectionRmsPx >= 0.0)
            require(triangulationUncertaintyMeters.isFinite() && triangulationUncertaintyMeters >= 0.0)
            require(parallaxDegrees.isFinite() && parallaxDegrees >= 0.0)
            require(angularDiversity in 0.0..1.0)
            require(inlierRatio in 0.0..1.0)
            require(cheiralityConfidence in 0.0..1.0)
            require(temporalUncertaintyMeters >= 0.0)
            require(calibrationDriftRisk in 0.0..1.0)
            require(rollingShutterRisk in 0.0..1.0)
            require(depthConfidence in 0.0..1.0)
            require(scaleUncertaintyMeters >= 0.0)
            require(supportingViews >= 1)
            require(sceneDepthMeters > 0.0)
        }
    }

    data class Result(
        val jointId: Int,
        val lateral95Meters: Double,
        val depth95Meters: Double,
        val radial95Meters: Double,
        val anisotropyRatio: Double,
        val confidence: Double,
        val gatePassed: Boolean,
        val degraded: Boolean,
        val warnings: List<String>,
    )

    data class Aggregate(
        val jointCount: Int,
        val medianLateral95Meters: Double,
        val p95Lateral95Meters: Double,
        val medianDepth95Meters: Double,
        val p95Depth95Meters: Double,
        val maxAnisotropyRatio: Double,
        val confidence: Double,
        val degradedJointCount: Int,
        val gatePassed: Boolean,
        val warnings: List<String>,
    )

    private val samples = ArrayDeque<Double>()
    private val states = LinkedHashMap<Int, Result>()
    private var closed = false

    @Synchronized
    fun evaluate(
        input: JointInput,
        lateralBudgetMeters: Double = 0.015,
        depthBudgetMeters: Double = 0.035,
    ): Result {
        check(!closed)
        require(lateralBudgetMeters > 0.0)
        require(depthBudgetMeters > 0.0)

        val parallaxRad = Math.toRadians(input.parallaxDegrees.coerceAtLeast(0.25))
        val geometricDepthAmplifier = (
            1.0 / max(0.02, sin(parallaxRad))
        ).coerceIn(1.0, 40.0)
        val reprojectionMeters = (
            input.reprojectionRmsPx * input.sceneDepthMeters * 0.001
        ).coerceAtLeast(0.0001)

        val lateral = max(
            input.triangulationUncertaintyMeters,
            reprojectionMeters * (
                0.35 +
                    0.65 / max(1.0, sqrt(input.supportingViews.toDouble()))
                ),
        )
        val depth = max(
            input.triangulationUncertaintyMeters * geometricDepthAmplifier,
            reprojectionMeters * geometricDepthAmplifier * 0.82,
        )

        val conditioningPenalty = (
            (1.0 - input.angularDiversity) * 0.28 +
                (1.0 - input.inlierRatio) * 0.26 +
                (1.0 - input.cheiralityConfidence) * 0.20 +
                input.calibrationDriftRisk * 0.12 +
                input.rollingShutterRisk * 0.08 +
                (1.0 - input.depthConfidence) * 0.06
            ).coerceIn(0.0, 0.96)

        val occlusionSigma =
            (
                1.0 - input.inlierRatio
            ) * input.sceneDepthMeters * 0.0025

        val calibrationSigma =
            input.calibrationDriftRisk *
                input.sceneDepthMeters *
                0.004

        val rollingShutterSigma =
            input.rollingShutterRisk *
                input.sceneDepthMeters *
                0.003

        val lateralBudget =
            correlatedBudgetEngine.fuse(
                listOf(
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.TRIANGULATION,
                        sigmaMeters = lateral,
                        confidence = input.cheiralityConfidence,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.REPROJECTION,
                        sigmaMeters = reprojectionMeters,
                        confidence = input.inlierRatio,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.TEMPORAL,
                        sigmaMeters = input.temporalUncertaintyMeters * 0.55,
                        confidence = input.depthConfidence,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.CALIBRATION,
                        sigmaMeters = calibrationSigma * 0.50,
                        confidence = input.angularDiversity,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.ROLLING_SHUTTER,
                        sigmaMeters = rollingShutterSigma * 0.50,
                        confidence = 1.0 - input.rollingShutterRisk * 0.50,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.OCCLUSION,
                        sigmaMeters = occlusionSigma,
                        confidence = input.inlierRatio,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.SCALE,
                        sigmaMeters = input.scaleUncertaintyMeters,
                        confidence = input.depthConfidence,
                    ),
                ),
            )

        val depthBudget =
            correlatedBudgetEngine.fuse(
                listOf(
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.TRIANGULATION,
                        sigmaMeters = depth,
                        confidence = input.cheiralityConfidence,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.REPROJECTION,
                        sigmaMeters = reprojectionMeters * geometricDepthAmplifier,
                        confidence = input.inlierRatio,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.TEMPORAL,
                        sigmaMeters = input.temporalUncertaintyMeters,
                        confidence = input.depthConfidence,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.CALIBRATION,
                        sigmaMeters = calibrationSigma,
                        confidence = input.angularDiversity,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.ROLLING_SHUTTER,
                        sigmaMeters = rollingShutterSigma,
                        confidence = 1.0 - input.rollingShutterRisk * 0.50,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.OCCLUSION,
                        sigmaMeters = occlusionSigma,
                        confidence = input.inlierRatio,
                    ),
                    CorrelatedUncertaintyBudgetEngine.Contributor(
                        source = CorrelatedUncertaintyBudgetEngine.Source.SCALE,
                        sigmaMeters = input.scaleUncertaintyMeters,
                        confidence = input.depthConfidence,
                    ),
                ),
            )

        val lateralAdjusted = max(
            lateral,
            lateralBudget.fusedSigmaMeters +
                conditioningPenalty * 0.010,
        )
        val depthAdjusted = max(
            depth,
            depthBudget.fusedSigmaMeters +
                conditioningPenalty * 0.018,
        )

        val radial = sqrt(
            lateralAdjusted * lateralAdjusted * 2.0 +
                depthAdjusted * depthAdjusted,
        )
        val anisotropy = (
            depthAdjusted / max(lateralAdjusted, 1e-6)
        ).coerceIn(1.0, 100.0)
        val confidence = (
            (1.0 - min(1.0, lateralAdjusted / lateralBudgetMeters)) * 0.38 +
                (1.0 - min(1.0, depthAdjusted / depthBudgetMeters)) * 0.42 +
                input.inlierRatio * 0.08 +
                input.cheiralityConfidence * 0.06 +
                input.angularDiversity * 0.06
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (input.parallaxDegrees < 2.0) add("low-parallax-depth-amplification")
            if (lateralBudget.independenceScore < 0.55) {
                add("lateral-uncertainty-correlation")
            }
            if (depthBudget.independenceScore < 0.55) {
                add("depth-uncertainty-correlation")
            }
            if (depthBudget.correlationPenalty > 1.35) {
                add("depth-uncertainty-correlation-inflation")
            }
            if (input.angularDiversity < 0.20) add("weak-angular-diversity")
            if (input.inlierRatio < 0.60) add("low-triangulation-inlier-ratio")
            if (input.calibrationDriftRisk > 0.30) add("calibration-drift-propagation")
            if (input.rollingShutterRisk > 0.50) add("rolling-shutter-propagation")
            if (anisotropy >= 6.0) add("strong-depth-anisotropy")
        }

        val passed = lateralAdjusted <= lateralBudgetMeters &&
            depthAdjusted <= depthBudgetMeters &&
            confidence >= 0.65

        val result = Result(
            jointId = input.jointId,
            lateral95Meters = lateralAdjusted,
            depth95Meters = depthAdjusted,
            radial95Meters = radial,
            anisotropyRatio = anisotropy,
            confidence = confidence,
            gatePassed = passed,
            degraded = !passed,
            warnings = warnings.distinct(),
        )
        states[input.jointId] = result
        samples.addLast(radial)
        while (samples.size > maximumSamples) samples.removeFirst()
        while (states.size > maximumJointStates) {
            states.entries.firstOrNull()?.let { states.remove(it.key) } ?: break
        }
        return result
    }

    @Synchronized
    fun aggregate(): Aggregate {
        check(!closed)
        val results = states.values.toList()
        if (results.isEmpty()) {
            return Aggregate(
                jointCount = 0,
                medianLateral95Meters = 0.0,
                p95Lateral95Meters = 0.0,
                medianDepth95Meters = 0.0,
                p95Depth95Meters = 0.0,
                maxAnisotropyRatio = 1.0,
                confidence = 0.0,
                degradedJointCount = 0,
                gatePassed = false,
                warnings = listOf("no-joint-uncertainty-evidence"),
            )
        }

        val lateral = results.map { it.lateral95Meters }
        val depth = results.map { it.depth95Meters }
        val confidence = results.map { it.confidence }.average()
        val degraded = results.count { it.degraded }
        return Aggregate(
            jointCount = results.size,
            medianLateral95Meters = quantile(lateral, 0.50),
            p95Lateral95Meters = quantile(lateral, 0.95),
            medianDepth95Meters = quantile(depth, 0.50),
            p95Depth95Meters = quantile(depth, 0.95),
            maxAnisotropyRatio = results.maxOf { it.anisotropyRatio },
            confidence = confidence,
            degradedJointCount = degraded,
            gatePassed = degraded == 0 && confidence >= 0.65,
            warnings = results.flatMap { it.warnings }.distinct().take(16),
        )
    }

    @Synchronized
    fun reset() {
        samples.clear()
        states.clear()
    }

    override fun close() {
        reset()
        closed = true
    }

    private fun quantile(values: List<Double>, q: Double): Double {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        val position = q.coerceIn(0.0, 1.0) * sorted.lastIndex
        val lower = position.toInt()
        val upper = min(lower + 1, sorted.lastIndex)
        val alpha = position - lower
        return sorted[lower] * (1.0 - alpha) + sorted[upper] * alpha
    }
}
