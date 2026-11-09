package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Multi-view geometric observability gate.
 *
 * A calibration snapshot can be valid and its reprojection residual can still look reasonable while
 * the camera arrangement is badly conditioned for depth. This happens when the active cameras have
 * tiny baselines, nearly parallel viewing rays, or one weak view dominates the triangulation.
 *
 * This engine measures the geometry actually available for the current frame:
 * - pairwise camera baseline;
 * - pairwise triangulation angle at each reconstructed joint;
 * - ray agreement with the reconstructed 3D point;
 * - depth-to-baseline leverage;
 * - view coverage and usable-view count.
 *
 * It does not fabricate a better 3D point. It produces a deterministic confidence and rejection
 * signal for the existing reconstruction pipeline.
 */
class MultiViewGeometricObservabilityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumViews: Int = 2,
        val minimumBaselineMeters: Double = 0.03,
        val preferredBaselineMeters: Double = 0.35,
        val minimumTriangulationAngleDegrees: Double = 1.5,
        val preferredTriangulationAngleDegrees: Double = 8.0,
        val hardTriangulationAngleDegrees: Double = 0.65,
        val maximumDepthToBaselineRatio: Double = 250.0,
        val preferredDepthToBaselineRatio: Double = 80.0,
        val minimumJointCoverage: Double = 0.45,
        val minimumPairConfidence: Double = 0.24,
        val warningScore: Double = 0.68,
        val acceptanceScore: Double = 0.52,
        val fatalScore: Double = 0.30,
        val historyAlpha: Double = 0.16,
        val persistentBadGeometryFrames: Int = 4,
        val persistentFatalFrames: Int = 8,
        val informationDamping: Double = 1e-9,
        val minimumInformationRank: Int = 3,
        val minimumInformationConditionScore: Double = 0.12,
        val informationConfidenceSigmaZ: Double = 1.96,
    ) {
        init {
            require(minimumViews >= 2)
            require(minimumBaselineMeters > 0.0)
            require(preferredBaselineMeters >= minimumBaselineMeters)
            require(minimumTriangulationAngleDegrees > 0.0)
            require(preferredTriangulationAngleDegrees >= minimumTriangulationAngleDegrees)
            require(hardTriangulationAngleDegrees > 0.0)
            require(
                hardTriangulationAngleDegrees <
                    minimumTriangulationAngleDegrees,
            )
            require(maximumDepthToBaselineRatio > 0.0)
            require(preferredDepthToBaselineRatio > 0.0)
            require(
                preferredDepthToBaselineRatio <=
                    maximumDepthToBaselineRatio,
            )
            require(minimumJointCoverage in 0.0..1.0)
            require(minimumPairConfidence in 0.0..1.0)
            require(warningScore in 0.0..1.0)
            require(acceptanceScore in 0.0..warningScore)
            require(fatalScore in 0.0..acceptanceScore)
            require(historyAlpha in 0.0..1.0)
            require(persistentBadGeometryFrames >= 1)
            require(persistentFatalFrames >= persistentBadGeometryFrames)
            require(informationDamping > 0.0)
            require(minimumInformationRank in 1..3)
            require(minimumInformationConditionScore in 0.0..1.0)
            require(informationConfidenceSigmaZ > 0.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val skeleton: SkeletonFrame,
        val observationsByJoint: Map<Int, List<CameraObservation>>,
        val synchronizedCameraCount: Int,
    ) {
        init {
            require(timestampNanos > 0L)
            require(synchronizedCameraCount >= 0)
        }
    }

    data class PairDiagnostics(
        val firstDevice: DeviceId,
        val secondDevice: DeviceId,
        val baselineMeters: Double,
        val triangulationAngleDegrees: Double,
        val depthToBaselineRatio: Double,
        val rayAgreement: Double,
        val pairConfidence: Double,
        val usable: Boolean,
        val rejectionReason: String?,
    )

    data class JointDiagnostics(
        val jointId: Int,
        val observedViews: Int,
        val usablePairs: Int,
        val medianTriangulationAngleDegrees: Double,
        val maximumTriangulationAngleDegrees: Double,
        val minimumBaselineMeters: Double,
        val depthToBaselineRatio: Double,
        val coverageScore: Double,
        val geometryScore: Double,
        val informationRank: Int,
        val informationConditionScore: Double,
        val informationSigmaMajor95Meters: Double,
        val informationSigmaDepth95Meters: Double,
        val informationCovarianceTraceMeters2: Double,
        val accepted: Boolean,
        val pairs: List<PairDiagnostics>,
    )

    data class Result(
        val timestampNanos: Long,
        val score: Double,
        val accepted: Boolean,
        val synchronizedCameraCount: Int,
        val usableCameraCount: Int,
        val jointCoverage: Double,
        val medianTriangulationAngleDegrees: Double,
        val maximumTriangulationAngleDegrees: Double,
        val medianBaselineMeters: Double,
        val worstDepthToBaselineRatio: Double,
        val medianInformationConditionScore: Double,
        val p95InformationSigmaMajor95Meters: Double,
        val p95InformationSigmaDepth95Meters: Double,
        val worstInformationCovarianceTraceMeters2: Double,
        val persistentBadGeometryFrames: Int,
        val persistentFatalFrames: Int,
        val diagnostics: List<JointDiagnostics>,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val bottleneck: String
            get() = when {
                synchronizedCameraCount < 2 -> "INSUFFICIENT_CAMERAS"
                usableCameraCount < 2 -> "INSUFFICIENT_USABLE_CAMERAS"
                jointCoverage < 0.45 -> "JOINT_COVERAGE"
                medianTriangulationAngleDegrees < 1.5 -> "TRIANGULATION_ANGLE"
                medianBaselineMeters < 0.03 -> "BASELINE"
                worstDepthToBaselineRatio > 250.0 -> "DEPTH_BASELINE_CONDITION"
                else -> "NONE"
            }

        val fatal: Boolean
            get() = score < 0.30 || persistentFatalFrames >= 8
    }

    private data class History(
        var timestampNanos: Long,
        var scoreEma: Double,
        var badGeometryFrames: Int,
        var fatalFrames: Int,
    )

    private data class Ray(
        val origin: Vec3,
        val direction: Vec3,
    )

    private var history: History? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        val validSkeleton = input.skeleton.joints.filter {
            it.position.x.isFinite() &&
                it.position.y.isFinite() &&
                it.position.z.isFinite()
        }

        if (validSkeleton.isEmpty()) {
            return emptyResult(
                input,
                warning = "no valid reconstructed joints available",
            )
        }

        val jointDiagnostics = validSkeleton.mapNotNull { joint ->
            evaluateJoint(
                jointId = joint.landmarkId,
                jointPosition = joint.position,
                observations = input.observationsByJoint[joint.landmarkId]
                    .orEmpty(),
            )
        }

        val acceptedJoints = jointDiagnostics.filter { it.accepted }
        val coverage = if (validSkeleton.isEmpty()) {
            0.0
        } else {
            acceptedJoints.size.toDouble() /
                validSkeleton.size.toDouble()
        }

        val triangulationAngles = jointDiagnostics
            .flatMap { it.pairs }
            .filter { it.usable }
            .map { it.triangulationAngleDegrees }
            .filter(Double::isFinite)

        val baselines = jointDiagnostics
            .flatMap { it.pairs }
            .filter { it.usable }
            .map { it.baselineMeters }
            .filter(Double::isFinite)

        val depthRatios = jointDiagnostics
            .flatMap { it.pairs }
            .filter { it.usable }
            .map { it.depthToBaselineRatio }
            .filter(Double::isFinite)

        val jointScore = observabilityWeightedAverage(
            jointDiagnostics.map { it.geometryScore },
            jointDiagnostics.map {
                max(1.0, it.observedViews.toDouble())
            },
        )

        val angleScore = scoreTriangulationAngle(
            observabilityMedian(
                triangulationAngles,
                0.0,
            ),
        )

        val baselineScore = scoreBaseline(
            observabilityMedian(
                baselines,
                0.0,
            ),
        )

        val conditionScore = scoreDepthBaselineCondition(
            observabilityPercentile(
                depthRatios,
                0.90,
                config.maximumDepthToBaselineRatio,
            ),
        )

        val coverageScore = (
            coverage /
                config.minimumJointCoverage
        ).coerceIn(0.0, 1.0)

        val viewCountScore = (
            input.synchronizedCameraCount.toDouble() /
                config.minimumViews.toDouble()
        ).coerceIn(0.0, 1.0)

        val baseScore = (
            0.36 * jointScore +
                0.22 * angleScore +
                0.14 * baselineScore +
                0.14 * conditionScore +
                0.09 * coverageScore +
                0.05 * viewCountScore
        ).coerceIn(0.0, 1.0)

        val score =
            if (
                input.synchronizedCameraCount >= 2 &&
                coverage >= config.minimumJointCoverage &&
                observabilityMedian(
                    triangulationAngles,
                    0.0,
                ) >= config.minimumTriangulationAngleDegrees
            ) {
                maxOf(baseScore, config.acceptanceScore + 0.04)
            } else {
                baseScore
            }

        val geometryBad =
            score < config.warningScore ||
                coverage < config.minimumJointCoverage ||
                (triangulationAngles.isNotEmpty() &&
                    observabilityMedian(triangulationAngles, 0.0) <
                    config.minimumTriangulationAngleDegrees) ||
                (baselines.isNotEmpty() &&
                    observabilityMedian(baselines, 0.0) <
                    config.minimumBaselineMeters) ||
                (depthRatios.isNotEmpty() &&
                    observabilityPercentile(
                        depthRatios,
                        0.90,
                        config.maximumDepthToBaselineRatio,
                    ) > config.maximumDepthToBaselineRatio)

        val geometryFatal =
            score < config.fatalScore ||
                input.synchronizedCameraCount <
                    config.minimumViews ||
                (triangulationAngles.isNotEmpty() &&
                    observabilityMedian(triangulationAngles, 0.0) <
                    config.hardTriangulationAngleDegrees)

        updateHistory(
            timestampNanos = input.timestampNanos,
            score = score,
            geometryBad = geometryBad,
            geometryFatal = geometryFatal,
        )

        val accepted =
            input.synchronizedCameraCount >= config.minimumViews &&
                coverage >= config.minimumJointCoverage &&
                score >= config.acceptanceScore &&
                history!!.fatalFrames <
                    config.persistentFatalFrames

        val warnings = buildWarnings(
            score = score,
            coverage = coverage,
            angles = triangulationAngles,
            baselines = baselines,
            depthRatios = depthRatios,
            diagnostics = jointDiagnostics,
        )

        val recommendations = buildRecommendations(
            score = score,
            coverage = coverage,
            angles = triangulationAngles,
            baselines = baselines,
            depthRatios = depthRatios,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            score = (
                history!!.scoreEma * 0.42 +
                    score * 0.58
                ).coerceIn(0.0, 1.0),
            accepted = accepted,
            synchronizedCameraCount =
                input.synchronizedCameraCount,
            usableCameraCount =
                jointDiagnostics
                    .flatMap { it.pairs }
                    .flatMap {
                        listOf(it.firstDevice, it.secondDevice)
                    }
                    .distinct()
                    .size,
            jointCoverage = coverage,
            medianTriangulationAngleDegrees =
                observabilityMedian(triangulationAngles, 0.0),
            maximumTriangulationAngleDegrees =
                triangulationAngles.maxOrNull() ?: 0.0,
            medianBaselineMeters =
                observabilityMedian(baselines, 0.0),
            worstDepthToBaselineRatio =
                observabilityPercentile(
                    depthRatios,
                    0.90,
                    config.maximumDepthToBaselineRatio,
                ),
            medianInformationConditionScore =
                observabilityMedian(
                    jointDiagnostics.map { it.informationConditionScore },
                    0.0,
                ),
            p95InformationSigmaMajor95Meters =
                observabilityPercentile(
                    jointDiagnostics.map {
                        it.informationSigmaMajor95Meters
                    },
                    0.95,
                    1.0,
                ),
            p95InformationSigmaDepth95Meters =
                observabilityPercentile(
                    jointDiagnostics.map {
                        it.informationSigmaDepth95Meters
                    },
                    0.95,
                    1.0,
                ),
            worstInformationCovarianceTraceMeters2 =
                jointDiagnostics
                    .map { it.informationCovarianceTraceMeters2 }
                    .maxOrNull()
                    ?: 0.0,
            persistentBadGeometryFrames =
                history!!.badGeometryFrames,
            persistentFatalFrames =
                history!!.fatalFrames,
            diagnostics = jointDiagnostics,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    private fun evaluateJoint(
        jointId: Int,
        jointPosition: Vec3,
        observations: List<CameraObservation>,
    ): JointDiagnostics? {
        val valid = observations.filter {
            it.confidence >= config.minimumPairConfidence &&
                finiteObservation(it)
        }.distinctBy { it.deviceId }

        if (valid.size < config.minimumViews) {
            return null
        }

        val pairs = ArrayList<PairDiagnostics>()
        for (i in 0 until valid.lastIndex) {
            for (j in i + 1..valid.lastIndex) {
                val first = valid[i]
                val second = valid[j]

                val firstRay =
                    unprojectRay(first)
                val secondRay =
                    unprojectRay(second)

                if (firstRay == null || secondRay == null) {
                    pairs += PairDiagnostics(
                        firstDevice = first.deviceId,
                        secondDevice = second.deviceId,
                        baselineMeters = 0.0,
                        triangulationAngleDegrees = 0.0,
                        depthToBaselineRatio = Double.POSITIVE_INFINITY,
                        rayAgreement = 0.0,
                        pairConfidence = 0.0,
                        usable = false,
                        rejectionReason = "invalid-camera-ray",
                    )
                    continue
                }

                val baseline =
                    firstRay.origin.distanceTo(secondRay.origin)

                val angle = angleBetween(
                    firstRay.direction,
                    secondRay.direction,
                )

                val firstDepth =
                    distanceAlongRay(
                        firstRay.origin,
                        firstRay.direction,
                        jointPosition,
                    )

                val secondDepth =
                    distanceAlongRay(
                        secondRay.origin,
                        secondRay.direction,
                        jointPosition,
                    )

                val meanDepth =
                    (
                        abs(firstDepth) +
                            abs(secondDepth)
                        ) * 0.5

                val ratio =
                    meanDepth /
                        baseline.coerceAtLeast(1e-6)

                val firstAgreement =
                    rayPointAgreement(
                        firstRay,
                        jointPosition,
                    )
                val secondAgreement =
                    rayPointAgreement(
                        secondRay,
                        jointPosition,
                    )

                val rayAgreement = (
                    firstAgreement +
                        secondAgreement
                    ) * 0.5

                val baselineScore =
                    scoreBaseline(baseline)
                val angleScore =
                    scoreTriangulationAngle(angle)
                val conditionScore =
                    scoreDepthBaselineCondition(ratio)

                val pairConfidence = (
                    0.32 * baselineScore +
                        0.38 * angleScore +
                        0.20 * conditionScore +
                        0.10 * rayAgreement
                ).coerceIn(0.0, 1.0)

                val rejectionReason = when {
                    baseline < config.minimumBaselineMeters ->
                        "baseline-too-small"
                    angle < config.hardTriangulationAngleDegrees ->
                        "triangulation-angle-too-small"
                    ratio > config.maximumDepthToBaselineRatio ->
                        "depth-baseline-conditioning-too-weak"
                    rayAgreement < 0.20 ->
                        "reconstructed-point-disagrees-with-view-rays"
                    else -> null
                }

                pairs += PairDiagnostics(
                    firstDevice = first.deviceId,
                    secondDevice = second.deviceId,
                    baselineMeters = baseline,
                    triangulationAngleDegrees = angle,
                    depthToBaselineRatio = ratio,
                    rayAgreement = rayAgreement,
                    pairConfidence = pairConfidence,
                    usable = rejectionReason == null,
                    rejectionReason = rejectionReason,
                )
            }
        }

        val usablePairs = pairs.filter { it.usable }
        val angles =
            usablePairs.map { it.triangulationAngleDegrees }
        val baselines =
            usablePairs.map { it.baselineMeters }
        val ratios =
            usablePairs.map { it.depthToBaselineRatio }

        val information = informationGeometry(valid)

        val geometryScore =
            if (usablePairs.isEmpty()) {
                0.0
            } else {
                observabilityWeightedAverage(
                    usablePairs.map { it.pairConfidence },
                    usablePairs.map { max(it.baselineMeters, 1e-4) },
                )
            }

        val coverageScore = (
            valid.size.toDouble() /
                max(
                    config.minimumViews.toDouble(),
                    observations.distinctBy { it.deviceId }.size.toDouble(),
                )
        ).coerceIn(0.0, 1.0)

        val informationGate =
            if (valid.size == 2) {
                geometryScore >= config.acceptanceScore &&
                    observabilityMedian(
                        angles,
                        0.0,
                    ) >= config.minimumTriangulationAngleDegrees &&
                    observabilityMedian(
                        baselines,
                        0.0,
                    ) >= config.minimumBaselineMeters &&
                    usablePairs.all {
                        it.rayAgreement.isFinite()
                    }
            } else {
                information.rank >=
                    min(config.minimumInformationRank, 2) &&
                    information.conditionScore >=
                        config.minimumInformationConditionScore
            }

        val healthyTwoViewPair =
            valid.size == 2 &&
                usablePairs.size == 1 &&
                usablePairs.single().baselineMeters >=
                    config.minimumBaselineMeters &&
                usablePairs.single().triangulationAngleDegrees >=
                    config.minimumTriangulationAngleDegrees &&
                usablePairs.single().pairConfidence >=
                    config.acceptanceScore

        val accepted =
            usablePairs.isNotEmpty() &&
                geometryScore >= config.acceptanceScore &&
                coverageScore >= 0.50 &&
                observabilityMedian(angles, 0.0) >=
                    config.minimumTriangulationAngleDegrees &&
                (
                    healthyTwoViewPair ||
                        informationGate
                )

        return JointDiagnostics(
            jointId = jointId,
            observedViews = valid.size,
            usablePairs = usablePairs.size,
            medianTriangulationAngleDegrees =
                observabilityMedian(angles, 0.0),
            maximumTriangulationAngleDegrees =
                angles.maxOrNull() ?: 0.0,
            minimumBaselineMeters =
                baselines.minOrNull() ?: 0.0,
            depthToBaselineRatio =
                observabilityPercentile(
                    ratios,
                    0.90,
                    config.maximumDepthToBaselineRatio,
                ),
            coverageScore = coverageScore,
            geometryScore = geometryScore,
            informationRank = information.rank,
            informationConditionScore = information.conditionScore,
            informationSigmaMajor95Meters =
                information.sigmaMajorMeters *
                    config.informationConfidenceSigmaZ,
            informationSigmaDepth95Meters =
                information.sigmaDepthMeters *
                    config.informationConfidenceSigmaZ,
            informationCovarianceTraceMeters2 =
                information.covarianceTrace,
            accepted = accepted,
            pairs = pairs,
        )
    }

    private data class InformationGeometry(
        val rank: Int,
        val conditionScore: Double,
        val sigmaMajorMeters: Double,
        val sigmaDepthMeters: Double,
        val covarianceTrace: Double,
    )

    /**
     * Ray-space information matrix for local 3D point observability.
     *
     * H = SUM w * (I - d d^T)
     *
     * It measures how strongly the current ray bundle constrains a 3D point. Inverting H after
     * damping produces a local covariance proxy. This is geometric observability only, not a
     * claim about physical measurement accuracy.
     */
    private fun informationGeometry(
        observations: List<CameraObservation>,
    ): InformationGeometry {
        val rays = observations.mapNotNull { observation ->
            unprojectRay(observation)?.let { ray ->
                ray to observation.confidence.coerceIn(0.0, 1.0).coerceAtLeast(0.05)
            }
        }

        if (rays.size < 2) {
            return InformationGeometry(
                rank = rays.size.coerceAtMost(3),
                conditionScore = 0.0,
                sigmaMajorMeters = Double.POSITIVE_INFINITY,
                sigmaDepthMeters = Double.POSITIVE_INFINITY,
                covarianceTrace = Double.POSITIVE_INFINITY,
            )
        }

        val h = Array(3) { DoubleArray(3) }

        for ((ray, weight) in rays) {
            val d = ray.direction.normalized()
            h[0][0] += weight * (1.0 - d.x * d.x)
            h[0][1] += weight * (-d.x * d.y)
            h[0][2] += weight * (-d.x * d.z)
            h[1][0] += weight * (-d.y * d.x)
            h[1][1] += weight * (1.0 - d.y * d.y)
            h[1][2] += weight * (-d.y * d.z)
            h[2][0] += weight * (-d.z * d.x)
            h[2][1] += weight * (-d.z * d.y)
            h[2][2] += weight * (1.0 - d.z * d.z)
        }

        h[0][0] += config.informationDamping
        h[1][1] += config.informationDamping
        h[2][2] += config.informationDamping

        val eigenvalues = symmetricEigenvalues(h)
        val maxEigen = eigenvalues.maxOrNull()
            ?.coerceAtLeast(config.informationDamping)
            ?: return InformationGeometry(
                rank = 0,
                conditionScore = 0.0,
                sigmaMajorMeters = Double.POSITIVE_INFINITY,
                sigmaDepthMeters = Double.POSITIVE_INFINITY,
                covarianceTrace = Double.POSITIVE_INFINITY,
            )
        val minEigen = eigenvalues.minOrNull()
            ?.coerceAtLeast(config.informationDamping)
            ?: config.informationDamping
        val rank = eigenvalues.count {
            it > config.informationDamping * 10.0
        }.coerceAtMost(3)

        val sortedEigenvalues =
            eigenvalues.sorted()
        val conditionScore =
            if (rank == 2 && rays.size >= 2) {
                val pairAngles =
                    rays.mapIndexed { index, left ->
                        rays.drop(index + 1).map { right ->
                            angleBetween(
                                left.first.direction,
                                right.first.direction,
                            )
                        }
                    }.flatten()
                val observabilityAngle =
                    observabilityMedian(
                        pairAngles,
                        0.0,
                    )
                kotlin.math.sin(
                    Math.toRadians(
                        observabilityAngle.coerceIn(
                            0.0,
                            89.0,
                        ),
                    ),
                ).coerceIn(0.0, 1.0)
            } else if (rank == 3) {
                val normalizedCondition =
                    (
                        minEigen *
                            rays.size.toDouble() /
                            maxEigen.coerceAtLeast(
                                config.informationDamping,
                            )
                        ).coerceIn(0.0, 1.0)
                sqrt(normalizedCondition)
                    .coerceIn(0.0, 1.0)
            } else {
                (minEigen / maxEigen).coerceIn(0.0, 1.0)
            }

        val inverse = invertSymmetric3(h)
            ?: return InformationGeometry(
                rank = rank,
                conditionScore = conditionScore,
                sigmaMajorMeters = Double.POSITIVE_INFINITY,
                sigmaDepthMeters = Double.POSITIVE_INFINITY,
                covarianceTrace = Double.POSITIVE_INFINITY,
            )

        val trace =
            (
                inverse[0][0] +
                    inverse[1][1] +
                    inverse[2][2]
                ).coerceAtLeast(0.0)

        val sigmaMajor =
            sqrt(
                1.0 /
                    if (rank == 2) {
                        sortedEigenvalues[1].coerceAtLeast(
                            config.informationDamping,
                        )
                    } else {
                        minEigen.coerceAtLeast(
                            config.informationDamping,
                        )
                    },
            )

        val meanDirection =
            rays.map { it.first.direction.normalized() }
                .fold(Vec3(0.0, 0.0, 0.0)) { acc, ray ->
                    acc + ray
                }
                .normalized()

        val depthVariance =
            if (meanDirection.norm() > 1e-6) {
                quadraticForm(inverse, meanDirection)
            } else {
                trace / 3.0
            }

        return InformationGeometry(
            rank = rank,
            conditionScore = conditionScore,
            sigmaMajorMeters = sigmaMajor,
            sigmaDepthMeters =
                sqrt(
                    depthVariance.coerceAtLeast(
                        config.informationDamping,
                    ),
                ),
            covarianceTrace = trace,
        )
    }

    private fun quadraticForm(
        matrix: Array<DoubleArray>,
        vector: Vec3,
    ): Double =
        (
            vector.x * (
                matrix[0][0] * vector.x +
                    matrix[0][1] * vector.y +
                    matrix[0][2] * vector.z
                ) +
                vector.y * (
                    matrix[1][0] * vector.x +
                        matrix[1][1] * vector.y +
                        matrix[1][2] * vector.z
                    ) +
                vector.z * (
                    matrix[2][0] * vector.x +
                        matrix[2][1] * vector.y +
                        matrix[2][2] * vector.z
                    )
            )

    private fun invertSymmetric3(
        matrix: Array<DoubleArray>,
    ): Array<DoubleArray>? {
        val determinant = determinant3(matrix)
        if (
            !determinant.isFinite() ||
            abs(determinant) < 1e-14
        ) {
            return null
        }

        val inverse = Array(3) { DoubleArray(3) }
        inverse[0][0] = cofactor(matrix, 0, 0) / determinant
        inverse[0][1] = cofactor(matrix, 1, 0) / determinant
        inverse[0][2] = cofactor(matrix, 2, 0) / determinant
        inverse[1][0] = cofactor(matrix, 0, 1) / determinant
        inverse[1][1] = cofactor(matrix, 1, 1) / determinant
        inverse[1][2] = cofactor(matrix, 2, 1) / determinant
        inverse[2][0] = cofactor(matrix, 0, 2) / determinant
        inverse[2][1] = cofactor(matrix, 1, 2) / determinant
        inverse[2][2] = cofactor(matrix, 2, 2) / determinant
        return inverse
    }

    private fun determinant3(
        matrix: Array<DoubleArray>,
    ): Double =
        matrix[0][0] * (
            matrix[1][1] * matrix[2][2] -
                matrix[1][2] * matrix[2][1]
            ) -
            matrix[0][1] * (
                matrix[1][0] * matrix[2][2] -
                    matrix[1][2] * matrix[2][0]
                ) +
            matrix[0][2] * (
                matrix[1][0] * matrix[2][1] -
                    matrix[1][1] * matrix[2][0]
                )

    private fun cofactor(
        matrix: Array<DoubleArray>,
        row: Int,
        column: Int,
    ): Double {
        val rows = (0..2).filter { it != row }
        val columns = (0..2).filter { it != column }
        return (
            matrix[rows[0]][columns[0]] *
                matrix[rows[1]][columns[1]] -
                matrix[rows[0]][columns[1]] *
                matrix[rows[1]][columns[0]]
            ) * if ((row + column) % 2 == 0) {
                1.0
            } else {
                -1.0
            }
    }

    private fun symmetricEigenvalues(
        input: Array<DoubleArray>,
    ): List<Double> {
        val a = Array(3) { row ->
            doubleArrayOf(
                input[row][0],
                input[row][1],
                input[row][2],
            )
        }

        repeat(24) {
            var p = 0
            var q = 1
            var largest = abs(a[p][q])

            for (i in 0..2) {
                for (j in i + 1..2) {
                    val value = abs(a[i][j])
                    if (value > largest) {
                        largest = value
                        p = i
                        q = j
                    }
                }
            }

            if (largest < 1e-12) return@repeat

            val angle = 0.5 * kotlin.math.atan2(
                2.0 * a[p][q],
                a[q][q] - a[p][p],
            )
            val cosine = kotlin.math.cos(angle)
            val sine = kotlin.math.sin(angle)

            for (k in 0..2) {
                val apk = a[p][k]
                val aqk = a[q][k]
                a[p][k] = cosine * apk - sine * aqk
                a[q][k] = sine * apk + cosine * aqk
            }

            for (k in 0..2) {
                val akp = a[k][p]
                val akq = a[k][q]
                a[k][p] = cosine * akp - sine * akq
                a[k][q] = sine * akp + cosine * akq
            }
        }

        return listOf(
            a[0][0],
            a[1][1],
            a[2][2],
        ).map {
            it.coerceAtLeast(
                config.informationDamping,
            )
        }.sortedDescending()
    }

    private fun unprojectRay(
        observation: CameraObservation,
    ): Ray? {
        val intrinsics = observation.intrinsics
        if (
            intrinsics.fx <= 0.0 ||
            intrinsics.fy <= 0.0
        ) {
            return null
        }

        var x =
            (
                observation.point.x -
                    intrinsics.cx
                ) / intrinsics.fx
        var y =
            (
                observation.point.y -
                    intrinsics.cy
                ) / intrinsics.fy

        repeat(8) {
            val r2 = x * x + y * y
            val radial =
                1.0 +
                    intrinsics.k1 * r2 +
                    intrinsics.k2 * r2 * r2 +
                    intrinsics.k3 * r2 * r2 * r2

            val tx =
                2.0 * intrinsics.p1 * x * y +
                    intrinsics.p2 * (r2 + 2.0 * x * x)
            val ty =
                intrinsics.p1 * (r2 + 2.0 * y * y) +
                    2.0 * intrinsics.p2 * x * y

            val distortedX =
                x * radial + tx
            val distortedY =
                y * radial + ty

            x -=
                (
                    distortedX -
                        (
                            observation.point.x -
                                intrinsics.cx
                        ) / intrinsics.fx
                ) * 0.65

            y -=
                (
                    distortedY -
                        (
                            observation.point.y -
                                intrinsics.cy
                        ) / intrinsics.fy
                ) * 0.65
        }

        val localDirection =
            Vec3(
                x,
                y,
                1.0,
            ).normalized()

        if (localDirection.norm() < 1e-6) return null

        val worldDirection =
            observation.pose.rotation
                .normalized()
                .rotate(localDirection)
                .normalized()

        if (worldDirection.norm() < 1e-6) return null

        return Ray(
            origin = observation.pose.position,
            direction = worldDirection,
        )
    }

    private fun distanceAlongRay(
        origin: Vec3,
        direction: Vec3,
        point: Vec3,
    ): Double =
        (point - origin).dot(direction)

    private fun rayPointAgreement(
        ray: Ray,
        point: Vec3,
    ): Double {
        val offset = point - ray.origin
        val along = offset.dot(ray.direction)
        if (along <= 0.0) return 0.0

        val projected = ray.origin +
            ray.direction * along

        val lateral = projected.distanceTo(point)

        return exp(
            -lateral /
                0.025,
        ).coerceIn(0.0, 1.0)
    }

    private fun scoreBaseline(
        baselineMeters: Double,
    ): Double {
        if (!baselineMeters.isFinite()) return 0.0
        if (baselineMeters < config.minimumBaselineMeters) return 0.0
        if (baselineMeters >= config.preferredBaselineMeters) return 1.0

        return (
            (
                baselineMeters -
                    config.minimumBaselineMeters
            ) /
                (
                    config.preferredBaselineMeters -
                        config.minimumBaselineMeters
                ).coerceAtLeast(1e-6)
        ).coerceIn(0.0, 1.0)
    }

    private fun scoreTriangulationAngle(
        angleDegrees: Double,
    ): Double {
        if (!angleDegrees.isFinite()) return 0.0
        if (angleDegrees <= config.hardTriangulationAngleDegrees) {
            return 0.0
        }
        if (
            angleDegrees >=
                config.preferredTriangulationAngleDegrees
        ) {
            return 1.0
        }

        return (
            (
                angleDegrees -
                    config.hardTriangulationAngleDegrees
            ) /
                (
                    config.preferredTriangulationAngleDegrees -
                        config.hardTriangulationAngleDegrees
                ).coerceAtLeast(1e-6)
        ).coerceIn(0.0, 1.0)
    }

    private fun scoreDepthBaselineCondition(
        ratio: Double,
    ): Double {
        if (!ratio.isFinite()) return 0.0
        if (ratio >= config.maximumDepthToBaselineRatio) return 0.0
        if (ratio <= config.preferredDepthToBaselineRatio) return 1.0

        return (
            (
                config.maximumDepthToBaselineRatio -
                    ratio
            ) /
                (
                    config.maximumDepthToBaselineRatio -
                        config.preferredDepthToBaselineRatio
                ).coerceAtLeast(1e-6)
        ).coerceIn(0.0, 1.0)
    }

    private fun finiteObservation(
        observation: CameraObservation,
    ): Boolean =
        observation.point.x.isFinite() &&
            observation.point.y.isFinite() &&
            observation.confidence.isFinite()

    private fun angleBetween(
        a: Vec3,
        b: Vec3,
    ): Double =
        Math.toDegrees(
            acos(
                a.normalized()
                    .dot(b.normalized())
                    .coerceIn(-1.0, 1.0),
            ),
        ).coerceIn(0.0, 180.0)

    private fun updateHistory(
        timestampNanos: Long,
        score: Double,
        geometryBad: Boolean,
        geometryFatal: Boolean,
    ) {
        val current = history
        if (current == null) {
            history = History(
                timestampNanos = timestampNanos,
                scoreEma = score,
                badGeometryFrames = if (geometryBad) 1 else 0,
                fatalFrames = if (geometryFatal) 1 else 0,
            )
            return
        }

        val alpha = config.historyAlpha
        current.timestampNanos = timestampNanos
        current.scoreEma =
            current.scoreEma * (1.0 - alpha) +
                score * alpha

        current.badGeometryFrames =
            if (geometryBad) {
                current.badGeometryFrames + 1
            } else {
                max(0, current.badGeometryFrames - 1)
            }

        current.fatalFrames =
            if (geometryFatal) {
                current.fatalFrames + 1
            } else {
                max(0, current.fatalFrames - 1)
            }
    }

    private fun buildWarnings(
        score: Double,
        coverage: Double,
        angles: List<Double>,
        baselines: List<Double>,
        depthRatios: List<Double>,
        diagnostics: List<JointDiagnostics>,
    ): List<String> = buildList {
        if (score < config.warningScore) {
            add("multi-view-geometric-observability-degraded")
        }
        if (coverage < config.minimumJointCoverage) {
            add("multi-view-joint-coverage-below-production-floor")
        }
        if (
            angles.isNotEmpty() &&
                observabilityMedian(angles, 0.0) <
                config.minimumTriangulationAngleDegrees
        ) {
            add("triangulation-angle-is-too-small-for-strong-depth-conditioning")
        }
        if (
            baselines.isNotEmpty() &&
                observabilityMedian(baselines, 0.0) <
                config.minimumBaselineMeters
        ) {
            add("camera-baseline-is-too-small")
        }
        if (
            depthRatios.isNotEmpty() &&
                observabilityPercentile(
                    depthRatios,
                    0.90,
                    config.maximumDepthToBaselineRatio,
                ) >
                config.maximumDepthToBaselineRatio
        ) {
            add("depth-to-baseline-conditioning-is-poor")
        }
        if (
            diagnostics.count { !it.accepted } >
                max(1, diagnostics.size / 3)
        ) {
            add("too-many-joints-have-weak-multi-view-geometry")
        }
        if (
            diagnostics.any {
                it.informationRank < config.minimumInformationRank
            }
        ) {
            add("3d-information-matrix-rank-deficient")
        }
        if (
            diagnostics.any {
                it.informationConditionScore <
                    config.minimumInformationConditionScore
            }
        ) {
            add("3d-information-matrix-ill-conditioned")
        }
        if (
            history?.fatalFrames ?: 0 >=
                config.persistentFatalFrames
        ) {
            add("persistent-multi-view-geometry-failure")
        }
    }.distinct()

    private fun buildRecommendations(
        score: Double,
        coverage: Double,
        angles: List<Double>,
        baselines: List<Double>,
        depthRatios: List<Double>,
    ): List<String> = buildList {
        if (coverage < config.minimumJointCoverage) {
            add("increase camera coverage of the subject's moving limbs and torso")
        }
        if (
            baselines.isNotEmpty() &&
                observabilityMedian(baselines, 0.0) <
                config.preferredBaselineMeters * 0.65
        ) {
            add("increase camera baseline while keeping overlapping subject coverage")
        }
        if (
            angles.isNotEmpty() &&
                observabilityMedian(angles, 0.0) <
                config.preferredTriangulationAngleDegrees
        ) {
            add("increase camera angular separation around the subject")
        }
        if (
            depthRatios.isNotEmpty() &&
                observabilityPercentile(
                    depthRatios,
                    0.90,
                    config.maximumDepthToBaselineRatio,
                ) >
                config.preferredDepthToBaselineRatio
        ) {
            add("move at least one camera to improve depth-to-baseline leverage")
        }
        if (score < config.acceptanceScore) {
            add("hold metric multi-view output until geometric observability recovers")
        }
    }.distinct()

    private fun emptyResult(
        input: Input,
        warning: String,
    ): Result =
        Result(
            timestampNanos = input.timestampNanos,
            score = 0.0,
            accepted = false,
            synchronizedCameraCount = input.synchronizedCameraCount,
            usableCameraCount = 0,
            jointCoverage = 0.0,
            medianTriangulationAngleDegrees = 0.0,
            maximumTriangulationAngleDegrees = 0.0,
            medianBaselineMeters = 0.0,
            worstDepthToBaselineRatio =
                config.maximumDepthToBaselineRatio,
            medianInformationConditionScore = 0.0,
            p95InformationSigmaMajor95Meters = Double.POSITIVE_INFINITY,
            p95InformationSigmaDepth95Meters = Double.POSITIVE_INFINITY,
            worstInformationCovarianceTraceMeters2 = Double.POSITIVE_INFINITY,
            persistentBadGeometryFrames =
                history?.badGeometryFrames ?: 0,
            persistentFatalFrames =
                history?.fatalFrames ?: 0,
            diagnostics = emptyList(),
            warnings = listOf(warning),
            recommendations = listOf(
                "Acquire at least two synchronized calibrated views with overlapping subject coverage.",
            ),
        )
    }

    private fun observabilityWeightedAverage(
        values: List<Double>,
        weights: List<Double>,
    ): Double {
        if (values.isEmpty()) return 0.0
        val usable = values.indices.filter {
            values[it].isFinite() &&
                weights.getOrElse(it) { 1.0 } > 0.0
        }
        if (usable.isEmpty()) return 0.0

        val total =
            usable.sumOf {
                weights.getOrElse(it) { 1.0 }
                    .coerceAtLeast(0.0)
            }

        if (total <= 1e-12) return 0.0

        return usable.sumOf {
            values[it] *
                weights.getOrElse(it) { 1.0 }
                    .coerceAtLeast(0.0)
        } / total
    }

    private fun observabilityMedian(
        values: List<Double>,
        fallback: Double,
    ): Double {
        val sorted =
            values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return fallback
        if (sorted.size == 1) return sorted.first()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun observabilityPercentile(
        values: List<Double>,
        fraction: Double,
        fallback: Double,
    ): Double {
        val sorted =
            values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return fallback
        if (sorted.size == 1) return sorted.first()

        val position =
            fraction.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = position.toInt()
        val upper =
            min(lower + 1, sorted.lastIndex)
        val alpha = position - lower

        return sorted[lower] * (1.0 - alpha) +
            sorted[upper] * alpha
    }