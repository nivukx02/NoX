package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Multi-actor temporal association and micro-motion preservation.
 *
 * This engine is deliberately model-agnostic. The perception front-end may be ML Kit, MediaPipe,
 * ONNX, NPU, or another detector. It receives per-person observations and maintains stable actor IDs
 * using global assignment, detector identity evidence, geometric overlap, velocity continuity, and
 * skeleton-shape continuity.
 *
 * Motion is evaluated in normalized image coordinates with nanosecond timestamps. That preserves
 * subpixel evidence numerically without claiming nanometer spatial resolution from commodity cameras.
 */
class MultiActorPoseAssociationEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maxActors: Int = 50,
        val maxMissedFrames: Int = 45,
        val minimumMatchScore: Double = 0.16,
        val maximumCenterDistanceNormalized: Double = 0.80,
        val detectorIdentityWeight: Double = 0.12,
        val appearanceWeight: Double = 0.30,
        val appearanceHardMismatchThreshold: Double = 0.24,
        val appearanceHardGateQuality: Double = 0.72,
        val iouWeight: Double = 0.20,
        val centerWeight: Double = 0.17,
        val velocityWeight: Double = 0.16,
        val skeletonWeight: Double = 0.15,
        val confidenceWeight: Double = 0.05,
        val worldJointContinuityBlend: Double = 0.08,
        val minimumWorldJointContinuityForHardGate: Double = 0.22,
        val velocitySmoothing: Double = 0.42,
        val predictionMaximumSpeedNormalizedPerSecond: Double = 5.0,
        val predictionMaximumAccelerationNormalizedPerSecond2: Double = 30.0,
        val predictionMaximumJerkNormalizedPerSecond3: Double = 140.0,
        val microMotionHistoryFrames: Int = 24,
        val minimumMicroMotionDeltaNormalized: Double = 1e-7,
        val microMotionGain: Double = 0.72,
        val crowdedSceneOverlapThreshold: Double = 0.35,
        val crowdedSceneMatchScoreBoost: Double = 0.10,
    ) {
        init {
            require(maxActors in 1..64)
            require(maxMissedFrames >= 0)
            require(minimumMatchScore in 0.0..1.0)
            require(maximumCenterDistanceNormalized > 0.0)
            require(detectorIdentityWeight >= 0.0)
            require(appearanceWeight >= 0.0)
            require(
                appearanceHardMismatchThreshold in 0.0..1.0
            )
            require(
                appearanceHardGateQuality in 0.0..1.0
            )
            require(iouWeight >= 0.0)
            require(centerWeight >= 0.0)
            require(velocityWeight >= 0.0)
            require(skeletonWeight >= 0.0)
            require(confidenceWeight >= 0.0)
            require(worldJointContinuityBlend in 0.0..0.25)
            require(
                minimumWorldJointContinuityForHardGate in 0.0..1.0
            )
            require(velocitySmoothing in 0.0..1.0)
            require(predictionMaximumSpeedNormalizedPerSecond > 0.0)
            require(predictionMaximumAccelerationNormalizedPerSecond2 > 0.0)
            require(predictionMaximumJerkNormalizedPerSecond3 > 0.0)
            require(microMotionHistoryFrames >= 2)
            require(minimumMicroMotionDeltaNormalized > 0.0)
            require(microMotionGain in 0.0..1.0)
            require(crowdedSceneOverlapThreshold in 0.0..1.0)
            require(crowdedSceneMatchScoreBoost in 0.0..0.30)
        }
    }

    data class Bounds(
        val left: Double,
        val top: Double,
        val right: Double,
        val bottom: Double,
    ) {
        val center: Vec2
            get() = Vec2(
                (left + right) * 0.5,
                (top + bottom) * 0.5,
            )

        val area: Double
            get() = (right - left).coerceAtLeast(0.0) *
                (bottom - top).coerceAtLeast(0.0)
    }

    data class Joint(
        val landmarkId: Int,
        val x: Double,
        val y: Double,
        val confidence: Double,
        val probabilitySigmaXNormalized: Double = 0.0,
        val probabilitySigmaYNormalized: Double = 0.0,
        val probabilityCalibrationScore: Double = 1.0,
    ) {
        init {
            require(
                probabilitySigmaXNormalized.isFinite() &&
                    probabilitySigmaXNormalized >= 0.0
            )
            require(
                probabilitySigmaYNormalized.isFinite() &&
                    probabilitySigmaYNormalized >= 0.0
            )
            require(
                probabilityCalibrationScore.isFinite() &&
                    probabilityCalibrationScore in 0.0..1.0
            )
        }
    }

    /**
     * Provider-neutral 3D body-space evidence retained alongside 2D observations.
     * This is a learned/body-space prior and is never treated as calibrated rig coordinates.
     */
    data class WorldJoint(
        val landmarkId: Int,
        val xMeters: Double,
        val yMeters: Double,
        val zMeters: Double,
        val confidence: Double,
    ) {
        val positionMeters: com.nivukx.nox.core.model.Vec3
            get() = com.nivukx.nox.core.model.Vec3(xMeters, yMeters, zMeters)
    }

    data class Observation(
        val sourceObservationId: Int,
        val detectorTrackingId: Int = -1,
        val subjectType: SubjectType = SubjectType.HUMAN,
        val center: Vec2,
        val bounds: Bounds,
        val confidence: Double,
        val joints: List<Joint>,
        val shapeSignature: List<Double> = emptyList(),
        val appearanceSignature: List<Double> = emptyList(),
        val appearanceFingerprint: List<Double> = emptyList(),
        val appearanceQuality: Double = 0.0,
        val worldJoints: List<WorldJoint> = emptyList(),
    )

    data class MicroMotion(
        val timestampDeltaNanos: Long,
        val maxDisplacementNormalized: Double,
        val rmsDisplacementNormalized: Double,
        val velocityNormalizedPerSecond: Double,
        val accelerationNormalizedPerSecond2: Double,
        val jerkNormalizedPerSecond3: Double,
        val preservedSignal: Double,
        val historyDepth: Int,
    )

    data class Result(
        val actorId: Long,
        val sourceObservationId: Int,
        val detectorTrackingId: Int,
        val subjectType: SubjectType,
        val center: Vec2,
        val bounds: Bounds,
        val confidence: Double,
        val predicted: Boolean,
        val missedFrames: Int,
        val microMotion: MicroMotion,
        val velocityXNormalizedPerSecond: Double,
        val velocityYNormalizedPerSecond: Double,
        val joints: List<Joint>,
        val shapeSignature: List<Double> = emptyList(),
        val predictionConfidence: Double = 0.0,
        val meanMotionInnovationNormalized: Double = 0.0,
        val acceleratedTrackCount: Int = 0,
        val globalAssignmentBounded: Boolean = true,
        val bodySpaceContinuity: Double = 0.0,
        val identityAmbiguity: Double = 0.0,
        val appearanceFingerprint: List<Double> = emptyList(),
        val appearanceQuality: Double = 0.0,
        val worldJoints: List<WorldJoint> = emptyList(),
    )

    private data class HistorySample(
        val timestampNanos: Long,
        val center: Vec2,
        val joints: List<Joint>,
    )

    private data class MutableTrack(
        var actorId: Long,
        var subjectType: SubjectType,
        var center: Vec2,
        var bounds: Bounds,
        var velocity: Vec2,
        var confidence: Double,
        val predictor: AdaptiveMotionPredictionEngine,
        var missedFrames: Int,
        var ageFrames: Long,
        var detectorTrackingId: Int,
        var sourceObservationId: Int,
        var lastTimestampNanos: Long,
        var joints: List<Joint>,
        var shapeSignature: List<Double>,
        var appearanceSignature: List<Double>,
        var appearanceFingerprint: List<Double>,
        var appearanceQuality: Double,
        var latestWorldJoints: List<WorldJoint>,
        var bodySpaceContinuity: Double,
        var identityAmbiguity: Double,
        var lastMicroMotion: MicroMotion,
        val history: ArrayDeque<HistorySample> = ArrayDeque(),
    )

    private data class Candidate(
        val trackIndex: Int,
        val observationIndex: Int,
        val score: Double,
    )

    private val tracks = LinkedHashMap<Long, MutableTrack>()
    private var lastUpdateTimestampNanos: Long = Long.MIN_VALUE
    private var nextActorId = 1L
    @Volatile
    private var runtimeMaxActors: Int = config.maxActors
    @Volatile
    private var runtimeMinimumMatchScore: Double = config.minimumMatchScore
    @Volatile
    private var runtimeMicroMotionRetention: Double = config.microMotionGain
    @Volatile
    private var runtimeMaxMissedFrames: Int = config.maxMissedFrames
    @Volatile
    private var runtimeCrowdedSceneOverlapThreshold: Double =
        config.crowdedSceneOverlapThreshold
    @Volatile
    private var runtimeCrowdedSceneMatchScoreBoost: Double =
        config.crowdedSceneMatchScoreBoost

    fun configure(
        maxActors: Int = runtimeMaxActors,
        minimumMatchScore: Double = runtimeMinimumMatchScore,
        microMotionRetention: Double = runtimeMicroMotionRetention,
        maxMissedFrames: Int = runtimeMaxMissedFrames,
        crowdedSceneOverlapThreshold: Double =
            runtimeCrowdedSceneOverlapThreshold,
        crowdedSceneMatchScoreBoost: Double =
            runtimeCrowdedSceneMatchScoreBoost,
    ) {
        require(maxActors in 1..64)
        require(minimumMatchScore in 0.0..1.0)
        require(microMotionRetention in 0.0..1.0)
        require(maxMissedFrames in 0..240)
        runtimeMaxActors = maxActors
        runtimeMinimumMatchScore = minimumMatchScore
        runtimeMicroMotionRetention = microMotionRetention
        runtimeMaxMissedFrames = maxMissedFrames
        runtimeCrowdedSceneOverlapThreshold =
            crowdedSceneOverlapThreshold.coerceIn(0.0, 1.0)
        runtimeCrowdedSceneMatchScoreBoost =
            crowdedSceneMatchScoreBoost.coerceIn(0.0, 0.30)
    }

    fun configuration(): RuntimeConfiguration =
        RuntimeConfiguration(
            maxActors = runtimeMaxActors,
            minimumMatchScore = runtimeMinimumMatchScore,
            microMotionRetention = runtimeMicroMotionRetention,
        )

    data class RuntimeConfiguration(
        val maxActors: Int,
        val minimumMatchScore: Double,
        val microMotionRetention: Double,
    )

    @Synchronized
    fun update(
        timestampNanos: Long,
        observations: List<Observation>,
    ): List<Result> {
        require(timestampNanos >= 0L)

        if (lastUpdateTimestampNanos != Long.MIN_VALUE) {
            if (timestampNanos <= lastUpdateTimestampNanos) {
                return current()
            }
        }
        lastUpdateTimestampNanos = timestampNanos
        val sanitized = observations
            .asSequence()
            .filter { it.confidence.isFinite() }
            .filter { it.center.x.isFinite() && it.center.y.isFinite() }
            .filter { it.bounds.left.isFinite() && it.bounds.top.isFinite() }
            .filter { it.bounds.right.isFinite() && it.bounds.bottom.isFinite() }
            .map(::sanitize)
            .sortedWith(
                compareByDescending<Observation> { it.confidence }
                    .thenBy { it.sourceObservationId },
            )
            .take(runtimeMaxActors)
            .toList()

        if (tracks.isEmpty() && sanitized.isEmpty()) return emptyList()

        val trackList = tracks.values.toList()
        val assignments = globalAssignment(
            trackList = trackList,
            observations = sanitized,
            timestampNanos = timestampNanos,
        )

        val matchedTracks = assignments.map { it.trackIndex }.toHashSet()
        val matchedObservations = assignments.map { it.observationIndex }.toHashSet()

        assignments.forEach { match ->
            val track = trackList[match.trackIndex]
            updateTrack(
                track = track,
                observation = sanitized[match.observationIndex],
                timestampNanos = timestampNanos,
            )
        }

        trackList.indices
            .filter { it !in matchedTracks }
            .forEach { index ->
                predictTrack(trackList[index], timestampNanos)
            }

        sanitized.indices
            .filter { it !in matchedObservations }
            .forEach { index ->
                createTrack(sanitized[index], timestampNanos)
            }

        expireTracks()

        return tracks.values
            .sortedBy { it.actorId }
            .take(runtimeMaxActors)
            .map(::snapshot)
    }

    @Synchronized
    fun current(): List<Result> = tracks.values
        .sortedBy { it.actorId }
        .take(runtimeMaxActors)
        .map(::snapshot)

    @Synchronized
    fun clear() {
        tracks.values.forEach {
            it.predictor.reset()
            it.history.clear()
        }
        tracks.clear()
        nextActorId = 1L
        lastUpdateTimestampNanos = Long.MIN_VALUE
    }

    private fun sanitize(observation: Observation): Observation =
        observation.copy(
            center = Vec2(
                observation.center.x.coerceIn(0.0, 1.0),
                observation.center.y.coerceIn(0.0, 1.0),
            ),
            bounds = Bounds(
                left = observation.bounds.left.coerceIn(0.0, 1.0),
                top = observation.bounds.top.coerceIn(0.0, 1.0),
                right = observation.bounds.right.coerceIn(0.0, 1.0),
                bottom = observation.bounds.bottom.coerceIn(0.0, 1.0),
            ).let { bounds ->
                if (bounds.right >= bounds.left && bounds.bottom >= bounds.top) {
                    bounds
                } else {
                    Bounds(
                        left = min(bounds.left, bounds.right),
                        top = min(bounds.top, bounds.bottom),
                        right = max(bounds.left, bounds.right),
                        bottom = max(bounds.top, bounds.bottom),
                    )
                }
            },
            confidence = observation.confidence.coerceIn(0.0, 1.0),
            joints = observation.joints.map {
                it.copy(
                    x = it.x.coerceIn(0.0, 1.0),
                    y = it.y.coerceIn(0.0, 1.0),
                    confidence = it.confidence.coerceIn(0.0, 1.0),
                )
            },
            shapeSignature = observation.shapeSignature
                .map { value ->
                    if (value.isFinite()) value.coerceIn(-2.0, 2.0) else 0.0
                },
        )

    private fun globalAssignment(
        trackList: List<MutableTrack>,
        observations: List<Observation>,
        timestampNanos: Long,
    ): List<Candidate> {
        if (trackList.isEmpty() || observations.isEmpty()) return emptyList()

        /*
         * The previous exhaustive search was exact but exponential. A 50-actor scene made that
         * complexity unsuitable for production. Hungarian assignment gives deterministic O(n^3)
         * matching with explicit dummy assignments for unmatched tracks/observations.
         */
        val rowCount = trackList.size
        val columnCount = observations.size
        val n = max(rowCount, columnCount)
        val unmatchedCost =
            (1.0 - runtimeMinimumMatchScore)
                .coerceIn(0.05, 0.95)
        val forbiddenCost = 1_000_000.0

        val cost = Array(n) {
            DoubleArray(n) { unmatchedCost }
        }

        for (row in 0 until rowCount) {
            for (column in 0 until columnCount) {
                val observation =
                    observations[column]
                val score =
                    matchScore(
                        track = trackList[row],
                        observation = observation,
                        timestampNanos = timestampNanos,
                    )
                val crowdingRisk =
                    crowdingRisk(
                        observation = observation,
                        observations = observations,
                    )
                val adaptiveMinimumScore =
                    (
                        runtimeMinimumMatchScore +
                            runtimeCrowdedSceneMatchScoreBoost *
                                crowdingRisk
                    ).coerceIn(
                        runtimeMinimumMatchScore,
                        0.95,
                    )
                cost[row][column] =
                    if (
                        score >= adaptiveMinimumScore &&
                        score.isFinite()
                    ) {
                        1.0 - score
                    } else {
                        forbiddenCost
                    }
            }
        }

        val assignment =
            hungarianMinimum(
                matrix = cost,
            )

        return assignment
            .mapIndexed { trackIndex, observationIndex ->
                if (
                    trackIndex >= rowCount ||
                    observationIndex !in 0 until columnCount
                ) {
                    null
                } else {
                    val selectedCost =
                        cost[trackIndex][observationIndex]

                    if (
                        selectedCost >= unmatchedCost ||
                        selectedCost >= forbiddenCost
                    ) {
                        null
                    } else {
                        Candidate(
                            trackIndex = trackIndex,
                            observationIndex = observationIndex,
                            score =
                                (
                                    1.0 - selectedCost
                                ).coerceIn(0.0, 1.0),
                        )
                    }
                }
            }
            .filterNotNull()
    }

    /**
     * Deterministic Hungarian minimization with one-based potentials.
     */
    private fun hungarianMinimum(
        matrix: Array<DoubleArray>,
    ): IntArray {
        val n = matrix.size
        if (n == 0) return IntArray(0)

        val u = DoubleArray(n + 1)
        val v = DoubleArray(n + 1)
        val p = IntArray(n + 1)
        val way = IntArray(n + 1)

        for (i in 1..n) {
            p[0] = i
            var column0 = 0
            val minValue =
                DoubleArray(n + 1) {
                    Double.POSITIVE_INFINITY
                }
            val used = BooleanArray(n + 1)

            do {
                used[column0] = true
                val row0 = p[column0]
                var delta = Double.POSITIVE_INFINITY
                var column1 = 0

                for (column in 1..n) {
                    if (used[column]) continue
                    val rawCost =
                        matrix[row0 - 1][column - 1]
                    val current =
                        rawCost -
                            u[row0] -
                            v[column]

                    if (current < minValue[column] - 1e-12) {
                        minValue[column] = current
                        way[column] = column0
                    }

                    if (
                        minValue[column] < delta - 1e-12 ||
                        (
                            abs(minValue[column] - delta) <= 1e-12 &&
                                column < column1
                            )
                    ) {
                        delta = minValue[column]
                        column1 = column
                    }
                }

                for (column in 0..n) {
                    if (used[column]) {
                        u[p[column]] += delta
                        v[column] -= delta
                    } else {
                        minValue[column] -= delta
                    }
                }

                column0 = column1
            } while (p[column0] != 0)

            do {
                val column1 = way[column0]
                p[column0] = p[column1]
                column0 = column1
            } while (column0 != 0)
        }

        val assignment = IntArray(n) { -1 }
        for (column in 1..n) {
            if (p[column] != 0) {
                assignment[p[column] - 1] = column - 1
            }
        }
        return assignment
    }


    private fun crowdingRisk(
        observation: Observation,
        observations: List<Observation>,
    ): Double {
        if (observations.size <= 1) return 0.0

        val strongestOverlap =
            observations
                .asSequence()
                .filter { it !== observation }
                .map { other ->
                    iou(
                        observation.bounds,
                        other.bounds,
                    )
                }
                .maxOrNull()
                ?: 0.0

        if (
            strongestOverlap <=
                runtimeCrowdedSceneOverlapThreshold
        ) {
            return 0.0
        }

        return (
            (
                strongestOverlap -
                    runtimeCrowdedSceneOverlapThreshold
            ) /
                (
                    1.0 -
                        runtimeCrowdedSceneOverlapThreshold
                ).coerceAtLeast(1e-6)
        ).coerceIn(0.0, 1.0)
    }

    private fun matchScore(
        track: MutableTrack,
        observation: Observation,
        timestampNanos: Long,
    ): Double {
        if (track.subjectType != observation.subjectType) return 0.0

        val predictedCenter = predictCenter(track, timestampNanos)
        val centerDistance = predictedCenter.distanceTo(observation.center)
        if (centerDistance > config.maximumCenterDistanceNormalized &&
            track.missedFrames == 0) return 0.0

        val diagonal = sqrt(
            max(
                (track.bounds.right - track.bounds.left) *
                    (track.bounds.right - track.bounds.left) +
                    (track.bounds.bottom - track.bounds.top) *
                    (track.bounds.bottom - track.bounds.top),
                1e-12,
            ),
        )
        val centerScore = (1.0 - centerDistance / diagonal).coerceIn(0.0, 1.0)
        val iouScore = iou(track.bounds, observation.bounds)
        val velocityScore = velocityContinuity(track, observation, timestampNanos)
        val skeletonScore = skeletonContinuity(track.joints, observation.joints)
        val shapeScore =
            signatureSimilarity(
                track.shapeSignature,
                observation.shapeSignature,
            )
        val appearanceSimilarity =
            centeredSignatureSimilarity(
                track.appearanceSignature,
                observation.appearanceSignature,
            )
        val appearanceFingerprintSimilarity =
            signatureSimilarity(
                track.appearanceFingerprint,
                observation.appearanceFingerprint,
            )
        val appearanceQuality =
            min(
                track.appearanceQuality,
                observation.appearanceQuality,
            ).coerceIn(0.0, 1.0)
        val appearanceEvidenceAvailable =
            appearanceQuality >= 0.20 &&
                track.appearanceSignature.isNotEmpty() &&
                observation.appearanceSignature.isNotEmpty()

        val appearanceDescriptorAgreement =
            if (appearanceEvidenceAvailable) {
                if (
                    track.appearanceFingerprint.isNotEmpty() &&
                        observation.appearanceFingerprint.isNotEmpty()
                ) {
                    (
                        appearanceSimilarity * 0.62 +
                            appearanceFingerprintSimilarity * 0.38
                    ).coerceIn(0.0, 1.0)
                } else {
                    appearanceSimilarity.coerceIn(0.0, 1.0)
                }
            } else {
                1.0
            }

        if (
            appearanceQuality >= config.appearanceHardGateQuality &&
            appearanceEvidenceAvailable &&
            appearanceDescriptorAgreement <
                config.appearanceHardMismatchThreshold
        ) {
            return 0.0
        }

        val appearanceEvidence =
            if (
                appearanceEvidenceAvailable
            ) {
                appearanceDescriptorAgreement *
                    appearanceQuality
            } else {
                0.0
            }
        val detectorIdScore = if (
            track.detectorTrackingId >= 0 &&
            observation.detectorTrackingId >= 0 &&
            track.detectorTrackingId == observation.detectorTrackingId
        ) 1.0 else 0.0

        val baseScore =
            (
                config.iouWeight * iouScore +
                    config.centerWeight * centerScore +
                    config.velocityWeight * velocityScore +
                    config.skeletonWeight * skeletonScore +
                    (
                        config.skeletonWeight * 0.55
                    ) * shapeScore +
                    config.appearanceWeight * appearanceEvidence +
                    config.detectorIdentityWeight * detectorIdScore +
                    config.confidenceWeight * observation.confidence
                ).coerceIn(0.0, 1.0)

        /*
         * Dense probabilistic landmarks carry uncertainty that should directly shape association.
         * A point with a broad heatmap posterior must not be treated as equally exact as a tightly
         * localized point. This factor is multiplicative so it cannot overpower strong appearance,
         * geometry or detector-identity evidence, but it can suppress improbable assignments.
         */
        val uncertaintyCompatibility =
            probabilisticJointContinuity(
                previous = track.joints,
                current = observation.joints,
            )

        val uncertaintyAdjustedScore =
            if (uncertaintyCompatibility >= 0.0) {
                baseScore *
                    (
                        0.78 +
                            0.22 *
                                uncertaintyCompatibility
                    )
            } else {
                baseScore
            }

        val bodySpaceScore =
            bodySpaceContinuity(
                previous = track.latestWorldJoints,
                current = observation.worldJoints,
            )

        if (
            bodySpaceScore >= 0.0 &&
                track.latestWorldJoints.size >= 4 &&
                observation.worldJoints.size >= 4 &&
                track.missedFrames <= 1 &&
                bodySpaceScore <
                    config.minimumWorldJointContinuityForHardGate
        ) {
            return 0.0
        }

        return if (bodySpaceScore >= 0.0) {
            (
                uncertaintyAdjustedScore *
                    (
                        1.0 -
                            config.worldJointContinuityBlend
                    ) +
                    bodySpaceScore *
                        config.worldJointContinuityBlend
            ).coerceIn(0.0, 1.0)
        } else {
            uncertaintyAdjustedScore
        }
    }

    private fun probabilisticJointContinuity(
        previous: List<Joint>,
        current: List<Joint>,
    ): Double {
        if (previous.isEmpty() || current.isEmpty()) return -1.0

        val currentById =
            current.associateBy { it.landmarkId }
        var weightedScore = 0.0
        var totalWeight = 0.0

        previous.forEach { before ->
            val after =
                currentById[before.landmarkId]
                    ?: return@forEach

            val sigmaX =
                (
                    before.probabilitySigmaXNormalized *
                        before.probabilitySigmaXNormalized +
                        after.probabilitySigmaXNormalized *
                            after.probabilitySigmaXNormalized
                    ).let(::sqrt)

            val sigmaY =
                (
                    before.probabilitySigmaYNormalized *
                        before.probabilitySigmaYNormalized +
                        after.probabilitySigmaYNormalized *
                            after.probabilitySigmaYNormalized
                    ).let(::sqrt)

            val sigma =
                max(
                    sqrt(
                        sigmaX * sigmaX +
                            sigmaY * sigmaY,
                    ),
                    1e-5,
                )

            val uncertaintyEvidenceAvailable =
                before.probabilitySigmaXNormalized > 0.0 ||
                    before.probabilitySigmaYNormalized > 0.0 ||
                    after.probabilitySigmaXNormalized > 0.0 ||
                    after.probabilitySigmaYNormalized > 0.0

            if (!uncertaintyEvidenceAvailable) {
                return@forEach
            }

            val distance =
                sqrt(
                    (before.x - after.x) *
                        (before.x - after.x) +
                        (before.y - after.y) *
                        (before.y - after.y),
                )

            val normalizedResidual =
                distance / sigma

            val robustScore =
                exp(
                    -0.5 *
                        min(
                            normalizedResidual *
                                normalizedResidual,
                            36.0,
                        ),
                ).coerceIn(0.0, 1.0)

            val evidenceWeight =
                sqrt(
                    max(
                        before.confidence *
                            after.confidence,
                        1e-6,
                    ),
                ) *
                    min(
                        before.probabilityCalibrationScore,
                        after.probabilityCalibrationScore,
                    ).coerceIn(0.0, 1.0)

            weightedScore += robustScore * evidenceWeight
            totalWeight += evidenceWeight
        }

        if (totalWeight <= 1e-9) return -1.0
        return (weightedScore / totalWeight)
            .coerceIn(0.0, 1.0)
    }

    private fun updateTrack(
        track: MutableTrack,
        observation: Observation,
        timestampNanos: Long,
    ) {
        val assignmentAmbiguity =
            assignmentAmbiguity(
                track = track,
                observation = observation,
                timestampNanos = timestampNanos,
            )
        val predictionUpdate =
            track.predictor.update(
                measurement = observation.center,
                timestampNanos = timestampNanos,
                observationConfidence = observation.confidence,
            )
        val predictorState = predictionUpdate.state
        val blendedVelocity = track.velocity.lerp(
            predictorState.velocity,
            config.velocitySmoothing,
        )

        val microMotion = analyzeMicroMotion(
            track = track,
            observation = observation,
            timestampNanos = timestampNanos,
        )

        track.velocity = blendedVelocity
        track.center = predictorState.position
        track.bounds = blendBounds(track.bounds, observation.bounds, 0.84)
        track.confidence = (
            track.confidence * 0.30 +
                observation.confidence * 0.70
        ).coerceIn(0.0, 1.0)
        track.missedFrames = 0
        track.ageFrames += 1
        track.detectorTrackingId = observation.detectorTrackingId
        track.sourceObservationId = observation.sourceObservationId
        track.lastTimestampNanos = timestampNanos
        val previousWorldJoints =
            track.latestWorldJoints

        val bodySpaceScore =

            bodySpaceContinuity(
                previous = previousWorldJoints,
                current = observation.worldJoints,
            )

        track.joints = observation.joints
        track.bodySpaceContinuity =
            bodySpaceScore.takeIf { it >= 0.0 } ?: 0.0

        track.latestWorldJoints = observation.worldJoints
        track.identityAmbiguity =
            assignmentAmbiguity
        track.shapeSignature =
            blendSignature(
                previous = track.shapeSignature,
                current = observation.shapeSignature,
                alpha = 0.34,
            )
        if (
            observation.appearanceQuality >= 0.20 &&
                observation.appearanceSignature.isNotEmpty()
        ) {
            track.appearanceSignature =
                blendSignature(
                    previous = track.appearanceSignature,
                    current = observation.appearanceSignature,
                    alpha =
                        (
                            0.18 +
                                observation.appearanceQuality * 0.34
                        ).coerceIn(0.18, 0.52),
                )
            track.appearanceFingerprint =
                blendSignature(
                    previous = track.appearanceFingerprint,
                    current = observation.appearanceFingerprint,
                    alpha =
                        (
                            0.20 +
                                observation.appearanceQuality * 0.30
                        ).coerceIn(0.20, 0.50),
                )
            track.appearanceQuality =
                (
                    track.appearanceQuality * 0.54 +
                        observation.appearanceQuality * 0.46
                    ).coerceIn(0.0, 1.0)
        } else {
            track.appearanceQuality =
                (
                    track.appearanceQuality * 0.96
                ).coerceIn(0.0, 1.0)
            if (track.appearanceFingerprint.isNotEmpty()) {
                track.appearanceFingerprint =
                    track.appearanceFingerprint
                        .map { it.coerceIn(0.0, 1.0) }
            }
        }
        track.lastMicroMotion = microMotion
        appendHistory(track, timestampNanos, observation.center, observation.joints)
    }

    private fun createTrack(
        observation: Observation,
        timestampNanos: Long,
    ) {
        if (tracks.size >= config.maxActors) return
        val id = nextActorId++
        val predictor =
            AdaptiveMotionPredictionEngine(
                AdaptiveMotionPredictionEngine.Config(
                    velocityGain =
                        config.velocitySmoothing
                            .coerceIn(0.05, 0.90),
                    maximumSpeedNormalizedPerSecond =
                        config.predictionMaximumSpeedNormalizedPerSecond,
                    maximumAccelerationNormalizedPerSecond2 =
                        config.predictionMaximumAccelerationNormalizedPerSecond2,
                    maximumJerkNormalizedPerSecond3 =
                        config.predictionMaximumJerkNormalizedPerSecond3,
                    maximumDtSeconds = 0.20,
                ),
            )
        predictor.initialize(
            measurement = observation.center,
            timestampNanos = timestampNanos,
            observationConfidence = observation.confidence,
        )

        val track = MutableTrack(
            actorId = id,
            subjectType = observation.subjectType,
            center = observation.center,
            bounds = observation.bounds,
            velocity = Vec2(0.0, 0.0),
            confidence = observation.confidence,
            predictor = predictor,
            missedFrames = 0,
            ageFrames = 1L,
            detectorTrackingId = observation.detectorTrackingId,
            sourceObservationId = observation.sourceObservationId,
            lastTimestampNanos = timestampNanos,
            joints = observation.joints,
            shapeSignature = observation.shapeSignature,
            appearanceSignature = observation.appearanceSignature,
            appearanceFingerprint = observation.appearanceFingerprint,
            appearanceQuality =
                observation.appearanceQuality.coerceIn(0.0, 1.0),
            latestWorldJoints = observation.worldJoints,
            bodySpaceContinuity = 0.0,
            identityAmbiguity = 0.0,
            lastMicroMotion = zeroMicroMotion(),
        )
        track.history.addLast(
            HistorySample(timestampNanos, observation.center, observation.joints),
        )
        tracks[id] = track
    }

    private fun predictTrack(
        track: MutableTrack,
        timestampNanos: Long,
    ) {
        val prediction =
            track.predictor.propagateOcclusion(timestampNanos)
        val dt = deltaSeconds(track.lastTimestampNanos, timestampNanos)
        val predictedCenter = prediction.position
        track.center = predictedCenter
        track.velocity = prediction.velocity
        val dx = predictedCenter.x - track.bounds.center.x
        val dy = predictedCenter.y - track.bounds.center.y
        track.bounds = Bounds(
            left = track.bounds.left + dx,
            top = track.bounds.top + dy,
            right = track.bounds.right + dx,
            bottom = track.bounds.bottom + dy,
        )
        track.confidence = (track.confidence * 0.82).coerceIn(0.0, 1.0)
        track.missedFrames += 1
        track.lastTimestampNanos = timestampNanos
        track.lastMicroMotion = MicroMotion(
            timestampDeltaNanos = (dt * 1e9).toLong().coerceAtLeast(0L),
            maxDisplacementNormalized = 0.0,
            rmsDisplacementNormalized = 0.0,
            velocityNormalizedPerSecond = prediction.velocity.norm(),
            accelerationNormalizedPerSecond2 =
                prediction.acceleration.norm(),
            jerkNormalizedPerSecond3 = 0.0,
            preservedSignal = 0.0,
            historyDepth = track.history.size,
        )
    }

    private fun expireTracks() {
        tracks.values.removeIf { it.missedFrames > runtimeMaxMissedFrames }
    }

    private fun appendHistory(
        track: MutableTrack,
        timestampNanos: Long,
        center: Vec2,
        joints: List<Joint>,
    ) {
        track.history.addLast(HistorySample(timestampNanos, center, joints))
        while (track.history.size > config.microMotionHistoryFrames) {
            track.history.removeFirst()
        }
    }

    private fun analyzeMicroMotion(
        track: MutableTrack,
        observation: Observation,
        timestampNanos: Long,
    ): MicroMotion {
        val previous = track.history.lastOrNull()
            ?: return zeroMicroMotion()
        val deltaNanos = (timestampNanos - previous.timestampNanos).coerceAtLeast(1L)
        val dt = deltaNanos / 1e9

        val displacements = mergeJoints(previous.joints, observation.joints)
            .map { it.first.distanceTo(it.second) }
            .filter(Double::isFinite)

        if (displacements.isEmpty()) {
            return zeroMicroMotion(deltaNanos)
        }

        val maxDisplacement = displacements.maxOrNull() ?: 0.0
        val rms = sqrt(displacements.map { it * it }.average())
        val velocity = rms / dt

        val previousVelocity = if (track.lastMicroMotion.timestampDeltaNanos > 0L) {
            track.lastMicroMotion.velocityNormalizedPerSecond
        } else 0.0
        val acceleration = abs(velocity - previousVelocity) / dt

        val previousAcceleration = track.lastMicroMotion.accelerationNormalizedPerSecond2
        val jerk = abs(acceleration - previousAcceleration) / dt

        val preserved = (
            rms / config.minimumMicroMotionDeltaNormalized
        ).coerceIn(0.0, 1.0) * runtimeMicroMotionRetention

        return MicroMotion(
            timestampDeltaNanos = deltaNanos,
            maxDisplacementNormalized = maxDisplacement,
            rmsDisplacementNormalized = rms,
            velocityNormalizedPerSecond = velocity,
            accelerationNormalizedPerSecond2 = acceleration,
            jerkNormalizedPerSecond3 = jerk,
            preservedSignal = preserved.coerceIn(0.0, 1.0),
            historyDepth = track.history.size,
        )
    }

    private fun bodySpaceContinuity(
        previous: List<WorldJoint>,
        current: List<WorldJoint>,
    ): Double {
        if (previous.isEmpty() || current.isEmpty()) {
            return -1.0
        }

        val previousById =
            previous
                .filter {
                    it.xMeters.isFinite() &&
                        it.yMeters.isFinite() &&
                        it.zMeters.isFinite() &&
                        it.confidence.isFinite()
                }
                .associateBy { it.landmarkId }

        val currentById =
            current
                .filter {
                    it.xMeters.isFinite() &&
                        it.yMeters.isFinite() &&
                        it.zMeters.isFinite() &&
                        it.confidence.isFinite()
                }
                .associateBy { it.landmarkId }

        val common =
            previousById.keys
                .intersect(currentById.keys)
                .take(24)

        if (common.size < 4) return -1.0

        val previousPoints =
            common.map { previousById.getValue(it).positionMeters }
        val currentPoints =
            common.map { currentById.getValue(it).positionMeters }

        val previousCenter = centroid(previousPoints)
        val currentCenter = centroid(currentPoints)

        val previousScale =
            robustBodyScale(previousPoints, previousCenter)
        val currentScale =
            robustBodyScale(currentPoints, currentCenter)

        if (
            previousScale <= 1e-6 ||
                currentScale <= 1e-6
        ) {
            return -1.0
        }

        var weightedError = 0.0
        var weightSum = 0.0

        common.forEachIndexed { index, landmarkId ->
            val previousPoint =
                previousPoints[index]
                    .minus(previousCenter)
                    .div(previousScale)
            val currentPoint =
                currentPoints[index]
                    .minus(currentCenter)
                    .div(currentScale)

            val confidence =
                min(
                    previousById
                        .getValue(landmarkId)
                        .confidence,
                    currentById
                        .getValue(landmarkId)
                        .confidence,
                ).coerceIn(0.05, 1.0)

            val delta =
                previousPoint
                    .minus(currentPoint)

            val error =
                delta.norm()

            weightedError +=
                error *
                    confidence
            weightSum += confidence
        }

        if (weightSum <= 1e-9) return -1.0

        val normalizedError =
            (
                weightedError /
                    weightSum
            ).coerceIn(0.0, 4.0)

        return (
            1.0 -
                normalizedError /
                    1.50
            ).coerceIn(0.0, 1.0)
    }

    private fun centroid(
        points: List<com.nivukx.nox.core.model.Vec3>,
    ): com.nivukx.nox.core.model.Vec3 {
        if (points.isEmpty()) {
            return com.nivukx.nox.core.model.Vec3(
                0.0,
                0.0,
                0.0,
            )
        }

        val count = points.size.toDouble()
        return com.nivukx.nox.core.model.Vec3(
            x = points.sumOf { it.x } / count,
            y = points.sumOf { it.y } / count,
            z = points.sumOf { it.z } / count,
        )
    }

    private fun robustBodyScale(
        points: List<com.nivukx.nox.core.model.Vec3>,
        center: com.nivukx.nox.core.model.Vec3,
    ): Double {
        val distances =
            points
                .map { it.distanceTo(center) }
                .filter {
                    it.isFinite() &&
                        it > 1e-6
                }
                .sorted()

        if (distances.isEmpty()) return 0.0
        val middle = distances.size / 2
        return if (distances.size % 2 == 0) {
            (
                distances[middle - 1] +
                    distances[middle]
                ) * 0.5
        } else {
            distances[middle]
        }
    }

    private fun assignmentAmbiguity(
        track: MutableTrack,
        observation: Observation,
        timestampNanos: Long,
    ): Double {
        val primary =
            matchScore(
                track = track,
                observation = observation,
                timestampNanos = timestampNanos,
            )

        if (!primary.isFinite()) return 1.0

        val alternatives =
            tracks.values
                .asSequence()
                .filter {
                    it.actorId != track.actorId &&
                        it.subjectType ==
                            observation.subjectType
                }
                .map {
                    matchScore(
                        track = it,
                        observation = observation,
                        timestampNanos = timestampNanos,
                    )
                }
                .filter(Double::isFinite)
                .maxOrNull()
                ?: 0.0

        return (
            1.0 -
                (
                    primary -
                        alternatives
                    ).coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)
    }

    private fun snapshot(track: MutableTrack): Result =
        Result(
            actorId = track.actorId,
            sourceObservationId = track.sourceObservationId,
            detectorTrackingId = track.detectorTrackingId,
            subjectType = track.subjectType,
            center = track.center,
            bounds = track.bounds,
            confidence = track.confidence,
            predicted = track.missedFrames > 0,
            missedFrames = track.missedFrames,
            microMotion = track.lastMicroMotion,
            predictionConfidence =
                track.predictor.snapshot().confidence,
            meanMotionInnovationNormalized =
                (
                    track.lastMicroMotion.velocityNormalizedPerSecond *
                        0.01
                    ).coerceIn(0.0, 2.0),
            acceleratedTrackCount =
                if (
                    track.predictor.snapshot().acceleration.norm() >
                        config.predictionMaximumAccelerationNormalizedPerSecond2 * 0.20
                ) 1 else 0,
            velocityXNormalizedPerSecond = track.velocity.x.toDouble(),
            velocityYNormalizedPerSecond = track.velocity.y.toDouble(),
            joints = track.joints,
            shapeSignature = track.shapeSignature,
            appearanceFingerprint =
                track.appearanceFingerprint.toList(),
            appearanceQuality =
                track.appearanceQuality.coerceIn(0.0, 1.0),
            worldJoints = track.latestWorldJoints,
            bodySpaceContinuity =
                track.bodySpaceContinuity.coerceIn(0.0, 1.0),
            identityAmbiguity =
                track.identityAmbiguity.coerceIn(0.0, 1.0),
        )

    private fun predictCenter(track: MutableTrack, timestampNanos: Long): Vec2 {
        return track.center + track.velocity * deltaSeconds(track.lastTimestampNanos, timestampNanos)
    }

    private fun velocityContinuity(
        track: MutableTrack,
        observation: Observation,
        timestampNanos: Long,
    ): Double {
        val dt = deltaSeconds(track.lastTimestampNanos, timestampNanos)
        if (dt <= 0.0) return 0.5
        val predicted = predictCenter(track, timestampNanos)
        val residual = predicted.distanceTo(observation.center)
        val speedScale = max(track.velocity.norm() * dt, 0.02)
        return (
            1.0 -
                residual /
                    max(speedScale * 2.5, 0.04)
        ).coerceIn(0.0, 1.0)
    }

    private fun skeletonContinuity(
        previous: List<Joint>,
        current: List<Joint>,
    ): Double {
        val currentMap = current.associateBy { it.landmarkId }
        val distances = previous.mapNotNull { old ->
            val now = currentMap[old.landmarkId] ?: return@mapNotNull null
            if (old.confidence < 0.10 || now.confidence < 0.10) return@mapNotNull null
            oldDistance(old, now)
        }
        if (distances.isEmpty()) return 0.5
        val mean = distances.average()
        return (1.0 - mean / 0.18).coerceIn(0.0, 1.0)
    }

    private fun oldDistance(a: Joint, b: Joint): Double =
        sqrt(
            (a.x - b.x) * (a.x - b.x) +
                (a.y - b.y) * (a.y - b.y),
        )

    private fun mergeJoints(
        previous: List<Joint>,
        current: List<Joint>,
    ): List<Pair<Vec2, Vec2>> {
        val map = current.associateBy { it.landmarkId }
        return previous.mapNotNull { old ->
            val now = map[old.landmarkId] ?: return@mapNotNull null
            if (old.confidence < 0.05 || now.confidence < 0.05) return@mapNotNull null
            Vec2(old.x, old.y) to Vec2(now.x, now.y)
        }
    }

    private fun deltaSeconds(fromNanos: Long, toNanos: Long): Double =
        (toNanos - fromNanos).coerceAtLeast(1L) / 1e9

    private fun blendBounds(a: Bounds, b: Bounds, alpha: Double): Bounds =
        Bounds(
            left = a.left * (1.0 - alpha) + b.left * alpha,
            top = a.top * (1.0 - alpha) + b.top * alpha,
            right = a.right * (1.0 - alpha) + b.right * alpha,
            bottom = a.bottom * (1.0 - alpha) + b.bottom * alpha,
        )

    private fun iou(a: Bounds, b: Bounds): Double {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection =
            (right - left).coerceAtLeast(0.0) *
                (bottom - top).coerceAtLeast(0.0)
        val union = a.area + b.area - intersection
        return if (union <= 1e-9) 0.0 else (intersection / union).coerceIn(0.0, 1.0)
    }

    private fun centeredSignatureSimilarity(
        previous: List<Double>,
        current: List<Double>,
    ): Double {
        if (previous.isEmpty() || current.isEmpty()) return 0.5

        val size = minOf(previous.size, current.size)
        if (size == 0) return 0.5

        var meanA = 0.0
        var meanB = 0.0
        var count = 0
        for (index in 0 until size) {
            val a = previous[index]
            val b = current[index]
            if (!a.isFinite() || !b.isFinite()) continue
            meanA += a
            meanB += b
            count += 1
        }
        if (count == 0) return 0.5
        meanA /= count.toDouble()
        meanB /= count.toDouble()

        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (index in 0 until size) {
            val a = previous[index]
            val b = current[index]
            if (!a.isFinite() || !b.isFinite()) continue
            val ca = a - meanA
            val cb = b - meanB
            dot += ca * cb
            normA += ca * ca
            normB += cb * cb
        }

        if (normA <= 1e-12 || normB <= 1e-12) {
            return signatureSimilarity(previous, current)
        }

        return (
            dot /
                sqrt(normA * normB).coerceAtLeast(1e-12)
        ).coerceIn(-1.0, 1.0) * 0.5 + 0.5
    }

    private fun signatureSimilarity(
        previous: List<Double>,
        current: List<Double>,
    ): Double {
        if (previous.isEmpty() || current.isEmpty()) return 0.5

        val size = minOf(previous.size, current.size)
        if (size == 0) return 0.5

        var dot = 0.0
        var normA = 0.0
        var normB = 0.0

        for (index in 0 until size) {
            val a = previous[index]
            val b = current[index]
            if (!a.isFinite() || !b.isFinite()) continue
            dot += a * b
            normA += a * a
            normB += b * b
        }

        if (normA <= 1e-12 || normB <= 1e-12) return 0.5

        return (
            dot /
                sqrt(
                    normA * normB,
                ).coerceAtLeast(1e-12)
            ).coerceIn(-1.0, 1.0) * 0.5 + 0.5
    }

    private fun blendSignature(
        previous: List<Double>,
        current: List<Double>,
        alpha: Double,
    ): List<Double> {
        if (previous.isEmpty()) return current.toList()
        if (current.isEmpty()) return previous.toList()

        val size = minOf(previous.size, current.size)
        return buildList(size) {
            for (index in 0 until size) {
                add(
                    previous[index] * (1.0 - alpha) +
                        current[index] * alpha,
                )
            }
        }
    }

    private fun zeroMicroMotion(deltaNanos: Long = 0L) = MicroMotion(
        timestampDeltaNanos = deltaNanos,
        maxDisplacementNormalized = 0.0,
        rmsDisplacementNormalized = 0.0,
        velocityNormalizedPerSecond = 0.0,
        accelerationNormalizedPerSecond2 = 0.0,
        jerkNormalizedPerSecond3 = 0.0,
        preservedSignal = 0.0,
        historyDepth = 0,
    )
}
