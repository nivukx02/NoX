package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.MultiActorPoseAssociationEngine
import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Geometry-aware cross-camera identity association.
 *
 * Each phone creates detector-local actor IDs. Those IDs are not globally meaningful. This engine
 * keeps a reference-camera actor table and compares incoming actors using:
 *  - calibrated cross-camera triangulation of stable body joints,
 *  - cross-view reprojection residuals,
 *  - normalized skeleton-shape similarity,
 *  - confidence and temporal continuity.
 *
 * The engine only assigns a global identity when the evidence crosses an explicit threshold. An
 * unmatched actor receives a new bounded global identity instead of being silently attached to the
 * nearest existing track.
 */
class CrossCameraActorIdentityFusionEngine(
    private val calibrationProvider: () -> CalibrationSnapshot?,
    private val maximumReferenceTracks: Int = 128,
    private val maximumSourceBindings: Int = 256,
    private val appearanceEmaAlpha: Double = 0.18,
    private val identityAmbiguityMargin: Double = 0.06,
    private val identityHysteresisBonus: Double = 0.08,
) {
    init {
        require(maximumReferenceTracks in 16..256)
        require(maximumSourceBindings >= maximumReferenceTracks)
        require(
            appearanceEmaAlpha.isFinite() &&
                appearanceEmaAlpha in 0.01..1.0,
        )
        require(
            identityAmbiguityMargin.isFinite() &&
                identityAmbiguityMargin in 0.0..0.50,
        )
        require(
            identityHysteresisBonus.isFinite() &&
                identityHysteresisBonus in 0.0..0.25,
        )
    }

    data class ActorObservation(
        val localActorId: Long,
        val confidence: Double,
        val inputWidth: Int,
        val inputHeight: Int,
        val joints: List<MultiActorPoseAssociationEngine.Joint>,
        val worldJoints: List<MultiActorPoseAssociationEngine.WorldJoint> = emptyList(),
        val appearanceSignature: List<Double> = emptyList(),
        val appearanceFingerprint: List<Double> = emptyList(),
        val appearanceQuality: Double = 0.0,
        val timestampNanos: Long,
        val cameraId: DeviceId,
    )

    data class IdentityAssignment(
        val sourceCameraId: DeviceId,
        val localActorId: Long,
        val globalActorId: Long,
        val score: Double,
        val triangulatedJointCount: Int,
        val reprojectionScore: Double,
        val shapeScore: Double,
        val temporalScore: Double,
        val worldShapeScore: Double = 0.0,
        val appearanceScore: Double = 0.0,
        /** Symmetric calibrated two-view epipolar evidence. */
        val epipolarConsistencyScore: Double = 0.0,
        val accepted: Boolean,
        val warnings: List<String>,
    )

    data class Result(
        val assignments: List<IdentityAssignment>,
        val unmatchedCount: Int,
        val warnings: List<String>,
    )

    private data class ReferenceTrack(
        val globalActorId: Long,
        val sourceCameraId: DeviceId,
        val localActorId: Long,
        var timestampNanos: Long,
        var confidence: Double,
        var inputWidth: Int,
        var inputHeight: Int,
        var joints: List<MultiActorPoseAssociationEngine.Joint>,
        var worldJoints: List<MultiActorPoseAssociationEngine.WorldJoint>,
        var appearanceSignature: List<Double>,
        var appearanceFingerprint: List<Double>,
        var appearanceQuality: Double,
    )

    private val references = LinkedHashMap<Long, ReferenceTrack>()
    private val sourceToGlobal = HashMap<Pair<DeviceId, Long>, Long>()
    private val triangulator =
        TriangulationEngine()
    private val symmetricEpipolarConsistencyEngine =
        SymmetricEpipolarConsistencyEngine()
    private val globalAssignmentEngine =
        MaximumWeightBipartiteAssignmentEngine(
            MaximumWeightBipartiteAssignmentEngine.Config(
                maxItems = 128,
            ),
        )
    private var nextGlobalId = 1_000_000L

    @Synchronized
    fun rememberReference(
        observations: List<ActorObservation>,
    ) {
        observations.forEach { observation ->
            val existing =
                sourceToGlobal[observation.cameraId to observation.localActorId]
            val globalId =
                existing ?: observation.localActorId.also {
                    sourceToGlobal[
                        observation.cameraId to
                            observation.localActorId
                    ] = it
                }

            val previous = references[globalId]
            if (
                previous == null ||
                observation.timestampNanos >= previous.timestampNanos
            ) {
                references[globalId] =
                    if (previous == null) {
                        ReferenceTrack(
                            globalActorId = globalId,
                            sourceCameraId = observation.cameraId,
                            localActorId = observation.localActorId,
                            timestampNanos = observation.timestampNanos,
                            confidence = observation.confidence,
                            inputWidth = observation.inputWidth,
                            inputHeight = observation.inputHeight,
                            joints = observation.joints,
                            worldJoints = observation.worldJoints,
                            appearanceSignature =
                                observation.appearanceSignature.toList(),
                            appearanceFingerprint =
                                observation.appearanceFingerprint.toList(),
                            appearanceQuality =
                                observation.appearanceQuality
                                    .coerceIn(0.0, 1.0),
                        )
                    } else {
                        previous.copy(
                            sourceCameraId = observation.cameraId,
                            localActorId = observation.localActorId,
                            timestampNanos = observation.timestampNanos,
                            confidence =
                                (
                                    previous.confidence *
                                        (1.0 - appearanceEmaAlpha) +
                                        observation.confidence *
                                        appearanceEmaAlpha
                                    ).coerceIn(0.0, 1.0),
                            inputWidth = observation.inputWidth,
                            inputHeight = observation.inputHeight,
                            joints = observation.joints,
                            worldJoints = observation.worldJoints,
                            appearanceSignature =
                                blendAppearanceVector(
                                    previous.appearanceSignature,
                                    observation.appearanceSignature,
                                ),
                            appearanceFingerprint =
                                blendAppearanceVector(
                                    previous.appearanceFingerprint,
                                    observation.appearanceFingerprint,
                                ),
                            appearanceQuality =
                                (
                                    previous.appearanceQuality *
                                        (1.0 - appearanceEmaAlpha) +
                                        observation.appearanceQuality
                                            .coerceIn(0.0, 1.0) *
                                        appearanceEmaAlpha
                                    ).coerceIn(0.0, 1.0),
                        )
                    }
            }
        }

        val newestTimestampNanos =
            observations.maxOfOrNull {
                it.timestampNanos
            } ?: Long.MIN_VALUE

        pruneReferences(newestTimestampNanos)
        enforceReferenceCapacity()
        enforceSourceBindingCapacity()
    }

    @Synchronized
    fun associate(
        observations: List<ActorObservation>,
        maximumAgeNanos: Long = 250_000_000L,
        acceptanceThreshold: Double = 0.56,
    ): Result {
        require(maximumAgeNanos > 0L)
        require(acceptanceThreshold in 0.0..1.0)

        val assignments = ArrayList<IdentityAssignment>(
            observations.size,
        )

        val candidateScores = HashMap<Pair<Int, Long>, PairScore>()
        val candidates = observations.flatMapIndexed { observationIndex, observation ->
            references.values
                .asSequence()
                .filter { it.sourceCameraId != observation.cameraId }
                .filter {
                    abs(
                        observation.timestampNanos -
                            it.timestampNanos,
                    ) <= maximumAgeNanos
                }
                .map { reference ->
                    val pairScore = score(
                        observation,
                        reference,
                    )
                    candidateScores[
                        observationIndex to reference.globalActorId
                    ] = pairScore
                    val stickyBinding =
                        sourceToGlobal[
                            observation.cameraId to
                                observation.localActorId
                        ] == reference.globalActorId
                    val hysteresisAdjustedScore =
                        (
                            pairScore.totalScore +
                                if (stickyBinding) {
                                    identityHysteresisBonus
                                } else {
                                    0.0
                                }
                        ).coerceIn(0.0, 1.0)
                    val assignmentScore =
                        if (
                            pairScore.epipolarEvaluatedJointCount >= 2 &&
                            pairScore.epipolarConsistencyScore < 0.30
                        ) {
                            0.0
                        } else {
                            hysteresisAdjustedScore
                        }
                    MaximumWeightBipartiteAssignmentEngine.Candidate(
                        leftIndex = observationIndex,
                        rightId = reference.globalActorId,
                        score = assignmentScore,
                    )
                }
                .toList()
        }

        val assignment = globalAssignmentEngine.assign(
            leftCount = observations.size,
            rightIds = references.values
                .map { it.globalActorId }
                .distinct()
                .sorted(),
            candidates = candidates,
            minimumAcceptedScore = acceptanceThreshold,
        )

        val assignedObservationIndices = HashSet<Int>()

        assignment.matches.forEach { match ->
            val observation = observations.getOrNull(match.leftIndex)
                ?: return@forEach
            val reference = references[match.rightId]
                ?: return@forEach
            val pairScore = candidateScores[
                match.leftIndex to match.rightId
            ] ?: return@forEach

            val competingScores =
                references.values
                    .asSequence()
                    .filter {
                        it.sourceCameraId != observation.cameraId
                    }
                    .filter {
                        abs(
                            observation.timestampNanos -
                                it.timestampNanos,
                        ) <= maximumAgeNanos
                    }
                    .mapNotNull {
                        candidateScores[
                            match.leftIndex to
                                it.globalActorId
                        ]?.totalScore
                    }
                    .sortedDescending()
                    .take(2)
                    .toList()

            val bestCompetingScore =
                competingScores.getOrNull(0)
                    ?: pairScore.totalScore
            val secondCompetingScore =
                competingScores.getOrNull(1)
                    ?: 0.0
            val ambiguityMargin =
                bestCompetingScore -
                    secondCompetingScore
            val stickyBinding =
                sourceToGlobal[
                    observation.cameraId to
                        observation.localActorId
                ] == reference.globalActorId
            val effectiveAmbiguityMargin =
                if (stickyBinding) {
                    identityAmbiguityMargin * 1.50
                } else {
                    identityAmbiguityMargin
                }
            val ambiguous =
                bestCompetingScore >= acceptanceThreshold &&
                    ambiguityMargin <
                        effectiveAmbiguityMargin

            assignments += IdentityAssignment(
                sourceCameraId = observation.cameraId,
                localActorId = observation.localActorId,
                globalActorId = reference.globalActorId,
                score = pairScore.totalScore,
                triangulatedJointCount = pairScore.triangulatedJointCount,
                reprojectionScore = pairScore.reprojectionScore,
                shapeScore = pairScore.shapeScore,
                temporalScore = pairScore.temporalScore,
                appearanceScore = pairScore.appearanceScore,
                epipolarConsistencyScore =
                    pairScore.epipolarConsistencyScore,
                epipolarEvaluatedJointCount =
                    pairScore.epipolarEvaluatedJointCount,
                accepted = !ambiguous,
                warnings =
                    pairScore.warnings +
                        if (ambiguous) {
                            listOf(
                                "CROSS_CAMERA_IDENTITY_AMBIGUOUS_MARGIN",
                            )
                        } else {
                            emptyList()
                        },
            )

            if (!ambiguous) {
                assignedObservationIndices += match.leftIndex
                sourceToGlobal[
                    observation.cameraId to observation.localActorId
                ] = reference.globalActorId
            }
        }

        observations
            .filterIndexed { index, _ ->
                index !in assignedObservationIndices
            }
            .forEach { observation ->
                val existing =
                    sourceToGlobal[
                        observation.cameraId to
                            observation.localActorId
                    ]

                val global =
                    existing
                        ?: allocateGlobalId(
                            observation,
                        )

                assignments +=
                    IdentityAssignment(
                        sourceCameraId =
                            observation.cameraId,
                        localActorId =
                            observation.localActorId,
                        globalActorId =
                            global,
                        score = 0.0,
                        triangulatedJointCount = 0,
                        reprojectionScore = 0.0,
                        shapeScore = 0.0,
                        temporalScore = 0.0,
                        epipolarConsistencyScore = 0.0,
                        accepted = false,
                        warnings =
                            listOf(
                                "CROSS_CAMERA_IDENTITY_NOT_QUALIFIED",
                            ),
                    )
            }

        return Result(
            assignments =
                assignments.sortedBy {
                    it.localActorId
                },
            unmatchedCount =
                assignments.count {
                    !it.accepted
                },
            warnings =
                assignments
                    .flatMap { it.warnings }
                    .distinct()
                    .takeLast(32),
        )
    }

    @Synchronized
    fun globalId(
        cameraId: DeviceId,
        localActorId: Long,
    ): Long? =
        sourceToGlobal[
            cameraId to localActorId
        ]

    @Synchronized
    fun reset() {
        references.clear()
        sourceToGlobal.clear()
        nextGlobalId = 1_000_000L
    }

    private data class PairScore(
        val totalScore: Double,
        val triangulatedJointCount: Int,
        val reprojectionScore: Double,
        val shapeScore: Double,
        val temporalScore: Double,
        val worldShapeScore: Double = 0.0,
        val appearanceScore: Double = 0.0,
        val epipolarConsistencyScore: Double = 0.0,
        val epipolarEvaluatedJointCount: Int = 0,
        val warnings: List<String>,
    )

    private fun blendAppearanceVector(
        previous: List<Double>,
        current: List<Double>,
    ): List<Double> {
        if (current.isEmpty()) return previous
        if (previous.size != current.size) return current.toList()

        return List(current.size) { index ->
            (
                previous[index] *
                    (1.0 - appearanceEmaAlpha) +
                    current[index] *
                    appearanceEmaAlpha
            ).coerceIn(-1.0, 1.0)
        }
    }

    private fun score(
        observation: ActorObservation,
        reference: ReferenceTrack,
    ): PairScore {
        val calibration =
            calibrationProvider()

        if (calibration == null) {
            return PairScore(
                totalScore = shapeFallback(
                    observation,
                    reference,
                ),
                triangulatedJointCount = 0,
                reprojectionScore = 0.0,
                shapeScore =
                    shapeSimilarity(
                        observation.joints,
                        reference.joints,
                    ),
                temporalScore = temporalSimilarity(
                    observation,
                    reference,
                ),
                worldShapeScore =
                    worldShapeSimilarity(
                        observation.worldJoints,
                        reference.worldJoints,
                    ),
                appearanceScore =
                    appearanceSimilarity(
                        observation,
                        reference,
                    ),
                epipolarConsistencyScore = 0.0,
                warnings =
                    listOf(
                        "CROSS_CAMERA_CALIBRATION_UNAVAILABLE",
                    ),
            )
        }

        val firstIntrinsics =
            calibration.intrinsics[reference.sourceCameraId]
        val firstPose =
            calibration.extrinsics[
                reference.sourceCameraId
            ]?.pose
        val secondIntrinsics =
            calibration.intrinsics[observation.cameraId]
        val secondPose =
            calibration.extrinsics[
                observation.cameraId
            ]?.pose

        if (
            firstIntrinsics == null ||
            firstPose == null ||
            secondIntrinsics == null ||
            secondPose == null
        ) {
            return PairScore(
                totalScore =
                    shapeFallback(
                        observation,
                        reference,
                    ),
                triangulatedJointCount = 0,
                reprojectionScore = 0.0,
                shapeScore =
                    shapeSimilarity(
                        observation.joints,
                        reference.joints,
                    ),
                temporalScore =
                    temporalSimilarity(
                        observation,
                        reference,
                    ),
                worldShapeScore =
                    worldShapeSimilarity(
                        observation.worldJoints,
                        reference.worldJoints,
                    ),
                epipolarConsistencyScore = 0.0,
                warnings =
                    listOf(
                        "CROSS_CAMERA_EXTRINSICS_UNAVAILABLE",
                    ),
            )
        }

        val candidateJoints =
            stableJointIds()
        val left =
            observation.joints
                .associateBy {
                    it.landmarkId
                }
        val right =
            reference.joints
                .associateBy {
                    it.landmarkId
                }

        var triangulated = 0
        var reprojectionQualitySum = 0.0
        var epipolarScoreSum = 0.0
        var epipolarEvaluated = 0

        candidateJoints.forEach { jointId ->
            val local = left[jointId]
            val remote = right[jointId]
            if (local == null || remote == null) {
                return@forEach
            }

            val a =
                observationToTriangulation(
                    observation = observation,
                    joint = local,
                    intrinsics = secondIntrinsics,
                    pose = secondPose,
                    calibrationUncertainty =
                        calibration.uncertaintyProfiles[observation.cameraId]
                            ?: com.nivukx.nox.core.model.CalibrationUncertaintyProfile(),
                )
            val b =
                referenceToTriangulation(
                    reference = reference,
                    joint = remote,
                    intrinsics = firstIntrinsics,
                    pose = firstPose,
                    calibrationUncertainty =
                        calibration.uncertaintyProfiles[reference.sourceCameraId]
                            ?: com.nivukx.nox.core.model.CalibrationUncertaintyProfile(),
                )

            val result =
                runCatching {
                    triangulator.robust(
                        listOf(a, b),
                    )
                }.getOrNull()

            if (result != null) {
                triangulated += 1
                val epipolar =
                    runCatching {
                        symmetricEpipolarConsistencyEngine.evaluate(
                            left =
                                SymmetricEpipolarConsistencyEngine.Observation(
                                    cameraPose = secondPose,
                                    intrinsics = secondIntrinsics,
                                    u = local.x,
                                    v = local.y,
                                    confidence = local.confidence,
                                ),
                            right =
                                SymmetricEpipolarConsistencyEngine.Observation(
                                    cameraPose = firstPose,
                                    intrinsics = firstIntrinsics,
                                    u = remote.x,
                                    v = remote.y,
                                    confidence = remote.confidence,
                                ),
                        )
                    }.getOrNull()
                if (epipolar != null) {
                    epipolarScoreSum += epipolar.score
                    epipolarEvaluated += 1
                }
                val residual =
                    result.rmsReprojectionPx
                        .coerceAtLeast(0.0)
                reprojectionQualitySum +=
                    expNegative(
                        residual / 2.5,
                    ) *
                        result.cheiralityRatio
            }
        }

        val reprojectionScore =
            if (triangulated == 0) {
                0.0
            } else {
                (
                    reprojectionQualitySum /
                        triangulated.toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        val epipolarConsistencyScore =
            if (epipolarEvaluated == 0) {
                0.0
            } else {
                (
                    epipolarScoreSum /
                        epipolarEvaluated.toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        val shapeScore =
            shapeSimilarity(
                observation.joints,
                reference.joints,
            )
        val temporalScore =
            temporalSimilarity(
                observation,
                reference,
            )
        val worldShapeScore =
            worldShapeSimilarity(
                observation.worldJoints,
                reference.worldJoints,
            )
        val appearanceScore =
            appearanceSimilarity(
                observation,
                reference,
            )
        val parallaxBonus =
            triangulated
                .toDouble()
                .div(
                    candidateJoints.size.toDouble(),
                )
                .coerceIn(0.0, 1.0)

        val total =
            (
                reprojectionScore * 0.31 +
                    epipolarConsistencyScore * 0.10 +
                    shapeScore * 0.17 +
                    temporalScore * 0.12 +
                    worldShapeScore * 0.10 +
                    appearanceScore * 0.10 +
                    parallaxBonus * 0.10
                ).coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (
                    appearanceScore > 0.0 &&
                    illuminationRobustSignatureSimilarity(
                        observation.appearanceSignature,
                        reference.appearanceSignature,
                    ) >= 0.70
                ) {
                    add(
                        "CROSS_CAMERA_ILLUMINATION_VARIANCE_COMPENSATED",
                    )
                }
                if (triangulated < 3) {
                    add("CROSS_CAMERA_STABLE_JOINT_SUPPORT_LOW")
                }
                if (reprojectionScore < 0.45) {
                    add("CROSS_CAMERA_REPROJECTION_SUPPORT_LOW")
                }
                if (epipolarEvaluated > 0 &&
                    epipolarConsistencyScore < 0.45
                ) {
                    add("CROSS_CAMERA_EPIPOLAR_SUPPORT_LOW")
                }
                if (
                    epipolarEvaluated >= 2 &&
                    epipolarConsistencyScore < 0.30
                ) {
                    add("CROSS_CAMERA_EPIPOLAR_ASSIGNMENT_BLOCKED")
                }
                if (shapeScore < 0.45) {
                    add("CROSS_CAMERA_SHAPE_SIMILARITY_LOW")
                }
                if (
                    observation.worldJoints.isNotEmpty() &&
                    reference.worldJoints.isNotEmpty() &&
                    worldShapeScore < 0.45
                ) {
                    add("CROSS_CAMERA_WORLD_SHAPE_SIMILARITY_LOW")
                }
            }

        return PairScore(
            totalScore = total,
            triangulatedJointCount = triangulated,
            reprojectionScore = reprojectionScore,
            shapeScore = shapeScore,
            temporalScore = temporalScore,
            worldShapeScore = worldShapeScore,
            appearanceScore = appearanceScore,
            epipolarConsistencyScore = epipolarConsistencyScore,
            epipolarEvaluatedJointCount = epipolarEvaluated,
            warnings = warnings,
        )
    }

    private fun shapeFallback(
        observation: ActorObservation,
        reference: ReferenceTrack,
    ): Double =
        (
            shapeSimilarity(
                observation.joints,
                reference.joints,
            ) * 0.47 +
                worldShapeSimilarity(
                    observation.worldJoints,
                    reference.worldJoints,
                ) * 0.20 +
                temporalSimilarity(
                    observation,
                    reference,
                ) * 0.28 +
                appearanceSimilarity(
                    observation,
                    reference,
                ) * 0.05
            ).coerceIn(0.0, 1.0)

    private fun appearanceSimilarity(
        observation: ActorObservation,
        reference: ReferenceTrack,
    ): Double {
        val quality =
            minOf(
                observation.appearanceQuality,
                reference.appearanceQuality,
            ).coerceIn(0.0, 1.0)

        if (
            quality < 0.20 ||
            observation.appearanceFingerprint.isEmpty() ||
            reference.appearanceFingerprint.isEmpty()
        ) {
            return 0.0
        }

        val size =
            minOf(
                observation.appearanceFingerprint.size,
                reference.appearanceFingerprint.size,
            )

        if (size == 0) return 0.0

        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        var finite = 0

        for (index in 0 until size) {
            val a = observation.appearanceFingerprint[index]
            val b = reference.appearanceFingerprint[index]
            if (!a.isFinite() || !b.isFinite()) continue
            dot += a * b
            normA += a * a
            normB += b * b
            finite += 1
        }

        if (
            finite < 16 ||
            normA <= 1e-12 ||
            normB <= 1e-12
        ) {
            return 0.0
        }

        val fingerprintCosine =
            (
                dot /
                    kotlin.math.sqrt(
                        normA * normB,
                    ).coerceAtLeast(1e-12)
            )
                .coerceIn(-1.0, 1.0)

        val fingerprintScore =
            (
                fingerprintCosine * 0.5 +
                    0.5
                )
                .coerceIn(0.0, 1.0)

        val signatureScore =
            illuminationRobustSignatureSimilarity(
                observation.appearanceSignature,
                reference.appearanceSignature,
            )

        val combined =
            if (signatureScore >= 0.0) {
                fingerprintScore * 0.55 +
                    signatureScore * 0.45
            } else {
                fingerprintScore
            }

        return (
            combined *
                quality
            )
            .coerceIn(0.0, 1.0)

    }

    /**
     * Compares the luma component of the exposure-normalized descriptor.
     *
     * YuvTrackAppearanceDescriptorEngine standardizes luma by the local mean/std-dev, so this
     * channel is intentionally less sensitive to global illumination gain/offset than the network
     * histogram fingerprint. Chroma and histogram data remain in the score for discriminability.
     */
    private fun illuminationRobustSignatureSimilarity(
        observation: List<Double>,
        reference: List<Double>,
    ): Double {
        if (
            observation.size < 9 ||
            reference.size < 9
        ) {
            return -1.0
        }

        val sampleCount =
            minOf(
                observation.size,
                reference.size,
            ) / 3

        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        var finite = 0

        for (sample in 0 until sampleCount) {
            val a =
                observation[
                    sample * 3
                ]
            val b =
                reference[
                    sample * 3
                ]

            if (!a.isFinite() || !b.isFinite()) continue

            dot += a * b
            normA += a * a
            normB += b * b
            finite++
        }

        if (
            finite < 8 ||
            normA <= 1e-12 ||
            normB <= 1e-12
        ) {
            return -1.0
        }

        val cosine =
            (
                dot /
                    kotlin.math.sqrt(
                        normA *
                            normB,
                    ).coerceAtLeast(1e-12)
                )
                .coerceIn(
                    -1.0,
                    1.0,
                )

        return (
            cosine * 0.5 +
                0.5
            ).coerceIn(
            0.0,
            1.0,
        )
    }

    private fun shapeSimilarity(
        a: List<MultiActorPoseAssociationEngine.Joint>,
        b: List<MultiActorPoseAssociationEngine.Joint>,
    ): Double {
        val first = a.associateBy { it.landmarkId }
        val second = b.associateBy { it.landmarkId }
        val ids =
            stableJointIds().filter {
                first[it] != null && second[it] != null
            }

        if (ids.size < 3) return 0.0

        val aCenter =
            center(first, ids)
        val bCenter =
            center(second, ids)
        val aScale =
            scale(first, ids, aCenter)
        val bScale =
            scale(second, ids, bCenter)

        var error = 0.0
        ids.forEach { id ->
            val pa =
                first.getValue(id)
            val pb =
                second.getValue(id)
            val ax =
                (pa.x - aCenter.first) /
                    aScale
            val ay =
                (pa.y - aCenter.second) /
                    aScale
            val bx =
                (pb.x - bCenter.first) /
                    bScale
            val by =
                (pb.y - bCenter.second) /
                    bScale
            error +=
                sqrt(
                    (ax - bx) * (ax - bx) +
                        (ay - by) * (ay - by),
                )
        }

        val mean =
            error /
                ids.size.toDouble()

        return expNegative(
            mean * 2.4,
        )
    }

    /**
     * Compares body-space bone-length signatures. Translation and global camera orientation cancel
     * out, making this useful for cross-camera identity continuity. The signal is intentionally a
     * secondary cue: learned world landmarks are not treated as calibrated rig geometry.
     */
    private fun worldShapeSimilarity(
        a: List<MultiActorPoseAssociationEngine.WorldJoint>,
        b: List<MultiActorPoseAssociationEngine.WorldJoint>,
    ): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0

        val first = a.associateBy { it.landmarkId }
        val second = b.associateBy { it.landmarkId }
        val pairs =
            listOf(
                0 to 11,
                0 to 12,
                11 to 12,
                11 to 23,
                12 to 24,
                23 to 24,
                11 to 24,
                12 to 23,
            )

        val errors = pairs.mapNotNull { (i, j) ->
            val p = first[i]
            val q = first[j]
            val r = second[i]
            val t = second[j]
            if (p == null || q == null || r == null || t == null) {
                null
            } else {
                val p3 = p.positionMeters
                val q3 = q.positionMeters
                val r3 = r.positionMeters
                val t3 = t.positionMeters
                val firstLength = p3.distanceTo(q3)
                val secondLength = r3.distanceTo(t3)
                if (
                    !firstLength.isFinite() ||
                    !secondLength.isFinite() ||
                    firstLength <= 1e-5 ||
                    secondLength <= 1e-5
                ) {
                    null
                } else {
                    abs(firstLength - secondLength) /
                        max(
                            max(firstLength, secondLength),
                            0.05,
                        )
                }
            }
        }

        if (errors.size < 3) return 0.0

        val sorted = errors.sorted()
        val trimmed =
            if (sorted.size >= 5) {
                sorted.drop(1).dropLast(1)
            } else {
                sorted
            }
        return expNegative(
            trimmed.average() * 5.0,
        )
    }

    private fun temporalSimilarity(
        observation: ActorObservation,
        reference: ReferenceTrack,
    ): Double {
        val deltaMillis =
            abs(
                observation.timestampNanos -
                    reference.timestampNanos,
            ).toDouble() / 1_000_000.0

        val confidence =
            min(
                observation.confidence,
                reference.confidence,
            )

        return (
            expNegative(deltaMillis / 80.0) *
                confidence
            ).coerceIn(0.0, 1.0)
    }

    private fun center(
        joints: Map<Int, MultiActorPoseAssociationEngine.Joint>,
        ids: List<Int>,
    ): Pair<Double, Double> {
        val inv =
            1.0 / ids.size.toDouble()
        return (
            ids.sumOf {
                joints.getValue(it).x
            } * inv
        ) to (
            ids.sumOf {
                joints.getValue(it).y
            } * inv
        )
    }

    private fun scale(
        joints: Map<Int, MultiActorPoseAssociationEngine.Joint>,
        ids: List<Int>,
        center: Pair<Double, Double>,
    ): Double =
        max(
            1e-4,
            sqrt(
                ids.map { id ->
                    val j =
                        joints.getValue(id)
                    val dx =
                        j.x - center.first
                    val dy =
                        j.y - center.second
                    dx * dx + dy * dy
                }.average(),
            ),
        )

    private fun observationToTriangulation(
        observation: ActorObservation,
        joint: MultiActorPoseAssociationEngine.Joint,
        intrinsics: CameraIntrinsics,
        pose: Pose,
        calibrationUncertainty:
            com.nivukx.nox.core.model.CalibrationUncertaintyProfile =
            com.nivukx.nox.core.model.CalibrationUncertaintyProfile(),
    ): TriangulationEngine.Observation =
        TriangulationEngine.Observation(
            cameraPose = pose,
            intrinsics = intrinsics,
            cameraId = observation.cameraId.value,
            u = joint.x *
                observation.inputWidth,
            v = joint.y *
                observation.inputHeight,
            weight =
                joint.confidence.toDouble(),
            sourceQuality =
                observation.confidence,
            calibrationUncertainty = calibrationUncertainty,
        )

    private fun referenceToTriangulation(
        reference: ReferenceTrack,
        joint: MultiActorPoseAssociationEngine.Joint,
        intrinsics: CameraIntrinsics,
        pose: Pose,
        calibrationUncertainty:
            com.nivukx.nox.core.model.CalibrationUncertaintyProfile =
            com.nivukx.nox.core.model.CalibrationUncertaintyProfile(),
    ): TriangulationEngine.Observation =
        TriangulationEngine.Observation(
            cameraPose = pose,
            intrinsics = intrinsics,
            cameraId = reference.sourceCameraId.value,
            u = joint.x *
                reference.inputWidth,
            v = joint.y *
                reference.inputHeight,
            weight =
                joint.confidence.toDouble(),
            sourceQuality =
                reference.confidence,
            calibrationUncertainty = calibrationUncertainty,
        )

    private fun allocateGlobalId(
        observation: ActorObservation,
    ): Long {
        val composite =
            (
                (
                    observation.cameraId.value.hashCode()
                        .toLong()
                        .and(0xFFFF_FFFFL)
                ) shl 32
            ) xor
                (
                    observation.localActorId
                        .and(0xFFFF_FFFFL)
                )

        val candidate =
            if (composite > 0L) {
                composite
            } else {
                nextGlobalId++
            }

        sourceToGlobal[
            observation.cameraId to
                observation.localActorId
        ] = candidate

        return candidate
    }

    private fun pruneReferences(
        newestTimestampNanos: Long,
    ) {
        if (newestTimestampNanos == Long.MIN_VALUE) {
            return
        }

        val cutoff =
            newestTimestampNanos -
                2_000_000_000L

        val removedGlobals = references
            .filterValues {
                it.timestampNanos < cutoff
            }
            .keys
        removedGlobals.forEach { references.remove(it) }
        if (removedGlobals.isNotEmpty()) {
            sourceToGlobal.entries.removeIf {
                it.value in removedGlobals
            }
        }
    }

    private fun enforceReferenceCapacity() {
        while (references.size > maximumReferenceTracks) {
            val victim =
                references.minByOrNull { (_, value) ->
                    value.timestampNanos
                }?.key ?: return
            references.remove(victim)
            sourceToGlobal.entries.removeIf { it.value == victim }
        }
    }

    private fun enforceSourceBindingCapacity() {
        if (sourceToGlobal.size <= maximumSourceBindings) return

        val activeGlobals = references.keys.toSet()
        sourceToGlobal.entries.removeIf {
            it.value !in activeGlobals
        }

        while (sourceToGlobal.size > maximumSourceBindings) {
            val candidate = sourceToGlobal.entries
                .sortedBy { (_, globalId) ->
                    references[globalId]?.timestampNanos
                        ?: Long.MIN_VALUE
                }
                .firstOrNull()
                ?.key ?: return

            sourceToGlobal.remove(candidate)
        }
    }

    private fun stableJointIds(): List<Int> =
        listOf(
            0,
            11,
            12,
            23,
            24,
        )

    private fun expNegative(value: Double): Double =
        kotlin.math.exp(
            -value.coerceAtLeast(0.0),
        )
}
