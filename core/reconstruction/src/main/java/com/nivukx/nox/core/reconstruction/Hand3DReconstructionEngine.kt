package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Hand3DFrame
import com.nivukx.nox.core.model.Hand3DLandmark
import com.nivukx.nox.core.model.Hand3DResult
import com.nivukx.nox.core.model.HandObservation
import com.nivukx.nox.core.model.HandSide
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Metric multi-view hand reconstruction.
 *
 * Cross-device tracker IDs are local to each phone and therefore are never used as the primary
 * correspondence key. The solver first associates hands by handedness plus calibrated wrist-ray
 * geometry, then triangulates each of the 21 landmarks with the shared robust body triangulator.
 *
 * The result is temporally stabilized and constrained by the hand's kinematic graph:
 * wrist -> palm -> finger chains plus the MCP palm fan.
 */
class Hand3DReconstructionEngine(
    private val triangulatorFactory: () -> TriangulationEngine = {
        TriangulationEngine(
            minParallaxDegrees = 0.55,
            huberDeltaPx = 2.5,
        )
    },
    private val maxTemporalSkewNanos: Long = 24_000_000L,
    private val maximumUncertaintyMeters: Double = 0.25,
    private val maximumWristReprojectionErrorPx: Double = 8.0,
    private val temporalAssociationWeight: Double = 1.35,
    private val maximumTemporalAssociationDistanceMeters: Double = 0.25,
    private val maximumTemporalStates: Int = 4096,
    private val maximumTemporalStateAgeNanos: Long = 2_000_000_000L,
    private val maximumTemporalVelocityMetersPerSecond: Double = 18.0,
    private val maximumTemporalAccelerationMetersPerSecond2: Double = 120.0,
    private val maximumTemporalJerkMetersPerSecond3: Double = 900.0,
    private val jointUncertaintyGate:
        HandJointUncertaintyGateEngine =
        HandJointUncertaintyGateEngine(),
) {

    data class CameraFrame(
        val deviceId: DeviceId,
        val timestampNanos: Long,
        val imageWidth: Int,
        val imageHeight: Int,
        val hands: List<HandObservation>,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
        val rollingShutterFraction: Double = 0.0,
        val frameDurationNanos: Long = 0L,
        val exposureTimeNanos: Long = 0L,
        val captureTimingConfidence: Double = 1.0,
        val calibrationUncertainty: CalibrationUncertaintyProfile =
            CalibrationUncertaintyProfile(),
    )

    private data class HandMember(
        val camera: CameraFrame,
        val hand: HandObservation,
    )

    private data class HandGroup(
        val groupId: Long,
        val side: HandSide,
        val anchor: HandMember,
        val members: MutableList<HandMember> = mutableListOf(),
    ) {
        init {
            members += anchor
        }
    }

    private data class TemporalState(
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var jerk: Vec3,
        var timestampNanos: Long,
        var confidence: Double,
    )

    private val temporal =
        HashMap<Long, TemporalState>(512)

    init {
        require(maximumTemporalStates >= 128)
        require(maximumTemporalStateAgeNanos > 0L)
    }

    private val assignmentSolver =
        BoundedAssignmentSolver(maxItems = 4)

    fun reconstruct(
        timestampNanos: Long,
        frames: List<CameraFrame>,
    ): Hand3DFrame? {
        evictExpiredTemporalState(timestampNanos)
        val usable = frames
            .filter {
                abs(it.timestampNanos - timestampNanos) <=
                    maxTemporalSkewNanos
            }
            .sortedBy { it.deviceId.value }

        if (usable.size < 2) return null

        val groups = associateHands(usable)

        val reconstructed = groups.mapNotNull { group ->
            if (group.members.size < 2) return@mapNotNull null

            buildHand(
                group = group,
                timestampNanos = timestampNanos,
            )
        }

        if (reconstructed.isEmpty()) return null

        enforceTemporalCapacity()

        return Hand3DFrame(
            timestampNanos = timestampNanos,
            hands = reconstructed,
            confidence = reconstructed
                .map { it.confidence }
                .average()
                .coerceIn(0.0, 1.0),
        )
    }

    @Synchronized
    fun reset() {
        temporal.clear()
    }

    @Synchronized
    fun configureJointUncertainty(
        config: HandJointUncertaintyGateEngine.Config,
    ) {
        jointUncertaintyGate.configure(config)
    }

    fun jointUncertaintyConfiguration():
        HandJointUncertaintyGateEngine.Config =
        jointUncertaintyGate.configuration()

    @Synchronized
    fun configureFromMotionCaptureCustomization(
        customization: MotionCaptureCustomization,
    ) {
        jointUncertaintyGate.configure(
            HandJointUncertaintyGateEngine.Config(
                minimumInputConfidence =
                    (
                        0.03 +
                            (1.0 -
                                customization.handGeometryStrictness) *
                                0.08
                        ).coerceIn(0.02, 0.15),
                uncertaintyFloorMeters =
                    (
                        0.0012 +
                            (1.0 -
                                customization.handGeometryStrictness) *
                                0.004
                        ).coerceIn(0.001, 0.01),
                uncertaintyCeilingMeters =
                    customization.maximumReconstructionUncertaintyMeters
                        .coerceIn(0.05, 0.75),
                confidenceExponent =
                    1.05 +
                        customization.handGeometryStrictness * 1.10,
                temporalExponent =
                    0.60 +
                        customization.handTemporalStrictness * 0.95,
                handExponent =
                    0.45 +
                        customization.handGeometryStrictness * 0.45,
                borderPenaltyStrength =
                    0.25 +
                        customization.handGeometryStrictness * 0.60,
                predictedPenaltyStrength =
                    0.25 +
                        customization.occlusionRecoveryConfidence * 0.60,
                unstablePenaltyStrength =
                    0.20 +
                        customization.handTemporalStrictness * 0.55,
                minimumFusionWeight =
                    (
                        0.006 +
                            customization.multiViewEvidenceAdaptivity *
                                0.018
                        ).coerceIn(0.005, 0.03),
                robustOutlierWeightFloor =
                    (
                        0.035 +
                            customization.handGeometryStrictness * 0.055
                        ).coerceIn(0.03, 0.10),
                fingerTipExtraSensitivity =
                    0.06 +
                        customization.handGeometryStrictness * 0.10,
                wristExtraSensitivity =
                    0.02 +
                        customization.handGeometryStrictness * 0.06,
                expectedImageMargin =
                    0.025 +
                        customization.handGeometryStrictness * 0.025,
                hardBorderMargin = 0.006,
                maximumVelocityMetersPerSecond =
                    (
                        8.0 +
                            (1.0 -
                                customization.handTemporalStrictness) *
                                10.0
                        ).coerceIn(8.0, 18.0),
            ),
        )
    }


    private fun associateHands(
        cameras: List<CameraFrame>,
    ): List<HandGroup> {
        val anchorCamera =
            cameras.maxByOrNull { frame ->
                frame.hands.count {
                    it.confidence >= 0.20
                }
            } ?: return emptyList()

        val anchors = anchorCamera.hands
            .filter { it.confidence >= 0.20 }
            .sortedByDescending {
                it.confidence
            }

        val groups = anchors.mapIndexed { index, hand ->
            HandGroup(
                groupId =
                    stableGroupId(
                        anchorDevice = anchorCamera.deviceId,
                        localHandId = hand.handId,
                        ordinal = index,
                    ),
                side = hand.side,
                anchor = HandMember(
                    camera = anchorCamera,
                    hand = hand,
                ),
            )
        }.toMutableList()

        cameras
            .filter { it.deviceId != anchorCamera.deviceId }
            .forEach { camera ->
                val candidates =
                    camera.hands
                        .filter { it.confidence >= 0.20 }
                        .toList()

                if (candidates.isEmpty()) {
                    return@forEach
                }

                val costs =
                    Array(groups.size) { groupIndex ->
                        DoubleArray(candidates.size) { candidateIndex ->
                            correspondenceScore(
                                groupId = groups[groupIndex].groupId,
                                anchor =
                                    groups[groupIndex].anchor,
                                candidateCamera = camera,
                                candidate =
                                    candidates[candidateIndex],
                            ) ?: Double.POSITIVE_INFINITY
                        }
                    }

                val assignment =
                    assignmentSolver.solve(
                        costs = costs,
                        unmatchedItemPenalty =
                            maximumWristReprojectionErrorPx * 1.35,
                        maxAcceptedCost =
                            maximumWristReprojectionErrorPx,
                    )

                assignment.matches.forEach { match ->
                    val group =
                        groups[match.itemIndex]
                    val candidate =
                        candidates[match.candidateIndex]

                    group.members +=
                        HandMember(
                            camera = camera,
                            hand = candidate,
                        )
                }
            }

        return groups.filter {
            it.members.size >= 2
        }
    }

    private fun correspondenceScore(
        groupId: Long,
        anchor: HandMember,
        candidateCamera: CameraFrame,
        candidate: HandObservation,
    ): Double? {
        if (
            anchor.hand.side != HandSide.UNKNOWN &&
            candidate.side != HandSide.UNKNOWN &&
            anchor.hand.side != candidate.side
        ) {
            return null
        }

        val anchorWrist =
            anchor.hand.landmarks.firstOrNull {
                it.index == 0
            } ?: return null

        val candidateWrist =
            candidate.landmarks.firstOrNull {
                it.index == 0
            } ?: return null

        val pair = listOf(
            triangulationObservation(
                camera = anchor.camera,
                landmark = anchorWrist,
                hand = anchor.hand,
            ),
            triangulationObservation(
                camera = candidateCamera,
                landmark = candidateWrist,
                hand = candidate,
            ),
        )

        val result =
            triangulatorFactory().robust(pair)
                ?: return null

        val uncertainty =
            result.uncertaintyMeters
                .coerceAtMost(maximumUncertaintyMeters * 4.0)

        if (
            result.rmsReprojectionPx >
                maximumWristReprojectionErrorPx
        ) {
            return null
        }

        val sidePenalty =
            if (
                anchor.hand.side == HandSide.UNKNOWN ||
                candidate.side == HandSide.UNKNOWN
            ) {
                0.5
            } else {
                0.0
            }

        val confidenceReward =
            (
                anchor.hand.confidence *
                    candidate.confidence
                ).coerceIn(0.0, 1.0)

        val temporalState =
            temporal[(groupId shl 8)]

        val candidateWristWorld =
            result.point

        val temporalResidual =
            temporalState?.position?.distanceTo(
                candidateWristWorld,
            )?.coerceAtMost(
                maximumTemporalAssociationDistanceMeters,
            ) ?: 0.0

        val temporalPenalty =
            if (temporalState == null) {
                0.0
            } else {
                temporalResidual *
                    temporalAssociationWeight
            }

        return result.rmsReprojectionPx +
            uncertainty * 6.0 +
            sidePenalty +
            (1.0 - confidenceReward) * 1.5 +
            temporalPenalty
    }

    private fun buildHand(
        group: HandGroup,
        timestampNanos: Long,
    ): Hand3DResult? {
        val grouped =
            HashMap<Int, MutableList<TriangulationEngine.Observation>>()
        val metricEvidenceScores = mutableListOf<Double>()
        val epipolarEvidenceScores = mutableListOf<Double>()
        val reprojectionErrorsPx = mutableListOf<Double>()

        group.members.forEach { member ->
            member.hand.landmarks.forEach { landmark ->
                if (landmark.confidence < 0.10) return@forEach

                grouped
                    .getOrPut(landmark.index) {
                        mutableListOf()
                    }
                    .add(
                        triangulationObservation(
                            camera = member.camera,
                            landmark = landmark,
                            hand = member.hand,
                        ),
                    )
            }
        }

        val landmarks =
            (0 until 21).mapNotNull { index ->
                val observations =
                    grouped[index].orEmpty()

                if (observations.size < 2) {
                    return@mapNotNull null
                }

                val result =
                    triangulatorFactory()
                        .robust(observations)
                        ?: return@mapNotNull null

                if (
                    result.rmsReprojectionPx.isFinite() &&
                        result.rmsReprojectionPx >= 0.0
                ) {
                    reprojectionErrorsPx +=
                        result.rmsReprojectionPx
                }

                if (
                    !result.point.x.isFinite() ||
                    !result.point.y.isFinite() ||
                    !result.point.z.isFinite()
                ) {
                    return@mapNotNull null
                }

                val uncertainty =
                    result.uncertaintyMeters
                        .coerceIn(
                            0.0,
                            maximumUncertaintyMeters,
                        )

                val metricEvidenceAuthority =
                    minOf(
                        result.metricEvidenceAuthority
                            .coerceIn(0.0, 1.0),
                        (
                            result.activeViewInformationGainScore * 0.55 +
                                result.activeViewRedundancyScore * 0.20 +
                                result.symmetricEpipolarConsistencyScore * 0.25
                            ).coerceIn(0.0, 1.0),
                    )
                metricEvidenceScores += metricEvidenceAuthority
                epipolarEvidenceScores +=
                    result.symmetricEpipolarConsistencyScore
                        .coerceIn(0.0, 1.0)

                val geometryConfidence =
                    (
                        (
                            1.0 -
                                uncertainty /
                                maximumUncertaintyMeters
                            ) *
                            result.conditionScore *
                            result.cheiralityRatio *
                            (
                                1.0 -
                                    min(
                                        1.0,
                                        result.rmsReprojectionPx /
                                            8.0,
                                    )
                                ) *
                            (
                                0.78 +
                                    0.22 *
                                        metricEvidenceAuthority
                            )
                        ).coerceIn(0.0, 1.0)

                val temporalKey =
                    (group.groupId shl 8) xor
                        index.toLong()

                val temporalResult =
                    stabilize(
                        key = temporalKey,
                        position = result.point,
                        confidence = geometryConfidence,
                        timestampNanos = timestampNanos,
                    )

                Hand3DLandmark(
                    index = index,
                    position = temporalResult.position,
                    confidence = temporalResult.confidence,
                    uncertaintyMeters =
                        max(
                            uncertainty,
                            temporalResult.uncertainty,
                        ),
                    observationCount =
                        observations.size,
                )
            }

        if (landmarks.size < 12) return null

        val constrained =
            enforceHandKinematics(landmarks)

        val meanUncertainty =
            constrained
                .map {
                    it.uncertaintyMeters
                }
                .average()

        val confidence = (
            constrained
                .map { it.confidence }
                .average() *
                exp(-meanUncertainty / 0.08)
            ).coerceIn(0.0, 1.0)

        val reprojection =
            reprojectionErrorsPx
                .filter { it.isFinite() && it >= 0.0 }
                .averageOrZero()

        val meanMetricEvidenceAuthority =
            metricEvidenceScores.averageOrZero()
        val meanEpipolarConsistency =
            epipolarEvidenceScores.averageOrZero()
        val strictSubMillimeterQualified =
            constrained.size == 21 &&
                meanUncertainty <= 0.0010 &&
                reprojection <= 0.75 &&
                confidence >= 0.90 &&
                meanMetricEvidenceAuthority >= 0.85 &&
                meanEpipolarConsistency >= 0.85 &&
                constrained.all {
                    it.uncertaintyMeters <= 0.0010 &&
                        it.confidence >= 0.85
                }

        return Hand3DResult(
            handId = group.groupId,
            side = group.side,
            timestampNanos = timestampNanos,
            landmarks = constrained,
            confidence = confidence,
            meanUncertaintyMeters = meanUncertainty,
            reprojectionErrorPx = reprojection,
            metricEvidenceAuthority =
                meanMetricEvidenceAuthority,
            symmetricEpipolarConsistencyScore =
                meanEpipolarConsistency,
            strictSubMillimeterQualified =
                strictSubMillimeterQualified,
        )
    }

    private fun triangulationObservation(
        camera: CameraFrame,
        landmark: com.nivukx.nox.core.model.HandLandmark,
        hand: HandObservation,
    ): TriangulationEngine.Observation {
        val gate =
            jointUncertaintyGate.evaluate(
                HandJointUncertaintyGateEngine.Input(
                    jointIndex = landmark.index,
                    normalizedX = landmark.normalizedX,
                    normalizedY = landmark.normalizedY,
                    landmarkConfidence = landmark.confidence,
                    handConfidence = hand.confidence,
                    handednessConfidence =
                        hand.handednessConfidence,
                    temporalStability =
                        hand.temporalStability.coerceIn(
                            0.0,
                            1.0,
                        ),
                    predicted = hand.predicted,
                    wristVelocityMetersPerSecond =
                        hand.wristVelocityMetersPerSecond.norm(),
                    observationFreshness =
                        (
                            1.0 -
                                (
                                    abs(
                                        hand.timestampNanos -
                                            camera.timestampNanos,
                                    ).toDouble() /
                                        maxTemporalSkewNanos
                                            .toDouble()
                                )
                        ).coerceIn(0.0, 1.0),
                ),
            )

        return TriangulationEngine.Observation(
            cameraPose = camera.pose,
            cameraId = camera.deviceId.value,
            intrinsics = camera.intrinsics,
            u = landmark.normalizedX *
                camera.imageWidth.toDouble(),
            v = landmark.normalizedY *
                camera.imageHeight.toDouble(),
            weight = gate.viewExpertWeight
                .coerceIn(0.0, 1.0),
            rollingShutterFraction =
                camera.rollingShutterFraction.coerceIn(
                    0.0,
                    1.0,
                ),
            motionBlurRisk =
                cameraOpticalMotionBlurRisk(camera),
            temporalAlignmentConfidence =
                camera.captureTimingConfidence
                    .coerceIn(0.0, 1.0),
            temporalUncertaintyNanos =
                abs(
                    camera.timestampNanos -
                        hand.timestampNanos,
                ).coerceAtLeast(0L),
            rowTimeOffsetNanos =
                (
                    (
                        landmark.normalizedY.coerceIn(0.0, 1.0) -
                            0.5
                        ) *
                        camera.frameDurationNanos.toDouble() *
                        camera.rollingShutterFraction.coerceIn(
                            0.0,
                            1.0,
                        )
                ).toLong(),
            calibrationUncertainty =
                camera.calibrationUncertainty,
        )
    }

    private fun cameraOpticalMotionBlurRisk(
        camera: CameraFrame,
    ): Double =
        estimateOpticalMotionBlurRisk(
            captureTimingConfidence =
                camera.captureTimingConfidence,
            rollingShutterSkewNanos =
                (
                    camera.rollingShutterFraction
                        .coerceIn(0.0, 1.0) *
                        camera.frameDurationNanos.toDouble()
                ).toLong(),
            frameDurationNanos =
                camera.frameDurationNanos,
            exposureTimeNanos =
                camera.exposureTimeNanos,
        )

    private data class Stabilized(
        val position: Vec3,
        val confidence: Double,
        val uncertainty: Double,
    )

    private fun stabilize(
        key: Long,
        position: Vec3,
        confidence: Double,
        timestampNanos: Long,
    ): Stabilized {
        val old =
            temporal[key]
                ?: return Stabilized(
                    position = position,
                    confidence = confidence,
                    uncertainty = 0.03,
                ).also {
                    temporal[key] =
                        TemporalState(
                            position = position,
                            velocity =
                                Vec3(
                                    0.0,
                                    0.0,
                                    0.0,
                                ),
                            acceleration =
                                Vec3(
                                    0.0,
                                    0.0,
                                    0.0,
                                ),
                            jerk =
                                Vec3(
                                    0.0,
                                    0.0,
                                    0.0,
                                ),
                            timestampNanos = timestampNanos,
                            confidence = confidence,
                        )
                }

        val dt =
            (
                (timestampNanos -
                    old.timestampNanos
                ).coerceAtLeast(1L) / 1e9
                ).coerceIn(
                    1.0 / 240.0,
                    0.20,
                )

        val dtSquared = dt * dt
        val dtCubed = dtSquared * dt

        val prediction =
            old.position +
                old.velocity * dt +
                old.acceleration *
                    (0.5 * dtSquared) +
                old.jerk *
                    (dtCubed / 6.0)

        val innovation =
            position.distanceTo(prediction)

        val gate = (
            0.015 +
                old.velocity.norm() * dt * 0.70 +
                (1.0 - confidence) * 0.035
            ).coerceIn(
                0.012,
                0.14,
            )

        val weight =
            if (innovation > gate) {
                0.08
            } else {
                (
                    0.18 +
                        0.68 * confidence +
                        0.14 *
                            (
                                1.0 -
                                    innovation /
                                    gate
                                )
                    ).coerceIn(
                        0.08,
                        0.92,
                    )
            }

        val corrected =
            prediction.lerp(
                position,
                weight,
            )

        val measuredVelocity =
            (
                corrected -
                    old.position
                ) / dt

        val boundedMeasuredVelocity =
            measuredVelocity.limitNorm(
                maximumTemporalVelocityMetersPerSecond,
            )

        val measuredAcceleration =
            (
                boundedMeasuredVelocity -
                    old.velocity
                ) / dt

        val boundedMeasuredAcceleration =
            measuredAcceleration.limitNorm(
                maximumTemporalAccelerationMetersPerSecond2,
            )

        val measuredJerk =
            (
                boundedMeasuredAcceleration -
                    old.acceleration
                ) / dt

        val boundedMeasuredJerk =
            measuredJerk.limitNorm(
                maximumTemporalJerkMetersPerSecond3,
            )

        old.velocity =
            old.velocity.lerp(
                boundedMeasuredVelocity,
                (
                    0.12 +
                        confidence * 0.55
                    ).coerceIn(
                        0.12,
                        0.67,
                    ),
            )
        old.acceleration =
            old.acceleration.lerp(
                boundedMeasuredAcceleration,
                (
                    0.10 +
                        confidence * 0.28
                    ).coerceIn(
                        0.10,
                        0.38,
                    ),
            )
        old.jerk =
            old.jerk.lerp(
                boundedMeasuredJerk,
                (
                    0.08 +
                        confidence * 0.22
                    ).coerceIn(
                        0.08,
                        0.30,
                    ),
            )

        old.position = corrected
        old.timestampNanos =
            timestampNanos
        old.confidence = (
            old.confidence * 0.28 +
                confidence * 0.72
            ).coerceIn(
                0.0,
                1.0,
            )

        return Stabilized(
            position = corrected,
            confidence = old.confidence *
                if (innovation > gate) {
                    0.86
                } else {
                    1.0
                },
            uncertainty = max(
                0.001,
                0.004 +
                    innovation * 0.15 +
                    old.acceleration.norm() * 0.0008 +
                    old.jerk.norm() * 0.00012,
            ),
        )
    }

    private fun Vec3.limitNorm(
        maximum: Double,
    ): Vec3 {
        val norm = norm()
        if (!norm.isFinite() || norm <= 1e-9) {
            return Vec3(
                0.0,
                0.0,
                0.0,
            )
        }
        val bounded =
            norm.coerceAtMost(
                maximum.coerceAtLeast(0.0),
            )
        if (bounded <= 1e-9) {
            return Vec3(
                0.0,
                0.0,
                0.0,
            )
        }
        return this * (bounded / norm)
    }

    private fun enforceHandKinematics(
        source: List<Hand3DLandmark>,
    ): List<Hand3DLandmark> {
        val map =
            source.associateBy {
                it.index
            }.toMutableMap()

        val bones =
            listOf(
                0 to 1, 1 to 2, 2 to 3, 3 to 4,
                0 to 5, 5 to 6, 6 to 7, 7 to 8,
                0 to 9, 9 to 10, 10 to 11, 11 to 12,
                0 to 13, 13 to 14, 14 to 15, 15 to 16,
                0 to 17, 17 to 18, 18 to 19, 19 to 20,
                5 to 9, 9 to 13, 13 to 17,
            )

        val referenceLengths =
            bones.mapNotNull { (a, b) ->
                val pa = map[a] ?: return@mapNotNull null
                val pb = map[b] ?: return@mapNotNull null

                Triple(
                    a,
                    b,
                    pa.position.distanceTo(
                        pb.position,
                    ),
                )
            }

        repeat(2) {
            referenceLengths.forEach {
                    (a, b, target) ->
                val pa = map[a] ?: return@forEach
                val pb = map[b] ?: return@forEach

                val delta =
                    pb.position -
                        pa.position

                val distance =
                    delta.norm()

                if (distance < 1e-7) {
                    return@forEach
                }

                val relative =
                    abs(distance - target) /
                        target.coerceAtLeast(
                            1e-6,
                        )

                if (relative > 0.20) {
                    return@forEach
                }

                val correction =
                    delta * (
                        (
                            distance -
                                target
                            ) /
                            distance *
                            0.18
                        )

                val wa =
                    pa.confidence.coerceIn(
                        0.05,
                        1.0,
                    )
                val wb =
                    pb.confidence.coerceIn(
                        0.05,
                        1.0,
                    )
                val total =
                    wa + wb

                map[a] =
                    pa.copy(
                        position =
                            pa.position +
                                correction *
                                (wa / total),
                        confidence = (
                            pa.confidence *
                                (
                                    1.0 -
                                        relative *
                                        0.35
                                    )
                            ).coerceIn(
                                0.0,
                                1.0,
                            ),
                    )

                map[b] =
                    pb.copy(
                        position =
                            pb.position -
                                correction *
                                (wb / total),
                        confidence = (
                            pb.confidence *
                                (
                                    1.0 -
                                        relative *
                                        0.35
                                    )
                            ).coerceIn(
                                0.0,
                                1.0,
                            ),
                    )
            }
        }

        return map.values
            .sortedBy { it.index }
    }

    private fun evictExpiredTemporalState(
        timestampNanos: Long,
    ) {
        if (temporal.isEmpty()) return
        val threshold =
            timestampNanos -
                maximumTemporalStateAgeNanos
        val iterator = temporal.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.value.timestampNanos < threshold) {
                iterator.remove()
            }
        }
    }

    private fun enforceTemporalCapacity() {
        if (temporal.size <= maximumTemporalStates) {
            return
        }

        val targetSize =
            (maximumTemporalStates * 0.90)
                .toInt()
                .coerceAtLeast(1)

        val removeCount =
            (temporal.size - targetSize)
                .coerceAtLeast(1)

        temporal.entries
            .asSequence()
            .sortedBy { it.value.timestampNanos }
            .take(removeCount)
            .map { it.key }
            .toList()
            .forEach(temporal::remove)
    }

    private fun stableGroupId(
        anchorDevice: DeviceId,
        localHandId: Long,
        ordinal: Int,
    ): Long {
        val deviceHash =
            abs(anchorDevice.value.hashCode())
                .toLong()
                .coerceAtLeast(1L)

        return (
            (deviceHash shl 32) xor
                (localHandId and 0xFFFF_FFFFL) xor
                ordinal.toLong()
            )
    }
}
