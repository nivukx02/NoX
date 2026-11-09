package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CameraExtrinsics
import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.reconstruction.BundleAdjustmentEngine
import com.nivukx.nox.core.reconstruction.TriangulationEngine
import com.nivukx.nox.core.nativeengine.calibration.NativeCalibrationConfig

/**
 * Self-calibration from synchronized actor landmarks.
 *
 * The rig is initialized from sensor/role geometry, then the actor itself becomes the temporary
 * calibration target. Corresponding 2D body landmarks are triangulated with the seed poses and
 * passed through robust bundle adjustment. This does not require a checkerboard to refine the rig,
 * but it still requires a reasonable initial orientation and sufficient multi-view parallax.
 */
class SelfCalibrationEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val nativeRefinementEnabled: Boolean = true,
        val nativeRefinementMinimumConfidence: Double = 0.72,
        val nativeRefinementMaximumRelativeResidualIncrease: Double = 0.01,
    ) {
        init {
            require(nativeRefinementMinimumConfidence in 0.0..1.0)
            require(
                nativeRefinementMaximumRelativeResidualIncrease in 0.0..0.10
            )
        }
    }

    private val uncertaintyEstimator = CalibrationUncertaintyEstimator()

    data class CameraSeed(
        val id: DeviceId,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
        val calibrationFocusDistanceDiopters: Double? = null,
    )

    data class LandmarkView(
        val cameraId: DeviceId,
        val timestampNanos: Long,
        val landmarks: Map<Int, Vec2>,
    )

    data class Result(
        val snapshot: CalibrationSnapshot,
        val reprojectionErrorPx: Double,
        val landmarksUsed: Int,
        val converged: Boolean,
        val iterations: Int,
        val nativeRefinementApplied: Boolean = false,
        val nativeRefinementConfidence: Double = 0.0,
        val nativeRefinementImprovementPx: Double = 0.0,
    )

    private val triangulator = TriangulationEngine(
        minParallaxDegrees = 0.25,
        huberDeltaPx = 3.0,
        iterations = 6,
    )

    private val optimizer = BundleAdjustmentEngine(
        maxIterations = 15,
        huberDeltaPx = 2.5,
        initialDamping = 1e-3,
    )

    fun refine(
        seeds: List<CameraSeed>,
        views: List<LandmarkView>,
        referenceDevice: DeviceId,
        nowNanos: Long = System.nanoTime(),
    ): Result? {
        if (seeds.size < 2 || views.size < 2) return null
        val seedMap = seeds.associateBy { it.id }
        if (!seedMap.containsKey(referenceDevice)) return null

        val synchronizedGroups =
            views
                .groupBy { it.timestampNanos }
                .mapNotNull { (_, group) ->
                    val cameraCount =
                        group.map { it.cameraId }.distinct().size

                    val sharedLandmarks =
                        group
                            .flatMap { it.landmarks.keys }
                            .groupingBy { it }
                            .eachCount()
                            .count { it.value >= 2 }

                    if (
                        cameraCount >= 2 &&
                        sharedLandmarks >= 6
                    ) {
                        group
                    } else {
                        null
                    }
                }
                .sortedWith(
                    compareByDescending<List<LandmarkView>> {
                        it.map { view ->
                            view.cameraId
                        }.distinct().size
                    }.thenByDescending {
                        it
                            .flatMap { view ->
                                view.landmarks.keys
                            }
                            .distinct()
                            .size
                    },
                )
                .take(8)

        if (synchronizedGroups.size < 2) {
            return null
        }

        val cameras = seeds.map { seed ->
            BundleAdjustmentEngine.Camera(
                id = seed.id.value,
                intrinsics = seed.intrinsics,
                pose = seed.pose,
                fixed = seed.id == referenceDevice,
            )
        }

        val cameraMap = cameras.associateBy { it.id }
        val points = mutableListOf<BundleAdjustmentEngine.Point>()
        val observations = mutableListOf<BundleAdjustmentEngine.Observation>()

        synchronizedGroups.forEachIndexed { groupIndex, group ->
            val landmarkIds =
                group
                    .flatMap { it.landmarks.keys }
                    .distinct()
                    .filter { landmarkId ->
                        group.count { it.landmarks.containsKey(landmarkId) } >= 2
                    }
                    .take(33)

            landmarkIds.forEach { landmarkId ->
                val samples =
                    group.mapNotNull { view ->
                        val pixel =
                            view.landmarks[landmarkId]
                                ?: return@mapNotNull null
                        Triple(
                            view.cameraId,
                            pixel,
                            view.timestampNanos,
                        )
                    }

                val triangulationObservations =
                    samples.mapNotNull { sample ->
                        val cameraSeed =
                            seedMap[sample.first]
                                ?: return@mapNotNull null

                        CameraObservation(
                            deviceId = cameraSeed.id,
                            point = sample.second,
                            confidence = 1.0,
                            intrinsics = cameraSeed.intrinsics,
                            pose = cameraSeed.pose,
                        )
                    }

                if (triangulationObservations.size < 2) {
                    return@forEach
                }

                val point =
                    triangulator.robust(
                        triangulationObservations.map(
                            TriangulationEngine::Observation,
                        ),
                    )?.point
                        ?: return@forEach

                if (
                    !point.x.isFinite() ||
                    !point.y.isFinite() ||
                    !point.z.isFinite()
                ) {
                    return@forEach
                }

                val pointId =
                    groupIndex * 1000 +
                        landmarkId

                points +=
                    BundleAdjustmentEngine.Point(
                        id = pointId,
                        position = point,
                    )

                samples.forEach { sample ->
                    if (
                        cameraMap.containsKey(
                            sample.first.value,
                        )
                    ) {
                        observations +=
                            BundleAdjustmentEngine.Observation(
                                cameraId = sample.first.value,
                                pointId = pointId,
                                pixel = sample.second,
                                confidence = 1.0,
                            )
                    }
                }
            }
        }

        if (points.size < 12 || observations.size < 24) return null

        val optimized = optimizer.optimize(
            inputCameras = cameras,
            inputPoints = points,
            observations = observations,
        )

        val initialIntrinsics =
            optimized.cameras.associate { camera ->
                DeviceId(camera.id) to camera.intrinsics
            }

        val initialExtrinsics =
            optimized.cameras.associate { camera ->
                DeviceId(camera.id) to CameraExtrinsics(
                    pose = camera.pose,
                    referenceDevice = referenceDevice,
                )
            }

        val optimizedPointsById =
            optimized.points.associateBy { it.id }

        val nativeObservations =
            observations.mapNotNull { observation ->
                val point =
                    optimizedPointsById[observation.pointId]
                        ?: return@mapNotNull null
                val cameraId =
                    DeviceId(observation.cameraId)
                if (cameraId !in initialIntrinsics.keys) {
                    return@mapNotNull null
                }

                CalibrationEngine.NativeCalibrationObservation(
                    cameraId = cameraId,
                    worldPoint = point.position,
                    pixel = observation.pixel,
                    weight =
                        observation.confidence
                            .coerceIn(0.0, 1.0),
                )
            }

        val baseSnapshot =
            CalibrationSnapshot(
                intrinsics = initialIntrinsics,
                extrinsics = initialExtrinsics,
                reprojectionErrorPx =
                    optimized.finalRmsPx,
                calibratedAtNanos = nowNanos,
            )

        var intrinsics = initialIntrinsics
        var extrinsics = initialExtrinsics
        var finalReprojectionErrorPx =
            optimized.finalRmsPx
        var nativeRefinementApplied = false
        var nativeRefinementConfidence = 0.0
        var nativeRefinementImprovementPx = 0.0

        if (
            config.nativeRefinementEnabled &&
            nativeObservations.isNotEmpty()
        ) {
            val nativeConfig =
                NativeCalibrationConfig(
                    maxIterations = 16,
                    minimumObservationsPerCamera = 12,
                    refineIntrinsics = true,
                    refineDistortion = true,
                    focalPriorSigmaRatio = 0.10,
                    principalPointPriorSigmaRatio = 0.05,
                    distortionPriorSigma = 0.05,
                    refinementDamping = 1e-3,
                    refinementIterations = 12,
                )

            val nativeEngine =
                com.nivukx.nox.core.nativeengine.calibration
                    .NativeCalibrationEngine()

            if (nativeEngine.isAvailable()) {
                val native =
                    runCatching {
                        CalibrationEngine().nativeRefine(
                            snapshot = baseSnapshot,
                            observations = nativeObservations,
                            reference = referenceDevice,
                            config = nativeConfig,
                        )
                    }.getOrNull()

                if (
                    native != null &&
                    native.valid &&
                    native.confidence >=
                        config.nativeRefinementMinimumConfidence &&
                    native.residualRmsPx.isFinite() &&
                    native.residualRmsPx <=
                        optimized.finalRmsPx *
                            (
                                1.0 +
                                    config.nativeRefinementMaximumRelativeResidualIncrease
                            )
                ) {
                    intrinsics =
                        native.cameras.associate {
                            DeviceId(it.cameraId) to
                                it.intrinsics
                        }
                    extrinsics =
                        native.cameras.associate {
                            DeviceId(it.cameraId) to
                                CameraExtrinsics(
                                    pose = it.pose,
                                    referenceDevice =
                                        referenceDevice,
                                )
                        }
                    finalReprojectionErrorPx =
                        native.residualRmsPx
                    nativeRefinementApplied = true
                    nativeRefinementConfidence =
                        native.confidence
                    nativeRefinementImprovementPx =
                        optimized.finalRmsPx -
                            native.residualRmsPx
                }
            }
        }

        val observationCounts =
            observations
                .groupingBy { it.cameraId }
                .eachCount()
        val maximumObservationCount =
            observationCounts.values.maxOrNull()
                ?.coerceAtLeast(1)
                ?: 1

        val calibrationFocusDistanceDiopters =
            seeds.mapNotNull { seed ->
                seed.calibrationFocusDistanceDiopters
                    ?.takeIf {
                        it.isFinite() && it >= 0.0
                    }
                    ?.let { seed.id to it }
            }.toMap()

        /*
         * Uncertainty metadata must describe the exact calibration snapshot that will be
         * published. Native intrinsic refinement can alter intrinsics and extrinsics after the
         * bundle-adjustment pass, so using optimized.cameras here would publish stale geometry
         * metadata.
         */
        val finalCameraPoses =
            intrinsics.keys.associateWith { deviceId ->
                extrinsics.getValue(deviceId).pose
            }

        val finalReprojectionRmsPx =
            finalReprojectionErrorPx

        val uncertaintyProfiles =
            finalCameraPoses.mapValues { (deviceId, finalPose) ->
                val participation =
                    (
                        observationCounts[deviceId.value]
                            ?: 0
                    ).toDouble() /
                        maximumObservationCount.toDouble()

                val cameraDepths =
                    optimized.points
                        .mapNotNull { point ->
                            val cameraPoint =
                                finalPose.rotation
                                    .conjugate()
                                    .normalized()
                                    .rotate(
                                        point.position -
                                            finalPose.position,
                                    )
                            cameraPoint.z.takeIf {
                                it.isFinite() && it > 0.0
                            }
                        }

                deviceId to uncertaintyEstimator.estimate(
                    CalibrationUncertaintyEstimator.CameraEvidence(
                        cameraId = deviceId.value,
                        intrinsics =
                            intrinsics.getValue(deviceId),
                        pose = finalPose,
                        landmarkCount =
                            observationCounts[deviceId.value]
                                ?: 0,
                        finalReprojectionRmsPx =
                            finalReprojectionRmsPx,
                        medianSceneDepthMeters =
                            medianFinite(
                                cameraDepths,
                            ),
                        maximumBaselineMeters =
                            uncertaintyEstimator.maximumBaseline(
                                referencePose =
                                    finalCameraPoses
                                        .getValue(referenceDevice),
                                poses =
                                    finalCameraPoses
                                        .filterKeys {
                                            it != deviceId
                                        }
                                        .values
                                        .toList(),
                            ),
                        cameraParticipationRatio =
                            participation.coerceIn(0.0, 1.0),
                    ),
                )
            }

        return Result(
            snapshot = CalibrationSnapshot(
                intrinsics = intrinsics,
                extrinsics = extrinsics,
                reprojectionErrorPx =
                    finalReprojectionErrorPx,
                calibratedAtNanos = nowNanos,
                uncertaintyProfiles = uncertaintyProfiles,
                calibrationFocusDistanceDiopters =
                    calibrationFocusDistanceDiopters,
                calibrationMethod =
                    if (nativeRefinementApplied) {
                        "SELF_CALIBRATION_NATIVE_INTRINSIC_REFINEMENT"
                    } else {
                        "SELF_CALIBRATION_BUNDLE_ADJUSTMENT"
                    },
                nativeIntrinsicRefinementApplied =
                    nativeRefinementApplied,
                nativeIntrinsicRefinementConfidence =
                    nativeRefinementConfidence,
                nativeIntrinsicRefinementImprovementPx =
                    nativeRefinementImprovementPx,
            ),
            reprojectionErrorPx =
                finalReprojectionErrorPx,
            landmarksUsed = optimized.points.size,
            converged =
                optimized.converged ||
                    nativeRefinementApplied,
            iterations = optimized.iterations,
            nativeRefinementApplied =
                nativeRefinementApplied,
            nativeRefinementConfidence =
                nativeRefinementConfidence,
            nativeRefinementImprovementPx =
                nativeRefinementImprovementPx,
        )
    }

    private fun medianFinite(
        values: List<Double>,
    ): Double {
        val finite = values
            .filter(Double::isFinite)
            .sorted()
        if (finite.isEmpty()) return 0.0
        val position = (finite.size - 1) * 0.50
        val lower = position.toInt()
        val upper = minOf(lower + 1, finite.lastIndex)
        val fraction = position - lower
        return finite[lower] * (1.0 - fraction) +
            finite[upper] * fraction
    }
}
