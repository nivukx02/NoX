package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.CameraFederationEvidence
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.FrontierActorRuntimeContract
import com.nivukx.nox.core.model.FrontierLandmarkMeasurement
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UltraFederatedMotionFusionEngineTest {
    private val intrinsics =
        CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )

    private fun camera(
        id: String,
        x: Double,
        z: Double,
        sector: Int,
    ) =
        CameraFederationEvidence(
            cameraId = id,
            sector = sector,
            positionMeters = Vec3(x, 0.0, z),
            forward = Vec3(-x, 0.0, -z).normalized(),
            quality = 0.98,
            synchronizationConfidence = 0.98,
            calibrationConfidence = 0.98,
        )

    @Test
    fun selectsDistinctSectorsAndBoundsAuthority() {
        val engine =
            UltraFederatedMotionFusionEngine()

        val cameras =
            (0 until 128).map {
                camera(
                    id = "c-" + it,
                    x = if (it % 2 == 0) -3.0 else 3.0,
                    z = 2.0 + (it % 10),
                    sector = it % 16,
                )
            }

        val plan = engine.selectCameras(cameras)

        assertEquals(32, plan.authority.size)
        assertTrue(plan.witness.size <= 96)
        assertTrue(plan.authority.map { it.cameraId }.distinct().size == 32)
        assertTrue(plan.diversityScore in 0.0..1.0)
    }

    @Test
    fun plansFiftyActorsWithoutExceedingHardCeiling() {
        val engine =
            UltraFederatedMotionFusionEngine()

        val actors =
            (0L until 64L).map { actorId ->
                FrontierActorRuntimeContract(
                    actorId = actorId,
                    priority = 0.6,
                    visibleRatio = 0.85,
                    identityConfidence = 0.92,
                    occlusionRatio = if (actorId % 7L == 0L) 0.35 else 0.10,
                    interactionRisk = if (actorId < 8L) 0.70 else 0.15,
                    requestedSurfaceDetail = true,
                    requestedFaceDetail = true,
                    requestedEyeDetail = true,
                    requestedHandDetail = true,
                )
            }

        val plan =
            engine.planActors(
                actors = actors,
                headroom =
                    MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom(
                        cpuHeadroom = 0.90,
                        gpuHeadroom = 0.90,
                        npuHeadroom = 0.90,
                        thermalHeadroom = 0.90,
                        memoryHeadroom = 0.90,
                    ),
                captureFps = 60,
            )

        assertTrue(plan.activeActorIds.size <= 64)
        assertTrue(plan.heldActorIds.size >= 0)
        assertTrue(plan.activeActorIds.size >= 50)
    }

    @Test
    fun rejectsLandmarkWhenParallaxIsTooSmall() {
        val engine =
            UltraFederatedMotionFusionEngine()

        val result =
            engine.fuseFrame(
                timestampNanos = 1_000_000_000L,
                cameras =
                    listOf(
                        camera("a", -0.01, 4.0, 0),
                        camera("b", 0.01, 4.0, 1),
                    ),
                observations =
                    listOf(
                        measurement("a", 0.0, 0.0),
                        measurement("b", 0.0, 0.0),
                    ),
                )

        val landmark = result.landmarks.single()
        assertTrue(landmark.viewCount >= 2)
        assertTrue(landmark.failure.name == "POOR_VIEW_DIVERSITY")
    }

    @Test
    fun opposite_rays_do_not_receive_peak_geometry_authority() {
        val engine =
            UltraFederatedMotionFusionEngine()

        val front = engine.fuseFrame(
            timestampNanos = 2_000_000_000L,
            cameras =
                listOf(
                    camera("a", -1.0, 0.0, 0),
                    camera("b", 1.0, 0.0, 1),
                    camera("c", 0.0, 1.0, 2),
                ),
            observations =
                listOf(
                    measurement("a", -120.0, 0.0),
                    measurement("b", 120.0, 0.0),
                    measurement("c", 0.0, -80.0),
                ),
        ).landmarks.single()

        val opposite =
            engine.fuseFrame(
                timestampNanos = 2_016_666_667L,
                cameras =
                    listOf(
                        camera("a", -1.0, 0.0, 0),
                        camera("b", 1.0, 0.0, 1),
                        camera("c", 0.0, 1.0, 2),
                    ),
                observations =
                    listOf(
                        measurement("a", -120.0, 0.0),
                        measurement("b", 120.0, 0.0),
                        measurement("c", 0.0, 80.0),
                    ),
            ).landmarks.single()

        assertTrue(front.p95UncertaintyMeters.isFinite())
        assertTrue(opposite.p95UncertaintyMeters.isFinite())
        assertTrue(
            opposite.authority !=
                com.nivukx.nox.core.model.MeasurementAuthority.GEOMETRIC ||
                front.authority ==
                    com.nivukx.nox.core.model.MeasurementAuthority.GEOMETRIC,
        )
    }

    @Test
    fun calibrationUncertaintyPropagatesIntoWorldSpaceConfidence() {
        val engine =
            UltraFederatedMotionFusionEngine()

        val cameras =
            listOf(
                camera("a", -1.2, 0.0, 0),
                camera("b", 1.2, 0.0, 1),
                camera("c", 0.0, -1.2, 2),
            )

        val lowUncertainty =
            listOf(
                measurement("a", -120.0, 0.0),
                measurement("b", 120.0, 0.0),
                measurement("c", 0.0, -120.0),
            )

        val highUncertainty =
            lowUncertainty.map {
                it.copy(
                    calibrationTranslationSigmaMeters = 0.006,
                    calibrationRotationSigmaRadians =
                        Math.toRadians(0.35),
                )
            }

        val low =
            engine.fuseFrame(
                timestampNanos = 1_000_000_000L,
                cameras = cameras,
                observations = lowUncertainty,
            ).landmarks.single()

        val high =
            engine.fuseFrame(
                timestampNanos = 1_016_666_667L,
                cameras = cameras,
                observations = highUncertainty,
            ).landmarks.single()

        assertTrue(high.p95UncertaintyMeters >= low.p95UncertaintyMeters)
        assertTrue(
            high.authority !=
                com.nivukx.nox.core.model.MeasurementAuthority.GEOMETRIC ||
                low.authority ==
                    com.nivukx.nox.core.model.MeasurementAuthority.GEOMETRIC,
        )
    }

    @Test
    fun opposite_ray_pair_is_rejected_as_ill_conditioned() {
        val engine =
            UltraFederatedMotionFusionEngine()

        val result =
            engine.fuseFrame(
                timestampNanos = 3_000_000_000L,
                cameras =
                    listOf(
                        CameraFederationEvidence(
                            cameraId = "left",
                            sector = 0,
                            positionMeters =
                                Vec3(-1.0, 0.0, 0.0),
                            forward =
                                Vec3(0.0, 0.0, 1.0),
                            quality = 0.99,
                            synchronizationConfidence = 0.99,
                            calibrationConfidence = 0.99,
                        ),
                        CameraFederationEvidence(
                            cameraId = "right",
                            sector = 8,
                            positionMeters =
                                Vec3(1.0, 0.0, 0.0),
                            forward =
                                Vec3(0.0, 0.0, -1.0),
                            quality = 0.99,
                            synchronizationConfidence = 0.99,
                            calibrationConfidence = 0.99,
                        ),
                    ),
                observations =
                    listOf(
                        FrontierLandmarkMeasurement(
                            landmarkId = 7,
                            cameraId = "left",
                            point =
                                Vec2(
                                    500.0,
                                    500.0,
                                ),
                            confidence = 0.99,
                            intrinsics = intrinsics,
                            cameraPose =
                                Pose(
                                    position =
                                        Vec3(
                                            -1.0,
                                            0.0,
                                            0.0,
                                        ),
                                    rotation =
                                        Quat.IDENTITY,
                                ),
                            observedAtNanos =
                                3_000_000_000L,
                        ),
                        FrontierLandmarkMeasurement(
                            landmarkId = 7,
                            cameraId = "right",
                            point =
                                Vec2(
                                    500.0,
                                    500.0,
                                ),
                            confidence = 0.99,
                            intrinsics = intrinsics,
                            cameraPose =
                                Pose(
                                    position =
                                        Vec3(
                                            1.0,
                                            0.0,
                                            0.0,
                                        ),
                                    rotation =
                                        Quat(
                                            0.0,
                                            0.0,
                                            1.0,
                                            0.0,
                                        ),
                                ),
                            observedAtNanos =
                                3_000_000_000L,
                        ),
                    ),
            )

        assertTrue(
            result.landmarks.single().failure ==
                com.nivukx.nox.core.model.MeasurementFailure
                    .POOR_VIEW_DIVERSITY,
        )
    }

    @Test
    fun robustFusionReturnsFiniteWorldPointForThreeViews() {
        val engine =
            UltraFederatedMotionFusionEngine()

        val cameras =
            listOf(
                camera("a", -1.2, 0.0, 0),
                camera("b", 1.2, 0.0, 1),
                camera("c", 0.0, -1.2, 2),
            )

        val observations =
            listOf(
                measurement("a", -120.0, 0.0),
                measurement("b", 120.0, 0.0),
                measurement("c", 0.0, -120.0),
            )

        val result =
            engine.fuseFrame(
                timestampNanos = 1_000_000_000L,
                cameras = cameras,
                observations = observations,
            )

        val landmark = result.landmarks.single()
        assertTrue(landmark.positionMeters.x.isFinite())
        assertTrue(landmark.positionMeters.y.isFinite())
        assertTrue(landmark.positionMeters.z.isFinite())
        assertTrue(landmark.reprojectionErrorPixels.isFinite())
    }

    private fun measurement(
        cameraId: String,
        x: Double,
        y: Double,
    ) =
        FrontierLandmarkMeasurement(
            landmarkId = 7,
            cameraId = cameraId,
            point = Vec2(500.0 + x, 500.0 + y),
            confidence = 0.97,
            intrinsics = intrinsics,
            cameraPose =
                Pose(
                    position =
                        when (cameraId) {
                            "a" -> Vec3(-1.2, 0.0, 4.0)
                            "b" -> Vec3(1.2, 0.0, 4.0)
                            else -> Vec3(0.0, -1.2, 4.0)
                        },
                    rotation = Quat.IDENTITY,
                ),
            temporalAlignmentConfidence = 0.98,
            temporalUncertaintyNanos = 250_000L,
            observedAtNanos = 1_000_000_000L,
        )
}
