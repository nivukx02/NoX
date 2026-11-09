package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UltraMultiActorCameraLoadBalancerGeometryTest {
    private fun camera(
        id: String,
        sector: Int,
        position: Vec3,
        forward: Vec3,
        visibility: Double = 0.95,
    ): FrontierCameraAuthorityFederationEngine.CameraEvidence =
        FrontierCameraAuthorityFederationEngine.CameraEvidence(
            cameraId = id,
            sector = sector,
            health = 1.0,
            synchronization = 1.0,
            calibration = 1.0,
            coverage = 1.0,
            actorVisibility = visibility,
            thermalHeadroom = 1.0,
            networkHeadroom = 1.0,
            freshness = 1.0,
            positionMeters = position,
            forward = forward,
        )

    @Test
    fun metricPlacementDiscriminatesRearView() {
        val balancer =
            UltraMultiActorCameraLoadBalancer(
                UltraMultiActorCameraLoadBalancer.Config(
                    authorityCamerasPerActor = 2,
                    minimumAuthorityCamerasPerActor = 2,
                    maximumAuthorityActorLoad = 8,
                ),
            )
        val actor =
            UltraMultiActorCameraLoadBalancer.Actor(
                actorId = 4,
                priority = 1.0,
                visibilityByCamera =
                    mapOf(
                        "front" to 0.95,
                        "rear" to 0.95,
                        "side" to 0.95,
                    ),
                worldPositionMeters =
                    Vec3(0.0, 0.0, 0.0),
            )
        val result =
            balancer.allocate(
                actors = listOf(actor),
                cameras =
                    listOf(
                        camera(
                            "front",
                            0,
                            Vec3(0.0, 0.0, 5.0),
                            Vec3(0.0, 0.0, -1.0),
                        ),
                        camera(
                            "side",
                            1,
                            Vec3(5.0, 0.0, 0.0),
                            Vec3(-1.0, 0.0, 0.0),
                        ),
                        camera(
                            "rear",
                            2,
                            Vec3(0.0, 0.0, -5.0),
                            Vec3(0.0, 0.0, 1.0),
                        ),
                    ),
            )
        val assignment = result.assignments.single()
        assertEquals(2, assignment.authorityCameraIds.size)
        assertTrue("front" in assignment.authorityCameraIds)
        assertTrue("rear" !in assignment.authorityCameraIds)
        assertTrue(assignment.viewDiversity > 0.45)
    }

    @Test
    fun difficult_actor_gets_extra_authority_depth_without_reducing_baseline() {
        val balancer =
            UltraMultiActorCameraLoadBalancer(
                UltraMultiActorCameraLoadBalancer.Config(
                    authorityCamerasPerActor = 4,
                    minimumAuthorityCamerasPerActor = 4,
                    maximumAuthorityActorLoad = 32,
                    maximumAdaptiveAuthorityBonus = 4,
                    maximumAdaptiveAuthorityCamerasPerActor = 8,
                ),
            )

        val cameras =
            (0 until 12).map { index ->
                val angle =
                    index.toDouble() /
                        12.0 *
                        Math.PI *
                        2.0
                camera(
                    id = "adaptive-$index",
                    sector = index % 12,
                    position =
                        Vec3(
                            kotlin.math.cos(angle) * 6.0,
                            1.6,
                            kotlin.math.sin(angle) * 6.0,
                        ),
                    forward =
                        Vec3(
                            -kotlin.math.cos(angle),
                            -0.18,
                            -kotlin.math.sin(angle),
                        ).normalized(),
                )
            }

        val easy =
            UltraMultiActorCameraLoadBalancer.Actor(
                actorId = 1,
                priority = 0.50,
                visibilityByCamera =
                    cameras.associate {
                        it.cameraId to 0.97
                    },
                worldPositionMeters =
                    Vec3(0.0, 0.0, 0.0),
                occlusionRisk = 0.10,
            )

        val difficult =
            easy.copy(
                actorId = 2,
                priority = 0.95,
                visibilityByCamera =
                    cameras.associate {
                        it.cameraId to 0.42
                    },
                occlusionRisk = 0.90,
            )

        val result =
            balancer.allocate(
                actors = listOf(easy, difficult),
                cameras = cameras,
            )

        assertEquals(
            4,
            result.requestedAuthorityDepthByActor[1],
        )
        assertTrue(
            (result.requestedAuthorityDepthByActor[2] ?: 0) > 4,
        )
        assertTrue(
            result.assignments.all {
                it.authorityCameraIds.size >= 4
            },
        )
    }

    @Test
    fun fiftyActorsRemainBoundedAndReceiveMinimumViews() {
        val balancer =
            UltraMultiActorCameraLoadBalancer(
                UltraMultiActorCameraLoadBalancer.Config(
                    maximumActors = 64,
                    authorityCamerasPerActor = 4,
                    minimumAuthorityCamerasPerActor = 4,
                    maximumAuthorityActorLoad = 16,
                ),
            )
        val cameras =
            (0 until 16).map { index ->
                val angle =
                    index.toDouble() /
                        16.0 *
                        Math.PI *
                        2.0
                camera(
                    id = "c$index",
                    sector = index % 8,
                    position =
                        Vec3(
                            kotlin.math.cos(angle) * 6.0,
                            1.5,
                            kotlin.math.sin(angle) * 6.0,
                        ),
                    forward =
                        Vec3(
                            -kotlin.math.cos(angle),
                            -0.18,
                            -kotlin.math.sin(angle),
                        ).normalized(),
                )
            }
        val actors =
            (0 until 50).map { actorId ->
                val angle =
                    actorId.toDouble() /
                        50.0 *
                        Math.PI *
                        2.0
                UltraMultiActorCameraLoadBalancer.Actor(
                    actorId = actorId,
                    priority =
                        0.5 +
                            (actorId % 5) * 0.1,
                    visibilityByCamera =
                        cameras.associate {
                            it.cameraId to 0.90
                        },
                    worldPositionMeters =
                        Vec3(
                            kotlin.math.cos(angle) * 0.9,
                            1.2,
                            kotlin.math.sin(angle) * 0.9,
                        ),
                )
            }
        val result =
            balancer.allocate(actors, cameras)
        assertEquals(50, result.assignments.size)
        assertTrue(
            result.assignments.all {
                it.authorityCameraIds.size >= 4
            },
        )
        assertTrue(
            (result.cameraAuthorityLoad.values.maxOrNull() ?: 0) <= 16,
        )
        assertEquals(0, result.overloadedCameraCount)
    }
}
