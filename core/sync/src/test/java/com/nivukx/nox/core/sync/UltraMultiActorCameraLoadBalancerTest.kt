package com.nivukx.nox.core.sync

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UltraMultiActorCameraLoadBalancerTest {

    private fun camera(
        id: String,
        sector: Int,
    ) =
        FrontierCameraAuthorityFederationEngine.CameraEvidence(
            cameraId = id,
            sector = sector,
            health = 0.98,
            synchronization = 0.98,
            calibration = 0.98,
            coverage = 0.95,
            actorVisibility = 0.90,
            thermalHeadroom = 0.92,
            networkHeadroom = 0.90,
            freshness = 0.99,
            actorVisibilityById =
                (0..63).associateWith {
                    if (id.endsWith(it.toString())) 0.99 else 0.88
                },
        )

    @Test
    fun keepsPerActorAuthorityAboveMinimum() {
        val balancer =
            UltraMultiActorCameraLoadBalancer(
                UltraMultiActorCameraLoadBalancer.Config(
                    authorityCamerasPerActor = 8,
                    minimumAuthorityCamerasPerActor = 4,
                    maximumAuthorityActorLoad = 16,
                ),
            )

        val cameras =
            (0 until 32).map {
                camera(
                    id = "camera-$it",
                    sector = it % 16,
                )
            }

        val actors =
            (0 until 50).map {
                UltraMultiActorCameraLoadBalancer.Actor(
                    actorId = it,
                    priority = 0.5 + (it % 5) * 0.05,
                    visibilityByCamera =
                        cameras.associate { camera ->
                            camera.cameraId to
                                (0.70 + (it % 3) * 0.08)
                        },
                )
            }

        val result =
            balancer.allocate(
                actors = actors,
                cameras = cameras,
            )

        assertEquals(50, result.assignments.size)
        assertTrue(
            result.assignments.all {
                it.authorityCameraIds.size >= 4
            },
        )
        assertTrue(
            result.minimumActorCoverage >= 0.70,
        )
        assertTrue(
            result.overloadedCameraCount == 0,
        )
    }

    @Test
    fun distributesAuthorityLoadAcrossCameras() {
        val balancer =
            UltraMultiActorCameraLoadBalancer(
                UltraMultiActorCameraLoadBalancer.Config(
                    authorityCamerasPerActor = 6,
                    minimumAuthorityCamerasPerActor = 4,
                    maximumAuthorityActorLoad = 12,
                ),
            )

        val cameras =
            (0 until 24).map {
                camera(
                    id = "camera-$it",
                    sector = it % 12,
                )
            }

        val actors =
            (0 until 40).map {
                UltraMultiActorCameraLoadBalancer.Actor(
                    actorId = it,
                    priority = 0.8,
                    visibilityByCamera =
                        cameras.associate { camera ->
                            camera.cameraId to
                                if (camera.sector == it % 12) {
                                    0.99
                                } else {
                                    0.78
                                }
                        },
                )
            }

        val result =
            balancer.allocate(
                actors = actors,
                cameras = cameras,
            )

        val loads =
            result.cameraAuthorityLoad.values

        assertTrue(loads.isNotEmpty())
        assertTrue(loads.maxOrNull()!! <= 12)
        assertTrue(
            loads.count { it > 0 } >= 12,
        )
        assertTrue(result.meanViewDiversity > 0.20)
    }

    @Test
    fun resetClearsHysteresisState() {
        val balancer =
            UltraMultiActorCameraLoadBalancer()

        val cameras =
            (0 until 8).map {
                camera("camera-$it", it)
            }

        val actor =
            UltraMultiActorCameraLoadBalancer.Actor(
                actorId = 1,
                priority = 0.9,
                visibilityByCamera =
                    cameras.associate {
                        it.cameraId to 0.9
                    },
            )

        balancer.allocate(
            actors = listOf(actor),
            cameras = cameras,
        )

        balancer.reset()

        val second =
            balancer.allocate(
                actors = listOf(actor),
                cameras = cameras,
            )

        assertEquals(1, second.assignments.size)
        assertTrue(second.assignments.single().score >= 0.0)
    }
}
