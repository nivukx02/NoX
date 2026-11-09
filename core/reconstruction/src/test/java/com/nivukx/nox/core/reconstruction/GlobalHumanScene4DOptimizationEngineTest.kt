package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class GlobalHumanScene4DOptimizationEngineTest {
    @Test
    fun globalRefinementReducesResidual() {
        val point = GlobalHumanScene4DOptimizationEngine.ActorPoint(
            1,2,Vec3(0.0,0.0,0.0),
            listOf(Vec3(0.10,0.0,0.0),Vec3(0.12,0.0,0.0)),
            0.95,
        )
        val result = GlobalHumanScene4DOptimizationEngine().optimize(
            cameras = listOf(GlobalHumanScene4DOptimizationEngine.CameraState("a",Vec3(0.0,0.0,0.0))),
            points = listOf(point),
        )
        assertTrue(result.finalResidualMeters < result.initialResidualMeters)
    }

    @Test
    fun globalPointBudgetIsBounded() {
        val engine=GlobalHumanScene4DOptimizationEngine(GlobalHumanScene4DOptimizationEngine.Config(maximumPoints=1))
        val points=listOf(
            GlobalHumanScene4DOptimizationEngine.ActorPoint(1,1,Vec3(0.0,0.0,0.0),listOf(Vec3(1.0,0.0,0.0)),1.0),
            GlobalHumanScene4DOptimizationEngine.ActorPoint(1,2,Vec3(0.0,0.0,0.0),listOf(Vec3(2.0,0.0,0.0)),1.0),
        )
        val result=engine.optimize(emptyList(),points)
        assertTrue(result.points.size == 1)
        assertTrue(result.warnings.contains("global-point-budget-truncated"))
    }
}


class GlobalHumanScene4DOptimizationEngineFrontierTest {
    @Test
    fun temporalAndSurfaceEvidenceContributeToRefinementDiagnostics() {
        val point = GlobalHumanScene4DOptimizationEngine.ActorPoint(
            actorId = 7,
            pointId = 11,
            positionMeters = Vec3(0.0, 0.0, 0.0),
            cameraObservations = listOf(
                Vec3(0.20, 0.0, 0.0),
                Vec3(0.22, 0.0, 0.0),
            ),
            confidence = 0.95,
            temporalObservations = listOf(
                GlobalHumanScene4DOptimizationEngine.TemporalObservation(
                    timestampNanos = 1_000_000_000L,
                    positionMeters = Vec3(0.18, 0.0, 0.0),
                    confidence = 0.9,
                ),
                GlobalHumanScene4DOptimizationEngine.TemporalObservation(
                    timestampNanos = 1_016_666_667L,
                    positionMeters = Vec3(0.19, 0.0, 0.0),
                    confidence = 0.9,
                ),
            ),
        )

        val result = GlobalHumanScene4DOptimizationEngine().optimize(
            cameras = listOf(
                GlobalHumanScene4DOptimizationEngine.CameraState(
                    cameraId = "a",
                    positionMeters = Vec3(0.0, 0.0, 0.0),
                    fixed = true,
                ),
            ),
            points = listOf(point),
            surfaceConstraints = listOf(
                GlobalHumanScene4DOptimizationEngine.SurfaceConstraint(
                    actorId = 7,
                    pointId = 11,
                    targetPositionMeters = Vec3(0.195, 0.0, 0.0),
                    weight = 1.0,
                    confidence = 0.95,
                ),
            ),
        )

        assertTrue(result.pointDiagnostics.size == 1)
        assertTrue(result.trajectoryDiagnostics.size == 1)
        assertTrue(result.finalResidualMeters <= result.initialResidualMeters)
        assertTrue(
            result.points.single().positionMeters.x > 0.05,
        )
    }

    @Test
    fun fixedReferenceCameraCannotReceiveTranslationCorrection() {
        val result = GlobalHumanScene4DOptimizationEngine().optimize(
            cameras = listOf(
                GlobalHumanScene4DOptimizationEngine.CameraState(
                    cameraId = "a",
                    positionMeters = Vec3(0.0, 0.0, 0.0),
                    fixed = true,
                ),
                GlobalHumanScene4DOptimizationEngine.CameraState(
                    cameraId = "b",
                    positionMeters = Vec3(1.0, 0.0, 0.0),
                ),
            ),
            points = listOf(
                GlobalHumanScene4DOptimizationEngine.ActorPoint(
                    actorId = 1,
                    pointId = 1,
                    positionMeters = Vec3(0.0, 0.0, 0.0),
                    cameraObservations = emptyList(),
                    confidence = 1.0,
                    cameraObservationById = mapOf(
                        "a" to Vec3(0.0, 0.0, 0.0),
                        "b" to Vec3(-0.05, 0.0, 0.0),
                    ),
                ),
            ),
        )

        assertTrue(
            result.cameras.first { it.cameraId == "a" }
                .correctionMeters.norm() < 1e-9,
        )
    }
}


class GlobalHumanScene4DOptimizationEngineInteractionTest {
    @Test
    fun interpenetratingActorPointsReceiveSeparationCorrection() {
        val result =
            GlobalHumanScene4DOptimizationEngine(
                GlobalHumanScene4DOptimizationEngine.Config(
                    minimumActorSeparationMeters = 0.20,
                    interactionSeparationWeight = 0.80,
                    maximumStepMeters = 0.10,
                ),
            ).optimize(
                cameras = listOf(
                    GlobalHumanScene4DOptimizationEngine.CameraState(
                        cameraId = "reference",
                        positionMeters = Vec3(0.0, 0.0, 0.0),
                        fixed = true,
                    ),
                ),
                points = listOf(
                    GlobalHumanScene4DOptimizationEngine.ActorPoint(
                        actorId = 1,
                        pointId = 1,
                        positionMeters = Vec3(0.0, 0.0, 1.0),
                        cameraObservations = listOf(
                            Vec3(0.0, 0.0, 1.0),
                        ),
                        confidence = 0.95,
                    ),
                    GlobalHumanScene4DOptimizationEngine.ActorPoint(
                        actorId = 2,
                        pointId = 1,
                        positionMeters = Vec3(0.02, 0.0, 1.0),
                        cameraObservations = listOf(
                            Vec3(0.02, 0.0, 1.0),
                        ),
                        confidence = 0.95,
                    ),
                ),
                interactionConstraints = listOf(
                    GlobalHumanScene4DOptimizationEngine.InteractionConstraint(
                        actorA = 1,
                        pointA = 1,
                        actorB = 2,
                        pointB = 1,
                        desiredDistanceMeters = 0.20,
                        contactProbability = 0.0,
                        confidence = 1.0,
                    ),
                ),
            )

        val first = result.points.first {
            it.actorId == 1
        }.positionMeters
        val second = result.points.first {
            it.actorId == 2
        }.positionMeters

        assertTrue(first.distanceTo(second) > 0.02)
        assertTrue(
            result.finalResidualMeters <=
                result.initialResidualMeters,
        )
    }

    @Test
    fun highContactProbabilityKeepsInteractionAtDesiredDistance() {
        val result =
            GlobalHumanScene4DOptimizationEngine(
                GlobalHumanScene4DOptimizationEngine.Config(
                    minimumActorSeparationMeters = 0.10,
                    interactionSeparationWeight = 0.55,
                    contactAttractionWeight = 0.35,
                ),
            ).optimize(
                cameras = emptyList(),
                points = listOf(
                    GlobalHumanScene4DOptimizationEngine.ActorPoint(
                        actorId = 3,
                        pointId = 4,
                        positionMeters = Vec3(0.0, 0.0, 0.0),
                        cameraObservations = emptyList(),
                        confidence = 1.0,
                    ),
                    GlobalHumanScene4DOptimizationEngine.ActorPoint(
                        actorId = 4,
                        pointId = 4,
                        positionMeters = Vec3(0.55, 0.0, 0.0),
                        cameraObservations = emptyList(),
                        confidence = 1.0,
                    ),
                ),
                interactionConstraints = listOf(
                    GlobalHumanScene4DOptimizationEngine.InteractionConstraint(
                        actorA = 3,
                        pointA = 4,
                        actorB = 4,
                        pointB = 4,
                        desiredDistanceMeters = 0.20,
                        contactProbability = 1.0,
                        confidence = 1.0,
                    ),
                ),
            )

        val first = result.points.first {
            it.actorId == 3
        }.positionMeters
        val second = result.points.first {
            it.actorId == 4
        }.positionMeters
        val separation = first.distanceTo(second)

        assertTrue(separation < 0.55)
        assertTrue(separation >= 0.10)
    }
}
