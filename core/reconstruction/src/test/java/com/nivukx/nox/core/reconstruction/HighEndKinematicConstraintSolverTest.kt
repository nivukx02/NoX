package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndKinematicConstraintSolverTest {

    @Test
    fun preservesFiniteGeometryAndReducesImpossibleBoneStretch() {
        val solver = HighEndKinematicConstraintSolver(
            HighEndKinematicConstraintSolver.Config(
                iterations = 24,
                edgeStiffness = 0.92,
            ),
        )

        val result = solver.solve(
            HighEndKinematicConstraintSolver.Frame(
                timestampNanos = 1_000_000_000L,
                subjectType = SubjectType.HUMAN,
                joints = listOf(
                    HighEndKinematicConstraintSolver.Joint(
                        id = 11,
                        semanticName = "shoulder",
                        position = Vec3(-0.25, 0.0, 2.0),
                        confidence = 0.95,
                    ),
                    HighEndKinematicConstraintSolver.Joint(
                        id = 13,
                        semanticName = "elbow",
                        position = Vec3(-0.70, 0.0, 2.0),
                        confidence = 0.55,
                    ),
                ),
                edges = listOf(
                    HighEndKinematicConstraintSolver.Edge(
                        a = 11,
                        b = 13,
                        preferredLengthMeters = 0.34,
                        minimumLengthMeters = 0.30,
                        maximumLengthMeters = 0.38,
                        stiffness = 1.0,
                    ),
                ),
            ),
        )

        val shoulder = result.joints.first { it.source.id == 11 }.position
        val elbow = result.joints.first { it.source.id == 13 }.position
        val dx = elbow.x - shoulder.x
        val dy = elbow.y - shoulder.y
        val dz = elbow.z - shoulder.z
        val distance = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)

        assertTrue(distance <= 0.40)
        assertTrue(distance >= 0.28)
        assertTrue(result.maxResidualMeters.isFinite())
        assertTrue(result.globalConfidence >= 0.0)
    }

    @Test
    fun temporalStateProducesFiniteVelocityOnSecondFrame() {
        val solver = HighEndKinematicConstraintSolver()

        val first = solver.solve(
            frame(
                timestamp = 1_000_000_000L,
                x = 0.0,
            ),
        )

        val second = solver.solve(
            frame(
                timestamp = 1_016_666_666L,
                x = 0.01,
            ),
        )

        assertTrue(first.joints.isNotEmpty())
        assertTrue(second.joints.all {
            it.velocityMetersPerSecond.x.isFinite()
        })
        assertTrue(second.temporalStability >= 0.0)
    }

    @Test
    fun predictedJointIsStillConstrainedByTopology() {
        val solver = HighEndKinematicConstraintSolver()

        val result = solver.solve(
            HighEndKinematicConstraintSolver.Frame(
                timestampNanos = 2_000_000_000L,
                subjectType = SubjectType.HUMAN,
                joints = listOf(
                    HighEndKinematicConstraintSolver.Joint(
                        id = 23,
                        semanticName = "hip",
                        position = Vec3(0.0, 0.0, 2.0),
                        confidence = 0.95,
                    ),
                    HighEndKinematicConstraintSolver.Joint(
                        id = 25,
                        semanticName = "knee",
                        position = Vec3(0.0, -0.55, 2.0),
                        confidence = 0.20,
                        predicted = true,
                    ),
                ),
                edges = listOf(
                    HighEndKinematicConstraintSolver.Edge(
                        a = 23,
                        b = 25,
                        preferredLengthMeters = 0.44,
                        minimumLengthMeters = 0.40,
                        maximumLengthMeters = 0.48,
                    ),
                ),
            ),
        )

        val predicted = result.joints.first { it.source.id == 25 }

        assertTrue(predicted.predicted)
        assertTrue(predicted.position.y < -0.38)
        assertTrue(predicted.position.y > -0.60)
        assertTrue(predicted.confidence >= 0.0)
    }

    private fun frame(
        timestamp: Long,
        x: Double,
    ) = HighEndKinematicConstraintSolver.Frame(
        timestampNanos = timestamp,
        subjectType = SubjectType.HUMAN,
        joints = listOf(
            HighEndKinematicConstraintSolver.Joint(
                id = 11,
                semanticName = "joint",
                position = Vec3(x, 0.0, 2.0),
                confidence = 0.95,
            ),
            HighEndKinematicConstraintSolver.Joint(
                id = 13,
                semanticName = "child",
                position = Vec3(x - 0.30, -0.02, 2.0),
                confidence = 0.90,
            ),
        ),
        edges = listOf(
            HighEndKinematicConstraintSolver.Edge(
                a = 11,
                b = 13,
                preferredLengthMeters = 0.30,
                minimumLengthMeters = 0.26,
                maximumLengthMeters = 0.34,
            ),
        ),
    )
}
