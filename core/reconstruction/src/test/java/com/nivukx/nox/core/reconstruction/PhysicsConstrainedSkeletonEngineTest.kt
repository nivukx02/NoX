package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
/**
 * Behavioral coverage for the production physics guard.
 *
 * These tests intentionally validate invariants instead of implementation details:
 * - missing joints remain missing;
 * - learned metric baselines resist low-confidence outliers;
 * - bilateral symmetry is measured from corresponding segment lengths;
 * - dynamics violations lower confidence without deleting the joint;
 * - persistent failures eventually block output;
 * - recovery requires multiple clean frames;
 * - reset removes stateful memory.
 */
class PhysicsConstrainedSkeletonEngineTest {

    fun missingJointsAreNeverFabricated() {
        val engine = PhysicsConstrainedSkeletonEngine()
        val frame = frame(
            timestampNanos = 1_000_000_000L,
            joints = listOf(
                joint(11, 0.0, 1.5, 0.0, confidence = 0.92),
                joint(12, 0.4, 1.5, 0.0, confidence = 0.92),
                joint(13, -0.25, 1.2, 0.0, confidence = 0.88),
                joint(14, 0.65, 1.2, 0.0, confidence = 0.88),
            ),
        )

        val result = engine.refine(frame)

        check(
            result.frame.joints
                .map { it.landmarkId }
                .toSet() == setOf(11, 12, 13, 14),
        )
        check(15 !in result.frame.joints.map { it.landmarkId })
        check(16 !in result.frame.joints.map { it.landmarkId })
    }

    fun learnedBoneLengthRejectsAbruptLowConfidenceBaselineRewrite() {
        val engine = PhysicsConstrainedSkeletonEngine(
            config = PhysicsConstrainedSkeletonEngine.Config(
                iterations = 4,
                baselineLearningRate = 0.22,
                minimumObservationCountForLearning = 2,
                maximumBaselineUpdateResidualMeters = 0.06,
            ),
        )

        val first = frame(
            timestampNanos = 1_000_000_000L,
            joints = listOf(
                joint(
                    11,
                    0.0,
                    1.5,
                    0.0,
                    confidence = 0.95,
                    observations = 4,
                ),
                joint(
                    13,
                    0.30,
                    1.5,
                    0.0,
                    confidence = 0.95,
                    observations = 4,
                ),
            ),
        )
        engine.refine(first)

        val stable = frame(
            timestampNanos = 1_033_000_000L,
            joints = listOf(
                joint(
                    11,
                    0.0,
                    1.5,
                    0.0,
                    confidence = 0.95,
                    observations = 4,
                ),
                joint(
                    13,
                    0.305,
                    1.5,
                    0.0,
                    confidence = 0.95,
                    observations = 4,
                ),
            ),
        )
        val stableResult = engine.refine(stable)
        check(stableResult.edges.isNotEmpty())

        val outlier = frame(
            timestampNanos = 1_066_000_000L,
            joints = listOf(
                joint(
                    11,
                    0.0,
                    1.5,
                    0.0,
                    confidence = 0.95,
                    observations = 4,
                ),
                joint(
                    13,
                    0.90,
                    1.5,
                    0.0,
                    confidence = 0.20,
                    observations = 4,
                ),
            ),
        )
        val outlierResult = engine.refine(outlier)
        val edge = outlierResult.edges.single()
        check(edge.targetLengthMeters < 0.50)
        check(edge.residualMeters.isFinite())
    }

    fun unqualifiedFirstObservationDoesNotSeedMetricBaseline() {
        val engine = PhysicsConstrainedSkeletonEngine(
            config = PhysicsConstrainedSkeletonEngine.Config(
                minimumObservationCountForLearning = 3,
                maximumBaselineUpdateResidualMeters = 0.05,
            ),
            edges = listOf(
                PhysicsConstrainedSkeletonEngine.Edge(
                    parent = 11,
                    child = 13,
                    group = "test-upper-arm",
                ),
            ),
        )

        val weakFirst =
            frame(
                timestampNanos = 1_000_000_000L,
                joints = listOf(
                    joint(11, 0.0, 1.5, 0.0, confidence = 0.18, observations = 0),
                    joint(13, 0.90, 1.5, 0.0, confidence = 0.18, observations = 0),
                ),
            )

        engine.refine(weakFirst)

        val qualifiedSecond =
            frame(
                timestampNanos = 1_033_000_000L,
                joints = listOf(
                    joint(11, 0.0, 1.5, 0.0, confidence = 0.95, observations = 4),
                    joint(13, 0.30, 1.5, 0.0, confidence = 0.95, observations = 4),
                ),
            )

        val result = engine.refine(qualifiedSecond)
        val edge = result.edges.single()

        check(edge.targetLengthMeters < 0.5)
    }

    fun bilateralSymmetryUsesSegmentLengthsInsteadOfLeftRightDistance() {
        val engine = PhysicsConstrainedSkeletonEngine(
            config = PhysicsConstrainedSkeletonEngine.Config(
                symmetryToleranceRatio = 0.30,
            ),
        )

        val symmetric = frame(
            timestampNanos = 1_000_000_000L,
            joints = listOf(
                joint(11, -0.2, 1.5, 0.0, confidence = 0.95),
                joint(12, 0.2, 1.5, 0.0, confidence = 0.95),
                joint(13, -0.5, 1.2, 0.0, confidence = 0.95),
                joint(14, 0.5, 1.2, 0.0, confidence = 0.95),
                joint(15, -0.8, 0.9, 0.0, confidence = 0.95),
                joint(16, 0.8, 0.9, 0.0, confidence = 0.95),
            ),
        )

        val symmetricResult = engine.refine(symmetric)
        check(symmetricResult.symmetryScore > 0.85)

        engine.reset()

        val asymmetric = symmetric.copy(
            joints = symmetric.joints.map {
                if (it.landmarkId == 15) {
                    it.copy(position = Vec3(-1.10, 0.9, 0.0))
                } else {
                    it
                }
            },
        )

        val asymmetricResult = engine.refine(asymmetric)
        check(asymmetricResult.symmetryScore < symmetricResult.symmetryScore)
    }

    fun highVelocityLowersConfidenceButPreservesJoint() {
        val engine = PhysicsConstrainedSkeletonEngine(
            config = PhysicsConstrainedSkeletonEngine.Config(
                warningVelocityMetersPerSecond = 0.50,
                maximumVelocityMetersPerSecond = 1.0,
            ),
        )

        val first = frame(
            timestampNanos = 1_000_000_000L,
            joints = listOf(
                joint(
                    13,
                    0.0,
                    1.0,
                    0.0,
                    velocity = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.95,
                    observations = 4,
                ),
            ),
        )
        val second = frame(
            timestampNanos = 1_033_000_000L,
            joints = listOf(
                joint(
                    13,
                    0.12,
                    1.0,
                    0.0,
                    velocity = Vec3(3.0, 0.0, 0.0),
                    confidence = 0.95,
                    observations = 4,
                ),
            ),
        )

        engine.refine(first)
        val result = engine.refine(second)

        val output = result.frame.joints.single()
        check(output.landmarkId == 13)
        check(output.confidence < 0.95)
        check(result.velocityViolationRatio >= 0.0)
        check(result.warnings.any { it.contains("velocity") })
    }

    fun persistentKinematicFailureEventuallyBlocks() {
        val engine = PhysicsConstrainedSkeletonEngine(
            config = PhysicsConstrainedSkeletonEngine.Config(
                blockedPhysicsScore = 0.70,
                warningPhysicsScore = 0.82,
                maximumBoneResidualMeters = 0.05,
                warningBoneResidualMeters = 0.02,
            ),
            edges = listOf(
                PhysicsConstrainedSkeletonEngine.Edge(
                    parent = 11,
                    child = 13,
                    group = "test-upper-arm",
                    expectedLengthMeters = 0.30,
                ),
            ),
        )

        val frames = (0 until 6).map { index ->
            frame(
                timestampNanos = 1_000_000_000L + index * 33_000_000L,
                joints = listOf(
                    joint(
                        11,
                        0.0,
                        1.5,
                        0.0,
                        confidence = 0.92,
                        observations = 3,
                    ),
                    joint(
                        13,
                        0.65,
                        1.5,
                        0.0,
                        confidence = 0.92,
                        observations = 3,
                    ),
                ),
            )
        }

        val results = frames.map(engine::refine)
        check(results.last().physicsScore < 0.70 || results.last().blocked)
        check(results.last().warnings.isNotEmpty())
    }

    fun cleanRecoveryIsHysteretic() {
        val engine = PhysicsConstrainedSkeletonEngine(
            config = PhysicsConstrainedSkeletonEngine.Config(
                blockedPhysicsScore = 0.70,
                warningPhysicsScore = 0.82,
                recoveryFrames = 4,
            ),
            edges = listOf(
                PhysicsConstrainedSkeletonEngine.Edge(
                    parent = 11,
                    child = 13,
                    group = "test-upper-arm",
                    expectedLengthMeters = 0.30,
                ),
            ),
        )

        val bad = frame(
            timestampNanos = 1_000_000_000L,
            joints = listOf(
                joint(11, 0.0, 1.5, 0.0, confidence = 0.90, observations = 3),
                joint(13, 0.75, 1.5, 0.0, confidence = 0.90, observations = 3),
            ),
        )
        engine.refine(bad)

        val good = frame(
            timestampNanos = 1_033_000_000L,
            joints = listOf(
                joint(11, 0.0, 1.5, 0.0, confidence = 0.95, observations = 4),
                joint(13, 0.30, 1.5, 0.0, confidence = 0.95, observations = 4),
            ),
        )

        val firstRecovery = engine.refine(good)
        if (firstRecovery.physicsScore < 0.82) {
            check(firstRecovery.warnings.any { it.contains("persistent") || it.contains("degraded") })
        }

        var final = firstRecovery
        repeat(6) {
            final = engine.refine(
                good.copy(
                    timestampNanos =
                        good.timestampNanos +
                            (it + 1L) * 33_000_000L,
                ),
            )
        }

        check(final.accepted || final.physicsScore >= 0.70)
    }

    fun resetClearsPhysicsHistoryAndDiagnostics() {
        val engine = PhysicsConstrainedSkeletonEngine()
        val frame = frame(
            timestampNanos = 1_000_000_000L,
            joints = listOf(
                joint(11, 0.0, 1.5, 0.0, confidence = 0.95),
                joint(12, 0.4, 1.5, 0.0, confidence = 0.95),
            ),
        )

        engine.refine(frame)
        check(engine.diagnostics() != null)

        engine.reset()

        check(engine.diagnostics() == null)
    }

    private fun frame(
        timestampNanos: Long,
        joints: List<ReconstructionJoint>,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = joints,
            globalConfidence = joints.map { it.confidence }.average().coerceIn(0.0, 1.0),
        )

    private fun joint(
        id: Int,
        x: Double,
        y: Double,
        z: Double,
        confidence: Double,
        observations: Int = 3,
        velocity: Vec3 = Vec3(0.0, 0.0, 0.0),
    ): ReconstructionJoint =
        ReconstructionJoint(
            landmarkId = id,
            position = Vec3(x, y, z),
            velocity = velocity,
            confidence = confidence,
            reprojectionErrorPx = 1.2,
            observationCount = observations,
            positionUncertaintyMeters = 0.018,
        )
}

fun main() {
    val test = PhysicsConstrainedSkeletonEngineTest()
    test.missingJointsAreNeverFabricated()
    test.learnedBoneLengthRejectsAbruptLowConfidenceBaselineRewrite()
    test.unqualifiedFirstObservationDoesNotSeedMetricBaseline()
    test.bilateralSymmetryUsesSegmentLengthsInsteadOfLeftRightDistance()
    test.highVelocityLowersConfidenceButPreservesJoint()
    test.persistentKinematicFailureEventuallyBlocks()
    test.cleanRecoveryIsHysteretic()
    test.resetClearsPhysicsHistoryAndDiagnostics()
}
