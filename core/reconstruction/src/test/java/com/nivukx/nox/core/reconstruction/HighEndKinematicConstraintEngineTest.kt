package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

class HighEndKinematicConstraintEngineTest {

    @Test
    fun explicitBoneConstraintReducesGeometricResidual() {
        val engine = HighEndKinematicConstraintEngine()

        val frame = frame(
            timestampNanos = 1_000_000_000L,
            root = Vec3(0.0, 0.0, 2.0),
            child = Vec3(0.0, 1.35, 2.0),
        )

        val result = engine.solve(
            HighEndKinematicConstraintEngine.Input(
                skeleton = frame,
                boneConstraints = listOf(
                    HighEndKinematicConstraintEngine.BoneConstraint(
                        a = 0,
                        b = 1,
                        preferredLengthMeters = 1.0,
                        minimumLengthMeters = 0.95,
                        maximumLengthMeters = 1.05,
                        stiffness = 1.0,
                        semantic = "root-head",
                    ),
                ),
                rootJointId = 0,
            ),
        )

        val outputLength = distance(
            result.skeleton.joints[0].position,
            result.skeleton.joints[1].position,
        )

        assertTrue(outputLength < 1.35)
        assertTrue(result.diagnostics.meanLengthResidualMeters > 0.0)
        assertTrue(result.diagnostics.geometricScore > 0.0)
    }

    @Test
    fun previousFrameAutomaticallySuppliesHumanBoneReference() {
        val engine = HighEndKinematicConstraintEngine()

        engine.solve(
            HighEndKinematicConstraintEngine.Input(
                skeleton = frame(
                    timestampNanos = 2_000_000_000L,
                    root = Vec3(0.0, 0.0, 2.0),
                    child = Vec3(0.0, 1.0, 2.0),
                ),
            ),
        )

        val result = engine.solve(
            HighEndKinematicConstraintEngine.Input(
                skeleton = frame(
                    timestampNanos = 2_016_666_667L,
                    root = Vec3(0.0, 0.0, 2.0),
                    child = Vec3(0.0, 1.45, 2.0),
                ),
            ),
        )

        val outputLength = distance(
            result.skeleton.joints[0].position,
            result.skeleton.joints[1].position,
        )

        assertTrue(outputLength < 1.45)
        assertTrue(
            result.diagnostics.activeConstraintCount >= 1,
        )
    }

    @Test
    fun extremeVelocityProducesQualityWarningWithoutNaNOutput() {
        val engine = HighEndKinematicConstraintEngine()

        engine.solve(
            HighEndKinematicConstraintEngine.Input(
                skeleton = frame(
                    timestampNanos = 3_000_000_000L,
                    root = Vec3(0.0, 0.0, 2.0),
                    child = Vec3(0.0, 1.0, 2.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                ),
            ),
        )

        val result = engine.solve(
            HighEndKinematicConstraintEngine.Input(
                skeleton = frame(
                    timestampNanos = 3_001_000_000L,
                    root = Vec3(3.0, 0.0, 2.0),
                    child = Vec3(3.0, 1.0, 2.0),
                    velocity = Vec3(3_000.0, 0.0, 0.0),
                    confidence = 0.42,
                ),
                cameraMotionRisk = 0.25,
                synchronizationConfidence = 0.92,
            ),
        )

        assertTrue(result.diagnostics.velocityViolations >= 1)
        assertTrue(
            result.diagnostics.warnings.any {
                it == "velocity-limit"
            },
        )
        assertTrue(
            result.skeleton.joints.all {
                it.position.x.isFinite() &&
                    it.position.y.isFinite() &&
                    it.position.z.isFinite()
            },
        )
    }

    @Test
    fun fallbackRootPrefersPelvisInsteadOfFoot() {
        val engine = HighEndKinematicConstraintEngine(
            HighEndKinematicConstraintEngine.Config(
                rootMotionLock = 1.0,
            ),
        )

        val frame = SkeletonFrame(
            timestampNanos = 5_000_000_000L,
            subjectType = SubjectType.HUMAN,
            trackId = 22L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 31,
                    position = Vec3(0.0, -1.0, 0.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.95,
                    reprojectionErrorPx = 1.0,
                    observationCount = 3,
                    positionUncertaintyMeters = 0.02,
                ),
                ReconstructionJoint(
                    landmarkId = 23,
                    position = Vec3(0.0, 0.0, 0.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.95,
                    reprojectionErrorPx = 1.0,
                    observationCount = 3,
                    positionUncertaintyMeters = 0.02,
                ),
            ),
            globalConfidence = 0.95,
        )

        val result = engine.solve(
            HighEndKinematicConstraintEngine.Input(
                skeleton = frame,
            ),
        )

        assertTrue(result.diagnostics.rootCorrectionCount == 0)
        assertTrue(
            result.skeleton.joints.first {
                it.landmarkId == 23
            }.position.y.isFinite(),
        )
    }

    @Test
    fun footContactCanLockAStationaryFootToTheCaptureFloor() {
        val engine = HighEndKinematicConstraintEngine(
            HighEndKinematicConstraintEngine.Config(
                footContactStrictness = 1.0,
            ),
        )

        val result = engine.solve(
            HighEndKinematicConstraintEngine.Input(
                skeleton = frame(
                    timestampNanos = 4_000_000_000L,
                    root = Vec3(0.0, 0.0, 2.0),
                    child = Vec3(0.0, 0.40, 2.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                ),
                footContacts = listOf(
                    HighEndKinematicConstraintEngine.FootContact(
                        jointId = 1,
                        floorHeightMeters = 0.0,
                        lockStrength = 1.0,
                        allowedVerticalDriftMeters = 0.01,
                        velocityThresholdMetersPerSecond = 0.01,
                    ),
                ),
                captureFloorHeightMeters = 0.0,
                floorLocked = true,
            ),
        )

        assertTrue(
            result.skeleton.joints[1].position.y <
                0.40,
        )
        assertTrue(
            result.diagnostics.footCorrectionCount >= 1,
        )
    }

    private fun frame(
        timestampNanos: Long,
        root: Vec3,
        child: Vec3,
        velocity: Vec3 = Vec3(0.0, 0.0, 0.12),
        confidence: Double = 0.94,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = SubjectType.HUMAN,
            trackId = 99L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 0,
                    position = root,
                    velocity = velocity,
                    confidence = confidence,
                    reprojectionErrorPx = 1.0,
                    observationCount = 4,
                    positionUncertaintyMeters = 0.015,
                ),
                ReconstructionJoint(
                    landmarkId = 1,
                    position = child,
                    velocity = velocity,
                    confidence = confidence,
                    reprojectionErrorPx = 1.0,
                    observationCount = 4,
                    positionUncertaintyMeters = 0.015,
                ),
            ),
            globalConfidence = confidence,
        )

    private fun distance(a: Vec3, b: Vec3): Double =
        sqrt(
            (a.x - b.x) * (a.x - b.x) +
                (a.y - b.y) * (a.y - b.y) +
                (a.z - b.z) * (a.z - b.z),
        )
}
