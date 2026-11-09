package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Test

class SkeletonReconstructionEngineTest {

    fun rollingShutterTrustIsBoundedAndConfigurable() {
        val engine =
            SkeletonReconstructionEngine()

        engine.configureObservability(
            SkeletonReconstructionEngine.ObservabilityTuning(
                rollingShutterCompensationTrust = 0.42,
            ),
        )

        check(
            engine.observabilityTuning()
                .rollingShutterCompensationTrust ==
                0.42,
        )
    }

    fun directionalUncertaintyIsCarriedIntoJointMetricUncertainty() {
        val intrinsics =
            CameraIntrinsics(
                fx = 1000.0,
                fy = 1000.0,
                cx = 500.0,
                cy = 500.0,
            )
        val point =
            Vec3(
                x = 0.18,
                y = 0.12,
                z = 4.0,
            )

        fun pixel(
            cameraX: Double,
        ): Vec2 =
            Vec2(
                x =
                    intrinsics.fx *
                        (point.x - cameraX) /
                        point.z +
                        intrinsics.cx,
                y =
                    intrinsics.fy *
                        point.y /
                        point.z +
                        intrinsics.cy,
            )

        val cameraA =
            DeviceId("camera-a")
        val cameraB =
            DeviceId("camera-b")

        val pA = pixel(0.0)
        val pB = pixel(1.1)

        val result =
            SkeletonReconstructionEngine()
                .reconstructDetailed(
                    timestampNanos = 1_000_000_000L,
                    subjectType = SubjectType.HUMAN,
                    trackId = 7L,
                    frames = listOf(
                        SkeletonReconstructionEngine.CameraFrame(
                            deviceId = cameraA,
                            timestampNanos = 1_000_000_000L,
                            observations = listOf(
                                CameraObservation(
                                    deviceId = cameraA,
                                    point = pA,
                                    confidence = 0.96,
                                    intrinsics = intrinsics,
                                    pose =
                                        Pose(
                                            position =
                                                Vec3(
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                ),
                                            rotation = Quat.IDENTITY,
                                        ),
                                    landmarkId = 11,
                                ),
                            ),
                        ),
                        SkeletonReconstructionEngine.CameraFrame(
                            deviceId = cameraB,
                            timestampNanos = 1_000_000_000L,
                            observations = listOf(
                                CameraObservation(
                                    deviceId = cameraB,
                                    point = pB,
                                    confidence = 0.95,
                                    intrinsics = intrinsics,
                                    pose =
                                        Pose(
                                            position =
                                                Vec3(
                                                    1.1,
                                                    0.0,
                                                    0.0,
                                                ),
                                            rotation = Quat.IDENTITY,
                                        ),
                                    landmarkId = 11,
                                ),
                            ),
                        ),
                    ),
                )

        check(result.diagnostics.size == 1)
        val diagnostic =
            result.diagnostics.single()

        check(
            diagnostic.directionalUncertaintyMeters.isFinite(),
        )
        check(
            diagnostic.directionalDepthUncertaintyMeters.isFinite(),
        )
        check(
            diagnostic.directionalUncertaintyConfidence in 0.0..1.0,
        )
        check(
            result.frame.joints
                .single()
                .positionUncertaintyMeters >=
                diagnostic.directionalUncertaintyMeters,
        )
    }

    fun directionalUncertaintyCanHoldAWeakDepthSolution() {
        val intrinsics =
            CameraIntrinsics(
                fx = 1000.0,
                fy = 1000.0,
                cx = 500.0,
                cy = 500.0,
            )
        val point =
            Vec3(
                x = 0.18,
                y = 0.12,
                z = 8.0,
            )

        fun pixel(
            cameraX: Double,
        ): Vec2 =
            Vec2(
                x =
                    intrinsics.fx *
                        (point.x - cameraX) /
                        point.z +
                        intrinsics.cx,
                y =
                    intrinsics.fy *
                        point.y /
                        point.z +
                        intrinsics.cy,
            )

        val engine =
            SkeletonReconstructionEngine()

        engine.configureObservability(
            SkeletonReconstructionEngine.ObservabilityTuning(
                minimumParallaxDegrees = 0.01,
                maximumDirectionalUncertaintyMeters = 0.05,
                maximumDepthUncertaintyMeters = 0.05,
                maximumUncertaintyAnisotropyRatio = 1.5,
            ),
        )

        val a =
            DeviceId("weak-a")
        val b =
            DeviceId("weak-b")
        val pa = pixel(0.0)
        val pb = pixel(0.10)

        val result =
            engine.reconstructDetailed(
                timestampNanos = 2_000_000_000L,
                subjectType = SubjectType.HUMAN,
                trackId = 9L,
                frames = listOf(
                    SkeletonReconstructionEngine.CameraFrame(
                        deviceId = a,
                        timestampNanos = 2_000_000_000L,
                        observations = listOf(
                            CameraObservation(
                                deviceId = a,
                                point = pa,
                                confidence = 0.92,
                                intrinsics = intrinsics,
                                pose =
                                    Pose(
                                        position =
                                            Vec3(
                                                0.0,
                                                0.0,
                                                0.0,
                                            ),
                                        rotation = Quat.IDENTITY,
                                    ),
                                landmarkId = 11,
                            ),
                        ),
                    ),
                    SkeletonReconstructionEngine.CameraFrame(
                        deviceId = b,
                        timestampNanos = 2_000_000_000L,
                        observations = listOf(
                            CameraObservation(
                                deviceId = b,
                                point = pb,
                                confidence = 0.92,
                                intrinsics = intrinsics,
                                pose =
                                    Pose(
                                        position =
                                            Vec3(
                                                0.10,
                                                0.0,
                                                0.0,
                                            ),
                                        rotation = Quat.IDENTITY,
                                    ),
                                landmarkId = 11,
                            ),
                        ),
                    ),
                ),
            )

        check(result.diagnostics.size == 1)
        val diagnostic =
            result.diagnostics.single()

        check(
            !diagnostic.accepted ||
                diagnostic.directionalUncertaintyAnisotropy >= 1.0,
        )
        check(
            diagnostic.directionalDepthUncertaintyMeters.isFinite(),
        )
    }
    @Test
    fun triangulationDiagnosticsExposeIndependentViewDiversity() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.20, 0.10, 4.0)

        fun pixel(cameraX: Double): Vec2 =
            Vec2(
                x = intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx,
                y = intrinsics.fy * point.y / point.z +
                    intrinsics.cy,
            )

        val devices = listOf(
            DeviceId("div-a"),
            DeviceId("div-b"),
            DeviceId("div-c"),
        )

        val positions = listOf(0.0, 1.0, 2.0)

        val frames = devices.zip(positions).map { (device, x) ->
            SkeletonReconstructionEngine.CameraFrame(
                deviceId = device,
                timestampNanos = 3_000_000_000L,
                observations = listOf(
                    CameraObservation(
                        deviceId = device,
                        point = pixel(x),
                        confidence = 0.95,
                        intrinsics = intrinsics,
                        pose = Pose(
                            position = Vec3(x, 0.0, 0.0),
                            rotation = Quat.IDENTITY,
                        ),
                        landmarkId = 21,
                    ),
                ),
            )
        }

        val result =
            SkeletonReconstructionEngine().reconstructDetailed(
                timestampNanos = 3_000_000_000L,
                subjectType = SubjectType.HUMAN,
                trackId = 21L,
                frames = frames,
            )

        val diagnostic = result.diagnostics.single()
        check(diagnostic.triangulationIndependentViewCount >= 3)
        check(diagnostic.triangulationViewDiversityScore > 0.50)
        check(
            diagnostic.triangulationDominantViewWeightRatio <= 0.82,
        )
        check(
            diagnostic.triangulationMetricEvidenceAuthority >= 0.60,
        )
        check(
            diagnostic.triangulationEpipolarConsistencyScore >= 0.90,
        )
        check(
            result.meanMetricEvidenceAuthority >= 0.60,
        )
        check(
            result.meanSymmetricEpipolarConsistencyScore >= 0.90,
        )
    }

}

fun main() {
    val test =
        SkeletonReconstructionEngineTest()

    test.rollingShutterTrustIsBoundedAndConfigurable()
    test.directionalUncertaintyIsCarriedIntoJointMetricUncertainty()
    test.directionalUncertaintyCanHoldAWeakDepthSolution()
}
