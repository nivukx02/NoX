package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertTrue

class SpatiotemporalBundleJointConsensusTest {

    @Test
    fun jointConsensusRejectsAClockFitThatOnlyImprovesOneCamera() {
        val engine =
            SpatiotemporalBundleAdjustmentEngine(
                SpatiotemporalBundleAdjustmentEngine.Config(
                    minimumObservations = 8,
                    maximumOffsetNanos = 20_000_000L,
                    searchStepNanos = 500_000L,
                    minimumJointConsensusConfidence = 0.70,
                    minimumJointImprovementRatio = 0.50,
                ),
            )

        val input = syntheticInput(
            offsetsNanos = mapOf(
                "cam-a" to 2_000_000.0,
                "cam-b" to -2_000_000.0,
            ),
        )

        val result =
            engine.optimize(
                cameras = input.cameras,
                trajectories = input.trajectories,
                observations = input.observations,
                referenceCameraId = "cam-a",
            )

        assertTrue(result.jointTemporalConsensusConfidence >= 0.70)
    }

    @Test
    fun jointConsensusReportsWhenCameraOffsetsAreNotCompatible() {
        val engine =
            SpatiotemporalBundleAdjustmentEngine(
                SpatiotemporalBundleAdjustmentEngine.Config(
                    minimumObservations = 8,
                    maximumOffsetNanos = 20_000_000L,
                    searchStepNanos = 1_000_000L,
                    minimumJointConsensusConfidence = 0.92,
                    maximumJointOffsetSpreadNanos = 2_000_000L,
                    minimumJointImprovementRatio = 0.50,
                ),
            )

        val input = syntheticInput(
            offsetsNanos = mapOf(
                "cam-a" to 10_000_000.0,
                "cam-b" to -10_000_000.0,
            ),
        )

        val result =
            engine.optimize(
                cameras = input.cameras,
                trajectories = input.trajectories,
                observations = input.observations,
                referenceCameraId = "cam-a",
            )

        assertTrue(
            result.warnings.contains(
                "joint-temporal-consensus-below-reference-threshold",
            ),
        )
    }

    private data class SyntheticInput(
        val cameras: List<SpatiotemporalBundleAdjustmentEngine.Camera>,
        val trajectories: List<SpatiotemporalBundleAdjustmentEngine.TrajectorySample>,
        val observations: List<SpatiotemporalBundleAdjustmentEngine.DynamicObservation>,
    )

    private fun syntheticInput(
        offsetsNanos: Map<String, Double>,
    ): SyntheticInput {
        val intrinsics =
            com.nivukx.nox.core.model.CameraIntrinsics(
                fx = 900.0,
                fy = 900.0,
                cx = 500.0,
                cy = 500.0,
            )
        val reference =
            SpatiotemporalBundleAdjustmentEngine.Camera(
                id = "ref",
                pose =
                    com.nivukx.nox.core.model.Pose(
                        position =
                            com.nivukx.nox.core.model.Vec3(0.0, 0.0, 0.0),
                        rotation =
                            com.nivukx.nox.core.model.Quat.IDENTITY,
                    ),
                intrinsics = intrinsics,
                fixed = true,
            )
        val cameras =
            offsetsNanos.keys.mapIndexed { index, id ->
                SpatiotemporalBundleAdjustmentEngine.Camera(
                    id = id,
                    pose =
                        com.nivukx.nox.core.model.Pose(
                            position =
                                com.nivukx.nox.core.model.Vec3(
                                    0.7 + index * 0.8,
                                    0.0,
                                    0.0,
                                ),
                            rotation =
                                com.nivukx.nox.core.model.Quat.IDENTITY,
                        ),
                    intrinsics = intrinsics,
                )
            }

        val trajectories =
            (0 until 32).map { index ->
                val timestamp =
                    1_000_000_000L +
                        index * 20_000_000L
                val seconds =
                    index * 0.020
                SpatiotemporalBundleAdjustmentEngine.TrajectorySample(
                    pointId = 1,
                    timestampNanos = timestamp,
                    position =
                        com.nivukx.nox.core.model.Vec3(
                            0.05 * seconds,
                            0.15 * kotlin.math.sin(seconds * 2.0),
                            4.0 + 0.12 * seconds,
                        ),
                )
            }

        val observations =
            cameras.flatMap { camera ->
                val delay =
                    offsetsNanos[camera.id] ?: 0.0
                trajectories.dropLast(1).map { sample ->
                    val delayed =
                        interpolate(
                            trajectories,
                            (
                                sample.timestampNanos +
                                    delay.toLong()
                            ),
                        ) ?: sample

                    SpatiotemporalBundleAdjustmentEngine.DynamicObservation(
                        cameraId = camera.id,
                        pointId = sample.pointId,
                        timestampNanos = sample.timestampNanos,
                        pixel =
                            project(
                                delayed.position,
                                camera,
                            ),
                        confidence = 1.0,
                    )
                }
            }

        return SyntheticInput(
            cameras = listOf(reference) + cameras,
            trajectories = trajectories,
            observations = observations,
        )
    }

    private fun interpolate(
        samples: List<SpatiotemporalBundleAdjustmentEngine.TrajectorySample>,
        timestampNanos: Long,
    ): SpatiotemporalBundleAdjustmentEngine.TrajectorySample? {
        if (
            timestampNanos <
                samples.first().timestampNanos ||
            timestampNanos >
                samples.last().timestampNanos
        ) {
            return null
        }

        for (index in 0 until samples.lastIndex) {
            val left = samples[index]
            val right = samples[index + 1]
            if (
                timestampNanos >= left.timestampNanos &&
                timestampNanos <= right.timestampNanos
            ) {
                val alpha =
                    (
                        timestampNanos -
                            left.timestampNanos
                    ).toDouble() /
                        (
                            right.timestampNanos -
                                left.timestampNanos
                        ).toDouble()
                return left.copy(
                    timestampNanos = timestampNanos,
                    position =
                        left.position.lerp(
                            right.position,
                            alpha,
                        ),
                )
            }
        }
        return null
    }

    private fun project(
        point: com.nivukx.nox.core.model.Vec3,
        camera: SpatiotemporalBundleAdjustmentEngine.Camera,
    ): com.nivukx.nox.core.model.Vec2 =
        com.nivukx.nox.core.model.Vec2(
            camera.intrinsics.fx * point.x /
                (point.z - camera.pose.position.z) +
                camera.intrinsics.cx,
            camera.intrinsics.fy * point.y /
                (point.z - camera.pose.position.z) +
                camera.intrinsics.cy,
        )
}
