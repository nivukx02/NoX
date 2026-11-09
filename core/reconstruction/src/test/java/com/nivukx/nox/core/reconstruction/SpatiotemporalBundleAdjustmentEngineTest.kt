package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3

class SpatiotemporalBundleAdjustmentEngineTest {

    private val intrinsics =
        CameraIntrinsics(
            fx = 900.0,
            fy = 900.0,
            cx = 500.0,
            cy = 500.0,
        )

    fun recovers_known_camera_time_offset() {
        val camera =
            SpatiotemporalBundleAdjustmentEngine.Camera(
                id = "cam-b",
                pose = Pose(
                    position = Vec3(0.8, 0.0, 0.0),
                    rotation = com.nivukx.nox.core.model.Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                fixed = false,
            )

        val trajectory =
            trajectory(
                pointId = 7,
                samples = 48,
            )

        val observations =
            observations(
                camera = camera,
                trajectory = trajectory,
                trueDelayNanos = 8_000_000L,
            )

        val result =
            SpatiotemporalBundleAdjustmentEngine(
                SpatiotemporalBundleAdjustmentEngine.Config(
                    maximumOffsetNanos = 20_000_000L,
                    searchStepNanos = 1_000_000L,
                    minimumObservations = 12,
                ),
            ).optimize(
                cameras = listOf(
                    camera.copy(
                        id = "ref",
                        pose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            com.nivukx.nox.core.model.Quat.IDENTITY,
                        ),
                        fixed = true,
                    ),
                    camera,
                ),
                trajectories = trajectory,
                observations = observations,
                referenceCameraId = "ref",
            )

        check(result.accepted)
        check(
            kotlin.math.abs(
                result.cameraOffsetsNanos["cam-b"]!! -
                    8_000_000.0,
            ) < 2_000_000.0,
        )
        check(result.finalRmsPx < result.initialRmsPx)
    }

    fun clamps_unobservable_camera_to_zero_confidence() {
        val camera =
            SpatiotemporalBundleAdjustmentEngine.Camera(
                id = "cam-b",
                pose = Pose(
                    Vec3(0.8, 0.0, 0.0),
                    com.nivukx.nox.core.model.Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
            )

        val result =
            SpatiotemporalBundleAdjustmentEngine().optimize(
                cameras = listOf(
                    camera.copy(
                        id = "ref",
                        fixed = true,
                    ),
                    camera,
                ),
                trajectories = emptyList(),
                observations = emptyList(),
                referenceCameraId = "ref",
            )

        check(!result.accepted)
        check(result.confidence == 0.0)
    }

    private fun trajectory(
        pointId: Int,
        samples: Int,
    ): List<SpatiotemporalBundleAdjustmentEngine.TrajectorySample> =
        (0 until samples).map { index ->
            val timestamp =
                1_000_000_000L +
                    index * 16_666_667L
            val seconds =
                index * 16.666667e-3
            SpatiotemporalBundleAdjustmentEngine.TrajectorySample(
                pointId = pointId,
                timestampNanos = timestamp,
                position = Vec3(
                    x = 0.04 * seconds,
                    y = 0.10 + 0.025 * kotlin.math.sin(seconds * 3.0),
                    z = 4.0 + 0.08 * seconds,
                ),
            )
        }

    private fun observations(
        camera: SpatiotemporalBundleAdjustmentEngine.Camera,
        trajectory: List<SpatiotemporalBundleAdjustmentEngine.TrajectorySample>,
        trueDelayNanos: Long,
    ): List<SpatiotemporalBundleAdjustmentEngine.DynamicObservation> =
        trajectory.dropLast(1).map { sample ->
            val delayed =
                interpolate(
                    trajectory,
                    sample.timestampNanos +
                        trueDelayNanos,
                )
                    ?: sample

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

    fun degraded_dense_evidence_is_downweighted() {
        val camera =
            SpatiotemporalBundleAdjustmentEngine.Camera(
                id = "cam-b",
                pose = Pose(
                    Vec3(0.8, 0.0, 0.0),
                    com.nivukx.nox.core.model.Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
            )

        val trajectory = trajectory(pointId = 9, samples = 48)
        val observations = observations(
            camera = camera,
            trajectory = trajectory,
            trueDelayNanos = 8_000_000L,
        ).mapIndexed { index, observation ->
            if (index % 5 == 0) {
                observation.copy(
                    uncertainty = 0.85,
                    visibility = 0.25,
                    temporalStability = 0.30,
                    crossViewAgreement = 0.20,
                )
            } else {
                observation
            }
        }

        val result =
            SpatiotemporalBundleAdjustmentEngine(
                SpatiotemporalBundleAdjustmentEngine.Config(
                    maximumOffsetNanos = 20_000_000L,
                    searchStepNanos = 1_000_000L,
                    minimumObservations = 12,
                ),
            ).optimize(
                cameras = listOf(
                    camera.copy(
                        id = "ref",
                        pose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            com.nivukx.nox.core.model.Quat.IDENTITY,
                        ),
                        fixed = true,
                    ),
                    camera,
                ),
                trajectories = trajectory,
                observations = observations,
                referenceCameraId = "ref",
            )

        check(result.accepted)
        check(
            result.warnings.contains(
                "dense-observation-quality-reduced",
            ),
        )
        check(result.finalRmsPx < result.initialRmsPx)
    }

    private fun interpolate(
        samples: List<SpatiotemporalBundleAdjustmentEngine.TrajectorySample>,
        timestampNanos: Long,
    ): SpatiotemporalBundleAdjustmentEngine.TrajectorySample? {
        val sorted = samples.sortedBy { it.timestampNanos }
        if (
            timestampNanos <
                sorted.first().timestampNanos ||
            timestampNanos >
                sorted.last().timestampNanos
        ) {
            return null
        }

        for (index in 0 until sorted.lastIndex) {
            val left = sorted[index]
            val right = sorted[index + 1]
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
                    position =
                        left.position.lerp(
                            right.position,
                            alpha,
                        ),
                    timestampNanos = timestampNanos,
                )
            }
        }

        return null
    }

    private fun project(
        point: Vec3,
        camera: SpatiotemporalBundleAdjustmentEngine.Camera,
    ): Vec2 {
        val local =
            camera.pose.rotation
                .conjugate()
                .normalized()
                .rotate(
                    point -
                        camera.pose.position,
                )

        return Vec2(
            camera.intrinsics.fx *
                local.x /
                local.z +
                camera.intrinsics.cx,
            camera.intrinsics.fy *
                local.y /
                local.z +
                camera.intrinsics.cy,
        )
    }
}

fun main() {
    val test = SpatiotemporalBundleAdjustmentEngineTest()
    test.recovers_known_camera_time_offset()
    test.clamps_unobservable_camera_to_zero_confidence()
    test.degraded_dense_evidence_is_downweighted()
}
