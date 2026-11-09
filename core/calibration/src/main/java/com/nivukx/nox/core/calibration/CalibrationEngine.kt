package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CameraExtrinsics
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.nativeengine.calibration.NativeCalibrationConfig
import com.nivukx.nox.core.nativeengine.calibration.NativeCalibrationEngine
import kotlin.math.abs

class CalibrationEngine {
    data class RigCamera(
        val id: DeviceId,
        val role: DeviceRole,
        val intrinsics: CameraIntrinsics,
        val centerMeters: Vec3,
        val forward: Vec3,
        val up: Vec3 = Vec3(0.0, 1.0, 0.0),
        val calibrationFocusDistanceDiopters: Double? = null,
    )

    data class RigResult(
        val snapshot: CalibrationSnapshot,
        val frame: SpatialRigFrame,
        val quality: Double,
    )

    fun fromReference(
        reference: DeviceId,
        intrinsics: Map<DeviceId, CameraIntrinsics>,
        cameraCenters: Map<DeviceId, Vec3>,
        forwardVectors: Map<DeviceId, Vec3>,
        upVector: Vec3 = Vec3(0.0, 1.0, 0.0),
        nowNanos: Long = System.nanoTime(),
    ): CalibrationSnapshot {
        require(reference in cameraCenters) { "Reference camera center missing" }
        val refCenter = cameraCenters.getValue(reference)
        val extrinsics =
            cameraCenters
                .map { entry ->
                    val device = entry.key
                    val center = entry.value
                    val forward =
                        forwardVectors[device]?.normalized()
                            ?: Vec3(0.0, 0.0, 1.0)
                    device to CameraExtrinsics(
                        pose = Pose(
                            position = center - refCenter,
                            rotation =
                                RigidTransform.alignAxes(
                                    forward,
                                    upVector,
                                ),
                        ),
                        referenceDevice = reference,
                    )
                }
                .toMap()
        return CalibrationSnapshot(
            intrinsics = intrinsics,
            extrinsics = extrinsics,
            reprojectionErrorPx = 0.0,
            calibratedAtNanos = nowNanos,
        )
    }

    fun calibrateRig(
        reference: DeviceId,
        cameras: List<RigCamera>,
        nowNanos: Long = System.nanoTime(),
    ): RigResult {
        require(cameras.isNotEmpty()) { "No cameras supplied" }
        val ref = cameras.firstOrNull { it.id == reference } ?: error("Reference device not found")
        val up = ref.up.normalized()
        val north = ref.forward.normalized()
        val east = up.cross(north).normalized()
        val correctedNorth = east.cross(up).normalized()

        val frame = SpatialRigFrame(
            origin = ref.centerMeters,
            north = correctedNorth,
            east = east,
            up = up,
            deviceRoles = cameras.associate { it.id to it.role },
        )

        val intrinsics = cameras.associate { it.id to it.intrinsics }
        val extrinsics = cameras.associate { camera ->
            camera.id to CameraExtrinsics(
                pose = Pose(
                    position = camera.centerMeters - ref.centerMeters,
                    rotation = RigidTransform.alignAxes(camera.forward, camera.up),
                ),
                referenceDevice = reference,
            )
        }

        val directionalQuality = cameras.map { camera ->
            val f = camera.forward.normalized()
            (1.0 - abs(f.dot(up)).coerceIn(0.0, 1.0))
        }.average()

        val baselineQuality = cameras
            .map { it.centerMeters.distanceTo(ref.centerMeters) }
            .filter { it > 1e-3 }
            .map { (1.0 - abs(it - 1.5) / 1.5).coerceIn(0.0, 1.0) }
            .average()
            .takeIf { !it.isNaN() } ?: 1.0

        val calibrationFocusDistanceDiopters =
            cameras.mapNotNull { camera ->
                camera.calibrationFocusDistanceDiopters
                    ?.takeIf {
                        it.isFinite() && it >= 0.0
                    }
                    ?.let { camera.id to it }
            }.toMap()

        val snapshot = CalibrationSnapshot(
            intrinsics = intrinsics,
            extrinsics = extrinsics,
            reprojectionErrorPx = 0.0,
            calibratedAtNanos = nowNanos,
            calibrationFocusDistanceDiopters =
                calibrationFocusDistanceDiopters,
        )
        return RigResult(
            snapshot = snapshot,
            frame = frame,
            quality = (0.75 * directionalQuality + 0.25 * baselineQuality).coerceIn(0.0, 1.0),
        )
    }
    data class NativeCalibrationObservation(
        val cameraId: DeviceId,
        val worldPoint: Vec3,
        val pixel: com.nivukx.nox.core.model.Vec2,
        val weight: Double = 1.0,
    ) {
        init {
            require(weight.isFinite() && weight >= 0.0)
        }
    }

    /**
     * Runs the native robust pose + intrinsic/distortion refinement stage against an existing
     * metric calibration snapshot. The reference camera remains fixed and all supplied observations
     * stay associated with their source camera.
     */
    fun nativeRefine(
        snapshot: CalibrationSnapshot,
        observations: List<NativeCalibrationObservation>,
        reference: DeviceId,
        config: NativeCalibrationConfig = NativeCalibrationConfig(),
    ): NativeCalibrationEngine.Result? {
        if (snapshot.intrinsics.size < 2 ||
            observations.isEmpty() ||
            reference !in snapshot.extrinsics) {
            return null
        }

        val cameraIds =
            snapshot.intrinsics.keys
                .intersect(snapshot.extrinsics.keys)
                .sortedBy { it.value }

        if (cameraIds.size < 2 ||
            cameraIds.size > 128 ||
            cameraIds.none { it == reference }) {
            return null
        }

        val observationsByCamera =
            observations
                .filter { it.cameraId in cameraIds }
                .groupBy { it.cameraId }

        if (cameraIds.any {
                observationsByCamera[it].orEmpty().size <
                    config.minimumObservationsPerCamera
            }) {
            return null
        }

        val inputs =
            cameraIds.map { cameraId ->
                val intrinsics =
                    snapshot.intrinsics.getValue(cameraId)
                val extrinsics =
                    snapshot.extrinsics.getValue(cameraId)

                NativeCalibrationEngine.CameraInput(
                    cameraId = cameraId.value,
                    intrinsics = intrinsics,
                    pose = extrinsics.pose,
                    fixed = cameraId == reference,
                    observations =
                        observationsByCamera
                            .getValue(cameraId)
                            .map {
                                NativeCalibrationEngine.Observation(
                                    worldPoint =
                                        it.worldPoint,
                                    pixel =
                                        it.pixel,
                                    weight =
                                        it.weight.coerceIn(
                                            0.0,
                                            1.0,
                                        ),
                                )
                            },
                )
            }

        return NativeCalibrationEngine().solve(
            cameras = inputs,
            config = config,
        )
    }

    fun nativeSeed(snapshot: CalibrationSnapshot): NativeCalibrationSeed =
        NativeCalibrationInputAdapter.fromSnapshot(
            snapshot = snapshot,
            config = NativeCalibrationConfig(),
        )

}
