package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.SilhouetteMask
import com.nivukx.nox.core.model.Vec3
import kotlin.math.max

/**
 * Dense-ish geometry path for phones that provide a silhouette mask.
 *
 * It computes a visual hull: a 3D point survives only when every participating camera projects it
 * inside its silhouette. This is geometry, not a fake monocular depth estimate. The resulting
 * occupancy can be converted to an explicit surface mesh by VoxelSurfaceMesher.
 */
class VisualHullReconstructor(
    private val resolution: Int = 80,
    private val alphaThreshold: Int = 96,
) {
    data class CameraView(
        val mask: SilhouetteMask,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
    )

    data class VoxelVolume(
        val origin: Vec3,
        val voxelSize: Double,
        val resolutionX: Int,
        val resolutionY: Int,
        val resolutionZ: Int,
        val occupied: BooleanArray,
        val timestampNanos: Long,
    ) {
        fun index(x: Int, y: Int, z: Int): Int = (z * resolutionY + y) * resolutionX + x
        fun contains(x: Int, y: Int, z: Int): Boolean =
            x in 0 until resolutionX && y in 0 until resolutionY && z in 0 until resolutionZ
    }

    fun reconstruct(
        views: List<CameraView>,
        minBounds: Vec3,
        maxBounds: Vec3,
        timestampNanos: Long,
    ): VoxelVolume {
        require(views.size >= 2) { "Visual hull requires at least two camera views" }
        require(maxBounds.x > minBounds.x && maxBounds.y > minBounds.y && maxBounds.z > minBounds.z)

        val extentX = maxBounds.x - minBounds.x
        val extentY = maxBounds.y - minBounds.y
        val extentZ = maxBounds.z - minBounds.z
        val maxExtent = maxOf(extentX, extentY, extentZ)
        val voxelSize = maxExtent / resolution.coerceAtLeast(8)
        val nx = max(1, kotlin.math.ceil(extentX / voxelSize).toInt() + 1)
        val ny = max(1, kotlin.math.ceil(extentY / voxelSize).toInt() + 1)
        val nz = max(1, kotlin.math.ceil(extentZ / voxelSize).toInt() + 1)
        val occupied = BooleanArray(nx * ny * nz)

        for (z in 0 until nz) {
            for (y in 0 until ny) {
                for (x in 0 until nx) {
                    val point = Vec3(
                        minBounds.x + (x + 0.5) * voxelSize,
                        minBounds.y + (y + 0.5) * voxelSize,
                        minBounds.z + (z + 0.5) * voxelSize,
                    )
                    if (views.all { insideSilhouette(point, it) }) {
                        occupied[(z * ny + y) * nx + x] = true
                    }
                }
            }
        }

        return VoxelVolume(
            origin = minBounds,
            voxelSize = voxelSize,
            resolutionX = nx,
            resolutionY = ny,
            resolutionZ = nz,
            occupied = occupied,
            timestampNanos = timestampNanos,
        )
    }

    private fun insideSilhouette(point: Vec3, view: CameraView): Boolean {
        val inverse = view.pose.rotation.conjugate().normalized()
        val cameraPoint = inverse.rotate(point - view.pose.position)
        if (cameraPoint.z <= 1e-6) return false

        val x = cameraPoint.x / cameraPoint.z
        val y = cameraPoint.y / cameraPoint.z
        val r2 = x * x + y * y
        val radial = 1.0 + view.intrinsics.k1 * r2 +
            view.intrinsics.k2 * r2 * r2 +
            view.intrinsics.k3 * r2 * r2 * r2
        val dx = 2.0 * view.intrinsics.p1 * x * y +
            view.intrinsics.p2 * (r2 + 2.0 * x * x)
        val dy = view.intrinsics.p1 * (r2 + 2.0 * y * y) +
            2.0 * view.intrinsics.p2 * x * y

        val u = (x * radial + dx) * view.intrinsics.fx + view.intrinsics.cx
        val v = (y * radial + dy) * view.intrinsics.fy + view.intrinsics.cy
        val px = kotlin.math.floor(u).toInt()
        val py = kotlin.math.floor(v).toInt()
        return view.mask.isInside(px, py, alphaThreshold)
    }
}
