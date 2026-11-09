package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3

/**
 * Derives a conservative metric reconstruction volume from confident 3D skeleton observations.
 *
 * The margin is intentionally configurable because clothing, props and fast extremities can extend
 * beyond the skeleton. The estimator never shrinks below the observed geometry.
 */
class ReconstructionBoundsEstimator(
    private val horizontalMarginMeters: Double = 0.35,
    private val verticalMarginMeters: Double = 0.25,
    private val minimumExtentMeters: Double = 0.50,
) {
    data class Bounds(
        val min: Vec3,
        val max: Vec3,
        val center: Vec3,
        val extent: Vec3,
    ) {
        fun expand(margin: Double): Bounds =
            Bounds(
                min = Vec3(min.x - margin, min.y - margin, min.z - margin),
                max = Vec3(max.x + margin, max.y + margin, max.z + margin),
                center = center,
                extent = Vec3(
                    extent.x + margin * 2.0,
                    extent.y + margin * 2.0,
                    extent.z + margin * 2.0,
                ),
            )
    }

    fun fromSkeleton(frame: SkeletonFrame): Bounds? {
        val points = frame.joints
            .filter { it.confidence >= 0.20 }
            .map { it.position }

        if (points.size < 4) return null

        val minX = points.minOf { it.x }
        val minY = points.minOf { it.y }
        val minZ = points.minOf { it.z }
        val maxX = points.maxOf { it.x }
        val maxY = points.maxOf { it.y }
        val maxZ = points.maxOf { it.z }

        val min = Vec3(
            minX - horizontalMarginMeters,
            minY - verticalMarginMeters,
            minZ - horizontalMarginMeters,
        )
        val max = Vec3(
            maxX + horizontalMarginMeters,
            maxY + verticalMarginMeters,
            maxZ + horizontalMarginMeters,
        )

        val extent = Vec3(
            max.x - min.x,
            max.y - min.y,
            max.z - min.z,
        )

        val correctedMin = Vec3(
            if (extent.x < minimumExtentMeters) (min.x + max.x) * 0.5 - minimumExtentMeters * 0.5 else min.x,
            if (extent.y < minimumExtentMeters) (min.y + max.y) * 0.5 - minimumExtentMeters * 0.5 else min.y,
            if (extent.z < minimumExtentMeters) (min.z + max.z) * 0.5 - minimumExtentMeters * 0.5 else min.z,
        )
        val correctedMax = Vec3(
            if (extent.x < minimumExtentMeters) (min.x + max.x) * 0.5 + minimumExtentMeters * 0.5 else max.x,
            if (extent.y < minimumExtentMeters) (min.y + max.y) * 0.5 + minimumExtentMeters * 0.5 else max.y,
            if (extent.z < minimumExtentMeters) (min.z + max.z) * 0.5 + minimumExtentMeters * 0.5 else max.z,
        )

        val finalExtent = correctedMax - correctedMin

        return Bounds(
            min = correctedMin,
            max = correctedMax,
            center = (correctedMin + correctedMax) * 0.5,
            extent = finalExtent,
        )
    }
}
