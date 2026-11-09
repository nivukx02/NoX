package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.SilhouetteMask
import com.nivukx.nox.core.model.Vec3
import kotlin.math.ceil

/**
 * Coarse-to-fine visual-hull reconstruction.
 *
 * Instead of evaluating the entire high-resolution voxel volume uniformly, a coarse occupancy pass
 * finds the subject envelope, then occupied cells are recursively refined. The output is a dense
 * occupancy volume suitable for meshing while keeping runtime proportional to the occupied region.
 */
class AdaptiveVisualHullReconstructor(
    private val coarseResolution: Int = 32,
    private val refinementLevels: Int = 2,
    private val alphaThreshold: Int = 96,
    private val minimumViewSupport: Double = 0.80,
    private val sampleProjectedCorners: Boolean = true,
) {
    data class CameraView(
        val mask: SilhouetteMask,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
    )

    data class Cell(
        val min: Vec3,
        val max: Vec3,
        val level: Int,
    ) {
        val center: Vec3 get() = (min + max) * 0.5

        fun children(): List<Cell> {
            val center = center
            return listOf(
                Cell(Vec3(min.x, min.y, min.z), Vec3(center.x, center.y, center.z), level + 1),
                Cell(Vec3(center.x, min.y, min.z), Vec3(max.x, center.y, center.z), level + 1),
                Cell(Vec3(min.x, center.y, min.z), Vec3(center.x, max.y, center.z), level + 1),
                Cell(Vec3(center.x, center.y, min.z), Vec3(max.x, max.y, center.z), level + 1),
                Cell(Vec3(min.x, min.y, center.z), Vec3(center.x, center.y, max.z), level + 1),
                Cell(Vec3(center.x, min.y, center.z), Vec3(max.x, center.y, max.z), level + 1),
                Cell(Vec3(min.x, center.y, center.z), Vec3(center.x, max.y, max.z), level + 1),
                Cell(center, Vec3(max.x, max.y, max.z), level + 1),
            )
        }
    }

    data class Reconstruction(
        val occupiedCells: List<Cell>,
        val timestampNanos: Long,
        val confidence: Double,
    )

    fun reconstruct(
        views: List<CameraView>,
        minBounds: Vec3,
        maxBounds: Vec3,
        timestampNanos: Long,
    ): Reconstruction {
        require(views.size >= 2) { "Adaptive visual hull requires at least two cameras" }
        require(
            minBounds.x < maxBounds.x &&
                minBounds.y < maxBounds.y &&
                minBounds.z < maxBounds.z,
        ) {
            "Visual-hull bounds must form a non-empty volume"
        }

        var cells = seedGrid(minBounds, maxBounds)
        var accepted = mutableListOf<Cell>()

        repeat(refinementLevels + 1) { level ->
            val inside = cells.filter { cell ->
                occupancyScore(cell, views) >= minimumViewSupport
            }

            if (level == refinementLevels) {
                accepted = inside.toMutableList()
            } else {
                cells = inside.flatMap { it.children() }
            }
        }

        val confidence = if (accepted.isEmpty()) 0.0
        else accepted.map { occupancyScore(it, views) }.average()

        return Reconstruction(
            occupiedCells = accepted,
            timestampNanos = timestampNanos,
            confidence = confidence,
        )
    }

    private fun seedGrid(minBounds: Vec3, maxBounds: Vec3): List<Cell> {
        val extent = maxBounds - minBounds
        val maximumExtent = maxOf(extent.x, extent.y, extent.z)
        require(maximumExtent > 1e-9) { "Visual-hull extent is degenerate" }
        val step = maximumExtent / coarseResolution.coerceAtLeast(4)

        val nx = ceil(extent.x / step).toInt().coerceAtLeast(1)
        val ny = ceil(extent.y / step).toInt().coerceAtLeast(1)
        val nz = ceil(extent.z / step).toInt().coerceAtLeast(1)

        val cells = ArrayList<Cell>(nx * ny * nz)
        for (z in 0 until nz) {
            for (y in 0 until ny) {
                for (x in 0 until nx) {
                    val min = Vec3(
                        minBounds.x + x * step,
                        minBounds.y + y * step,
                        minBounds.z + z * step,
                    )
                    val max = Vec3(
                        min.x + step,
                        min.y + step,
                        min.z + step,
                    )
                    cells += Cell(min, max, 0)
                }
            }
        }
        return cells
    }

    private fun occupancyScore(
        cell: Cell,
        views: List<CameraView>,
    ): Double {
        val corners = cellCorners(cell)
        var supportingViews = 0.0

        views.forEach { view ->
            val projectedCorners = corners.mapNotNull { project(it, view) }
            if (projectedCorners.isEmpty()) return@forEach

            val minX = kotlin.math.floor(projectedCorners.minOf { it.first }).toInt()
            val maxX = kotlin.math.ceil(projectedCorners.maxOf { it.first }).toInt()
            val minY = kotlin.math.floor(projectedCorners.minOf { it.second }).toInt()
            val maxY = kotlin.math.ceil(projectedCorners.maxOf { it.second }).toInt()

            val samples = if (sampleProjectedCorners) {
                listOf(
                    (minX + maxX) / 2 to (minY + maxY) / 2,
                    minX to minY,
                    maxX to minY,
                    minX to maxY,
                    maxX to maxY,
                )
            } else {
                listOf(
                    (minX + maxX) / 2 to (minY + maxY) / 2,
                )
            }

            val occupancy = samples.count { (x, y) ->
                view.mask.isInside(
                    x.coerceIn(0, view.mask.width - 1),
                    y.coerceIn(0, view.mask.height - 1),
                    alphaThreshold,
                )
            }.toDouble() / samples.size.coerceAtLeast(1)

            if (occupancy >= 0.60) supportingViews += occupancy
        }

        return supportingViews.toDouble() / views.size.coerceAtLeast(1)
    }

    private fun project(
        point: Vec3,
        view: CameraView,
    ): Pair<Double, Double>? {
        val cameraPoint = view.pose.rotation
            .conjugate()
            .normalized()
            .rotate(point - view.pose.position)

        if (cameraPoint.z <= 1e-6) return null

        var x = cameraPoint.x / cameraPoint.z
        var y = cameraPoint.y / cameraPoint.z

        val k = view.intrinsics
        val r2 = x * x + y * y
        val radial = 1.0 + k.k1 * r2 + k.k2 * r2 * r2 + k.k3 * r2 * r2 * r2
        val dx = 2.0 * k.p1 * x * y + k.p2 * (r2 + 2.0 * x * x)
        val dy = k.p1 * (r2 + 2.0 * y * y) + 2.0 * k.p2 * x * y

        x = x * radial + dx
        y = y * radial + dy

        return x * k.fx + k.cx to y * k.fy + k.cy
    }

    private fun cellCorners(cell: Cell): List<Vec3> {
        val x0 = cell.min.x
        val y0 = cell.min.y
        val z0 = cell.min.z
        val x1 = cell.max.x
        val y1 = cell.max.y
        val z1 = cell.max.z

        return listOf(
            Vec3(x0, y0, z0),
            Vec3(x1, y0, z0),
            Vec3(x0, y1, z0),
            Vec3(x1, y1, z0),
            Vec3(x0, y0, z1),
            Vec3(x1, y0, z1),
            Vec3(x0, y1, z1),
            Vec3(x1, y1, z1),
        )
    }
}
