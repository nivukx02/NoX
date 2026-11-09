package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.Vec3

/**
 * Geometry-safe per-vertex normal estimation.
 *
 * Face normals are area weighted so large, stable triangles contribute more strongly than tiny
 * slivers. Degenerate or out-of-range triangles are ignored. Vertices without valid support return
 * the zero vector rather than a fabricated normal.
 */
object MeshNormalEstimator {

    fun estimate(
        vertices: List<MeshVertex>,
        triangles: List<MeshTriangle>,
    ): List<Vec3> {
        if (vertices.isEmpty()) return emptyList()

        val accumulated =
            MutableList(vertices.size) {
                Vec3(0.0, 0.0, 0.0)
            }

        triangles.forEach { triangle ->
            if (
                triangle.a !in vertices.indices ||
                triangle.b !in vertices.indices ||
                triangle.c !in vertices.indices
            ) {
                return@forEach
            }

            if (
                triangle.a == triangle.b ||
                triangle.b == triangle.c ||
                triangle.a == triangle.c
            ) {
                return@forEach
            }

            val a = vertices[triangle.a].position
            val b = vertices[triangle.b].position
            val c = vertices[triangle.c].position

            val ab = b - a
            val ac = c - a
            val normal = ab.cross(ac)

            if (
                !normal.isFinite() ||
                normal.norm() < 1e-10
            ) {
                return@forEach
            }

            accumulated[triangle.a] =
                accumulated[triangle.a] + normal
            accumulated[triangle.b] =
                accumulated[triangle.b] + normal
            accumulated[triangle.c] =
                accumulated[triangle.c] + normal
        }

        return accumulated.map { normal ->
            if (!normal.isFinite() || normal.norm() < 1e-10) {
                Vec3(0.0, 0.0, 0.0)
            } else {
                normal.normalized()
            }
        }
    }

    private fun Vec3.isFinite(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            z.isFinite()
}
