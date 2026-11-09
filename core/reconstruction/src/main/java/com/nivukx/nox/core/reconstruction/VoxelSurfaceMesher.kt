package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3

/**
 * Converts a visual-hull voxel volume into a watertight block surface.
 *
 * This intentionally uses exposed voxel faces instead of claiming to implement marching cubes.
 * It is deterministic, mobile-friendly, and provides a real mesh that later GPU/offline passes can
 * smooth or remesh.
 */
class VoxelSurfaceMesher {
    private data class Key(val x: Int, val y: Int, val z: Int)

    fun mesh(volume: VisualHullReconstructor.VoxelVolume): ReconstructionMesh {
        val vertices = mutableListOf<MeshVertex>()
        val triangles = mutableListOf<MeshTriangle>()
        val vertexMap = HashMap<Key, Int>()

        fun vertex(x: Int, y: Int, z: Int): Int {
            val key = Key(x, y, z)
            return vertexMap.getOrPut(key) {
                val p = volume.origin + Vec3(
                    x * volume.voxelSize,
                    y * volume.voxelSize,
                    z * volume.voxelSize,
                )
                vertices += MeshVertex(p)
                vertices.lastIndex
            }
        }

        fun face(points: IntArray) {
            val a = vertex(points[0], points[1], points[2])
            val b = vertex(points[3], points[4], points[5])
            val c = vertex(points[6], points[7], points[8])
            val d = vertex(points[9], points[10], points[11])
            triangles += MeshTriangle(a, b, c)
            triangles += MeshTriangle(a, c, d)
        }

        for (z in 0 until volume.resolutionZ) {
            for (y in 0 until volume.resolutionY) {
                for (x in 0 until volume.resolutionX) {
                    if (!volume.occupied[volume.index(x, y, z)]) continue
                    if (!volume.contains(x - 1, y, z) || !volume.occupied[volume.index(x - 1, y, z)]) {
                        face(intArrayOf(x, y, z, x, y + 1, z, x, y + 1, z + 1, x, y, z + 1))
                    }
                    if (!volume.contains(x + 1, y, z) || !volume.occupied[volume.index(x + 1, y, z)]) {
                        face(intArrayOf(x + 1, y, z + 1, x + 1, y + 1, z + 1, x + 1, y + 1, z, x + 1, y, z))
                    }
                    if (!volume.contains(x, y - 1, z) || !volume.occupied[volume.index(x, y - 1, z)]) {
                        face(intArrayOf(x, y, z + 1, x + 1, y, z + 1, x + 1, y, z, x, y, z))
                    }
                    if (!volume.contains(x, y + 1, z) || !volume.occupied[volume.index(x, y + 1, z)]) {
                        face(intArrayOf(x, y + 1, z, x + 1, y + 1, z, x + 1, y + 1, z + 1, x, y + 1, z + 1))
                    }
                    if (!volume.contains(x, y, z - 1) || !volume.occupied[volume.index(x, y, z - 1)]) {
                        face(intArrayOf(x, y, z, x + 1, y, z, x + 1, y + 1, z, x, y + 1, z))
                    }
                    if (!volume.contains(x, y, z + 1) || !volume.occupied[volume.index(x, y, z + 1)]) {
                        face(intArrayOf(x, y + 1, z + 1, x + 1, y + 1, z + 1, x + 1, y, z + 1, x, y, z + 1))
                    }
                }
            }
        }

        val normals =
            MeshNormalEstimator.estimate(
                vertices = vertices,
                triangles = triangles,
            )
        val normalCoverage =
            (
                normals.count { it.norm() > 0.5 }.toDouble() /
                    normals.size.coerceAtLeast(1)
                ).coerceIn(0.0, 1.0)

        return ReconstructionMesh(
            vertices = vertices,
            triangles = triangles,
            timestampNanos = volume.timestampNanos,
            confidence =
                if (vertices.isEmpty()) {
                    0.0
                } else {
                    0.72 + 0.28 * normalCoverage
                }.coerceIn(0.0, 1.0),
            normals = normals,
        )
    }
}
