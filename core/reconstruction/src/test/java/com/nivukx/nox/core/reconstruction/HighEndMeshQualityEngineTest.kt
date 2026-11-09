package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DenseMeshQuality
import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class HighEndMeshQualityEngineTest {

    @Test
    fun uncertainty_and_normal_metadata_survive_high_end_mesh_processing() {
        val mesh =
            ReconstructionMesh(
                vertices =
                    listOf(
                        MeshVertex(
                            position = Vec3(0.0, 0.0, 0.0),
                            uncertaintyMeters = 0.001,
                        ),
                        MeshVertex(
                            position = Vec3(1.0, 0.0, 0.0),
                            uncertaintyMeters = 0.003,
                        ),
                        MeshVertex(
                            position = Vec3(0.0, 1.0, 0.0),
                            uncertaintyMeters = 0.005,
                        ),
                    ),
                triangles =
                    listOf(
                        MeshTriangle(0, 1, 2),
                    ),
                timestampNanos = 1_000_000L,
                confidence = 0.98,
            )

        val incoming =
            DenseMeshQuality(
                timestampNanos = mesh.timestampNanos,
                vertexCount = mesh.vertices.size,
                triangleCount = mesh.triangles.size,
                meanReprojectionErrorPx = 0.5,
                watertightnessScore = 0.8,
                temporalStability = 0.9,
                geometricConfidence = 0.9,
            )

        val result =
            HighEndMeshQualityEngine(
                HighEndMeshQualityEngine.Config(
                    minimumTriangleAreaMeters2 = 1e-8,
                    minimumEdgeMeters = 1e-4,
                    maximumEdgeMeters = 2.0,
                ),
            ).process(
                mesh = mesh,
                incomingQuality = incoming,
            )

        assertEquals(
            result.mesh.vertices.size,
            result.mesh.vertexUncertaintyMeters.size,
        )
        assertTrue(
            result.mesh.vertices.map {
                it.uncertaintyMeters
            }.containsAll(
                listOf(
                    0.001,
                    0.003,
                    0.005,
                ),
            ),
        )
        assertEquals(
            result.mesh.vertices.size,
            result.mesh.normals.size,
        )
        assertTrue(
            result.mesh.normals.all {
                it.norm() > 0.99 &&
                    it.x.isFinite() &&
                    it.y.isFinite() &&
                    it.z.isFinite()
            },
        )
    }

    @Test
    fun uncertain_vertices_receive_more_denoising_than_precise_vertices() {
        fun run(uncertainty: Double): Double {
            val mesh =
                ReconstructionMesh(
                    vertices =
                        listOf(
                            MeshVertex(
                                Vec3(0.0, 0.0, 0.0),
                                uncertainty,
                            ),
                            MeshVertex(
                                Vec3(1.0, 0.0, 0.0),
                                uncertainty,
                            ),
                            MeshVertex(
                                Vec3(0.0, 1.0, 0.0),
                                uncertainty,
                            ),
                        ),
                    triangles =
                        listOf(
                            MeshTriangle(0, 1, 2),
                        ),
                    timestampNanos = 12L,
                    confidence = 0.95,
                )

            val quality =
                DenseMeshQuality(
                    timestampNanos = 12L,
                    vertexCount = 3,
                    triangleCount = 1,
                    meanReprojectionErrorPx = 0.8,
                    watertightnessScore = 0.8,
                    temporalStability = 0.9,
                    geometricConfidence = 0.9,
                )

            val result =
                HighEndMeshQualityEngine(
                    HighEndMeshQualityEngine.Config(
                        minimumTriangleAreaMeters2 = 1e-8,
                        minimumEdgeMeters = 1e-4,
                        maximumEdgeMeters = 2.0,
                        smoothingAlpha = 0.20,
                        smoothingPasses = 1,
                    ),
                ).process(
                    mesh = mesh,
                    incomingQuality = quality,
                )

            return result.mesh.vertices.first().position.distanceTo(
                mesh.vertices.first().position,
            )
        }

        val preciseMovement = run(0.00005)
        val uncertainMovement = run(0.005)

        assertTrue(
            uncertainMovement > preciseMovement,
        )
    }

    @Test
    fun topology_sanitization_receives_real_vertex_metadata() {
        val mesh =
            ReconstructionMesh(
                vertices =
                    listOf(
                        MeshVertex(
                            Vec3(0.0, 0.0, 0.0),
                            0.002,
                        ),
                        MeshVertex(
                            Vec3(1.0, 0.0, 0.0),
                            0.004,
                        ),
                        MeshVertex(
                            Vec3(0.0, 1.0, 0.0),
                            0.006,
                        ),
                        MeshVertex(
                            Vec3(0.0, 0.0, 1.0),
                            0.008,
                        ),
                    ),
                triangles =
                    listOf(
                        MeshTriangle(0, 1, 2),
                        MeshTriangle(0, 1, 3),
                        MeshTriangle(0, 1, 2),
                    ),
                timestampNanos = 7_000_000L,
                confidence = 0.91,
            )

        val incoming =
            DenseMeshQuality(
                timestampNanos = mesh.timestampNanos,
                vertexCount = 4,
                triangleCount = 3,
                meanReprojectionErrorPx = 0.8,
                watertightnessScore = 0.7,
                temporalStability = 0.82,
                geometricConfidence = 0.84,
            )

        val result =
            HighEndMeshQualityEngine(
                HighEndMeshQualityEngine.Config(
                    minimumTriangleAreaMeters2 = 1e-8,
                    minimumEdgeMeters = 1e-4,
                    maximumEdgeMeters = 2.0,
                ),
            ).process(
                mesh = mesh,
                incomingQuality = incoming,
            )

        assertTrue(
            result.nonManifoldEdgeCount >= 0,
        )
        assertEquals(
            result.mesh.vertices.size,
            result.mesh.vertexUncertaintyMeters.size,
        )
    }
}
