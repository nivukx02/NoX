package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class CanonicalSurfaceTemporalPriorEngineTest {

    @Test
    fun stableDeformationIsAcceptedWithHighCorrespondenceConfidence() {
        val engine = CanonicalSurfaceTemporalPriorEngine()

        val canonical =
            tetraMesh(
                offset = Vec3(0.0, 0.0, 0.0),
                scale = 1.0,
            )
        val deformed =
            tetraMesh(
                offset = Vec3(0.01, -0.005, 0.002),
                scale = 1.03,
            )

        val result =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 1_000_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = deformed,
                    upstreamConfidence = 0.94,
                ),
            )

        assertTrue(result.correspondenceCoverage > 0.90)
        assertTrue(result.normalAgreement > 0.80)
        assertTrue(result.edgeStretchScore > 0.70)
        assertTrue(result.accepted)
        assertTrue(result.confidence > 0.70)
    }

    @Test
    fun extremeStretchIsRejectedAsNonPhysicalSurfaceDeformation() {
        val engine = CanonicalSurfaceTemporalPriorEngine()

        val canonical =
            tetraMesh(
                offset = Vec3(0.0, 0.0, 0.0),
                scale = 1.0,
            )
        val stretched =
            tetraMesh(
                offset = Vec3(0.0, 0.0, 0.0),
                scale = 2.0,
            )

        val result =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 2_000_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = stretched,
                    upstreamConfidence = 0.92,
                ),
            )

        assertTrue(result.edgeStretchScore < 0.50)
        assertTrue(!result.accepted)
        assertTrue(
            result.warnings.any {
                it.contains("edge") ||
                    it.contains("deformation")
            },
        )
    }

    @Test
    fun suddenTemporalSurfaceJumpIsRejectedByDeformationHistory() {
        val engine = CanonicalSurfaceTemporalPriorEngine(
            CanonicalSurfaceTemporalPriorEngine.Config(
                temporalDisplacementWarningMeters = 0.05,
                temporalDisplacementRejectMeters = 0.12,
                minimumConfidenceForAcceptance = 0.50,
            ),
        )

        val canonical =
            tetraMesh(
                offset = Vec3(0.0, 0.0, 0.0),
                scale = 1.0,
            )

        val stable =
            tetraMesh(
                offset = Vec3(0.01, 0.0, 0.0),
                scale = 1.01,
            )

        val jumped =
            tetraMesh(
                offset = Vec3(0.16, 0.16, 0.16),
                scale = 1.01,
            )

        val first =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 10_000_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = stable,
                    upstreamConfidence = 0.95,
                ),
            )
        assertTrue(first.accepted)

        val second =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 10_033_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = jumped,
                    upstreamConfidence = 0.95,
                ),
            )

        assertTrue(
            second.p95TemporalDisplacementMeters >
                first.p95TemporalDisplacementMeters,
        )
        assertTrue(
            second.temporalDeformationConsistency <
                first.temporalDeformationConsistency,
        )
        assertTrue(!second.accepted)
        assertTrue(
            second.warnings.any {
                it.contains("temporal-deformation")
            },
        )
    }

    @Test
    fun acceptedSurfaceCarriesSdfFlowConsistencyTelemetry() {
        val engine = CanonicalSurfaceTemporalPriorEngine()

        val canonical =
            tetraMesh(
                offset = Vec3(0.0, 0.0, 0.0),
                scale = 1.0,
            )
        val firstMesh =
            tetraMesh(
                offset = Vec3(0.01, 0.0, 0.0),
                scale = 1.01,
            )
        val secondMesh =
            tetraMesh(
                offset = Vec3(0.03, 0.02, 0.0),
                scale = 1.02,
            )

        val first =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 7_000_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = firstMesh,
                    upstreamConfidence = 0.96,
                ),
            )

        val second =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 7_033_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = secondMesh,
                    upstreamConfidence = 0.96,
                ),
            )

        assertTrue(first.normalFlowConsistency.isFinite())
        assertTrue(second.normalFlowConsistency.isFinite())
        assertTrue(second.normalFlowConsistency >= 0.0)
        assertTrue(second.normalFlowConsistency <= 1.0)
        assertTrue(second.accepted || second.confidence > 0.35)
    }

    @Test
    fun largeCorrespondenceFailureDoesNotFabricateMissingSurface() {
        val engine = CanonicalSurfaceTemporalPriorEngine()

        val canonical =
            tetraMesh(
                offset = Vec3(0.0, 0.0, 0.0),
                scale = 1.0,
            )
        val farAway =
            tetraMesh(
                offset = Vec3(2.0, 2.0, 2.0),
                scale = 1.0,
            )

        val result =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 3_000_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = farAway,
                    upstreamConfidence = 0.90,
                ),
            )

        assertTrue(result.correspondenceCoverage < 0.50)
        assertTrue(!result.accepted)
        assertTrue(
            result.warnings.any {
                it.contains("correspondence")
            },
        )
    }


    @Test
    fun stale_surface_frame_does_not_rewrite_canonical_temporal_history() {
        val engine = CanonicalSurfaceTemporalPriorEngine()

        val canonical =
            tetraMesh(
                offset = Vec3(0.0, 0.0, 0.0),
                scale = 1.0,
            )

        val current =
            tetraMesh(
                offset = Vec3(0.01, 0.0, 0.0),
                scale = 1.01,
            )

        val first =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 11_000_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = current,
                    upstreamConfidence = 0.95,
                ),
            )

        val stale =
            engine.evaluate(
                CanonicalSurfaceTemporalPriorEngine.Input(
                    timestampNanos = 10_900_000_000L,
                    canonicalMesh = canonical,
                    currentMesh = tetraMesh(
                        offset = Vec3(1.0, 1.0, 1.0),
                        scale = 2.0,
                    ),
                    upstreamConfidence = 0.95,
                ),
            )

        assertTrue(
            stale.timestampNanos ==
                first.timestampNanos,
        )
        assertTrue(
            stale.warnings.any {
                it.contains("stale")
            },
        )
    }

    private fun tetraMesh(
        offset: Vec3,
        scale: Double,
    ): ReconstructionMesh {
        val base = listOf(
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
        )

        return ReconstructionMesh(
            vertices = base.map {
                MeshVertex(
                    Vec3(
                        it.x * scale + offset.x,
                        it.y * scale + offset.y,
                        it.z * scale + offset.z,
                    ),
                )
            },
            triangles = listOf(
                MeshTriangle(0, 1, 2),
                MeshTriangle(0, 1, 3),
                MeshTriangle(0, 2, 3),
                MeshTriangle(1, 2, 3),
            ),
            timestampNanos = 1L,
            confidence = 0.95,
        )
    }
}
