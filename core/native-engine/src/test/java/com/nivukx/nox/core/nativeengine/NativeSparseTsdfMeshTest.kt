package com.nivukx.nox.core.nativeengine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Test

class NativeSparseTsdfMeshTest {
    @Test
    fun packedMeshContractCarriesIndexedTopologyAndUncertainty() {
        val mesh = NativeSparseTsdfMesh(
            vertexPositionsNormalsAndUncertainty = FloatArray(28),
            triangleIndices = intArrayOf(0, 1, 2, 0, 2, 3),
            vertexCount = 4,
            triangleCount = 2,
            confidence = 0.75,
        )
        assertEquals(4, mesh.vertexCount)
        assertEquals(2, mesh.triangleCount)
        assertEquals(0.75, mesh.confidence, 0.0)
        assertEquals(28, mesh.vertexPositionsNormalsAndUncertainty.size)
        assertArrayEquals(intArrayOf(0, 1, 2, 0, 2, 3), mesh.triangleIndices)
    }

    @Test(expected = IllegalArgumentException::class)
    fun packedMeshRejectsOutOfRangeIndices() {
        NativeSparseTsdfMesh(
            vertexPositionsNormalsAndUncertainty = FloatArray(21),
            triangleIndices = intArrayOf(0, 1, 9),
            vertexCount = 3,
            triangleCount = 1,
            confidence = 0.5,
        )
    }
    @Test
    fun gaussianTelemetryRejectsInvalidRanges() {
        GaussianSurfaceTelemetry(
            primitiveCount = 10,
            fusedObservations = 20,
            rejectedObservations = 2,
            dynamicPrimitives = 3,
            meanConfidence = 0.91,
            meanUncertaintyMeters = 0.0012,
            meanDynamicProbability = 0.12,
            temporalStability = 0.94,
            trustedPrimitiveRatio = 0.88,
        )
    }

    @Test
    fun gaussianObservationCarriesMetricEvidence() {
        val observation = GaussianSurfaceObservation(
            positionX = 0.1,
            positionY = 0.2,
            positionZ = 1.4,
            normalX = 0.0,
            normalY = 0.0,
            normalZ = 1.0,
            velocityX = 0.02,
            velocityY = 0.0,
            velocityZ = 0.0,
            varianceX = 0.0004,
            varianceY = 0.0004,
            varianceZ = 0.0008,
            confidence = 0.95,
            dynamicProbability = 0.08,
            temporalConsistency = 0.97,
            timestampNanos = 1_000_000_000L,
        )

        assertEquals(0.95, observation.confidence, 0.0)
        assertEquals(1_000_000_000L, observation.timestampNanos)
    }


}


class NativeSparseTsdfTelemetryFallbackContractTest {
    @Test
    fun unavailableTelemetryRemainsFiniteAndSafeForUiConsumers() {
        val telemetry =
            GaussianSurfaceTelemetry(
                primitiveCount = 0L,
                fusedObservations = 0L,
                rejectedObservations = 0L,
                dynamicPrimitives = 0L,
                meanConfidence = 0.0,
                meanUncertaintyMeters = 0.0,
                meanDynamicProbability = 1.0,
                temporalStability = 0.0,
                trustedPrimitiveRatio = 0.0,
                rejectedTriangleCount = 0L,
                nativeTelemetryAvailable = false,
                deformationNodeCount = 0L,
                deformationBindingCount = 0L,
                deformationRejectedBindingCount = 0L,
                deformationMeanNodeConfidence = 0.0,
                deformationMeanBindingConfidence = 0.0,
                deformationP95StrainRatio = 0.0,
                deformationMaximumStrainRatio = 0.0,
                deformationTemporalStability = 0.0,
                deformationTrustedBindingRatio = 0.0,
            )

        assertTrue(telemetry.meanUncertaintyMeters.isFinite())
        assertTrue(telemetry.deformationP95StrainRatio.isFinite())
        assertTrue(telemetry.deformationMaximumStrainRatio.isFinite())
        assertTrue(telemetry.rejectedTriangleCount >= 0L)
    }
}

class NativeSparseTsdfMeshSafetyTest {
    @Test(expected = IllegalArgumentException::class)
    fun rejectsVertexCountWhoseIntProductWrapsToArrayLength() {
        NativeSparseTsdfMesh(
            vertexPositionsNormalsAndUncertainty = FloatArray(3),
            triangleIndices = IntArray(0),
            vertexCount = 613_566_757,
            triangleCount = 0,
            confidence = 0.5,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsTriangleCountWhoseIntProductWrapsToArrayLength() {
        NativeSparseTsdfMesh(
            vertexPositionsNormalsAndUncertainty = FloatArray(0),
            triangleIndices = intArrayOf(0, 0),
            vertexCount = 0,
            triangleCount = 1_431_655_766,
            confidence = 0.5,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteVertexPayloads() {
        val vertices = FloatArray(7).also { it[4] = Float.POSITIVE_INFINITY }
        NativeSparseTsdfMesh(
            vertexPositionsNormalsAndUncertainty = vertices,
            triangleIndices = intArrayOf(0, 0, 0),
            vertexCount = 1,
            triangleCount = 1,
            confidence = 0.5,
        )
    }
}

