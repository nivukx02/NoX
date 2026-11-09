package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshTriangle
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceMeshWireCodecTest {

    private fun frame(
        pointCount: Int = 468,
        triangleCount: Int = 900,
        trackingId: Int? = 42,
    ): FaceMeshFrame {
        val points = (0 until pointCount).map { index ->
            FaceMeshPoint3D(
                index = index,
                position = Vec3(
                    x = index.toDouble() * 0.25,
                    y = index.toDouble() * 0.10,
                    z = -index.toDouble() * 0.05,
                ),
            )
        }

        val triangles = (0 until triangleCount).map { index ->
            val ring = index / pointCount
            val a = index % pointCount
            val b = (a + 1 + ring) % pointCount
            val c = (a + 2 + ring * 2) % pointCount
            FaceMeshTriangle(a, b, c)
        }

        return FaceMeshFrame(
            timestampNanos = 4_000_000_000L,
            imageWidth = 1920,
            imageHeight = 1080,
            rotationDegrees = 90,
            trackingId = trackingId,
            boundsLeft = 420f,
            boundsTop = 170f,
            boundsRight = 1480f,
            boundsBottom = 890f,
            confidence = 0.94,
            pointCoverage = pointCount.toDouble() / 468.0,
            triangleCoverage = triangleCount.toDouble() / 960.0,
            points = points,
            triangles = triangles,
        )
    }

    @Test
    fun roundTripPreservesStableDenseMeshSchema() {
        val source = frame()
        val encoded = FaceMeshWireCodec.encode(source)
        val decoded = FaceMeshWireCodec.decode(encoded)

        assertEquals(source.timestampNanos, decoded.timestampNanos)
        assertEquals(source.imageWidth, decoded.imageWidth)
        assertEquals(source.imageHeight, decoded.imageHeight)
        assertEquals(source.rotationDegrees, decoded.rotationDegrees)
        assertEquals(source.trackingId, decoded.trackingId)
        assertEquals(source.pointCount, decoded.pointCount)
        assertEquals(source.triangleCount, decoded.triangleCount)
        assertEquals(
            source.points.first().index,
            decoded.points.first().index,
        )
        assertEquals(
            source.points.first().position.x,
            decoded.points.first().position.x,
            1e-5,
        )
        assertEquals(
            source.points.last().position.z,
            decoded.points.last().position.z,
            1e-5,
        )
        assertEquals(
            source.triangles[100].a,
            decoded.triangles[100].a,
        )
    }

    @Test
    fun transportStaysUnderControlPacketBudget() {
        val encoded = FaceMeshWireCodec.encode(frame())
        assertTrue(
            encoded.size <
                NoXWireProtocol.MAX_PACKET_BYTES,
        )
    }

    @Test
    fun decodeRejectsTruncatedPayload() {
        val encoded = FaceMeshWireCodec.encode(frame())
        val truncated = encoded.copyOf(encoded.size - 1)

        runCatching {
            FaceMeshWireCodec.decode(truncated)
        }.onSuccess {
            throw AssertionError("truncated mesh unexpectedly decoded")
        }
    }

    @Test
    fun decodeRejectsTrailingBytes() {
        val encoded = FaceMeshWireCodec.encode(frame())
        val corrupted =
            encoded + byteArrayOf(0x7f, 0x2a)

        runCatching {
            FaceMeshWireCodec.decode(corrupted)
        }.onSuccess {
            throw AssertionError("trailing bytes unexpectedly accepted")
        }
    }

    @Test
    fun sparseFramesRemainValidAndDoNotClaimDenseGeometry() {
        val sparse = frame(
            pointCount = 12,
            triangleCount = 6,
            trackingId = null,
        )

        val decoded =
            FaceMeshWireCodec.decode(
                FaceMeshWireCodec.encode(sparse),
            )

        assertEquals(12, decoded.pointCount)
        assertEquals(6, decoded.triangleCount)
        assertTrue(!decoded.isDense)
        assertEquals(null, decoded.trackingId)
    }

    @Test
    fun versionedPayloadIncludesTrackAssociation() {
        val source = frame(
            pointCount = 64,
            triangleCount = 62,
            trackingId = 77,
        )

        val decoded =
            FaceMeshWireCodec.decode(
                FaceMeshWireCodec.encode(source),
            )

        assertEquals(77, decoded.trackingId)
    }

    @Test
    fun coordinateNaNCannotSurviveModelValidation() {
        runCatching {
            FaceMeshFrame(
                timestampNanos = 1L,
                imageWidth = 100,
                imageHeight = 100,
                rotationDegrees = 0,
                trackingId = null,
                boundsLeft = 0f,
                boundsTop = 0f,
                boundsRight = 80f,
                boundsBottom = 80f,
                confidence = 0.8,
                pointCoverage = 0.8,
                triangleCoverage = 0.8,
                points = listOf(
                    FaceMeshPoint3D(
                        index = 0,
                        position = Vec3(
                            Double.NaN,
                            0.0,
                            0.0,
                        ),
                    ),
                ),
                triangles = emptyList(),
            )
        }.onSuccess {
            throw AssertionError("NaN coordinate passed model validation")
        }
    }

    @Test
    fun duplicatePointIdsAreRejectedAtModelBoundary() {
        val duplicate = listOf(
            FaceMeshPoint3D(0, Vec3(0.0, 0.0, 0.0)),
            FaceMeshPoint3D(0, Vec3(1.0, 0.0, 0.0)),
        )

        runCatching {
            FaceMeshFrame(
                timestampNanos = 1L,
                imageWidth = 100,
                imageHeight = 100,
                rotationDegrees = 0,
                trackingId = null,
                boundsLeft = 0f,
                boundsTop = 0f,
                boundsRight = 80f,
                boundsBottom = 80f,
                confidence = 0.8,
                pointCoverage = 0.8,
                triangleCoverage = 0.8,
                points = duplicate,
                triangles = emptyList(),
            )
        }.onSuccess {
            throw AssertionError("duplicate mesh point id was accepted")
        }
    }
}
