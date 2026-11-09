package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FaceMeshChunkedWireCodecTest {

    @Test
    fun fullFaceMeshRoundTripsThroughBoundedByteFragments() {
        val points =
            (0 until 468).map { index ->
                FaceMeshPoint3D(
                    index = index,
                    position =
                        Vec3(
                            x = index * 0.001,
                            y = index * 0.002,
                            z = 0.1 + index * 0.0001,
                        ),
                )
            }

        val triangles =
            (0 until 466).map { index ->
                FaceMeshWireCodecTestTriangle(
                    index,
                    index + 1,
                    index + 2,
                )
            }

        val frame =
            FaceMeshFrame(
                timestampNanos = 14_000_000_000L,
                imageWidth = 3840,
                imageHeight = 2160,
                rotationDegrees = 0,
                trackingId = 7,
                boundsLeft = 0f,
                boundsTop = 0f,
                boundsRight = 1f,
                boundsBottom = 1f,
                confidence = 0.98,
                pointCoverage = 1.0,
                triangleCoverage = 0.99,
                points = points,
                triangles =
                    triangles.map {
                        com.nivukx.nox.core.model.FaceMeshTriangle(
                            a = it.a,
                            b = it.b,
                            c = it.c,
                        )
                    },
            )

        val chunks =
            FaceMeshChunkedWireCodec.encode(
                frame = frame,
                streamId = "face-rig",
                frameSequence = 14L,
            )

        assertTrue(chunks.size >= 2)

        val decoded =
            FaceMeshChunkedWireCodec.decode(
                chunks.reversed(),
            )

        assertEquals(468, decoded.points.size)
        assertEquals(466, decoded.triangles.size)
        assertEquals(frame.timestampNanos, decoded.timestampNanos)
    }

    private data class FaceMeshWireCodecTestTriangle(
        val a: Int,
        val b: Int,
        val c: Int,
    )
}
