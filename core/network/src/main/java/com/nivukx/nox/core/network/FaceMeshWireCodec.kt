package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshTriangle
import com.nivukx.nox.core.model.Vec3
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Bounded binary transport for dense facial geometry.
 *
 * A frame contains up to the documented 468 mesh points and the detector's triangle topology.
 * Coordinates stay in image-space units because ML Kit's Z value is scaled like X/Y rather than
 * being metric depth. The payload is deterministic and comfortably below the NoX control packet
 * ceiling.
 */
object FaceMeshWireCodec {
    private const val VERSION = 2
    private const val MAX_POINTS = 468
    private const val MAX_TRIANGLES = 960
    private const val HEADER_BYTES = 56
    private const val POINT_BYTES = 16
    private const val TRIANGLE_BYTES = 12

    fun encode(frame: FaceMeshFrame): ByteArray {
        val points = frame.points
            .filter { it.index in 0 until MAX_POINTS }
            .distinctBy { it.index }
            .take(MAX_POINTS)

        val validPointIds = points.map { it.index }.toHashSet()
        val triangles = frame.triangles
            .filter {
                it.a in validPointIds &&
                    it.b in validPointIds &&
                    it.c in validPointIds &&
                    it.a != it.b &&
                    it.a != it.c &&
                    it.b != it.c
            }
            .distinct()
            .take(MAX_TRIANGLES)

        val capacity =
            HEADER_BYTES +
                points.size * POINT_BYTES +
                triangles.size * TRIANGLE_BYTES

        require(capacity <= NoXWireProtocol.MAX_PACKET_BYTES - 32) {
            "Face mesh payload exceeds transport safety budget"
        }

        return ByteBuffer
            .allocate(capacity)
            .order(ByteOrder.BIG_ENDIAN)
            .apply {
                putInt(VERSION)
                putLong(frame.timestampNanos)
                putInt(frame.imageWidth.coerceAtLeast(1))
                putInt(frame.imageHeight.coerceAtLeast(1))
                putInt(frame.rotationDegrees)
                putInt(frame.trackingId ?: -1)
                putFloat(frame.boundsLeft)
                putFloat(frame.boundsTop)
                putFloat(frame.boundsRight)
                putFloat(frame.boundsBottom)
                putFloat(frame.confidence.coerceIn(0.0, 1.0).toFloat())
                putFloat(frame.pointCoverage.coerceIn(0.0, 1.0).toFloat())
                putShort(points.size.toShort())
                putShort(triangles.size.toShort())

                points.forEach { point ->
                    putInt(point.index)
                    putFloat(point.position.x.toFloat())
                    putFloat(point.position.y.toFloat())
                    putFloat(point.position.z.toFloat())
                }

                triangles.forEach { triangle ->
                    putInt(triangle.a)
                    putInt(triangle.b)
                    putInt(triangle.c)
                }
            }
            .array()
    }

    fun decode(bytes: ByteArray): FaceMeshFrame {
        require(bytes.size >= HEADER_BYTES) {
            "Face mesh payload too small"
        }

        val buffer = ByteBuffer
            .wrap(bytes)
            .order(ByteOrder.BIG_ENDIAN)

        require(buffer.int == VERSION) {
            "Unsupported face mesh payload version"
        }

        val timestamp = buffer.long
        val width = buffer.int
        val height = buffer.int
        val rotation = buffer.int
        val trackingId = buffer.int.takeIf { it >= 0 }
        val left = buffer.float
        val top = buffer.float
        val right = buffer.float
        val bottom = buffer.float
        val confidence = buffer.float.toDouble().coerceIn(0.0, 1.0)
        val pointCoverage = buffer.float.toDouble().coerceIn(0.0, 1.0)
        val pointCount = buffer.short.toInt() and 0xFFFF
        val triangleCount = buffer.short.toInt() and 0xFFFF

        require(pointCount in 0..MAX_POINTS)
        require(triangleCount in 0..MAX_TRIANGLES)

        val expected =
            HEADER_BYTES +
                pointCount * POINT_BYTES +
                triangleCount * TRIANGLE_BYTES
        require(bytes.size == expected) {
            "Invalid face mesh payload length"
        }

        val points = ArrayList<FaceMeshPoint3D>(pointCount)
        val seen = HashSet<Int>(pointCount)

        repeat(pointCount) {
            val index = buffer.int
            val x = buffer.float.toDouble()
            val y = buffer.float.toDouble()
            val z = buffer.float.toDouble()

            require(index in 0 until MAX_POINTS)
            require(seen.add(index)) {
                "Duplicate face mesh point index"
            }
            require(
                x.isFinite() &&
                    y.isFinite() &&
                    z.isFinite(),
            )

            points += FaceMeshPoint3D(
                index = index,
                position = Vec3(x, y, z),
            )
        }

        val triangles = ArrayList<FaceMeshTriangle>(triangleCount)
        repeat(triangleCount) {
            val triangle = FaceMeshTriangle(
                a = buffer.int,
                b = buffer.int,
                c = buffer.int,
            )
            require(
                triangle.a in seen &&
                    triangle.b in seen &&
                    triangle.c in seen,
            ) {
                "Face mesh triangle references unknown point"
            }
            triangles += triangle
        }

        require(buffer.remaining() == 0) {
            "Trailing face mesh payload bytes"
        }

        return FaceMeshFrame(
            timestampNanos = timestamp,
            imageWidth = width,
            imageHeight = height,
            rotationDegrees = rotation,
            trackingId = trackingId,
            boundsLeft = left,
            boundsTop = top,
            boundsRight = right,
            boundsBottom = bottom,
            confidence = confidence,
            pointCoverage = pointCoverage,
            triangleCoverage =
                (triangleCount.toDouble() / MAX_TRIANGLES.toDouble())
                    .coerceIn(0.0, 1.0),
            points = points,
            triangles = triangles,
        )
    }
}
