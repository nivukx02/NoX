package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceMeshGeometryEngineTest {

    @Test
    fun resultConvertsToExistingDetectorNeutralModelFrame() {
        val result =
            FaceMeshGeometryEngine.Result(
                timestampNanos = 42_000_000L,
                trackingId = 314,
                boundsLeft = 10f,
                boundsTop = 20f,
                boundsRight = 490f,
                boundsBottom = 500f,
                points = listOf(
                    FaceMeshGeometryEngine.Point(
                        index = 7,
                        positionImageSpace = Vec3(120.0, 220.0, -4.0),
                        confidence = 1.0,
                    ),
                    FaceMeshGeometryEngine.Point(
                        index = 9,
                        positionImageSpace = Vec3(180.0, 240.0, 3.0),
                        confidence = 1.0,
                    ),
                    FaceMeshGeometryEngine.Point(
                        index = 11,
                        positionImageSpace = Vec3(240.0, 210.0, 1.0),
                        confidence = 1.0,
                    ),
                ),
                triangles = listOf(
                    FaceMeshGeometryEngine.Triangle(
                        a = 7,
                        b = 9,
                        c = 11,
                    ),
                ),
                imageWidth = 640,
                imageHeight = 480,
                rotationDegrees = 90,
                confidence = 0.91,
                pointCoverage = 0.98,
                triangleCoverage = 0.97,
                faceAreaRatio = 0.42,
                depthSpreadImageUnits = 7.0,
            )

        val frame: FaceMeshFrame = result.toModelFrame()

        assertEquals(42_000_000L, frame.timestampNanos)
        assertEquals(314, frame.trackingId)
        assertEquals(640, frame.imageWidth)
        assertEquals(480, frame.imageHeight)
        assertEquals(90, frame.rotationDegrees)
        assertEquals(3, frame.pointCount)
        assertEquals(1, frame.triangleCount)
        assertEquals(0.91, frame.confidence, 1e-9)
        assertEquals(0.98, frame.pointCoverage, 1e-9)
        assertEquals(0.97, frame.triangleCoverage, 1e-9)
        assertEquals(7, frame.points[0].index)
        assertEquals(
            -4.0,
            frame.points[0].position.z,
            1e-9,
        )
        assertTrue(frame.points.all {
            it.position.x.isFinite() &&
                it.position.y.isFinite() &&
                it.position.z.isFinite()
        })
        assertEquals(7, frame.triangles[0].a)
        assertEquals(9, frame.triangles[0].b)
        assertEquals(11, frame.triangles[0].c)
    }
}
