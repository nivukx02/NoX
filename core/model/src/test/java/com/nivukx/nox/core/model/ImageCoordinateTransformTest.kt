package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Test

class ImageCoordinateTransformTest {

    @Test
    fun rotationZeroPreservesSensorCoordinates() {
        val point =
            ImageCoordinateTransform.detectorToSensor(
                point = Vec2(320.0, 180.0),
                rawWidth = 1920,
                rawHeight = 1080,
                rotationDegrees = 0,
            )

        assertEquals(320.0, point.x, 0.000001)
        assertEquals(180.0, point.y, 0.000001)
    }

    @Test
    fun rotationNinetyMapsPortraitDetectorBackToSensor() {
        val point =
            ImageCoordinateTransform.detectorToSensor(
                point = Vec2(100.0, 300.0),
                rawWidth = 1920,
                rawHeight = 1080,
                rotationDegrees = 90,
            )

        assertEquals(300.0, point.x, 0.000001)
        assertEquals(979.0, point.y, 0.000001)
    }

    @Test
    fun rotationOneEightyFlipsBothAxes() {
        val point =
            ImageCoordinateTransform.detectorToSensor(
                point = Vec2(100.0, 200.0),
                rawWidth = 1920,
                rawHeight = 1080,
                rotationDegrees = 180,
            )

        assertEquals(1819.0, point.x, 0.000001)
        assertEquals(879.0, point.y, 0.000001)
    }

    @Test
    fun normalizationRemainsBounded() {
        val point =
            ImageCoordinateTransform.detectorToNormalizedSensor(
                point = Vec2(
                    x = -500.0,
                    y = 5000.0,
                ),
                rawWidth = 1920,
                rawHeight = 1080,
                rotationDegrees = 270,
            )

        assertEquals(true, point.x in 0.0..1.0)
        assertEquals(true, point.y in 0.0..1.0)
    }
}
