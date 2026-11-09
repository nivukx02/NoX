package com.nivukx.nox.core.model

/**
 * Canonical Camera2 coordinate conversion utilities.
 *
 * Detector libraries such as ML Kit expose geometry in the rotated image coordinate system while
 * Camera2 sensor metadata, focus regions and metric triangulation operate in the sensor plane.
 * Keeping this transform in one shared module prevents subtle 90/180/270 degree divergence across
 * perception, autofocus and reconstruction.
 */
object ImageCoordinateTransform {

    fun normalizeRotation(rotationDegrees: Int): Int {
        val normalized =
            (rotationDegrees % 360 + 360) % 360

        return when {
            normalized < 45 -> 0
            normalized < 135 -> 90
            normalized < 225 -> 180
            normalized < 315 -> 270
            else -> 0
        }
    }

    fun detectorToSensor(
        point: Vec2,
        rawWidth: Int,
        rawHeight: Int,
        rotationDegrees: Int,
    ): Vec2 {
        val width =
            rawWidth.coerceAtLeast(1)
        val height =
            rawHeight.coerceAtLeast(1)

        val rotation =
            normalizeRotation(rotationDegrees)

        val sensor =
            when (rotation) {
                90 -> Vec2(
                    point.y,
                    height - 1.0 - point.x,
                )

                180 -> Vec2(
                    width - 1.0 - point.x,
                    height - 1.0 - point.y,
                )

                270 -> Vec2(
                    width - 1.0 - point.y,
                    point.x,
                )

                else -> point
            }

        return Vec2(
            sensor.x.coerceIn(
                0.0,
                (width - 1).toDouble(),
            ),
            sensor.y.coerceIn(
                0.0,
                (height - 1).toDouble(),
            ),
        )
    }

    fun detectorToNormalizedSensor(
        point: Vec2,
        rawWidth: Int,
        rawHeight: Int,
        rotationDegrees: Int,
    ): Vec2 {
        val sensor =
            detectorToSensor(
                point = point,
                rawWidth = rawWidth,
                rawHeight = rawHeight,
                rotationDegrees = rotationDegrees,
            )

        return Vec2(
            (
                sensor.x /
                    rawWidth.coerceAtLeast(1)
                        .toDouble()
                ).coerceIn(0.0, 1.0),
            (
                sensor.y /
                    rawHeight.coerceAtLeast(1)
                        .toDouble()
                ).coerceIn(0.0, 1.0),
        )
    }
}
