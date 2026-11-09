package com.nivukx.nox.core.model

class SpatiotemporalObservationTest {

    private inline fun assertThrows(block: () -> Unit) {
        val threw =
            try {
                block()
                false
            } catch (_: IllegalArgumentException) {
                true
            }
        check(threw) { "Expected IllegalArgumentException" }
    }

    fun zeroRowOffsetUsesFrameTimestamp() {
        val observation =
            SpatiotemporalObservation(
                cameraId = "cam-a",
                landmarkId = 7,
                x = 320.0,
                y = 0.0,
                confidence = 0.9,
                frameTimestampNanos = 1_000_000_000L,
                rowTimestampOffsetNanos = 0L,
                temporalAlignmentOffsetNanos = 0L,
                rollingShutterConfidence = 1.0,
                sensorConfidence = 1.0,
            )

        check(observation.sampleTimestampNanos() == 1_000_000_000L)
    }

    fun combinesRowAndAlignmentOffsetDeterministically() {
        val observation =
            SpatiotemporalObservation(
                cameraId = "cam-b",
                landmarkId = 11,
                x = 640.0,
                y = 720.0,
                confidence = 0.8,
                frameTimestampNanos = 5_000_000_000L,
                rowTimestampOffsetNanos = 4_000_000L,
                temporalAlignmentOffsetNanos = -1_500_000L,
                rollingShutterConfidence = 0.92,
                sensorConfidence = 0.88,
            )

        check(observation.sampleTimestampNanos() == 5_002_500_000L)
    }

    fun rejectsInvalidConfidenceAndNonFiniteCoordinates() {
        assertThrows {
            SpatiotemporalObservation(
                cameraId = "cam-c",
                landmarkId = 1,
                x = Double.NaN,
                y = 20.0,
                confidence = 0.5,
                frameTimestampNanos = 1L,
                rowTimestampOffsetNanos = 0L,
                temporalAlignmentOffsetNanos = 0L,
                rollingShutterConfidence = 1.0,
                sensorConfidence = 1.0,
            )
        }
    }

    fun rejectsConfidenceOutsideRange() {
        assertThrows {
            SpatiotemporalObservation(
                cameraId = "cam-d",
                landmarkId = 1,
                x = 0.0,
                y = 0.0,
                confidence = 1.5,
                frameTimestampNanos = 1L,
                rowTimestampOffsetNanos = 0L,
                temporalAlignmentOffsetNanos = 0L,
                rollingShutterConfidence = 1.0,
                sensorConfidence = 1.0,
            )
        }
    }
}

fun main() {
    val test = SpatiotemporalObservationTest()
    test.zeroRowOffsetUsesFrameTimestamp()
    test.combinesRowAndAlignmentOffsetDeterministically()
    test.rejectsInvalidConfidenceAndNonFiniteCoordinates()
    test.rejectsConfidenceOutsideRange()
}
