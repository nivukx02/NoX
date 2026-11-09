package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DistributedTimingBeaconObservationBufferTest {

    @Test
    fun observationsRemainBoundedPerCameraAndCalibrateAsOneSet() {
        val buffer =
            DistributedTimingBeaconObservationBuffer(
                DistributedTimingBeaconObservationBuffer.Config(
                    maximumDevices = 2,
                    maximumObservationsPerDevice = 4,
                ),
            )

        repeat(8) { index ->
            buffer.add(
                VisualTimingBeaconCalibrationEngine.Observation(
                    deviceId = DeviceId("cam-a"),
                    deviceTimestampNanos =
                        1_000_000_000L +
                            index * 20_000_000L,
                    beaconTimestampNanos =
                        1_002_000_000L +
                            index * 20_000_000L,
                    exposureTimeNanos = 2_000_000L,
                    rollingShutterSkewNanos = 500_000L,
                    normalizedRow = 0.5,
                    decodeConfidence = 0.98,
                ),
            )
        }

        assertEquals(
            4,
            buffer.observationCount(DeviceId("cam-a")),
        )
        assertEquals(
            1,
            buffer.deviceCount(),
        )

        val result =
            buffer.calibrate(
                engine =
                    VisualTimingBeaconCalibrationEngine(
                        VisualTimingBeaconCalibrationEngine.Config(
                            minimumSamplesPerCamera = 3,
                        ),
                    ),
            )

        assertTrue(result.deviceCount == 1)
    }

    @Test
    fun firstObservationForNewDeviceIsRetained() {
        val buffer =
            DistributedTimingBeaconObservationBuffer()

        val accepted =
            buffer.add(
                VisualTimingBeaconCalibrationEngine.Observation(
                    deviceId = DeviceId("first-device"),
                    deviceTimestampNanos = 1_000L,
                    beaconTimestampNanos = 2_000L,
                    exposureTimeNanos = 1_000L,
                    rollingShutterSkewNanos = 0L,
                    normalizedRow = 0.5,
                    decodeConfidence = 0.95,
                ),
            )

        assertTrue(accepted)
        assertEquals(1, buffer.deviceCount())
        assertEquals(
            1,
            buffer.observationCount(DeviceId("first-device")),
        )
    }

    @Test
    fun staleObservationsAreExpiredWithoutGrowingMemory() {
        val buffer =
            DistributedTimingBeaconObservationBuffer(
                DistributedTimingBeaconObservationBuffer.Config(
                    maximumDevices = 4,
                    maximumObservationsPerDevice = 8,
                    maximumAgeNanos = 100L,
                ),
            )

        buffer.add(
            VisualTimingBeaconCalibrationEngine.Observation(
                deviceId = DeviceId("cam-a"),
                deviceTimestampNanos = 1_000L,
                beaconTimestampNanos = 2_000L,
                exposureTimeNanos = 1L,
                rollingShutterSkewNanos = 0L,
                normalizedRow = 0.5,
                decodeConfidence = 0.95,
            ),
        )

        buffer.expire(1_101L)

        assertEquals(
            0,
            buffer.observationCount(DeviceId("cam-a")),
        )
        assertEquals(0, buffer.deviceCount())
    }
}
