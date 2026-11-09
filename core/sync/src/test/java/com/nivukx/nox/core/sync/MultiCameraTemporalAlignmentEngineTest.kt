package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FrameStamp
import com.nivukx.nox.core.model.RollingShutterModel
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiCameraTemporalAlignmentEngineTest {

    private fun frame(
        device: String,
        sensorNanos: Long,
        monotonicNanos: Long = sensorNanos,
        exposureNanos: Long = 10_000_000L,
        domain: MultiCameraTemporalAlignmentEngine.TimestampDomain =
            MultiCameraTemporalAlignmentEngine.TimestampDomain.SENSOR,
        row: Double = 0.5,
        rollingShutter: RollingShutterModel? = null,
        comparable: Boolean = true,
        hardwareSync: Boolean = false,
    ): MultiCameraTemporalAlignmentEngine.FrameInput =
        MultiCameraTemporalAlignmentEngine.FrameInput(
            deviceId = DeviceId(device),
            stamp = FrameStamp(
                sequence = sensorNanos / 1_000_000L,
                sensorTimestampNanos = sensorNanos,
                monotonicTimestampNanos = monotonicNanos,
                exposureTimeNanos = exposureNanos,
            ),
            domain = domain,
            hardwareSynchronized = hardwareSync,
            rollingShutter = rollingShutter,
            sourceIsComparableAcrossDevices = comparable,
            normalizedRow = row,
        )

    @Test
    fun synchronizedSensorFramesAreAccepted() {
        val engine = MultiCameraTemporalAlignmentEngine()

        val result = engine.alignGroup(
            frames = listOf(
                frame("cam-a", 1_000_000_000L, hardwareSync = true),
                frame("cam-b", 1_001_000_000L, hardwareSync = true),
                frame("cam-c", 999_000_000L, hardwareSync = true),
            ),
            referenceDevice = DeviceId("cam-a"),
        )

        assertTrue(result.accepted)
        assertTrue(result.acceptedDeviceCount >= 2)
        assertTrue(result.confidence > 0.60)
        assertTrue(result.spreadNanos <= 3_000_000L)
    }

    @Test
    fun unknownTimestampDomainIsRejectedForMetricAlignment() {
        val engine = MultiCameraTemporalAlignmentEngine()

        val result = engine.alignGroup(
            frames = listOf(
                frame(
                    device = "cam-a",
                    sensorNanos = 1_000_000_000L,
                    domain =
                        MultiCameraTemporalAlignmentEngine.TimestampDomain.UNKNOWN,
                ),
                frame(
                    device = "cam-b",
                    sensorNanos = 1_001_000_000L,
                    domain =
                        MultiCameraTemporalAlignmentEngine.TimestampDomain.SENSOR,
                ),
            ),
        )

        assertFalse(
            result.samples
                .first { it.deviceId == DeviceId("cam-a") }
                .accepted,
        )
        assertTrue(
            result.warnings.any {
                it.contains("timestamp-domain")
            },
        )
    }

    @Test
    fun rollingShutterCompensationChangesRowTimeDeterministically() {
        val engine = MultiCameraTemporalAlignmentEngine()

        val result = engine.alignGroup(
            frames = listOf(
                frame(
                    device = "cam-a",
                    sensorNanos = 1_000_000_000L,
                    exposureNanos = 4_000_000L,
                    row = 0.0,
                    rollingShutter =
                        RollingShutterModel(
                            readoutTimeNanos = 12_000_000L,
                            exposureTimeNanos = 4_000_000L,
                            directionSign = 1,
                        ),
                ),
                frame(
                    device = "cam-b",
                    sensorNanos = 1_006_000_000L,
                    row = 0.5,
                ),
            ),
            referenceDevice = DeviceId("cam-b"),
        )

        val rowSample =
            result.samples.first {
                it.deviceId == DeviceId("cam-a")
            }

        assertTrue(
            rowSample.rollingShutterOffsetNanos < 0L,
        )
        assertTrue(
            rowSample.exposureCenterOffsetNanos == 2_000_000L,
        )
    }

    @Test
    fun robustTemporalStatisticsKeepTransientOutliersVisible() {
        val engine = MultiCameraTemporalAlignmentEngine()

        val result =
            engine.alignGroup(
                frames = listOf(
                    frame(
                        device = "cam-a",
                        sensorNanos = 5_000_000_000L,
                    ),
                    frame(
                        device = "cam-b",
                        sensorNanos = 5_001_000_000L,
                    ),
                    frame(
                        device = "cam-c",
                        sensorNanos = 5_018_000_000L,
                    ),
                ),
                referenceDevice = DeviceId("cam-a"),
            )

        assertTrue(
            result.warnings.any {
                it.contains("frame-spread-above-preferred-bound")
            },
        )
        assertTrue(result.maximumAbsoluteOffsetNanos >= 17_000_000L)
        assertTrue(result.confidence < 0.85)
    }

    @Test
    fun acceptedSubframeAlignmentAugmentsClockTelemetry() {
        val engine = MultiCameraTemporalAlignmentEngine()

        val result =
            SubframeMotionAlignmentEngine.Result(
                offsetNanos = 6_250_000.0,
                confidence = 0.87,
                correlation = 0.96,
                overlapSamples = 24,
                samplePeriodNanos = 16_666_667.0,
                accepted = true,
                warnings = emptyList(),
            )

        val telemetry =
            engine.applySubframeAlignment(
                deviceId = DeviceId("remote"),
                result = result,
            )

        assertTrue(
            telemetry.subframeOffsetEwmaNanos >
                5_000_000.0,
        )
        assertTrue(
            telemetry.subframeOffsetEwmaNanos <
                7_500_000.0,
        )
        assertTrue(
            telemetry.subframeConfidence > 0.80,
        )
    }

    @Test
    fun directSubframeEvidenceUpdatesTelemetryWithoutSyntheticCorrelation() {
        val engine =
            MultiCameraTemporalAlignmentEngine()

        val telemetry =
            engine.applySubframeOffsetEvidence(
                deviceId = DeviceId("remote-direct"),
                offsetNanos = 4_750_000.0,
                confidence = 0.91,
            )

        assertTrue(
            telemetry.subframeOffsetEwmaNanos >
                4_000_000.0,
        )
        assertTrue(
            telemetry.subframeOffsetEwmaNanos <
                5_500_000.0,
        )
        assertTrue(
            telemetry.subframeConfidence >
                0.85,
        )
    }

    @Test
    fun pairIngestionBuildsStableClockTelemetry() {
        val engine = MultiCameraTemporalAlignmentEngine()

        repeat(8) { index ->
            val base = 2_000_000_000L +
                index * 33_333_333L

            engine.ingestPair(
                reference =
                    frame("master", base),
                sample =
                    frame(
                        device = "remote",
                        sensorNanos = base + 2_000_000L,
                    ),
            )
        }

        val telemetry =
            engine.telemetry(DeviceId("remote"))

        assertTrue(telemetry.samples == 8)
        assertTrue(
            telemetry.offsetEwmaNanos > 1_000_000.0,
        )
        assertTrue(
            telemetry.offsetEwmaNanos < 3_000_000.0,
        )
        assertTrue(telemetry.confidence > 0.50)
    }
}


class MultiCameraTemporalAlignmentEngineClockDriftTest {
    @Test
    fun robustPeerClockDriftTracksLongRunRateAndRejectsOneTransientOffset() {
        val engine =
            MultiCameraTemporalAlignmentEngine(
                MultiCameraTemporalAlignmentEngine.Config(
                    maximumPeerClockDriftPpm = 250.0,
                    peerClockDriftAlpha = 0.24,
                    historyPerDevice = 16,
                ),
            )

        repeat(10) { index ->
            val masterTimestamp =
                10_000_000_000L + index * 100_000_000L
            val driftOffsetNanos =
                index * 10_000L
            val transientOutlier =
                if (index == 5) 2_000_000L else 0L

            engine.ingestPair(
                reference = frame(
                    device = "master",
                    sensorNanos = masterTimestamp,
                ),
                sample = frame(
                    device = "remote",
                    sensorNanos =
                        masterTimestamp +
                            2_000_000L +
                            driftOffsetNanos +
                            transientOutlier,
                ),
            )
        }

        val telemetry =
            engine.telemetry(DeviceId("remote"))

        assertTrue(
            telemetry.peerClockDriftPpm > 40.0,
        )
        assertTrue(
            telemetry.peerClockDriftPpm < 180.0,
        )
        assertTrue(
            kotlin.math.abs(telemetry.offsetJitterNanos) <
                1_500_000.0,
        )
    }

    @Test
    fun peerClockDriftProjectionRemainsBoundedDuringAlignment() {
        val engine =
            MultiCameraTemporalAlignmentEngine(
                MultiCameraTemporalAlignmentEngine.Config(
                    maximumPeerClockDriftPpm = 100.0,
                    peerClockDriftAlpha = 0.50,
                ),
            )

        repeat(8) { index ->
            val base =
                20_000_000_000L +
                    index * 250_000_000L
            engine.ingestPair(
                reference = frame(
                    device = "master",
                    sensorNanos = base,
                ),
                sample = frame(
                    device = "remote",
                    sensorNanos =
                        base +
                            1_500_000L +
                            index * 25_000L,
                ),
            )
        }

        val result =
            engine.alignGroup(
                frames = listOf(
                    frame(
                        device = "master",
                        sensorNanos = 23_000_000_000L,
                    ),
                    frame(
                        device = "remote",
                        sensorNanos = 23_000_000_000L + 1_800_000L,
                    ),
                ),
                referenceDevice = DeviceId("master"),
            )

        val remote =
            result.samples.single {
                it.deviceId == DeviceId("remote")
            }

        assertTrue(remote.accepted)
        assertTrue(
            kotlin.math.abs(remote.estimatedClockOffsetNanos) <=
                40_000_000L,
        )
    }
}
