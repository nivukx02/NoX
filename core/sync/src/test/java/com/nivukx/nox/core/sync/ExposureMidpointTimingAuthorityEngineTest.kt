package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.FrameTelemetry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ExposureMidpointTimingAuthorityEngineTest {
    private fun telemetry(
        sensor: Long = 1_000_000L,
        exposure: Long = 10_000L,
        frame: Long = 16_666L,
        skew: Long = 2_000L,
        readout: Long? = 1_000_000L,
    ) = FrameTelemetry(
        sensorTimestampNanos = sensor,
        monotonicTimestampNanos = sensor + 5_000L,
        exposureTimeNanos = exposure,
        sensorSensitivityIso = 400,
        frameDurationNanos = frame,
        rollingShutterSkewNanos = skew,
        aeState = null,
        afState = null,
        awbState = null,
        lensFocusDistanceDiopters = null,
        focusDistanceMeters = null,
        readoutTimestampNanos = readout,
    )

    @Test
    fun exposureMidpointIsUsedAsAuthoritativeGeometryTime() {
        val result = ExposureMidpointTimingAuthorityEngine().resolve(telemetry(), rowCount = 1)
        assertTrue(result.valid)
        assertEquals(1_005_000L, result.evidence!!.exposureMidpointNanos)
        assertEquals(2_000L, result.evidence.readoutDurationNanos)
        assertTrue(result.evidence.hardwareAuthoritative)
    }

    @Test
    fun rollingShutterTimingIsDistributedAcrossRows() {
        val result = ExposureMidpointTimingAuthorityEngine().resolve(telemetry(skew = 9_000L), 10)
        assertEquals(1_000.0, result.evidence!!.rowTimeNanos, 1e-9)
        assertEquals(1_000_500.0, result.evidence.rowTimestamp(0, 10), 1e-9)
        assertEquals(1_009_500.0, result.evidence.rowTimestamp(9, 10), 1e-9)
    }

    @Test
    fun missingHardwareReadoutUsesSensorOnlyAuthority() {
        val result = ExposureMidpointTimingAuthorityEngine().resolve(telemetry(readout = null))
        assertEquals(ExposureMidpointTimingAuthorityEngine.Authority.SENSOR_ONLY, result.authority)
        assertFalse(result.evidence!!.hardwareAuthoritative)
    }

    @Test
    fun duplicateTimestampInputIsDeterministic() {
        val engine = ExposureMidpointTimingAuthorityEngine()
        val a = engine.resolve(telemetry())
        val b = engine.resolve(telemetry())
        assertEquals(a, b)
    }

    @Test
    fun hostileReadoutOffsetLowersAuthorityButStillPreservesTiming() {
        val result = ExposureMidpointTimingAuthorityEngine().resolve(telemetry(readout = 300_000_000L))
        assertEquals(ExposureMidpointTimingAuthorityEngine.Authority.SENSOR_AND_READOUT_INFERRED, result.authority)
        assertTrue(result.warnings.isNotEmpty())
        assertTrue(result.evidence != null)
    }

    @Test
    fun invalidMetadataFailsClosed() {
        val result = ExposureMidpointTimingAuthorityEngine().resolve(telemetry(exposure = 600_000_000L))
        assertFalse(result.valid)
        assertEquals(ExposureMidpointTimingAuthorityEngine.Authority.INVALID, result.authority)
    }
}