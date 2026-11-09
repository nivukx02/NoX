package com.nivukx.nox.core.sync

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraSynchronizationQualityEngineTest {

    @Test
    fun tightlySynchronizedRigPasses() {
        val engine = CameraSynchronizationQualityEngine(
            CameraSynchronizationQualityEngine.Config(
                minimumSamples = 4,
            ),
        )

        val samples = listOf("A", "B", "C").flatMap { camera ->
            (0 until 4).map { index ->
                CameraSynchronizationQualityEngine.CameraSample(
                    cameraId = camera,
                    timestampNanos = 1_000_000_000L +
                        index * 16_666_666L,
                    clockOffsetNanos = if (camera == "A") 150_000L else 280_000L,
                    roundTripNanos = 8_000_000L,
                    packetLoss = 0.001,
                    jitterNanos = 120_000L,
                )
            }
        }

        val result = engine.evaluate(
            samples = samples,
            fastMotion = true,
        )

        assertTrue(result.cameras.size == 3)
        assertTrue(result.rigScore > 0.70)
    }

    @Test
    fun poorJitterAndLossAreNotProductionReady() {
        val engine = CameraSynchronizationQualityEngine(
            CameraSynchronizationQualityEngine.Config(
                minimumSamples = 4,
            ),
        )

        val samples = (0 until 4).map { index ->
            CameraSynchronizationQualityEngine.CameraSample(
                cameraId = "A",
                timestampNanos = 2_000_000_000L +
                    index * 16_666_666L,
                clockOffsetNanos = 9_000_000L,
                roundTripNanos = 250_000_000L,
                packetLoss = 0.18,
                jitterNanos = 12_000_000L,
            )
        }

        val result = engine.evaluate(samples)

        assertFalse(result.ready)
        assertTrue(result.recommendations.isNotEmpty())
    }
    @Test
    fun acceptedClockGraphContributesToRigAuthority() {
        val graph =
            DistributedClockAuthorityEngine().solve(
                nodes =
                    listOf(
                        DistributedClockAuthorityEngine.Node(
                            DeviceId("master"),
                            0.99,
                        ),
                        DistributedClockAuthorityEngine.Node(
                            DeviceId("cam-a"),
                            0.96,
                        ),
                    ),
                edges =
                    listOf(
                        DistributedClockAuthorityEngine.Edge(
                            from = DeviceId("master"),
                            to = DeviceId("cam-a"),
                            offsetNanos = 1_000_000L,
                            residualNanos = 200_000L,
                            confidence = 0.98,
                        ),
                    ),
            )

        val result =
            CameraSynchronizationQualityEngine().evaluate(
                samples =
                    listOf(
                        sample("cam-a"),
                        sample("cam-b"),
                    ),
                fastMotion = true,
                clockGraph = graph,
            )

        assertEquals(
            graph.confidence,
            result.clockGraphConfidence,
        )
        assertEquals(
            graph.accepted,
            result.clockGraphAccepted,
        )
    }

    @Test
    fun rejectedClockGraphPreventsReferenceReadyState() {
        val graph =
            DistributedClockAuthorityEngine(
                DistributedClockAuthorityEngine.Config(
                    maximumCycleResidualNanos = 1_000_000L,
                    minimumCycleConsistency = 0.90,
                ),
            ).solve(
                nodes =
                    listOf(
                        DistributedClockAuthorityEngine.Node(
                            DeviceId("a"),
                            0.95,
                        ),
                        DistributedClockAuthorityEngine.Node(
                            DeviceId("b"),
                            0.95,
                    )
                ),
                edges =
                    listOf(
                        DistributedClockAuthorityEngine.Edge(
                            from = DeviceId("a"),
                            to = DeviceId("b"),
                            offsetNanos = 20_000_000L,
                            residualNanos = 0L,
                            confidence = 0.95,
                        ),
                    ),
            )

        val result =
            CameraSynchronizationQualityEngine().evaluate(
                samples =
                    listOf(
                        sample("a"),
                        sample("b"),
                    ),
                fastMotion = true,
                clockGraph = graph,
            )

        assertTrue(!result.ready || result.clockGraphAccepted == false)
    }

    @Test
    fun hardwareSynchronizedSensorClockRanksAboveSoftwareOnlyClock() {
        val engine = CameraSynchronizationQualityEngine()

        val hardwareSamples =
            (0 until 8).map { index ->
                CameraSynchronizationQualityEngine.CameraSample(
                    cameraId = "hw",
                    timestampNanos =
                        index * 10_000_000L,
                    clockOffsetNanos = 100_000L,
                    roundTripNanos = 2_000_000L,
                    packetLoss = 0.0,
                    jitterNanos = 100_000L,
                    timestampDomain =
                        MultiCameraTemporalAlignmentEngine.TimestampDomain.SENSOR,
                    hardwareSynchronized = true,
                    sourceComparableAcrossDevices = true,
                    hardwareReadoutTimingConfidence = 0.99,
                )
            }

        val softwareSamples =
            hardwareSamples.map {
                it.copy(
                    cameraId = "sw",
                    timestampDomain =
                        MultiCameraTemporalAlignmentEngine.TimestampDomain.MONOTONIC,
                    hardwareSynchronized = false,
                    sourceComparableAcrossDevices = false,
                    hardwareReadoutTimingConfidence = 0.0,
                )
            }

        val result =
            engine.evaluate(
                hardwareSamples + softwareSamples,
            )

        val hardware =
            result.cameras.first {
                it.cameraId == "hw"
            }
        val software =
            result.cameras.first {
                it.cameraId == "sw"
            }

        assertTrue(
            hardware.timestampAuthorityScore >
                software.timestampAuthorityScore,
        )
        assertTrue(
            hardware.score >
                software.score,
        )
    }

}
