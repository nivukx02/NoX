package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiCameraSynchronizationQualityEngineTest {

    @Test
    fun cleanRigIsSynchronized() {
        val engine = MultiCameraSynchronizationQualityEngine()

        val result = engine.evaluate(
            timestampNanos = 1_000_000_000L,
            samples = listOf(
                sample("front", 800_000L, 20_000_000L, 0.005, 20.0),
                sample("rear", -900_000L, 22_000_000L, 0.004, 24.0),
                sample("left", 1_100_000L, 24_000_000L, 0.006, 18.0),
            ),
        )

        assertEquals(
            MultiCameraSynchronizationQualityEngine.State.SYNCHRONIZED,
            result.state,
        )
        assertTrue(result.confidence >= 0.65)
        assertTrue(result.healthy)
    }

    @Test
    fun badPeerDegradesTheRig() {
        val engine = MultiCameraSynchronizationQualityEngine()

        val result = engine.evaluate(
            timestampNanos = 2_000_000_000L,
            samples = listOf(
                sample("front", 900_000L, 20_000_000L, 0.005, 30.0),
                sample("rear", 1_200_000L, 25_000_000L, 0.007, 25.0),
                sample("bad", 48_000_000L, 240_000_000L, 0.34, 900.0),
            ),
        )

        assertTrue(
            result.state ==
                MultiCameraSynchronizationQualityEngine.State.DEGRADED ||
                result.state ==
                MultiCameraSynchronizationQualityEngine.State.UNSAFE,
        )
        assertTrue(result.warnings.isNotEmpty())
        assertTrue(result.recommendations.isNotEmpty())
    }

    @Test
    fun poorSubframeAlignmentDegradesOtherwiseCleanRig() {
        val engine =
            MultiCameraSynchronizationQualityEngine()

        val result =
            engine.evaluate(
                timestampNanos = 5_000_000_000L,
                samples = listOf(
                    sample(
                        id = "front",
                        offsetNanos = 700_000L,
                        roundTripNanos = 20_000_000L,
                        packetLoss = 0.004,
                        ageMillis = 20.0,
                        subframeOffsetNanos = 9_000_000.0,
                        subframeConfidence = 0.36,
                    ),
                    sample(
                        id = "rear",
                        offsetNanos = -800_000L,
                        roundTripNanos = 22_000_000L,
                        packetLoss = 0.004,
                        ageMillis = 22.0,
                        subframeOffsetNanos = 8_500_000.0,
                        subframeConfidence = 0.40,
                    ),
                ),
            )

        assertTrue(
            result.state ==
                MultiCameraSynchronizationQualityEngine.State.DEGRADED ||
                result.state ==
                MultiCameraSynchronizationQualityEngine.State.UNSAFE,
        )
        assertTrue(
            result.warnings.any {
                it.contains(
                    "sub-frame",
                    ignoreCase = true,
                )
            },
        )
    }

    @Test
    fun precisionContractBlocksRigWithMillisecondClockSkew() {
        val engine =
            MultiCameraSynchronizationQualityEngine(
                MultiCameraSynchronizationQualityEngine.Config(
                    precisionContractEnabled = true,
                ),
            )

        val result =
            engine.evaluate(
                timestampNanos = 6_000_000_000L,
                samples = listOf(
                    sample(
                        id = "front",
                        offsetNanos = 1_600_000L,
                        roundTripNanos = 20_000_000L,
                        packetLoss = 0.004,
                        ageMillis = 20.0,
                        subframeOffsetNanos = 200_000.0,
                        subframeConfidence = 0.96,
                    ),
                    sample(
                        id = "rear",
                        offsetNanos = -1_700_000L,
                        roundTripNanos = 22_000_000L,
                        packetLoss = 0.004,
                        ageMillis = 22.0,
                        subframeOffsetNanos = 220_000.0,
                        subframeConfidence = 0.95,
                    ),
                ),
            )

        assertFalse(result.precisionQualified)
        assertTrue(result.precisionBlockers.isNotEmpty())
        assertTrue(
            result.precisionBlockers.any {
                it.contains("one-millisecond")
            },
        )
        assertEquals(
            MultiCameraSynchronizationQualityEngine.State.UNSAFE,
            result.state,
        )
    }

    @Test
    fun precisionContractRequiresStableQualifiedWindow() {
        val engine =
            MultiCameraSynchronizationQualityEngine(
                MultiCameraSynchronizationQualityEngine.Config(
                    precisionContractEnabled = true,
                    precisionMinimumStableEvaluations = 3,
                ),
            )

        fun cleanResult(time: Long) =
            engine.evaluate(
                timestampNanos = time,
                samples = listOf(
                    sample(
                        id = "front",
                        offsetNanos = 120_000L,
                        roundTripNanos = 18_000_000L,
                        packetLoss = 0.001,
                        ageMillis = 12.0,
                        subframeOffsetNanos = 120_000.0,
                        subframeConfidence = 0.98,
                    ),
                    sample(
                        id = "rear",
                        offsetNanos = -140_000L,
                        roundTripNanos = 19_000_000L,
                        packetLoss = 0.001,
                        ageMillis = 13.0,
                        subframeOffsetNanos = 100_000.0,
                        subframeConfidence = 0.98,
                    ),
                ),
            )

        val first = cleanResult(7_000_000_000L)
        val second = cleanResult(8_000_000_000L)
        val third = cleanResult(9_000_000_000L)

        assertFalse(first.precisionQualified)
        assertFalse(second.precisionQualified)
        assertTrue(third.precisionQualified)
        assertEquals(3, third.precisionQualifiedStreak)
        assertTrue(third.maximumClockDriftPpm <= 80.0)
    }

    @Test
    fun outOfOrderTimestampDoesNotCorruptClockDriftHistory() {
        val engine =
            MultiCameraSynchronizationQualityEngine(
                MultiCameraSynchronizationQualityEngine.Config(
                    precisionContractEnabled = true,
                ),
            )

        engine.evaluate(
            timestampNanos = 10_000_000_000L,
            samples = listOf(
                sample("front", 120_000L, 18_000_000L, 0.001, 12.0),
                sample("rear", -140_000L, 19_000_000L, 0.001, 13.0),
            ),
        )
        val result = engine.evaluate(
            timestampNanos = 9_000_000_000L,
            samples = listOf(
                sample("front", 200_000L, 18_000_000L, 0.001, 12.0),
                sample("rear", -210_000L, 19_000_000L, 0.001, 13.0),
            ),
        )

        assertTrue(
            result.peers.any { peer ->
                peer.warnings.any { it == "out-of-order-timestamp" }
            },
        )
    }

    @Test
    fun resetClearsPeerHistory() {
        val engine = MultiCameraSynchronizationQualityEngine()

        engine.evaluate(
            timestampNanos = 3_000_000_000L,
            samples = listOf(
                sample("front", 12_000_000L, 80_000_000L, 0.08, 300.0),
            ),
        )

        engine.reset()

        val result = engine.evaluate(
            timestampNanos = 4_000_000_000L,
            samples = emptyList(),
        )

        assertEquals(
            MultiCameraSynchronizationQualityEngine.State.NO_PEERS,
            result.state,
        )
        assertEquals(1.0, result.confidence, 0.0001)
    }

    private fun sample(
        id: String,
        offsetNanos: Long,
        roundTripNanos: Long,
        packetLoss: Double,
        ageMillis: Double,
        subframeOffsetNanos: Double = 0.0,
        subframeConfidence: Double = 1.0,
    ) =
        MultiCameraSynchronizationQualityEngine.PeerSample(
            deviceId = DeviceId("nox-$id"),
            offsetNanos = offsetNanos,
            roundTripNanos = roundTripNanos,
            packetLoss = packetLoss,
            sampleAgeMillis = ageMillis,
            subframeOffsetNanos =
                subframeOffsetNanos,
            subframeConfidence =
                subframeConfidence,
        )
    @Test
    fun precisionModeRejectsNonConsistentMultiMasterGraph() {
        val graph =
            DistributedClockAuthorityEngine.Result(
                offsets = emptyMap(),
                estimates = emptyList(),
                authorityNodes = listOf(
                    DeviceId("a"),
                ),
                cycleConsistency = 0.62,
                graphCoverage = 0.98,
                residualP95Nanos = 8_000_000.0,
                confidence = 0.71,
                accepted = false,
                warnings = listOf(
                    "CLOCK_GRAPH_CYCLE_INCONSISTENCY",
                ),
            )

        val result =
            MultiCameraSynchronizationQualityEngine(
                MultiCameraSynchronizationQualityEngine.Config(
                    precisionContractEnabled = true,
                    precisionMinimumStableEvaluations = 1,
                ),
            ).evaluate(
                timestampNanos = 2_000_000_000L,
                samples =
                    listOf(
                        PeerSample(
                            deviceId = DeviceId("a"),
                            offsetNanos = 100_000L,
                            roundTripNanos = 5_000_000L,
                            packetLoss = 0.0,
                            sampleAgeMillis = 10.0,
                        ),
                        PeerSample(
                            deviceId = DeviceId("b"),
                            offsetNanos = 120_000L,
                            roundTripNanos = 5_000_000L,
                            packetLoss = 0.0,
                            sampleAgeMillis = 10.0,
                        ),
                    ),
                clockGraph = graph,
            )

        assertFalse(result.precisionQualified)
        assertTrue(
            result.precisionBlockers.contains(
                "precision-multi-master-clock-consensus-below-contract",
            ),
        )
    }

    @Test
    fun absentClockGraphPreservesStandardSynchronizationMode() {
        val result =
            MultiCameraSynchronizationQualityEngine().evaluate(
                timestampNanos = 3_000_000_000L,
                samples =
                    listOf(
                        PeerSample(
                            deviceId = DeviceId("a"),
                            offsetNanos = 100_000L,
                            roundTripNanos = 5_000_000L,
                            packetLoss = 0.0,
                            sampleAgeMillis = 10.0,
                        ),
                        PeerSample(
                            deviceId = DeviceId("b"),
                            offsetNanos = 120_000L,
                            roundTripNanos = 5_000_000L,
                            packetLoss = 0.0,
                            sampleAgeMillis = 10.0,
                        ),
                    ),
            )

        assertTrue(result.precisionQualified)
        assertTrue(result.clockGraphConfidence == null)
    }

}
