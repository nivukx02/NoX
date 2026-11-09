package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DistributedClockAuthorityEngineTest {

    @Test
    fun redundantAnchorGraphProducesStableOffsets() {
        val engine =
            DistributedClockAuthorityEngine()

        val result =
            engine.solve(
                nodes =
                    listOf(
                        node("master", 0.99),
                        node("cam-a", 0.96),
                        node("cam-b", 0.95),
                        node("cam-c", 0.93),
                    ),
                edges =
                    listOf(
                        edge("master", "cam-a", 2_000_000L, 0.98),
                        edge("master", "cam-b", -1_000_000L, 0.97),
                        edge("cam-a", "cam-c", 1_000_000L, 0.96),
                        edge("cam-b", "cam-c", 4_000_000L, 0.95),
                        edge("master", "cam-c", 3_000_000L, 0.98),
                    ),
            )

        assertTrue(result.accepted)
        assertTrue(result.confidence > 0.85)
        assertEquals(
            setOf(
                DeviceId("master"),
                DeviceId("cam-a"),
                DeviceId("cam-b"),
                DeviceId("cam-c"),
            ),
            result.offsets.keys,
        )
        assertTrue(result.cycleConsistency > 0.90)
    }

    @Test
    fun inconsistentClockCycleIsRejectedInsteadOfAveragedAway() {
        val engine =
            DistributedClockAuthorityEngine(
                DistributedClockAuthorityEngine.Config(
                    maximumCycleResidualNanos = 2_000_000L,
                    minimumCycleConsistency = 0.85,
                ),
            )

        val result =
            engine.solve(
                nodes =
                    listOf(
                        node("master", 0.99),
                        node("a", 0.96),
                        node("b", 0.96),
                    ),
                edges =
                    listOf(
                        edge("master", "a", 2_000_000L, 0.98),
                        edge("a", "b", -3_000_000L, 0.98),
                        edge("b", "master", 4_500_000L, 0.98),
                    ),
            )

        assertTrue(!result.accepted)
        assertTrue(
            result.warnings.contains(
                "CLOCK_GRAPH_CYCLE_INCONSISTENCY",
            ),
        )
    }

    @Test
    fun weakAnchorIsNotAllowedToBecomeAuthority() {
        val result =
            DistributedClockAuthorityEngine(
                DistributedClockAuthorityEngine.Config(
                    minimumNodeAuthority = 0.80,
                ),
            ).solve(
                nodes =
                    listOf(
                        node("weak", 0.40),
                        node("cam", 0.95),
                    ),
                edges =
                    listOf(
                        edge("weak", "cam", 1_000_000L, 0.95),
                    ),
            )

        assertEquals(
            listOf(DeviceId("cam")),
            result.authorityNodes,
        )
        assertTrue(
            result.warnings.contains(
                "CLOCK_GRAPH_SINGLE_AUTHORITY",
            ),
        )
    }

    private fun node(
        id: String,
        authority: Double,
    ) =
        DistributedClockAuthorityEngine.Node(
            deviceId = DeviceId(id),
            timingAuthority = authority,
        )

    private fun edge(
        from: String,
        to: String,
        offsetNanos: Long,
        confidence: Double,
    ) =
        DistributedClockAuthorityEngine.Edge(
            from = DeviceId(from),
            to = DeviceId(to),
            offsetNanos = offsetNanos,
            residualNanos = 400_000L,
            confidence = confidence,
        )
    @Test
    fun hardwareAndBeaconBackedNodeRanksAboveUnknownClockSource() {
        val engine =
            DistributedClockAuthorityEngine(
                DistributedClockAuthorityEngine.Config(
                    minimumNodeAuthority = 0.50,
                ),
            )

        val nodes =
            listOf(
                DistributedClockAuthorityEngine.Node(
                    deviceId = com.nivukx.nox.core.model.DeviceId("hw"),
                    timingAuthority = 0.94,
                    timestampDomain =
                        MultiCameraTemporalAlignmentEngine.TimestampDomain.SENSOR,
                    hardwareSynchronized = true,
                    visualBeaconConfidence = 0.96,
                ),
                DistributedClockAuthorityEngine.Node(
                    deviceId = com.nivukx.nox.core.model.DeviceId("unknown"),
                    timingAuthority = 0.94,
                    timestampDomain =
                        MultiCameraTemporalAlignmentEngine.TimestampDomain.UNKNOWN,
                    hardwareSynchronized = false,
                    visualBeaconConfidence = 0.0,
                ),
            )

        val edges =
            listOf(
                DistributedClockAuthorityEngine.Edge(
                    from = nodes[0].deviceId,
                    to = nodes[1].deviceId,
                    offsetNanos = 1_000L,
                    confidence = 0.98,
                ),
            )

        val result = engine.solve(nodes, edges)

        val hw =
            result.estimates.first {
                it.deviceId == com.nivukx.nox.core.model.DeviceId("hw")
            }
        val unknown =
            result.estimates.first {
                it.deviceId ==
                    com.nivukx.nox.core.model.DeviceId("unknown")
            }

        assertTrue(hw.authority > unknown.authority)
    }

}
