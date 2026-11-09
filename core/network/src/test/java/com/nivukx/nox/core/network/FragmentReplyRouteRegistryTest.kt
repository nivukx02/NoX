package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals

class FragmentReplyRouteRegistryTest {

    @Test
    fun reverseRouteIsStoredWithExpiry() {
        val registry =
            FragmentReplyRouteRegistry()

        val route =
            listOf(
                DeviceId("relay-b"),
                DeviceId("relay-a"),
                DeviceId("cam"),
            )

        registry.remember(
            endpoint = "destination",
            evidenceType =
                NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
            streamId = "stream",
            frameSequence = 10L,
            route = route,
            nowNanos = 100L,
        )

        assertEquals(
            route,
            registry.find(
                endpoint = "destination",
                evidenceType =
                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                streamId = "stream",
                frameSequence = 10L,
                nowNanos = 101L,
            ),
        )

        registry.expire(500_000_001L)

        assertEquals(
            null,
            registry.find(
                endpoint = "destination",
                evidenceType =
                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                streamId = "stream",
                frameSequence = 10L,
                nowNanos = 500_000_002L,
            ),
        )
    }
    @Test
    fun routeRegistryPreservesRelayOrderForReturnPath() {
        val registry =
            FragmentReplyRouteRegistry()

        val reverseRoute =
            listOf(
                DeviceId("relay-b"),
                DeviceId("relay-a"),
                DeviceId("cam-source"),
            )

        registry.remember(
            endpoint = "cam-source",
            evidenceType =
                NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
            streamId = "pose",
            frameSequence = 20L,
            route = reverseRoute,
            nowNanos = 1_000L,
        )

        assertEquals(
            listOf(
                DeviceId("relay-b"),
                DeviceId("relay-a"),
                DeviceId("cam-source"),
            ),
            registry.find(
                endpoint = "cam-source",
                evidenceType =
                    NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                streamId = "pose",
                frameSequence = 20L,
                nowNanos = 1_001L,
            ),
        )
    }


}
