package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertTrue

class DistributedFragmentTransportTest {

    @Test
    fun routeBudgetLeavesRoomForNestedDenseFragment() {
        val origin = DeviceId("origin-camera")
        val destination = DeviceId("destination-camera")
        val route =
            listOf(
                origin,
                DeviceId("relay-01"),
                DeviceId("relay-02"),
                DeviceId("relay-03"),
                destination,
            )

        val routeBudget =
            DistributedTransportEnvelopeCodec
                .maxPayloadBytesForRoute(
                    origin = origin,
                    destination = destination,
                    route = route,
                )

        val fragmentBudget =
            routeBudget -
                36 -
                28 -
                96

        assertTrue(routeBudget > 4_000)
        assertTrue(fragmentBudget > 60_000)

        val frame =
            com.nivukx.nox.core.network.DenseLandmarkWireCodec.Frame(
                timestampNanos = 14_000_000_000L,
                landmarks =
                    (0 until 2_000).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 33,
                            landmarkId = index % 33,
                            normalizedX = 0.3f,
                            normalizedY = 0.4f,
                            confidence = 0.95f,
                            uncertainty = 0.05f,
                            visibility = 1.0f,
                            contactProbability = 0.0f,
                            temporalStability = 0.96f,
                            crossViewAgreement = 0.94f,
                            reprojectionResidualNormalized = 0.02f,
                            predicted = false,
                        )
                    },
            )

        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame = frame,
                streamId = "x".repeat(96),
                frameSequence = 123L,
                maximumPayloadBytes = fragmentBudget,
            )

        assertTrue(
            chunks.all {
                it.payload.size <= fragmentBudget
            },
        )
    }
}
