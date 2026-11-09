package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DepthRepresentation
import com.nivukx.nox.core.model.MotionCaptureMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionPeerCapabilityNegotiationEngineTest {

    @Test
    fun negotiatesCommonContract() {
        val engine = MotionPeerCapabilityNegotiationEngine()
        val result = engine.negotiate(
            peers = listOf(peer("A"), peer("B", maxFps = 90)),
            requestedMode = MotionCaptureMode.MULTI_VIEW_RGB,
            requestedCaptureFps = 60,
            requestedInferenceFps = 60,
            preferMetricDepth = true,
            requestedSubjects = 2,
        )

        assertTrue(result.accepted)
        assertEquals(60, result.contract?.captureFps)
        assertEquals(60, result.contract?.inferenceFps)
        assertEquals(
            DepthRepresentation.METRIC_METERS,
            result.contract?.depthRepresentation,
        )
        assertEquals(2, result.contract?.maxSubjects)
    }

    @Test
    fun rejectsPeerThatCannotPersistMotionOnlyOutput() {
        val engine = MotionPeerCapabilityNegotiationEngine()
        val result = engine.negotiate(
            peers = listOf(
                peer("A"),
                peer(
                    id = "B",
                    motionOnly = false,
                ),
            ),
            requestedMode = MotionCaptureMode.MULTI_VIEW_RGB,
            requestedCaptureFps = 30,
            requestedInferenceFps = 30,
        )

        assertFalse(result.accepted)
        assertTrue(result.rejections.isNotEmpty())
    }

    private fun peer(
        id: String,
        maxFps: Int = 60,
        motionOnly: Boolean = true,
        timingAuthority: Double = 0.96,
        visualTimingBeacon: Boolean = true,
    ) =
        MotionPeerCapabilityNegotiationEngine.Peer(
            id = id,
            protocolVersion = 2,
            sessionSchema = 10,
            supportedModes = setOf(
                MotionCaptureMode.MONOCULAR_RGB,
                MotionCaptureMode.MULTI_VIEW_RGB,
            ),
            maximumCaptureFps = maxFps,
            maximumInferenceFps = maxFps,
            supportedDepth = setOf(
                DepthRepresentation.METRIC_METERS,
                DepthRepresentation.RELATIVE_ORDERED_DEPTH,
            ),
            supportsFace = true,
            supportsEyes = true,
            supportsHands = true,
            supportsDenseMesh = true,
            supportsMotionOnlyExport = motionOnly,
            maxSubjects = 4,
            timingAuthority = timingAuthority,
            supportsVisualTimingBeacon = visualTimingBeacon,
            supportsDenseLandmarkFragmentation = true,
            supportsMultiActorPoseFragmentation = true,
        )
    @Test
    fun referenceGradeNegotiationRejectsPeerWithoutTimingAuthority() {
        val engine = MotionPeerCapabilityNegotiationEngine()

        val result =
            engine.negotiate(
                peers =
                    listOf(
                        peer(
                            "A",
                            timingAuthority = 0.97,
                            visualTimingBeacon = true,
                        ),
                        peer(
                            "B",
                            timingAuthority = 0.42,
                            visualTimingBeacon = false,
                        ),
                    ),
                requestedMode =
                    MotionCaptureMode.MULTI_VIEW_RGB,
                requestedCaptureFps = 120,
                requestedInferenceFps = 120,
                requestedSubjects = 50,
                requireHighPrecisionTimingAuthority = true,
                requireFragmentedEvidence = true,
            )

        assertTrue(!result.accepted)
        assertTrue(
            result.rejections.any {
                it.peerId == "B" &&
                    it.reasons.any { reason ->
                        reason.contains("timing authority")
                    }
            },
        )
    }

    @Test
    fun distributedHighEndContractNegotiatesFragmentedCrowdEvidence() {
        val engine = MotionPeerCapabilityNegotiationEngine()

        val result =
            engine.negotiate(
                peers =
                    listOf(
                        peer(
                            "A",
                            timingAuthority = 0.96,
                            visualTimingBeacon = true,
                            maxSubjects = 64,
                        ),
                        peer(
                            "B",
                            timingAuthority = 0.94,
                            visualTimingBeacon = true,
                            maxSubjects = 64,
                        ),
                    ),
                requestedMode =
                    MotionCaptureMode.MULTI_VIEW_RGB,
                requestedCaptureFps = 120,
                requestedInferenceFps = 120,
                requestedSubjects = 50,
                requireHighPrecisionTimingAuthority = true,
                requireFragmentedEvidence = true,
            )

        assertTrue(result.accepted)
        assertTrue(
            result.contract?.fragmentedEvidence == true,
        )
        assertTrue(
            result.contract?.timingAuthority ?: 0.0 >= 0.94,
        )
    }

}
