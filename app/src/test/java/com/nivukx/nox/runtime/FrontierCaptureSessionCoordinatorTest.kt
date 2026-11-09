package com.nivukx.nox.runtime

import com.nivukx.nox.core.ai.FrontierMotionCaptureCoordinator
import com.nivukx.nox.core.reconstruction.FrontierReconstructionOrchestrator
import com.nivukx.nox.core.network.FrontierCameraTransportBudgetEngine
import java.nio.file.Files
import java.nio.file.Path
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierCaptureSessionCoordinatorTest {
    @Test
    fun underfundedTransportMarksSessionTransportLimited() {
        val motionInput = FrontierMotionCaptureCoordinator.Input(
            timestampNanos = 1_000L,
            rayObservations = emptyList(),
        )
        val channels = (0 until 32).map { id ->
            FrontierCameraTransportBudgetEngine.CameraChannel(
                cameraId = "c$id",
                authoritative = true,
                quality = 0.90,
                packetLoss = 0.01,
                latencyMillis = 10.0,
                requestedKbps = 10_000.0,
            )
        }
        val result = FrontierCaptureSessionCoordinator().process(
            FrontierCaptureSessionCoordinator.Input(
                timestampNanos = 1_000L,
                motion = motionInput,
                transportChannels = channels,
                linkCapacityKbps = 10_000.0,
            ),
        )
        assertEquals(
            FrontierCaptureSessionCoordinator.SessionState.TRANSPORT_LIMITED,
            result.state,
        )
        assertTrue(result.transport.budget.authoritativeUnderfunded > 0)
    }
    @Test
    fun weakDistributedRigBlocksAuthorityAtSessionBoundary() {
        val root = FrontierCaptureSessionCoordinator()
        val rays = listOf(
            com.nivukx.nox.core.ai.WorldSpaceMotionProposalEngine.RayObservation(
                actorId = 1,
                pointId = 7,
                cameraId = "c0",
                timestampNanos = 2_000L,
                rayOriginMeters = com.nivukx.nox.core.model.Vec3(-0.5,0.0,0.0),
                rayDirection = com.nivukx.nox.core.model.Vec3(0.4472135955,0.0,0.8944271910),
                pixelConfidence = 0.99,
                timingConfidence = 0.99,
                geometricUncertaintyMeters = 0.01,
            ),
            com.nivukx.nox.core.ai.WorldSpaceMotionProposalEngine.RayObservation(
                actorId = 1,
                pointId = 7,
                cameraId = "c1",
                timestampNanos = 2_000L,
                rayOriginMeters = com.nivukx.nox.core.model.Vec3(0.5,0.0,0.0),
                rayDirection = com.nivukx.nox.core.model.Vec3(-0.4472135955,0.0,0.8944271910),
                pixelConfidence = 0.99,
                timingConfidence = 0.99,
                geometricUncertaintyMeters = 0.01,
            ),
        )
        val rig = FrontierCaptureSessionCoordinator.RigEvidenceInput(
            cameras = listOf(
                com.nivukx.nox.core.sync.FrontierRigEvidenceCoordinator.CameraInput(
                    evidence =
                        com.nivukx.nox.core.sync.FrontierCameraAuthorityFederationEngine.CameraEvidence(
                            cameraId = "c0",
                            sector = 0,
                            health = 0.95,
                            synchronization = 0.95,
                            calibration = 0.95,
                            coverage = 0.90,
                            actorVisibility = 0.95,
                            thermalHeadroom = 0.95,
                            networkHeadroom = 0.95,
                            freshness = 1.0,
                        ),
                    peerSample = null,
                ),
            ),
            actorIds = setOf(1),
        )
        val result = root.process(
            FrontierCaptureSessionCoordinator.Input(
                timestampNanos = 2_000L,
                motion = FrontierMotionCaptureCoordinator.Input(
                    timestampNanos = 2_000L,
                    rayObservations = rays,
                ),
                transportChannels = emptyList(),
                linkCapacityKbps = 1_000.0,
                rigEvidence = rig,
            ),
        )
        assertEquals(
            FrontierCaptureSessionCoordinator.SessionState.EVIDENCE_LIMITED,
            result.state,
        )
        assertTrue(result.rigEvidence != null)
        assertTrue(!result.rigEvidence!!.authoritativeReady)
        root.close()
    }
    @Test
    fun closeReleasesInjectedReconstructionAndIsIdempotent() {
        val reconstruction = FrontierReconstructionOrchestrator()
        val rigEvidence = com.nivukx.nox.core.sync.FrontierRigEvidenceCoordinator()
        val transport = com.nivukx.nox.core.network.FrontierDistributedTransportOrchestrator()
        val coordinator =
            FrontierCaptureSessionCoordinator(
                reconstruction = reconstruction,
                rigEvidenceAuthority = rigEvidence,
                transport = transport,
            )

        coordinator.close()
        coordinator.close()

        assertThrows(IllegalStateException::class.java) {
            reconstruction.process(
                FrontierReconstructionOrchestrator.Input(
                    timestampNanos = 3_000L,
                    regions = emptyList(),
                    surfacePoints = emptyList(),
                    globalCameras = emptyList(),
                    globalPoints = emptyList(),
                ),
            )
        }
        assertThrows(IllegalStateException::class.java) {
            rigEvidence.configure(
                com.nivukx.nox.core.model.HighEndMotionCaptureSettings(),
            )
        }
        assertThrows(IllegalStateException::class.java) {
            transport.peerQuality(
                endpoint = "peer-after-close",
                nowNanos = 4_000L,
            )
        }
    }

    @Test
    fun everySessionResultCarriesDecisionTrace() {
        val coordinator = FrontierCaptureSessionCoordinator()
        val motionInput = FrontierMotionCaptureCoordinator.Input(
            timestampNanos = 2_000L,
            rayObservations = emptyList(),
        )
        val result = coordinator.process(
            FrontierCaptureSessionCoordinator.Input(
                timestampNanos = 2_000L,
                motion = motionInput,
                transportChannels = listOf(
                    FrontierCameraTransportBudgetEngine.CameraChannel(
                        cameraId = "c0",
                        authoritative = true,
                        quality = 0.95,
                        packetLoss = 0.01,
                        latencyMillis = 10.0,
                        requestedKbps = 2_000.0,
                    ),
                ),
                linkCapacityKbps = 5_000.0,
            ),
        )
        assertTrue(result.decisionTrace.sequence > 0L)
        assertEquals(64, result.decisionTrace.evidenceDigest.length)
        assertTrue(coordinator.decisionLedgerDigest().isNotBlank())
    }

    @Test
    fun depthFrameSubmissionsAreComputedBeforeCheckpointStageSelection() {
        val candidates = listOf(
            Path.of("src/main/java/com/nivukx/nox/runtime/FrontierCaptureSessionCoordinator.kt"),
            Path.of("app/src/main/java/com/nivukx/nox/runtime/FrontierCaptureSessionCoordinator.kt"),
        )
        val source = candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("FrontierCaptureSessionCoordinator.kt was not found")

        val submission = source.indexOf("val depthFrameSubmissions =")
        val checkpoint = source.indexOf("val shouldCheckpoint =")
        val checkpointUse = source.indexOf("depthFrameSubmissions.any")

        assertTrue(submission >= 0)
        assertTrue(checkpoint > submission)
        assertTrue(checkpointUse > checkpoint)
    }

}