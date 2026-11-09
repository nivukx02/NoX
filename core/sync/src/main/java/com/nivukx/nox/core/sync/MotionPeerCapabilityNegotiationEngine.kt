
package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.model.DepthRepresentation

/**
 * Negotiates a common capture contract across heterogeneous Android devices.
 *
 * The negotiation never enables a feature that one peer cannot satisfy. Unsupported metric depth
 * remains explicitly relative, and motion-only persistence stays mandatory for every accepted peer.
 */
class MotionPeerCapabilityNegotiationEngine {

    data class Peer(
        val id: String,
        val protocolVersion: Int,
        val sessionSchema: Int,
        val supportedModes: Set<MotionCaptureMode>,
        val maximumCaptureFps: Int,
        val maximumInferenceFps: Int,
        val supportedDepth: Set<DepthRepresentation>,
        val supportsFace: Boolean,
        val supportsEyes: Boolean,
        val supportsHands: Boolean,
        val supportsDenseMesh: Boolean,
        val supportsMotionOnlyExport: Boolean,
        val maxSubjects: Int,
        val timingAuthority: Double = 0.0,
        val supportsVisualTimingBeacon: Boolean = false,
        val supportsDenseLandmarkFragmentation: Boolean = false,
        val supportsMultiActorPoseFragmentation: Boolean = false,
    )

    data class Contract(
        val protocolVersion: Int,
        val sessionSchema: Int,
        val mode: MotionCaptureMode,
        val captureFps: Int,
        val inferenceFps: Int,
        val depthRepresentation: DepthRepresentation?,
        val face: Boolean,
        val eyes: Boolean,
        val hands: Boolean,
        val denseMesh: Boolean,
        val motionOnlyExport: Boolean,
        val maxSubjects: Int,
        val timingAuthority: Double = 0.0,
        val fragmentedEvidence: Boolean = false,
        val visualTimingBeacon: Boolean = false,
    )

    data class Rejection(
        val peerId: String,
        val reasons: List<String>,
    )

    data class Result(
        val accepted: Boolean,
        val contract: Contract?,
        val peers: List<String>,
        val rejections: List<Rejection>,
        val diagnostics: List<String>,
    )

    fun negotiate(
        peers: List<Peer>,
        requestedMode: MotionCaptureMode,
        requestedCaptureFps: Int,
        requestedInferenceFps: Int,
        preferMetricDepth: Boolean = true,
        requestedSubjects: Int = 1,
        requireHighPrecisionTimingAuthority: Boolean = false,
        requireFragmentedEvidence: Boolean = false,
    ): Result {
        if (peers.isEmpty()) {
            return Result(
                accepted = false,
                contract = null,
                peers = emptyList(),
                rejections = emptyList(),
                diagnostics = listOf("no peers supplied"),
            )
        }

        val rejections = peers.mapNotNull { peer ->
            val reasons = mutableListOf<String>()

            if (requestedMode !in peer.supportedModes) {
                reasons += "requested capture mode unsupported"
            }

            if (peer.protocolVersion < 1) {
                reasons += "unsupported protocol"
            }

            if (peer.sessionSchema < 10) {
                reasons += "motion session schema below production floor"
            }

            if (!peer.supportsMotionOnlyExport) {
                reasons += "peer cannot satisfy motion-only export contract"
            }

            if (peer.maxSubjects < requestedSubjects) {
                reasons += "peer supports fewer subjects than requested"
            }

            if (
                requireHighPrecisionTimingAuthority &&
                peer.timingAuthority < 0.90
            ) {
                reasons +=
                    "peer timing authority is below the reference threshold"
            }

            if (
                requireHighPrecisionTimingAuthority &&
                !peer.supportsVisualTimingBeacon &&
                peer.timingAuthority < 0.95
            ) {
                reasons +=
                    "peer lacks an independent visual timing authority source"
            }

            if (
                requireFragmentedEvidence &&
                (
                    !peer.supportsDenseLandmarkFragmentation ||
                        !peer.supportsMultiActorPoseFragmentation
                )
            ) {
                reasons +=
                    "peer cannot transport fragmented high-density evidence"
            }

            if (reasons.isNotEmpty()) {
                Rejection(
                    peerId = peer.id,
                    reasons = reasons,
                )
            } else {
                null
            }
        }

        if (rejections.isNotEmpty()) {
            return Result(
                accepted = false,
                contract = null,
                peers = peers.map { it.id },
                rejections = rejections,
                diagnostics = listOf(
                    "acceptedPeers=0",
                    "rejections=" + rejections.size,
                ),
            )
        }

        val protocol = peers.minOf { it.protocolVersion }
        val schema = peers.minOf { it.sessionSchema }

        val captureFps = minOf(
            requestedCaptureFps.coerceIn(15, 240),
            peers.minOf { it.maximumCaptureFps.coerceAtLeast(15) },
        )

        val inferenceFps = minOf(
            requestedInferenceFps.coerceIn(5, 240),
            captureFps,
            peers.minOf { it.maximumInferenceFps.coerceAtLeast(5) },
        )

        val commonDepth =
            peers.map { it.supportedDepth }
                .reduce { acc, next -> acc.intersect(next) }

        val depth = when {
            commonDepth.isEmpty() -> null
            preferMetricDepth &&
                DepthRepresentation.METRIC_METERS in commonDepth ->
                DepthRepresentation.METRIC_METERS
            DepthRepresentation.RELATIVE_ORDERED_DEPTH in commonDepth ->
                DepthRepresentation.RELATIVE_ORDERED_DEPTH
            else ->
                commonDepth.first()
        }

        val contract = Contract(
            protocolVersion = protocol,
            sessionSchema = schema,
            mode = requestedMode,
            captureFps = captureFps,
            inferenceFps = inferenceFps,
            depthRepresentation = depth,
            face = peers.all { it.supportsFace },
            eyes = peers.all { it.supportsEyes },
            hands = peers.all { it.supportsHands },
            denseMesh = peers.all { it.supportsDenseMesh },
            motionOnlyExport = peers.all { it.supportsMotionOnlyExport },
            maxSubjects = minOf(
                requestedSubjects.coerceAtLeast(1),
                peers.minOf { it.maxSubjects.coerceAtLeast(1) },
            ),
            timingAuthority =
                peers.minOf {
                    it.timingAuthority
                        .coerceIn(0.0, 1.0)
                },
            fragmentedEvidence =
                peers.all {
                    it.supportsDenseLandmarkFragmentation &&
                        it.supportsMultiActorPoseFragmentation
                },
            visualTimingBeacon =
                peers.all {
                    it.supportsVisualTimingBeacon
                },
        )

        return Result(
            accepted = true,
            contract = contract,
            peers = peers.map { it.id },
            rejections = emptyList(),
            diagnostics = listOf(
                "protocolVersion=" + protocol,
                "sessionSchema=" + schema,
                "captureFps=" + captureFps,
                "inferenceFps=" + inferenceFps,
                "depth=" + (depth?.name ?: "none"),
                "face=" + contract.face,
                "eyes=" + contract.eyes,
                "hands=" + contract.hands,
                "denseMesh=" + contract.denseMesh,
                "maxSubjects=" + contract.maxSubjects,
                "timingAuthority=" + contract.timingAuthority,
                "fragmentedEvidence=" + contract.fragmentedEvidence,
                "visualTimingBeacon=" + contract.visualTimingBeacon,
                "motionOnly=" + contract.motionOnlyExport,
            ),
        )
    }
}
