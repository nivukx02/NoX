package com.nivukx.nox.core.network

/**
 * Semantic congestion policy for distributed evidence transport.
 *
 * Control and timing-authority traffic is kept unshed at the congestion boundary. Reconstruction
 * evidence may additionally declare itself critical when the operator's quality contract requires
 * lossless delivery. Ordinary high-rate perception streams remain shed-able under pressure so a
 * degraded peer cannot exhaust the queue.
 */
class PeerTransportSheddingPolicy(
    private val maximumCriticalOutstanding: Int = 512,
    private val maximumHighOutstanding: Int = 192,
    private val maximumNormalOutstanding: Int = 96,
    private val maximumLowOutstanding: Int = 48,
    private val smallPacketBytes: Int = 4 * 1024,
) {

    enum class Class {
        CRITICAL,
        HIGH,
        NORMAL,
        LOW,
    }

    init {
        require(maximumCriticalOutstanding > 0)
        require(maximumHighOutstanding > 0)
        require(maximumNormalOutstanding > 0)
        require(maximumLowOutstanding > 0)
        require(smallPacketBytes in 1024..16 * 1024)
    }

    fun classify(
        type: NoXWireProtocol.Type,
    ): Class =
        when (type) {
            NoXWireProtocol.Type.HELLO,
            NoXWireProtocol.Type.PING,
            NoXWireProtocol.Type.PONG,
            NoXWireProtocol.Type.START,
            NoXWireProtocol.Type.STOP,
            NoXWireProtocol.Type.ERROR,
            NoXWireProtocol.Type.TIMING_BEACON_OBSERVATION,
            NoXWireProtocol.Type.EVIDENCE_FRAGMENT_FEEDBACK ->
                Class.CRITICAL

            NoXWireProtocol.Type.HEARTBEAT,
            NoXWireProtocol.Type.CALIBRATION,
            NoXWireProtocol.Type.CALIBRATION_RESULT,
            NoXWireProtocol.Type.DEVICE_DESCRIPTOR,
            NoXWireProtocol.Type.CAPTURE_ACK ->
                Class.HIGH

            NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
            NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
            NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
            NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAME,
            NoXWireProtocol.Type.FACE_FRAME,
            NoXWireProtocol.Type.EYE_FRAME,
            NoXWireProtocol.Type.HAND_FRAME,
            NoXWireProtocol.Type.POSE_FRAME ->
                Class.NORMAL

            NoXWireProtocol.Type.DEPTH_FRAGMENT,
            NoXWireProtocol.Type.DEPTH_FRAME,
            NoXWireProtocol.Type.CALIBRATION_FRAME,
            NoXWireProtocol.Type.FACE_CONTOUR_FRAME ->
                Class.LOW

            else ->
                Class.NORMAL
        }

    fun shouldShed(
        type: NoXWireProtocol.Type,
        packetBytes: Int,
        outstandingPayloads: Int,
        reconstructionCritical: Boolean = false,
    ): Boolean {
        require(packetBytes >= 0)
        require(outstandingPayloads >= 0)

        if (
            reconstructionCritical ||
                packetBytes <= smallPacketBytes
        ) {
            return false
        }

        if (classify(type) == Class.CRITICAL) {
            return false
        }

        val limit =
            when (classify(type)) {
                Class.HIGH ->
                    maximumHighOutstanding
                Class.NORMAL ->
                    maximumNormalOutstanding
                Class.LOW ->
                    maximumLowOutstanding
                Class.CRITICAL ->
                    maximumCriticalOutstanding
            }

        return outstandingPayloads >= limit
    }
}
