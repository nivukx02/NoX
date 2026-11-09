package com.nivukx.nox.core.network

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/**
 * Compact binary transport for the complete admitted multi-actor pose graph.
 *
 * Version 3 carries:
 * - bounded 2D body joints;
 * - learned/body-space 3D landmarks used as reconstruction priors;
 * - short-term prediction confidence and innovation diagnostics;
 * - deterministic assignment provenance.
 *
 * The semantic payload supports up to sixty-four actors. A fully populated fifty-actor frame with
 * 2D, world-space and appearance evidence may exceed one physical transport packet, so callers
 * must use MultiActorPoseChunkedWireCodec for oversized frames. Version 1 and version 2 payloads
 * remain decodable so a peer upgrade does not require a flag-day migration.
 *
 * Body-space 3D coordinates are transport evidence only. They are not declared to be calibrated rig
 * coordinates and must not bypass the multi-view geometric reconstruction authority.
 */
object MultiActorPoseWireCodec {

    private const val CURRENT_VERSION = 3
    private const val VERSION_2 = 2
    private const val APPEARANCE_FINGERPRINT_DIMENSIONS = 16
    private const val APPEARANCE_FINGERPRINT_BYTES = 17
    private const val LEGACY_VERSION = 1
    private const val FRAME_HEADER_BYTES = 24
    private const val ACTOR_HEADER_BYTES = 51
    private const val EXTENSION_HEADER_BYTES = 16
    private const val JOINT_BYTES = 16
    private const val WORLD_JOINT_BYTES = 20
    private const val MAX_ACTORS = 64
    private const val MAX_JOINTS_PER_ACTOR = 33
    private const val MAX_WORLD_JOINTS_PER_ACTOR = 33
    private const val MAX_SOURCE_ID_BYTES = 128

    data class Joint(
        val landmarkId: Int,
        val x: Float,
        val y: Float,
        val confidence: Float,
    )

    data class WorldJoint(
        val landmarkId: Int,
        val xMeters: Float,
        val yMeters: Float,
        val zMeters: Float,
        val confidence: Float,
    )

    data class Actor(
        val actorId: Long,
        val confidence: Float,
        val predicted: Boolean,
        val missedFrames: Int,
        val centerX: Float,
        val centerY: Float,
        val boundsLeft: Float,
        val boundsTop: Float,
        val boundsRight: Float,
        val boundsBottom: Float,
        val velocityX: Float,
        val velocityY: Float,
        val joints: List<Joint>,
        val worldJoints: List<WorldJoint> = emptyList(),
        val predictionConfidence: Float = 0f,
        val meanMotionInnovationNormalized: Float = 0f,
        val acceleratedTrack: Boolean = false,
        val globalAssignmentBounded: Boolean = true,
        val appearanceFingerprint: List<Float> = emptyList(),
        val appearanceQuality: Float = 0f,
    )

    data class Frame(
        val timestampNanos: Long,
        val sourceDeviceId: String,
        val inputWidth: Int,
        val inputHeight: Int,
        val actors: List<Actor>,
    )

    fun encodedSize(frame: Frame): Int {
        val sourceLength = sourceIdBytes(frame.sourceDeviceId).size
        val actors = frame.actors
        require(
            actors.distinctBy { it.actorId }.size ==
                actors.size,
        ) {
            "Duplicate actor IDs cannot be encoded"
        }
        require(actors.size <= MAX_ACTORS) {
            "Multi-actor frame exceeds the 64-actor transport envelope"
        }
        val actorBytes =
            actors.sumOf(::encodedActorBytes)

        return FRAME_HEADER_BYTES +
            sourceLength +
            actorBytes
    }

    fun requiresFragmentation(frame: Frame): Boolean {
        require(frame.timestampNanos >= 0L)
        require(frame.inputWidth > 0)
        require(frame.inputHeight > 0)
        require(frame.sourceDeviceId.isNotBlank())
        return encodedSize(frame) >
            NoXWireProtocol.MAX_PACKET_BYTES - 64
    }
    fun encode(frame: Frame): ByteArray {
        require(frame.timestampNanos >= 0L)
        require(frame.inputWidth > 0)
        require(frame.inputHeight > 0)
        require(frame.sourceDeviceId.isNotBlank())

        val sourceBytes = sourceIdBytes(frame.sourceDeviceId)
        val actors = frame.actors
        require(
            actors.distinctBy { it.actorId }.size ==
                actors.size,
        ) {
            "Duplicate actor IDs cannot be encoded"
        }
        require(actors.size <= MAX_ACTORS) {
            "Multi-actor frame exceeds the 64-actor transport envelope"
        }

        actors.forEach(::validateActor)

        val totalBytes =
            FRAME_HEADER_BYTES +
                sourceBytes.size +
                actors.sumOf(::encodedActorBytes)

        require(totalBytes <= NoXWireProtocol.MAX_PACKET_BYTES - 64) {
            "Multi-actor pose payload too large: $totalBytes bytes"
        }

        val buffer =
            ByteBuffer
                .allocate(totalBytes)
                .order(ByteOrder.BIG_ENDIAN)

        buffer.putInt(CURRENT_VERSION)
        buffer.putLong(frame.timestampNanos)
        buffer.putInt(frame.inputWidth)
        buffer.putInt(frame.inputHeight)
        buffer.putShort(actors.size.toShort())
        buffer.putShort(sourceBytes.size.toShort())
        buffer.put(sourceBytes)

        actors.forEach { actor ->
            val joints =
                actor.joints
                    .distinctBy { it.landmarkId }
                    .take(MAX_JOINTS_PER_ACTOR)
                    .sortedBy { it.landmarkId }

            val worldJoints =
                actor.worldJoints
                    .distinctBy { it.landmarkId }
                    .take(MAX_WORLD_JOINTS_PER_ACTOR)
                    .sortedBy { it.landmarkId }

            buffer.putLong(actor.actorId)
            buffer.putFloat(actor.confidence.coerceIn(0f, 1f))
            buffer.put(if (actor.predicted) 1 else 0)
            buffer.putShort(
                actor.missedFrames
                    .coerceIn(0, Short.MAX_VALUE.toInt())
                    .toShort(),
            )
            buffer.putFloat(actor.centerX)
            buffer.putFloat(actor.centerY)
            buffer.putFloat(actor.boundsLeft)
            buffer.putFloat(actor.boundsTop)
            buffer.putFloat(actor.boundsRight)
            buffer.putFloat(actor.boundsBottom)
            buffer.putFloat(actor.velocityX)
            buffer.putFloat(actor.velocityY)
            buffer.putShort(joints.size.toShort())
            buffer.putShort(0)

            buffer.putFloat(actor.predictionConfidence.coerceIn(0f, 1f))
            buffer.putFloat(actor.meanMotionInnovationNormalized.coerceAtLeast(0f))
            buffer.put(if (actor.acceleratedTrack) 1 else 0)
            buffer.put(if (actor.globalAssignmentBounded) 1 else 0)
            buffer.putShort(worldJoints.size.toShort())
            buffer.putInt(0)

            joints.forEach { joint ->
                buffer.putInt(joint.landmarkId)
                buffer.putFloat(joint.x)
                buffer.putFloat(joint.y)
                buffer.putFloat(joint.confidence.coerceIn(0f, 1f))
            }

            worldJoints.forEach { joint ->
                buffer.putInt(joint.landmarkId)
                buffer.putFloat(joint.xMeters)
                buffer.putFloat(joint.yMeters)
                buffer.putFloat(joint.zMeters)
                buffer.putFloat(joint.confidence.coerceIn(0f, 1f))
            }

            val fingerprint =
                actor.appearanceFingerprint
                    .take(APPEARANCE_FINGERPRINT_DIMENSIONS)
                    .map { it.coerceIn(0f, 1f) }

            buffer.put(
                (
                    actor.appearanceQuality
                        .coerceIn(0f, 1f) * 255f
                ).toInt()
                    .coerceIn(0, 255)
                    .toByte(),
            )

            repeat(APPEARANCE_FINGERPRINT_DIMENSIONS) { index ->
                val value =
                    fingerprint.getOrElse(index) { 0f }
                        .coerceIn(0f, 1f)
                buffer.put(
                    (
                        value * 127f
                    ).toInt()
                        .coerceIn(0, 127)
                        .toByte(),
                )
            }
        }

        return buffer.array()
    }

    fun decode(bytes: ByteArray): Frame {
        require(bytes.size >= FRAME_HEADER_BYTES + 1) {
            "Multi-actor pose payload is too small"
        }

        val buffer =
            ByteBuffer.wrap(bytes)
                .order(ByteOrder.BIG_ENDIAN)

        val version = buffer.int
        require(
            version == LEGACY_VERSION ||
                version == VERSION_2 ||
                version == CURRENT_VERSION
        ) {
            "Unsupported multi-actor pose codec version: $version"
        }

        val timestampNanos = buffer.long
        val width = buffer.int
        val height = buffer.int
        val actorCount = buffer.short.toInt().and(0xFFFF)
        val sourceLength = buffer.short.toInt().and(0xFFFF)

        require(timestampNanos >= 0L) {
            "Negative multi-actor timestamp"
        }
        require(width > 0 && height > 0)
        require(actorCount in 0..MAX_ACTORS)
        require(sourceLength in 1..MAX_SOURCE_ID_BYTES)
        require(buffer.remaining() >= sourceLength)

        val sourceDeviceId =
            ByteArray(sourceLength)
                .also(buffer::get)
                .toString(Charsets.UTF_8)

        require(sourceDeviceId.isNotBlank()) {
            "Multi-actor source device ID is blank"
        }

        val actors = ArrayList<Actor>(actorCount)

        repeat(actorCount) {
            require(buffer.remaining() >= ACTOR_HEADER_BYTES) {
                "Truncated multi-actor header"
            }

            val actorId = buffer.long
            val confidence = buffer.float
            val predictedByte = buffer.get().toInt()
            val missedFrames = buffer.short.toInt().and(0xFFFF)
            val centerX = buffer.float
            val centerY = buffer.float
            val boundsLeft = buffer.float
            val boundsTop = buffer.float
            val boundsRight = buffer.float
            val boundsBottom = buffer.float
            val velocityX = buffer.float
            val velocityY = buffer.float
            val jointCount = buffer.short.toInt().and(0xFFFF)
            buffer.short

            require(actorId >= 0L)
            require(predictedByte == 0 || predictedByte == 1)
            require(jointCount in 0..MAX_JOINTS_PER_ACTOR)

            var predictionConfidence = 0f
            var meanMotionInnovationNormalized = 0f
            var acceleratedTrack = false
            var globalAssignmentBounded = true
            var worldJointCount = 0
            var appearanceQuality = 0f
            val appearanceFingerprint =
                ArrayList<Float>(
                    APPEARANCE_FINGERPRINT_DIMENSIONS,
                )

            if (version >= 2) {
                require(buffer.remaining() >= EXTENSION_HEADER_BYTES) {
                    "Truncated multi-actor evidence extension"
                }

                predictionConfidence =
                    buffer.float
                        .coerceIn(0f, 1f)

                meanMotionInnovationNormalized =
                    buffer.float
                        .coerceAtLeast(0f)

                val acceleratedByte = buffer.get().toInt()
                val boundedByte = buffer.get().toInt()

                require(acceleratedByte == 0 || acceleratedByte == 1)
                require(boundedByte == 0 || boundedByte == 1)

                acceleratedTrack = acceleratedByte != 0
                globalAssignmentBounded = boundedByte != 0

                worldJointCount =
                    buffer.short
                        .toInt()
                        .and(0xFFFF)

                buffer.int

                require(
                    worldJointCount in
                        0..MAX_WORLD_JOINTS_PER_ACTOR,
                )
            }

            validateFinite("confidence", confidence)
            validateFinite("centerX", centerX)
            validateFinite("centerY", centerY)
            validateFinite("boundsLeft", boundsLeft)
            validateFinite("boundsTop", boundsTop)
            validateFinite("boundsRight", boundsRight)
            validateFinite("boundsBottom", boundsBottom)
            validateFinite("velocityX", velocityX)
            validateFinite("velocityY", velocityY)

            val requiredBytes =
                jointCount * JOINT_BYTES +
                    worldJointCount * WORLD_JOINT_BYTES +
                    if (version >= CURRENT_VERSION) {
                        APPEARANCE_FINGERPRINT_BYTES
                    } else {
                        0
                    }

            require(buffer.remaining() >= requiredBytes) {
                "Truncated multi-actor evidence payload"
            }

            val joints = ArrayList<Joint>(jointCount)

            repeat(jointCount) {
                val landmarkId = buffer.int
                val x = buffer.float
                val y = buffer.float
                val jointConfidence = buffer.float

                validateFinite("joint.x", x)
                validateFinite("joint.y", y)
                validateFinite("joint.confidence", jointConfidence)

                joints +=
                    Joint(
                        landmarkId = landmarkId,
                        x = x,
                        y = y,
                        confidence = jointConfidence.coerceIn(0f, 1f),
                    )
            }

            val worldJoints =
                ArrayList<WorldJoint>(worldJointCount)

            repeat(worldJointCount) {
                val landmarkId = buffer.int
                val xMeters = buffer.float
                val yMeters = buffer.float
                val zMeters = buffer.float
                val jointConfidence = buffer.float

                validateFinite("world.xMeters", xMeters)
                validateFinite("world.yMeters", yMeters)
                validateFinite("world.zMeters", zMeters)
                validateFinite("world.confidence", jointConfidence)

                worldJoints +=
                    WorldJoint(
                        landmarkId = landmarkId,
                        xMeters = xMeters,
                        yMeters = yMeters,
                        zMeters = zMeters,
                        confidence =
                            jointConfidence
                                .coerceIn(0f, 1f),
                    )
            }

            if (version >= CURRENT_VERSION) {
                appearanceQuality =
                    (
                        buffer.get().toInt() and 0xFF
                    ) / 255f

                repeat(
                    APPEARANCE_FINGERPRINT_DIMENSIONS,
                ) {
                    appearanceFingerprint +=
                        (
                            (
                                buffer.get().toInt()
                            )
                                .coerceIn(0, 127)
                                .toFloat() /
                                127f
                        ).coerceIn(0f, 1f)
                }
            }

            actors +=
                Actor(
                    actorId = actorId,
                    confidence = confidence.coerceIn(0f, 1f),
                    predicted = predictedByte != 0,
                    missedFrames = missedFrames,
                    centerX = centerX,
                    centerY = centerY,
                    boundsLeft = boundsLeft,
                    boundsTop = boundsTop,
                    boundsRight = boundsRight,
                    boundsBottom = boundsBottom,
                    velocityX = velocityX,
                    velocityY = velocityY,
                    joints = joints,
                    worldJoints = worldJoints,
                    predictionConfidence =
                        predictionConfidence.coerceIn(0f, 1f),
                    meanMotionInnovationNormalized =
                        meanMotionInnovationNormalized.coerceAtLeast(0f),
                    acceleratedTrack = acceleratedTrack,
                    globalAssignmentBounded = globalAssignmentBounded,
                    appearanceFingerprint =
                        appearanceFingerprint.toList(),
                    appearanceQuality =
                        appearanceQuality.coerceIn(0f, 1f),
                )
        }

        require(!buffer.hasRemaining()) {
            "Unexpected trailing bytes in multi-actor pose payload"
        }

        return Frame(
            timestampNanos = max(0L, timestampNanos),
            sourceDeviceId = sourceDeviceId,
            inputWidth = width,
            inputHeight = height,
            actors = actors,
        )
    }

    private fun encodedActorBytes(
        actor: Actor,
    ): Int =
        ACTOR_HEADER_BYTES +
            EXTENSION_HEADER_BYTES +
            JOINT_BYTES *
            actor.joints
                .distinctBy { it.landmarkId }
                .size
                .coerceAtMost(MAX_JOINTS_PER_ACTOR) +
            WORLD_JOINT_BYTES *
            actor.worldJoints
                .distinctBy { it.landmarkId }
                .size
                .coerceAtMost(MAX_WORLD_JOINTS_PER_ACTOR) +
            APPEARANCE_FINGERPRINT_BYTES

    private fun validateActor(actor: Actor) {
        require(actor.actorId >= 0L)
        require(actor.confidence.isFinite())
        require(actor.centerX.isFinite() && actor.centerY.isFinite())
        require(
            actor.boundsLeft.isFinite() &&
                actor.boundsTop.isFinite() &&
                actor.boundsRight.isFinite() &&
                actor.boundsBottom.isFinite()
        )
        require(actor.velocityX.isFinite() && actor.velocityY.isFinite())
        require(actor.joints.size <= MAX_JOINTS_PER_ACTOR)
        require(actor.worldJoints.size <= MAX_WORLD_JOINTS_PER_ACTOR)
        require(actor.predictionConfidence.isFinite())
        require(actor.meanMotionInnovationNormalized.isFinite())
        require(actor.appearanceQuality.isFinite())
        require(
            actor.appearanceFingerprint.size <=
                APPEARANCE_FINGERPRINT_DIMENSIONS,
        )
        actor.appearanceFingerprint.forEach { value ->
            require(value.isFinite())
        }

        actor.joints.forEach { joint ->
            require(joint.x.isFinite() && joint.y.isFinite())
            require(joint.confidence.isFinite())
        }

        actor.worldJoints.forEach { joint ->
            require(
                joint.xMeters.isFinite() &&
                    joint.yMeters.isFinite() &&
                    joint.zMeters.isFinite(),
            )
            require(joint.confidence.isFinite())
        }
    }

    private fun sourceIdBytes(
        sourceDeviceId: String,
    ): ByteArray {
        require(sourceDeviceId.isNotBlank())

        val bytes =
            sourceDeviceId.toByteArray(Charsets.UTF_8)

        require(
            bytes.isNotEmpty() &&
                bytes.size <= MAX_SOURCE_ID_BYTES,
        )

        return bytes
    }

    private fun validateFinite(
        field: String,
        value: Float,
    ) {
        require(value.isFinite()) {
            "Non-finite $field in multi-actor payload"
        }
    }
}
