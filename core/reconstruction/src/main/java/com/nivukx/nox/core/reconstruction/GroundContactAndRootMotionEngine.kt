
package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Estimates support contacts and stabilizes root motion for articulated subjects.
 *
 * A monocular pose can be geometrically plausible while still producing the classic mocap
 * artifacts of foot skating, floating feet and frame-to-frame root drift. This engine uses
 * velocity, acceleration, vertical stability, support symmetry and a slowly learned ground plane
 * to turn those heuristics into explicit evidence. It is deliberately conservative: uncertain
 * contacts are reported as uncertain instead of being forced to the floor.
 */
class GroundContactAndRootMotionEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val footVelocityThresholdMetersPerSecond: Double = 0.075,
        val footAccelerationThresholdMetersPerSecond2: Double = 1.40,
        val verticalDriftThresholdMeters: Double = 0.022,
        val contactConfidenceThreshold: Double = 0.55,
        val maximumTrustedFootUncertaintyMeters: Double = 0.08,
        val minimumContactUncertaintyScore: Double = 0.42,
        val groundAdaptationAlpha: Double = 0.06,
        val rootSmoothingAlpha: Double = 0.30,
        val footCorrectionAlpha: Double = 0.60,
        val maximumGroundCorrectionMeters: Double = 0.18,
        val maximumRootCorrectionMeters: Double = 0.12,
        val minimumSupportSeparationMeters: Double = 0.06,
        val maximumSupportSeparationMeters: Double = 2.50,
        val historyFrames: Int = 24,
    ) {
        init {
            require(footVelocityThresholdMetersPerSecond > 0.0)
            require(footAccelerationThresholdMetersPerSecond2 > 0.0)
            require(verticalDriftThresholdMeters > 0.0)
            require(contactConfidenceThreshold in 0.0..1.0)
            require(maximumTrustedFootUncertaintyMeters > 0.0)
            require(minimumContactUncertaintyScore in 0.0..1.0)
            require(groundAdaptationAlpha in 0.0..1.0)
            require(rootSmoothingAlpha in 0.0..1.0)
            require(footCorrectionAlpha in 0.0..1.0)
            require(maximumGroundCorrectionMeters > 0.0)
            require(maximumRootCorrectionMeters > 0.0)
            require(minimumSupportSeparationMeters >= 0.0)
            require(maximumSupportSeparationMeters > minimumSupportSeparationMeters)
            require(historyFrames >= 2)
        }
    }

    data class Contact(
        val landmarkId: Int,
        val confidence: Double,
        val stable: Boolean,
        val slipping: Boolean,
        val groundHeightMeters: Double,
        val verticalVelocityMetersPerSecond: Double,
        val speedMetersPerSecond: Double,
        val supportScore: Double,
        val uncertaintyScore: Double,
    )

    data class Plane(
        val normal: Vec3,
        val offsetMeters: Double,
        val confidence: Double,
    )

    data class RootMotion(
        val rootPositionMeters: Vec3,
        val rootVelocityMetersPerSecond: Vec3,
        val rootAccelerationMetersPerSecond2: Vec3,
        val corrected: Boolean,
        val correctionMeters: Double,
        val driftRisk: Double,
    )

    data class Result(
        val skeleton: SkeletonFrame,
        val plane: Plane,
        val contacts: List<Contact>,
        val rootMotion: RootMotion,
        val contactCount: Int,
        val stableContactCount: Int,
        val footSkateRisk: Double,
        val rootDriftRisk: Double,
        val confidence: Double,
        val diagnostics: List<String>,
    ) {
        val grounded: Boolean
            get() = stableContactCount > 0
    }

    private data class FootState(
        val timestampNanos: Long,
        val position: Vec3,
        val velocity: Vec3,
    )

    private val history = ArrayDeque<SkeletonFrame>()
    private val leftFootHistory = ArrayDeque<FootState>()
    private val rightFootHistory = ArrayDeque<FootState>()
    private var groundHeight: Double? = null
    private var previousRoot: Vec3? = null
    private var previousRootVelocity = Vec3(0.0, 0.0, 0.0)
    private var previousTimestampNanos: Long = Long.MIN_VALUE

    @Synchronized
    fun process(
        frame: SkeletonFrame,
    ): Result {
        history.addLast(frame)
        while (history.size > config.historyFrames) {
            history.removeFirst()
        }

        val dtSeconds =
            if (
                previousTimestampNanos == Long.MIN_VALUE ||
                frame.timestampNanos <= previousTimestampNanos
            ) {
                1.0 / 60.0
            } else {
                (
                    frame.timestampNanos -
                        previousTimestampNanos
                    ).toDouble() / 1e9
            }.coerceIn(1e-4, 0.50)

        val leftFoot = footJoint(frame, left = true)
        val rightFoot = footJoint(frame, left = false)

        val contacts = buildList {
            leftFoot?.let {
                add(
                    evaluateFoot(
                        frame = frame,
                        joint = it,
                        left = true,
                        dtSeconds = dtSeconds,
                    ),
                )
            }
            rightFoot?.let {
                add(
                    evaluateFoot(
                        frame = frame,
                        joint = it,
                        left = false,
                        dtSeconds = dtSeconds,
                    ),
                )
            }
        }

        updateGroundPlane(contacts)

        val stableContacts = contacts.map { contact ->
            val refinedGround = groundHeight ?: contact.groundHeightMeters
            contact.copy(
                confidence = (
                    contact.confidence * 0.62 +
                        planeConfidence() * 0.24 +
                        contact.uncertaintyScore * 0.14
                    ).coerceIn(0.0, 1.0),
                stable =
                    contact.stable &&
                        abs(contact.groundHeightMeters - refinedGround) <
                            config.verticalDriftThresholdMeters * 2.0,
                groundHeightMeters = refinedGround,
            )
        }

        val correctedFrame = correctFeetAndRoot(
            frame = frame,
            contacts = stableContacts,
            dtSeconds = dtSeconds,
        )

        val correctedRoot = rootJoint(correctedFrame)
        val rootMotion = updateRootMotion(
            timestampNanos = correctedFrame.timestampNanos,
            root = correctedRoot?.position,
            dtSeconds = dtSeconds,
        )

        val footSkateRisk = (
            stableContacts
                .map { contact ->
                    if (contact.stable && contact.speedMetersPerSecond >
                        config.footVelocityThresholdMetersPerSecond
                    ) {
                        (
                            contact.speedMetersPerSecond /
                                (config.footVelocityThresholdMetersPerSecond * 4.0)
                            ).coerceIn(0.0, 1.0)
                    } else {
                        0.0
                    }
                }
                .averageOrZero()
        )

        val rootDriftRisk = (
            rootMotion.driftRisk * 0.72 +
                footSkateRisk * 0.28
            ).coerceIn(0.0, 1.0)

        val contactConfidence =
            stableContacts.map { it.confidence }.averageOrZero()

        val confidence = (
            contactConfidence * 0.35 +
                (1.0 - footSkateRisk) * 0.30 +
                (1.0 - rootDriftRisk) * 0.20 +
                rootMotionConfidence(rootMotion) * 0.15
            ).coerceIn(0.0, 1.0)

        val diagnostics = listOf(
            "subject=" + frame.subjectType.name,
            "contacts=" + stableContacts.size,
            "stableContacts=" + stableContacts.count { it.stable },
            "groundHeight=" + format(groundHeight ?: Double.NaN),
            "groundConfidence=" + format(planeConfidence()),
            "footSkateRisk=" + format(footSkateRisk),
            "rootDriftRisk=" + format(rootDriftRisk),
            "rootCorrectionMeters=" + format(rootMotion.correctionMeters),
            "meanFootUncertainty=" +
                format(
                    stableContacts
                        .map { it.uncertaintyScore }
                        .averageOrZero(),
                ),
            "confidence=" + format(confidence),
        )

        previousTimestampNanos = correctedFrame.timestampNanos

        return Result(
            skeleton = correctedFrame,
            plane = Plane(
                normal = Vec3(0.0, 1.0, 0.0),
                offsetMeters = -(groundHeight ?: 0.0),
                confidence = planeConfidence(),
            ),
            contacts = stableContacts,
            rootMotion = rootMotion,
            contactCount = stableContacts.size,
            stableContactCount = stableContacts.count { it.stable },
            footSkateRisk = footSkateRisk,
            rootDriftRisk = rootDriftRisk,
            confidence = confidence,
            diagnostics = diagnostics,
        )
    }

    @Synchronized
    fun reset() {
        history.clear()
        leftFootHistory.clear()
        rightFootHistory.clear()
        groundHeight = null
        previousRoot = null
        previousRootVelocity = Vec3(0.0, 0.0, 0.0)
        previousTimestampNanos = Long.MIN_VALUE
    }

    private fun evaluateFoot(
        frame: SkeletonFrame,
        joint: ReconstructionJoint,
        left: Boolean,
        dtSeconds: Double,
    ): Contact {
        val buffer = if (left) leftFootHistory else rightFootHistory

        val previous = buffer.lastOrNull()
        val velocity =
            if (previous == null) {
                joint.velocity
            } else {
                (
                    joint.position -
                        previous.position
                ) / dtSeconds
            }

        val speed = magnitude(velocity)
        val verticalSpeed = abs(velocity.y)

        val previousVelocity = previous?.velocity
        val acceleration = if (previousVelocity == null) {
            0.0
        } else {
            magnitude(
                (velocity - previousVelocity) /
                    dtSeconds,
            )
        }

        val stabilityScore = (
            1.0 -
                speed /
                    (config.footVelocityThresholdMetersPerSecond * 3.0)
            ).coerceIn(0.0, 1.0)

        val verticalScore = (
            1.0 -
                verticalSpeed /
                    (config.footVelocityThresholdMetersPerSecond * 2.5)
            ).coerceIn(0.0, 1.0)

        val accelerationScore = (
            1.0 -
                acceleration /
                    (config.footAccelerationThresholdMetersPerSecond2 * 2.0)
            ).coerceIn(0.0, 1.0)

        val observationScore =
            joint.confidence.coerceIn(0.0, 1.0)

        val uncertaintyScore =
            if (joint.positionUncertaintyMeters.isFinite()) {
                exp(
                    -joint.positionUncertaintyMeters /
                        config.maximumTrustedFootUncertaintyMeters
                ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val supportScore = (
            observationScore * 0.34 +
                stabilityScore * 0.24 +
                verticalScore * 0.17 +
                accelerationScore * 0.10 +
                uncertaintyScore * 0.15
            ).coerceIn(0.0, 1.0)

        val stable =
            supportScore >= config.contactConfidenceThreshold &&
                uncertaintyScore >= config.minimumContactUncertaintyScore &&
                speed <= config.footVelocityThresholdMetersPerSecond * 1.8 &&
                verticalSpeed <= config.footVelocityThresholdMetersPerSecond * 1.6

        val slipping =
            stable &&
                speed > config.footVelocityThresholdMetersPerSecond

        val ground = groundHeight ?: joint.position.y

        buffer.addLast(
            FootState(
                timestampNanos = frame.timestampNanos,
                position = joint.position,
                velocity = velocity,
            ),
        )

        while (buffer.size > config.historyFrames) {
            buffer.removeFirst()
        }

        return Contact(
            landmarkId = joint.landmarkId,
            confidence = supportScore,
            stable = stable,
            slipping = slipping,
            groundHeightMeters = ground,
            verticalVelocityMetersPerSecond = velocity.y,
            speedMetersPerSecond = speed,
            supportScore = supportScore,
            uncertaintyScore = uncertaintyScore,
        )
    }

    private fun updateGroundPlane(
        contacts: List<Contact>,
    ) {
        val candidates = contacts
            .filter {
                it.stable &&
                    it.confidence >= config.contactConfidenceThreshold
            }
            .map { it.groundHeightMeters }

        if (candidates.isEmpty()) return

        val median = median(candidates)

        groundHeight = groundHeight?.let { previous ->
            previous +
                (median - previous) *
                config.groundAdaptationAlpha
        } ?: median
    }

    private fun correctFeetAndRoot(
        frame: SkeletonFrame,
        contacts: List<Contact>,
        dtSeconds: Double,
    ): SkeletonFrame {
        if (contacts.isEmpty()) return frame

        val ground = groundHeight ?: return frame

        val contactById = contacts.associateBy { it.landmarkId }

        val correctedJoints = frame.joints.map { joint ->
            val contact = contactById[joint.landmarkId]
                ?: return@map joint

            if (!contact.stable) return@map joint

            val desiredY = ground
            val correction = (
                desiredY - joint.position.y
            ).coerceIn(
                -config.maximumGroundCorrectionMeters,
                config.maximumGroundCorrectionMeters,
            )

            if (abs(correction) < 1e-4) {
                return@map joint
            }

            val correctionScaled =
                correction *
                    config.footCorrectionAlpha *
                    contact.confidence.coerceIn(0.0, 1.0) *
                    contact.uncertaintyScore.coerceIn(0.0, 1.0)

            joint.copy(
                position = joint.position.copy(
                    y = joint.position.y +
                        correctionScaled,
                ),
                velocity = joint.velocity.copy(
                    y = joint.velocity.y * (
                        1.0 -
                            config.footCorrectionAlpha *
                            contact.confidence
                    ),
                ),
                confidence = (
                    joint.confidence *
                        (0.92 + contact.confidence * 0.08)
                    ).coerceIn(0.0, 1.0),
                positionUncertaintyMeters = (
                    joint.positionUncertaintyMeters +
                        abs(correctionScaled) * 0.12
                    ).coerceAtLeast(0.0),
            )
        }

        val intermediate = frame.copy(
            joints = correctedJoints,
        )

        return stabilizeRoot(
            frame = intermediate,
            contacts = contacts,
            dtSeconds = dtSeconds,
        )
    }

    private fun stabilizeRoot(
        frame: SkeletonFrame,
        contacts: List<Contact>,
        dtSeconds: Double,
    ): SkeletonFrame {
        val root = rootJoint(frame) ?: return frame
        val support = contacts.filter { it.stable }

        if (support.isEmpty()) return frame

        val meanFootX = support.mapNotNull {
            frame.joints
                .firstOrNull { joint ->
                    joint.landmarkId == it.landmarkId
                }
                ?.position
                ?.x
        }.averageOrZero()

        val meanFootZ = support.mapNotNull {
            frame.joints
                .firstOrNull { joint ->
                    joint.landmarkId == it.landmarkId
                }
                ?.position
                ?.z
        }.averageOrZero()

        val supportCenter = Vec3(
            meanFootX,
            root.position.y,
            meanFootZ,
        )

        val oldRoot = previousRoot
        val expected = oldRoot?.let {
            it + previousRootVelocity * dtSeconds
        } ?: root.position

        val drift = Vec3(
            root.position.x - expected.x,
            0.0,
            root.position.z - expected.z,
        )

        val driftMagnitude = magnitude(drift)
        val normalizedDrift = (
            driftMagnitude /
                config.maximumRootCorrectionMeters
            ).coerceIn(0.0, 1.0)

        val correction = if (driftMagnitude >
            config.maximumRootCorrectionMeters
        ) {
            drift * (
                config.maximumRootCorrectionMeters /
                    driftMagnitude.coerceAtLeast(1e-8)
            )
        } else {
            drift
        }

        val desiredRoot = root.position -
            correction *
            config.rootSmoothingAlpha.coerceIn(0.0, 1.0)

        val supportInfluence = if (support.size >= 2 &&
            supportSeparation(frame, support) in
            config.minimumSupportSeparationMeters..
            config.maximumSupportSeparationMeters
        ) {
            0.26
        } else {
            0.08
        }

        val stabilized = root.position.lerp(
            Vec3(
                desiredRoot.x * (1.0 - supportInfluence) +
                    supportCenter.x * supportInfluence,
                root.position.y,
                desiredRoot.z * (1.0 - supportInfluence) +
                    supportCenter.z * supportInfluence,
            ),
            config.rootSmoothingAlpha,
        )

        val delta = stabilized - root.position

        return frame.copy(
            joints = frame.joints.map { joint ->
                if (joint.landmarkId == root.landmarkId) {
                    joint.copy(
                        position = stabilized,
                        velocity = joint.velocity - delta / dtSeconds.coerceAtLeast(1e-4),
                        positionUncertaintyMeters = (
                            joint.positionUncertaintyMeters +
                                magnitude(delta) * 0.08
                            ).coerceAtLeast(0.0),
                    )
                } else {
                    joint
                }
            },
        )
    }

    private fun updateRootMotion(
        timestampNanos: Long,
        root: Vec3?,
        dtSeconds: Double,
    ): RootMotion {
        if (root == null) {
            return RootMotion(
                rootPositionMeters = Vec3(0.0, 0.0, 0.0),
                rootVelocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
                rootAccelerationMetersPerSecond2 = Vec3(0.0, 0.0, 0.0),
                corrected = false,
                correctionMeters = 0.0,
                driftRisk = 1.0,
            )
        }

        val previous = previousRoot
        val velocity = if (previous == null) {
            Vec3(0.0, 0.0, 0.0)
        } else {
            (root - previous) / dtSeconds.coerceAtLeast(1e-4)
        }

        val acceleration = (
            velocity -
                previousRootVelocity
            ) / dtSeconds.coerceAtLeast(1e-4)

        val expected = previous?.let {
            it + previousRootVelocity * dtSeconds
        } ?: root

        val correction = root - expected

        val driftRisk = (
            magnitude(correction) /
                config.maximumRootCorrectionMeters
            ).coerceIn(0.0, 1.0)

        val result = RootMotion(
            rootPositionMeters = root,
            rootVelocityMetersPerSecond = velocity,
            rootAccelerationMetersPerSecond2 = acceleration,
            corrected = magnitude(correction) >
                config.maximumRootCorrectionMeters * 0.25,
            correctionMeters = magnitude(correction),
            driftRisk = driftRisk,
        )

        previousRootVelocity = velocity
        previousRoot = root

        return result
    }

    private fun footJoint(
        frame: SkeletonFrame,
        left: Boolean,
    ): ReconstructionJoint? {
        val candidates = if (left) {
            intArrayOf(31, 29, 27)
        } else {
            intArrayOf(32, 30, 28)
        }

        return candidates
            .asSequence()
            .mapNotNull { id ->
                frame.joints.firstOrNull {
                    it.landmarkId == id
                }
            }
            .maxByOrNull { it.confidence }
    }

    private fun rootJoint(
        frame: SkeletonFrame,
    ): ReconstructionJoint? =
        frame.joints
            .filter {
                it.landmarkId == 23 ||
                    it.landmarkId == 24 ||
                    it.landmarkId == 25
            }
            .maxByOrNull { it.confidence }

    private fun supportSeparation(
        frame: SkeletonFrame,
        contacts: List<Contact>,
    ): Double {
        if (contacts.size < 2) return 0.0

        val a = frame.joints.firstOrNull {
            it.landmarkId == contacts[0].landmarkId
        } ?: return 0.0
        val b = frame.joints.firstOrNull {
            it.landmarkId == contacts[1].landmarkId
        } ?: return 0.0

        return a.position.distanceTo(b.position)
    }

    private fun planeConfidence(): Double =
        when {
            groundHeight == null -> 0.25
            history.size < 3 -> 0.45
            else ->
                (
                    history.size.toDouble() /
                        config.historyFrames.toDouble()
                    ).coerceIn(0.0, 1.0)
        }

    private fun rootMotionConfidence(
        root: RootMotion,
    ): Double =
        (
            1.0 -
                root.driftRisk * 0.80
        ).coerceIn(0.0, 1.0)

    private fun magnitude(value: Vec3): Double =
        sqrt(
            value.x * value.x +
                value.y * value.y +
                value.z * value.z,
        )

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        return if (sorted.size % 2 == 0) {
            (
                sorted[sorted.size / 2 - 1] +
                    sorted[sorted.size / 2]
                ) * 0.5
        } else {
            sorted[sorted.size / 2]
        }
    }

    private fun format(value: Double): String =
        if (!value.isFinite()) {
            "nan"
        } else {
            "%.5f".format(java.util.Locale.US, value)
        }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
