package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Global 4D consistency evaluator.
 *
 * This engine does not solve camera poses, deformable geometry, or scene segmentation. Those
 * responsibilities remain with their existing authoritative engines. Instead it evaluates whether
 * the already-solved human trajectory, camera/reprojection quality, rig trust, physical skeleton
 * constraints, and optional static/dynamic scene evidence agree in one global operating frame.
 *
 * The result is deliberately conservative:
 * - missing scene decomposition is reported as missing evidence rather than fabricated;
 * - a single bad frame cannot redefine the global baseline;
 * - persistent disagreement enters a degraded state;
 * - recovery requires repeated independent healthy frames;
 * - root motion is derived only from observed pelvis landmarks;
 * - support/contact evidence comes from the physical constraint result when available.
 *
 * This layer is designed to complement the current multiview pipeline and mirrors the global
 * human-scene-camera consistency direction in recent 4D reconstruction research without pretending
 * to contain an external neural reconstruction model.
 */
class Global4DConsistencyEngine(
    private var config: Config = Config(),
) {

    enum class State {
        WARMING,
        STABLE,
        DEGRADED,
        PERSISTENT_FAILURE,
        RECOVERING,
    }

    data class Config(
        val minimumRootConfidence: Double = 0.28,
        val preferredRootConfidence: Double = 0.70,
        val maximumRootSpeedMetersPerSecond: Double = 8.0,
        val maximumRootAccelerationMetersPerSecond2: Double = 120.0,
        val maximumRootJerkMetersPerSecond3: Double = 420.0,
        val rootTemporalWeight: Double = 0.16,
        val physicsWeight: Double = 0.18,
        val reprojectionWeight: Double = 0.16,
        val temporalWeight: Double = 0.14,
        val synchronizationWeight: Double = 0.12,
        val rigTrustWeight: Double = 0.10,
        val cameraMotionWeight: Double = 0.08,
        val sceneWeight: Double = 0.06,
        val directionalUncertaintyWeight: Double = 0.08,
        val faceEyeWeight: Double = 0.07,
        val rigStabilityWeight: Double = 0.09,
        val dynamicMetrologyWeight: Double = 0.08,
        val gaussian4DWeight: Double = 0.10,
        val gaussianStrainWeight: Double = 0.10,
        val maximumGaussianP95StrainRatio: Double = 0.35,
        val maximumGaussianMaximumStrainRatio: Double = 0.75,
        val minimumGaussianTrustedRatio: Double = 0.55,
        val minimumGaussianTemporalStability: Double = 0.70,
        val minimumGaussianNodeConfidence: Double = 0.60,
        val minimumGaussianBindingConfidence: Double = 0.60,
        val maximumRigTranslationDriftMeters: Double = 0.0015,
        val maximumRigRotationDriftRadians: Double = 0.0025,
        val maximumDynamicMetrologyP95Millimeters: Double = 1.0,
        val maximumDynamicMetrologyRmseMillimeters: Double = 0.75,
        val maximumDynamicMetrologyMaxErrorMillimeters: Double = 2.0,
        val minimumDynamicMetrologyConfidence: Double = 0.60,
        val maximumDirectionalAnisotropy: Double = 14.0,
        val minimumDirectionalConfidence: Double = 0.20,
        val maximumDirectionalDepthUncertaintyMeters: Double = 0.16,
        val minimumFaceEyeConfidence: Double = 0.24,
        val staticCoverageMinimum: Double = 0.18,
        val dynamicCoverageMaximum: Double = 0.78,
        val unknownCoverageMaximum: Double = 0.42,
        val sceneAmbiguityPenalty: Double = 0.34,
        val rootMotionConsistencyWeight: Double = 0.54,
        val rootVelocityAgreementWeight: Double = 0.08,
        val contactConsistencyWeight: Double = 0.26,
        val symmetryWeight: Double = 0.20,
        val bodyScaleWeight: Double = 0.08,
        val maximumBodyScaleChangeRatioPerFrame: Double = 0.065,
        val historyAlpha: Double = 0.20,
        val warningScore: Double = 0.64,
        val failureScore: Double = 0.42,
        val persistentFailureFrames: Int = 3,
        val recoveryFrames: Int = 4,
        val monotonicJitterToleranceNanos: Long = 1_000_000L,
    ) {
        init {
            require(minimumRootConfidence in 0.0..1.0)
            require(preferredRootConfidence in 0.0..1.0)
            require(maximumRootSpeedMetersPerSecond > 0.0)
            require(maximumRootAccelerationMetersPerSecond2 > 0.0)
            require(maximumRootJerkMetersPerSecond3 > 0.0)
            require(rootTemporalWeight >= 0.0)
            require(physicsWeight >= 0.0)
            require(reprojectionWeight >= 0.0)
            require(temporalWeight >= 0.0)
            require(synchronizationWeight >= 0.0)
            require(rigTrustWeight >= 0.0)
            require(cameraMotionWeight >= 0.0)
            require(sceneWeight >= 0.0)
            require(directionalUncertaintyWeight >= 0.0)
            require(faceEyeWeight >= 0.0)
            require(rigStabilityWeight >= 0.0)
            require(dynamicMetrologyWeight >= 0.0)
            require(gaussian4DWeight >= 0.0)
            require(gaussianStrainWeight >= 0.0)
            require(maximumGaussianP95StrainRatio > 0.0)
            require(maximumGaussianMaximumStrainRatio >= maximumGaussianP95StrainRatio)
            require(minimumGaussianTrustedRatio in 0.0..1.0)
            require(minimumGaussianTemporalStability in 0.0..1.0)
            require(minimumGaussianNodeConfidence in 0.0..1.0)
            require(minimumGaussianBindingConfidence in 0.0..1.0)
            require(maximumRigTranslationDriftMeters > 0.0)
            require(maximumRigRotationDriftRadians > 0.0)
            require(maximumDynamicMetrologyP95Millimeters > 0.0)
            require(maximumDynamicMetrologyRmseMillimeters > 0.0)
            require(maximumDynamicMetrologyMaxErrorMillimeters > 0.0)
            require(minimumDynamicMetrologyConfidence in 0.0..1.0)
            require(maximumDirectionalAnisotropy >= 1.0)
            require(minimumDirectionalConfidence in 0.0..1.0)
            require(maximumDirectionalDepthUncertaintyMeters > 0.0)
            require(minimumFaceEyeConfidence in 0.0..1.0)
            require(staticCoverageMinimum in 0.0..1.0)
            require(dynamicCoverageMaximum in 0.0..1.0)
            require(unknownCoverageMaximum in 0.0..1.0)
            require(sceneAmbiguityPenalty in 0.0..1.0)
            require(rootMotionConsistencyWeight >= 0.0)
            require(rootVelocityAgreementWeight >= 0.0)
            require(contactConsistencyWeight >= 0.0)
            require(symmetryWeight >= 0.0)
            require(historyAlpha in 0.0..1.0)
            require(warningScore in 0.0..1.0)
            require(failureScore in 0.0..warningScore)
            require(persistentFailureFrames >= 1)
            require(recoveryFrames >= 1)
            require(monotonicJitterToleranceNanos >= 0L)
        }
    }

    data class Gaussian4DEvidence(
        val temporalStability: Double,
        val trustedRatio: Double,
        val meanNodeConfidence: Double,
        val meanBindingConfidence: Double,
        val p95StrainRatio: Double,
        val maximumStrainRatio: Double,
    ) {
        init {
            require(
                temporalStability.isFinite() &&
                    temporalStability in 0.0..1.0,
            )
            require(
                trustedRatio.isFinite() &&
                    trustedRatio in 0.0..1.0,
            )
            require(
                meanNodeConfidence.isFinite() &&
                    meanNodeConfidence in 0.0..1.0,
            )
            require(
                meanBindingConfidence.isFinite() &&
                    meanBindingConfidence in 0.0..1.0,
            )
            require(
                p95StrainRatio.isFinite() &&
                    p95StrainRatio >= 0.0,
            )
            require(
                maximumStrainRatio.isFinite() &&
                    maximumStrainRatio >= 0.0,
            )
        }
    }

    data class Input(
        val skeleton: SkeletonFrame,
        val physicsScore: Double,
        val physicsAccepted: Boolean,
        val reprojectionScore: Double,
        val temporalScore: Double,
        val synchronizationConfidence: Double,
        val rigTrust: Double,
        val cameraMotionRisk: Double,
        val directionalUncertaintyConfidence: Double = 1.0,
        val directionalUncertaintyAnisotropy: Double = 1.0,
        val directionalUncertaintyDepthMeters: Double = 0.0,
        val faceEyeConfidence: Double = 1.0,
        val faceEyeAccepted: Boolean = true,
        val faceEyeOutOfDistributionRisk: Double = 0.0,
        val rigStabilityConfidence: Double = 1.0,
        val rigTranslationDriftMeters: Double = 0.0,
        val rigRotationDriftRadians: Double = 0.0,
        val rigUnstable: Boolean = false,
        val dynamicMetrologyP95Millimeters: Double = 0.0,
        val dynamicMetrologyRmseMillimeters: Double = 0.0,
        val dynamicMetrologyMaxErrorMillimeters: Double = 0.0,
        val dynamicMetrologyConfidence: Double = 1.0,
        val dynamicMetrologyValid: Boolean = true,
        val gaussianTemporalStability: Double = 1.0,
        val gaussianTrustedRatio: Double = 1.0,
        val gaussianMeanNodeConfidence: Double = 1.0,
        val gaussianMeanBindingConfidence: Double = 1.0,
        val gaussianP95StrainRatio: Double = 0.0,
        val gaussianMaximumStrainRatio: Double = 0.0,
        val gaussianEvidenceAvailable: Boolean = false,
        val externalContactConfidence: Double = 1.0,
        val externalContactEvidenceAvailable: Boolean = false,
        val sceneDecompositionConfidence: Double? = null,
        val staticCoverage: Double? = null,
        val dynamicCoverage: Double? = null,
        val unknownCoverage: Double? = null,
        val sceneEvidenceAvailable: Boolean =
            sceneDecompositionConfidence != null &&
                staticCoverage != null &&
                dynamicCoverage != null &&
                unknownCoverage != null,
        val bodyMetricEvidenceAuthority: Double = 1.0,
        val faceMetricEvidenceAuthority: Double = 1.0,
        val handMetricEvidenceAuthority: Double = 1.0,
        val bodyEpipolarConsistency: Double = 1.0,
        val faceEpipolarConsistency: Double = 1.0,
        val handEpipolarConsistency: Double = 1.0,
    ) {
        init {
            require(physicsScore in 0.0..1.0)
            require(reprojectionScore in 0.0..1.0)
            require(temporalScore in 0.0..1.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(rigTrust in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
            require(directionalUncertaintyConfidence in 0.0..1.0)
            require(directionalUncertaintyAnisotropy >= 1.0)
            require(
                directionalUncertaintyDepthMeters.isFinite() &&
                    directionalUncertaintyDepthMeters >= 0.0,
            )
            require(faceEyeConfidence in 0.0..1.0)
            require(faceEyeOutOfDistributionRisk in 0.0..1.0)
            require(rigStabilityConfidence in 0.0..1.0)
            require(rigTranslationDriftMeters.isFinite() && rigTranslationDriftMeters >= 0.0)
            require(rigRotationDriftRadians.isFinite() && rigRotationDriftRadians >= 0.0)
            require(dynamicMetrologyP95Millimeters.isFinite() && dynamicMetrologyP95Millimeters >= 0.0)
            require(dynamicMetrologyRmseMillimeters.isFinite() && dynamicMetrologyRmseMillimeters >= 0.0)
            require(dynamicMetrologyMaxErrorMillimeters.isFinite() && dynamicMetrologyMaxErrorMillimeters >= 0.0)
            require(dynamicMetrologyConfidence in 0.0..1.0)
            require(gaussianTemporalStability in 0.0..1.0)
            require(gaussianTrustedRatio in 0.0..1.0)
            require(gaussianMeanNodeConfidence in 0.0..1.0)
            require(gaussianMeanBindingConfidence in 0.0..1.0)
            require(gaussianP95StrainRatio.isFinite() && gaussianP95StrainRatio >= 0.0)
            require(gaussianMaximumStrainRatio.isFinite() && gaussianMaximumStrainRatio >= 0.0)
            require(externalContactConfidence in 0.0..1.0)
            require(bodyMetricEvidenceAuthority in 0.0..1.0)
            require(faceMetricEvidenceAuthority in 0.0..1.0)
            require(handMetricEvidenceAuthority in 0.0..1.0)
            require(bodyEpipolarConsistency in 0.0..1.0)
            require(faceEpipolarConsistency in 0.0..1.0)
            require(handEpipolarConsistency in 0.0..1.0)
            if (sceneDecompositionConfidence != null) {
                require(sceneDecompositionConfidence in 0.0..1.0)
            }
            if (staticCoverage != null) require(staticCoverage in 0.0..1.0)
            if (dynamicCoverage != null) require(dynamicCoverage in 0.0..1.0)
            if (unknownCoverage != null) require(unknownCoverage in 0.0..1.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val score: Double,
        val rootMotionConsistency: Double,
        val rootVelocityAgreement: Double = 1.0,
        val rootSpeedMetersPerSecond: Double,
        val rootAccelerationMetersPerSecond2: Double,
        val rootJerkMetersPerSecond3: Double,
        val physicsConsistency: Double,
        val contactConsistency: Double,
        val externalContactConsistency: Double = 1.0,
        val symmetryConsistency: Double,
        val cameraHumanConsistency: Double,
        val sceneConsistency: Double,
        val sceneEvidenceAvailable: Boolean,
        val temporalConsistency: Double,
        val directionalUncertaintyConsistency: Double = 1.0,
        val faceEyeConsistency: Double = 1.0,
        val bodyScaleConsistency: Double = 1.0,
        val rigStabilityConsistency: Double = 1.0,
        val dynamicMetrologyConsistency: Double = 1.0,
        val gaussian4DConsistency: Double = 1.0,
        val gaussianStrainConsistency: Double = 1.0,
        val gaussianEvidenceAvailable: Boolean = false,
        val crossModalMetricAuthority: Double = 1.0,
        val crossModalEpipolarAuthority: Double = 1.0,
        val persistentFailure: Boolean,
        val accepted: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class StateData(
        var timestampNanos: Long = Long.MIN_VALUE,
        var rootPosition: Vec3? = null,
        var rootVelocity: Vec3 = Vec3(0.0, 0.0, 0.0),
        var rootAcceleration: Vec3 = Vec3(0.0, 0.0, 0.0),
        var previousAccelerationMagnitude: Double = 0.0,
        var previousBodyScaleMeters: Double? = null,
        var scoreEma: Double = 0.0,
        var failureFrames: Int = 0,
        var healthyFrames: Int = 0,
        var state: State = State.WARMING,
        var initialized: Boolean = false,
    )

    private val state = StateData()

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig
        state.scoreEma =
            state.scoreEma.coerceIn(0.0, 1.0)
    }

    fun configuration(): Config = config

    @Synchronized
    fun snapshot(): Result? = lastResult

    private var lastResult: Result? = null

    @Synchronized
    fun reset() {
        state.timestampNanos = Long.MIN_VALUE
        state.rootPosition = null
        state.rootVelocity = Vec3(0.0, 0.0, 0.0)
        state.rootAcceleration = Vec3(0.0, 0.0, 0.0)
        state.previousAccelerationMagnitude = 0.0
        state.previousBodyScaleMeters = null
        state.scoreEma = 0.0
        state.failureFrames = 0
        state.healthyFrames = 0
        state.state = State.WARMING
        state.initialized = false
        lastResult = null
    }

    @Synchronized
    fun evaluate(input: Input): Result {
        val timestamp = input.skeleton.timestampNanos

        if (state.initialized && timestamp <= state.timestampNanos) {
            return if (lastResult != null) {
                duplicateOrJitterResult(
                    timestampNanos = timestamp,
                    previousTimestampNanos = state.timestampNanos,
                )
            } else {
                duplicateTimestampResult()
            }
        }

        val dtSeconds =
            if (!state.initialized) {
                1.0 / 30.0
            } else {
                (
                    timestamp -
                        state.timestampNanos
                    ).coerceAtLeast(1L) / 1e9
            }.coerceIn(1.0 / 240.0, 0.25)

        val root = deriveRoot(input.skeleton)
        val rootMetrics =
            updateRootState(
                root = root,
                timestampNanos = timestamp,
                dtSeconds = dtSeconds,
            )

        val physics =
            (
                input.physicsScore *
                    if (input.physicsAccepted) 1.0 else 0.44
                ).coerceIn(0.0, 1.0)

        val physicsContact =
            deriveContactConsistency(
                physicsAccepted = input.physicsAccepted,
                physicsScore = input.physicsScore,
            )

        val contact =
            if (input.externalContactEvidenceAvailable) {
                (
                    physicsContact * 0.60 +
                        input.externalContactConfidence.coerceIn(
                            0.0,
                            1.0,
                        ) * 0.40
                    ).coerceIn(0.0, 1.0)
            } else {
                physicsContact
            }

        val externalContactConsistency =
            if (input.externalContactEvidenceAvailable) {
                input.externalContactConfidence.coerceIn(
                    0.0,
                    1.0,
                )
            } else {
                1.0
            }

        val symmetry =
            deriveSymmetryConsistency(
                input.skeleton,
            )

        val bodyScaleConsistency =
            deriveBodyScaleConsistency(
                input.skeleton,
            )

        val cameraHuman =
            (
                input.reprojectionScore * 0.34 +
                    input.temporalScore * 0.20 +
                    input.synchronizationConfidence * 0.18 +
                    input.rigTrust * 0.16 +
                    (
                        1.0 -
                            input.cameraMotionRisk
                        ).coerceIn(0.0, 1.0) * 0.12
                ).coerceIn(0.0, 1.0)

        val scene =
            deriveSceneConsistency(
                input,
            )

        val rigStability =
            rigStabilityConsistency(
                input,
            )

        val dynamicMetrology =
            dynamicMetrologyConsistency(
                input,
            )

        val gaussian4D =
            gaussian4DConsistency(
                input,
            )

        val gaussianStrain =
            gaussianStrainConsistency(
                input,
            )

        val rootScore =
            rootMetrics.consistencyScore

        val rootVelocityAgreement =
            reportedRootVelocityAgreement(
                skeleton = input.skeleton,
                measuredVelocity = state.rootVelocity,
            )

        val baseScore =
            weightedGeometricMean(
                listOf(
                    rootScore to config.rootTemporalWeight,
                    rootVelocityAgreement to
                        config.rootVelocityAgreementWeight,
                    physics to config.physicsWeight,
                    input.reprojectionScore to
                        config.reprojectionWeight,
                    input.temporalScore to
                        config.temporalWeight,
                    input.synchronizationConfidence to
                        config.synchronizationWeight,
                    input.rigTrust to config.rigTrustWeight,
                    (
                        1.0 -
                            input.cameraMotionRisk
                        ).coerceIn(0.0, 1.0) to
                        config.cameraMotionWeight,
                    scene to config.sceneWeight,
                    directionalUncertaintyConsistency(
                        input,
                    ) to config.directionalUncertaintyWeight,
                    faceEyeConsistency(
                        input,
                    ) to config.faceEyeWeight,
                    bodyScaleConsistency to config.bodyScaleWeight,
                    rigStability to config.rigStabilityWeight,
                    dynamicMetrology to config.dynamicMetrologyWeight,
                    gaussian4D to config.gaussian4DWeight,
                    gaussianStrain to config.gaussianStrainWeight,
                    crossModalMetricAuthority(input) to 0.05,
                    crossModalEpipolarAuthority(input) to 0.08,
                ),
            )

        val crossCheck =
            (
                rootScore * config.rootMotionConsistencyWeight +
                    contact * config.contactConsistencyWeight +
                    symmetry * config.symmetryWeight
                ).coerceIn(0.0, 1.0)

        val rawScore =
            (
                baseScore * 0.78 +
                    crossCheck * 0.22
                ).coerceIn(0.0, 1.0)

        val immediateFailure =
            input.physicsScore < config.failureScore ||
                !input.physicsAccepted ||
                input.reprojectionScore < config.failureScore ||
                input.temporalScore < config.failureScore ||
                input.synchronizationConfidence < config.failureScore ||
                input.rigTrust < config.failureScore ||
                input.cameraMotionRisk > 0.90 ||
                input.directionalUncertaintyConfidence <
                    config.minimumDirectionalConfidence ||
                input.directionalUncertaintyAnisotropy >=
                    config.maximumDirectionalAnisotropy ||
                input.directionalUncertaintyDepthMeters >
                    config.maximumDirectionalDepthUncertaintyMeters ||
                !input.faceEyeAccepted ||
                input.faceEyeConfidence <
                    config.minimumFaceEyeConfidence ||
                input.faceEyeOutOfDistributionRisk >= 0.92 ||
                crossModalMetricAuthority(input) < 0.16 ||
                crossModalEpipolarAuthority(input) < 0.18 ||
                rigStability < config.failureScore ||
                dynamicMetrology < config.failureScore ||
                (
                    input.gaussianEvidenceAvailable &&
                        (
                            gaussian4D < config.failureScore ||
                                gaussianStrain < config.failureScore ||
                                input.gaussianP95StrainRatio >
                                    config.maximumGaussianP95StrainRatio ||
                                input.gaussianMaximumStrainRatio >
                                    config.maximumGaussianMaximumStrainRatio ||
                                input.gaussianTrustedRatio <
                                    config.minimumGaussianTrustedRatio
                        )
                    ) ||
                (
                    input.externalContactEvidenceAvailable &&
                        input.externalContactConfidence < 0.24
                    ) ||
                rootScore < config.failureScore ||
                rootVelocityAgreement < 0.18 ||
                (
                    input.sceneEvidenceAvailable &&
                        scene < config.failureScore
                    )

        state.scoreEma =
            if (!state.initialized) {
                rawScore
            } else {
                state.scoreEma *
                    (1.0 - config.historyAlpha) +
                    rawScore * config.historyAlpha
            }

        if (immediateFailure) {
            state.failureFrames += 1
            state.healthyFrames = 0
        } else {
            state.failureFrames =
                max(0, state.failureFrames - 1)
            if (state.scoreEma >= config.warningScore) {
                state.healthyFrames += 1
            } else {
                state.healthyFrames = 0
            }
        }

        if (!state.initialized) {
            state.state = if (immediateFailure) {
                State.DEGRADED
            } else {
                State.WARMING
            }
        } else if (
            state.state == State.PERSISTENT_FAILURE &&
            !immediateFailure &&
            rawScore >= config.warningScore &&
            state.healthyFrames >= 1
        ) {
            state.state = State.RECOVERING
        } else if (
            state.failureFrames >=
                config.persistentFailureFrames
        ) {
            state.state = State.PERSISTENT_FAILURE
        } else if (
            state.scoreEma < config.failureScore
        ) {
            state.state = State.DEGRADED
        } else if (
            state.healthyFrames > 0 &&
            state.state != State.STABLE
        ) {
            state.state = State.RECOVERING
        } else if (
            state.scoreEma >= config.warningScore
        ) {
            state.state = State.STABLE
        } else {
            state.state = State.DEGRADED
        }

        val persistentFailure =
            state.failureFrames >=
                config.persistentFailureFrames ||
                state.state == State.PERSISTENT_FAILURE

        val accepted =
            state.scoreEma >= config.failureScore &&
                !persistentFailure &&
                !immediateFailure &&
                physics >= config.failureScore

        state.timestampNanos = timestamp
        state.initialized = true

        val warnings =
            buildWarnings(
                input = input,
                stateValue = state.state,
                score = state.scoreEma,
                rootScore = rootScore,
                rootVelocityAgreement = rootVelocityAgreement,
                bodyScaleConsistency = bodyScaleConsistency,
                contactScore = contact,
                sceneScore = scene,
                rootMetrics = rootMetrics,
            )

        val recommendations =
            buildRecommendations(
                input = input,
                stateValue = state.state,
                rootMetrics = rootMetrics,
                sceneScore = scene,
            )

        val result =
            Result(
                timestampNanos = timestamp,
                state = state.state,
                score = state.scoreEma.coerceIn(0.0, 1.0),
                rootMotionConsistency =
                    rootScore.coerceIn(0.0, 1.0),
                rootVelocityAgreement =
                    rootVelocityAgreement.coerceIn(0.0, 1.0),
                rootSpeedMetersPerSecond =
                    rootMetrics.speed,
                rootAccelerationMetersPerSecond2 =
                    rootMetrics.acceleration,
                rootJerkMetersPerSecond3 =
                    rootMetrics.jerk,
                physicsConsistency = physics,
                contactConsistency = contact,
                externalContactConsistency =
                    externalContactConsistency,
                symmetryConsistency = symmetry,
                cameraHumanConsistency = cameraHuman,
                sceneConsistency = scene,
                sceneEvidenceAvailable =
                    input.sceneEvidenceAvailable,
                temporalConsistency =
                    (
                        input.temporalScore *
                            input.synchronizationConfidence
                        ).coerceIn(0.0, 1.0),
                directionalUncertaintyConsistency =
                    directionalUncertaintyConsistency(input),
                faceEyeConsistency =
                    faceEyeConsistency(input),
                bodyScaleConsistency =
                    bodyScaleConsistency,
                rigStabilityConsistency =
                    rigStability,
                dynamicMetrologyConsistency =
                    dynamicMetrology,
                gaussian4DConsistency =
                    gaussian4D,
                gaussianStrainConsistency =
                    gaussianStrain,
                gaussianEvidenceAvailable =
                    input.gaussianEvidenceAvailable,
                crossModalMetricAuthority =
                    crossModalMetricAuthority(input),
                crossModalEpipolarAuthority =
                    crossModalEpipolarAuthority(input),
                persistentFailure = persistentFailure,
                accepted = accepted,
                warnings = warnings,
                recommendations = recommendations,
            )
        lastResult = result
        return result
    }

    private data class RootMetrics(
        val speed: Double,
        val acceleration: Double,
        val jerk: Double,
        val consistencyScore: Double,
    )

    private fun deriveRoot(
        skeleton: SkeletonFrame,
    ): Vec3? {
        val pelvisLeft = skeleton.joints.firstOrNull {
            it.landmarkId == 23
        }

        val pelvisRight = skeleton.joints.firstOrNull {
            it.landmarkId == 24
        }

        return when {
            pelvisLeft != null &&
                pelvisRight != null ->
                (pelvisLeft.position + pelvisRight.position) * 0.5

            pelvisLeft != null ->
                pelvisLeft.position

            pelvisRight != null ->
                pelvisRight.position

            else ->
                null
        }
    }

    private fun updateRootState(
        root: Vec3?,
        timestampNanos: Long,
        dtSeconds: Double,
    ): RootMetrics {
        if (root == null ||
            !root.x.isFinite() ||
            !root.y.isFinite() ||
            !root.z.isFinite()
        ) {
            return RootMetrics(
                speed = 0.0,
                acceleration = 0.0,
                jerk = 0.0,
                consistencyScore = 0.54,
            )
        }

        val previousRoot = state.rootPosition

        val measuredVelocity =
            if (previousRoot == null) {
                Vec3(0.0, 0.0, 0.0)
            } else {
                (root - previousRoot) /
                    dtSeconds.coerceAtLeast(1e-5)
            }

        val previousVelocity =
            state.rootVelocity

        val acceleration =
            (measuredVelocity - previousVelocity) /
                dtSeconds.coerceAtLeast(1e-5)

        val accelerationMagnitude =
            acceleration.norm()

        val jerk =
            abs(
                accelerationMagnitude -
                    state.previousAccelerationMagnitude,
            ) /
                dtSeconds.coerceAtLeast(1e-5)

        val speed =
            measuredVelocity.norm()

        val speedScore =
            (
                1.0 -
                    speed /
                        config.maximumRootSpeedMetersPerSecond
                ).coerceIn(0.0, 1.0)

        val jerkScore =
            (
                1.0 -
                    jerk /
                        config.maximumRootJerkMetersPerSecond3
                ).coerceIn(0.0, 1.0)

        val velocityAgreement =
            if (previousRoot == null) {
                0.76
            } else {
                (
                    1.0 -
                        (
                            measuredVelocity -
                                previousVelocity
                            ).norm() /
                            config.maximumRootSpeedMetersPerSecond
                    ).coerceIn(0.0, 1.0)
            }

        val confidence =
            rootConfidence(
                root = root,
                previousRoot = previousRoot,
                dtSeconds = dtSeconds,
            )

        state.rootPosition = root
        state.rootVelocity = measuredVelocity
        state.rootAcceleration = acceleration
        state.previousAccelerationMagnitude =
            accelerationMagnitude

        val consistency =
            (
                speedScore * 0.28 +
                    jerkScore * 0.24 +
                    velocityAgreement * 0.22 +
                    confidence * 0.26
                ).coerceIn(0.0, 1.0)

        return RootMetrics(
            speed = speed,
            acceleration = accelerationMagnitude,
            jerk = jerk,
            consistencyScore = consistency,
        )
    }

    private fun rootConfidence(
        root: Vec3,
        previousRoot: Vec3?,
        dtSeconds: Double,
    ): Double {
        if (!root.x.isFinite() ||
            !root.y.isFinite() ||
            !root.z.isFinite()
        ) {
            return 0.0
        }

        val finiteScore = 1.0
        val continuityScore =
            if (previousRoot == null) {
                0.76
            } else {
                (
                    1.0 -
                        root.distanceTo(previousRoot) /
                            (
                                config.maximumRootSpeedMetersPerSecond *
                                    dtSeconds.coerceIn(
                                        1.0 / 240.0,
                                        0.25,
                                    )
                                )
                    ).coerceIn(0.0, 1.0)
            }

        return (
            finiteScore * 0.54 +
                continuityScore * 0.46
            ).coerceIn(0.0, 1.0)
    }

    private fun reportedRootVelocityAgreement(
        skeleton: SkeletonFrame,
        measuredVelocity: Vec3,
    ): Double {
        val velocities =
            skeleton.joints
                .filter {
                    it.landmarkId == 23 ||
                        it.landmarkId == 24
                }
                .map { it.velocity }
                .filter {
                    it.x.isFinite() &&
                        it.y.isFinite() &&
                        it.z.isFinite()
                }

        if (velocities.isEmpty()) {
            return 0.72
        }

        val reported =
            velocities.reduce { left, right ->
                left + right
            } / velocities.size.toDouble()

        val disagreement =
            (reported - measuredVelocity).norm()

        return kotlin.math.exp(
            -disagreement /
                config.maximumRootSpeedMetersPerSecond
                    .coerceAtLeast(1e-6),
        ).coerceIn(0.0, 1.0)
    }

    private fun deriveContactConsistency(
        physicsAccepted: Boolean,
        physicsScore: Double,
    ): Double =
        (
            physicsScore * 0.72 +
                if (physicsAccepted) 1.0 else 0.24
            ).coerceIn(0.0, 1.0)

    private fun deriveSymmetryConsistency(
        skeleton: SkeletonFrame,
    ): Double {
        val leftShoulder = skeleton.joints.firstOrNull {
            it.landmarkId == 11
        }
        val rightShoulder = skeleton.joints.firstOrNull {
            it.landmarkId == 12
        }
        val leftHip = skeleton.joints.firstOrNull {
            it.landmarkId == 23
        }
        val rightHip = skeleton.joints.firstOrNull {
            it.landmarkId == 24
        }

        val leftSegment =
            leftShoulder?.position?.let { left ->
                leftHip?.position?.let { hip ->
                    left.distanceTo(hip)
                }
            }
        val rightSegment =
            rightShoulder?.position?.let { right ->
                rightHip?.position?.let { hip ->
                    right.distanceTo(hip)
                }
            }

        val segments = listOfNotNull(
            leftSegment,
            rightSegment,
        )

        if (segments.size < 2) return 0.70

        val mean = segments.average().coerceAtLeast(1e-6)
        val difference =
            abs(segments[0] - segments[1]) /
                mean

        return (
            1.0 -
                difference
            ).coerceIn(0.0, 1.0)
    }

    private fun deriveSceneConsistency(
        input: Input,
    ): Double {
        if (!input.sceneEvidenceAvailable) {
            return 0.72
        }

        val static =
            input.staticCoverage?.coerceIn(0.0, 1.0) ?: 0.0
        val dynamic =
            input.dynamicCoverage?.coerceIn(0.0, 1.0) ?: 0.0
        val unknown =
            input.unknownCoverage?.coerceIn(0.0, 1.0) ?: 1.0
        val decomposition =
            input.sceneDecompositionConfidence
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        val staticSupport =
            if (static >= config.staticCoverageMinimum) {
                min(
                    1.0,
                    static /
                        config.staticCoverageMinimum,
                )
            } else {
                static /
                    config.staticCoverageMinimum
            }

        val dynamicIsolation =
            (
                1.0 -
                    max(
                        0.0,
                        dynamic -
                            config.dynamicCoverageMaximum,
                    )
            ).coerceIn(0.0, 1.0)

        val ambiguity =
            1.0 -
                (
                    unknown /
                        config.unknownCoverageMaximum
                ).coerceIn(0.0, 1.0) *
                    config.sceneAmbiguityPenalty

        return (
            decomposition * 0.42 +
                staticSupport * 0.24 +
                dynamicIsolation * 0.20 +
                ambiguity * 0.14
            ).coerceIn(0.0, 1.0)
    }

    private fun rigStabilityConsistency(
        input: Input,
    ): Double {
        val translation =
            kotlin.math.exp(
                -input.rigTranslationDriftMeters /
                    config.maximumRigTranslationDriftMeters
            )
        val rotation =
            kotlin.math.exp(
                -input.rigRotationDriftRadians /
                    config.maximumRigRotationDriftRadians
            )
        return (
            input.rigStabilityConfidence *
                translation *
                rotation *
                if (input.rigUnstable) 0.18 else 1.0
            ).coerceIn(0.0, 1.0)
    }

    private fun dynamicMetrologyConsistency(
        input: Input,
    ): Double {
        val p95 =
            kotlin.math.exp(
                -input.dynamicMetrologyP95Millimeters /
                    config.maximumDynamicMetrologyP95Millimeters
            )
        val rmse =
            kotlin.math.exp(
                -input.dynamicMetrologyRmseMillimeters /
                    config.maximumDynamicMetrologyRmseMillimeters
            )
        val maxError =
            kotlin.math.exp(
                -input.dynamicMetrologyMaxErrorMillimeters /
                    config.maximumDynamicMetrologyMaxErrorMillimeters
            )
        val validity =
            if (input.dynamicMetrologyValid) 1.0 else 0.20
        val confidence =
            (
                input.dynamicMetrologyConfidence /
                    config.minimumDynamicMetrologyConfidence
            ).coerceIn(0.0, 1.0)

        return (
            p95 * 0.40 +
                rmse * 0.25 +
                maxError * 0.20 +
                confidence * 0.15
            ) * validity
    }

    private fun gaussian4DConsistency(
        input: Input,
    ): Double {
        if (!input.gaussianEvidenceAvailable) {
            return 1.0
        }

        return (
            input.gaussianTemporalStability * 0.34 +
                input.gaussianTrustedRatio * 0.28 +
                input.gaussianMeanNodeConfidence * 0.18 +
                input.gaussianMeanBindingConfidence * 0.20
            ).coerceIn(0.0, 1.0)
    }

    private fun gaussianStrainConsistency(
        input: Input,
    ): Double {
        if (!input.gaussianEvidenceAvailable) {
            return 0.72
        }

        val p95 = input.gaussianP95StrainRatio.coerceAtLeast(0.0)
        val maximum = input.gaussianMaximumStrainRatio.coerceAtLeast(0.0)

        val p95Score = (
            1.0 -
                p95 /
                    (config.maximumGaussianP95StrainRatio * 1.50)
                        .coerceAtLeast(1e-6)
        ).coerceIn(0.0, 1.0)
        val maximumScore = (
            1.0 -
                maximum /
                    (config.maximumGaussianMaximumStrainRatio * 1.50)
                        .coerceAtLeast(1e-6)
        ).coerceIn(0.0, 1.0)

        return (
            p95Score * 0.58 +
                maximumScore * 0.42
            ).coerceIn(0.0, 1.0)
    }

    private fun crossModalMetricAuthority(input: Input): Double =
        (
            0.34 * input.bodyMetricEvidenceAuthority +
                0.33 * input.faceMetricEvidenceAuthority +
                0.33 * input.handMetricEvidenceAuthority
            ).coerceIn(0.0, 1.0)

    private fun crossModalEpipolarAuthority(input: Input): Double =
        minOf(
            input.bodyEpipolarConsistency,
            input.faceEpipolarConsistency,
            input.handEpipolarConsistency,
        ).coerceIn(0.0, 1.0)

    private fun buildWarnings(
        input: Input,
        stateValue: State,
        score: Double,
        rootScore: Double,
        rootVelocityAgreement: Double,
        bodyScaleConsistency: Double,
        contactScore: Double,
        sceneScore: Double,
        rootMetrics: RootMetrics,
    ): List<String> = buildList {
        if (stateValue == State.WARMING) {
            add("global-4d-consistency-warming")
        }
        if (stateValue == State.DEGRADED) {
            add("global-4d-consistency-degraded")
        }
        if (stateValue == State.PERSISTENT_FAILURE) {
            add("global-4d-consistency-persistent-failure")
        }
        if (stateValue == State.RECOVERING) {
            add("global-4d-consistency-recovering")
        }
        if (score < config.warningScore) {
            add("global-4d-score-below-production-target")
        }
        if (rigStabilityConsistency(input) < config.warningScore) {
            add("global-4d-rig-stability-evidence")
        }
        if (dynamicMetrologyConsistency(input) < config.warningScore) {
            add("global-4d-dynamic-metrology-evidence")
        }
        if (crossModalMetricAuthority(input) < config.warningScore) {
            add("global-4d-cross-modal-metric-authority")
        }
        if (crossModalEpipolarAuthority(input) < config.warningScore) {
            add("global-4d-cross-modal-epipolar-authority")
        }
        if (
            input.gaussianEvidenceAvailable &&
            gaussian4DConsistency(input) < config.warningScore
        ) {
            add("global-4d-gaussian-temporal-evidence-limited")
        }
        if (
            input.gaussianEvidenceAvailable &&
            gaussianStrainConsistency(input) < config.warningScore
        ) {
            add("global-4d-gaussian-deformation-strain-elevated")
        }
        if (
            input.gaussianEvidenceAvailable &&
            input.gaussianTrustedRatio < config.minimumGaussianTrustedRatio
        ) {
            add("global-4d-gaussian-trusted-ratio-low")
        }
        if (rootScore < config.warningScore) {
            add("global-4d-root-motion-inconsistency")
        }
        if (rootVelocityAgreement < config.warningScore) {
            add("global-4d-root-velocity-evidence-mismatch")
        }
        if (bodyScaleConsistency < config.warningScore) {
            add("global-4d-body-scale-continuity-degraded")
        }
        if (rootMetrics.speed >
            config.maximumRootSpeedMetersPerSecond
        ) {
            add("global-4d-root-speed-envelope-exceeded")
        }
        if (
            rootMetrics.acceleration >
                config.maximumRootAccelerationMetersPerSecond2
        ) {
            add(
                "global-4d-root-acceleration-envelope-exceeded",
            )
        }
        if (rootMetrics.jerk >
            config.maximumRootJerkMetersPerSecond3
        ) {
            add("global-4d-root-jerk-envelope-exceeded")
        }
        if (input.cameraMotionRisk > 0.78) {
            add("global-4d-camera-motion-risk-high")
        }
        if (input.synchronizationConfidence < 0.62) {
            add("global-4d-synchronization-degraded")
        }
        if (input.rigTrust < 0.62) {
            add("global-4d-rig-trust-degraded")
        }
        if (
            input.directionalUncertaintyConfidence <
                config.minimumDirectionalConfidence
        ) {
            add("global-4d-directional-uncertainty-confidence-low")
        }
        if (
            input.directionalUncertaintyAnisotropy >=
                config.maximumDirectionalAnisotropy
        ) {
            add("global-4d-directional-uncertainty-anisotropy-high")
        }
        if (
            input.directionalUncertaintyDepthMeters >
                config.maximumDirectionalDepthUncertaintyMeters
        ) {
            add("global-4d-depth-uncertainty-elevated")
        }
        if (
            !input.faceEyeAccepted ||
            input.faceEyeConfidence <
                config.minimumFaceEyeConfidence
        ) {
            add("global-4d-face-eye-integrity-limited")
        }
        if (input.faceEyeOutOfDistributionRisk >= 0.70) {
            add("global-4d-face-eye-ood-risk-elevated")
        }
        if (contactScore < config.warningScore) {
            add("global-4d-contact-consistency-limited")
        }
        if (
            input.externalContactEvidenceAvailable &&
                input.externalContactConfidence < 0.20
        ) {
            add("global-4d-external-contact-evidence-weak")
        }
        if (input.sceneEvidenceAvailable.not()) {
            add("scene-decomposition-not-available")
        } else if (sceneScore < config.warningScore) {
            add("global-4d-scene-consistency-limited")
        }
    }.distinct()

    private fun buildRecommendations(
        input: Input,
        stateValue: State,
        rootMetrics: RootMetrics,
        sceneScore: Double,
    ): List<String> = buildList {
        if (
            stateValue == State.PERSISTENT_FAILURE ||
            stateValue == State.DEGRADED
        ) {
            add(
                "Hold long metric segments until human, camera and reconstruction evidence return to global consistency.",
            )
        }

        if (rootMetrics.jerk >
            config.maximumRootJerkMetersPerSecond3
        ) {
            add(
                "Inspect temporal alignment and fast root motion before smoothing the subject trajectory.",
            )
        }

        if (input.cameraMotionRisk > 0.78) {
            add(
                "Strengthen camera-motion compensation before trusting global human-scene alignment.",
            )
        }

        if (
            input.synchronizationConfidence < 0.62
        ) {
            add(
                "Improve cross-camera timing alignment before committing a globally consistent solve.",
            )
        }

        if (
            input.gaussianEvidenceAvailable &&
            (
                gaussianStrainConsistency(input) < config.warningScore ||
                    input.gaussianTrustedRatio <
                        config.minimumGaussianTrustedRatio
            )
        ) {
            add(
                "Hold dense 4D deformation until Gaussian temporal stability and strain evidence recover.",
            )
        }

        if (
            input.rigTrust < 0.62
        ) {
            add(
                "Revalidate rig geometry and camera contribution weights.",
            )
        }

        if (
            input.directionalUncertaintyAnisotropy >=
                config.maximumDirectionalAnisotropy ||
            input.directionalUncertaintyDepthMeters >
                config.maximumDirectionalDepthUncertaintyMeters
        ) {
            add(
                "Improve depth-axis observability and camera baseline diversity before trusting metric global alignment.",
            )
        }

        if (
            !input.faceEyeAccepted ||
            input.faceEyeConfidence <
                config.minimumFaceEyeConfidence
        ) {
            add(
                "Keep head/eye state in degraded mode until face/eye geometry returns to the production envelope.",
            )
        }

        if (
            input.sceneEvidenceAvailable &&
            sceneScore < config.warningScore
        ) {
            add(
                "Run dynamic/static scene refinement before final TSDF integration or novel-view export.",
            )
        }

        if (
            input.sceneEvidenceAvailable.not()
        ) {
            add(
                "Mark scene consistency as unresolved until dynamic/static decomposition evidence is available.",
            )
        }
    }.distinct().take(8)

    private fun deriveBodyScaleConsistency(
        skeleton: SkeletonFrame,
    ): Double {
        val edges =
            listOf(
                11 to 12,
                23 to 24,
                11 to 23,
                12 to 24,
                11 to 13,
                13 to 15,
                12 to 14,
                14 to 16,
            )

        val joints =
            skeleton.joints.associateBy { it.landmarkId }

        val lengths =
            edges.mapNotNull { (a, b) ->
                val left = joints[a] ?: return@mapNotNull null
                val right = joints[b] ?: return@mapNotNull null
                val length =
                    left.position.distanceTo(right.position)
                length.takeIf {
                    it.isFinite() && it > 0.01
                }
            }

        if (lengths.isEmpty()) {
            return 0.70
        }

        val sorted = lengths.sorted()
        val current =
            if (sorted.size % 2 == 1) {
                sorted[sorted.size / 2]
            } else {
                (
                    sorted[sorted.size / 2 - 1] +
                        sorted[sorted.size / 2]
                    ) * 0.5
            }

        val previous = state.previousBodyScaleMeters
        state.previousBodyScaleMeters = current

        if (previous == null || previous <= 1e-6) {
            return 0.82
        }

        val ratio =
            (
                current / previous
            ).coerceIn(0.50, 1.50)

        val relativeChange =
            abs(ratio - 1.0)

        return (
            1.0 -
                relativeChange /
                    config.maximumBodyScaleChangeRatioPerFrame
        ).coerceIn(0.0, 1.0)
    }

    private fun directionalUncertaintyConsistency(
        input: Input,
    ): Double {
        val confidence =
            input.directionalUncertaintyConfidence
                .coerceIn(0.0, 1.0)
        val anisotropyScore =
            (
                1.0 -
                    (
                        input.directionalUncertaintyAnisotropy - 1.0
                    ) /
                        (
                            config.maximumDirectionalAnisotropy - 1.0
                        ).coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)
        val depthScore =
            (
                1.0 -
                    input.directionalUncertaintyDepthMeters /
                        config.maximumDirectionalDepthUncertaintyMeters
                ).coerceIn(0.0, 1.0)

        return (
            confidence * 0.46 +
                anisotropyScore * 0.28 +
                depthScore * 0.26
            ).coerceIn(0.0, 1.0)
    }

    private fun faceEyeConsistency(
        input: Input,
    ): Double {
        val acceptance =
            if (input.faceEyeAccepted) 1.0 else 0.38
        val confidence =
            input.faceEyeConfidence.coerceIn(0.0, 1.0)
        val ood =
            1.0 -
                input.faceEyeOutOfDistributionRisk
                    .coerceIn(0.0, 1.0)

        return (
            confidence * 0.54 +
                acceptance * 0.22 +
                ood * 0.24
            ).coerceIn(0.0, 1.0)
    }

    private fun weightedGeometricMean(
        values: List<Pair<Double, Double>>,
    ): Double {
        if (values.isEmpty()) return 0.0

        var totalWeight = 0.0
        var weightedLog = 0.0

        values.forEach { (value, weight) ->
            val sanitized =
                value.coerceIn(0.015, 1.0)
            val boundedWeight =
                weight.coerceAtLeast(0.0)
            totalWeight += boundedWeight
            weightedLog +=
                kotlin.math.ln(sanitized) *
                    boundedWeight
        }

        return if (totalWeight <= 1e-9) {
            0.0
        } else {
            kotlin.math.exp(
                weightedLog /
                    totalWeight,
            ).coerceIn(0.0, 1.0)
        }
    }

    private fun duplicateOrJitterResult(
        timestampNanos: Long,
        previousTimestampNanos: Long,
    ): Result {
        val previous = lastResult
            ?: return duplicateTimestampResult()

        val backwardsNanos =
            (previousTimestampNanos - timestampNanos)
                .coerceAtLeast(0L)

        val withinJitterTolerance =
            backwardsNanos <=
                config.monotonicJitterToleranceNanos

        return previous.copy(
            timestampNanos = previousTimestampNanos,
            warnings = (
                previous.warnings +
                    if (withinJitterTolerance) {
                        "global-4d-timestamp-jitter-sample-ignored"
                    } else {
                        "global-4d-out-of-order-timestamp-ignored"
                    }
                ).distinct(),
            recommendations = (
                previous.recommendations +
                    "Retain monotonic global 4D state; do not reinject duplicate or older samples into kinematic estimation."
                ).distinct().take(8),
        )
    }

    private fun duplicateTimestampResult(): Result =
        Result(
            timestampNanos = state.timestampNanos,
            state = state.state,
            score = state.scoreEma.coerceIn(0.0, 1.0),
            rootMotionConsistency = state.scoreEma.coerceIn(0.0, 1.0),
            rootSpeedMetersPerSecond = state.rootVelocity.norm(),
            rootAccelerationMetersPerSecond2 =
                state.rootAcceleration.norm(),
            rootJerkMetersPerSecond3 = 0.0,
            physicsConsistency = state.scoreEma.coerceIn(0.0, 1.0),
            contactConsistency = state.scoreEma.coerceIn(0.0, 1.0),
            symmetryConsistency = state.scoreEma.coerceIn(0.0, 1.0),
            cameraHumanConsistency = state.scoreEma.coerceIn(0.0, 1.0),
            sceneConsistency = 0.72,
            sceneEvidenceAvailable = false,
            temporalConsistency = state.scoreEma.coerceIn(0.0, 1.0),
            directionalUncertaintyConsistency = 0.0,
            faceEyeConsistency = 0.0,
            bodyScaleConsistency = 0.0,
            gaussian4DConsistency = 0.0,
            gaussianStrainConsistency = 0.0,
            gaussianEvidenceAvailable = false,
            persistentFailure =
                state.state == State.PERSISTENT_FAILURE,
            accepted =
                state.state != State.PERSISTENT_FAILURE &&
                    state.scoreEma >= config.failureScore,
            warnings = listOf(
                "global-4d-duplicate-or-out-of-order-timestamp",
            ),
            recommendations = listOf(
                "Discard duplicate global consistency telemetry and retain the latest monotonic sample.",
            ),
        )
}
