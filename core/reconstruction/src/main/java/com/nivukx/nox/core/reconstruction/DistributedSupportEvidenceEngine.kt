package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId

/**
 * Cheap post-solve witness pass for cameras outside the authoritative reconstruction subset.
 *
 * The expensive pose solver remains bounded by the selected-camera budget. Every additional
 * calibrated camera can still participate as a validation witness by reprojecting the reconstructed
 * skeleton into that view. This improves 128-camera federation coverage without multiplying the
 * full nonlinear solve by 4x.
 *
 * Support evidence is never used to invent or directly move a joint. It can:
 * - strengthen independent cross-view agreement when healthy;
 * - expose systematic contradictions in cameras that were not selected for the expensive solve;
 * - identify weak support cameras for operator diagnostics;
 * - reduce global confidence when many independent support cameras disagree.
 */
class DistributedSupportEvidenceEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumSupportingCameras: Int = 1,
        val maximumSupportingCameras: Int = 96,
        val minimumSupportScore: Double = 0.42,
        val contradictionScore: Double = 0.45,
        val hardContradictionRisk: Double = 0.72,
        val supportAgreementWeight: Double = 0.08,
        val maximumWitnessResidualPx: Double = 18.0,
    ) {
        init {
            require(minimumSupportingCameras >= 1)
            require(maximumSupportingCameras >= minimumSupportingCameras)
            require(minimumSupportScore in 0.0..1.0)
            require(contradictionScore in 0.0..1.0)
            require(hardContradictionRisk in 0.0..1.0)
            require(supportAgreementWeight in 0.0..0.25)
            require(maximumWitnessResidualPx > 0.0)
        }
    }

    data class Input(
        val skeleton:
            com.nivukx.nox.core.model.SkeletonFrame,
        val selectedCameraIds: Set<String>,
        val cameraFrames:
            List<MultiViewReprojectionConsistencyEngine.CameraFrame>,
        val targetTimestampNanos: Long,
        val synchronizationConfidence: Double = 1.0,
        val cameraMotionRisk: Double = 0.0,
        val calibrationReprojectionErrorPx: Double = 0.0,
    ) {
        init {
            require(targetTimestampNanos > 0L)
            require(synchronizationConfidence in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
            require(calibrationReprojectionErrorPx >= 0.0)
        }
    }

    data class Result(
        val score: Double,
        val supportingCameraCount: Int,
        val rejectedSupportingCameraCount: Int,
        val supportingJointCount: Int,
        val contradictedJointCount: Int,
        val contradictionRisk: Double,
        val meanResidualPx: Double,
        val p95ResidualPx: Double,
        val inlierRatio: Double,
        val authoritativeSupport: Boolean,
        val warnings: List<String>,
    ) {
        val independentAgreementContribution: Double
            get() =
                (
                    score *
                        inlierRatio *
                        (
                            supportingCameraCount /
                                (supportingCameraCount + 4.0)
                        )
                ).coerceIn(0.0, 1.0)
    }

    @Synchronized
    fun evaluate(
        input: Input,
    ): Result {
        val supportFrames =
            input.cameraFrames
                .asSequence()
                .filter {
                    it.deviceId.value !in
                        input.selectedCameraIds
                }
                .filter {
                    it.observations.isNotEmpty()
                }
                .take(config.maximumSupportingCameras)
                .toList()

        if (supportFrames.isEmpty()) {
            return Result(
                score = 0.0,
                supportingCameraCount = 0,
                rejectedSupportingCameraCount = 0,
                supportingJointCount = 0,
                contradictedJointCount = 0,
                contradictionRisk = 0.0,
                meanResidualPx = 0.0,
                p95ResidualPx = 0.0,
                inlierRatio = 0.0,
                authoritativeSupport = false,
                warnings =
                    listOf(
                        "SUPPORT_EVIDENCE_NO_NON_SELECTED_CAMERAS",
                    ),
            )
        }

        val observer =
            MultiViewReprojectionConsistencyEngine(
                MultiViewReprojectionConsistencyEngine.Config(
                    minimumViewsPerJoint = 1,
                    hardRejectResidualPx =
                        config.maximumWitnessResidualPx,
                    fatalResidualPx =
                        config.maximumWitnessResidualPx * 2.5,
                    globalAcceptanceScore = 0.35,
                    globalSoftWarningScore = 0.62,
                ),
            )

        val evaluation =
            observer.evaluate(
                MultiViewReprojectionConsistencyEngine.Input(
                    skeleton = input.skeleton,
                    cameraFrames = supportFrames,
                    targetTimestampNanos = input.targetTimestampNanos,
                    calibrationReprojectionErrorPx =
                        input.calibrationReprojectionErrorPx,
                    synchronizationConfidence =
                        input.synchronizationConfidence,
                    cameraMotionRisk =
                        input.cameraMotionRisk,
                ),
            )

        val diagnostics = evaluation.diagnostics

        val supportingJoints =
            diagnostics.count {
                it.acceptedViews > 0 &&
                    it.finalScore >= config.minimumSupportScore
            }

        val contradictedJoints =
            diagnostics.count {
                it.acceptedViews == 0 ||
                    it.finalScore <
                    config.contradictionScore
            }

        val totalJoints =
            diagnostics.size.coerceAtLeast(1)

        val contradictionRisk =
            (
                contradictedJoints.toDouble() /
                    totalJoints.toDouble()
            ).coerceIn(0.0, 1.0) *
                (
                    supportFrames.size /
                        (supportFrames.size + 3.0)
                ).coerceIn(0.0, 1.0)

        val rejectedCameraIds =
            diagnostics
                .flatMap { it.views }
                .filter {
                    it.rejected &&
                        it.reprojectionErrorPx.isFinite() &&
                        it.reprojectionErrorPx >
                        config.contradictionScore *
                            config.maximumWitnessResidualPx
                }
                .map { it.deviceId.value }
                .distinct()

        val supportingCameraCount =
            supportFrames.count { frame ->
                diagnostics.any { diagnostic ->
                    diagnostic.views.any {
                        !it.rejected &&
                            it.deviceId == frame.deviceId
                    }
                }
            }

        val score =
            (
                evaluation.score * 0.56 +
                    evaluation.viewAgreementScore * 0.20 +
                    evaluation.inlierRatio * 0.16 +
                    evaluation.temporalConsistencyScore * 0.08
                ).coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (
                    supportFrames.size <
                        config.minimumSupportingCameras
                ) {
                    add(
                        "SUPPORT_EVIDENCE_SUPPORT_CAMERA_COUNT_LOW",
                    )
                }
                if (
                    score <
                        config.minimumSupportScore
                ) {
                    add(
                        "SUPPORT_EVIDENCE_SUPPORT_SCORE_LOW",
                    )
                }
                if (
                    contradictionRisk >=
                        config.contradictionScore
                ) {
                    add(
                        "SUPPORT_EVIDENCE_CONTRADICTION_HIGH",
                    )
                }
                if (
                    contradictionRisk >=
                        config.hardContradictionRisk
                ) {
                    add(
                        "SUPPORT_EVIDENCE_HARD_CONTRADICTION",
                    )
                }
                if (rejectedCameraIds.isNotEmpty()) {
                    add(
                        "SUPPORT_EVIDENCE_WEAK_CAMERAS:" +
                            rejectedCameraIds
                                .take(8)
                                .joinToString(","),
                    )
                }
            }.distinct()

        return Result(
            score = score,
            supportingCameraCount = supportingCameraCount,
            rejectedSupportingCameraCount =
                supportFrames.size -
                    supportingCameraCount,
            supportingJointCount = supportingJoints,
            contradictedJointCount = contradictedJoints,
            contradictionRisk = contradictionRisk,
            meanResidualPx =
                evaluation.meanResidualPx,
            p95ResidualPx =
                evaluation.p95ResidualPx,
            inlierRatio =
                evaluation.inlierRatio,
            authoritativeSupport =
                supportingCameraCount >=
                    config.minimumSupportingCameras &&
                    score >=
                    config.minimumSupportScore &&
                    contradictionRisk <
                    config.contradictionScore,
            warnings = warnings,
        )
    }
}
