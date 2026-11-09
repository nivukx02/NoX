package com.nivukx.nox

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.ui.GlassCard
import com.nivukx.nox.ui.GlassMetric
import com.nivukx.nox.ui.SectionTitle
import com.nivukx.nox.ui.StatusPill

/**
 * High-level operator decision surface.
 *
 * This panel intentionally summarizes existing evidence rather than duplicating the individual
 * subsystem dashboards. It keeps the primary operator decision visible: whether NoX currently has
 * enough coherent evidence to trust the reconstructed motion state.
 */
@Composable
fun ReconstructionIntelligencePanel(
    productionQuality: Double,
    productionPolicy: String,
    productionBottleneck: String,
    productionOutputAllowed: Boolean,
    motionPhase: String,
    motionDynamicsScore: Double,
    dynamicStress: Double,
    identityContinuity: Double,
    identitySwapRisk: Double,
    identityAction: String,
    identityEvidenceFusionScore: Double = 0.0,
    identitySelectedModalityCount: Int = 0,
    identityEvidenceWarnings: List<String> = emptyList(),
    faceEyeFusionConfidence: Double = 0.0,
    faceEyeFusionMode: String = "HOLD",
    faceEyeIndependentEvidenceSupport: Double = 0.0,
    faceEyeReliabilityTrend: Double = 0.0,
    faceEyeSaccadeVelocityDegreesPerSecond: Double = 0.0,
    faceEyeFusionAccepted: Boolean = false,
    faceEyeFusionWarnings: List<String> = emptyList(),
    faceEyeCaptureDomainConfidence: Double = 0.0,
    faceEyeLightingConfidence: Double = 0.0,
    faceEyeScaleConfidence: Double = 0.0,
    faceEyeUncertaintyCalibrationConfidence: Double = 0.0,
    faceEyeUncertaintyCalibrationReady: Boolean = false,
    faceUncertaintyCalibrationMultiplier: Double = 1.0,
    eyeUncertaintyCalibrationMultiplier: Double = 1.0,
    faceUncertaintyCoverage: Double = 0.0,
    eyeUncertaintyCoverage: Double = 0.0,
    faceEyeFailureAttribution: List<String> = emptyList(),
    faceEyeGeometricIntegrity: Double = 0.0,
    faceEyeFaceFreshnessScore: Double = 0.0,
    faceEyeGazeAgreementScore: Double = 0.0,
    faceEyeRayGeometryCondition: Double? = null,
    faceEyeIpdConsistencyScore: Double = 1.0,
    faceEyeOutOfDistributionRisk: Double = 0.0,
    faceEyeShapeConfidence: Double = 1.0,
    faceEyeEyeDynamicConsistency: Double = 1.0,
    faceEyeHardFailureReasons: List<String> = emptyList(),
    faceEyeCalibrationDriftScore: Double = 0.0,
    faceEyeRecalibrationRecommended: Boolean = false,
    faceEyeCalibrationDriftWarnings: List<String> = emptyList(),
    adaptiveViewEvidenceGlobalRigTrust: Double = 1.0,
    cameraContributionInformationGainScore: Double = 0.0,
    surfaceConfidence: Double,
    denseTsdfSurfaceNormalObservability: Double = 1.0,
    denseTsdfMeanEvidenceWeight: Double = 0.0,
    temporalDepthConsistency: Double = 0.0,
    trustedJointRatio: Double,
    physicsConstraintScore: Double = 1.0,
    physicsConstraintAccepted: Boolean = true,
    physicsConstraintMeanBoneResidualMeters: Double = 0.0,
    physicsConstraintP95BoneResidualMeters: Double = 0.0,
    physicsConstraintMaxBoneResidualMeters: Double = 0.0,
    physicsConstraintSymmetryScore: Double = 1.0,
    physicsConstraintFloorContactScore: Double = 1.0,
    physicsConstraintCorrectedJointCount: Int = 0,
    physicsConstraintConstrainedEdgeCount: Int = 0,
    physicsConstraintWarnings: List<String> = emptyList(),
    directionalUncertaintyMajor95Meters: Double = 0.0,
    directionalUncertaintyDepthMeters: Double = 0.0,
    directionalUncertaintyAnisotropy: Double = 1.0,
    directionalUncertaintyConfidence: Double = 1.0,
    directionalUncertaintyDegradedJointCount: Int = 0,
    directionalUncertaintyWarnings: List<String> = emptyList(),
    global4DScore: Double = 1.0,
    global4DAccepted: Boolean = true,
    global4DState: String = "WARMING",
    global4DPersistentFailure: Boolean = false,
    global4DRootMotionConsistency: Double = 1.0,
    global4DRootVelocityAgreement: Double = 1.0,
    global4DSceneConsistency: Double = 0.72,
    global4DDirectionalUncertaintyConsistency: Double = 1.0,
    global4DFaceEyeConsistency: Double = 1.0,
    global4DWarnings: List<String> = emptyList(),
    worldSpaceConsistencyScore: Double = 1.0,
    worldSpaceAccepted: Boolean = true,
    worldSpaceTemporalStability: Double = 1.0,
    worldSpaceMotionContinuity: Double = 1.0,
    worldSpaceRejectedJointCount: Int = 0,
    worldSpaceRobustOutlierRatio: Double = 0.0,
    worldSpaceWarnings: List<String> = emptyList(),
    modifier: Modifier = Modifier,
) {
    val operatorState =
        when {
            !productionOutputAllowed -> "HOLD"
            global4DPersistentFailure -> "GLOBAL 4D HOLD"
            !global4DAccepted -> "VERIFY GLOBAL 4D"
            global4DScore < 0.58 -> "GLOBAL 4D DEGRADED"
            faceEyeOutOfDistributionRisk >= 0.92 ->
                "FACE / EYE OOD"
            faceEyeFusionMode == "HOLD" ->
                "FACE / EYE HOLD"
            !faceEyeFusionAccepted -> "VERIFY FACE / EYE"
            faceEyeFusionConfidence < 0.55 -> "FACE / EYE DEGRADED"
            adaptiveViewEvidenceGlobalRigTrust < 0.45 ->
                "VERIFY CAMERA RIG"
            adaptiveViewEvidenceGlobalRigTrust < 0.62 ->
                "RIG DEGRADED"
            !physicsConstraintAccepted -> "VERIFY PHYSICS"
            physicsConstraintScore < 0.58 -> "PHYSICS DEGRADED"
            identitySwapRisk >= 0.55 -> "VERIFY IDENTITY"
            dynamicStress >= 0.72 -> "HIGH DYNAMICS"
            productionQuality >= 0.82 -> "REFERENCE READY"
            productionQuality >= 0.62 -> "TRACKING"
            else -> "ACQUIRING"
        }

    GlassCard(
        modifier.fillMaxWidth(),
        intensity = 0.99f,
    ) {
        Column(
            Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                SectionTitle(
                    "RECONSTRUCTION INTELLIGENCE",
                    "One operator surface for motion, identity and production-quality decisions.",
                )
                StatusPill(
                    operatorState,
                    productionOutputAllowed && identitySwapRisk < 0.55,
                )
            }

            Text(
                buildString {
                    append(productionPolicy)
                    append(" / ")
                    append(productionBottleneck)
                    append(" / ")
                    append(identityAction)
                    append(" / ")
                    append(motionPhase)
                },
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Production",
                    "%.0f%%".format(productionQuality * 100.0),
                    productionPolicy,
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Identity",
                    "%.0f%%".format(identityContinuity * 100.0),
                    "continuity",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Swap risk",
                    "%.0f%%".format(identitySwapRisk * 100.0),
                    identityAction,
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Evidence fusion",
                    "%.0f%%".format(
                        identityEvidenceFusionScore * 100.0,
                    ),
                    identitySelectedModalityCount.toString() + " modalities",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Face / eye",
                    "%.0f%%".format(
                        faceEyeFusionConfidence * 100.0,
                    ),
                    if (faceEyeFusionAccepted) {
                        "accepted"
                    } else {
                        "degraded"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Fusion mode",
                    faceEyeFusionMode.replace("_", " "),
                    "%.0f%% support".format(
                        faceEyeIndependentEvidenceSupport * 100.0,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Capture domain",
                    "%.0f%%".format(
                        faceEyeCaptureDomainConfidence * 100.0,
                    ),
                    "lighting %.0f%%".format(
                        faceEyeLightingConfidence * 100.0,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Face scale",
                    "%.0f%%".format(
                        faceEyeScaleConfidence * 100.0,
                    ),
                    if (faceEyeScaleConfidence < 0.55) {
                        "reframe"
                    } else {
                        "usable"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Reliability trend",
                    "%+.1f".format(
                        faceEyeReliabilityTrend * 100.0,
                    ),
                    if (faceEyeReliabilityTrend < -0.08) {
                        "declining"
                    } else if (faceEyeReliabilityTrend > 0.08) {
                        "recovering"
                    } else {
                        "stable"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Gaze burst",
                    "%.0f°/s".format(
                        faceEyeSaccadeVelocityDegreesPerSecond,
                    ),
                    if (faceEyeSaccadeVelocityDegreesPerSecond >= 700.0) {
                        "high-frequency priority"
                    } else {
                        "steady"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "View gain",
                    "%.0f%%".format(
                        cameraContributionInformationGainScore * 100.0,
                    ),
                    "uncertainty reduction",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Geometry",
                    "%.0f%%".format(
                        faceEyeGeometricIntegrity * 100.0,
                    ),
                    "%.0f%% gaze".format(
                        faceEyeGazeAgreementScore * 100.0,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Root velocity",
                    "%.0f%%".format(
                        global4DRootVelocityAgreement * 100.0,
                    ),
                    "%.0f%% root".format(
                        global4DRootMotionConsistency * 100.0,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "World-space",
                    "%.0f%%".format(
                        worldSpaceConsistencyScore * 100.0,
                    ),
                    if (worldSpaceAccepted) {
                        "%.0f%% continuity".format(
                            worldSpaceMotionContinuity * 100.0,
                        )
                    } else {
                        "HOLD / VERIFY"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Ray conditioning",
                    faceEyeRayGeometryCondition?.let {
                        "%.0f%%".format(it * 100.0)
                    } ?: "N/A",
                    "binocular geometry",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "IPD consistency",
                    "%.0f%%".format(
                        faceEyeIpdConsistencyScore * 100.0,
                    ),
                    "session-calibrated",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Face shape",
                    "%.0f%%".format(
                        faceEyeShapeConfidence * 100.0,
                    ),
                    if (faceEyeHardFailureReasons.any {
                        it.startsWith("face-")
                    }) {
                        "GATE LIMITED"
                    } else {
                        "geometry accepted"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Eye dynamics",
                    "%.0f%%".format(
                        faceEyeEyeDynamicConsistency * 100.0,
                    ),
                    if (faceEyeEyeDynamicConsistency < 0.55) {
                        "high dynamics"
                    } else {
                        "stable"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Face / eye OOD",
                    "%.0f%%".format(
                        faceEyeOutOfDistributionRisk * 100.0,
                    ),
                    if (faceEyeOutOfDistributionRisk >= 0.72) {
                        "elevated"
                    } else {
                        "stable"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Calibration drift",
                    "%.0f%%".format(
                        faceEyeCalibrationDriftScore * 100.0,
                    ),
                    if (faceEyeRecalibrationRecommended) {
                        "RECALIBRATE"
                    } else {
                        "stable"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Uncertainty calibration",
                    "%.0f%%".format(
                        faceEyeUncertaintyCalibrationConfidence * 100.0,
                    ),
                    if (faceEyeUncertaintyCalibrationReady) {
                        "coverage %.0f/%.0f%%".format(
                            faceUncertaintyCoverage * 100.0,
                            eyeUncertaintyCoverage * 100.0,
                        )
                    } else {
                        "warm-up"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Face / eye widening",
                    "%.2fx".format(
                        faceUncertaintyCalibrationMultiplier,
                    ),
                    "%.2fx eye".format(
                        eyeUncertaintyCalibrationMultiplier,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Rig trust",
                    "%.0f%%".format(
                        adaptiveViewEvidenceGlobalRigTrust * 100.0,
                    ),
                    if (adaptiveViewEvidenceGlobalRigTrust >= 0.62) {
                        "global calibration/sync"
                    } else {
                        "rig degraded"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Surface",
                    "%.0f%%".format(surfaceConfidence * 100.0),
                    "dense evidence",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "View geometry",
                    "%.0f%%".format(
                        denseTsdfSurfaceNormalObservability * 100.0,
                    ),
                    "TSDF normal observability",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "TSDF evidence",
                    "%.0f%%".format(
                        denseTsdfMeanEvidenceWeight * 100.0,
                    ),
                    "weighted fusion",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Depth temporal",
                    "%.0f%%".format(
                        temporalDepthConsistency * 100.0,
                    ),
                    "camera compensated",
                    Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "3D uncertainty",
                    "%.0f mm".format(
                        directionalUncertaintyMajor95Meters * 1000.0,
                    ),
                    "%.0f mm depth".format(
                        directionalUncertaintyDepthMeters * 1000.0,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Anisotropy",
                    "%.1fx".format(
                        directionalUncertaintyAnisotropy,
                    ),
                    "%.0f%% confidence".format(
                        directionalUncertaintyConfidence * 100.0,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Depth geometry",
                    if (
                        directionalUncertaintyDepthMeters >
                            directionalUncertaintyMajor95Meters * 0.72
                    ) {
                        "ELEVATED"
                    } else {
                        "CONTROLLED"
                    },
                    directionalUncertaintyDegradedJointCount.toString() +
                        " joints",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Global 4D",
                    "%.0f%%".format(global4DScore * 100.0),
                    global4DState +
                        " / root %.0f%%".format(
                            global4DRootMotionConsistency * 100.0,
                        ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Scene consistency",
                    "%.0f%%".format(
                        global4DSceneConsistency * 100.0,
                    ),
                    if (global4DAccepted) "human + camera" else "recovery",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "4D geometry",
                    "%.0f%%".format(
                        global4DDirectionalUncertaintyConsistency * 100.0,
                    ),
                    "uncertainty",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "4D face / eye",
                    "%.0f%%".format(
                        global4DFaceEyeConsistency * 100.0,
                    ),
                    "head integrity",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Physics guard",
                    "%.0f%%".format(physicsConstraintScore * 100.0),
                    if (physicsConstraintAccepted) {
                        "kinematic envelope"
                    } else {
                        "HOLD / VERIFY"
                    },
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Bone P95",
                    "%.0f mm".format(
                        physicsConstraintP95BoneResidualMeters * 1000.0,
                    ),
                    "%.0f mm max".format(
                        physicsConstraintMaxBoneResidualMeters * 1000.0,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Symmetry",
                    "%.0f%%".format(
                        physicsConstraintSymmetryScore * 100.0,
                    ),
                    "%.0f%% floor".format(
                        physicsConstraintFloorContactScore * 100.0,
                    ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Corrections",
                    physicsConstraintCorrectedJointCount.toString(),
                    physicsConstraintConstrainedEdgeCount.toString() + " edges",
                    Modifier.weight(1f),
                )
            }

            if (
                faceEyeUncertaintyCalibrationConfidence > 0.0 &&
                (
                    !faceEyeUncertaintyCalibrationReady ||
                        faceUncertaintyCoverage < 0.84 ||
                        eyeUncertaintyCoverage < 0.84
                    )
            ) {
                Text(
                    "uncertainty calibration requires more stable face/eye evidence",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (faceEyeCalibrationDriftWarnings.isNotEmpty()) {
                Text(
                    faceEyeCalibrationDriftWarnings
                        .take(2)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (faceEyeFusionWarnings.isNotEmpty()) {
                Text(
                    faceEyeFusionWarnings
                        .take(2)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (faceEyeFailureAttribution.isNotEmpty()) {
                Text(
                    faceEyeFailureAttribution
                        .take(3)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (faceEyeHardFailureReasons.isNotEmpty()) {
                Text(
                    faceEyeHardFailureReasons
                        .take(3)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (identityEvidenceWarnings.isNotEmpty()) {
                Text(
                    identityEvidenceWarnings
                        .take(2)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (global4DWarnings.isNotEmpty()) {
                Text(
                    global4DWarnings
                        .take(2)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (directionalUncertaintyWarnings.isNotEmpty()) {
                Text(
                    directionalUncertaintyWarnings
                        .take(2)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (worldSpaceWarnings.isNotEmpty()) {
                Text(
                    worldSpaceWarnings
                        .take(2)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (physicsConstraintWarnings.isNotEmpty()) {
                Text(
                    physicsConstraintWarnings
                        .take(2)
                        .joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

                        Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Motion state",
                    "%.0f%%".format(motionDynamicsScore * 100.0),
                    motionPhase,
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Dynamics stress",
                    "%.0f%%".format(dynamicStress * 100.0),
                    "jerk regime",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Trusted joints",
                    "%.0f%%".format(trustedJointRatio * 100.0),
                    "3D state",
                    Modifier.weight(1f),
                )
            }

            val guidance =
                when {
                    !productionOutputAllowed ->
                        "Hold export and improve the active production bottleneck before trusting the final solve."
                    !faceEyeFusionAccepted ->
                        "Face/eye evidence is outside the configured production envelope. Hold gaze-sensitive output until binocular and temporal agreement recovers."
                    adaptiveViewEvidenceGlobalRigTrust < 0.45 ->
                        "Global camera-rig trust is low. Hold metric reconstruction until calibration and synchronization evidence recover."
                    adaptiveViewEvidenceGlobalRigTrust < 0.62 ->
                        "Camera-rig evidence is degraded. Preserve continuity, but avoid committing high-precision multi-view geometry."
                    cameraContributionInformationGainScore < 0.48 ->
                        "Current camera selection has limited information gain. Prefer additional calibrated viewpoints before committing dense geometry."
                    !physicsConstraintAccepted ->
                        "Physics consistency is outside the configured production envelope. Preserve the solve, but hold metric output until geometry recovers."
                    identitySwapRisk >= 0.55 ->
                        "Keep the current track locked and prefer additional view coverage before accepting an identity switch."
                    dynamicStress >= 0.72 ->
                        "Preserve temporal samples and prioritize fast body/hand updates during this high-dynamics interval."
                    surfaceConfidence < 0.55 ->
                        "Keep skeleton output authoritative and accumulate stronger surface evidence before dense export."
                    else ->
                        "Current evidence is coherent enough for normal operator capture and downstream refinement."
                }

            Text(
                guidance,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
