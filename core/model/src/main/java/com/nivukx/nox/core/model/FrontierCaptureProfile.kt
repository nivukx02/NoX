package com.nivukx.nox.core.model

import java.security.MessageDigest

/** Immutable frontier capture policy that layers on top of HighEndMotionCaptureSettings. */
data class FrontierCaptureProfile(
    val schemaVersion: Int = 3,
    val id: String = "frontier-precision",
    val resolution: CameraResolutionPreset = CameraResolutionPreset.AUTO,
    val targetFps: Int = 60,
    val inferenceFps: Int = 30,
    val shutterMicros: Long? = 8_000L,
    val iso: Int? = 400,
    val preciseColorTemperatureEnabled: Boolean = false,
    val colorTemperatureKelvin: Int = 5000,
    val colorTint: Int = 0,
    val stabilizationEnabled: Boolean = false,
    val rawEnabled: Boolean = true,
    val depthEnabled: Boolean = true,
    val denseSurfaceEnabled: Boolean = true,
    val gaussianRefinementEnabled: Boolean = true,
    val sparseViewRecoveryEnabled: Boolean = true,
    val worldSpaceLearnedProposalsEnabled: Boolean = true,
    val targetActorCount: Int = 50,
    val hardActorCeiling: Int = 64,
    val authoritativeCameraLimit: Int = 32,
    val globalCameraLimit: Int = 128,
    val precisionTargetMillimeters: Double = 1.0,
    val minimumIndependentViews: Int = 3,
    val minimumEmpiricalSamples: Int = 600,
    val reducedTransparency: Boolean = false,
    val reducedMotion: Boolean = false,
    val maximumFrameBudgetMillis: Double = 16.67,
    val maximumClockSkewMillis: Double = 4.0,
    val maximumCalibrationResidualPixels: Double = 2.0,
    val minimumRigConfidence: Double = 0.90,
    val maximumThermalRisk: Double = 0.60,
    val maximumMemoryPressure: Double = 0.75,
    val minimumTransportHeadroom: Double = 0.55,
    val referenceQualificationRequired: Boolean = true,
    val maximumMotionBlurRiskForAuthority: Double = 0.30,
    val minimumEpipolarConsistencyForAuthority: Double = 0.80,
    val minimumMetricEvidenceAuthorityForAuthority: Double = 0.78,
    val minimumCaptureAuthorityScore: Double = 0.88,
    val precisionAuthorityMinimumScore: Double = 0.93,
    val precisionAuthorityMinimumSamples: Int = 600,
) {
    init {
        require(schemaVersion in 1..3)
        require(id.isNotBlank())
        require(targetFps in 15..240)
        require(inferenceFps in 5..240)
        require(inferenceFps <= targetFps)
        require(shutterMicros == null || shutterMicros > 0L)
        require(iso == null || iso in 50..25_600)
        require(colorTemperatureKelvin in 2000..10000)
        require(colorTint in -100..100)
        require(targetActorCount in 1..64)
        require(hardActorCeiling in targetActorCount..64)
        require(authoritativeCameraLimit in 2..32)
        require(globalCameraLimit in authoritativeCameraLimit..128)
        require(precisionTargetMillimeters.isFinite() && precisionTargetMillimeters > 0.0)
        require(minimumIndependentViews >= 2)
        require(minimumEmpiricalSamples >= 600)
        require(maximumFrameBudgetMillis.isFinite() && maximumFrameBudgetMillis in 1.0..1000.0)
        require(maximumClockSkewMillis.isFinite() && maximumClockSkewMillis in 0.1..100.0)
        require(maximumCalibrationResidualPixels.isFinite() && maximumCalibrationResidualPixels in 0.05..50.0)
        require(minimumRigConfidence.isFinite() && minimumRigConfidence in 0.0..1.0)
        require(maximumThermalRisk.isFinite() && maximumThermalRisk in 0.0..1.0)
        require(maximumMemoryPressure.isFinite() && maximumMemoryPressure in 0.0..1.0)
        require(minimumTransportHeadroom.isFinite() && minimumTransportHeadroom in 0.0..1.0)
        require(maximumMotionBlurRiskForAuthority.isFinite() && maximumMotionBlurRiskForAuthority in 0.0..1.0)
        require(minimumEpipolarConsistencyForAuthority.isFinite() && minimumEpipolarConsistencyForAuthority in 0.0..1.0)
        require(minimumMetricEvidenceAuthorityForAuthority.isFinite() && minimumMetricEvidenceAuthorityForAuthority in 0.0..1.0)
        require(minimumCaptureAuthorityScore.isFinite() && minimumCaptureAuthorityScore in 0.0..1.0)
        require(precisionAuthorityMinimumScore.isFinite() && precisionAuthorityMinimumScore in 0.0..1.0)
        require(precisionAuthorityMinimumSamples >= 1)
    }

    fun canonicalForm(): String = listOf(schemaVersion, id, resolution.name, targetFps, inferenceFps, shutterMicros, iso, stabilizationEnabled, rawEnabled, depthEnabled, denseSurfaceEnabled, gaussianRefinementEnabled, sparseViewRecoveryEnabled, worldSpaceLearnedProposalsEnabled, targetActorCount, hardActorCeiling, authoritativeCameraLimit, globalCameraLimit, precisionTargetMillimeters, minimumIndependentViews, minimumEmpiricalSamples, reducedTransparency, reducedMotion, maximumFrameBudgetMillis, maximumClockSkewMillis, maximumCalibrationResidualPixels, minimumRigConfidence, maximumThermalRisk, maximumMemoryPressure, minimumTransportHeadroom, maximumMotionBlurRiskForAuthority, minimumEpipolarConsistencyForAuthority, minimumMetricEvidenceAuthorityForAuthority, minimumCaptureAuthorityScore, precisionAuthorityMinimumScore, precisionAuthorityMinimumSamples, referenceQualificationRequired,
        preciseColorTemperatureEnabled, colorTemperatureKelvin, colorTint
    ).joinToString("|")

    fun fingerprint(): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(canonicalForm().toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }
    }

    fun applyTo(base: HighEndMotionCaptureSettings = HighEndMotionCaptureSettings()): HighEndMotionCaptureSettings = base.copy(cameraResolutionPreset = resolution, cameraManualExposureMicros = shutterMicros ?: base.cameraManualExposureMicros, cameraManualIso = iso ?: base.cameraManualIso, cameraOisEnabled = stabilizationEnabled, cameraElectronicStabilizationEnabled = stabilizationEnabled, cameraRawSidecarEnabled = rawEnabled, cameraDepthSidecarEnabled = depthEnabled, targetFps = targetFps.coerceIn(24, 240), inferenceFps = inferenceFps.coerceIn(15, 240), distributedGlobalCameraEnvelope = globalCameraLimit, distributedAuthoritativeCameraBudget = authoritativeCameraLimit, distributedMinimumTransportHeadroom = minimumTransportHeadroom, maximumMotionBlurRiskForAuthority = maximumMotionBlurRiskForAuthority, minimumEpipolarConsistencyForAuthority = minimumEpipolarConsistencyForAuthority, minimumMetricEvidenceAuthorityForAuthority = minimumMetricEvidenceAuthority, minimumCaptureAuthorityScore = minimumCaptureAuthorityScore, precisionAuthorityMinimumScore = precisionAuthorityMinimumScore, precisionAuthorityMinimumSamples = precisionAuthorityMinimumSamples, ultraPrecisionTargetP95ErrorMillimeters = precisionTargetMillimeters.coerceIn(0.05, 5.0), ultraPrecisionMinimumEvidenceSamples = minimumEmpiricalSamples, ultraPrecisionMaximumThermalRisk = maximumThermalRisk, ultraPrecisionMaximumMemoryPressure = maximumMemoryPressure, frontierPrecisionQualificationRequiresReference = referenceQualificationRequired,
            frontierMinimumObservedViews = minimumIndependentViews.coerceIn(2, 16),
            cameraPreciseColorTemperatureEnabled =
                preciseColorTemperatureEnabled,
            cameraColorTemperatureKelvin =
                colorTemperatureKelvin,
            cameraColorTint =
                colorTint, frontierSurfaceFusionEnabled = denseSurfaceEnabled, frontierCameraAuthorityEnabled = true)

    companion object {
        fun frontierPrecision(): FrontierCaptureProfile =
            FrontierCaptureProfile(
                resolution = CameraResolutionPreset.AUTO,
                targetFps = 120,
                inferenceFps = 60,
                shutterMicros = 4_000L,
                iso = 200,
                stabilizationEnabled = false,
                rawEnabled = true,
                depthEnabled = true,
                denseSurfaceEnabled = true,
                gaussianRefinementEnabled = true,
                sparseViewRecoveryEnabled = true,
                worldSpaceLearnedProposalsEnabled = true,
                precisionTargetMillimeters = 1.0,
                minimumIndependentViews = 4,
                minimumEmpiricalSamples = 600,
                maximumFrameBudgetMillis = 8.33,
                maximumClockSkewMillis = 4.0,
                maximumCalibrationResidualPixels = 1.5,
                minimumRigConfidence = 0.95,
                minimumTransportHeadroom = 0.65,
                maximumMotionBlurRiskForAuthority = 0.30,
                minimumEpipolarConsistencyForAuthority = 0.80,
                minimumMetricEvidenceAuthorityForAuthority = 0.78,
                minimumCaptureAuthorityScore = 0.88,
                precisionAuthorityMinimumScore = 0.93,
                precisionAuthorityMinimumSamples = 1_200,
            )
    }
}