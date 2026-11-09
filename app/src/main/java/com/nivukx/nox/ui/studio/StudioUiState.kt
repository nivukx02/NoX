package com.nivukx.nox.ui.studio

import com.nivukx.nox.core.model.ProductionCaptureProfile

enum class StudioSection {
    CAPTURE, RIG, CALIBRATION, SYNC, RECONSTRUCTION,
    FACE_EYE, HANDS, SCENE, TIMELINE, DIAGNOSTICS, EXPORT, ADVANCED_SETTINGS
}

data class StudioUiState(
    val section: StudioSection,
    val telemetry: StudioTelemetryState,
    val captureProfileId: String = ProductionCaptureProfile.default.id,
    val customizationPanelVisible: Boolean = false,
)

object StudioUiStateReducer {
    fun reduce(
        telemetry: StudioTelemetryState,
        selected: StudioSection = StudioSection.RECONSTRUCTION,
        captureProfileId: String = ProductionCaptureProfile.default.id,
    ): StudioUiState =
        StudioUiState(
            section = selected,
            telemetry = telemetry,
            captureProfileId = ProductionCaptureProfile.byId(captureProfileId).id,
        )

    fun selectCaptureProfile(
        state: StudioUiState,
        profileId: String,
    ): StudioUiState =
        state.copy(
            captureProfileId = ProductionCaptureProfile.byId(profileId).id,
        )

    fun toggleCustomizationPanel(
        state: StudioUiState,
    ): StudioUiState =
        state.copy(
            customizationPanelVisible = !state.customizationPanelVisible,
        )

    fun exportEnabled(telemetry: StudioTelemetryState): Boolean =
        telemetry.exportReady == true &&
            telemetry.calibrationReady == true &&
            telemetry.nativeEngineHealthy == true &&
            telemetry.systemConfidence?.let { it >= 0.50 } == true
}