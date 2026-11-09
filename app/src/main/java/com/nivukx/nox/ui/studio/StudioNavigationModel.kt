package com.nivukx.nox.ui.studio

enum class StudioDestinationGroup(
    val order: Int,
) {
    CAPTURE(0),
    RIG(1),
    SOLVE(2),
    EDIT(3),
    OUTPUT(4),
}

enum class MotionStudioDestination(
    val label: String,
    val shortLabel: String,
    val contentDescription: String,
    val group: StudioDestinationGroup,
) {
    CAPTURE(
        label = "Capture",
        shortLabel = "CAP",
        contentDescription = "Capture and operator recording controls",
        group = StudioDestinationGroup.CAPTURE,
    ),
    RIG(
        label = "Rig",
        shortLabel = "RIG",
        contentDescription = "Camera rig topology and device roles",
        group = StudioDestinationGroup.RIG,
    ),
    CALIBRATION(
        label = "Calibration",
        shortLabel = "CAL",
        contentDescription = "Camera calibration, drift and quality gates",
        group = StudioDestinationGroup.RIG,
    ),
    SYNC(
        label = "Sync",
        shortLabel = "SYNC",
        contentDescription = "Camera clock synchronization and temporal alignment",
        group = StudioDestinationGroup.RIG,
    ),
    RECONSTRUCTION(
        label = "Reconstruction",
        shortLabel = "RECON",
        contentDescription = "Live three-dimensional reconstruction and quality evidence",
        group = StudioDestinationGroup.SOLVE,
    ),
    FACE_EYE(
        label = "Face + Eye",
        shortLabel = "FACE",
        contentDescription = "Facial geometry, eye tracking and gaze reconstruction",
        group = StudioDestinationGroup.SOLVE,
    ),
    HANDS(
        label = "Hands",
        shortLabel = "HAND",
        contentDescription = "Hand and finger kinematics reconstruction",
        group = StudioDestinationGroup.SOLVE,
    ),
    SCENE(
        label = "Scene",
        shortLabel = "SCN",
        contentDescription = "Scene geometry, lighting and spatial context",
        group = StudioDestinationGroup.SOLVE,
    ),
    TIMELINE(
        label = "Timeline",
        shortLabel = "TIME",
        contentDescription = "Motion timeline, take markers and temporal editing",
        group = StudioDestinationGroup.EDIT,
    ),
    DIAGNOSTICS(
        label = "Diagnostics",
        shortLabel = "DIAG",
        contentDescription = "Runtime evidence, performance and subsystem diagnostics",
        group = StudioDestinationGroup.EDIT,
    ),
    EXPORT(
        label = "Export",
        shortLabel = "OUT",
        contentDescription = "Validate and export the motion-only session artifact",
        group = StudioDestinationGroup.OUTPUT,
    ),
}

data class StudioNavigationState(
    val selected: MotionStudioDestination = MotionStudioDestination.RECONSTRUCTION,
    val inspectorVisible: Boolean = true,
) {
    val selectedGroup: StudioDestinationGroup
        get() = selected.group
}

object StudioNavigationReducer {
    fun select(
        state: StudioNavigationState,
        destination: MotionStudioDestination,
    ): StudioNavigationState = state.copy(selected = destination)

    fun toggleInspector(state: StudioNavigationState): StudioNavigationState =
        state.copy(inspectorVisible = !state.inspectorVisible)
}
