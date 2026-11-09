package com.nivukx.nox.core.model

enum class ReconstructionVisualizationMode {
    SOLID_SKELETON,
    DENSE_SURFACE,
    SURFELS,
    WIREFRAME,
    CAMERA_FRUSTA,
    UNCERTAINTY,
    REPROJECTION_HEATMAP,
}

data class ReconstructionRenderSettings(
    val mode: ReconstructionVisualizationMode = ReconstructionVisualizationMode.DENSE_SURFACE,
    val showCameras: Boolean = true,
    val showSkeleton: Boolean = true,
    val showFaceMesh: Boolean = true,
    val showHands: Boolean = true,
    val showGazeRay: Boolean = true,
    val showSceneGeometry: Boolean = true,
    val showUncertainty: Boolean = true,
    val showResiduals: Boolean = false,
    val showContactPoints: Boolean = true,
    val pointSizePx: Double = 2.0,
    val skeletonLineWidthPx: Double = 2.5,
    val meshOpacity: Double = 0.92,
) {
    init {
        require(pointSizePx in 0.25..12.0)
        require(skeletonLineWidthPx in 0.5..8.0)
        require(meshOpacity in 0.05..1.0)
    }
}
