package com.nivukx.nox.core.model

/**
 * Shared metric hand reconstruction contracts.
 *
 * These types live in core:model so capture/session persistence and reconstruction can exchange hand
 * geometry without creating an upward module dependency.
 */
data class Hand3DLandmark(
    val index: Int,
    val position: Vec3,
    val confidence: Double,
    val uncertaintyMeters: Double,
    val observationCount: Int,
)

data class Hand3DResult(
    val handId: Long,
    val side: HandSide,
    val timestampNanos: Long,
    val landmarks: List<Hand3DLandmark>,
    val confidence: Double,
    val meanUncertaintyMeters: Double,
    val reprojectionErrorPx: Double,
    val metricEvidenceAuthority: Double = 0.0,
    val symmetricEpipolarConsistencyScore: Double = 1.0,
    val strictSubMillimeterQualified: Boolean = false,
) {
    init {
        require(handId >= 0L)
        require(timestampNanos >= 0L)
        require(landmarks.size <= 21)
        require(confidence in 0.0..1.0)
        require(
            meanUncertaintyMeters.isFinite() &&
                meanUncertaintyMeters >= 0.0,
        )
        require(
            reprojectionErrorPx.isFinite() &&
                reprojectionErrorPx >= 0.0,
        )
        require(metricEvidenceAuthority in 0.0..1.0)
        require(
            symmetricEpipolarConsistencyScore in 0.0..1.0,
        )
    }
}

data class Hand3DFrame(
    val timestampNanos: Long,
    val hands: List<Hand3DResult>,
    val confidence: Double,
)
