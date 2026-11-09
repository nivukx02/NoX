package com.nivukx.nox.core.reconstruction

import kotlin.math.exp

/**
 * Conservative confidence model for metric joint reconstruction.
 */
class JointQualityModel(
    private val targetParallaxDegrees: Double = 8.0,
    private val targetReprojectionPx: Double = 2.5,
    private val uncertaintyScaleMeters: Double = 0.08,
) {
    fun score(
        parallaxDegrees: Double,
        reprojectionErrorPx: Double,
        inliers: Int,
        observations: Int,
        uncertaintyMeters: Double,
        detectorConfidence: Double,
        cheiralityRatio: Double = 1.0,
        conditionScore: Double = 1.0,
    ): Double {
        val parallax = (parallaxDegrees / targetParallaxDegrees).coerceIn(0.0, 1.0)
        val reprojection = exp(
            -reprojectionErrorPx.coerceAtLeast(0.0) /
                targetReprojectionPx.coerceAtLeast(0.1),
        ).coerceIn(0.0, 1.0)
        val inlierRatio = (inliers.toDouble() / observations.coerceAtLeast(1)).coerceIn(0.0, 1.0)
        val uncertainty = exp(
            -uncertaintyMeters.coerceAtLeast(0.0) /
                uncertaintyScaleMeters.coerceAtLeast(0.001),
        ).coerceIn(0.0, 1.0)

        return (
            0.20 * parallax +
                0.28 * reprojection +
                0.16 * inlierRatio +
                0.16 * uncertainty +
                0.08 * detectorConfidence.coerceIn(0.0, 1.0) +
                0.06 * cheiralityRatio.coerceIn(0.0, 1.0) +
                0.06 * conditionScore.coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)
    }
}
