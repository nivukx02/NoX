package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.FrontierLandmark3dResult
import com.nivukx.nox.core.model.MeasurementAuthority
import com.nivukx.nox.core.model.MeasurementFailure
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FrontierReconstructionCoordinatorAuthorityTest {

    private val coordinator =
        FrontierReconstructionCoordinator()

    private fun landmark(
        authority: MeasurementAuthority,
        failure: MeasurementFailure = MeasurementFailure.NONE,
        precision: Double = 0.99,
    ): FrontierLandmark3dResult =
        FrontierLandmark3dResult(
            landmarkId = 7,
            timestampNanos = 1_000L,
            positionMeters = Vec3(0.0, 0.0, 1.0),
            covarianceMeters2 = 1.0e-8,
            p95UncertaintyMeters = 0.0001,
            reprojectionErrorPixels = 0.1,
            viewCount = 6,
            angularDiversityDegrees = 20.0,
            cameraIds = listOf("a", "b", "c"),
            authority = authority,
            failure = failure,
        )

    @Test
    fun observedHighFeasibilityIsNotAuthoritative() {
        assertFalse(
            coordinator.isAuthoritativeLandmark(
                landmark =
                    landmark(
                        authority = MeasurementAuthority.OBSERVED,
                    ),
                minimumConfidence = 0.60,
            ),
        )
    }

    @Test
    fun geometricHighFeasibilityIsAuthoritative() {
        assertTrue(
            coordinator.isAuthoritativeLandmark(
                landmark =
                    landmark(
                        authority = MeasurementAuthority.GEOMETRIC,
                    ),
                minimumConfidence = 0.60,
            ),
        )
    }

    @Test
    fun failedGeometricLandmarkIsNotAuthoritative() {
        assertFalse(
            coordinator.isAuthoritativeLandmark(
                landmark =
                    landmark(
                        authority = MeasurementAuthority.GEOMETRIC,
                        failure = MeasurementFailure.HIGH_UNCERTAINTY,
                    ),
                minimumConfidence = 0.60,
            ),
        )
    }
}
