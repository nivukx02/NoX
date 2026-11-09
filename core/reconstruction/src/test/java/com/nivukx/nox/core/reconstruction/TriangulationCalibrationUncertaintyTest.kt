package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertTrue
import org.junit.Test

class TriangulationCalibrationUncertaintyTest {

    private val intrinsics = CameraIntrinsics(
        fx = 1000.0,
        fy = 1000.0,
        cx = 500.0,
        cy = 400.0,
    )

    private val point = Vec3(0.0, 0.0, 3.0)

    @Test
    fun correlatedCalibrationEnvelopeInflates3dUncertainty() {
        val engine = TriangulationUncertaintyEngine()

        val baseObservations = observations(
            profile = CalibrationUncertaintyProfile(),
        )
        val uncertainObservations = observations(
            profile = CalibrationUncertaintyProfile(
                translationSigmaMeters = 0.0010,
                rotationSigmaRadians = 0.0010,
                intrinsicRelativeSigma = 0.0010,
                evidenceConfidence = 0.95,
            ),
        )

        val baseline = engine.estimate(
            TriangulationUncertaintyEngine.Input(
                point = point,
                observations = baseObservations,
                residualsPx = listOf(0.5, 0.6),
                robustResidualScalePx = 0.5,
            ),
        )

        val calibrated = engine.estimate(
            TriangulationUncertaintyEngine.Input(
                point = point,
                observations = uncertainObservations,
                residualsPx = listOf(0.5, 0.6),
                robustResidualScalePx = 0.5,
            ),
        )

        assertTrue(!baseline.calibrationUncertaintyApplied)
        assertTrue(calibrated.calibrationUncertaintyApplied)
        assertTrue(
            calibrated.sigmaMajorMeters >=
                baseline.sigmaMajorMeters,
        )
        assertTrue(
            calibrated.calibrationUncertaintyMeters > 0.0,
        )
    }

    @Test
    fun largeCalibrationUncertaintyReducesMetricEvidenceWeight() {
        val engine = TriangulationEngine()

        fun profile(
            translation: Double,
            rotationDegrees: Double,
            intrinsic: Double,
        ) = CalibrationUncertaintyProfile(
            translationSigmaMeters = translation,
            rotationSigmaRadians =
                Math.toRadians(rotationDegrees),
            intrinsicRelativeSigma = intrinsic,
            evidenceConfidence = 0.95,
        )

        val stable = listOf(
            TriangulationEngine.Observation(
                cameraPose = Pose(
                    position = Vec3(0.0, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                cameraId = "a",
                u = 500.0,
                v = 400.0,
                calibrationUncertainty =
                    profile(0.001, 0.05, 0.001),
            ),
            TriangulationEngine.Observation(
                cameraPose = Pose(
                    position = Vec3(1.0, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                cameraId = "b",
                u = 333.3333333333,
                v = 400.0,
                calibrationUncertainty =
                    profile(0.001, 0.05, 0.001),
            ),
        )

        val degraded =
            stable.map {
                it.copy(
                    calibrationUncertainty =
                        profile(0.020, 1.0, 0.020),
                )
            }

        val stableResult = engine.robust(stable)
        val degradedResult = engine.robust(degraded)

        assertTrue(stableResult != null)
        assertTrue(degradedResult != null)
        assertTrue(
            degradedResult!!.metricEvidenceAuthority <=
                stableResult!!.metricEvidenceAuthority,
        )
    }

    @Test
    fun missingCalibrationEvidenceDoesNotPretendZeroMeansPerfect() {
        val engine = TriangulationUncertaintyEngine()
        val result = engine.estimate(
            TriangulationUncertaintyEngine.Input(
                point = point,
                observations = observations(
                    profile = CalibrationUncertaintyProfile(),
                ),
            ),
        )

        assertTrue(!result.calibrationUncertaintyApplied)
        assertTrue(result.calibrationUncertaintyMeters == 0.0)
    }

    private fun observations(
        profile: CalibrationUncertaintyProfile,
    ): List<TriangulationUncertaintyEngine.Observation> =
        listOf(
            TriangulationUncertaintyEngine.Observation(
                cameraPose = Pose(
                    position = Vec3(0.0, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                u = 500.0,
                v = 400.0,
                calibrationUncertainty = profile,
            ),
            TriangulationUncertaintyEngine.Observation(
                cameraPose = Pose(
                    position = Vec3(1.0, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                u = 166.6666666667,
                v = 400.0,
                calibrationUncertainty = profile,
            ),
        )
}
