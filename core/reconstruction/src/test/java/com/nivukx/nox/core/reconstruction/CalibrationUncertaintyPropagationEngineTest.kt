package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class CalibrationUncertaintyPropagationEngineTest {

    @Test
    fun rotationalUncertaintyGrowsWithRange() {
        val engine = CalibrationUncertaintyPropagationEngine()
        val profile = CalibrationUncertaintyProfile(
            rotationSigmaRadians = 0.001,
            evidenceConfidence = 1.0,
        )

        val near = engine.propagate(
            pointWorld = Vec3(0.0, 0.0, 1.0),
            cameraPose = Pose(),
            profile = profile,
        )
        val far = engine.propagate(
            pointWorld = Vec3(0.0, 0.0, 5.0),
            cameraPose = Pose(),
            profile = profile,
        )

        assertTrue(
            far.rotationalSigmaMeters >
                near.rotationalSigmaMeters,
        )
        assertTrue(
            far.equivalentSigmaMeters >
                near.equivalentSigmaMeters,
        )
    }

    @Test
    fun missingEvidenceDoesNotPretendCalibrationIsPrecise() {
        val result =
            CalibrationUncertaintyPropagationEngine().propagate(
                pointWorld = Vec3(1.0, 2.0, 3.0),
                cameraPose = Pose(),
                profile = CalibrationUncertaintyProfile(),
            )

        assertEquals(0.0, result.equivalentSigmaMeters, 1e-12)
        assertEquals(false, result.applied)
        assertEquals(0.0, result.confidence, 1e-12)
    }

    @Test
    fun translationAndIntrinsicTermsProduceFinitePositiveEnvelope() {
        val result =
            CalibrationUncertaintyPropagationEngine().propagate(
                pointWorld = Vec3(1.0, 0.0, 2.0),
                cameraPose = Pose(),
                profile = CalibrationUncertaintyProfile(
                    translationSigmaMeters = 0.001,
                    intrinsicRelativeSigma = 0.002,
                    evidenceConfidence = 0.9,
                ),
            )

        assertTrue(result.applied)
        assertTrue(result.equivalentSigmaMeters > 0.0)
        assertTrue(result.covariance[0][0].isFinite())
        assertTrue(result.covariance[1][1].isFinite())
        assertTrue(result.covariance[2][2].isFinite())
    }
}
