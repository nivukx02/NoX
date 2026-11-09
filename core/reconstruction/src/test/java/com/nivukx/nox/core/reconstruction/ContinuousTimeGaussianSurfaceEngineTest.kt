package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContinuousTimeGaussianSurfaceEngineTest {
    @Test fun interpolatesContinuousTimeSurfaceState() {
        val e=ContinuousTimeGaussianSurfaceEngine()
        val c=UncertaintyEnvelope(1e-4,1e-4,1e-4)
        e.observe(1L,ContinuousTimeGaussianSurfaceEngine.Sample(1_000L,Vec3(0.0,0.0,0.0),Vec3(1.0,0.0,0.0),covariance=c,confidence=0.95,origin=EvidenceOrigin.OBSERVED))
        e.observe(1L,ContinuousTimeGaussianSurfaceEngine.Sample(2_000L,Vec3(0.001,0.0,0.0),Vec3(1.0,0.0,0.0),covariance=c,confidence=0.95,origin=EvidenceOrigin.OBSERVED))
        val q=e.query(1L,1_500L)!!
        assertTrue(q.interpolated)
        assertEquals(0.0005,q.positionMeters.x,1e-6)
    }

    @Test fun predictionInflatesUncertaintyAndChangesOrigin() {
        val e=ContinuousTimeGaussianSurfaceEngine()
        e.observe(2L,ContinuousTimeGaussianSurfaceEngine.Sample(1_000L,Vec3(0.0,0.0,0.0),Vec3(1.0,0.0,0.0),covariance=UncertaintyEnvelope(1e-4,1e-4,1e-4),confidence=0.9,origin=EvidenceOrigin.OBSERVED))
        val q=e.query(2L,1_050L)!!
        assertEquals(EvidenceOrigin.TEMPORALLY_PROPAGATED,q.origin)
        assertTrue(q.covariance.xxMeters2>1e-4)
    }
    @Test
    fun quintic_time_interpolation_uses_measured_acceleration_without_nonfinite_output() {
        val e =
            ContinuousTimeGaussianSurfaceEngine(
                ContinuousTimeGaussianSurfaceEngine.Config(
                    maximumQuinticDeviationMeters = 0.20,
                    maximumHermiteDeviationMeters = 0.20,
                    maximumAccelerationMetersPerSecond2 = 20.0,
                ),
            )
        val c =
            UncertaintyEnvelope(
                1e-4,
                1e-4,
                1e-4,
                confidence = 0.98,
            )

        e.observe(
            7L,
            ContinuousTimeGaussianSurfaceEngine.Sample(
                timestampNanos = 0L,
                positionMeters = Vec3(0.0,0.0,0.0),
                velocityMetersPerSecond = Vec3(0.0,0.0,0.0),
                accelerationMetersPerSecond2 = Vec3(2.0,0.0,0.0),
                covariance = c,
                confidence = 0.98,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )
        e.observe(
            7L,
            ContinuousTimeGaussianSurfaceEngine.Sample(
                timestampNanos = 100_000_000L,
                positionMeters = Vec3(0.02,0.0,0.0),
                velocityMetersPerSecond = Vec3(0.40,0.0,0.0),
                accelerationMetersPerSecond2 = Vec3(2.0,0.0,0.0),
                covariance = c,
                confidence = 0.98,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )

        val q =
            e.query(
                fieldId = 7L,
                timestampNanos = 50_000_000L,
            )!!

        assertTrue(q.positionMeters.finite())
        assertTrue(q.velocityMetersPerSecond.finite())
        assertTrue(q.confidence > 0.80)
        assertTrue(!q.kinematicFallback)
    }

    @Test
    fun jerk_aware_extrapolation_is_bounded_and_lowers_confidence() {
        val engine =
            ContinuousTimeGaussianSurfaceEngine(
                ContinuousTimeGaussianSurfaceEngine.Config(
                    maximumPredictionSeconds = 0.20,
                    maximumJerkMetersPerSecond3 = 30.0,
                ),
            )

        val covariance =
            UncertaintyEnvelope(
                1e-4,
                1e-4,
                1e-4,
                confidence = 0.98,
            )

        engine.observe(
            19L,
            ContinuousTimeGaussianSurfaceEngine.Sample(
                timestampNanos = 1_000_000_000L,
                positionMeters = Vec3(0.0, 0.0, 0.0),
                velocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
                accelerationMetersPerSecond2 = Vec3(0.0, 0.0, 0.0),
                covariance = covariance,
                confidence = 0.98,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )
        engine.observe(
            19L,
            ContinuousTimeGaussianSurfaceEngine.Sample(
                timestampNanos = 1_100_000_000L,
                positionMeters = Vec3(0.0, 0.0, 0.0),
                velocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
                accelerationMetersPerSecond2 = Vec3(3.0, 0.0, 0.0),
                covariance = covariance,
                confidence = 0.98,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )

        val query =
            engine.query(
                fieldId = 19L,
                timestampNanos = 1_150_000_000L,
            )!!

        assertTrue(query.positionMeters.finite())
        assertTrue(query.velocityMetersPerSecond.finite())
        assertTrue(query.jerkMetersPerSecond3.finite())
        assertTrue(
            query.jerkMetersPerSecond3.norm() <=
                30.0 + 1e-9,
        )
        assertTrue(query.confidence < 0.98)
        assertTrue(
            query.covariance.confidence <
                0.98,
        )
    }

    @Test
    fun oversizedInterpolationGapReducesConfidence() {
        val e = ContinuousTimeGaussianSurfaceEngine(
            ContinuousTimeGaussianSurfaceEngine.Config(
                maximumInterpolationGapSeconds = 0.0005,
            ),
        )
        val c = UncertaintyEnvelope(1e-4,1e-4,1e-4,confidence=0.95)
        e.observe(
            3L,
            ContinuousTimeGaussianSurfaceEngine.Sample(
                1_000L,
                Vec3(0.0,0.0,0.0),
                Vec3(0.5,0.0,0.0),
                covariance = c,
                confidence = 0.95,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )
        e.observe(
            3L,
            ContinuousTimeGaussianSurfaceEngine.Sample(
                3_000_000L,
                Vec3(0.002,0.0,0.0),
                Vec3(0.5,0.0,0.0),
                covariance = c,
                confidence = 0.95,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )
        val query = e.query(3L, 1_500_000L)!!
        assertTrue(query.confidence < 0.95)
        assertTrue(query.covariance.confidence < 0.95)
    }
    @Test
    fun excessiveHermiteOvershootFallsBackAndLowersConfidence() {
        val e = ContinuousTimeGaussianSurfaceEngine(
            ContinuousTimeGaussianSurfaceEngine.Config(
                maximumHermiteDeviationMeters = 0.01,
                maximumVelocityMetersPerSecond = 5.0,
            ),
        )
        val c = UncertaintyEnvelope(1e-4,1e-4,1e-4,confidence=0.95)
        e.observe(
            4L,
            ContinuousTimeGaussianSurfaceEngine.Sample(
                0L,
                Vec3(0.0,0.0,0.0),
                Vec3(5.0,0.0,0.0),
                covariance = c,
                confidence = 0.95,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )
        e.observe(
            4L,
            ContinuousTimeGaussianSurfaceEngine.Sample(
                100_000_000L,
                Vec3(0.10,0.0,0.0),
                Vec3(-5.0,0.0,0.0),
                covariance = c,
                confidence = 0.95,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )
        val query = e.query(4L, 50_000_000L)!!
        assertTrue(query.kinematicFallback)
        assertTrue(query.confidence < 0.95)
        assertTrue(query.positionMeters.x in 0.0..0.10)
    }
}