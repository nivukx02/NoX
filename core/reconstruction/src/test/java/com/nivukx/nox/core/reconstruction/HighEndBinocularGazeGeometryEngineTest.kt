package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndBinocularGazeGeometryEngineTest {

    @Test
    fun coherentBinocularRaysProduceTrustedGeometry() {
        val engine = HighEndBinocularGazeGeometryEngine()

        val left = eye(
            EyeSide.LEFT,
            Vec3(-0.0325, 0.0, 0.0),
            Vec3(0.035, 0.0, 1.0),
        )
        val right = eye(
            EyeSide.RIGHT,
            Vec3(0.0325, 0.0, 0.0),
            Vec3(-0.035, 0.0, 1.0),
        )

        val result = engine.evaluate(left, right)

        assertTrue(result.interpupillaryDistanceMeters > 0.06)
        assertTrue(result.angularDisagreementDegrees < 6.0)
        assertTrue(result.rayMissDistanceMeters < 0.01)
        assertTrue(result.confidence > 0.70)
        assertTrue(result.warnings.none {
            it.contains("hard-failure")
        })
    }

    @Test
    fun nearParallelGazeRaysExposeWeakGeometryConditioning() {
        val engine = HighEndBinocularGazeGeometryEngine()

        val left = eye(
            EyeSide.LEFT,
            Vec3(-0.0325, 0.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
        )
        val right = eye(
            EyeSide.RIGHT,
            Vec3(0.0325, 0.0, 0.0),
            Vec3(0.001, 0.0, 1.0),
        )

        val result = engine.evaluate(left, right)

        assertTrue(
            result.rayGeometryCondition in 0.0..1.0,
        )
        assertTrue(
            result.rayGeometryCondition < 0.08,
        )
        assertTrue(
            result.warnings.any {
                it.contains("near-parallel")
            },
        )
    }

    @Test
    fun weakUpstreamEpipolarEvidenceBlocksPrecisionQualification() {
        val engine = HighEndBinocularGazeGeometryEngine()

        val left =
            eye(
                EyeSide.LEFT,
                Vec3(-0.0325, 0.0, 0.0),
                Vec3(0.035, 0.0, 1.0),
                epipolarScore = 0.25,
            )
        val right =
            eye(
                EyeSide.RIGHT,
                Vec3(0.0325, 0.0, 0.0),
                Vec3(-0.035, 0.0, 1.0),
                epipolarScore = 0.25,
            )

        val result = engine.evaluate(left, right)

        assertTrue(result.epipolarEvidenceScore < 0.30)
        assertTrue(!result.precisionQualified)
        assertTrue(
            result.warnings.any {
                it.contains("epipolar-evidence")
            },
        )
    }

    @Test
    fun divergentRaysAreDegraded() {
        val engine = HighEndBinocularGazeGeometryEngine()

        val left = eye(
            EyeSide.LEFT,
            Vec3(-0.0325, 0.0, 0.0),
            Vec3(0.20, 0.0, 1.0),
        )
        val right = eye(
            EyeSide.RIGHT,
            Vec3(0.0325, 0.0, 0.0),
            Vec3(0.20, 0.0, 1.0),
        )

        val result = engine.evaluate(left, right)

        assertTrue(result.angularDisagreementDegrees < 1.0)
        assertTrue(result.rayMissDistanceMeters > 0.03)
        assertTrue(
            result.warnings.any {
                it.contains("ray-miss") ||
                    it.contains("convergence")
            },
        )
        assertTrue(result.confidence < 0.80)
    }

    private fun eye(
        side: EyeSide,
        center: Vec3,
        direction: Vec3,
        epipolarScore: Double = 1.0,
    ) = Eye3DReconstructionEngine.Result(
        eye = side,
        centerWorld = center,
        gazeDirectionWorld = direction.normalized(),
        confidence = 0.96,
        angularSpreadDegrees = 1.0,
        observations = 4,
        centerUncertaintyMeters = 0.006,
        gazeStability = 0.92,
        angularVelocityDegreesPerSecond = 40.0,
        parallaxDegrees = 4.0,
        symmetricEpipolarConsistencyScore = epipolarScore,
    )
}


class HighEndBinocularGazeGeometryEngineStrictPrecisionTest {
    @Test
    fun strict_submillimeter_gaze_requires_measured_point_uncertainty_below_one_millimeter() {
        val engine =
            HighEndBinocularGazeGeometryEngine(
                HighEndBinocularGazeGeometryEngine.Config(
                    strictQualificationMaxGazeUncertaintyMeters =
                        0.001,
                ),
            )

        fun eye(
            side: EyeSide,
            uncertainty: Double,
        ) =
            Eye3DReconstructionEngine.Result(
                eye = side,
                centerWorld =
                    if (side == EyeSide.LEFT) {
                        Vec3(-0.0325, 0.0, 0.0)
                    } else {
                        Vec3(0.0325, 0.0, 0.0)
                    },
                gazeDirectionWorld =
                    if (side == EyeSide.LEFT) {
                        Vec3(0.08, 0.0, 1.0).normalized()
                    } else {
                        Vec3(-0.08, 0.0, 1.0).normalized()
                    },
                confidence = 0.99,
                angularSpreadDegrees = 0.2,
                observations = 12,
                uncertainty95MajorMeters = uncertainty,
                uncertaintyMajorMeters = uncertainty,
                uncertaintyMiddleMeters = uncertainty,
                uncertaintyMinorMeters = uncertainty,
                uncertaintyDepthMeters = uncertainty,
                uncertaintyConfidence = 0.99,
                conditionScore = 0.98,
                cheiralityRatio = 1.0,
                robustReprojectionP95Px = 0.25,
                viewDiversityScore = 0.98,
                gazeStability = 0.99,
                parallaxDegrees = 8.0,
                symmetricEpipolarConsistencyScore = 0.99,
            )

        val precise =
            engine.evaluate(
                eye(EyeSide.LEFT, 0.0004),
                eye(EyeSide.RIGHT, 0.0004),
            )
        val uncertain =
            engine.evaluate(
                eye(EyeSide.LEFT, 0.003),
                eye(EyeSide.RIGHT, 0.003),
            )

        assertTrue(precise.strictSubMillimeterQualified)
        assertTrue(!uncertain.strictSubMillimeterQualified)
        assertTrue(
            uncertain.warnings.contains(
                "binocular-gaze-submillimeter-not-qualified",
            ),
        )
    }

class HighEndBinocularGazeGeometryEngineUncertaintyTest {
    @Test
    fun centerUncertaintyLowersGazeConfidenceAndRaisesPointUncertainty() {
        fun eye(
            side: EyeSide,
            uncertainty: Double,
        ) = Eye3DReconstructionEngine.Result(
            eye = side,
            centerWorld = if (side == EyeSide.LEFT) {
                Vec3(-0.032, 0.0, 0.0)
            } else {
                Vec3(0.032, 0.0, 0.0)
            },
            gazeDirectionWorld = if (side == EyeSide.LEFT) {
                Vec3(0.032, 0.0, 1.0).normalized()
            } else {
                Vec3(-0.032, 0.0, 1.0).normalized()
            },
            confidence = 0.99,
            angularSpreadDegrees = 0.5,
            observations = 8,
            uncertainty95MajorMeters = uncertainty,
            uncertaintyMajorMeters = uncertainty * 0.8,
            uncertaintyMiddleMeters = uncertainty * 0.5,
            uncertaintyMinorMeters = uncertainty * 0.3,
            uncertaintyDepthMeters = uncertainty * 0.7,
            uncertaintyConfidence = 1.0,
            conditionScore = 0.95,
            cheiralityRatio = 1.0,
            robustReprojectionP95Px = 0.5,
            viewDiversityScore = 0.95,
            gazeStability = 0.95,
            parallaxDegrees = 8.0,
        )

        val engine = HighEndBinocularGazeGeometryEngine()
        val precise = engine.evaluate(
            eye(EyeSide.LEFT, 0.001),
            eye(EyeSide.RIGHT, 0.001),
        )
        val uncertain = engine.evaluate(
            eye(EyeSide.LEFT, 0.025),
            eye(EyeSide.RIGHT, 0.025),
        )

        assertTrue(
            uncertain.gazePointUncertaintyMeters >
                precise.gazePointUncertaintyMeters,
        )
        assertTrue(uncertain.confidence < precise.confidence)
    }
}
