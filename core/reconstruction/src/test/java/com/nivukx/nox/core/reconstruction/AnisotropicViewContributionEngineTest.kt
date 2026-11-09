package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class AnisotropicViewContributionEngineTest {

    private fun observation(
        cameraId: Int,
        position: Vec3,
        u: Double = 960.0,
        v: Double = 540.0,
        confidence: Double = 0.90,
    ) =
        TriangulationEngine.Observation(
            cameraPose = Pose(
                position = position,
                rotation = Quat.IDENTITY,
            ),
            intrinsics = CameraIntrinsics(
                fx = 1200.0,
                fy = 1200.0,
                cx = 960.0,
                cy = 540.0,
                imageWidth = 1920,
                imageHeight = 1080,
            ),
            cameraId = cameraId.toString(),
            u = u,
            v = v,
            weight = confidence,
            temporalAlignmentConfidence = 0.95,
            observationUncertainty = 0.10,
            visibility = 0.95,
            sourceQuality = confidence,
        )

    @Test
    fun complementaryViewsReceivePrincipalAxisEvidence() {
        val engine = AnisotropicViewContributionEngine()

        val result = engine.rank(
            candidates = listOf(
                AnisotropicViewContributionEngine.Candidate(
                    observation = observation(
                        0,
                        Vec3(0.0, 0.0, 0.0),
                    ),
                ),
                AnisotropicViewContributionEngine.Candidate(
                    observation = observation(
                        1,
                        Vec3(0.25, 0.0, 0.0),
                    ),
                ),
                AnisotropicViewContributionEngine.Candidate(
                    observation = observation(
                        2,
                        Vec3(0.0, 0.30, 0.0),
                    ),
                ),
            ),
        )

        assertTrue(result.selected.isNotEmpty())
        assertTrue(result.weakestEigenvalue.isFinite())
        assertTrue(result.strongestEigenvalue.isFinite())
        assertTrue(result.conditionScore in 0.0..1.0)
        assertTrue(
            result.selected.all {
                it.totalScore in 0.0..1.0
            },
        )
    }

    @Test
    fun rankDeficientConfigurationRaisesWeakAxisWarning() {
        val engine = AnisotropicViewContributionEngine()

        val result = engine.rank(
            candidates = listOf(
                AnisotropicViewContributionEngine.Candidate(
                    observation =
                        observation(
                            0,
                            Vec3(0.0, 0.0, 0.0),
                        ),
                ),
                AnisotropicViewContributionEngine.Candidate(
                    observation =
                        observation(
                            1,
                            Vec3(0.01, 0.0, 0.0),
                        ),
                ),
            ),
        )

        assertTrue(
            result.warnings.any {
                it == "3d-information-weak-axis-dominant" ||
                    it == "ray-information-rank-deficient"
            } ||
                result.conditionScore < 0.20,
        )
    }

    @Test
    fun selectedViewsRemainDeterministic() {
        val engine = AnisotropicViewContributionEngine()
        val candidates = listOf(
            AnisotropicViewContributionEngine.Candidate(
                observation = observation(
                    0,
                    Vec3(0.0, 0.0, 0.0),
                    confidence = 0.92,
                ),
            ),
            AnisotropicViewContributionEngine.Candidate(
                observation = observation(
                    1,
                    Vec3(0.40, 0.0, 0.0),
                    confidence = 0.90,
                ),
            ),
            AnisotropicViewContributionEngine.Candidate(
                observation = observation(
                    2,
                    Vec3(0.0, 0.40, 0.0),
                    confidence = 0.91,
                ),
            ),
        )

        val a = engine.rank(candidates)
        val b = engine.rank(candidates)

        assertTrue(
            a.selected.map { it.cameraId } ==
                b.selected.map { it.cameraId },
        )
    }
}
