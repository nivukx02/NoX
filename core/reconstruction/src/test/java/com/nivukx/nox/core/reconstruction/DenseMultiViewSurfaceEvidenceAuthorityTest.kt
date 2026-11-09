package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class DenseMultiViewSurfaceEvidenceAuthorityTest {

    private val intrinsics =
        CameraIntrinsics(
            fx = 1200.0,
            fy = 1200.0,
            cx = 600.0,
            cy = 500.0,
        )

    private fun observation(
        key: DenseMultiViewSurfaceReconstructionEngine.PointKey,
        target: Vec3,
        cameraX: Double,
        cameraId: String,
        calibrationSigmaMeters: Double = 0.001,
        rotationSigmaRadians: Double = 0.001,
        timingUncertaintyNanos: Long = 1_000_000L,
    ): DenseMultiViewSurfaceReconstructionEngine.PointObservation {
        val u =
            intrinsics.fx *
                (target.x - cameraX) /
                target.z +
                intrinsics.cx
        val v =
            intrinsics.fy *
                target.y /
                target.z +
                intrinsics.cy

        return DenseMultiViewSurfaceReconstructionEngine.PointObservation(
            key = key,
            observation = TriangulationEngine.Observation(
                cameraPose = Pose(
                    Vec3(cameraX, 0.0, 0.0),
                    Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                cameraId = cameraId,
                u = u,
                v = v,
                weight = 0.98,
                temporalAlignmentConfidence = 0.98,
                temporalUncertaintyNanos = timingUncertaintyNanos,
                observationUncertainty = 0.02,
                visibility = 0.98,
                sourceQuality = 0.98,
                crossViewAgreement = 0.98,
                temporalStability = 0.98,
                calibrationUncertainty =
                    com.nivukx.nox.core.model.CalibrationUncertaintyProfile(
                        translationSigmaMeters = calibrationSigmaMeters,
                        rotationSigmaRadians = rotationSigmaRadians,
                        intrinsicRelativeSigma = 0.003,
                        evidenceConfidence = 0.98,
                    ),
            ),
            semanticConfidence = 0.98,
            visibility = 0.98,
            sourceDiversity = 0.98,
            photometricConsistency = 0.98,
            silhouetteConsistency = 0.98,
            motionBoundaryConfidence = 0.98,
            lightingStability = 0.98,
        )
    }

    @Test
    fun densePointExposesTriangulatorViewEvidence() {
        val engine = DenseMultiViewSurfaceReconstructionEngine()
        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 100,
            )
        val target = Vec3(0.2, 0.1, 4.0)

        val result =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = listOf(
                        observation(key, target, -1.0, "a"),
                        observation(key, target, 1.0, "b"),
                        observation(key, target, 0.0, "c"),
                    ),
                ),
            )

        val point = result.points.single()
        assertTrue(point.activeViewInformationGainScore in 0.0..1.0)
        assertTrue(point.activeViewRedundancyScore in 0.0..1.0)
        assertTrue(point.metricEvidenceAuthority in 0.0..1.0)
        assertEquals(
            point.metricEvidenceAuthority,
            (
                point.activeViewInformationGainScore * 0.60 +
                    point.activeViewRedundancyScore * 0.40
                ).coerceIn(0.0, 1.0),
            1e-9,
        )
        assertEquals(
            point.metricEvidenceAuthority,
            result.meanMetricEvidenceAuthority,
            1e-9,
        )
        engine.close()
    }

    @Test
    fun precisionQualificationRequiresMetricEvidenceAuthority() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    minimumActiveViewInformationGain = 0.98,
                    minimumMetricEvidenceAuthorityForPrecision = 0.98,
                    minimumPointUncertaintyMeters = 0.50,
                    hardPointUncertaintyMeters = 0.80,
                    preferredParallaxDegrees = 1.0,
                ),
            )
        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 101,
            )
        val target = Vec3(0.2, 0.1, 4.0)

        val result =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = listOf(
                        observation(key, target, 0.0, "a"),
                        observation(
                            key,
                            target,
                            0.55,
                            "b",
                            calibrationSigmaMeters = 0.030,
                            rotationSigmaRadians = 0.025,
                            timingUncertaintyNanos = 30_000_000L,
                        ),
                    ),
                ),
            )

        val point = result.points.single()
        assertTrue(
            !point.precisionQualified ||
                point.metricEvidenceAuthority >= 0.98,
        )
        engine.close()
    }
}
