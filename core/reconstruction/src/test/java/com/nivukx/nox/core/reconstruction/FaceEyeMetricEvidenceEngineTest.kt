package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshTriangle
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceEyeMetricEvidenceEngineTest {
    private val engine = FaceEyeMetricEvidenceEngine()

    private fun face() = FaceMeshFrame(
        timestampNanos = 1_000_000L,
        imageWidth = 1920,
        imageHeight = 1080,
        rotationDegrees = 0,
        boundsLeft = 400f,
        boundsTop = 200f,
        boundsRight = 1500f,
        boundsBottom = 900f,
        confidence = 0.96,
        pointCoverage = 0.94,
        triangleCoverage = 0.90,
        points = (0 until 4).map {
            FaceMeshPoint3D(
                index = it,
                position = Vec3(it.toDouble(), 0.0, 0.0),
            )
        },
        triangles = listOf(
            FaceMeshTriangle(0, 1, 2),
            FaceMeshTriangle(0, 2, 3),
        ),
    )

    private fun eye() = Eye3DReconstructionEngine.Result(
        eye = com.nivukx.nox.core.model.EyeSide.LEFT,
        centerWorld = Vec3(0.0, 0.0, 1.0),
        gazeDirectionWorld = Vec3(0.0, 0.0, 1.0),
        confidence = 0.92,
        angularSpreadDegrees = 2.0,
        observations = 4,
        centerUncertaintyMeters = 0.001,
        uncertainty95MajorMeters = 0.003,
        uncertaintyMajorMeters = 0.002,
        uncertaintyMiddleMeters = 0.0015,
        uncertaintyMinorMeters = 0.001,
        uncertaintyDepthMeters = 0.0012,
        uncertaintyAnisotropyRatio = 3.0,
        uncertaintyDepthDominance = 0.6,
        uncertaintyConfidence = 0.94,
        conditionScore = 0.90,
        cheiralityRatio = 0.98,
        robustReprojectionP95Px = 0.8,
        viewDiversityScore = 0.90,
        gazeStability = 0.92,
        angularVelocityDegreesPerSecond = 20.0,
        parallaxDegrees = 4.0,
        metricEvidenceAuthority = 0.92,
        symmetricEpipolarConsistencyScore = 0.94,
        meanIrisRadiusNormalized = 0.22,
        meanIrisStructureConfidence = 0.92,
        meanEyelidOcclusion = 0.08,
    )

    @Test
    fun highQualityFaceEyeEvidenceIsMetricReady() {
        val result = engine.evaluate(
            FaceEyeMetricEvidenceEngine.Input(
                face = face(),
                leftEye = eye(),
                rightEye = eye().copy(
                    eye = com.nivukx.nox.core.model.EyeSide.RIGHT,
                ),
                calibrationConfidence = 0.95,
                temporalStability = 0.92,
                gazeCalibrationConfidence = 0.94,
                faceCalibrationDrift = 0.04,
                headMotionConfidence = 0.90,
            ),
        )
        assertEquals(
            FaceEyeMetricEvidenceEngine.State.METRIC_READY,
            result.state,
        )
        assertTrue(result.confidence > 0.76)
    }

    @Test
    fun weakEyeEpipolarEvidenceBlocksMetricPromotion() {
        val weakEye =
            eye().copy(
                symmetricEpipolarConsistencyScore = 0.22,
            )

        val result =
            engine.evaluate(
                FaceEyeMetricEvidenceEngine.Input(
                    face = face(),
                    leftEye = weakEye,
                    rightEye = weakEye.copy(
                        eye = com.nivukx.nox.core.model.EyeSide.RIGHT,
                    ),
                    calibrationConfidence = 0.95,
                    temporalStability = 0.92,
                    gazeCalibrationConfidence = 0.94,
                    faceCalibrationDrift = 0.04,
                    headMotionConfidence = 0.90,
                ),
            )

        assertTrue(result.state != FaceEyeMetricEvidenceEngine.State.METRIC_READY)
        assertTrue(
            result.warnings.contains(
                "EYE_EPIPOLAR_CONSISTENCY_LOW",
            ),
        )
        assertTrue(result.epipolarConsistency < 0.55)
    }

    @Test
    fun weakEyeMetricAuthorityBlocksMetricPromotion() {
        val weakEye =
            eye().copy(
                metricEvidenceAuthority = 0.14,
            )

        val result =
            engine.evaluate(
                FaceEyeMetricEvidenceEngine.Input(
                    face = face(),
                    leftEye = weakEye,
                    rightEye = weakEye.copy(
                        eye = com.nivukx.nox.core.model.EyeSide.RIGHT,
                    ),
                    calibrationConfidence = 0.95,
                    temporalStability = 0.92,
                    gazeCalibrationConfidence = 0.94,
                    faceCalibrationDrift = 0.04,
                    headMotionConfidence = 0.90,
                ),
            )

        assertTrue(result.state == FaceEyeMetricEvidenceEngine.State.HOLD)
        assertTrue(result.metricEvidenceAuthority < 0.20)
    }

    @Test
    fun low_iris_structure_evidence_is_reported_without_destroying_other_metrics() {
        val weakEye =
            eye().copy(
                meanIrisStructureConfidence = 0.18,
                meanIrisRadiusNormalized = 0.04,
            )

        val result =
            engine.evaluate(
                FaceEyeMetricEvidenceEngine.Input(
                    face = face(),
                    leftEye = weakEye,
                    rightEye = weakEye.copy(
                        eye = com.nivukx.nox.core.model.EyeSide.RIGHT,
                    ),
                    calibrationConfidence = 0.95,
                    temporalStability = 0.92,
                    gazeCalibrationConfidence = 0.94,
                    faceCalibrationDrift = 0.04,
                    headMotionConfidence = 0.90,
                ),
            )

        assertEquals(
            0.18,
            result.irisStructureConfidence,
        )
        assertTrue(
            result.warnings.contains(
                "IRIS_STRUCTURE_CONFIDENCE_LOW",
            ),
        )
        assertTrue(
            result.eyeP95UncertaintyMm < 8.0,
        )
    }

    @Test
    fun highDriftBlocksMetricFaceEyeOutput() {
        val result = engine.evaluate(
            FaceEyeMetricEvidenceEngine.Input(
                face = face(),
                leftEye = eye(),
                rightEye = eye().copy(
                    eye = com.nivukx.nox.core.model.EyeSide.RIGHT,
                ),
                calibrationConfidence = 0.86,
                temporalStability = 0.85,
                gazeCalibrationConfidence = 0.85,
                faceCalibrationDrift = 0.40,
                headMotionConfidence = 0.90,
            ),
        )
        assertTrue(
            result.warnings.contains("FACE_EYE_CALIBRATION_DRIFT_TOO_HIGH"),
        )
    }
}
