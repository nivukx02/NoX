package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiCameraCalibrationDriftEngineTest {

    private fun camera(
        id: String,
        x: Double,
        y: Double = 0.0,
        z: Double = 0.0,
    ): MultiCameraCalibrationDriftEngine.BaselineCamera =
        MultiCameraCalibrationDriftEngine.BaselineCamera(
            deviceId = DeviceId(id),
            pose = Pose(
                position = Vec3(x, y, z),
                rotation = Quat.IDENTITY,
            ),
        )

    private fun current(
        id: String,
        x: Double,
        y: Double = 0.0,
        z: Double = 0.0,
        reprojection: Double = 0.0,
        thermal: Double = 0.0,
        vibration: Double = 0.0,
        temporalAlignmentConfidence: Double = 1.0,
        subframeOffsetMillis: Double = 0.0,
    ): MultiCameraCalibrationDriftEngine.CurrentCamera =
        MultiCameraCalibrationDriftEngine.CurrentCamera(
            deviceId = DeviceId(id),
            pose = Pose(
                position = Vec3(x, y, z),
                rotation = Quat.IDENTITY,
            ),
            reprojectionErrorPx = reprojection,
            thermalRisk = thermal,
            vibrationRisk = vibration,
            temporalAlignmentConfidence =
                temporalAlignmentConfidence,
            subframeOffsetMillis =
                subframeOffsetMillis,
        )

    @Test
    fun metrology_profile_is_strict_but_does_not_claim_accuracy_by_itself() {
        val config =
            MultiCameraCalibrationDriftEngine.Config.metrologyQualification()

        assertTrue(config.translationWarningMeters < 0.001)
        assertTrue(config.rotationWarningDegrees < 0.10)
        assertTrue(config.reprojectionWarningPx <= 0.50)
        assertTrue(config.maximumSubframeOffsetMillis <= 0.50)

        val engine =
            MultiCameraCalibrationDriftEngine(config)
        engine.establishBaseline(
            cameras = listOf(
                camera("A", 0.0),
                camera("B", 1.0),
                camera("C", 0.0, 0.0, 1.0),
            ),
            timestampNanos = 1L,
        )

        val result =
            engine.observe(
                timestampNanos = 20_000_000L,
                cameras = listOf(
                    current("A", 0.0),
                    current("B", 1.0002),
                    current("C", 0.0, 0.0, 1.0),
                ),
            )

        assertTrue(result.rigTrust > 0.80)
        assertTrue(!result.hardFailure)
    }

    @Test
    fun stableRigMaintainsHighTrust() {
        val engine = MultiCameraCalibrationDriftEngine()
        engine.establishBaseline(
            cameras = listOf(
                camera("A", 0.0),
                camera("B", 1.0),
                camera("C", 0.0, 0.0, 1.0),
            ),
            timestampNanos = 1L,
        )

        val result = engine.observe(
            timestampNanos = 20_000_000L,
            cameras = listOf(
                current("A", 0.0),
                current("B", 1.0),
                current("C", 0.0, 0.0, 1.0),
            ),
        )

        assertTrue(result.rigTrust > 0.80)
        assertTrue(!result.hardFailure)
        assertTrue(!result.recalibrationRecommended)
    }

    @Test
    fun translationDriftReducesTrust() {
        val engine = MultiCameraCalibrationDriftEngine()
        engine.establishBaseline(
            cameras = listOf(
                camera("A", 0.0),
                camera("B", 1.0),
            ),
            timestampNanos = 1L,
        )

        repeat(12) { index ->
            engine.observe(
                timestampNanos = (index + 1L) * 20_000_000L,
                cameras = listOf(
                    current("A", 0.0),
                    current("B", 1.08),
                ),
            )
        }

        val result = engine.latest()!!
        assertTrue(result.geometryTrust < 0.80)
        assertTrue(result.recalibrationRecommended)
    }

    @Test
    fun low_confidence_drift_is_degraded_before_it_becomes_a_hard_failure() {
        val engine =
            MultiCameraCalibrationDriftEngine(
                MultiCameraCalibrationDriftEngine.Config(
                    minimumHardFailureConfidence = 0.80,
                    persistentHardFailureSamples = 3,
                ),
            )

        engine.establishBaseline(
            cameras = listOf(
                camera("A", 0.0),
                camera("B", 1.0),
            ),
            timestampNanos = 1L,
        )

        val transient =
            engine.observe(
                timestampNanos = 40_000_000L,
                cameras = listOf(
                    current("A", 0.0),
                    current(
                        "B",
                        1.20,
                        reprojection = 6.0,
                    ).copy(confidence = 0.35),
                ),
            )

        assertTrue(!transient.hardFailure)
        assertTrue(transient.degraded)
        assertTrue(transient.recalibrationRecommended)
    }

    @Test
    fun pairBaselineHardFailureHoldsMultiViewOutput() {
        val engine = MultiCameraCalibrationDriftEngine()
        engine.establishBaseline(
            cameras = listOf(
                camera("A", 0.0),
                camera("B", 1.0),
            ),
            timestampNanos = 1L,
        )

        val result = engine.observe(
            timestampNanos = 40_000_000L,
            cameras = listOf(
                current("A", 0.0),
                current("B", 1.20),
            ),
        )

        assertTrue(result.hardFailure)
        assertTrue(engine.shouldHoldMultiViewOutput())
    }

    @Test
    fun poorTemporalAlignmentRecommendsRecalibrationWithoutGeometryDrift() {
        val engine =
            MultiCameraCalibrationDriftEngine()

        engine.establishBaseline(
            cameras = listOf(
                camera("A", 0.0),
                camera("B", 1.0),
            ),
            timestampNanos = 1L,
        )

        val result =
            engine.observe(
                timestampNanos = 40_000_000L,
                cameras = listOf(
                    current(
                        "A",
                        0.0,
                        temporalAlignmentConfidence = 0.32,
                        subframeOffsetMillis = 9.0,
                    ),
                    current(
                        "B",
                        1.0,
                        temporalAlignmentConfidence = 0.38,
                        subframeOffsetMillis = 8.0,
                    ),
                ),
            )

        assertTrue(result.temporalTrust < 0.60)
        assertTrue(result.recalibrationRecommended)
        assertTrue(
            result.warnings.any {
                it.contains(
                    "temporal",
                    ignoreCase = true,
                )
            },
        )
    }

    @Test
    fun non_finite_runtime_camera_evidence_is_rejected_at_input_boundary() {
        val camera =
            CurrentCamera(
                deviceId = DeviceId("runtime"),
                pose =
                    Pose(
                        position = Vec3(0.0, 0.0, 0.0),
                        rotation = com.nivukx.nox.core.model.Quat.IDENTITY,
                    ),
                reprojectionErrorPx = 1.0,
                confidence = 0.9,
                thermalRisk = 0.1,
                vibrationRisk = 0.1,
                temporalAlignmentConfidence = 0.9,
                subframeOffsetMillis = 0.5,
            )

        assertTrue(camera.confidence.isFinite())
        assertTrue(camera.subframeOffsetMillis.isFinite())
    }

    @Test(expected = IllegalArgumentException::class)
    fun nan_reprojection_is_rejected() {
        CurrentCamera(
            deviceId = DeviceId("nan"),
            pose =
                Pose(
                    position = Vec3(0.0, 0.0, 0.0),
                    rotation = com.nivukx.nox.core.model.Quat.IDENTITY,
                ),
            reprojectionErrorPx = Double.NaN,
        )
    }

    @Test
    fun missingBaselineCannotAuthorizeMultiView() {
        val engine = MultiCameraCalibrationDriftEngine()
        val result = engine.observe(
            timestampNanos = 100L,
            cameras = listOf(
                current("A", 0.0),
                current("B", 1.0),
            ),
        )

        assertTrue(result.hardFailure)
        assertTrue(result.recalibrationRecommended)
        assertTrue(engine.shouldHoldMultiViewOutput())
    }
    @Test
    fun cameraAndPairTrustAggregationRegression() {
        val engine = MultiCameraCalibrationDriftEngine()

        engine.establishBaseline(
            cameras = listOf(
                camera("A", 0.0),
                camera("B", 1.0),
            ),
            timestampNanos = 1L,
        )

        val result = engine.observe(
            timestampNanos = 20_000_000L,
            cameras = listOf(
                current("A", 0.0),
                current("B", 1.0),
            ),
        )

        assertTrue(result.cameraDrift.isNotEmpty())
        assertTrue(result.pairDrift.isNotEmpty())
        assertTrue(result.rigTrust > 0.80)
    }

}
