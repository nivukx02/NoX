package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.calibration.MetrologyPrecisionQualificationCoordinator
import com.nivukx.nox.core.calibration.MultiCameraCalibrationDriftEngine
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MetrologyPrecisionQualificationCoordinatorTest {

    private fun stableCalibration():
        MultiCameraCalibrationDriftEngine.Snapshot {
        val engine =
            MultiCameraCalibrationDriftEngine(
                MultiCameraCalibrationDriftEngine
                    .Config
                    .metrologyQualification(),
            )

        engine.establishBaseline(
            cameras = listOf(
                camera("A", 0.0),
                camera("B", 1.0),
                camera("C", 0.0, 0.0, 1.0),
            ),
            timestampNanos = 1L,
        )

        return engine.observe(
            timestampNanos = 20_000_000L,
            cameras = listOf(
                current("A", 0.0),
                current("B", 1.0),
                current("C", 0.0, 0.0, 1.0),
            ),
        )
    }

    private fun precisionReport():
        CapturePrecisionBenchmarkEngine.Report {
        val engine =
            CapturePrecisionBenchmarkEngine(
                CapturePrecisionBenchmarkEngine.Config(
                    p50LimitMm = 0.75,
                    p95LimitMm = 0.75,
                    p99LimitMm = 1.75,
                    jitterLimitMm = 0.50,
                    lagLimitMs = 5.0,
                    scaleDriftLimitPercentPerMinute = 0.25,
                    minimumSamples = 8,
                ),
            )

        val samples =
            buildList {
                repeat(8) { frame ->
                    repeat(2) { joint ->
                        val reference =
                            Vec3(
                                x = joint * 0.10,
                                y = frame * 0.01,
                                z = 1.0 + joint * 0.02,
                            )
                        add(
                            CapturePrecisionBenchmarkEngine.JointSample(
                                timestampNanos =
                                    (frame + 1L) * 10_000_000L,
                                jointId = joint,
                                estimate = reference,
                                reference = reference,
                                confidence = 0.99,
                                estimatedScale = 1.0,
                                referenceScale = 1.0,
                            ),
                        )
                    }
                }
            }

        val surfaceSamples =
            buildList {
                repeat(8) { frame ->
                    val point =
                        Vec3(
                            0.10,
                            frame * 0.01,
                            1.0,
                        )
                    add(
                        CapturePrecisionBenchmarkEngine.SurfaceSample(
                            timestampNanos =
                                (frame + 1L) * 10_000_000L,
                            estimate = point,
                            reference = point,
                            estimateNormal =
                                Vec3(0.0, 0.0, 1.0),
                            referenceNormal =
                                Vec3(0.0, 0.0, 1.0),
                            confidence = 0.99,
                        ),
                    )
                }
            }

        return engine.evaluate(
            CapturePrecisionBenchmarkEngine.Input(
                captureId = "metrology-test",
                referenceType =
                    CapturePrecisionBenchmarkEngine.ReferenceType.SYNTHETIC_GROUND_TRUTH,
                samples = samples,
                surfaceSamples = surfaceSamples,
            ),
        )
    }

    private fun camera(
        id: String,
        x: Double,
        y: Double = 0.0,
        z: Double = 0.0,
    ): MultiCameraCalibrationDriftEngine.BaselineCamera =
        MultiCameraCalibrationDriftEngine.BaselineCamera(
            deviceId = DeviceId(id),
            pose =
                Pose(
                    position = Vec3(x, y, z),
                    rotation = Quat.IDENTITY,
                ),
        )

    private fun current(
        id: String,
        x: Double,
        y: Double = 0.0,
        z: Double = 0.0,
    ): MultiCameraCalibrationDriftEngine.CurrentCamera =
        MultiCameraCalibrationDriftEngine.CurrentCamera(
            deviceId = DeviceId(id),
            pose =
                Pose(
                    position = Vec3(x, y, z),
                    rotation = Quat.IDENTITY,
                ),
            reprojectionErrorPx = 0.1,
            confidence = 0.99,
            thermalRisk = 0.02,
            vibrationRisk = 0.02,
            temporalAlignmentConfidence = 0.99,
            subframeOffsetMillis = 0.05,
        )

    @Test
    fun stable_low_error_capture_enters_metrology_qualification() {
        val result =
            MetrologyPrecisionQualificationCoordinator().qualify(
                precisionReport = precisionReport(),
                calibrationSnapshot = stableCalibration(),
            )

        assertTrue(result.qualified)
        assertTrue(result.calibrationQualified)
        assertTrue(result.precisionQualified)
        assertTrue(
            result.status ==
                MetrologyPrecisionQualificationCoordinator
                    .Status.QUALIFICATION_READY,
        )
    }

    @Test
    fun calibration_failure_blocks_even_exact_precision() {
        val calibration =
            stableCalibration().copy(
                hardFailure = true,
                recalibrationRecommended = true,
                rigTrust = 0.20,
                geometryTrust = 0.20,
            )

        val result =
            MetrologyPrecisionQualificationCoordinator().qualify(
                precisionReport = precisionReport(),
                calibrationSnapshot = calibration,
            )

        assertFalse(result.qualified)
        assertFalse(result.calibrationQualified)
        assertTrue(
            result.status ==
                MetrologyPrecisionQualificationCoordinator
                    .Status.BLOCKED,
        )
        assertTrue(
            result.blockingReasons.contains(
                "metrology-calibration-not-qualified",
            ),
        )
    }
}
