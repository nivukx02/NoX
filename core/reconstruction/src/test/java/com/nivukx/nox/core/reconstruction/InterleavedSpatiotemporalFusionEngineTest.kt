package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class InterleavedSpatiotemporalFusionEngineTest {
    @Test
    fun interleavedCamerasCreateHigherEffectiveTemporalSampling() {
        val engine = InterleavedSpatiotemporalFusionEngine(
            InterleavedSpatiotemporalFusionEngine.Config(
                maximumTemporalWindowNanos = 30_000_000L,
                minimumSamplesForAuthoritativeOutput = 2,
            ),
        )

        val frames = listOf(
            frame("A", 0L, 0L, 100.0),
            frame("B", 8_000_000L, 0L, 101.0),
            frame("A", 16_000_000L, 1L, 102.0),
            frame("B", 24_000_000L, 1L, 103.0),
        )

        val result = engine.fuseDetailed(12_000_000L, frames)

        assertEquals(2, result.frame.cameraCount)
        assertEquals(1, result.frame.joints.size)
        assertTrue(result.frame.effectiveInputRateHz > 100.0)
        assertTrue(result.frame.temporalConsistency > 0.80)
        assertTrue(result.frame.spatialAgreement > 0.80)
        assertTrue(result.frame.joints.single().position.x in 101.0..102.0)
    }

    @Test
    fun grossOutlierDoesNotDominateFusion() {
        val engine = InterleavedSpatiotemporalFusionEngine(
            InterleavedSpatiotemporalFusionEngine.Config(
                minimumSamplesForAuthoritativeOutput = 2,
                outlierDistancePixels = 20.0,
            ),
        )

        val frames = listOf(
            frame("A", 0L, 0L, 100.0, 0.98),
            frame("B", 10_000_000L, 0L, 101.0, 0.98),
            frame("C", 10_000_000L, 0L, 400.0, 1.0),
        )

        val result = engine.fuse(10_000_000L, frames)
        val joint = result.joints.single()

        assertTrue(joint.position.x < 110.0)
        assertTrue(joint.supportingCameras >= 2)
    }

    @Test
    fun boundedPredictionIsMarkedAndNeverUnlimited() {
        val engine = InterleavedSpatiotemporalFusionEngine(
            InterleavedSpatiotemporalFusionEngine.Config(
                maximumPredictionNanos = 5_000_000L,
                maximumTemporalWindowNanos = 30_000_000L,
            ),
        )

        val frames = listOf(
            frame("A", 0L, 0L, 100.0),
            frame("B", 10_000_000L, 1L, 110.0),
            frame("A", 20_000_000L, 2L, 120.0),
        )

        val result = engine.fuse(24_000_000L, frames)

        assertFalse(result.joints.isEmpty())
        assertTrue(result.joints.single().predicted)
        assertTrue(
            result.warnings.any {
                it.contains("bounded temporal prediction")
            },
        )
    }

    @Test
    fun measured_phase_offsets_align_interleaved_samples_before_fusion() {
        val engine =
            InterleavedSpatiotemporalFusionEngine(
                InterleavedSpatiotemporalFusionEngine.Config(
                    maximumTemporalWindowNanos = 20_000_000L,
                    maximumSampleSkewNanos = 4_000_000L,
                    minimumSamplesForAuthoritativeOutput = 2,
                ),
            )

        val frames =
            listOf(
                InterleavedSpatiotemporalFusionEngine.CameraFrame(
                    cameraId = "A",
                    timestampNanos = 0L,
                    sequence = 0L,
                    phaseOffsetNanos = 0L,
                    observations = listOf(
                        InterleavedSpatiotemporalFusionEngine.JointObservation(
                            jointId = 7,
                            position = Vec2(100.0, 200.0),
                            confidence = 0.98,
                        ),
                    ),
                ),
                InterleavedSpatiotemporalFusionEngine.CameraFrame(
                    cameraId = "B",
                    timestampNanos = 8_000_000L,
                    sequence = 0L,
                    phaseOffsetNanos = -8_000_000L,
                    observations = listOf(
                        InterleavedSpatiotemporalFusionEngine.JointObservation(
                            jointId = 7,
                            position = Vec2(100.5, 200.0),
                            confidence = 0.98,
                        ),
                    ),
                ),
            )

        val result =
            engine.fuseDetailed(
                targetTimestampNanos = 0L,
                frames = frames,
            )

        assertEquals(2, result.frame.cameraCount)
        assertTrue(result.frame.authoritative)
        assertTrue(result.frame.temporalConsistency > 0.90)
        assertTrue(result.frame.joints.single().supportingCameras >= 2)
    }

    @Test
    fun low_confidence_sparse_samples_do_not_become_authoritative() {
        val engine =
            InterleavedSpatiotemporalFusionEngine(
                InterleavedSpatiotemporalFusionEngine.Config(
                    minimumConfidence = 0.80,
                    minimumSamplesForAuthoritativeOutput = 2,
                ),
            )

        val result =
            engine.fuse(
                targetTimestampNanos = 10_000_000L,
                frames = listOf(
                    frame(
                        camera = "A",
                        timestamp = 10_000_000L,
                        sequence = 0L,
                        x = 100.0,
                        confidence = 0.50,
                    ),
                    frame(
                        camera = "B",
                        timestamp = 10_000_000L,
                        sequence = 0L,
                        x = 100.5,
                        confidence = 0.55,
                    ),
                ),
            )

        assertTrue(result.joints.isEmpty())
        assertFalse(result.authoritative)
        assertTrue(
            result.warnings.any {
                it.contains("No joint")
            },
        )
    }

    @Test
    fun calibrated_epipolar_consistency_preserves_valid_stereo_evidence() {
        val engine =
            InterleavedSpatiotemporalFusionEngine(
                InterleavedSpatiotemporalFusionEngine.Config(
                    minimumConfidence = 0.20,
                    minimumSamplesForAuthoritativeOutput = 2,
                    outlierDistancePixels = 40.0,
                ),
            )

        val intrinsics =
            CameraIntrinsics(
                fx = 1_000.0,
                fy = 1_000.0,
                cx = 320.0,
                cy = 240.0,
                k1 = 0.0,
                k2 = 0.0,
                k3 = 0.0,
                p1 = 0.0,
                p2 = 0.0,
            )

        val frames =
            listOf(
                InterleavedSpatiotemporalFusionEngine.CameraFrame(
                    cameraId = "left",
                    timestampNanos = 10_000_000L,
                    sequence = 0L,
                    cameraPose =
                        Pose(
                            position = Vec3(0.0, 0.0, 0.0),
                            rotation = Quat.IDENTITY,
                        ),
                    intrinsics = intrinsics,
                    observations =
                        listOf(
                            InterleavedSpatiotemporalFusionEngine.JointObservation(
                                jointId = 7,
                                position = Vec2(320.0, 240.0),
                                confidence = 0.99,
                            ),
                        ),
                ),
                InterleavedSpatiotemporalFusionEngine.CameraFrame(
                    cameraId = "right",
                    timestampNanos = 10_000_000L,
                    sequence = 0L,
                    cameraPose =
                        Pose(
                            position = Vec3(1.0, 0.0, 0.0),
                            rotation = Quat.IDENTITY,
                        ),
                    intrinsics = intrinsics,
                    observations =
                        listOf(
                            InterleavedSpatiotemporalFusionEngine.JointObservation(
                                jointId = 7,
                                position = Vec2(120.0, 240.0),
                                confidence = 0.99,
                            ),
                        ),
                ),
            )

        val result =
            engine.fuseDetailed(
                targetTimestampNanos = 10_000_000L,
                frames = frames,
            )

        assertEquals(2, result.frame.cameraCount)
        assertTrue(result.frame.authoritative)
        assertTrue(
            result.joints.single().supportingCameras == 2,
        )
    }

    @Test
    fun temporally_distant_views_are_not_forced_through_same_moment_epipolar_gate() {
        val engine =
            InterleavedSpatiotemporalFusionEngine(
                InterleavedSpatiotemporalFusionEngine.Config(
                    minimumSamplesForAuthoritativeOutput = 2,
                    maximumEpipolarTemporalDeltaNanos = 2_000_000L,
                ),
            )

        val intrinsics =
            CameraIntrinsics(
                fx = 1_000.0,
                fy = 1_000.0,
                cx = 320.0,
                cy = 240.0,
            )

        val frames =
            listOf(
                InterleavedSpatiotemporalFusionEngine.CameraFrame(
                    cameraId = "A",
                    timestampNanos = 0L,
                    sequence = 0L,
                    cameraPose =
                        Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                    intrinsics = intrinsics,
                    observations =
                        listOf(
                            InterleavedSpatiotemporalFusionEngine.JointObservation(
                                jointId = 9,
                                position = Vec2(320.0, 240.0),
                                confidence = 0.99,
                            ),
                        ),
                ),
                InterleavedSpatiotemporalFusionEngine.CameraFrame(
                    cameraId = "B",
                    timestampNanos = 8_000_000L,
                    sequence = 0L,
                    cameraPose =
                        Pose(
                            Vec3(1.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                    intrinsics = intrinsics,
                    observations =
                        listOf(
                            InterleavedSpatiotemporalFusionEngine.JointObservation(
                                jointId = 9,
                                position = Vec2(140.0, 240.0),
                                confidence = 0.99,
                            ),
                        ),
                ),
            )

        val result =
            engine.fuseDetailed(
                targetTimestampNanos = 4_000_000L,
                frames = frames,
            )

        assertTrue(result.frame.cameraCount == 2)
        assertTrue(result.frame.joints.single().supportingCameras == 2)
        assertTrue(result.frame.authoritative)
    }

    @Test
    fun staleEvidenceIsRejected() {
        val engine = InterleavedSpatiotemporalFusionEngine(
            InterleavedSpatiotemporalFusionEngine.Config(
                maximumTemporalWindowNanos = 5_000_000L,
                maximumPredictionNanos = 5_000_000L,
            ),
        )

        val result = engine.fuse(
            targetTimestampNanos = 100_000_000L,
            frames = listOf(
                frame("A", 0L, 0L, 100.0),
                frame("B", 1_000_000L, 0L, 100.5),
            ),
        )

        assertTrue(result.joints.isEmpty())
        assertFalse(result.authoritative)
    }

    private fun frame(
        camera: String,
        timestamp: Long,
        sequence: Long,
        x: Double,
        confidence: Double = 1.0,
    ) =
        InterleavedSpatiotemporalFusionEngine.CameraFrame(
            cameraId = camera,
            timestampNanos = timestamp,
            sequence = sequence,
            observations = listOf(
                InterleavedSpatiotemporalFusionEngine.JointObservation(
                    jointId = 7,
                    position = Vec2(x, 200.0),
                    confidence = confidence,
                ),
            ),
        )
}
