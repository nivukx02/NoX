package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraExtrinsics
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Test

class CalibrationSnapshotWireCodecTest {
    @Test
    fun roundTripsOptimizedRig() {
        val reference = DeviceId("master")
        val peer = DeviceId("peer")
        val snapshot = CalibrationSnapshot(
            intrinsics = mapOf(
                reference to CameraIntrinsics(
                    1000.0, 1001.0, 500.0, 400.0,
                    imageWidth = 1000,
                    imageHeight = 800,
                ),
                peer to CameraIntrinsics(
                    900.0, 901.0, 450.0, 360.0,
                    imageWidth = 900,
                    imageHeight = 720,
                ),
            ),
            extrinsics = mapOf(
                reference to CameraExtrinsics(
                    Pose(Vec3(0.0, 0.0, 0.0), Quat.IDENTITY),
                    reference,
                ),
                peer to CameraExtrinsics(
                    Pose(Vec3(1.2, 0.2, -0.4), Quat(0.99, 0.02, 0.03, -0.01).normalized()),
                    reference,
                ),
            ),
            reprojectionErrorPx = 1.42,
            calibratedAtNanos = 123456789L,
        )

        val decoded = CalibrationSnapshotWireCodec.decode(
            CalibrationSnapshotWireCodec.encode(snapshot),
        )

        assertEquals(snapshot.intrinsics.size, decoded.intrinsics.size)
        assertEquals(snapshot.reprojectionErrorPx, decoded.reprojectionErrorPx, 1e-9)
        assertEquals(
            snapshot.extrinsics.getValue(peer).pose.position.z,
            decoded.extrinsics.getValue(peer).pose.position.z,
            1e-9,
        )
        assertEquals(reference, decoded.extrinsics.getValue(peer).referenceDevice)
    }

    @Test
    fun roundTripsCalibrationFocusStateInV3() {
        val reference = DeviceId("master")
        val peer = DeviceId("peer")
        val snapshot = CalibrationSnapshot(
            intrinsics = mapOf(
                reference to CameraIntrinsics(1000.0, 1000.0, 500.0, 400.0),
                peer to CameraIntrinsics(1001.0, 1002.0, 501.0, 401.0),
            ),
            extrinsics = mapOf(
                reference to CameraExtrinsics(Pose(), reference),
                peer to CameraExtrinsics(
                    Pose(Vec3(1.0, 0.0, 0.0), Quat.IDENTITY),
                    reference,
                ),
            ),
            reprojectionErrorPx = 0.21,
            calibratedAtNanos = 99L,
            calibrationFocusDistanceDiopters = mapOf(
                reference to 2.5,
                peer to 2.6,
            ),
        )

        val decoded =
            CalibrationSnapshotWireCodec.decode(
                CalibrationSnapshotWireCodec.encode(snapshot),
            )

        assertEquals(
            2.5,
            decoded.calibrationFocusDistanceDiopters[
                reference
            ],
            1e-12,
        )
        assertEquals(
            2.6,
            decoded.calibrationFocusDistanceDiopters[
                peer
            ],
            1e-12,
        )
    }

    @Test
    fun v2RemainsCompatibleWithoutFocusState() {
        val reference = DeviceId("master")
        val snapshot = CalibrationSnapshot(
            intrinsics = mapOf(
                reference to CameraIntrinsics(1000.0, 1000.0, 500.0, 400.0),
            ),
            extrinsics = mapOf(
                reference to CameraExtrinsics(Pose(), reference),
            ),
            reprojectionErrorPx = 0.50,
            calibratedAtNanos = 100L,
            uncertaintyProfiles = mapOf(
                reference to CalibrationUncertaintyProfile(
                    translationSigmaMeters = 0.001,
                    rotationSigmaRadians = 0.0005,
                    intrinsicRelativeSigma = 0.001,
                    evidenceConfidence = 0.8,
                ),
            ),
            calibrationFocusDistanceDiopters = mapOf(
                reference to 3.0,
            ),
        )

        val decoded =
            CalibrationSnapshotWireCodec.decode(
                CalibrationSnapshotWireCodec.encode(
                    snapshot,
                    requestedVersion = 2,
                ),
            )

        assertEquals(
            0.001,
            decoded.uncertaintyProfiles[
                reference
            ]?.translationSigmaMeters,
            1e-12,
        )
        assertEquals(
            emptyMap(),
            decoded.calibrationFocusDistanceDiopters,
        )
    }

    @Test
    fun roundTripsPerCameraUncertaintyProfile() {
        val reference = DeviceId("master")
        val snapshot = CalibrationSnapshot(
            intrinsics = mapOf(
                reference to CameraIntrinsics(
                    fx = 1000.0,
                    fy = 1000.0,
                    cx = 500.0,
                    cy = 400.0,
                    imageWidth = 1000,
                    imageHeight = 800,
                ),
            ),
            extrinsics = mapOf(
                reference to CameraExtrinsics(
                    Pose(),
                    reference,
                ),
            ),
            reprojectionErrorPx = 0.42,
            calibratedAtNanos = 77L,
            uncertaintyProfiles = mapOf(
                reference to CalibrationUncertaintyProfile(
                    translationSigmaMeters = 0.0008,
                    rotationSigmaRadians = 0.00045,
                    intrinsicRelativeSigma = 0.0007,
                    evidenceConfidence = 0.93,
                ),
            ),
        )

        val decoded =
            CalibrationSnapshotWireCodec.decode(
                CalibrationSnapshotWireCodec.encode(snapshot),
            )
        val profile =
            decoded.uncertaintyProfiles.getValue(reference)

        assertEquals(
            0.0008,
            profile.translationSigmaMeters,
            1e-12,
        )
        assertEquals(
            0.00045,
            profile.rotationSigmaRadians,
            1e-12,
        )
        assertEquals(
            0.0007,
            profile.intrinsicRelativeSigma,
            1e-12,
        )
        assertEquals(
            0.93,
            profile.evidenceConfidence,
            1e-12,
        )
    }
    @Test
    fun roundTripsV4CalibrationProvenance() {
        val reference = DeviceId("master")
        val snapshot =
            CalibrationSnapshot(
                intrinsics =
                    mapOf(
                        reference to
                            CameraIntrinsics(
                                1200.0,
                                1200.0,
                                640.0,
                                360.0,
                                imageWidth = 1280,
                                imageHeight = 720,
                            ),
                    ),
                extrinsics =
                    mapOf(
                        reference to
                            CameraExtrinsics(
                                Pose(),
                                reference,
                            ),
                    ),
                reprojectionErrorPx = 0.18,
                calibratedAtNanos = 123L,
                calibrationMethod =
                    "SELF_CALIBRATION_NATIVE_INTRINSIC_REFINEMENT",
                nativeIntrinsicRefinementApplied = true,
                nativeIntrinsicRefinementConfidence = 0.97,
                nativeIntrinsicRefinementImprovementPx = 0.21,
                timingAuthorityConfidence = 0.98,
                timingAuthorityResidualP95Milliseconds = 0.62,
            )

        val decoded =
            CalibrationSnapshotWireCodec.decode(
                CalibrationSnapshotWireCodec.encode(snapshot),
            )

        assertEquals(
            snapshot.calibrationMethod,
            decoded.calibrationMethod,
        )
        assertEquals(
            snapshot.nativeIntrinsicRefinementApplied,
            decoded.nativeIntrinsicRefinementApplied,
        )
        assertEquals(
            snapshot.nativeIntrinsicRefinementConfidence,
            decoded.nativeIntrinsicRefinementConfidence,
            1e-12,
        )
        assertEquals(
            snapshot.nativeIntrinsicRefinementImprovementPx,
            decoded.nativeIntrinsicRefinementImprovementPx,
            1e-12,
        )
        assertEquals(
            snapshot.timingAuthorityConfidence,
            decoded.timingAuthorityConfidence,
            1e-12,
        )
        assertEquals(
            snapshot.timingAuthorityResidualP95Milliseconds,
            decoded.timingAuthorityResidualP95Milliseconds,
            1e-12,
        )
    }

    @Test
    fun v3DecodeUsesSafeLegacyProvenanceDefaults() {
        val reference = DeviceId("master")
        val snapshot =
            CalibrationSnapshot(
                intrinsics =
                    mapOf(
                        reference to
                            CameraIntrinsics(
                                1000.0,
                                1000.0,
                                500.0,
                                400.0,
                            ),
                    ),
                extrinsics =
                    mapOf(
                        reference to
                            CameraExtrinsics(
                                Pose(),
                                reference,
                            ),
                    ),
                reprojectionErrorPx = 0.50,
                calibratedAtNanos = 100L,
            )

        val decoded =
            CalibrationSnapshotWireCodec.decode(
                CalibrationSnapshotWireCodec.encode(
                    snapshot,
                    requestedVersion = 3,
                ),
            )

        assertEquals("UNKNOWN", decoded.calibrationMethod)
        assertEquals(
            false,
            decoded.nativeIntrinsicRefinementApplied,
        )
        assertEquals(
            0.0,
            decoded.timingAuthorityConfidence,
            0.0,
        )
    }

}
