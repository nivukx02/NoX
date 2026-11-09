package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FaceContour
import com.nivukx.nox.core.model.FaceContourFrame
import com.nivukx.nox.core.model.FaceContourKind
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class Face3DReconstructionEvidenceTest {

    private val intrinsics =
        CameraIntrinsics(
            fx = 1200.0,
            fy = 1200.0,
            cx = 640.0,
            cy = 360.0,
            imageWidth = 1280,
            imageHeight = 720,
        )

    private val leftCamera = DeviceId("face-a")
    private val rightCamera = DeviceId("face-b")

    private fun contour(
        kind: FaceContourKind,
        cameraX: Double,
    ): FaceContour {
        val world =
            listOf(
                Vec3(0.15, 0.10, 4.0),
                Vec3(0.18, 0.12, 4.0),
            )
        val points =
            world.map { p ->
                Vec2(
                    intrinsics.fx *
                        (p.x - cameraX) /
                        p.z +
                        intrinsics.cx,
                    intrinsics.fy *
                        p.y /
                        p.z +
                        intrinsics.cy,
                )
            }
        return FaceContour(
            kind = kind,
            points = points,
            confidence = 0.98,
        )
    }

    private fun frame(
        cameraX: Double,
        timestampNanos: Long,
    ): Face3DReconstructionEngine.CameraFrame =
        Face3DReconstructionEngine.CameraFrame(
            deviceId =
                if (cameraX == 0.0) leftCamera else rightCamera,
            timestampNanos = timestampNanos,
            face =
                FaceContourFrame(
                    timestampNanos = timestampNanos,
                    boundsLeft = 300f,
                    boundsTop = 120f,
                    boundsRight = 980f,
                    boundsBottom = 640f,
                    yawDegrees = 0f,
                    rollDegrees = 0f,
                    contours =
                        FaceContourKind.entries
                            .take(10)
                            .map {
                                contour(
                                    kind = it,
                                    cameraX = cameraX,
                                )
                            },
                    imageWidth = 1280,
                    imageHeight = 720,
                    rotationDegrees = 0,
                    captureTimingConfidence = 0.98,
                    rollingShutterSkewNanos = 1_000_000L,
                    frameDurationNanos = 16_666_667L,
                    exposureTimeNanos = 2_000_000L,
                ),
        )

    @Test
    fun faceMeshRetainsMetricUncertaintyAndEpipolarAuthority() {
        val engine =
            Face3DReconstructionEngine()

        val result =
            engine.reconstruct(
                targetTimestampNanos = 1_000_000_000L,
                frames =
                    listOf(
                        frame(
                            cameraX = 0.0,
                            timestampNanos = 1_000_000_000L,
                        ),
                        frame(
                            cameraX = 1.0,
                            timestampNanos = 1_000_000_000L,
                        ),
                    ),
                intrinsics =
                    mapOf(
                        leftCamera to intrinsics,
                        rightCamera to intrinsics,
                    ),
                extrinsics =
                    mapOf(
                        leftCamera to
                            Pose(Vec3(0.0, 0.0, 0.0)),
                        rightCamera to
                            Pose(Vec3(1.0, 0.0, 0.0)),
                    ),
                calibrationUncertaintyProfiles =
                    mapOf(
                        leftCamera to
                            CalibrationUncertaintyProfile(
                                translationSigmaMeters = 0.001,
                                rotationSigmaRadians = 0.001,
                                intrinsicRelativeSigma = 0.002,
                                evidenceConfidence = 0.98,
                            ),
                        rightCamera to
                            CalibrationUncertaintyProfile(
                                translationSigmaMeters = 0.001,
                                rotationSigmaRadians = 0.001,
                                intrinsicRelativeSigma = 0.002,
                                evidenceConfidence = 0.98,
                            ),
                    ),
            )

        assertTrue(result != null)
        result ?: return

        assertTrue(result.validPointCount >= 20)
        assertEquals(
            result.validPointCount,
            result.mesh.vertices.size,
        )
        assertEquals(
            result.validPointCount,
            result.mesh.vertexUncertaintyMeters.size,
        )
        assertTrue(
            result.meanEpipolarConsistencyScore > 0.90,
        )
        assertTrue(
            result.meanMetricEvidenceAuthority > 0.50,
        )
        assertTrue(
            !result.strictSubMillimeterQualified,
        )
        assertTrue(
            result.mesh.vertices.all {
                it.uncertaintyMeters.isFinite() &&
                    it.uncertaintyMeters >= 0.0
            },
        )
    }
}
