package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.HandLandmark
import com.nivukx.nox.core.model.HandObservation
import com.nivukx.nox.core.model.HandSide
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import kotlin.math.max
import kotlin.test.assertTrue
import org.junit.Test

class Hand3DReconstructionEvidenceTest {

    private val intrinsics =
        CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
            imageWidth = 1000,
            imageHeight = 1000,
        )

    private val points =
        List(21) { index ->
            Vec3(
                x = ((index % 7) - 3) * 0.035,
                y = ((index / 7) - 1) * 0.045,
                z = 3.5 + index * 0.010,
            )
        }

    private fun hand(
        cameraX: Double,
        timestampNanos: Long,
        timingConfidence: Double,
        predicted: Boolean = false,
    ): Hand3DReconstructionEngine.CameraFrame {
        val landmarks =
            points.mapIndexed { index, world ->
                val projectedX =
                    (
                        intrinsics.fx *
                            (world.x - cameraX) /
                            world.z +
                            intrinsics.cx
                    ) / 1000.0
                val projectedY =
                    (
                        intrinsics.fy *
                            world.y /
                            world.z +
                            intrinsics.cy
                    ) / 1000.0
                HandLandmark(
                    index = index,
                    normalizedX = projectedX.coerceIn(0.0, 1.0),
                    normalizedY = projectedY.coerceIn(0.0, 1.0),
                    normalizedZ = 0.0,
                    worldPositionMeters = world,
                    confidence = 0.98,
                )
            }

        val observation =
            HandObservation(
                handId = if (cameraX < 0.5) 1L else 2L,
                side = HandSide.RIGHT,
                handednessConfidence = 0.98,
                landmarks = landmarks,
                confidence = 0.98,
                timestampNanos = timestampNanos,
                wristVelocityMetersPerSecond =
                    Vec3(0.0, 0.0, 0.0),
                temporalStability = 0.98,
                predicted = predicted,
            )

        return Hand3DReconstructionEngine.CameraFrame(
            deviceId =
                if (cameraX < 0.5) {
                    DeviceId("hand-a")
                } else {
                    DeviceId("hand-b")
                },
            timestampNanos = timestampNanos,
            imageWidth = 1000,
            imageHeight = 1000,
            hands = listOf(observation),
            intrinsics = intrinsics,
            pose =
                Pose(
                    position = Vec3(cameraX, 0.0, 0.0),
                ),
            rollingShutterFraction = 0.04,
            frameDurationNanos = 16_666_667L,
            exposureTimeNanos = 1_500_000L,
            captureTimingConfidence = timingConfidence,
            calibrationUncertainty =
                CalibrationUncertaintyProfile(
                    translationSigmaMeters = 0.001,
                    rotationSigmaRadians = 0.001,
                    intrinsicRelativeSigma = 0.002,
                    evidenceConfidence = 0.98,
                ),
        )
    }

    @Test
    fun highQualityTwoViewHandPreservesMetricAuthority() {
        val engine = Hand3DReconstructionEngine()

        val result =
            engine.reconstruct(
                timestampNanos = 1_000_000_000L,
                frames =
                    listOf(
                        hand(
                            cameraX = 0.0,
                            timestampNanos = 1_000_000_000L,
                            timingConfidence = 0.98,
                        ),
                        hand(
                            cameraX = 1.0,
                            timestampNanos = 1_000_000_000L,
                            timingConfidence = 0.98,
                        ),
                    ),
            )

        assertTrue(result != null)
        result ?: return

        assertTrue(result.hands.isNotEmpty())
        val hand = result.hands.single()
        assertTrue(hand.landmarks.size >= 12)
        assertTrue(hand.metricEvidenceAuthority > 0.45)
        assertTrue(
            hand.symmetricEpipolarConsistencyScore > 0.45,
        )
        assertTrue(
            hand.meanUncertaintyMeters >= 0.0,
        )
        assertTrue(
            hand.confidence in 0.0..1.0,
        )
    }

    @Test
    fun degradedTimingDoesNotCreateArtificialConfidence() {
        val engine = Hand3DReconstructionEngine()
        val good =
            engine.reconstruct(
                timestampNanos = 1_000_000_000L,
                frames =
                    listOf(
                        hand(0.0, 1_000_000_000L, 0.98),
                        hand(1.0, 1_000_000_000L, 0.98),
                    ),
            )!!
        engine.reset()
        val degraded =
            engine.reconstruct(
                timestampNanos = 1_000_000_000L,
                frames =
                    listOf(
                        hand(0.0, 1_000_000_000L, 0.45),
                        hand(1.0, 1_000_000_000L, 0.45),
                    ),
            )!!

        assertTrue(
            degraded.confidence <=
                max(good.confidence, 1e-9),
        )
    }
}
