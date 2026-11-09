package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FaceMeshPoint
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceEyeFusionEngineTest {

    @Test
    fun binocularGazeProducesStableFusedState() {
        val engine = FaceEyeFusionEngine()
        val result = engine.fuse(
            FaceEyeFusionEngine.Input(
                timestampNanos = 1_000_000_000L,
                face = face(),
                eyes = eyes(1_000_000_000L),
            ),
        )

        assertTrue(result.state.confidence >= 0.0)
        assertTrue(result.state.gazeStability in 0.0..1.0)
        assertTrue(result.binocularAgreement > 0.50)
        assertTrue(result.state.combinedGazeCamera.z > 0.5)
    }

    @Test
    fun lowConfidenceLargeGazeJumpIsRejected() {
        val engine = FaceEyeFusionEngine()
        val first = engine.fuse(
            FaceEyeFusionEngine.Input(
                timestampNanos = 1_000_000_000L,
                face = face(),
                eyes = eyes(1_000_000_000L),
            ),
        )

        val jumpedEyes = EyeTrackingFrame(
            timestampNanos = 1_016_666_667L,
            trackingId = 4,
            left = EyeObservation(
                eye = EyeSide.LEFT,
                center = Vec2(150.0, 165.0),
                irisCenter = Vec2(180.0, 190.0),
                normalizedOffset = Vec2(0.80, 0.80),
                openness = 0.92,
                blink = false,
                confidence = 0.24,
                gazeDirection = Vec3(-0.95, 0.05, 0.30),
            ),
            right = EyeObservation(
                eye = EyeSide.RIGHT,
                center = Vec2(250.0, 165.0),
                irisCenter = Vec2(220.0, 190.0),
                normalizedOffset = Vec2(-0.80, 0.80),
                openness = 0.90,
                blink = false,
                confidence = 0.20,
                gazeDirection = Vec3(-0.90, 0.10, 0.35),
            ),
            combinedGaze = Vec3(-0.92, 0.08, 0.32),
            fixationConfidence = 0.18,
        )

        val second = engine.fuse(
            FaceEyeFusionEngine.Input(
                timestampNanos = 1_016_666_667L,
                face = face().copy(
                    timestampNanos = 1_016_666_667L,
                ),
                eyes = jumpedEyes,
            ),
        )

        assertTrue(
            first.state.combinedGazeCamera.dot(
                second.state.combinedGazeCamera,
            ) > 0.70,
        )
        assertTrue(
            second.rejectionReasons.any {
                it.contains("innovation")
            },
        )
    }

    @Test
    fun uncertaintyDecompositionSeparatesStableAndDegradedEvidence() {
        val engine = FaceEyeFusionEngine()

        val stable = engine.fuse(
            FaceEyeFusionEngine.Input(
                timestampNanos = 2_000_000_000L,
                face = face().copy(
                    timestampNanos = 2_000_000_000L,
                    captureTimingConfidence = 0.98,
                ),
                eyes = eyes(2_000_000_000L),
            ),
        )

        val degraded = engine.fuse(
            FaceEyeFusionEngine.Input(
                timestampNanos = 2_016_666_667L,
                face = face().copy(
                    timestampNanos = 2_016_666_667L,
                    captureTimingConfidence = 0.35,
                ),
                eyes = EyeTrackingFrame(
                    timestampNanos = 2_016_666_667L,
                    trackingId = 4,
                    left = EyeObservation(
                        eye = EyeSide.LEFT,
                        center = Vec2(150.0, 165.0),
                        irisCenter = Vec2(175.0, 190.0),
                        normalizedOffset = Vec2(0.70, 0.70),
                        openness = 0.40,
                        blink = true,
                        confidence = 0.18,
                        gazeDirection = Vec3(-0.82, 0.16, 0.55),
                    ),
                    right = EyeObservation(
                        eye = EyeSide.RIGHT,
                        center = Vec2(250.0, 165.0),
                        irisCenter = Vec2(225.0, 190.0),
                        normalizedOffset = Vec2(-0.70, 0.70),
                        openness = 0.42,
                        blink = true,
                        confidence = 0.16,
                        gazeDirection = Vec3(0.78, 0.20, 0.56),
                    ),
                    combinedGaze = Vec3(-0.70, 0.10, 0.60),
                    fixationConfidence = 0.16,
                ),
            ),
        )

        assertTrue(stable.uncertainty.total in 0.0..1.0)
        assertTrue(degraded.uncertainty.total in 0.0..1.0)
        assertTrue(
            degraded.uncertainty.total >=
                stable.uncertainty.total,
        )
        assertTrue(
            degraded.uncertainty.outOfDistributionRisk >=
                stable.uncertainty.outOfDistributionRisk,
        )
        assertTrue(
            degraded.uncertainty.modelMismatch >= 0.0,
        )
    }


    @Test
    fun uncertaintyThresholdReducesAuthoritativeConfidence() {
        val engine =
            FaceEyeFusionEngine(
                FaceEyeFusionEngine.Config(
                    hardUncertainty = 0.30,
                ),
            )

        val result =
            engine.fuse(
                FaceEyeFusionEngine.Input(
                    timestampNanos = 3_000_000_000L,
                    face =
                        face().copy(
                            timestampNanos = 3_000_000_000L,
                            captureTimingConfidence = 0.25,
                        ),
                    eyes =
                        EyeTrackingFrame(
                            timestampNanos = 3_000_000_000L,
                            trackingId = 4,
                            left = null,
                            right = null,
                            combinedGaze = Vec3(0.0, 0.0, 1.0),
                            fixationConfidence = 0.0,
                        ),
                ),
            )

        assertTrue(
            !result.uncertainty.trustworthy,
        )
        assertTrue(
            result.rejectionReasons.any {
                it.contains("uncertainty")
            },
        )
        assertTrue(
            result.state.confidence < 0.80,
        )
    }

    @Test
    fun advancedCustomizationChangesFusionConfiguration() {
        val engine = FaceEyeFusionEngine()

        val baseline = engine.configuration()

        engine.configureCustomization(
            headPoseStrictness = 1.0,
            gazePriority = 1.0,
            eyeTemporalStrictness = 1.0,
            semanticMotionSensitivity = 1.0,
        )

        val tuned = engine.configuration()

        assertTrue(
            tuned.minimumFaceConfidence >
                baseline.minimumFaceConfidence,
        )
        assertTrue(
            tuned.minimumEyeConfidence >
                baseline.minimumEyeConfidence,
        )
        assertTrue(
            tuned.binocularAgreementWeight >
                baseline.binocularAgreementWeight,
        )
        assertTrue(
            tuned.temporalSmoothing >
                baseline.temporalSmoothing,
        )
    }

    @Test
    fun rapid_gaze_reversal_reports_bounded_angular_acceleration() {
        val engine = FaceEyeFusionEngine()

        engine.fuse(
            input = fixtureInput(
                timestampNanos = 6_000_000_000L,
                gazeX = 0.0,
                gazeY = 0.0,
            ),
        )
        engine.fuse(
            input = fixtureInput(
                timestampNanos = 6_016_666_667L,
                gazeX = 0.20,
                gazeY = 0.0,
            ),
        )

        val result =
            engine.fuse(
                input = fixtureInput(
                    timestampNanos = 6_033_333_334L,
                    gazeX = -0.16,
                    gazeY = 0.02,
                ),
            )

        val acceleration =
            result.diagnostics
                .first {
                    it.startsWith(
                        "gazeAngularAccelerationDps2=",
                    )
                }
                .substringAfter("=")
                .toDouble()

        assertTrue(acceleration.isFinite())
        assertTrue(
            kotlin.math.abs(acceleration) <=
                18_000.0 + 1e-6,
        )
    }

    @Test
    fun staleEyeEvidenceDoesNotBecomeFresh() {
        val engine = FaceEyeFusionEngine()
        val result = engine.fuse(
            FaceEyeFusionEngine.Input(
                timestampNanos = 1_500_000_000L,
                face = face(),
                eyes = eyes(1_000_000_000L),
            ),
        )

        assertTrue(!result.eyeFresh)
        assertTrue(result.rejectionReasons.isNotEmpty())
    }

    private fun face() = FaceFrame(
        timestampNanos = 1_000_000_000L,
        trackingId = 4,
        boundsLeft = 100f,
        boundsTop = 80f,
        boundsRight = 300f,
        boundsBottom = 320f,
        yawDegrees = 4f,
        rollDegrees = -2f,
        points = listOf(
            FaceMeshPoint(0, Vec2(140.0, 160.0), 0.95),
            FaceMeshPoint(1, Vec2(260.0, 160.0), 0.95),
        ),
        leftEyeCenter = Vec2(150.0, 165.0),
        rightEyeCenter = Vec2(250.0, 165.0),
        leftEyeOpenProbability = 0.96,
        rightEyeOpenProbability = 0.94,
    )

    private fun eyes(timestamp: Long) = EyeTrackingFrame(
        timestampNanos = timestamp,
        trackingId = 4,
        left = EyeObservation(
            eye = EyeSide.LEFT,
            center = Vec2(150.0, 165.0),
            irisCenter = Vec2(151.0, 164.0),
            normalizedOffset = Vec2(0.04, 0.01),
            openness = 0.94,
            blink = false,
            confidence = 0.88,
            gazeDirection = Vec3(0.03, 0.00, 0.999),
        ),
        right = EyeObservation(
            eye = EyeSide.RIGHT,
            center = Vec2(250.0, 165.0),
            irisCenter = Vec2(249.0, 164.0),
            normalizedOffset = Vec2(-0.03, 0.01),
            openness = 0.93,
            blink = false,
            confidence = 0.90,
            gazeDirection = Vec3(0.02, 0.00, 0.999),
        ),
        combinedGaze = Vec3(0.025, 0.0, 0.999),
        fixationConfidence = 0.89,
    )
}
