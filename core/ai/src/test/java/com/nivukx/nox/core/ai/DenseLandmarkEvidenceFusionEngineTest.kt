package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import com.nivukx.nox.core.model.Vec3

class DenseLandmarkEvidenceFusionEngineTest {

    private fun observation(
        cameraId: Int,
        timestampNanos: Long,
        x: Double,
        y: Double,
        confidence: Double = 0.95,
        uncertainty: Double = 0.05,
        visibility: Double = 1.0,
        contactProbability: Double = 0.0,
        residual: Double = 0.002,
        motionBoundaryConfidence: Double = 1.0,
        appearanceConsistency: Double = 1.0,
        depthConfidence: Double = 1.0,
        viewSelectionScore: Double = 1.0,
        segmentationConsistency: Double = 0.0,
        segmentationEvidenceAvailable: Boolean = false,
        calibrationConfidence: Double = 0.95,
        calibrationEvidenceAvailable: Boolean = true,
    ): DenseLandmarkEvidenceFusionEngine.CameraObservation =
        DenseLandmarkEvidenceFusionEngine.CameraObservation(
            cameraId = cameraId,
            timestampNanos = timestampNanos,
            normalizedX = x,
            normalizedY = y,
            confidence = confidence,
            uncertainty = uncertainty,
            visibility = visibility,
            contactProbability = contactProbability,
            cameraQuality = 0.95,
            reprojectionResidualNormalized = residual,
            depthMeters = 2.0,
            motionBoundaryConfidence = motionBoundaryConfidence,
            appearanceConsistency = appearanceConsistency,
            depthConfidence = depthConfidence,
            viewSelectionScore = viewSelectionScore,
            segmentationConsistency = segmentationConsistency,
            segmentationEvidenceAvailable = segmentationEvidenceAvailable,
            calibrationConfidence = calibrationConfidence,
            calibrationEvidenceAvailable = calibrationEvidenceAvailable,
        )

    private fun frame(
        timestampNanos: Long,
        vararg observations:
            DenseLandmarkEvidenceFusionEngine.CameraObservation,
    ) =
        DenseLandmarkEvidenceFusionEngine.FrameInput(
            timestampNanos = timestampNanos,
            observations = observations.map {
                DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                    subjectId = 0,
                    landmarkId = 7,
                ) to it
            },
        )

    @Test
    fun multiViewFusionRejectsLargeOutlier() {
        val engine = DenseLandmarkEvidenceFusionEngine()

        val result = engine.process(
            frame(
                1_000_000_000L,
                observation(0, 1_000_000_000L, 0.40, 0.50),
                observation(1, 1_000_000_000L, 0.401, 0.501),
                observation(2, 1_000_000_000L, 0.399, 0.499),
                observation(
                    3,
                    1_000_000_000L,
                    1.20,
                    1.20,
                    confidence = 0.20,
                ),
            ),
        )

        val landmark = result.landmarks.single()
        assertTrue(
            kotlin.math.abs(landmark.normalizedX - 0.40) < 0.02,
        )
        assertTrue(
            landmark.rejectedViewCount >= 1 ||
                landmark.acceptedViewCount < 4,
        )
        assertTrue(landmark.crossViewAgreement > 0.40)
        assertTrue(result.crossViewEvidenceAvailable)
        assertEquals(3, result.supportingCameraCount)
        engine.close()
    }

    @Test
    fun temporalInnovationAndPhysicalEvidenceContributeToFusion() {
        val engine = DenseLandmarkEvidenceFusionEngine()
        val key = DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 9)

        engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_000_000_000L,
                observations = listOf(
                    key to observation(
                        0, 1_000_000_000L, 0.20, 0.30,
                    ),
                    key to observation(
                        1, 1_000_000_000L, 0.201, 0.301,
                    ),
                ),
            ),
        )

        val result = engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_016_000_000L,
                observations = listOf(
                    key to observation(
                        0, 1_016_000_000L, 0.208, 0.308,
                        motionBoundaryConfidence = 0.95,
                        appearanceConsistency = 0.93,
                        depthConfidence = 0.96,
                        viewSelectionScore = 0.98,
                    ),
                    key to observation(
                        1, 1_016_000_000L, 0.209, 0.309,
                        motionBoundaryConfidence = 0.92,
                        appearanceConsistency = 0.91,
                        depthConfidence = 0.94,
                        viewSelectionScore = 0.96,
                    ),
                ),
            ),
        )

        val landmark = result.landmarks.single()
        assertTrue(landmark.temporalInnovationScore > 0.60)
        assertTrue(landmark.motionBoundaryConfidence > 0.90)
        assertTrue(landmark.appearanceConsistency > 0.90)
        assertTrue(landmark.depthConfidence > 0.90)
        assertTrue(result.geometricConfidence > 0.45)
        assertTrue(result.viewDiversity > 0.25)
        assertTrue(result.temporalInnovationP95 < 0.60)
        engine.close()
    }

    @Test
    fun singleViewEvidenceDoesNotPretendCrossViewFailure() {
        val engine = DenseLandmarkEvidenceFusionEngine()

        val result = engine.process(
            frame(
                1_000_000_000L,
                observation(
                    0,
                    1_000_000_000L,
                    0.4,
                    0.5,
                ),
            ),
        )

        assertTrue(!result.crossViewEvidenceAvailable)
        assertTrue(result.crossViewAgreement == 0.0)
        assertTrue(
            result.warnings.none {
                it.contains("cross-view")
            },
        )
        engine.close()
    }

    @Test
    fun nonFiniteEvidenceFailsClosed() {
        val engine = DenseLandmarkEvidenceFusionEngine()

        val result = engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_000_000_000L,
                observations = listOf(
                    DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 1) to
                        observation(
                            cameraId = 0,
                            timestampNanos = 1_000_000_000L,
                            x = Double.NaN,
                            y = 0.5,
                        ),
                ),
            ),
        )

        assertEquals(0.0, result.meanConfidence)
        assertTrue(result.hardFailure)
        assertTrue(result.nonFiniteObservations == 1L)
        engine.close()
    }

    @Test
    fun depth_temporal_prediction_preserves_forward_motion_within_bounds() {
        val engine =
            DenseLandmarkEvidenceFusionEngine(
                DenseLandmarkEvidenceFusionEngine.Config(
                    occlusionPredictionHorizonMillis = 80.0,
                    maximumPredictionDepthSpeedMetersPerSecond = 12.0,
                    maximumPredictionDepthAccelerationMetersPerSecond2 = 60.0,
                ),
            )
        val key =
            DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                subjectId = 0,
                landmarkId = 51,
            )

        fun observation(
            timestamp: Long,
            depth: Double,
        ) =
            DenseLandmarkEvidenceFusionEngine.CameraObservation(
                cameraId = 0,
                timestampNanos = timestamp,
                normalizedX = 0.50,
                normalizedY = 0.50,
                depthMeters = depth,
                confidence = 0.98,
                uncertainty = 0.01,
                visibility = 1.0,
                cameraQuality = 0.99,
                reprojectionResidualNormalized = 0.001,
                depthResidualMeters = 0.001,
            )

        engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_000_000_000L,
                observations =
                    listOf(
                        key to
                            observation(
                                1_000_000_000L,
                                2.00,
                            ),
                    ),
            ),
        )
        engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_016_666_667L,
                observations =
                    listOf(
                        key to
                            observation(
                                1_016_666_667L,
                                1.96,
                            ),
                    ),
            ),
        )

        val predicted =
            engine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = 1_033_333_334L,
                    observations = emptyList(),
                ),
            )
        val landmark = predicted.landmarks.single()

        assertTrue(landmark.depthMeters != null)
        assertTrue(landmark.depthMeters!! in 1.80..1.98)
        assertTrue(
            landmark.state ==
                DenseLandmarkEvidenceFusionEngine.EvidenceState.TEMPORALLY_PREDICTED,
        )
        engine.close()
    }

    @Test
    fun shortOcclusionUsesBoundedPredictionAndLongGapWithdrawsIt() {
        val engine = DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                occlusionPredictionHorizonMillis = 80.0,
            ),
        )

        engine.process(
            frame(
                1_000_000_000L,
                observation(0, 1_000_000_000L, 0.30, 0.50),
                observation(1, 1_000_000_000L, 0.301, 0.501),
            ),
        )

        val predicted = engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_040_000_000L,
                observations = emptyList(),
            ),
        )

        assertTrue(
            predicted.landmarks.any {
                it.state ==
                    DenseLandmarkEvidenceFusionEngine.EvidenceState
                        .TEMPORALLY_PREDICTED
            },
        )

        val withdrawn = engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_200_000_000L,
                observations = emptyList(),
            ),
        )

        assertTrue(
            withdrawn.landmarks.none {
                it.state ==
                    DenseLandmarkEvidenceFusionEngine.EvidenceState
                        .TEMPORALLY_PREDICTED
            },
        )
        engine.close()
    }

    @Test
    fun contactUsesHysteresisInsteadOfSingleFrameFlapping() {
        val engine = DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                contactHysteresisFrames = 2,
            ),
        )

        val key = DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 8)

        fun processFrame(
            timestamp: Long,
            probability: Double,
        ) =
            engine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = timestamp,
                    observations = listOf(
                        key to observation(
                            cameraId = 0,
                            timestampNanos = timestamp,
                            x = 0.5,
                            y = 0.5,
                            contactProbability = probability,
                        ),
                    ),
                ),
            )

        val approaching = processFrame(1_000_000_000L, 0.90)
        assertTrue(
            approaching.landmarks.single().contactState ==
                DenseLandmarkEvidenceFusionEngine.ContactState.APPROACHING,
        )

        val contact = processFrame(1_016_000_000L, 0.92)
        assertTrue(
            contact.landmarks.single().contactState ==
                DenseLandmarkEvidenceFusionEngine.ContactState.CONTACT,
        )

        val stillContact = processFrame(1_032_000_000L, 0.50)
        assertTrue(
            stillContact.landmarks.single().contactState ==
                DenseLandmarkEvidenceFusionEngine.ContactState.CONTACT,
        )
        engine.close()
    }

    @Test
    fun staleEvidenceIsRejectedAndStateMemoryRemainsBounded() {
        val engine = DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                maximumLandmarksPerSubject = 32,
                maximumStoredLandmarks = 32,
                staleObservationMillis = 50.0,
            ),
        )

        repeat(100) { index ->
            val timestamp = 1_000_000_000L +
                index * 20_000_000L
            val key =
                DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                    subjectId = 0,
                    landmarkId = index % 32,
                )
            val result = engine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = timestamp,
                    observations = listOf(
                        key to observation(
                            0,
                            timestamp,
                            0.2,
                            0.3,
                        ),
                    ),
                ),
            )
            assertTrue(result.landmarks.isNotEmpty())
        }

        val stale = engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 4_000_000_000L,
                observations = listOf(
                    DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 1) to
                        observation(
                            cameraId = 1,
                            timestampNanos = 1_000_000_000L,
                            x = 0.2,
                            y = 0.3,
                        ),
                ),
            ),
        )

        assertTrue(stale.staleObservations >= 1L)
        assertTrue(engine.telemetry().activeLandmarkCount <= 32)
        engine.close()
    }

    @Test
    fun cameraTimestampReversalIsHardRejected() {
        val engine = DenseLandmarkEvidenceFusionEngine()
        val first = engine.process(
            frame(
                1_000_000_000L,
                observation(0, 1_000_000_000L, 0.3, 0.4),
            ),
        )
        assertTrue(!first.hardFailure)

        val reversed = engine.process(
            frame(
                1_100_000_000L,
                observation(0, 900_000_000L, 0.31, 0.41),
            ),
        )

        assertTrue(reversed.hardFailure)
        assertTrue(
            engine.diagnostics().any {
                it.contains("camera timestamp reversal")
            },
        )
        engine.close()
    }

    @Test
    fun frameAdmissionAndPerLandmarkEvidenceAreBounded() {
        val engine = DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                maximumLandmarksPerSubject = 16,
                maximumObservationsPerFrame = 8,
                maximumObservationsPerLandmark = 2,
                maximumCameras = 2,
                maximumStoredLandmarks = 16,
            ),
        )

        val observations = (0 until 32).map { index ->
            DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                subjectId = 0,
                landmarkId = index % 4,
            ) to observation(
                cameraId = index % 2,
                timestampNanos = 1_000_000_000L,
                x = 0.4 + index * 0.001,
                y = 0.5,
            )
        }

        val result = engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_000_000_000L,
                observations = observations,
            ),
        )

        assertTrue(result.rejectedObservations > 0L)
        assertTrue(result.acceptedObservations <= 8L)
        assertTrue(engine.telemetry().activeLandmarkCount <= 16)
        engine.close()
    }

}


class DenseLandmarkEvidenceFusionEngineCalibratedRayTest {
    @Test
    fun convergentWorldRaysStrengthenCrossViewAgreement() {
        val engine = DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                calibratedRayAgreementWeight = 0.30,
                maximumSymmetricRayDistanceMeters = 0.08,
            ),
        )

        val key = DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 41)
        val observations = listOf(
            key to DenseLandmarkEvidenceFusionEngine.CameraObservation(
                cameraId = 0,
                timestampNanos = 1_000_000_000L,
                normalizedX = 0.5,
                normalizedY = 0.5,
                depthMeters = 2.0,
                confidence = 0.99,
                uncertainty = 0.02,
                visibility = 1.0,
                cameraQuality = 0.99,
                reprojectionResidualNormalized = 0.001,
                depthResidualMeters = 0.001,
                worldRayOriginMeters = Vec3(-0.25, 0.0, 0.0),
                worldRayDirection = Vec3(0.25, 0.0, 2.0),
            ),
            key to DenseLandmarkEvidenceFusionEngine.CameraObservation(
                cameraId = 1,
                timestampNanos = 1_000_000_000L,
                normalizedX = 0.5,
                normalizedY = 0.5,
                depthMeters = 2.0,
                confidence = 0.99,
                uncertainty = 0.02,
                visibility = 1.0,
                cameraQuality = 0.99,
                reprojectionResidualNormalized = 0.001,
                depthResidualMeters = 0.001,
                worldRayOriginMeters = Vec3(0.25, 0.0, 0.0),
                worldRayDirection = Vec3(-0.25, 0.0, 2.0),
            ),
        )

        val result = engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 1_000_000_000L,
                observations = observations,
            ),
        )
        val fused = result.landmarks.single()

        assertTrue(fused.crossViewAgreement > 0.75)
        assertTrue(fused.acceptedViewCount == 2)
        engine.close()
    }

    @Test
    fun contradictoryWorldRaysReduceCrossViewAgreementEvenWithStrongDetectors() {
        val engine = DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                calibratedRayAgreementWeight = 0.40,
                maximumSymmetricRayDistanceMeters = 0.01,
            ),
        )

        val key = DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 42)
        fun obs(cameraId: Int, origin: Vec3, direction: Vec3) =
            DenseLandmarkEvidenceFusionEngine.CameraObservation(
                cameraId = cameraId,
                timestampNanos = 2_000_000_000L,
                normalizedX = 0.5,
                normalizedY = 0.5,
                depthMeters = 2.0,
                confidence = 0.99,
                uncertainty = 0.01,
                visibility = 1.0,
                cameraQuality = 0.99,
                reprojectionResidualNormalized = 0.001,
                depthResidualMeters = 0.001,
                worldRayOriginMeters = origin,
                worldRayDirection = direction,
            )

        val result = engine.process(
            DenseLandmarkEvidenceFusionEngine.FrameInput(
                timestampNanos = 2_000_000_000L,
                observations = listOf(
                    key to obs(
                        0,
                        Vec3(-0.25, 0.0, 0.0),
                        Vec3(0.0, 0.0, 1.0),
                    ),
                    key to obs(
                        1,
                        Vec3(0.25, 1.0, 0.0),
                        Vec3(0.0, 1.0, 0.0),
                    ),
                ),
            ),
        )
        val fused = result.landmarks.single()

        assertTrue(fused.crossViewAgreement < 0.75)
        engine.close()
    }
}


class DenseLandmarkEvidenceFusionEngineCalibrationUncertaintyTest {
    @Test
    fun calibrationUncertaintyLimitsDenseAuthority() {
        val engine = DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                calibratedRayAgreementWeight = 0.25,
                calibrationTranslationSigmaScaleMeters = 0.01,
                calibrationRotationSigmaScaleRadians = 0.005,
                calibrationIntrinsicSigmaScale = 0.005,
            ),
        )

        val key = DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 91)

        fun observation(
            cameraId: Int,
            calibrationUncertainty: Double,
            calibrationConfidence: Double,
        ) =
            DenseLandmarkEvidenceFusionEngine.CameraObservation(
                cameraId = cameraId,
                timestampNanos = 3_000_000_000L,
                normalizedX = 0.50,
                normalizedY = 0.50,
                depthMeters = 2.0,
                confidence = 0.99,
                uncertainty = 0.01,
                visibility = 1.0,
                cameraQuality = 0.99,
                reprojectionResidualNormalized = 0.001,
                depthResidualMeters = 0.001,
                worldRayOriginMeters =
                    if (cameraId == 0) {
                        Vec3(-0.2, 0.0, 0.0)
                    } else {
                        Vec3(0.2, 0.0, 0.0)
                    },
                worldRayDirection =
                    if (cameraId == 0) {
                        Vec3(0.2, 0.0, 2.0)
                    } else {
                        Vec3(-0.2, 0.0, 2.0)
                    },
                calibrationUncertaintyMeters = calibrationUncertainty,
                calibrationConfidence = calibrationConfidence,
            )

        val precise =
            engine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = 3_000_000_000L,
                    observations = listOf(
                        key to observation(0, 0.001, 0.99),
                        key to observation(1, 0.001, 0.99),
                    ),
                ),
            ).landmarks.single()

        val degraded =
            engine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = 3_016_666_667L,
                    observations = listOf(
                        DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 92) to
                            observation(0, 0.08, 0.25),
                        DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 92) to
                            observation(1, 0.08, 0.25),
                    ),
                ),
            ).landmarks.single()

        assertTrue(precise.calibrationConfidence > degraded.calibrationConfidence)
        assertTrue(precise.confidence > degraded.confidence)
    }
}


class DenseLandmarkSegmentationEvidenceAvailabilityTest {
    @Test
    fun fusedLandmarkKeepsMeasuredSegmentationSeparateFromUnknownEvidence() {
        val engine = DenseLandmarkEvidenceFusionEngine()
        val measured =
            engine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = 8_000_000_000L,
                    observations =
                        listOf(
                            DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 3) to
                                observation(
                                    0,
                                    8_000_000_000L,
                                    0.40,
                                    0.50,
                                    segmentationConsistency = 0.32,
                                    segmentationEvidenceAvailable = true,
                                ),
                            DenseLandmarkEvidenceFusionEngine.LandmarkKey(0, 3) to
                                observation(
                                    1,
                                    8_000_000_000L,
                                    0.401,
                                    0.501,
                                    segmentationConsistency = 0.84,
                                    segmentationEvidenceAvailable = true,
                                ),
                        ),
                ),
            ).landmarks.single()

        assertTrue(measured.segmentationEvidenceAvailable)
        assertTrue(measured.segmentationConsistency in 0.32..0.84)

        val unknown =
            engine.process(
                frame(
                    8_016_000_000L,
                    observation(0, 8_016_000_000L, 0.40, 0.50),
                    observation(1, 8_016_000_000L, 0.401, 0.501),
                ),
            ).landmarks.single()

        assertTrue(!unknown.segmentationEvidenceAvailable)
        assertEquals(0.0, unknown.segmentationConsistency, 1e-6)
        engine.close()
    }
}


class DenseLandmarkCalibrationEvidenceAvailabilityTest {
    @Test
    fun unknownCalibrationDoesNotClaimPerfectMetricConfidenceOrErase2dTracking() {
        val engine = DenseLandmarkEvidenceFusionEngine()
        val result =
            engine.process(
                frame(
                    9_000_000_000L,
                    observation(
                        0,
                        9_000_000_000L,
                        0.40,
                        0.50,
                        calibrationEvidenceAvailable = false,
                    ),
                    observation(
                        1,
                        9_000_000_000L,
                        0.401,
                        0.501,
                        calibrationEvidenceAvailable = false,
                    ),
                ),
            ).landmarks.single()

        assertTrue(!result.calibrationEvidenceAvailable)
        assertEquals(0.0, result.calibrationConfidence, 1e-6)
        assertTrue(result.confidence > 0.0)
        assertTrue(result.diagnostics.any { it.contains("calibration confidence is unknown") })
        engine.close()
    }
}
