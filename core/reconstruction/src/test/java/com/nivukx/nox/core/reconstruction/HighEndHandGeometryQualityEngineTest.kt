package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Hand3DFrame
import com.nivukx.nox.core.model.Hand3DLandmark
import com.nivukx.nox.core.model.Hand3DResult
import com.nivukx.nox.core.model.HandSide
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndHandGeometryQualityEngineTest {

    @Test
    fun strict_submillimeter_requirement_is_visible_as_a_quality_warning_when_unqualified() {
        val hand =
            handResult(
                strictSubMillimeterQualified = false,
                confidence = 0.94,
                meanUncertaintyMeters = 0.0015,
                metricEvidenceAuthority = 0.90,
                epipolarScore = 0.90,
            )

        val result =
            HighEndHandGeometryQualityEngine()
                .evaluate(
                    Hand3DFrame(
                        timestampNanos = 1_000_000L,
                        hands = listOf(hand),
                        confidence = hand.confidence,
                    ),
                )

        assertTrue(
            result.hands.single().warnings.contains(
                "hand-submillimeter-metric-output-not-qualified",
            ),
        )
    }

    @Test
    fun strict_submillimeter_flag_is_preserved_when_metric_contract_is_satisfied() {
        val hand =
            handResult(
                strictSubMillimeterQualified = true,
                confidence = 0.96,
                meanUncertaintyMeters = 0.0005,
                metricEvidenceAuthority = 0.95,
                epipolarScore = 0.95,
            )

        val result =
            HighEndHandGeometryQualityEngine()
                .evaluate(
                    Hand3DFrame(
                        timestampNanos = 2_000_000L,
                        hands = listOf(hand),
                        confidence = hand.confidence,
                    ),
                )

        assertTrue(
            result.hands.single().strictSubMillimeterQualified,
        )
    }

    @Test
    fun coherentHandProducesHighGeometryConfidence() {
        val engine = HighEndHandGeometryQualityEngine()
        val frame = frame()

        engine.evaluate(frame)
        val result = engine.evaluate(
            frame.copy(timestampNanos = frame.timestampNanos + 16_666_667L),
        )

        assertTrue(result.handCount == 1)
        assertTrue(result.confidence > 0.70)
        assertTrue(result.hardFailureCount == 0)
        assertTrue(result.landmarkCoverage > 0.95)
        assertTrue(result.warnings.none { it.contains("hard-failure") })
    }

    @Test
    fun lowConfidenceFirstFrameCannotPoisonLongitudinalAnatomyBaseline() {
        val engine =
            HighEndHandGeometryQualityEngine(
                HighEndHandGeometryQualityEngine.Config(
                    longitudinalBaselineMinimumSamples = 3,
                    longitudinalBaselineUpdateToleranceRatio = 0.18,
                ),
            )

        engine.evaluate(
            scaledFrame(
                timestamp = 1_000_000_000L,
                scale = 1.34,
                confidence = 0.30,
            ),
        )

        repeat(4) { index ->
            engine.evaluate(
                frame().copy(
                    timestampNanos =
                        1_033_000_000L +
                            index * 33_000_000L,
                ),
            )
        }

        val result =
            engine.evaluate(
                frame().copy(
                    timestampNanos = 1_200_000_000L,
                    hands = listOf(
                        frame().hands.single().copy(
                            landmarks =
                                handLandmarks().map {
                                    it.copy(
                                        confidence = 0.96,
                                    )
                                },
                        ),
                    ),
                ),
            )

        assertTrue(
            result.hands.single()
                .longitudinalSegmentConsistency > 0.70,
        )
        assertTrue(
            result.warnings.none {
                it.contains("longitudinal-anatomy-hard")
            },
        )
    }

    @Test
    fun persistentHandScaleDriftIsSeparatedFromInstantSegmentConsistency() {
        val engine =
            HighEndHandGeometryQualityEngine(
                HighEndHandGeometryQualityEngine.Config(
                    longitudinalBaselineMinimumSamples = 3,
                    longitudinalBaselineUpdateToleranceRatio = 0.18,
                ),
            )

        repeat(5) { index ->
            engine.evaluate(
                frame().copy(
                    timestampNanos =
                        2_000_000_000L +
                            index * 33_000_000L,
                ),
            )
        }

        val drifted =
            engine.evaluate(
                scaledFrame(
                    timestamp = 2_200_000_000L,
                    scale = 1.30,
                    confidence = 0.96,
                ),
            )

        val hand = drifted.hands.single()

        assertTrue(
            hand.longitudinalSegmentConsistency < 0.60,
        )
        assertTrue(
            drifted.warnings.any {
                it.contains("longitudinal-anatomy")
            },
        )
    }

    @Test
    fun severeSegmentFailureIsFlagged() {
        val engine = HighEndHandGeometryQualityEngine()
        val good = frame()
        engine.evaluate(good)

        val badLandmarks = good.hands.single().landmarks.map { landmark ->
            if (landmark.index == 12) {
                landmark.copy(
                    position = landmark.position +
                        Vec3(0.45, 0.42, 0.30),
                )
            } else {
                landmark
            }
        }

        val bad = good.copy(
            timestampNanos = good.timestampNanos + 16_666_667L,
            hands = listOf(
                good.hands.single().copy(
                    landmarks = badLandmarks,
                ),
            ),
        )

        val result = engine.evaluate(bad)

        assertTrue(result.hardFailureCount >= 1)
        assertTrue(
            result.warnings.any {
                it.contains("hand-temporal") ||
                    it.contains("hand-segment")
            },
        )
        assertTrue(result.confidence < 0.75)
    }

    private fun handResult(
        strictSubMillimeterQualified: Boolean,
        confidence: Double,
        meanUncertaintyMeters: Double,
        metricEvidenceAuthority: Double,
        epipolarScore: Double,
    ): Hand3DResult =
        Hand3DResult(
            handId = 11L,
            side = HandSide.RIGHT,
            timestampNanos = 1_000_000L,
            landmarks = handLandmarks().map {
                it.copy(
                    confidence =
                        confidence.coerceAtLeast(0.90),
                    uncertaintyMeters =
                        meanUncertaintyMeters.coerceAtLeast(0.0001),
                )
            },
            confidence = confidence,
            meanUncertaintyMeters =
                meanUncertaintyMeters,
            reprojectionErrorPx = 0.50,
            metricEvidenceAuthority =
                metricEvidenceAuthority,
            symmetricEpipolarConsistencyScore =
                epipolarScore,
            strictSubMillimeterQualified =
                strictSubMillimeterQualified,
        )

    private fun scaledFrame(
        timestamp: Long,
        scale: Double,
        confidence: Double,
    ): Hand3DFrame =
        frame().copy(
            timestampNanos = timestamp,
            hands = listOf(
                frame().hands.single().copy(
                    timestampNanos = timestamp,
                    landmarks =
                        handLandmarks().map {
                            it.copy(
                                position =
                                    Vec3(
                                        it.position.x * scale,
                                        it.position.y * scale,
                                        it.position.z * scale,
                                    ),
                                confidence = confidence,
                            )
                        },
                    confidence = confidence,
                ),
            ),
            confidence = confidence,
        )

    private fun frame(): Hand3DFrame =
        Hand3DFrame(
            timestampNanos = 1_000_000_000L,
            hands = listOf(
                Hand3DResult(
                    handId = 7L,
                    side = HandSide.LEFT,
                    timestampNanos = 1_000_000_000L,
                    landmarks = handLandmarks(),
                    confidence = 0.95,
                    meanUncertaintyMeters = 0.012,
                    reprojectionErrorPx = 1.2,
                ),
            ),
            confidence = 0.95,
        )

    private fun handLandmarks(): List<Hand3DLandmark> =
        listOf(
            p(0, 0.00, 0.00, 0.00),
            p(1, 0.03, 0.02, 0.00),
            p(2, 0.07, 0.035, 0.00),
            p(3, 0.11, 0.040, 0.00),
            p(4, 0.14, 0.035, 0.00),
            p(5, 0.00, 0.030, 0.00),
            p(6, 0.00, 0.090, 0.00),
            p(7, 0.00, 0.150, 0.00),
            p(8, 0.00, 0.205, 0.00),
            p(9, 0.02, 0.030, 0.00),
            p(10, 0.02, 0.100, 0.00),
            p(11, 0.02, 0.170, 0.00),
            p(12, 0.02, 0.235, 0.00),
            p(13, 0.045, 0.026, 0.00),
            p(14, 0.050, 0.085, 0.00),
            p(15, 0.050, 0.145, 0.00),
            p(16, 0.050, 0.200, 0.00),
            p(17, 0.068, 0.018, 0.00),
            p(18, 0.078, 0.064, 0.00),
            p(19, 0.080, 0.112, 0.00),
            p(20, 0.081, 0.158, 0.00),
        )

    private fun p(
        index: Int,
        x: Double,
        y: Double,
        z: Double,
    ) = Hand3DLandmark(
        index = index,
        position = Vec3(x, y, z),
        confidence = 0.96,
        uncertaintyMeters = 0.008,
        observationCount = 4,
    )
}
