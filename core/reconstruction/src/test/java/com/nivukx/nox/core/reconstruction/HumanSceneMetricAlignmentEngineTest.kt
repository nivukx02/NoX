package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HumanSceneMetricAlignmentEngineTest {
    @Test
    fun scaleAndContactEvidenceProduceConsistentWorldFrame() {
        val engine = HumanSceneMetricAlignmentEngine(
            HumanSceneMetricAlignmentEngine.Config(minimumEvidence = 2),
        )

        val result = engine.align(
            HumanSceneMetricAlignmentEngine.Input(
                humanLandmarks = listOf(
                    HumanSceneMetricAlignmentEngine.HumanLandmark(
                        id = 1,
                        position = Vec3(0.0, 0.0, 0.0),
                        confidence = 0.98,
                        contactCandidate = true,
                    ),
                    HumanSceneMetricAlignmentEngine.HumanLandmark(
                        id = 2,
                        position = Vec3(0.0, 1.0, 0.0),
                        confidence = 0.97,
                    ),
                ),
                sceneAnchors = listOf(
                    HumanSceneMetricAlignmentEngine.SceneAnchor(
                        id = 10,
                        position = Vec3(0.20, 0.0, 0.0),
                        normal = Vec3(0.0, 1.0, 0.0),
                        confidence = 0.99,
                    ),
                    HumanSceneMetricAlignmentEngine.SceneAnchor(
                        id = 11,
                        position = Vec3(0.20, 2.0, 0.0),
                        normal = Vec3(0.0, 1.0, 0.0),
                        confidence = 0.99,
                    ),
                ),
                scaleEvidence = listOf(
                    HumanSceneMetricAlignmentEngine.ScaleEvidence(
                        segmentId = "height",
                        observedLengthMeters = 2.0,
                        referenceLengthMeters = 2.0,
                        confidence = 0.98,
                    ),
                    HumanSceneMetricAlignmentEngine.ScaleEvidence(
                        segmentId = "torso",
                        observedLengthMeters = 0.99,
                        referenceLengthMeters = 1.0,
                        confidence = 0.96,
                    ),
                ),
                gravityEvidence = listOf(
                    HumanSceneMetricAlignmentEngine.GravityEvidence(
                        direction = Vec3(0.0, 1.0, 0.0),
                        confidence = 0.99,
                    ),
                ),
                contacts = listOf(
                    HumanSceneMetricAlignmentEngine.ContactConstraint(
                        landmarkId = 1,
                        sceneAnchorId = 10,
                        confidence = 0.99,
                    ),
                ),
            ),
        )

        assertTrue(result.metricScale != null)
        assertTrue(result.scaleConfidence > 0.70)
        assertTrue(result.contactInlierRatio >= 0.5)
        assertTrue(result.worldConsistency > 0.70)
    }

    @Test
    fun missingScaleEvidenceCannotBecomeAuthoritative() {
        val engine = HumanSceneMetricAlignmentEngine()

        val result = engine.align(
            HumanSceneMetricAlignmentEngine.Input(
                humanLandmarks = listOf(
                    HumanSceneMetricAlignmentEngine.HumanLandmark(
                        id = 1,
                        position = Vec3(0.0, 0.0, 0.0),
                        confidence = 0.95,
                    ),
                ),
                sceneAnchors = listOf(
                    HumanSceneMetricAlignmentEngine.SceneAnchor(
                        id = 2,
                        position = Vec3(0.0, 0.0, 0.0),
                        normal = Vec3(0.0, 1.0, 0.0),
                        confidence = 0.95,
                    ),
                ),
                scaleEvidence = emptyList(),
            ),
        )

        assertTrue(!result.accepted)
        assertTrue(result.metricScale == null)
        assertTrue(
            result.warnings.any {
                it.contains("Metric scale evidence")
            },
        )
    }
}
