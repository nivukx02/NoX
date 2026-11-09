package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReconstructionQualityAnalyzerTest {

    @Test
    fun rejectsLowConfidenceHighUncertaintyGeometry() {
        val analyzer = ReconstructionQualityAnalyzer()
        val frame = SkeletonFrame(
            timestampNanos = 1_000_000L,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = List(8) { index ->
                ReconstructionJoint(
                    landmarkId = index,
                    position = Vec3(index.toDouble(), 0.0, 2.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.18,
                    reprojectionErrorPx = 2.0,
                    observationCount = 3,
                    positionUncertaintyMeters = 0.90,
                )
            },
            globalConfidence = 0.20,
        )

        val evaluation = analyzer.evaluate(
            frame = frame,
            camerasUsed = 3,
            clockSkewNanos = 5_000_000L,
        )

        assertFalse(evaluation.accepted)
        assertTrue(
            evaluation.reasons.contains("mean-depth-uncertainty") ||
                evaluation.reasons.contains("joint-depth-uncertainty"),
        )
    }

    @Test
    fun acceptsWellConditionedMetricGeometry() {
        val analyzer = ReconstructionQualityAnalyzer()
        val frame = SkeletonFrame(
            timestampNanos = 2_000_000L,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = List(12) { index ->
                ReconstructionJoint(
                    landmarkId = index,
                    position = Vec3(index * 0.05, 0.1, 2.0),
                    velocity = Vec3(0.01, 0.0, 0.0),
                    confidence = 0.88,
                    reprojectionErrorPx = 1.2,
                    observationCount = 5,
                    positionUncertaintyMeters = 0.035,
                )
            },
            globalConfidence = 0.91,
        )

        val evaluation = analyzer.evaluate(
            frame = frame,
            camerasUsed = 4,
            clockSkewNanos = 2_000_000L,
        )

        assertTrue(evaluation.accepted)
        assertTrue(evaluation.score >= 0.55)
    }
}
