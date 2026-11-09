package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.HandTrackingFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TrackedObject
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Cross-modal motion and subject assessment.
 *
 * Pose, face, eye, hand and generic-object detectors provide complementary evidence. Treating them
 * as independent UI numbers leaves useful information on the table. This engine produces a single
 * semantic assessment used by downstream policy and diagnostics without pretending that a detector
 * result is ground truth.
 */
class AdvancedMotionFusionEngine {

    data class SubjectAssessment(
        val type: SubjectType,
        val trackingId: Long?,
        val confidence: Double,
        val spatialImportance: Double,
        val motionImportance: Double,
        val semanticMotionScore: Double = 0.0,
        val cameraMotionRisk: Double = 0.0,
    )

    data class Assessment(
        val timestampNanos: Long,
        val dominantSubject: SubjectAssessment?,
        val semanticCoverage: Double,
        val humanConfidence: Double,
        val animalConfidence: Double,
        val objectConfidence: Double,
        val faceCoverage: Double,
        val eyeCoverage: Double,
        val handCoverage: Double,
        val motionComplexity: Double,
        val occlusionRisk: Double,
        val attentionConfidence: Double,
        val semanticMotion: SemanticMotionEstimator.GlobalMotion,
        val subjectCount: Int = 0,
        val activeSubjects: Int = 0,
        val movingSubjects: Int = 0,
        val relativeMotionEnergy: Double = 0.0,
        val commonModeEnergy: Double = 0.0,
        val meanSpeedNormalizedPerSecond: Double = 0.0,
        val maxSpeedNormalizedPerSecond: Double = 0.0,
        val maxAccelerationNormalizedPerSecond2: Double = 0.0,
        val meanJerkNormalizedPerSecond3: Double = 0.0,
        val subjects: List<SubjectAssessment> = emptyList(),
    )

    private val semanticMotionEstimator = SemanticMotionEstimator()

    fun assess(
        timestampNanos: Long,
        faces: List<FaceFrame>,
        eyes: List<EyeTrackingFrame>,
        hands: HandTrackingFrame,
        objects: List<TrackedObject>,
        subjectTracks: List<SubjectTrackManager.Track> = emptyList(),
        motion: MotionCaptureEngine.Quality,
        frameWidth: Int = 1920,
        frameHeight: Int = 1080,
    ): Assessment {
        val semanticMotion =
            semanticMotionEstimator.update(
                tracks = subjectTracks,
                timestampNanos = timestampNanos,
                frameWidth = frameWidth,
                frameHeight = frameHeight,
            )

        val humanObject = objects
            .filter { it.subjectType == SubjectType.HUMAN }
            .maxByOrNull { it.confidence * boxArea(it.left, it.top, it.right, it.bottom) }

        val animalObject = objects
            .filter { it.subjectType == SubjectType.ANIMAL }
            .maxByOrNull { it.confidence * boxArea(it.left, it.top, it.right, it.bottom) }

        val genericObject = objects
            .filter { it.subjectType == SubjectType.OBJECT }
            .maxByOrNull { it.confidence * boxArea(it.left, it.top, it.right, it.bottom) }

        val bestFace = faces.maxByOrNull { face ->
            val area = (
                (face.boundsRight - face.boundsLeft).coerceAtLeast(0f) *
                    (face.boundsBottom - face.boundsTop).coerceAtLeast(0f)
                ).toDouble()
            facePointQuality(face) * sqrt(area.coerceAtLeast(1.0))
        }

        val faceConfidence = bestFace?.let(::facePointQuality) ?: 0.0
        val eyeConfidence = if (eyes.isEmpty()) {
            0.0
        } else {
            eyes
                .flatMap { frame ->
                    listOfNotNull(
                        frame.left?.confidence,
                        frame.right?.confidence,
                    )
                }
                .takeIf { it.isNotEmpty() }
                ?.average()
                ?: 0.0
        }
        val handConfidence =
            if (hands.staleEvidence) {
                (hands.meanConfidence * 0.18).coerceIn(0.0, 1.0)
            } else {
                hands.meanConfidence
            }
        val humanConfidence = max(
            faceConfidence,
            humanObject?.confidence ?: 0.0,
        )
        val animalConfidence = animalObject?.confidence ?: 0.0
        val objectConfidence = genericObject?.confidence ?: 0.0

        val candidates = buildList {
            humanObject?.let {
                add(
                    SubjectAssessment(
                        type = SubjectType.HUMAN,
                        trackingId = it.trackingId.takeIf { id -> id >= 0 }?.toLong(),
                        confidence = humanConfidence,
                        spatialImportance = boxArea(
                            it.left,
                            it.top,
                            it.right,
                            it.bottom,
                        ).coerceIn(0.0, 1.0),
                        motionImportance = (
                            0.50 * motion.motionScore +
                                0.50 * semanticMotionFor(
                                    it.trackingId,
                                    semanticMotion,
                                )
                            ).coerceIn(0.0, 1.0),
                        semanticMotionScore = semanticMotionFor(
                            it.trackingId,
                            semanticMotion,
                        ),
                        cameraMotionRisk = semanticMotion.cameraMotionRisk,
                    ),
                )
            }
            bestFace?.let {
                add(
                    SubjectAssessment(
                        type = SubjectType.HUMAN,
                        trackingId = it.trackingId?.toLong(),
                        confidence = humanConfidence,
                        spatialImportance = (
                            boxArea(
                                it.boundsLeft,
                                it.boundsTop,
                                it.boundsRight,
                                it.boundsBottom,
                            )
                        ).coerceIn(0.0, 1.0),
                        motionImportance = (
                            0.50 * motion.motionScore +
                                0.50 * semanticMotion.globalEnergy
                            ).coerceIn(0.0, 1.0),
                        semanticMotionScore = semanticMotion.globalEnergy,
                        cameraMotionRisk = semanticMotion.cameraMotionRisk,
                    ),
                )
            }
            animalObject?.let {
                add(
                    SubjectAssessment(
                        type = SubjectType.ANIMAL,
                        trackingId = it.trackingId.takeIf { id -> id >= 0 }?.toLong(),
                        confidence = animalConfidence,
                        spatialImportance = boxArea(
                            it.left,
                            it.top,
                            it.right,
                            it.bottom,
                        ).coerceIn(0.0, 1.0),
                        motionImportance = (
                            0.50 * motion.motionScore +
                                0.50 * semanticMotionFor(
                                    it.trackingId,
                                    semanticMotion,
                                )
                            ).coerceIn(0.0, 1.0),
                        semanticMotionScore = semanticMotionFor(
                            it.trackingId,
                            semanticMotion,
                        ),
                        cameraMotionRisk = semanticMotion.cameraMotionRisk,
                    ),
                )
            }
            genericObject?.let {
                add(
                    SubjectAssessment(
                        type = SubjectType.OBJECT,
                        trackingId = it.trackingId.takeIf { id -> id >= 0 }?.toLong(),
                        confidence = objectConfidence,
                        spatialImportance = boxArea(
                            it.left,
                            it.top,
                            it.right,
                            it.bottom,
                        ).coerceIn(0.0, 1.0),
                        motionImportance = (
                            0.50 * motion.motionScore +
                                0.50 * semanticMotionFor(
                                    it.trackingId,
                                    semanticMotion,
                                )
                            ).coerceIn(0.0, 1.0),
                        semanticMotionScore = semanticMotionFor(
                            it.trackingId,
                            semanticMotion,
                        ),
                        cameraMotionRisk = semanticMotion.cameraMotionRisk,
                    ),
                )
            }
        }

        val dominant = candidates.maxByOrNull {
            it.confidence * (
                0.55 +
                    0.25 * it.spatialImportance +
                    0.20 * it.motionImportance
                )
        }

        val semanticCoverage = listOf(
            if (faces.isNotEmpty()) 1.0 else 0.0,
            if (hands.trackedHands > 0) 1.0 else 0.0,
            if (objects.isNotEmpty()) 1.0 else 0.0,
        ).average()

        val motionComplexity = (
            0.30 * motion.motionScore +
                0.22 * semanticMotion.globalEnergy +
                0.12 * semanticMotion.meanAccelerationNormalizedPerSecond2
                    .div(18.0)
                    .coerceIn(0.0, 1.0) +
                0.16 * motion.jitterNormalized +
                0.10 * motion.occlusionRatio +
                0.10 * (1.0 - motion.temporalStability)
            ).coerceIn(0.0, 1.0)

        val occlusionRisk = (
            0.48 * motion.occlusionRatio +
                0.16 * (1.0 - motion.meanConfidence) +
                0.14 * motion.jitterNormalized +
                0.12 * (1.0 - semanticMotion.confidence) +
                0.10 * semanticMotion.cameraMotionRisk
            ).coerceIn(0.0, 1.0)

        val attentionConfidence = (
            0.30 * (dominant?.confidence ?: 0.0) +
                0.16 * semanticCoverage +
                0.16 * motion.temporalStability +
                0.14 * semanticMotion.temporalStability +
                0.14 * (1.0 - occlusionRisk) +
                0.10 * motion.dynamicsConfidence

            ).coerceIn(0.0, 1.0)

        return Assessment(
            timestampNanos = timestampNanos,
            dominantSubject = dominant,
            semanticCoverage = semanticCoverage,
            humanConfidence = humanConfidence,
            animalConfidence = animalConfidence,
            objectConfidence = objectConfidence,
            faceCoverage = if (faces.isEmpty()) 0.0 else faceConfidence,
            eyeCoverage = eyeConfidence,
            handCoverage = handConfidence,
            motionComplexity = motionComplexity,
            occlusionRisk = occlusionRisk,
            attentionConfidence = attentionConfidence,
            semanticMotion = semanticMotion,
        )
    }

    @Synchronized
    fun reset() {
        semanticMotionEstimator.clear()
    }

    private fun semanticMotionFor(
        trackingId: Int?,
        global: SemanticMotionEstimator.GlobalMotion,
    ): Double {
        if (trackingId == null || trackingId < 0) {
            return global.globalEnergy
        }

        return global.subjects
            .firstOrNull { it.trackId == trackingId.toLong() }
            ?.motionScore
            ?: global.globalEnergy
    }

    private fun facePointQuality(face: FaceFrame): Double {
        if (face.points.isEmpty()) return 0.0
        val pointConfidence = face.points.map { it.confidence }.average()
        val eyeConfidence = (
            face.leftEyeOpenProbability.coerceIn(0.0, 1.0) +
                face.rightEyeOpenProbability.coerceIn(0.0, 1.0)
            ) * 0.5
        return (
            0.78 * pointConfidence +
                0.22 * eyeConfidence
            ).coerceIn(0.0, 1.0)
    }

    private fun boxArea(
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
    ): Double {
        val area = (
            (right - left).coerceAtLeast(0f) *
                (bottom - top).coerceAtLeast(0f)
            ).toDouble()
        return area / 1_000_000.0
    }
}
