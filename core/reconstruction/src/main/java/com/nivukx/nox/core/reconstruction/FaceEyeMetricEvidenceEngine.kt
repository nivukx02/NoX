package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.EyeSide
import kotlin.math.max
import kotlin.math.min

/**
 * Metric face/eye evidence authority.
 *
 * Face tracking and gaze output are frequently treated as visually convincing even when geometric
 * evidence is weak. This engine keeps dense face coverage, binocular eye geometry, calibration,
 * temporal stability and uncertainty separate, then decides whether the result may be promoted to
 * metric downstream consumers.
 */
class FaceEyeMetricEvidenceEngine(
    private val config: Config = Config(),
) {
    enum class State {
        NO_DATA,
        VISUAL_ONLY,
        GEOMETRIC,
        METRIC_READY,
        HOLD,
    }

    data class Config(
        val minimumFaceConfidence: Double = 0.55,
        val minimumPointCoverage: Double = 0.72,
        val minimumTriangleCoverage: Double = 0.62,
        val minimumEyeConfidence: Double = 0.62,
        val minimumBinocularObservations: Int = 2,
        val maximumEyeAngularSpreadDegrees: Double = 12.0,
        val maximumEyeP95UncertaintyMm: Double = 8.0,
        val minimumIrisStructureConfidence: Double = 0.38,
        val minimumCalibrationConfidence: Double = 0.82,
        val minimumTemporalStability: Double = 0.78,
        val maximumCalibrationDrift: Double = 0.18,
        val metricThreshold: Double = 0.76,
        val geometricThreshold: Double = 0.54,
    )

    data class Input(
        val face: FaceMeshFrame?,
        val leftEye: Eye3DReconstructionEngine.Result?,
        val rightEye: Eye3DReconstructionEngine.Result?,
        val calibrationConfidence: Double,
        val temporalStability: Double,
        val gazeCalibrationConfidence: Double,
        val faceCalibrationDrift: Double,
        val headMotionConfidence: Double,
    )

    data class Result(
        val state: State,
        val confidence: Double,
        val faceConfidence: Double,
        val faceCoverage: Double,
        val eyeConfidence: Double,
        val binocularConfidence: Double,
        val gazeCalibrationConfidence: Double,
        val temporalStability: Double,
        val calibrationDrift: Double,
        val eyeP95UncertaintyMm: Double,
        val eyeAngularSpreadDegrees: Double,
        val irisStructureConfidence: Double = 0.0,
        val irisRadiusNormalized: Double = 0.0,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun evaluate(input: Input): Result {
        val face = input.face
        val faceConfidence = face?.confidence ?: 0.0
        val faceCoverage = face?.let {
            (
                it.pointCoverage * 0.55 +
                    it.triangleCoverage * 0.45
            ).coerceIn(0.0, 1.0)
        } ?: 0.0

        val eyeResults = listOfNotNull(
            input.leftEye,
            input.rightEye,
        )
        val eyeConfidence = if (eyeResults.isEmpty()) {
            0.0
        } else {
            eyeResults.map { it.confidence }.average()
        }

        val binocularAvailable =
            input.leftEye != null &&
                input.rightEye != null

        val binocularConsistency =
            if (binocularAvailable) {
                val spread = (
                    input.leftEye!!.angularSpreadDegrees +
                        input.rightEye!!.angularSpreadDegrees
                    ) / 2.0
                (
                    1.0 -
                        spread /
                            config.maximumEyeAngularSpreadDegrees
                    ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val eyeP95UncertaintyMm =
            eyeResults
                .map {
                    it.uncertainty95MajorMeters * 1000.0
                }
                .maxOrNull()
                ?: Double.POSITIVE_INFINITY

        val eyeMetricEvidenceAuthority =
            eyeResults
                .map {
                    it.metricEvidenceAuthority
                        .coerceIn(0.0, 1.0)
                }
                .takeIf { it.isNotEmpty() }
                ?.minOrNull()
                ?: 0.0
        val eyeEpipolarConsistency =
            eyeResults
                .map {
                    it.symmetricEpipolarConsistencyScore
                        .coerceIn(0.0, 1.0)
                }
        val irisStructureConfidence =
            eyeResults
                .map {
                    it.meanIrisStructureConfidence
                        .coerceIn(0.0, 1.0)
                }
                .takeIf { it.isNotEmpty() }
                ?.minOrNull()
                ?: 0.0
        val irisRadiusNormalized =
            eyeResults
                .map {
                    it.meanIrisRadiusNormalized
                        .coerceIn(0.0, 1.0)
                }
                .takeIf { it.isNotEmpty() }
                ?.average()
                ?: 0.0
                .takeIf { it.isNotEmpty() }
                ?.minOrNull()
                ?: 0.0

        val uncertaintyScore =
            inverse(
                eyeP95UncertaintyMm,
                config.maximumEyeP95UncertaintyMm * 0.40,
                config.maximumEyeP95UncertaintyMm,
            )

        val calibrationScore = (
            input.calibrationConfidence *
                0.62 +
                input.gazeCalibrationConfidence *
                0.38
            ).coerceIn(0.0, 1.0)

        val temporalScore = (
            input.temporalStability * 0.70 +
                input.headMotionConfidence * 0.30
            ).coerceIn(0.0, 1.0)

        val driftScore =
            inverse(
                input.faceCalibrationDrift,
                config.maximumCalibrationDrift * 0.45,
                config.maximumCalibrationDrift,
            )

        val coverageScore = (
            faceConfidence * 0.25 +
                faceCoverage * 0.25 +
                eyeConfidence * 0.25 +
                binocularConsistency * 0.15 +
                uncertaintyScore * 0.10
            ).coerceIn(0.0, 1.0)

        val metricEvidenceAuthority =
            (
                0.58 * eyeMetricEvidenceAuthority +
                    0.42 * uncertaintyScore
                ).coerceIn(0.0, 1.0)
        val epipolarConsistency =
            eyeEpipolarConsistency

        val confidence = (
            coverageScore * 0.28 +
                calibrationScore * 0.22 +
                temporalScore * 0.18 +
                driftScore * 0.10 +
                uncertaintyScore * 0.10 +
                metricEvidenceAuthority * 0.06 +
                epipolarConsistency * 0.06
            ).coerceIn(0.0, 1.0)

        val hardHold =
            face == null ||
                faceConfidence < config.minimumFaceConfidence * 0.60 ||
                faceCoverage < config.minimumPointCoverage * 0.55 ||
                input.calibrationConfidence < 0.40 ||
                input.faceCalibrationDrift >=
                    config.maximumCalibrationDrift * 1.8 ||
                eyeResults.any {
                    it.robustReprojectionP95Px > 8.0
                } ||
                eyeResults.any {
                    it.symmetricEpipolarConsistencyScore < 0.18
                } ||
                eyeResults.any {
                    it.metricEvidenceAuthority < 0.16
                }

        val metricReady =
            !hardHold &&
                faceConfidence >= config.minimumFaceConfidence &&
                faceCoverage >= config.minimumPointCoverage &&
                eyeConfidence >= config.minimumEyeConfidence &&
                binocularAvailable &&
                eyeResults.size >= config.minimumBinocularObservations &&
                eyeAngularSpread(eyeResults) <=
                    config.maximumEyeAngularSpreadDegrees &&
                eyeP95UncertaintyMm <=
                    config.maximumEyeP95UncertaintyMm &&
                calibrationScore >=
                    config.minimumCalibrationConfidence &&
                temporalScore >=
                    config.minimumTemporalStability &&
                driftScore >= 0.75 &&
                eyeMetricEvidenceAuthority >= 0.55 &&
                eyeEpipolarConsistency >= 0.55 &&
                confidence >= config.metricThreshold

        val geometricReady =
            !hardHold &&
                faceCoverage >= config.minimumPointCoverage * 0.90 &&
                eyeResults.isNotEmpty() &&
                calibrationScore >= 0.62 &&
                temporalScore >= 0.55 &&
                confidence >= config.geometricThreshold

        val state = when {
            hardHold -> State.HOLD
            metricReady -> State.METRIC_READY
            geometricReady -> State.GEOMETRIC
            face != null -> State.VISUAL_ONLY
            else -> State.NO_DATA
        }

        val warnings = buildList {
            if (face == null) add("FACE_MESH_UNAVAILABLE")
            if (faceConfidence < config.minimumFaceConfidence) {
                add("FACE_CONFIDENCE_BELOW_METRIC_TARGET")
            }
            if (faceCoverage < config.minimumPointCoverage) {
                add("FACE_SURFACE_COVERAGE_BELOW_METRIC_TARGET")
            }
            if (!binocularAvailable) {
                add("BINOCULAR_EYE_GEOMETRY_UNAVAILABLE")
            }
            if (eyeConfidence < config.minimumEyeConfidence) {
                add("EYE_CONFIDENCE_BELOW_METRIC_TARGET")
            }
            if (eyeP95UncertaintyMm > config.maximumEyeP95UncertaintyMm) {
                add("EYE_POSITION_UNCERTAINTY_TOO_HIGH")
            }
            if (input.calibrationConfidence < config.minimumCalibrationConfidence) {
                add("FACE_EYE_CALIBRATION_BELOW_METRIC_TARGET")
            }
            if (input.temporalStability < config.minimumTemporalStability) {
                add("FACE_EYE_TEMPORAL_STABILITY_BELOW_METRIC_TARGET")
            }
            if (input.faceCalibrationDrift > config.maximumCalibrationDrift) {
                add("FACE_EYE_CALIBRATION_DRIFT_TOO_HIGH")
            }
            if (eyeResults.any { it.viewDiversityScore < 0.65 }) {
                add("EYE_VIEW_DIVERSITY_REDUCED")
            }
            if (eyeMetricEvidenceAuthority < 0.55) {
                add("EYE_METRIC_EVIDENCE_AUTHORITY_LOW")
            }
            if (eyeEpipolarConsistency < 0.55) {
                add("EYE_EPIPOLAR_CONSISTENCY_LOW")
            }
            if (
                irisStructureConfidence <
                    config.minimumIrisStructureConfidence
            ) {
                add("IRIS_STRUCTURE_CONFIDENCE_LOW")
            }
        }.distinct()

        val recommendations = buildList {
            if (!binocularAvailable) {
                add("Acquire synchronized left and right eye geometry from independent views.")
            }
            if (eyeP95UncertaintyMm > config.maximumEyeP95UncertaintyMm) {
                add("Increase camera baseline and face-region pixel coverage before increasing eye-model complexity.")
            }
            if (faceCoverage < config.minimumPointCoverage) {
                add("Improve facial visibility and reduce occlusion around the eyes and nose bridge.")
            }
            if (input.calibrationConfidence < config.minimumCalibrationConfidence) {
                add("Refresh calibrated face-camera extrinsics before exporting metric gaze.")
            }
            if (input.faceCalibrationDrift > config.maximumCalibrationDrift) {
                add("Hold metric face/eye output until the face-camera rig is recalibrated.")
            }
            if (
                irisStructureConfidence <
                    config.minimumIrisStructureConfidence
            ) {
                add(
                    "Improve eye-region illumination, focus and pixel coverage before relying on iris-scale detail.",
                )
            }
        }.distinct()

        return Result(
            state = state,
            confidence = confidence,
            faceConfidence = faceConfidence,
            faceCoverage = faceCoverage,
            eyeConfidence = eyeConfidence,
            binocularConfidence = binocularConsistency,
            gazeCalibrationConfidence = input.gazeCalibrationConfidence,
            temporalStability = temporalScore,
            calibrationDrift = input.faceCalibrationDrift,
            eyeP95UncertaintyMm = eyeP95UncertaintyMm,
            eyeAngularSpreadDegrees = eyeAngularSpread(eyeResults),
            irisStructureConfidence =
                irisStructureConfidence,
            irisRadiusNormalized =
                irisRadiusNormalized,
            metricEvidenceAuthority = metricEvidenceAuthority,
            epipolarConsistency = epipolarConsistency,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun eyeAngularSpread(
        eyes: List<Eye3DReconstructionEngine.Result>,
    ): Double =
        if (eyes.isEmpty()) {
            Double.POSITIVE_INFINITY
        } else {
            eyes.map { it.angularSpreadDegrees }.average()
        }

    private fun inverse(
        value: Double,
        preferred: Double,
        hard: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= preferred) return 1.0
        if (value >= hard) return 0.0
        return (
            1.0 -
                (value - preferred) /
                    (hard - preferred).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }
}
