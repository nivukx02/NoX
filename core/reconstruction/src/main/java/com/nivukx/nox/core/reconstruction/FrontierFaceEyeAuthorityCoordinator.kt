package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.PrecisionAuthoritySnapshot
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3

/** Unified face + eye authority layer with explicit detector freshness gates. */
class FrontierFaceEyeAuthorityCoordinator(
    private val faceAuthority: FrontierFaceSurfaceAuthorityEngine = FrontierFaceSurfaceAuthorityEngine(),
    private val gazeEngine: WorldSpaceGazeReconstructionEngine = WorldSpaceGazeReconstructionEngine(),
    private val maximumFaceFreshnessNanos: Long = 120_000_000L,
    private val maximumEyeFreshnessNanos: Long = 80_000_000L,
    private val requireBinocularAuthority: Boolean = true,
    private val maximumBinocularDisagreementDegrees: Double = 12.0,
    private val maximumFaceEyeSkewNanos: Long = 45_000_000L,
) : AutoCloseable {

    data class Input(
        val timestampNanos: Long,
        val faceFrames: List<Pair<DeviceId, FaceMeshFrame>>,
        val calibration: CalibrationSnapshot,
        val headRotation: Quat,
        val eyeRays: List<WorldSpaceGazeReconstructionEngine.EyeRay>,
        val eyeTimestampNanos: Long,
    )

    data class Result(
        val face: FrontierFaceSurfaceAuthorityEngine.Result,
        val gaze: WorldSpaceGazeReconstructionEngine.Result,
        val combinedConfidence: Double,
        val authoritative: Boolean,
        val precisionAuthority: PrecisionAuthoritySnapshot,
        val warnings: List<String>,
    )

    init {
        require(maximumFaceFreshnessNanos > 0L)
        require(maximumEyeFreshnessNanos > 0L)
        require(maximumBinocularDisagreementDegrees > 0.0)
        require(maximumFaceEyeSkewNanos > 0L)
    }

    @Synchronized
    fun process(input: Input): Result {
        require(input.timestampNanos >= 0L)
        require(input.eyeTimestampNanos >= 0L)
        val freshFaceFrames = input.faceFrames.filter { (_, frame) ->
            frame.timestampNanos <= input.timestampNanos &&
                input.timestampNanos - frame.timestampNanos <= maximumFaceFreshnessNanos
        }
        freshFaceFrames.forEach { (deviceId, frame) -> faceAuthority.push(deviceId, frame) }
        val face = faceAuthority.reconstruct(input.timestampNanos, input.calibration)
        val eyeFresh =
            input.eyeTimestampNanos <= input.timestampNanos &&
                input.timestampNanos - input.eyeTimestampNanos <= maximumEyeFreshnessNanos
        val gaze = if (eyeFresh) {
            gazeEngine.reconstruct(input.headRotation, input.eyeRays)
        } else {
            WorldSpaceGazeReconstructionEngine.Result(
                directionWorld = Vec3(0.0,0.0,1.0),
                angularUncertaintyDegrees = 180.0,
                inlierRatio = 0.0,
                binocularAgreementDegrees = 180.0,
                valid = false,
                warnings = listOf("eye-evidence-stale"),
            )
        }
        val faceScore = face.metric?.confidence ?: 0.0
        val gazeScore = if (gaze.valid) {
            (
                gaze.confidence *
                    gaze.inlierRatio *
                    gaze.leftRightAgreementScore *
                    (1.0 - gaze.angularUncertaintyDegrees / 180.0)
                        .coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)
        } else 0.0
        val newestFaceTimestamp = freshFaceFrames.maxOfOrNull { it.second.timestampNanos }
        val faceEyeSkewSafe =
            newestFaceTimestamp == null ||
                kotlin.math.abs(newestFaceTimestamp - input.eyeTimestampNanos) <=
                    maximumFaceEyeSkewNanos
        val combined = (faceScore * 0.72 + gazeScore * 0.28).coerceIn(0.0,1.0)
        val leftSupport = input.eyeRays.any { !it.blink && it.eye == WorldSpaceGazeReconstructionEngine.Eye.LEFT && it.confidence >= 0.15 && it.eyelidOpenness > 0.10 }
        val rightSupport = input.eyeRays.any { !it.blink && it.eye == WorldSpaceGazeReconstructionEngine.Eye.RIGHT && it.confidence >= 0.15 && it.eyelidOpenness > 0.10 }
        val binocularAgreementSafe =
            gaze.binocularAgreementDegrees.isFinite() &&
                gaze.binocularAgreementDegrees <= maximumBinocularDisagreementDegrees
        val binocularReady =
            !requireBinocularAuthority ||
                (leftSupport && rightSupport && binocularAgreementSafe)
        val authoritative =
            face.authoritative &&
                gaze.valid &&
                binocularReady &&
                faceEyeSkewSafe &&
                combined >= 0.70
        return Result(
            face=face,
            gaze=gaze,
            combinedConfidence=combined,
            authoritative=authoritative,
            precisionAuthority=face.precisionAuthority,
            warnings = buildList {
                addAll(face.provenanceWarnings)
                addAll(gaze.warnings)
                if (
                    freshFaceFrames.isEmpty() &&
                        input.faceFrames.isNotEmpty()
                ) {
                    add("face-evidence-stale")
                }
                if (
                    requireBinocularAuthority &&
                        (!leftSupport || !rightSupport)
                ) {
                    add("binocular-eye-support-insufficient")
                }
                if (
                    requireBinocularAuthority &&
                        !binocularAgreementSafe &&
                        gaze.valid
                ) {
                    add("binocular-gaze-disagreement-high")
                }
                if (!faceEyeSkewSafe) {
                    add("face-eye-temporal-skew-high")
                }
            }.distinct(),
        )
    }

    override fun close() { faceAuthority.close() }
}