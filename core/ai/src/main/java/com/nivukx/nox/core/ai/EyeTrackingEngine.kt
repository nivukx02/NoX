package com.nivukx.nox.core.ai

import android.media.Image
import com.nivukx.nox.core.model.EyeGazeCalibrationProfile
import com.nivukx.nox.core.model.ImageCoordinateTransform
import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import java.nio.ByteBuffer
import java.util.PriorityQueue
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Dedicated eye-motion engine.
 *
 * This tracker is deliberately split from face detection:
 *  - ML Kit supplies stable eye centers, head pose and eye-open probability.
 *  - The luminance plane is searched inside a tight eye ROI for the darkest coherent iris/pupil
 *    region.
 *  - Coordinates are mapped back from ML Kit's rotated frame into the Camera2 sensor plane.
 *  - A per-track temporal filter stabilizes iris position and gaze without smearing blinks.
 *
 * It is a monocular eye tracker, not a medical gaze estimator. The output is a camera-relative gaze
 * direction and confidence that can be triangulated across multiple NoX face cameras.
 */
class EyeTrackingEngine(
    private val gazeCalibrationProvider: () -> EyeGazeCalibrationProfile? = { null },
    private val minimumFaceDimensionPx: Float = 80f,
    private val minimumOpenProbability: Double = 0.12,
    private val blinkCloseThreshold: Double = 0.20,
    private val blinkOpenThreshold: Double = 0.42,
    private val maximumTrackAgeNanos: Long = 1_500_000_000L,
    private val maximumTracks: Int = 128,
) : AutoCloseable {
    init {
        require(minimumFaceDimensionPx.isFinite() && minimumFaceDimensionPx > 0f)
        require(minimumOpenProbability.isFinite() && minimumOpenProbability in 0.0..1.0)
        require(blinkCloseThreshold.isFinite() && blinkCloseThreshold in 0.0..1.0)
        require(blinkOpenThreshold.isFinite() && blinkOpenThreshold in 0.0..1.0)
        require(blinkOpenThreshold >= blinkCloseThreshold)
        require(maximumTrackAgeNanos > 0L)
        require(maximumTracks in 1..4_096)
    }

    private data class TrackState(
        var leftGaze: Vec3 = Vec3(0.0, 0.0, 1.0),
        var rightGaze: Vec3 = Vec3(0.0, 0.0, 1.0),
        var leftIris: Vec2? = null,
        var rightIris: Vec2? = null,
        var blinking: Boolean = false,
        var lastTimestampNanos: Long = 0L,
        var lastOutput: EyeTrackingFrame? = null,
    )

    private data class EyeSearchResult(
        val center: Vec2?,
        val normalizedOffset: Vec2,
        val confidence: Double,
        val irisRadiusNormalized: Double = 0.0,
        val irisStructureConfidence: Double = 0.0,
    )

    private val tracks = ConcurrentHashMap<Int, TrackState>()

    companion object {
        fun isStaleTimestamp(
            previousTimestampNanos: Long,
            currentTimestampNanos: Long,
        ): Boolean =
            previousTimestampNanos > 0L &&
                currentTimestampNanos > 0L &&
                currentTimestampNanos <= previousTimestampNanos
        /**
         * Bounded temporal evidence scale for eye tracking.
         *
         * Eye ROIs are small and therefore more sensitive to frame timing, readout skew, and
         * exposure-related motion blur than full-body landmarks. This factor never creates
         * confidence; it only reduces trust when measured capture timing is degraded.
         */
        fun temporalQualityScale(
            captureTimingConfidence: Double,
            rollingShutterSkewNanos: Long,
            frameDurationNanos: Long,
            exposureTimeNanos: Long,
        ): Double {
            require(captureTimingConfidence in 0.0..1.0)
            require(rollingShutterSkewNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)

            val frameDuration = maxOf(
                frameDurationNanos,
                1_000_000L,
            ).toDouble()
            val readoutRatio = (
                rollingShutterSkewNanos.toDouble() /
                    frameDuration
            ).coerceAtLeast(0.0)
            val exposureRatio = (
                exposureTimeNanos.toDouble() /
                    frameDuration
            ).coerceIn(0.0, 1.0)

            return (
                captureTimingConfidence *
                    exp(
                        -0.35 * readoutRatio -
                            0.20 * exposureRatio,
                    )
                ).coerceIn(0.05, 1.0)
        }
        data class SubpixelSample(
            val offsetX: Int,
            val offsetY: Int,
            val darknessScore: Double,
            val gradientScore: Double,
        )

        fun irisCandidateConfidence(
            compactness: Double,
            roiHalfWidth: Double,
            candidateQuality: Double,
        ): Double {
            require(compactness.isFinite() && compactness >= 0.0)
            require(roiHalfWidth.isFinite() && roiHalfWidth > 0.0)
            require(candidateQuality.isFinite() && candidateQuality in 0.0..1.0)

            return (
                (
                    1.0 -
                        sqrt(compactness) /
                            max(roiHalfWidth, 1.0)
                ).coerceIn(0.0, 1.0) *
                    candidateQuality
            ).coerceIn(0.0, 1.0)
        }

        fun subpixelCentroid(
            samples: List<SubpixelSample>,
        ): Vec2 {
            if (samples.isEmpty()) {
                return Vec2(0.0, 0.0)
            }

            var weightSum = 0.0
            var xSum = 0.0
            var ySum = 0.0

            samples.forEach { sample ->
                val weight =
                    (
                        sample.darknessScore.coerceIn(0.0, 1.0) * 0.72 +
                            sample.gradientScore.coerceIn(0.0, 1.0) * 0.28
                        ) *
                        exp(
                            -0.5 *
                                (
                                    sample.offsetX * sample.offsetX +
                                        sample.offsetY * sample.offsetY
                                    ).toDouble(),
                        )

                if (weight <= 0.0) return@forEach

                weightSum += weight
                xSum += sample.offsetX * weight
                ySum += sample.offsetY * weight
            }

            if (weightSum <= 1e-9) {
                return Vec2(0.0, 0.0)
            }

            return Vec2(
                x = (xSum / weightSum).coerceIn(-0.75, 0.75),
                y = (ySum / weightSum).coerceIn(-0.75, 0.75),
            )
        }

        fun irisStructureScore(
            darknessScore: Double,
            annulusContrast: Double,
            radialSymmetry: Double,
            gradientScore: Double,
            temporalPrior: Double,
        ): Double = (
            darknessScore.coerceIn(0.0, 1.0) * 0.18 +
                annulusContrast.coerceIn(0.0, 1.0) * 0.28 +
                radialSymmetry.coerceIn(0.0, 1.0) * 0.20 +
                gradientScore.coerceIn(0.0, 1.0) * 0.18 +
                temporalPrior.coerceIn(0.0, 1.0) * 0.16
            ).coerceIn(0.0, 1.0)

    }

    fun process(
        image: Image,
        rotationDegrees: Int,
        face: FaceFrame,
        captureTimingConfidence: Double = 1.0,
        rollingShutterSkewNanos: Long = 0L,
        frameDurationNanos: Long = 0L,
        exposureTimeNanos: Long = 0L,
    ): EyeTrackingFrame {
        if (!isUsableFaceEvidence(face)) return emptyFrame(face)
        val temporalQuality =
            try {
                temporalQualityScale(
                    captureTimingConfidence = captureTimingConfidence,
                    rollingShutterSkewNanos = rollingShutterSkewNanos,
                    frameDurationNanos = frameDurationNanos,
                    exposureTimeNanos = exposureTimeNanos,
                )
            } catch (_: IllegalArgumentException) {
                return emptyFrame(face)
            }

        val faceWidth = face.boundsRight - face.boundsLeft
        val faceHeight = face.boundsBottom - face.boundsTop
        if (max(faceWidth, faceHeight) < minimumFaceDimensionPx) return emptyFrame(face)

        val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
        val stateKey = face.trackingId ?: stableFaceKey(face)
        val existingState = tracks[stateKey]
        if (
            existingState != null &&
            isStaleTimestamp(
                previousTimestampNanos = existingState.lastTimestampNanos,
                currentTimestampNanos = face.timestampNanos,
            )
        ) {
            return existingState.lastOutput ?: emptyFrame(face)
        }

        val source =
            try {
                LumaPlane(image)
            } catch (_: IllegalArgumentException) {
                return emptyFrame(face)
            } catch (_: NoSuchElementException) {
                return emptyFrame(face)
            }

        pruneExpiredTracks(face.timestampNanos)
        if (!tracks.containsKey(stateKey)) {
            enforceTrackCapacity()
        }
        val left = face.leftEyeCenter?.let {
            findEye(
                source,
                it,
                face.rightEyeCenter,
                EyeSide.LEFT,
                normalizedRotation,
                face,
                existingState?.leftIris,
            )        }
        val right = face.rightEyeCenter?.let {
            findEye(
                source,
                it,
                face.leftEyeCenter,
                EyeSide.RIGHT,
                normalizedRotation,
                face,
                existingState?.rightIris,
            )        }

        val state = tracks.getOrPut(stateKey) { TrackState() }
        val negotiatedFrameSeconds =
            if (frameDurationNanos > 0L) {
                frameDurationNanos
                    .coerceIn(
                        1_000_000L,
                        1_000_000_000L,
                    )
                    .toDouble() / 1e9
            } else {
                1.0 / 30.0
            }
        val dt = if (state.lastTimestampNanos == 0L) {
            negotiatedFrameSeconds
        } else {
            (
                (
                    face.timestampNanos -
                        state.lastTimestampNanos
                    ).coerceAtLeast(1L) / 1e9
                ).coerceIn(1.0 / 240.0, 0.25)
        }

        val leftOpen = face.leftEyeOpenProbability
        val rightOpen = face.rightEyeOpenProbability
        val blinkCandidate = leftOpen < blinkCloseThreshold && rightOpen < blinkCloseThreshold
        val openedAgain = leftOpen > blinkOpenThreshold || rightOpen > blinkOpenThreshold

        state.blinking = when {
            state.blinking && openedAgain -> false
            !state.blinking && blinkCandidate -> true
            else -> state.blinking
        }

        val leftObservation = left?.let {
            buildObservation(
                image = image,
                side = EyeSide.LEFT,
                eyeCenter = face.leftEyeCenter,
                result = it,
                openness = leftOpen,
                blink = state.blinking,
                stateGaze = state.leftGaze,
                headYaw = face.yawDegrees.toDouble(),
                headRoll = face.rollDegrees.toDouble(),
                previousIris = state.leftIris,
                dt = dt,
                temporalQuality = temporalQuality,
            )
        }

        val rightObservation = right?.let {
            buildObservation(
                image = image,
                side = EyeSide.RIGHT,
                eyeCenter = face.rightEyeCenter,
                result = it,
                openness = rightOpen,
                blink = state.blinking,
                stateGaze = state.rightGaze,
                headYaw = face.yawDegrees.toDouble(),
                headRoll = face.rollDegrees.toDouble(),
                previousIris = state.rightIris,
                dt = dt,
                temporalQuality = temporalQuality,
            )
        }

        if (leftObservation != null) {
            state.leftGaze = leftObservation.gazeDirection
            state.leftIris = leftObservation.irisCenter
        }
        if (rightObservation != null) {
            state.rightGaze = rightObservation.gazeDirection
            state.rightIris = rightObservation.irisCenter
        }

        state.lastTimestampNanos = face.timestampNanos

        val valid = listOfNotNull(leftObservation, rightObservation)
        val combined = if (valid.isEmpty()) Vec3(0.0, 0.0, 1.0)
        else valid.map { it.gazeDirection }.reduce { a, b -> (a + b).normalized() }

        val confidence = if (valid.isEmpty()) {
            0.0
        } else {
            valid.map { it.confidence }.average()
        }

        val output =
            EyeTrackingFrame(
                timestampNanos = face.timestampNanos,
                trackingId = face.trackingId,
                left = leftObservation,
                right = rightObservation,
                combinedGaze = combined,
                fixationConfidence = confidence,
                captureTimingConfidence =
                    captureTimingConfidence.coerceIn(0.0, 1.0),
                rollingShutterSkewNanos =
                    rollingShutterSkewNanos.coerceAtLeast(0L),
                frameDurationNanos =
                    frameDurationNanos.coerceAtLeast(0L),
                exposureTimeNanos =
                    exposureTimeNanos.coerceAtLeast(0L),
            )
        state.lastOutput = output
        return output
    }

    private fun buildObservation(
        image: Image,
        side: EyeSide,
        eyeCenter: Vec2?,
        result: EyeSearchResult,
        openness: Double,
        blink: Boolean,
        stateGaze: Vec3,
        headYaw: Double,
        headRoll: Double,
        previousIris: Vec2?,
        dt: Double,
        temporalQuality: Double,
    ): EyeObservation {
        val measured = result.center
        val smoothed = if (measured == null || previousIris == null) {
            measured
        } else {
            val speed = measured.distanceTo(previousIris) / dt
            val baseAlpha =
                (
                    0.72 +
                        0.18 *
                            (speed / 800.0)
                                .coerceIn(0.0, 1.0)
                    ).coerceIn(0.72, 0.90)
            val timingAdjustedAlpha =
                baseAlpha *
                    (
                        0.65 +
                            0.35 * temporalQuality.coerceIn(0.0, 1.0)
                        ).coerceIn(0.65, 1.0)
            previousIris.lerp(measured, timingAdjustedAlpha)
        }

        val openWeight = if (openness >= minimumOpenProbability) openness else 0.0
        val blinkWeight = if (blink) 0.15 else 1.0
        val confidence = (
            result.confidence *
                openWeight.coerceIn(0.0, 1.0) *
                blinkWeight *
                temporalQuality.coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)

        val gaze = gazeFromOffset(
            result.normalizedOffset,
            headYaw,
            headRoll,
        )
        val stableGaze =
            if (blink) {
                // Preserve the last stable gaze vector while both eyes are closed. Replacing it
                // with low-confidence dark-ROI noise during a blink creates a visible gaze spike.
                stateGaze
            } else {
                stateGaze
                    .lerp(
                        gaze,
                        confidence.coerceIn(0.08, 0.9),
                    )
                    .normalized()
            }

        return EyeObservation(
            eye = side,
            center = result.center ?: eyeCenter ?: Vec2(0.0, 0.0),
            irisCenter = smoothed,
            normalizedOffset = result.normalizedOffset,
            openness = openness,
            blink = blink,
            confidence = confidence,
            gazeDirection = stableGaze,
            irisRadiusNormalized =
                result.irisRadiusNormalized.coerceIn(
                    0.0,
                    1.0,
                ),
            irisStructureConfidence =
                result.irisStructureConfidence.coerceIn(
                    0.0,
                    1.0,
                ),
            eyelidOcclusion =
                (
                    1.0 -
                        openness.coerceIn(0.0, 1.0)
                ).coerceIn(0.0, 1.0),
        )
    }

    private fun findEye(
        source: LumaPlane,
        detectorEyeCenter: Vec2,
        oppositeDetectorEyeCenter: Vec2?,
        side: EyeSide,
        rotationDegrees: Int,
        face: FaceFrame,
        previousIris: Vec2?,
    ): EyeSearchResult {
        val mapped = mapDetectorPointToSensor(
            detectorEyeCenter,
            source.width,
            source.height,
            rotationDegrees,
        )
        val opposite = oppositeDetectorEyeCenter?.let {
            mapDetectorPointToSensor(it, source.width, source.height, rotationDegrees)
        }

        val interEyeDistance = opposite?.let {
            mapped.distanceTo(it)
        } ?: ((face.boundsRight - face.boundsLeft) * 0.24).toDouble()

        val halfWidth = (interEyeDistance * 0.16).coerceIn(6.0, 54.0)
        val halfHeight = (interEyeDistance * 0.095).coerceIn(4.0, 30.0)

        val axisSign = if (side == EyeSide.LEFT) 1.0 else -1.0
        val sampleMinX = (mapped.x - halfWidth).toInt()
        val sampleMaxX = (mapped.x + halfWidth).toInt()
        val sampleMinY = (mapped.y - halfHeight).toInt()
        val sampleMaxY = (mapped.y + halfHeight).toInt()

        val topCandidates = PriorityQueue<Pair<Vec2, Double>>(
            96,
            compareBy { it.second },
        )

        for (y in sampleMinY..sampleMaxY) {
            for (x in sampleMinX..sampleMaxX) {
                if (!source.valid(x, y)) continue

                val nx = (x - mapped.x) / halfWidth
                val ny = (y - mapped.y) / halfHeight
                if (nx * nx + ny * ny > 1.0) continue

                val luma = source.sample(x, y).toDouble()
                val falloff = exp(-0.5 * (nx * nx + ny * ny))
                val gradient = source.gradientStrength(x, y)
                val darknessScore =
                    ((255.0 - luma) / 255.0)
                        .coerceIn(0.0, 1.0)
                val score = (
                    (
                        darknessScore * 0.72 +
                            gradient * 0.28
                        ) * 255.0
                    ) * falloff
                val candidate =
                    Vec2(x.toDouble(), y.toDouble()) to score

                if (topCandidates.size < 96) {
                    topCandidates.add(candidate)
                } else if (score > topCandidates.peek().second) {
                    topCandidates.poll()
                    topCandidates.add(candidate)
                }
            }
        }

        if (topCandidates.size < 20) {
            return EyeSearchResult(null, Vec2(0.0, 0.0), 0.0)
        }

        val candidates =
            topCandidates
                .toList()
                .map { (point, darkness) ->
                    val quality =
                        irisCandidateQuality(
                            source = source,
                            center = point,
                            halfWidth = halfWidth,
                            halfHeight = halfHeight,
                            darkness = darkness,
                            previousIris = previousIris,
                        )
                    point to quality
                }
                .sortedByDescending { it.second }
                .take(80)
        val offset = Vec2(
            x = ((cx - mapped.x) / halfWidth * axisSign).coerceIn(-1.0, 1.0),
            y = ((cy - mapped.y) / halfHeight).coerceIn(-1.0, 1.0),
        )

        val compactness = candidates.map {
            val dx = it.first.x - cx
            val dy = it.first.y - cy
            dx * dx + dy * dy
        }.average()

        val bestCandidate = candidates.firstOrNull()
        val subpixelOffset =
            bestCandidate?.let { (point, _) ->
                val samples =
                    buildList {
                        for (dy in -1..1) {
                            for (dx in -1..1) {
                                val x = point.x.toInt() + dx
                                val y = point.y.toInt() + dy
                                if (!source.valid(x, y)) continue
                                val luma =
                                    source.sample(
                                        x,
                                        y,
                                    ).toDouble()
                                add(
                                    EyeTrackingEngine.SubpixelSample(
                                        offsetX = dx,
                                        offsetY = dy,
                                        darknessScore =
                                            (
                                                (255.0 - luma) /
                                                    255.0
                                            ).coerceIn(
                                                0.0,
                                                1.0,
                                            ),
                                        gradientScore =
                                            source.gradientStrength(
                                                x,
                                                y,
                                            ).coerceIn(
                                                0.0,
                                                1.0,
                                            ),
                                    ),
                                )
                            }
                        }
                    }
                EyeTrackingEngine.subpixelCentroid(samples)
            } ?: Vec2(0.0, 0.0)

        val refinedCenter =
            bestCandidate?.let { (point, _) ->
                Vec2(
                    x = point.x + subpixelOffset.x,
                    y = point.y + subpixelOffset.y,
                )
            }

        val refinedCompactness =
            candidates.map {
                val dx = it.first.x -
                    (refinedCenter?.x ?: cx)
                val dy = it.first.y -
                    (refinedCenter?.y ?: cy)
                dx * dx + dy * dy
            }.average()

        val confidence =
            EyeTrackingEngine.irisCandidateConfidence(
                compactness = refinedCompactness,
                roiHalfWidth = halfWidth,
                candidateQuality =
                    bestCandidate?.second ?: 0.0,
            )

        val center =
            refinedCenter ?: Vec2(cx, cy)
        val refinedOffset =
            Vec2(
                x =
                    (
                        (center.x - mapped.x) /
                            halfWidth *
                            axisSign
                    ).coerceIn(-1.0, 1.0),
                y =
                    (
                        (center.y - mapped.y) /
                            halfHeight
                    ).coerceIn(-1.0, 1.0),
            )

        val irisScale =
            bestCandidate?.let { (point, _) ->
                estimateIrisRadiusNormalized(
                    source = source,
                    center = point,
                    halfWidth = halfWidth,
                    halfHeight = halfHeight,
                )
            } ?: 0.0

        return EyeSearchResult(
            center = center,
            normalizedOffset = refinedOffset,
            confidence = confidence,
            irisRadiusNormalized = irisScale,
            irisStructureConfidence =
                (bestCandidate?.second ?: 0.0)
                    .coerceIn(0.0, 1.0),
        )
    }

    private fun estimateIrisRadiusNormalized(
        source: LumaPlane,
        center: Vec2,
        halfWidth: Double,
        halfHeight: Double,
    ): Double {
        val maxRadius =
            min(
                halfWidth,
                halfHeight,
            ) * 0.82

        if (maxRadius < 2.0) return 0.0

        val candidates =
            (1..7).map { index ->
                maxRadius *
                    (0.32 + 0.08 * index)
            }

        val scored =
            candidates.mapNotNull { radius ->
                val samples = ArrayList<Double>(16)
                for (index in 0 until 12) {
                    val theta =
                        (
                            2.0 *
                                Math.PI *
                                index
                        ) / 12.0
                    val x =
                        (
                            center.x +
                                cos(theta) * radius
                            ).toInt()
                    val y =
                        (
                            center.y +
                                sin(theta) * radius
                            ).toInt()

                    if (source.valid(x, y)) {
                        samples +=
                            source.sample(
                                x,
                                y,
                            ).toDouble()
                    }
                }

                if (samples.size < 8) {
                    return@mapNotNull null
                }

                val mean =
                    samples.average()

                val variance =
                    samples.sumOf { sample ->
                        val delta =
                            sample - mean
                        delta * delta
                    } / samples.size.toDouble()

                val symmetry =
                    (
                        1.0 -
                            sqrt(variance) /
                                64.0
                    ).coerceIn(
                        0.0,
                        1.0,
                    )

                val centerLuma =
                    source.sample(
                        center.x.toInt(),
                        center.y.toInt(),
                    ).toDouble()

                val contrast =
                    (
                        (mean - centerLuma) /
                            128.0
                    ).coerceIn(
                        0.0,
                        1.0,
                    )

                val score =
                    contrast * 0.68 +
                        symmetry * 0.32

                radius to score
            }

        val best =
            scored.maxByOrNull { it.second }
                ?: return 0.0

        return (
            best.first /
                min(
                    halfWidth,
                    halfHeight,
                ).coerceAtLeast(1.0)
            ).coerceIn(
                0.0,
                1.0,
            )
    }

    private fun irisCandidateQuality(
        source: LumaPlane,
        center: Vec2,
        halfWidth: Double,
        halfHeight: Double,
        darkness: Double,
        previousIris: Vec2?,
    ): Double {
        val centerLumaScore =
            (darkness / 255.0).coerceIn(0.0, 1.0)
        val localGradientScore =
            source.gradientStrength(
                center.x.toInt(),
                center.y.toInt(),
            )

        val ringRadius =
            min(halfWidth, halfHeight) * 0.52
        if (ringRadius < 1.5) {
            return centerLumaScore * 0.20
        }

        val samples = ArrayList<Double>(8)
        for (index in 0 until 8) {
            val theta =
                (2.0 * Math.PI * index) / 8.0
            val x =
                (center.x + cos(theta) * ringRadius).toInt()
            val y =
                (center.y + sin(theta) * ringRadius).toInt()
            if (source.valid(x, y)) {
                samples += source.sample(x, y).toDouble()
            }
        }

        if (samples.size < 5) {
            return centerLumaScore * 0.25
        }

        val ringMean = samples.average()
        val ringVariance =
            samples.sumOf {
                val d = it - ringMean
                d * d
            } / samples.size.toDouble()
        val ringStd = sqrt(ringVariance)

        val centerApproxLuma =
            (255.0 - darkness).coerceIn(0.0, 255.0)
        val annulusContrast =
            ((ringMean - centerApproxLuma) / 128.0)
                .coerceIn(0.0, 1.0)
        val radialSymmetry =
            (1.0 - ringStd / 64.0)
                .coerceIn(0.0, 1.0)

        val temporalPrior =
            previousIris?.let { previous ->
                val dx = center.x - previous.x
                val dy = center.y - previous.y
                val distance = sqrt(dx * dx + dy * dy)
                exp(
                    -distance /
                        max(
                            min(halfWidth, halfHeight) * 1.35,
                            2.0,
                        ),
                ).coerceIn(0.0, 1.0)
            } ?: 0.72

        return (
            centerLumaScore * 0.18 +
                annulusContrast * 0.28 +
                radialSymmetry * 0.20 +
                localGradientScore * 0.18 +
                temporalPrior * 0.16
        ).coerceIn(0.0, 1.0)
    }
    private fun stableFaceKey(
        face: FaceFrame,
    ): Int {
        val left = face.boundsLeft.toDouble()
        val top = face.boundsTop.toDouble()
        val right = face.boundsRight.toDouble()
        val bottom = face.boundsBottom.toDouble()
        val centerX = (left + right) * 0.5
        val centerY = (top + bottom) * 0.5
        val width = (right - left).coerceAtLeast(1.0)
        val height = (bottom - top).coerceAtLeast(1.0)

        fun quantize(value: Double): Int =
            (value / 16.0).toInt()

        var hash = 17
        hash = 31 * hash + quantize(left)
        hash = 31 * hash + quantize(top)
        hash = 31 * hash + quantize(right)
        hash = 31 * hash + quantize(bottom)
        hash = 31 * hash + quantize(centerX)
        hash = 31 * hash + quantize(centerY)
        hash = 31 * hash + quantize(width)
        hash = 31 * hash + quantize(height)
        return if (hash == -1) Int.MIN_VALUE + 17 else hash
    }

    private fun gazeFromOffset(
        offset: Vec2,
        headYawDegrees: Double,
        headRollDegrees: Double,
    ): Vec3 {
        val calibration = gazeCalibrationProvider()?.predict(offset)

        val relativeYaw = calibration?.x ?: (offset.x * 0.24)
        val relativePitch = calibration?.y ?: (offset.y * 0.16)

        val yaw = headYawDegrees * Math.PI / 180.0 + relativeYaw
        val roll = headRollDegrees * Math.PI / 180.0

        val x = cos(relativePitch) * sin(yaw)
        val y = sin(relativePitch)
        val z = cos(relativePitch) * cos(yaw)

        val rollX = x * cos(roll) - y * sin(roll)
        val rollY = x * sin(roll) + y * cos(roll)

        return Vec3(rollX, rollY, z).normalized()
    }

    private fun pruneExpiredTracks(nowTimestampNanos: Long) {
        tracks.entries.removeIf { (_, state) ->
            state.lastTimestampNanos > 0L &&
                nowTimestampNanos -
                    state.lastTimestampNanos >
                maximumTrackAgeNanos
        }
    }

    private fun enforceTrackCapacity() {
        if (tracks.size < maximumTracks) return

        val removable =
            tracks.entries
                .minByOrNull { (_, state) ->
                    state.lastTimestampNanos
                }
                ?.key

        if (removable != null) {
            tracks.remove(removable)
        }
    }

    private fun isUsableFaceEvidence(face: FaceFrame): Boolean {
        if (face.timestampNanos <= 0L) return false
        if (
            !face.boundsLeft.isFinite() ||
            !face.boundsTop.isFinite() ||
            !face.boundsRight.isFinite() ||
            !face.boundsBottom.isFinite() ||
            face.boundsRight <= face.boundsLeft ||
            face.boundsBottom <= face.boundsTop
        ) {
            return false
        }
        if (
            !face.leftEyeOpenProbability.isFinite() ||
            face.leftEyeOpenProbability !in 0.0..1.0 ||
            !face.rightEyeOpenProbability.isFinite() ||
            face.rightEyeOpenProbability !in 0.0..1.0
        ) {
            return false
        }
        if (!face.yawDegrees.isFinite() || !face.rollDegrees.isFinite()) return false
        val left = face.leftEyeCenter
        if (left != null && (!left.x.isFinite() || !left.y.isFinite())) return false
        val right = face.rightEyeCenter
        if (right != null && (!right.x.isFinite() || !right.y.isFinite())) return false
        return true
    }

    private fun emptyFrame(face: FaceFrame): EyeTrackingFrame =
        EyeTrackingFrame(
            timestampNanos = face.timestampNanos,
            trackingId = face.trackingId,
            left = null,
            right = null,
            combinedGaze = Vec3(0.0, 0.0, 1.0),
            fixationConfidence = 0.0,
        )


    private class LumaPlane(image: Image) {
        private val plane =
            image.planes.firstOrNull()
                ?: throw IllegalArgumentException("Eye tracking image has no planes")
        val width: Int = image.width
        val height: Int = image.height
        private val rowStride: Int = plane.rowStride
        private val pixelStride: Int = plane.pixelStride
        private val buffer: ByteBuffer = plane.buffer.duplicate()

        init {
            require(width > 0 && height > 0)
            require(rowStride > 0 && pixelStride > 0)
            val lastByteIndex =
                (height - 1L) * rowStride.toLong() +
                    (width - 1L) * pixelStride.toLong()
            require(lastByteIndex >= 0L && lastByteIndex < buffer.limit().toLong()) {
                "Eye tracking luma plane is truncated"
            }
        }

        fun valid(x: Int, y: Int): Boolean =
            x in 0 until width && y in 0 until height

        fun sample(x: Int, y: Int): Int {
            val index =
                y.toLong() * rowStride.toLong() +
                    x.toLong() * pixelStride.toLong()
            if (index < 0L || index >= buffer.limit().toLong()) return 255
            return buffer.get(index.toInt()).toInt() and 0xFF
        }
    }

    private fun Vec2.distanceTo(other: Vec2): Double {
        val dx = x - other.x
        val dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun Vec2.lerp(other: Vec2, alpha: Double): Vec2 =
        Vec2(
            x * (1.0 - alpha) + other.x * alpha,
            y * (1.0 - alpha) + other.y * alpha,
        )

    private fun mapDetectorPointToSensor(
        point: Vec2,
        rawWidth: Int,
        rawHeight: Int,
        rotationDegrees: Int,
    ): Vec2 =
        ImageCoordinateTransform.detectorToSensor(
            point = point,
            rawWidth = rawWidth,
            rawHeight = rawHeight,
            rotationDegrees = rotationDegrees,
        )

    override fun close() {
        tracks.clear()
    }
}
