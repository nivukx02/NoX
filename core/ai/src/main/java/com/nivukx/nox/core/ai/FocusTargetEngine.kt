package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FocusTarget
import com.nivukx.nox.core.model.FocusTargetMode
import com.nivukx.nox.core.model.ImageCoordinateTransform
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TrackedObject
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Closed-loop focus target selector for Motion Capture NoX.
 *
 * The selector converts all detector-space geometry into the Camera2 sensor plane before emitting
 * a focus target. Eye iris coordinates are already sensor-space because EyeTrackingEngine searches
 * the raw luminance plane. Face and object boxes originate in detector/rotation space and therefore
 * pass through the same explicit rotation transform before normalization.
 *
 * Target priority:
 *   EYE > FACE > SUBJECT
 *
 * The state machine is deliberately hysteretic:
 *   - small detector jitter is smoothed;
 *   - a target does not jump for a marginal confidence improvement;
 *   - a better semantic target can pre-empt a lower-priority target;
 *   - stale targets expire rather than holding focus forever on a disappeared subject.
 *
 * The engine is pure perception/control policy. Camera2 owns the hardware request and AF trigger.
 */
class FocusTargetEngine(
    private val minimumEyeConfidence: Double = 0.38,
    private val minimumFaceSizePx: Double = 80.0,
    private val minimumSubjectConfidence: Double = 0.50,
    private val smoothingAlpha: Double = 0.34,
    private val switchMargin: Double = 0.12,
    private val staleAfterNanos: Long = 350_000_000L,
    private val edgeGuard: Double = 0.025,
) {

    private var last: FocusTarget? = null

    @Synchronized
    fun select(
        timestampNanos: Long,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int,
        faces: List<FaceFrame>,
        eyes: List<EyeTrackingFrame>,
        objects: List<TrackedObject>,
    ): FocusTarget? {
        val width = imageWidth.coerceAtLeast(1)
        val height = imageHeight.coerceAtLeast(1)

        val humanPresent =
            faces.isNotEmpty() ||
                objects.any {
                    it.subjectType == SubjectType.HUMAN &&
                        it.confidence >= minimumSubjectConfidence
                }

        val candidates =
            buildList {
                if (humanPresent) {
                    addAll(
                        eyeCandidates(
                            width = width,
                            height = height,
                            rotationDegrees = rotationDegrees,
                            faces = faces,
                            eyes = eyes,
                            timestampNanos = timestampNanos,
                        ),
                    )
                    addAll(
                        faceCandidates(
                            width = width,
                            height = height,
                            rotationDegrees = rotationDegrees,
                            faces = faces,
                            timestampNanos = timestampNanos,
                        ),
                    )
                }

                addAll(
                    subjectCandidates(
                        width = width,
                        height = height,
                        rotationDegrees = rotationDegrees,
                        objects = objects,
                        timestampNanos = timestampNanos,
                    ),
                )
            }

        val selected =
            candidates
                .filter { it.confidence > 0.05 }
                .filter { isInsideGuardBand(it) }
                .maxWithOrNull(
                    compareBy<FocusTarget> {
                        priority(it.mode)
                    }.thenBy {
                        it.confidence
                    },
                )

        if (selected == null) {
            val previous = last
            if (
                previous != null &&
                timestampNanos -
                    previous.timestampNanos >
                    staleAfterNanos
            ) {
                last = null
            }
            return last
        }

        val previous = last
        if (previous == null) {
            last = selected
            return selected
        }

        val previousPriority =
            priority(previous.mode)
        val selectedPriority =
            priority(selected.mode)

        val semanticPromotion =
            selectedPriority >
                previousPriority

        val confidencePromotion =
            selectedPriority ==
                previousPriority &&
                selected.confidence >=
                    previous.confidence +
                        switchMargin

        val sameTrack =
            selected.trackingId != null &&
                selected.trackingId ==
                    previous.trackingId

        val target =
            when {
                semanticPromotion -> {
                    selected
                }

                confidencePromotion -> {
                    selected
                }

                sameTrack -> {
                    previous.lerp(
                        selected,
                        smoothingAlpha,
                    ).copy(
                        mode = previous.mode,
                        trackingId =
                            selected.trackingId
                                ?: previous.trackingId,
                        subjectType =
                            if (
                                selected.subjectType !=
                                    SubjectType.UNKNOWN
                            ) {
                                selected.subjectType
                            } else {
                                previous.subjectType
                            },
                    )
                }

                else -> {
                    previous.lerp(
                        selected,
                        smoothingAlpha * 0.55,
                    )
                }
            }

        last =
            target.copy(
                normalizedX =
                    target.normalizedX.coerceIn(
                        edgeGuard,
                        1.0 - edgeGuard,
                    ),
                normalizedY =
                    target.normalizedY.coerceIn(
                        edgeGuard,
                        1.0 - edgeGuard,
                    ),
                confidence =
                    target.confidence.coerceIn(
                        0.0,
                        1.0,
                    ),
            )

        return last
    }

    @Synchronized
    fun reset() {
        last = null
    }

    private fun eyeCandidates(
        width: Int,
        height: Int,
        rotationDegrees: Int,
        faces: List<FaceFrame>,
        eyes: List<EyeTrackingFrame>,
        timestampNanos: Long,
    ): List<FocusTarget> =
        eyes.mapNotNull { frame ->
            if (
                frame.fixationConfidence <
                    minimumEyeConfidence
            ) {
                return@mapNotNull null
            }

            val face =
                faces.firstOrNull {
                    it.trackingId ==
                        frame.trackingId
                } ?: faces.minByOrNull {
                    eyeFaceDistance(
                        eye = frame,
                        face = it,
                        width = width,
                        height = height,
                        rotationDegrees = rotationDegrees,
                    )
                } ?: return@mapNotNull null

            val observations =
                listOfNotNull(
                    frame.left,
                    frame.right,
                )

            val usable =
                observations
                    .filter {
                        it.confidence >=
                            minimumEyeConfidence &&
                            !it.blink &&
                            it.irisCenter != null
                    }

            if (usable.isEmpty()) {
                return@mapNotNull null
            }

            val strongest =
                usable.maxByOrNull {
                    it.confidence
                } ?: return@mapNotNull null

            val target =
                sensorPointFromEye(
                    strongest,
                    width,
                    height,
                )

            FocusTarget(
                timestampNanos = timestampNanos,
                normalizedX =
                    target.first,
                normalizedY =
                    target.second,
                confidence =
                    (
                        strongest.confidence *
                            frame.fixationConfidence *
                            eyeSizeBoost(face)
                        ).coerceIn(
                            0.0,
                            1.0,
                        ),
                mode = FocusTargetMode.EYE,
                trackingId =
                    face.trackingId,
                subjectType =
                    SubjectType.HUMAN,
            )
        }

    private fun faceCandidates(
        width: Int,
        height: Int,
        rotationDegrees: Int,
        faces: List<FaceFrame>,
        timestampNanos: Long,
    ): List<FocusTarget> =
        faces.mapNotNull { face ->
            val faceWidth =
                (
                    face.boundsRight -
                        face.boundsLeft
                    ).coerceAtLeast(
                        0f,
                    ).toDouble()

            val faceHeight =
                (
                    face.boundsBottom -
                        face.boundsTop
                    ).coerceAtLeast(
                        0f,
                    ).toDouble()

            if (
                max(faceWidth, faceHeight) <
                    minimumFaceSizePx
            ) {
                return@mapNotNull null
            }

            val center =
                detectorPointToSensor(
                    x =
                        (
                            face.boundsLeft +
                                face.boundsRight
                            ) * 0.5,
                    y =
                        (
                            face.boundsTop +
                                face.boundsBottom
                            ) * 0.5,
                    width = width,
                    height = height,
                    rotationDegrees = rotationDegrees,
                )

            val sizeConfidence =
                min(
                    1.0,
                    max(
                        faceWidth,
                        faceHeight,
                    ) / 320.0,
                )

            FocusTarget(
                timestampNanos = timestampNanos,
                normalizedX =
                    center.first,
                normalizedY =
                    center.second,
                confidence =
                    (
                        0.45 +
                            sizeConfidence * 0.40
                        ).coerceIn(
                            0.0,
                            1.0,
                        ),
                mode = FocusTargetMode.FACE,
                trackingId =
                    face.trackingId,
                subjectType =
                    SubjectType.HUMAN,
            )
        }

    private fun subjectCandidates(
        width: Int,
        height: Int,
        rotationDegrees: Int,
        objects: List<TrackedObject>,
        timestampNanos: Long,
    ): List<FocusTarget> =
        objects
            .filter {
                it.confidence >=
                    minimumSubjectConfidence &&
                    it.subjectType !=
                        SubjectType.UNKNOWN
            }
            .mapNotNull { objectTrack ->
                val center =
                    detectorPointToSensor(
                        x =
                            (
                                objectTrack.left +
                                    objectTrack.right
                                ) * 0.5,
                        y =
                            (
                                objectTrack.top +
                                    objectTrack.bottom
                                ) * 0.5,
                        width = width,
                        height = height,
                        rotationDegrees = rotationDegrees,
                    )

                FocusTarget(
                    timestampNanos = timestampNanos,
                    normalizedX = center.first,
                    normalizedY = center.second,
                    confidence =
                        objectTrack.confidence.coerceIn(
                            0.0,
                            1.0,
                        ),
                    mode =
                        FocusTargetMode.SUBJECT,
                    trackingId =
                        objectTrack.trackingId,
                    subjectType =
                        objectTrack.subjectType,
                )
            }

    private fun sensorPointFromEye(
        eye: EyeObservation,
        width: Int,
        height: Int,
    ): Pair<Double, Double> {
        val iris =
            eye.irisCenter
                ?: eye.center

        return Pair(
            (iris.x / width)
                .coerceIn(0.0, 1.0),
            (iris.y / height)
                .coerceIn(0.0, 1.0),
        )
    }

    private fun detectorPointToSensor(
        x: Double,
        y: Double,
        width: Int,
        height: Int,
        rotationDegrees: Int,
    ): Pair<Double, Double> {
        val normalized =
            ImageCoordinateTransform.detectorToNormalizedSensor(
                point = Vec2(x, y),
                rawWidth = width,
                rawHeight = height,
                rotationDegrees = rotationDegrees,
            )

        return Pair(
            normalized.x,
            normalized.y,
        )
    }

    private fun eyeFaceDistance(
        eye: EyeTrackingFrame,
        face: FaceFrame,
        width: Int,
        height: Int,
        rotationDegrees: Int,
    ): Double {
        val faceCenter =
            detectorPointToSensor(
                x =
                    (
                        face.boundsLeft +
                            face.boundsRight
                        ) * 0.5,
                y =
                    (
                        face.boundsTop +
                            face.boundsBottom
                        ) * 0.5,
                width = width,
                height = height,
                rotationDegrees = rotationDegrees,
            )

        val cx = faceCenter.first * width
        val cy = faceCenter.second * height

        val point =
            eye.left?.irisCenter
                ?: eye.right?.irisCenter
                ?: eye.left?.center
                ?: eye.right?.center
                ?: return Double.POSITIVE_INFINITY

        val dx = point.x - cx
        val dy = point.y - cy
        return sqrt(
            dx * dx +
                dy * dy,
        )
    }

    private fun eyeSizeBoost(
        face: FaceFrame,
    ): Double =
        min(
            1.0,
            max(
                face.boundsRight -
                    face.boundsLeft,
                face.boundsBottom -
                    face.boundsTop,
            ).coerceAtLeast(1f) /
                360.0,
        ).let {
            0.82 +
                it * 0.18
        }

    private fun isInsideGuardBand(
        target: FocusTarget,
    ): Boolean =
        target.normalizedX in
            edgeGuard..(1.0 - edgeGuard) &&
            target.normalizedY in
            edgeGuard..(1.0 - edgeGuard)

    private fun priority(
        mode: FocusTargetMode,
    ): Int =
        when (mode) {
            FocusTargetMode.EYE -> 4
            FocusTargetMode.FACE -> 3
            FocusTargetMode.SUBJECT -> 2
            FocusTargetMode.CENTER -> 1
        }
}
