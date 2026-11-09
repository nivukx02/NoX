package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Production-grade subject-lock and focus arbitration layer.
 *
 * The engine deliberately does not depend on a detector implementation. Camera/ML frontends
 * convert their observations into [Observation] and receive a deterministic [Decision]. This
 * keeps the decision layer usable for humans, animals, vehicles, props and other moving objects.
 *
 * Design goals:
 * - stable identities without requiring detector-provided track IDs
 * - motion prediction with acceleration and jerk limiting
 * - explicit camera-motion compensation
 * - eye/gaze-aware focus selection when valid eye observations exist
 * - graceful degradation to face/head/body/object targets
 * - occlusion grace windows and reacquisition
 * - bounded compute and memory per frame
 * - deterministic hysteresis so UI/AF hardware is not driven by frame-to-frame noise
 * - confidence/quality values that represent evidence, not invented certainty
 */
class ProductionSubjectLockEngine(
    private val config: Config = Config(),
) {
    data class Config(
        // Focus/subject arbitration must follow the same 64-actor ceiling as the motion graph.
        val maxTracks: Int = 64,
        val maxCandidatesForGlobalAssignment: Int = 64,
        val candidateTimeoutNanos: Long = 700_000_000L,
        val hardTrackTimeoutNanos: Long = 1_800_000_000L,
        val lockAcquireNanos: Long = 220_000_000L,
        val lockReleaseNanos: Long = 420_000_000L,
        val focusSwitchPenalty: Double = 0.22,
        val focusSwitchMargin: Double = 0.10,
        val predictionHorizonNanos: Long = 80_000_000L,
        val maximumPredictionSeconds: Double = 0.18,
        val maximumSpeedNormPerSecond: Double = 4.5,
        val maximumAccelerationNormPerSecond2: Double = 18.0,
        val maximumJerkNormPerSecond3: Double = 80.0,
        val maxAssignmentDistanceNorm: Double = 0.42,
        val minimumObservationConfidence: Double = 0.08,
        val minimumEyeConfidence: Double = 0.34,
        val minimumGazeConfidence: Double = 0.30,
        val occlusionDecaySeconds: Double = 0.42,
        val cameraMotionSuppression: Double = 0.75,
        val focusDamping: Double = 0.82,
        val focusVelocityLimitNormPerSecond: Double = 3.2,
        val focusAccelerationLimitNormPerSecond2: Double = 24.0,
        val saliencyWeight: Double = 0.12,
        val motionWeight: Double = 0.20,
        val confidenceWeight: Double = 0.23,
        val persistenceWeight: Double = 0.18,
        val eyeWeight: Double = 0.17,
        val proximityWeight: Double = 0.10,
    ) {
        init {
            require(maxTracks in 1..128)
            require(
                maxCandidatesForGlobalAssignment in
                    1..maxTracks
            )
            require(candidateTimeoutNanos > 0L)
            require(hardTrackTimeoutNanos >= candidateTimeoutNanos)
            require(lockAcquireNanos > 0L)
            require(lockReleaseNanos >= lockAcquireNanos)
            require(maxPredictionHorizonNanosSafe() > 0L)
        }

        private fun maxPredictionHorizonNanosSafe(): Long =
            predictionHorizonNanos.coerceIn(1L, 1_000_000_000L)
    }

    enum class SubjectKind {
        HUMAN,
        ANIMAL,
        OBJECT,
        UNKNOWN,
    }

    enum class FocusSource {
        LEFT_EYE,
        RIGHT_EYE,
        GAZE_FUSION,
        FACE_CENTER,
        HEAD_CENTER,
        SUBJECT_CENTER,
        PREDICTED_SUBJECT,
        HOLD_LAST_LOCK,
    }

    enum class LockState {
        SEARCHING,
        CANDIDATE,
        LOCKED,
        OCCLUDED,
        REACQUIRING,
    }

    data class EyeObservation(
        val x: Double,
        val y: Double,
        val confidence: Double,
        val openness: Double = 1.0,
    ) {
        init {
            require(x.isFinite() && y.isFinite())
            require(confidence.isFinite())
            require(openness.isFinite())
        }
    }

    data class GazeObservation(
        val x: Double,
        val y: Double,
        val confidence: Double,
    ) {
        init {
            require(x.isFinite() && y.isFinite())
            require(confidence.isFinite())
        }
    }

    data class Observation(
        val detectorId: Long = -1L,
        val kind: SubjectKind = SubjectKind.UNKNOWN,
        val left: Double,
        val top: Double,
        val right: Double,
        val bottom: Double,
        val confidence: Double,
        val saliency: Double = 0.0,
        val faceCenterX: Double? = null,
        val faceCenterY: Double? = null,
        val headCenterX: Double? = null,
        val headCenterY: Double? = null,
        val leftEye: EyeObservation? = null,
        val rightEye: EyeObservation? = null,
        val gaze: GazeObservation? = null,
    ) {
        val centerX: Double get() = (left + right) * 0.5
        val centerY: Double get() = (top + bottom) * 0.5
        val width: Double get() = (right - left).coerceAtLeast(0.0)
        val height: Double get() = (bottom - top).coerceAtLeast(0.0)
        val area: Double get() = (width * height).coerceIn(0.0, 1.0)

        fun finite(): Boolean =
            left.isFinite() &&
                top.isFinite() &&
                right.isFinite() &&
                bottom.isFinite() &&
                confidence.isFinite() &&
                saliency.isFinite()
    }

    data class CameraMotion(
        val vx: Double = 0.0,
        val vy: Double = 0.0,
        val confidence: Double = 0.0,
    ) {
        fun norm(): Double = hypot(vx, vy).coerceAtMost(8.0)
    }

    data class FocusTarget(
        val x: Double,
        val y: Double,
        val source: FocusSource,
        val confidence: Double,
        val trackId: Long?,
        val locked: Boolean,
    )

    data class TrackTelemetry(
        val id: Long,
        val kind: SubjectKind,
        val confidence: Double,
        val stability: Double,
        val motion: Double,
        val occlusion: Double,
        val ageNanos: Long,
        val missedNanos: Long,
        val predictedX: Double,
        val predictedY: Double,
        val score: Double,
        val focusable: Boolean,
    )

    data class Decision(
        val timestampNanos: Long,
        val lockState: LockState,
        val lockedTrackId: Long?,
        val focus: FocusTarget?,
        val selected: TrackTelemetry?,
        val tracks: List<TrackTelemetry>,
        val cameraMotionRisk: Double,
        val globalConfidence: Double,
        val targetSwitchCount: Long,
        val computeCostEstimate: Int,
    )

    private data class State(
        var id: Long,
        var kind: SubjectKind,
        var detectorId: Long,
        var x: Double,
        var y: Double,
        var w: Double,
        var h: Double,
        var vx: Double,
        var vy: Double,
        var ax: Double,
        var ay: Double,
        var jx: Double,
        var jy: Double,
        var confidence: Double,
        var saliency: Double,
        var eyeConfidence: Double,
        var gazeConfidence: Double,
        var lastSeenNanos: Long,
        var firstSeenNanos: Long,
        var missedFrames: Int,
        var appearanceScore: Double,
        var quality: Double,
        var lockEvidence: Double,
        var unlockEvidence: Double,
        var focusX: Double,
        var focusY: Double,
        var focusVx: Double,
        var focusVy: Double,
        var locked: Boolean,
        var lastSource: FocusSource = FocusSource.SUBJECT_CENTER,
        var lastObservationConfidence: Double = 0.0,
    ) {
        fun ageNanos(now: Long): Long =
            (now - firstSeenNanos).coerceAtLeast(0L)

        fun missedNanos(now: Long): Long =
            (now - lastSeenNanos).coerceAtLeast(0L)
    }

    private val states = LinkedHashMap<Long, State>()
    private var runtimeMaxTracks = config.maxTracks
    private var nextId = 1L
    private var lockedTrackId: Long? = null
    private var lastTimestampNanos = Long.MIN_VALUE
    private var lastCameraMotion = CameraMotion()
    private var targetSwitchCount = 0L
    private var lastFocus = FocusTarget(
        x = 0.5,
        y = 0.5,
        source = FocusSource.HOLD_LAST_LOCK,
        confidence = 0.0,
        trackId = null,
        locked = false,
    )

    @Synchronized
    fun configureMaximumSubjects(
        maximumSubjects: Int,
    ) {
        runtimeMaxTracks =
            maximumSubjects.coerceIn(
                1,
                config.maxTracks,
            )
        trimTracksIfNeeded()
    }

    private fun trimTracksIfNeeded() {
        if (states.size <= runtimeMaxTracks) return

        val now = lastTimestampNanos.coerceAtLeast(0L)
        val victims = states.values
            .asSequence()
            .filter { it.id != lockedTrackId }
            .sortedWith(
                compareBy<State> { it.locked }
                    .thenBy { it.confidence }
                    .thenByDescending { it.missedNanos(now) }
                    .thenBy { it.lastSeenNanos },
            )
            .toList()

        var index = 0
        while (states.size > runtimeMaxTracks && index < victims.size) {
            val victim = victims[index++]
            states.remove(victim.id)
        }

        if (states.size > runtimeMaxTracks) {
            val fallback = states.keys.firstOrNull { it != lockedTrackId }
            if (fallback != null) states.remove(fallback)
        }

        if (lockedTrackId != null && !states.containsKey(lockedTrackId)) {
            lockedTrackId = null
        }
    }

    @Synchronized
    fun update(
        timestampNanos: Long,
        observations: List<Observation>,
        cameraMotion: CameraMotion = CameraMotion(),
    ): Decision {
        require(timestampNanos >= 0L) { "timestampNanos must be non-negative" }

        val dtSeconds = when {
            lastTimestampNanos == Long.MIN_VALUE -> 1.0 / 30.0
            timestampNanos <= lastTimestampNanos -> 1.0 / 120.0
            else -> (
                (timestampNanos - lastTimestampNanos).toDouble() * 1e-9
            ).coerceIn(1.0 / 240.0, 0.25)
        }

        lastTimestampNanos = timestampNanos
        lastCameraMotion = sanitizeCameraMotion(cameraMotion)

        val candidates = observations
            .asSequence()
            .filter { it.finite() }
            .filter { it.confidence >= config.minimumObservationConfidence }
            .map(::sanitizeObservation)
            .take(config.maxTracks * 2)
            .toList()

        predictAll(
            dtSeconds = dtSeconds,
            cameraMotion = lastCameraMotion,
        )

        val assignment = assign(candidates)
        val matchedIds = BooleanArray(states.size)
        val stateEntries = states.values.toList()

        assignment.forEachIndexed { candidateIndex, stateIndex ->
            if (candidateIndex !in candidates.indices || stateIndex !in stateEntries.indices) {
                return@forEachIndexed
            }
            val state = stateEntries[stateIndex]
            val candidate = candidates[candidateIndex]
            updateMatched(
                state = state,
                observation = candidate,
                timestampNanos = timestampNanos,
                dtSeconds = dtSeconds,
            )
            matchedIds[stateIndex] = true
        }

        val assignedCandidateIndices = assignment
            .withIndex()
            .filter { it.value >= 0 }
            .map { it.index }
            .toHashSet()
        candidates.forEachIndexed { candidateIndex, observation ->
            if (candidateIndex !in assignedCandidateIndices) {
                spawn(
                    observation = observation,
                    timestampNanos = timestampNanos,
                    dtSeconds = dtSeconds,
                )
            }
        }

        pruneExpired(timestampNanos)
        updateLockEvidence(timestampNanos, dtSeconds)

        val best = chooseBestTrack(timestampNanos)
        val selectedState = best?.let { states[it] }

        updateLockedSelection(
            bestId = best,
            timestampNanos = timestampNanos,
            dtSeconds = dtSeconds,
        )

        val focus = selectedState?.let {
            deriveFocusTarget(
                state = it,
                timestampNanos = timestampNanos,
                dtSeconds = dtSeconds,
            )
        } ?: retainLastFocus(timestampNanos)

        val telemetry = states.values
            .sortedByDescending { state ->
                selectionScore(state, timestampNanos)
            }
            .take(config.maxTracks)
            .map { state ->
                telemetry(state, timestampNanos)
            }

        val confidence = computeGlobalConfidence(
            observations = candidates,
            selected = selectedState,
            timestampNanos = timestampNanos,
        )

        return Decision(
            timestampNanos = timestampNanos,
            lockState = lockState(selectedState, timestampNanos),
            lockedTrackId = lockedTrackId,
            focus = focus,
            selected = selectedState?.let { telemetry(it, timestampNanos) },
            tracks = telemetry,
            cameraMotionRisk = cameraMotionRisk(lastCameraMotion),
            globalConfidence = confidence,
            targetSwitchCount = targetSwitchCount,
            computeCostEstimate = estimateComputeCost(
                candidateCount = candidates.size,
                trackCount = states.size,
            ),
        )
    }

    @Synchronized
    fun reset() {
        states.clear()
        runtimeMaxTracks = config.maxTracks
        nextId = 1L
        lockedTrackId = null
        lastTimestampNanos = Long.MIN_VALUE
        lastCameraMotion = CameraMotion()
        targetSwitchCount = 0L
        lastFocus = FocusTarget(
            x = 0.5,
            y = 0.5,
            source = FocusSource.HOLD_LAST_LOCK,
            confidence = 0.0,
            trackId = null,
            locked = false,
        )
    }

    @Synchronized
    fun snapshot(nowNanos: Long = lastTimestampNanos.coerceAtLeast(0L)): List<TrackTelemetry> =
        states.values
            .sortedByDescending { selectionScore(it, nowNanos) }
            .take(config.maxTracks)
            .map { telemetry(it, nowNanos) }

    private fun sanitizeCameraMotion(value: CameraMotion): CameraMotion =
        CameraMotion(
            vx = finiteOrZero(value.vx).coerceIn(-8.0, 8.0),
            vy = finiteOrZero(value.vy).coerceIn(-8.0, 8.0),
            confidence = finiteOrZero(value.confidence).coerceIn(0.0, 1.0),
        )

    private fun sanitizeObservation(input: Observation): Observation {
        val l = input.left.coerceIn(-0.25, 1.25)
        val t = input.top.coerceIn(-0.25, 1.25)
        val r = input.right.coerceIn(-0.25, 1.25)
        val b = input.bottom.coerceIn(-0.25, 1.25)

        val correctedLeft = min(l, r)
        val correctedRight = max(l, r)
        val correctedTop = min(t, b)
        val correctedBottom = max(t, b)

        return input.copy(
            left = correctedLeft,
            top = correctedTop,
            right = correctedRight,
            bottom = correctedBottom,
            confidence = input.confidence.coerceIn(0.0, 1.0),
            saliency = input.saliency.coerceIn(0.0, 1.0),
            faceCenterX = input.faceCenterX?.coerceIn(0.0, 1.0),
            faceCenterY = input.faceCenterY?.coerceIn(0.0, 1.0),
            headCenterX = input.headCenterX?.coerceIn(0.0, 1.0),
            headCenterY = input.headCenterY?.coerceIn(0.0, 1.0),
            leftEye = input.leftEye?.let(::sanitizeEye),
            rightEye = input.rightEye?.let(::sanitizeEye),
            gaze = input.gaze?.let(::sanitizeGaze),
        )
    }

    private fun sanitizeEye(value: EyeObservation): EyeObservation =
        value.copy(
            x = value.x.coerceIn(0.0, 1.0),
            y = value.y.coerceIn(0.0, 1.0),
            confidence = value.confidence.coerceIn(0.0, 1.0),
            openness = value.openness.coerceIn(0.0, 1.0),
        )

    private fun sanitizeGaze(value: GazeObservation): GazeObservation =
        value.copy(
            x = value.x.coerceIn(0.0, 1.0),
            y = value.y.coerceIn(0.0, 1.0),
            confidence = value.confidence.coerceIn(0.0, 1.0),
        )

    private fun predictAll(
        dtSeconds: Double,
        cameraMotion: CameraMotion,
    ) {
        states.values.forEach { state ->
            val cameraWeight = (
                cameraMotion.confidence * config.cameraMotionSuppression
            ).coerceIn(0.0, 1.0)

            val compensatedVx = state.vx - cameraMotion.vx * cameraWeight
            val compensatedVy = state.vy - cameraMotion.vy * cameraWeight

            val speed = hypot(compensatedVx, compensatedVy)
            val boundedSpeed = speed.coerceAtMost(config.maximumSpeedNormPerSecond)
            val speedScale = if (speed > 1e-9) boundedSpeed / speed else 1.0

            state.vx = compensatedVx * speedScale
            state.vy = compensatedVy * speedScale

            val nextAx = (
                state.ax +
                    state.jx * dtSeconds
            ).coerceIn(
                -config.maximumAccelerationNormPerSecond2,
                config.maximumAccelerationNormPerSecond2,
            )
            val nextAy = (
                state.ay +
                    state.jy * dtSeconds
            ).coerceIn(
                -config.maximumAccelerationNormPerSecond2,
                config.maximumAccelerationNormPerSecond2,
            )

            state.ax = nextAx
            state.ay = nextAy

            val px = (
                state.x +
                    state.vx * dtSeconds +
                    0.5 * state.ax * dtSeconds * dtSeconds
            )
            val py = (
                state.y +
                    state.vy * dtSeconds +
                    0.5 * state.ay * dtSeconds * dtSeconds
            )

            state.x = px.coerceIn(-0.5, 1.5)
            state.y = py.coerceIn(-0.5, 1.5)
            state.focusX = state.x.coerceIn(0.0, 1.0)
            state.focusY = state.y.coerceIn(0.0, 1.0)
        }
    }

    private fun assign(
        candidates: List<Observation>,
    ): IntArray {
        if (
            candidates.isEmpty() ||
            states.isEmpty()
        ) {
            return IntArray(0)
        }

        val stateEntries =
            states.values.toList()
        val boundedCandidates =
            candidates.take(
                config.maxCandidatesForGlobalAssignment,
            )

        if (
            boundedCandidates.isEmpty() ||
            stateEntries.isEmpty()
        ) {
            return IntArray(
                candidates.size,
            ) { -1 }
        }

        val rowCount =
            boundedCandidates.size
        val columnCount =
            min(
                stateEntries.size,
                config.maxCandidatesForGlobalAssignment,
            )
        val n =
            max(rowCount, columnCount)

        /*
         * Preserve the previous dynamic-programming behaviour: skipping a candidate cost 0.35,
         * while an association above 1.0 is semantically invalid. Padding the square matrix with
         * skip cost gives the Hungarian solver explicit unmatched rows/columns.
         */
        val skipCost = 0.35
        val forbiddenCost = 1_000_000.0
        val costs =
            Array(n) {
                DoubleArray(n) {
                    skipCost
                }
            }

        for (row in 0 until rowCount) {
            for (
                column in 0 until columnCount
            ) {
                val cost =
                    associationCost(
                        boundedCandidates[row],
                        stateEntries[column],
                    )

                costs[row][column] =
                    if (
                        cost.isFinite() &&
                        cost <= 1.0
                    ) {
                        cost
                    } else {
                        forbiddenCost
                    }
            }
        }

        val assignment =
            hungarianMinimize(
                costs = costs,
                forbiddenCost = forbiddenCost,
            )

        val result =
            IntArray(
                candidates.size,
            ) { -1 }

        for (
            row in 0 until rowCount
        ) {
            val column =
                assignment[row]

            if (
                column !in
                0 until columnCount
            ) {
                continue
            }

            val cost =
                costs[row][column]

            if (
                cost >=
                    forbiddenCost
            ) {
                continue
            }

            /*
             * A match is accepted only when it beats the explicit unmatched penalty. This mirrors
             * the former recursive solver while making complexity O(n^3), n <= 64.
             */
            if (cost >= skipCost) {
                continue
            }

            result[row] =
                column
        }

        return result
    }

    private fun hungarianMinimize(
        costs: Array<DoubleArray>,
        forbiddenCost: Double,
    ): IntArray {
        val n =
            costs.size
        if (n == 0) {
            return IntArray(0)
        }

        val u =
            DoubleArray(n + 1)
        val v =
            DoubleArray(n + 1)
        val p =
            IntArray(n + 1)
        val way =
            IntArray(n + 1)

        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv =
                DoubleArray(n + 1) {
                    Double.POSITIVE_INFINITY
                }
            val used =
                BooleanArray(n + 1)

            do {
                used[j0] = true
                val i0 =
                    p[j0]
                var delta =
                    Double.POSITIVE_INFINITY
                var j1 = 0

                for (j in 1..n) {
                    if (used[j]) {
                        continue
                    }

                    val currentCost =
                        if (
                            costs[i0 - 1][j - 1]
                                >= forbiddenCost
                        ) {
                            1e12
                        } else {
                            costs[i0 - 1][j - 1]
                        }

                    val current =
                        currentCost -
                            u[i0] -
                            v[j]

                    if (
                        current <
                            minv[j]
                    ) {
                        minv[j] =
                            current
                        way[j] =
                            j0
                    }

                    if (
                        minv[j] <
                            delta
                    ) {
                        delta =
                            minv[j]
                        j1 =
                            j
                    }
                }

                if (
                    !delta.isFinite()
                ) {
                    break
                }

                for (j in 0..n) {
                    if (used[j]) {
                        u[p[j]] +=
                            delta
                        v[j] -=
                            delta
                    } else {
                        minv[j] -=
                            delta
                    }
                }

                j0 = j1
            } while (
                j0 != 0 &&
                p[j0] != 0
            )

            if (
                j0 == 0
            ) {
                continue
            }

            do {
                val j1 =
                    way[j0]
                p[j0] =
                    p[j1]
                j0 =
                    j1
            } while (
                j0 != 0
            )
        }

        val rowToColumn =
            IntArray(n) { -1 }

        for (j in 1..n) {
            if (p[j] != 0) {
                rowToColumn[
                    p[j] - 1
                ] =
                    j - 1
            }
        }

        return rowToColumn
    }

    private fun associationCost(
        observation: Observation,
        state: State,
    ): Double {
        val dx = observation.centerX - state.x
        val dy = observation.centerY - state.y
        val centerDistance = hypot(dx, dy)

        val sizeDelta = abs(
            (
                sqrt(max(observation.width * observation.height, 1e-8)) -
                    sqrt(max(state.w * state.h, 1e-8))
            ),
        ).coerceAtMost(1.0)

        val kindPenalty = if (
            observation.kind != SubjectKind.UNKNOWN &&
            state.kind != SubjectKind.UNKNOWN &&
            observation.kind != state.kind
        ) 0.34 else 0.0

        val detectorBoost = if (
            observation.detectorId >= 0L &&
            state.detectorId >= 0L &&
            observation.detectorId == state.detectorId
        ) {
            -0.26
        } else {
            0.0
        }

        val persistenceBoost = if (state.missedFrames == 0) -0.08 else 0.0

        return (
            0.62 * (centerDistance / config.maxAssignmentDistanceNorm.coerceAtLeast(0.01)) +
                0.16 * sizeDelta +
                0.22 * kindPenalty +
                detectorBoost +
                persistenceBoost
            ).coerceIn(0.0, 4.0)
    }

    private fun updateMatched(
        state: State,
        observation: Observation,
        timestampNanos: Long,
        dtSeconds: Double,
    ) {
        val residualX = observation.centerX - state.x
        val residualY = observation.centerY - state.y

        val normalizedDt = dtSeconds.coerceIn(1.0 / 240.0, 0.25)

        val alpha = (
            0.35 +
                observation.confidence * 0.50 +
                state.quality * 0.10
        ).coerceIn(0.20, 0.95)

        val beta = (
            0.025 +
                observation.confidence * 0.12
        ).coerceIn(0.02, 0.16)

        val gamma = (
            0.004 +
                observation.confidence * 0.025
        ).coerceIn(0.003, 0.04)

        val correctionX = residualX * alpha
        val correctionY = residualY * alpha

        val previousVx = state.vx
        val previousVy = state.vy

        state.x += correctionX
        state.y += correctionY

        state.vx += beta * residualX / normalizedDt
        state.vy += beta * residualY / normalizedDt

        state.ax += gamma * (
            (
                state.vx - previousVx
            ) / normalizedDt - state.ax
        )
        state.ay += gamma * (
            (
                state.vy - previousVy
            ) / normalizedDt - state.ay
        )

        state.jx = (
            (
                state.ax - state.jx * normalizedDt
            ) / normalizedDt
        ).coerceIn(
            -config.maximumJerkNormPerSecond3,
            config.maximumJerkNormPerSecond3,
        )
        state.jy = (
            (
                state.ay - state.jy * normalizedDt
            ) / normalizedDt
        ).coerceIn(
            -config.maximumJerkNormPerSecond3,
            config.maximumJerkNormPerSecond3,
        )

        val confidence = observation.confidence.coerceIn(0.0, 1.0)
        val eyeConfidence = eyeConfidence(observation)
        val gazeConfidence = observation.gaze?.confidence?.coerceIn(0.0, 1.0) ?: 0.0

        state.w = blend(state.w, observation.width, 0.42 + confidence * 0.35)
        state.h = blend(state.h, observation.height, 0.42 + confidence * 0.35)
        state.confidence = blend(state.confidence, confidence, 0.32)
        state.saliency = blend(state.saliency, observation.saliency, 0.28)
        state.eyeConfidence = blend(state.eyeConfidence, eyeConfidence, 0.35)
        state.gazeConfidence = blend(state.gazeConfidence, gazeConfidence, 0.34)
        state.appearanceScore = blend(
            state.appearanceScore,
            appearanceCompatibility(state.kind, observation.kind),
            0.24,
        )
        state.quality = updateQuality(
            current = state.quality,
            confidence = confidence,
            eye = eyeConfidence,
            residual = hypot(residualX, residualY),
            missedFrames = state.missedFrames,
        )
        state.detectorId = if (observation.detectorId >= 0L) {
            observation.detectorId
        } else {
            state.detectorId
        }
        state.kind = fuseKind(state.kind, observation.kind)
        state.lastSeenNanos = timestampNanos
        state.missedFrames = 0
        state.lastObservationConfidence = confidence
    }

    private fun spawn(
        observation: Observation,
        timestampNanos: Long,
        dtSeconds: Double,
    ) {
        if (states.size >= config.maxTracks) {
            evictLowestPriority(timestampNanos)
        }

        val id = nextId++
        val eye = eyeConfidence(observation)
        val gaze = observation.gaze?.confidence?.coerceIn(0.0, 1.0) ?: 0.0

        states[id] = State(
            id = id,
            kind = observation.kind,
            detectorId = observation.detectorId,
            x = observation.centerX,
            y = observation.centerY,
            w = observation.width,
            h = observation.height,
            vx = 0.0,
            vy = 0.0,
            ax = 0.0,
            ay = 0.0,
            jx = 0.0,
            jy = 0.0,
            confidence = observation.confidence,
            saliency = observation.saliency,
            eyeConfidence = eye,
            gazeConfidence = gaze,
            lastSeenNanos = timestampNanos,
            firstSeenNanos = timestampNanos,
            missedFrames = 0,
            appearanceScore = 0.5,
            quality = (
                0.35 * observation.confidence +
                    0.25 * eye +
                    0.20 * gaze +
                    0.20
            ).coerceIn(0.0, 1.0),
            lockEvidence = 0.0,
            unlockEvidence = 0.0,
            focusX = (
                observation.gaze?.x?.takeIf { gaze >= config.minimumGazeConfidence }
                    ?: observation.leftEye?.x?.takeIf { eye >= config.minimumEyeConfidence }
                    ?: observation.rightEye?.x?.takeIf { eye >= config.minimumEyeConfidence }
                    ?: observation.faceCenterX
                    ?: observation.headCenterX
                    ?: observation.centerX
            ).coerceIn(0.0, 1.0),
            focusY = (
                observation.gaze?.y?.takeIf { gaze >= config.minimumGazeConfidence }
                    ?: observation.leftEye?.y?.takeIf { eye >= config.minimumEyeConfidence }
                    ?: observation.rightEye?.y?.takeIf { eye >= config.minimumEyeConfidence }
                    ?: observation.faceCenterY
                    ?: observation.headCenterY
                    ?: observation.centerY
            ).coerceIn(0.0, 1.0),
            focusVx = 0.0,
            focusVy = 0.0,
            locked = false,
            lastSource = FocusSource.SUBJECT_CENTER,
            lastObservationConfidence = observation.confidence,
        )

        if (dtSeconds > 0.0) {
            states[id]?.quality = states[id]?.quality?.coerceIn(0.0, 1.0) ?: 0.0
        }
    }

    private fun updateLockEvidence(
        nowNanos: Long,
        dtSeconds: Double,
    ) {
        states.values.forEach { state ->
            val selection = selectionScore(state, nowNanos)
            val evidenceGain = dtSeconds * (
                0.42 +
                    state.quality * 1.10 +
                    state.eyeConfidence * 0.35 +
                    state.gazeConfidence * 0.28
            )

            if (selection >= 0.56 && state.missedFrames == 0) {
                state.lockEvidence = (
                    state.lockEvidence + evidenceGain
                ).coerceAtMost(1.8)
            } else {
                state.lockEvidence = (
                    state.lockEvidence - dtSeconds * 0.46
                ).coerceAtLeast(0.0)
            }

            if (state.locked) {
                val releasePenalty = (
                    if (state.missedFrames > 0) 0.60 else 0.16
                )
                state.unlockEvidence = (
                    state.unlockEvidence + dtSeconds * releasePenalty
                ).coerceAtMost(1.8)

                if (
                    state.unlockEvidence >= config.lockReleaseNanos * 1e-9
                ) {
                    state.locked = false
                    state.unlockEvidence = 0.0
                }
            } else {
                state.unlockEvidence = 0.0
            }
        }
    }

    private fun chooseBestTrack(nowNanos: Long): Long? {
        if (states.isEmpty()) return null

        return states.values
            .asSequence()
            .filter { state ->
                state.missedNanos(nowNanos) <= config.hardTrackTimeoutNanos
            }
            .maxByOrNull { state ->
                selectionScore(state, nowNanos)
            }
            ?.id
    }

    private fun updateLockedSelection(
        bestId: Long?,
        timestampNanos: Long,
        dtSeconds: Double,
    ) {
        val current = lockedTrackId
        val currentScore = current?.let { states[it] }
            ?.let { selectionScore(it, timestampNanos) }
            ?: -1.0

        val bestScore = bestId?.let { states[it] }
            ?.let { selectionScore(it, timestampNanos) }
            ?: -1.0

        if (current == null) {
            val candidate = bestId?.let { states[it] }
            if (
                candidate != null &&
                bestScore >= 0.56 &&
                candidate.lockEvidence >= config.lockAcquireNanos * 1e-9
            ) {
                candidate.locked = true
                lockedTrackId = candidate.id
            }
            return
        }

        val currentState = states[current]
        if (currentState == null) {
            lockedTrackId = null
            return
        }

        if (
            bestId != null &&
            bestId != current &&
            bestScore > currentScore + config.focusSwitchMargin
        ) {
            val competing = states[bestId]
            if (
                competing != null &&
                competing.lockEvidence >= config.lockAcquireNanos * 1e-9
            ) {
                currentState.locked = false
                competing.locked = true
                lockedTrackId = competing.id
                targetSwitchCount++
            }
        } else if (
            currentState.missedNanos(timestampNanos) >
                config.lockReleaseNanos
        ) {
            currentState.locked = false
            lockedTrackId = bestId
            bestId?.let { states[it]?.locked = true }
            targetSwitchCount += if (bestId != null) 1 else 0
        } else {
            currentState.locked = true
        }

        val decay = exp(
            -dtSeconds /
                config.occlusionDecaySeconds.coerceAtLeast(0.05),
        )
        if (currentState.missedFrames > 0) {
            currentState.confidence *= decay
            currentState.quality = blend(
                currentState.quality,
                currentState.confidence,
                0.14,
            )
        }
    }

    private fun deriveFocusTarget(
        state: State,
        timestampNanos: Long,
        dtSeconds: Double,
    ): FocusTarget {
        val observationFresh = (
            state.missedNanos(timestampNanos) <= 70_000_000L
        )

        val raw = when {
            observationFresh && state.kind == SubjectKind.HUMAN &&
                state.gazeConfidence >= config.minimumGazeConfidence -> {
                FocusTarget(
                    x = state.focusX,
                    y = state.focusY,
                    source = FocusSource.GAZE_FUSION,
                    confidence = (
                        0.56 * state.confidence +
                            0.44 * state.gazeConfidence
                        ).coerceIn(0.0, 1.0),
                    trackId = state.id,
                    locked = state.locked,
                )
            }

            observationFresh && state.kind == SubjectKind.HUMAN &&
                state.eyeConfidence >= config.minimumEyeConfidence -> {
                FocusTarget(
                    x = state.focusX,
                    y = state.focusY,
                    source = if (
                        state.lastSource == FocusSource.RIGHT_EYE
                    ) {
                        FocusSource.RIGHT_EYE
                    } else {
                        FocusSource.LEFT_EYE
                    },
                    confidence = (
                        0.62 * state.confidence +
                            0.38 * state.eyeConfidence
                        ).coerceIn(0.0, 1.0),
                    trackId = state.id,
                    locked = state.locked,
                )
            }

            observationFresh && state.kind == SubjectKind.HUMAN -> {
                FocusTarget(
                    x = state.focusX,
                    y = state.focusY,
                    source = FocusSource.FACE_CENTER,
                    confidence = (
                        0.70 * state.confidence +
                            0.30 * state.quality
                        ).coerceIn(0.0, 1.0),
                    trackId = state.id,
                    locked = state.locked,
                )
            }

            observationFresh -> {
                FocusTarget(
                    x = state.focusX,
                    y = state.focusY,
                    source = if (state.missedFrames == 0) {
                        FocusSource.SUBJECT_CENTER
                    } else {
                        FocusSource.PREDICTED_SUBJECT
                    },
                    confidence = (
                        0.74 * state.confidence +
                            0.26 * state.quality
                        ).coerceIn(0.0, 1.0),
                    trackId = state.id,
                    locked = state.locked,
                )
            }

            else -> {
                FocusTarget(
                    x = state.x.coerceIn(0.0, 1.0),
                    y = state.y.coerceIn(0.0, 1.0),
                    source = FocusSource.PREDICTED_SUBJECT,
                    confidence = (
                        state.quality *
                            exp(
                                -state.missedNanos(timestampNanos) *
                                    1e-9 /
                                    config.occlusionDecaySeconds
                            )
                    ).coerceIn(0.0, 1.0),
                    trackId = state.id,
                    locked = state.locked,
                )
            }
        }

        state.lastSource = raw.source

        val filtered = smoothFocus(
            previousX = lastFocus.x,
            previousY = lastFocus.y,
            targetX = raw.x,
            targetY = raw.y,
            confidence = raw.confidence,
            dtSeconds = dtSeconds,
        )

        val output = raw.copy(
            x = filtered.first,
            y = filtered.second,
        )
        lastFocus = output
        return output
    }

    private fun retainLastFocus(
        nowNanos: Long,
    ): FocusTarget {
        val age = if (lastTimestampNanos == Long.MIN_VALUE) {
            Long.MAX_VALUE
        } else {
            (nowNanos - lastTimestampNanos).coerceAtLeast(0L)
        }

        val confidence = (
            lastFocus.confidence *
                exp(-age * 1e-9 / 0.45)
        ).coerceIn(0.0, 1.0)

        return lastFocus.copy(
            confidence = confidence,
            locked = false,
            source = FocusSource.HOLD_LAST_LOCK,
        )
    }

    private fun smoothFocus(
        previousX: Double,
        previousY: Double,
        targetX: Double,
        targetY: Double,
        confidence: Double,
        dtSeconds: Double,
    ): Pair<Double, Double> {
        val gain = (
            (1.0 - config.focusDamping) *
                (0.35 + confidence * 0.90)
        ).coerceIn(0.04, 0.85)

        var vx = (targetX - previousX) / dtSeconds.coerceAtLeast(1.0 / 240.0)
        var vy = (targetY - previousY) / dtSeconds.coerceAtLeast(1.0 / 240.0)

        val speed = hypot(vx, vy)
        if (speed > config.focusVelocityLimitNormPerSecond) {
            val scale = config.focusVelocityLimitNormPerSecond / speed
            vx *= scale
            vy *= scale
        }

        val predictedX = previousX + vx * dtSeconds
        val predictedY = previousY + vy * dtSeconds

        val nextX = blend(previousX, predictedX, gain)
            .coerceIn(0.0, 1.0)
        val nextY = blend(previousY, predictedY, gain)
            .coerceIn(0.0, 1.0)

        return nextX to nextY
    }

    private fun selectionScore(
        state: State,
        nowNanos: Long,
    ): Double {
        val freshness = exp(
            -state.missedNanos(nowNanos) * 1e-9 /
                config.occlusionDecaySeconds.coerceAtLeast(0.05),
        ).coerceIn(0.0, 1.0)

        val persistence = (
            state.ageNanos(nowNanos).toDouble() /
                2_000_000_000.0
        ).coerceIn(0.0, 1.0)

        val focusProximity = (
            1.0 -
                hypot(
                    state.focusX - 0.5,
                    state.focusY - 0.5,
                ) / 0.70710678
        ).coerceIn(0.0, 1.0)

        val motion = (
            hypot(state.vx, state.vy) /
                config.maximumSpeedNormPerSecond
        ).coerceIn(0.0, 1.0)

        val score = (
            config.confidenceWeight * state.confidence +
                config.persistenceWeight * (
                    0.5 * persistence +
                        0.5 * freshness
                    ) +
                config.eyeWeight * (
                    0.55 * state.eyeConfidence +
                        0.45 * state.gazeConfidence
                    ) +
                config.saliencyWeight * state.saliency +
                config.motionWeight * motion +
                config.proximityWeight * focusProximity +
                0.10 * state.appearanceScore
            )

        val lockPenalty = if (
            lockedTrackId != null &&
            lockedTrackId != state.id
        ) {
            config.focusSwitchPenalty
        } else {
            0.0
        }

        return (
            score - lockPenalty
        ).coerceIn(0.0, 1.0)
    }

    private fun telemetry(
        state: State,
        nowNanos: Long,
    ): TrackTelemetry =
        TrackTelemetry(
            id = state.id,
            kind = state.kind,
            confidence = state.confidence.coerceIn(0.0, 1.0),
            stability = state.quality.coerceIn(0.0, 1.0),
            motion = (
                hypot(state.vx, state.vy) /
                    config.maximumSpeedNormPerSecond
            ).coerceIn(0.0, 1.0),
            occlusion = (
                1.0 -
                    exp(
                        -state.missedNanos(nowNanos) * 1e-9 /
                            config.occlusionDecaySeconds.coerceAtLeast(0.05),
                    )
            ).coerceIn(0.0, 1.0),
            ageNanos = state.ageNanos(nowNanos),
            missedNanos = state.missedNanos(nowNanos),
            predictedX = state.x.coerceIn(-0.5, 1.5),
            predictedY = state.y.coerceIn(-0.5, 1.5),
            score = selectionScore(state, nowNanos),
            focusable = (
                state.confidence >= config.minimumObservationConfidence &&
                    state.quality >= 0.20 &&
                    state.missedNanos(nowNanos) <=
                        config.hardTrackTimeoutNanos
            ),
        )

    private fun lockState(
        selected: State?,
        nowNanos: Long,
    ): LockState {
        val locked = lockedTrackId?.let { states[it] }
        return when {
            locked == null && selected == null -> LockState.SEARCHING
            locked == null -> LockState.CANDIDATE
            locked.missedNanos(nowNanos) > 0L -> LockState.OCCLUDED
            selected?.id == locked.id -> LockState.LOCKED
            else -> LockState.REACQUIRING
        }
    }

    private fun computeGlobalConfidence(
        observations: List<Observation>,
        selected: State?,
        timestampNanos: Long,
    ): Double {
        if (observations.isEmpty() && selected == null) return 0.0

        val observationConfidence = if (observations.isEmpty()) {
            0.0
        } else {
            observations
                .map { it.confidence.coerceIn(0.0, 1.0) }
                .average()
        }

        val selectedConfidence = selected?.let { state ->
            val freshness = exp(
                -state.missedNanos(timestampNanos) * 1e-9 /
                    config.occlusionDecaySeconds.coerceAtLeast(0.05),
            ).coerceIn(0.0, 1.0)
            (
                0.54 * state.confidence +
                    0.28 * state.quality +
                    0.18 * freshness
            ).coerceIn(0.0, 1.0)
        } ?: 0.0

        return (
            0.42 * observationConfidence +
                0.58 * selectedConfidence
        ).coerceIn(0.0, 1.0)
    }

    private fun cameraMotionRisk(cameraMotion: CameraMotion): Double = (
        cameraMotion.norm() /
            config.maximumSpeedNormPerSecond
    ).coerceIn(0.0, 1.0) *
        cameraMotion.confidence.coerceIn(0.0, 1.0)

    private fun estimateComputeCost(
        candidateCount: Int,
        trackCount: Int,
    ): Int {
        val assignmentCost = if (
            candidateCount <= config.maxCandidatesForGlobalAssignment &&
            trackCount <= config.maxCandidatesForGlobalAssignment
        ) {
            candidateCount * max(trackCount, 1)
        } else {
            candidateCount * max(trackCount, 1) / 2 + candidateCount
        }

        val predictionCost = trackCount * 4
        val scoringCost = trackCount * 6
        return (assignmentCost + predictionCost + scoringCost).coerceAtMost(10_000)
    }

    private fun pruneExpired(nowNanos: Long) {
        val iterator = states.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            val state = entry.value
            if (state.missedNanos(nowNanos) > config.hardTrackTimeoutNanos) {
                if (lockedTrackId == state.id) {
                    lockedTrackId = null
                }
                iterator.remove()
            } else if (
                state.missedNanos(nowNanos) > config.candidateTimeoutNanos
            ) {
                state.missedFrames = (
                    state.missedFrames + 1
                ).coerceAtMost(Int.MAX_VALUE)
            }
        }
    }

    private fun evictLowestPriority(nowNanos: Long) {
        val victim = states.values
            .minByOrNull { selectionScore(it, nowNanos) }
            ?: return

        if (lockedTrackId == victim.id) {
            val replacement = states.values
                .filter { it.id != victim.id }
                .maxByOrNull { selectionScore(it, nowNanos) }
            lockedTrackId = replacement?.id
        }

        states.remove(victim.id)
    }

    private fun updateQuality(
        current: Double,
        confidence: Double,
        eye: Double,
        residual: Double,
        missedFrames: Int,
    ): Double {
        val residualScore = exp(
            -residual * residual / 0.04,
        ).coerceIn(0.0, 1.0)

        val missPenalty = exp(
            -missedFrames.toDouble() / 3.0,
        ).coerceIn(0.0, 1.0)

        val target = (
            0.46 * confidence +
                0.18 * eye +
                0.24 * residualScore +
                0.12 * missPenalty
        ).coerceIn(0.0, 1.0)

        return blend(
            current = current,
            measured = target,
            alpha = 0.24 + confidence * 0.30,
        )
    }

    private fun eyeConfidence(observation: Observation): Double {
        val left = observation.leftEye?.confidence?.coerceIn(0.0, 1.0) ?: 0.0
        val right = observation.rightEye?.confidence?.coerceIn(0.0, 1.0) ?: 0.0
        val presence = when {
            left > 0.0 && right > 0.0 -> 0.5 * (left + right)
            left > 0.0 -> left * 0.84
            right > 0.0 -> right * 0.84
            else -> 0.0
        }
        return (
            presence *
                (
                    0.60 +
                        0.40 * (
                            (
                                observation.leftEye?.openness ?: 0.0
                            ).coerceIn(0.0, 1.0) +
                                (
                                    observation.rightEye?.openness ?: 0.0
                                ).coerceIn(0.0, 1.0)
                        ).coerceAtMost(2.0) / 2.0
                    )
        ).coerceIn(0.0, 1.0)
    }

    private fun appearanceCompatibility(
        previous: SubjectKind,
        current: SubjectKind,
    ): Double = when {
        previous == current -> 1.0
        previous == SubjectKind.UNKNOWN || current == SubjectKind.UNKNOWN -> 0.65
        else -> 0.15
    }

    private fun fuseKind(
        previous: SubjectKind,
        current: SubjectKind,
    ): SubjectKind = when {
        current == SubjectKind.UNKNOWN -> previous
        previous == SubjectKind.UNKNOWN -> current
        previous == current -> current
        else -> current
    }

    private fun finiteOrZero(value: Double): Double =
        if (value.isFinite()) value else 0.0

    private fun blend(
        current: Double,
        measured: Double,
        alpha: Double,
    ): Double {
        val a = alpha.coerceIn(0.0, 1.0)
        return current * (1.0 - a) + measured * a
    }
}
