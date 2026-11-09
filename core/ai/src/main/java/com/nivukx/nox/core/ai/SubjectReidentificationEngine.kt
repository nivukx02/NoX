package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Detector-independent identity persistence.
 *
 * Detector IDs are session-local and can reset after occlusion, model reinitialization or camera
 * handoff. This engine maintains a stable NoX identity using geometry, motion state, semantic class
 * and temporal continuity without storing RGB appearance embeddings.
 */
class SubjectReidentificationEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumMissedFrames: Int = 45,
        val minimumMatchScore: Double = 0.48,
        val velocityWeight: Double = 0.22,
        val geometryWeight: Double = 0.32,
        val scaleWeight: Double = 0.16,
        val semanticsWeight: Double = 0.18,
        val temporalWeight: Double = 0.12,
        val maximumCenterDistance: Double = 0.32,
        val maximumScaleRatio: Double = 3.2,
    ) {
        init {
            require(maximumMissedFrames >= 0)
            require(minimumMatchScore in 0.0..1.0)
            listOf(
                velocityWeight,
                geometryWeight,
                scaleWeight,
                semanticsWeight,
                temporalWeight,
            ).forEach { require(it >= 0.0) }
            require(
                velocityWeight +
                    geometryWeight +
                    scaleWeight +
                    semanticsWeight +
                    temporalWeight > 0.0
            )
            require(maximumCenterDistance > 0.0)
            require(maximumScaleRatio >= 1.0)
        }
    }

    data class Identity(
        val stableId: Long,
        val currentTrackId: Long,
        val subjectType: SubjectType,
        val centerX: Double,
        val centerY: Double,
        val width: Double,
        val height: Double,
        val velocityX: Double,
        val velocityY: Double,
        val confidence: Double,
        val ageFrames: Long,
        val missedFrames: Int,
        val lastTimestampNanos: Long,
    )

    data class Assignment(
        val timestampNanos: Long,
        val identities: List<Identity>,
        val createdCount: Int,
        val reassociatedCount: Int,
        val droppedCount: Int,
    )

    private data class MutableIdentity(
        var stableId: Long,
        var currentTrackId: Long,
        var subjectType: SubjectType,
        var centerX: Double,
        var centerY: Double,
        var width: Double,
        var height: Double,
        var velocityX: Double,
        var velocityY: Double,
        var confidence: Double,
        var ageFrames: Long,
        var missedFrames: Int,
        var lastTimestampNanos: Long,
    )

    private data class Candidate(
        val identityId: Long,
        val trackId: Long,
        val score: Double,
    )

    private val identities = LinkedHashMap<Long, MutableIdentity>()
    private var nextStableId = 1L
    private var lastUpdateTimestampNanos: Long = Long.MIN_VALUE
    private var lastAssignment: Assignment? = null

    @Synchronized
    fun update(
        tracks: List<SubjectTrackManager.Track>,
        timestampNanos: Long,
    ): Assignment {
        require(timestampNanos >= 0L)

        if (timestampNanos <= lastUpdateTimestampNanos) {
            return lastAssignment
                ?: Assignment(
                    timestampNanos = timestampNanos,
                    identities = current(),
                    createdCount = 0,
                    reassociatedCount = 0,
                    droppedCount = 0,
                )
        }

        val assignments =
            globalAssignment(
                identities = identities.values.toList(),
                tracks = tracks,
                timestampNanos = timestampNanos,
            )

        var created = 0
        var reassociated = 0

        tracks.forEach { track ->
            val matched = assignments.firstOrNull {
                it.trackId == track.id
            }

            if (matched != null) {
                val identity =
                    identities[matched.identityId]
                        ?: return@forEach

                if (identity.currentTrackId != track.id) {
                    reassociated += 1
                }

                updateIdentity(
                    identity = identity,
                    track = track,
                    timestampNanos = timestampNanos,
                    score = matched.score,
                )
            } else {
                val stableId = nextStableId++
                val identity = MutableIdentity(
                    stableId = stableId,
                    currentTrackId = track.id,
                    subjectType = track.subjectType,
                    centerX = track.centerX.toDouble(),
                    centerY = track.centerY.toDouble(),
                    width = track.width.toDouble(),
                    height = track.height.toDouble(),
                    velocityX = track.velocityX.toDouble(),
                    velocityY = track.velocityY.toDouble(),
                    confidence = track.confidence,
                    ageFrames = 1L,
                    missedFrames = 0,
                    lastTimestampNanos = timestampNanos,
                )

                identities[stableId] = identity
                created += 1
            }
        }

        identities.values.forEach { identity ->
            val visible =
                tracks.any {
                    it.id == identity.currentTrackId
                }

            if (!visible) {
                identity.missedFrames += 1
                identity.ageFrames += 1
                identity.confidence *= 0.94
            }
        }

        val expired =
            identities.values
                .filter {
                    it.missedFrames >
                        config.maximumMissedFrames
                }
                .map { it.stableId }

        expired.forEach {
            identities.remove(it)
        }

        val snapshots = identities.values
            .filter { it.missedFrames <= config.maximumMissedFrames }
            .map(::snapshot)
            .sortedBy { it.stableId }

        val result =
            Assignment(
                timestampNanos = timestampNanos,
                identities = snapshots,
                createdCount = created,
                reassociatedCount = reassociated,
                droppedCount = expired.size,
            )
        lastUpdateTimestampNanos = timestampNanos
        lastAssignment = result
        return result
    }

    @Synchronized
    fun current(): List<Identity> =
        identities.values
            .map(::snapshot)
            .sortedBy { it.stableId }

    @Synchronized
    fun reset() {
        identities.clear()
        nextStableId = 1L
        lastUpdateTimestampNanos = Long.MIN_VALUE
        lastAssignment = null
    }

    private fun globalAssignment(
        identities: List<MutableIdentity>,
        tracks: List<SubjectTrackManager.Track>,
        timestampNanos: Long,
    ): List<AssignmentPair> {
        if (identities.isEmpty() || tracks.isEmpty()) {
            return emptyList()
        }

        val rows = identities.size
        val columns = tracks.size
        val size = max(rows, columns)

        val unmatchedCost =
            (1.0 - config.minimumMatchScore)
                .coerceIn(0.05, 0.95)
        val forbiddenCost = 1_000_000.0

        val matrix =
            Array(size) {
                DoubleArray(size) {
                    unmatchedCost
                }
            }

        for (row in identities.indices) {
            for (column in tracks.indices) {
                val score =
                    score(
                        identity = identities[row],
                        track = tracks[column],
                        timestampNanos = timestampNanos,
                    )

                matrix[row][column] =
                    if (
                        score.isFinite() &&
                        score >=
                            config.minimumMatchScore
                    ) {
                        1.0 - score
                    } else {
                        forbiddenCost
                    }
            }
        }

        val assignment =
            hungarianMinimum(matrix)

        return assignment.mapIndexedNotNull { identityIndex, trackColumn ->
            if (
                identityIndex >= rows ||
                trackColumn !in tracks.indices
            ) {
                null
            } else {
                val selected =
                    matrix[identityIndex][trackColumn]

                if (
                    selected >= unmatchedCost ||
                    selected >= forbiddenCost
                ) {
                    null
                } else {
                    AssignmentPair(
                        identityId =
                            identities[identityIndex]
                                .stableId,
                        trackId =
                            tracks[trackColumn]
                                .id,
                        score =
                            (
                                1.0 -
                                    selected
                                ).coerceIn(
                                0.0,
                                1.0,
                            ),
                    )
                }
            }
        }
    }

    private fun hungarianMinimum(
        matrix: Array<DoubleArray>,
    ): IntArray {
        val n = matrix.size
        if (n == 0) return IntArray(0)

        val u = DoubleArray(n + 1)
        val v = DoubleArray(n + 1)
        val p = IntArray(n + 1)
        val way = IntArray(n + 1)

        for (row in 1..n) {
            p[0] = row
            var column0 = 0
            val minValue =
                DoubleArray(n + 1) {
                    Double.POSITIVE_INFINITY
                }
            val used =
                BooleanArray(n + 1)

            do {
                used[column0] = true
                val row0 = p[column0]
                var delta = Double.POSITIVE_INFINITY
                var column1 = 0

                for (column in 1..n) {
                    if (used[column]) continue

                    val current =
                        matrix[row0 - 1][column - 1] -
                            u[row0] -
                            v[column]

                    if (
                        current <
                            minValue[column]
                    ) {
                        minValue[column] = current
                        way[column] = column0
                    }

                    if (
                        minValue[column] <
                            delta
                    ) {
                        delta =
                            minValue[column]
                        column1 = column
                    }
                }

                for (column in 0..n) {
                    if (used[column]) {
                        u[p[column]] += delta
                        v[column] -= delta
                    } else {
                        minValue[column] -= delta
                    }
                }

                column0 = column1
            } while (p[column0] != 0)

            do {
                val column1 = way[column0]
                p[column0] = p[column1]
                column0 = column1
            } while (column0 != 0)
        }

        val result =
            IntArray(n) {
                -1
            }

        for (column in 1..n) {
            if (p[column] != 0) {
                result[p[column] - 1] =
                    column - 1
            }
        }

        return result
    }

    private fun score(
        identity: MutableIdentity,
        track: SubjectTrackManager.Track,
        timestampNanos: Long,
    ): Double {
        if (identity.subjectType != track.subjectType) {
            return -1.0
        }

        val dt = (
            timestampNanos -
                identity.lastTimestampNanos
        ).coerceAtLeast(1L) / 1e9

        val predictedX =
            identity.centerX +
                identity.velocityX * dt
        val predictedY =
            identity.centerY +
                identity.velocityY * dt

        val distance = sqrt(
            (predictedX - track.centerX).toDouble() *
                (predictedX - track.centerX).toDouble() +
                (predictedY - track.centerY).toDouble() *
                (predictedY - track.centerY).toDouble(),
        )

        if (distance > config.maximumCenterDistance) {
            return -1.0
        }

        val centerScore =
            1.0 -
                (
                    distance /
                        config.maximumCenterDistance
                ).coerceIn(0.0, 1.0)

        val scaleRatioW =
            ratio(
                identity.width,
                track.width.toDouble(),
            )
        val scaleRatioH =
            ratio(
                identity.height,
                track.height.toDouble(),
            )

        if (
            scaleRatioW > config.maximumScaleRatio ||
            scaleRatioH > config.maximumScaleRatio
        ) {
            return -1.0
        }

        val scaleScore =
            (
                1.0 -
                    (
                        abs(
                            kotlin.math.ln(
                                scaleRatioW,
                            ),
                        ) +
                            abs(
                                kotlin.math.ln(
                                    scaleRatioH,
                                ),
                            )
                    ) * 0.5
            ).coerceIn(0.0, 1.0)

        val velocityDistance =
            sqrt(
                (
                    identity.velocityX -
                        track.velocityX
                ) * (
                    identity.velocityX -
                        track.velocityX
                ) +
                    (
                        identity.velocityY -
                            track.velocityY
                    ) * (
                        identity.velocityY -
                            track.velocityY
                    ),
            )

        val velocityScore =
            expDecay(
                velocityDistance,
                1.4,
            )

        val confidenceScore =
            min(
                identity.confidence,
                track.confidence,
            ).coerceIn(0.0, 1.0)

        val temporalScore =
            (
                1.0 -
                    identity.missedFrames.toDouble() /
                    max(
                        1,
                        config.maximumMissedFrames,
                    )
            ).coerceIn(0.0, 1.0)

        return (
            centerScore * config.geometryWeight +
                scaleScore * config.scaleWeight +
                velocityScore * config.velocityWeight +
                confidenceScore * config.semanticsWeight +
                temporalScore * config.temporalWeight
            ).coerceIn(0.0, 1.0)
    }

    private fun updateIdentity(
        identity: MutableIdentity,
        track: SubjectTrackManager.Track,
        timestampNanos: Long,
        score: Double,
    ) {
        val smoothing =
            (
                0.46 +
                    score * 0.32
            ).coerceIn(0.20, 0.84)

        identity.currentTrackId = track.id
        identity.centerX =
            lerp(
                identity.centerX,
                track.centerX.toDouble(),
                smoothing,
            )
        identity.centerY =
            lerp(
                identity.centerY,
                track.centerY.toDouble(),
                smoothing,
            )
        identity.width =
            lerp(
                identity.width,
                track.width.toDouble(),
                smoothing,
            )
        identity.height =
            lerp(
                identity.height,
                track.height.toDouble(),
                smoothing,
            )
        identity.velocityX =
            lerp(
                identity.velocityX,
                track.velocityX.toDouble(),
                smoothing,
            )
        identity.velocityY =
            lerp(
                identity.velocityY,
                track.velocityY.toDouble(),
                smoothing,
            )
        identity.confidence =
            lerp(
                identity.confidence,
                track.confidence,
                smoothing,
            ).coerceIn(0.0, 1.0)
        identity.ageFrames += 1L
        identity.missedFrames = 0
        identity.lastTimestampNanos = timestampNanos
    }

    private fun snapshot(
        identity: MutableIdentity,
    ): Identity =
        Identity(
            stableId = identity.stableId,
            currentTrackId = identity.currentTrackId,
            subjectType = identity.subjectType,
            centerX = identity.centerX,
            centerY = identity.centerY,
            width = identity.width,
            height = identity.height,
            velocityX = identity.velocityX,
            velocityY = identity.velocityY,
            confidence = identity.confidence,
            ageFrames = identity.ageFrames,
            missedFrames = identity.missedFrames,
            lastTimestampNanos = identity.lastTimestampNanos,
        )

    private fun ratio(
        a: Double,
        b: Double,
    ): Double =
        max(
            a,
            b,
        ) / min(
            max(a, 1e-5),
            max(b, 1e-5),
        )

    private fun expDecay(
        distance: Double,
        scale: Double,
    ): Double =
        kotlin.math.exp(
            -distance /
                max(
                    1e-6,
                    scale,
                ),
        )

    private fun lerp(
        a: Double,
        b: Double,
        t: Double,
    ): Double =
        a + (b - a) * t

    private data class AssignmentPair(
        val identityId: Long,
        val trackId: Long,
        val score: Double,
    )
}
