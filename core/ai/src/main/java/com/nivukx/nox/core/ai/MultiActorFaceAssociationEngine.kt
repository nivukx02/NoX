package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.Vec2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Associates face detections with global body actors.
 *
 * Face detectors and body trackers maintain separate identity namespaces. This engine creates an
 * explicit mapping using calibrated image geometry, containment, scale agreement and detector
 * continuity. Ambiguous associations are rejected instead of merging facial geometry into an
 * arbitrary actor.
 */
class MultiActorFaceAssociationEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val trackingContinuityWeight: Double = 0.10,
        val maximumTrackingContinuityAgeNanos: Long = 500_000_000L,
        val maximumRememberedFaces: Int = 128,
    ) {
        init {
            require(trackingContinuityWeight in 0.0..0.30)
            require(maximumTrackingContinuityAgeNanos > 0L)
            require(maximumRememberedFaces in 16..512)
        }
    }


    data class Actor(
        val actorId: Long,
        val center: Vec2,
        val boundsLeft: Double,
        val boundsTop: Double,
        val boundsRight: Double,
        val boundsBottom: Double,
        val confidence: Double,
    )

    data class Assignment(
        val actorId: Long?,
        val faceTrackingId: Int?,
        val score: Double,
        val accepted: Boolean,
        val reason: String,
    )

    data class Result(
        val assignments: List<Assignment>,
        val unassignedFaces: Int,
        val warnings: List<String>,
    )

    fun associate(
        actors: List<Actor>,
        faces: List<FaceFrame>,
        minimumScore: Double = 0.50,
    ): Result {
        require(minimumScore in 0.0..1.0)

        val latestTimestamp =
            faces.maxOfOrNull { it.timestampNanos } ?: 0L
        val trackingIterator = trackingHistory.entries.iterator()
        while (trackingIterator.hasNext()) {
            val memory = trackingIterator.next().value
            if (
                latestTimestamp >= memory.first &&
                latestTimestamp - memory.first >
                    config.maximumTrackingContinuityAgeNanos
            ) {
                trackingIterator.remove()
            }
        }

        if (faces.isEmpty()) {
            return Result(
                assignments = emptyList(),
                unassignedFaces = 0,
                warnings = emptyList(),
            )
        }

        if (actors.isEmpty()) {
            return Result(
                assignments = faces.map { face ->
                    Assignment(
                        actorId = null,
                        faceTrackingId = face.trackingId,
                        score = 0.0,
                        accepted = false,
                        reason = "NO_ACTOR_TRACKS_AVAILABLE",
                    )
                },
                unassignedFaces = faces.size,
                warnings =
                    listOf("NO_ACTOR_TRACKS_AVAILABLE_FOR_FACE_ASSOCIATION"),
            )
        }

        val scores =
            Array(faces.size) {
                DoubleArray(actors.size)
            }

        faces.forEachIndexed { faceIndex, face ->
            val faceCenter =
                Vec2(
                    (
                        face.boundsLeft +
                            face.boundsRight
                        ) * 0.5,
                    (
                        face.boundsTop +
                            face.boundsBottom
                        ) * 0.5,
                )
            val faceWidth =
                (
                    face.boundsRight -
                        face.boundsLeft
                    ).toDouble().coerceAtLeast(1.0)
            val faceHeight =
                (
                    face.boundsBottom -
                        face.boundsTop
                    ).toDouble().coerceAtLeast(1.0)
            val faceArea =
                faceWidth * faceHeight
            val faceConfidence =
                face.points
                    .map { it.confidence.toDouble() }
                    .averageOrZero()

            actors.forEachIndexed { actorIndex, actor ->
                val actorWidth =
                    (
                        actor.boundsRight -
                            actor.boundsLeft
                        ).coerceAtLeast(1e-6)
                val actorHeight =
                    (
                        actor.boundsBottom -
                            actor.boundsTop
                        ).coerceAtLeast(1e-6)
                val actorArea =
                    actorWidth * actorHeight

                val inside =
                    faceCenter.x in
                        actor.boundsLeft..actor.boundsRight &&
                        faceCenter.y in
                        actor.boundsTop..actor.boundsBottom

                val distance =
                    hypot(
                        faceCenter.x - actor.center.x,
                        faceCenter.y - actor.center.y,
                    )

                val actorScale =
                    hypot(actorWidth, actorHeight)
                        .coerceAtLeast(1e-6)

                val spatial =
                    (
                        1.0 -
                            distance / (actorScale * 0.75)
                    ).coerceIn(0.0, 1.0)

                val areaRatio =
                    min(
                        faceArea / actorArea.coerceAtLeast(1e-6),
                        actorArea.coerceAtLeast(1e-6) / faceArea,
                    ).coerceIn(0.0, 1.0)

                val trackingBonus =
                    if (face.trackingId != null) {
                        val previous =
                            trackingHistory[face.trackingId]
                        if (
                            previous != null &&
                            previous.second == actor.actorId &&
                            face.timestampNanos >= previous.first &&
                            face.timestampNanos - previous.first <=
                                config.maximumTrackingContinuityAgeNanos
                        ) {
                            config.trackingContinuityWeight
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }

                scores[faceIndex][actorIndex] =
                    (
                        (if (inside) 1.0 else 0.0) * 0.36 +
                            spatial * 0.27 +
                            areaRatio * 0.17 +
                            faceConfidence * 0.08 +
                            actor.confidence * 0.04 +
                            trackingBonus
                    ).coerceIn(0.0, 1.0)
            }
        }

        val assignment =
            hungarianMaximum(scores)
        val usedFaces = HashSet<Int>()
        val usedActors = HashSet<Long>()
        val assignments =
            ArrayList<Assignment>(faces.size)

        assignment.forEachIndexed { faceIndex, actorIndex ->
            if (
                faceIndex !in faces.indices ||
                actorIndex !in actors.indices
            ) {
                return@forEachIndexed
            }

            val score = scores[faceIndex][actorIndex]
            if (score < minimumScore) {
                return@forEachIndexed
            }

            val face = faces[faceIndex]
            val actor = actors[actorIndex]
            val continuityUsed =
                trackingContinuityWasStrong(
                    face = face,
                    actor = actor,
                )

            usedFaces += faceIndex
            usedActors += actor.actorId

            face.trackingId?.let {
                trackingHistory[it] =
                    face.timestampNanos to actor.actorId
            }

            assignments +=
                Assignment(
                    actorId = actor.actorId,
                    faceTrackingId = face.trackingId,
                    score = score,
                    accepted = true,
                    reason =
                        if (continuityUsed) {
                            "FACE_TO_ACTOR_GLOBAL_ASSIGNMENT_WITH_TRACKING_CONTINUITY"
                        } else {
                            "FACE_TO_ACTOR_GLOBAL_ASSIGNMENT"
                        },
                )
        }

        while (trackingHistory.size > config.maximumRememberedFaces) {
            trackingHistory.remove(trackingHistory.keys.first())
        }

        faces.forEachIndexed { index, face ->
            if (index !in usedFaces) {
                assignments +=
                    Assignment(
                        actorId = null,
                        faceTrackingId = face.trackingId,
                        score = 0.0,
                        accepted = false,
                        reason = "FACE_ACTOR_ASSOCIATION_AMBIGUOUS",
                    )
            }
        }

        return Result(
            assignments = assignments,
            unassignedFaces =
                faces.size - usedFaces.size,
            warnings =
                buildList {
                    if (assignments.any { !it.accepted }) {
                        add("ONE_OR_MORE_FACES_LEFT_UNASSIGNED")
                    }
                    if (faces.size > actors.size) {
                        add("FACE_COUNT_EXCEEDS_ACTOR_TRACK_CAPACITY")
                    }
                }.distinct(),
        )
    }

    private fun trackingContinuityWasStrong(
        face: FaceFrame,
        actor: Actor,
    ): Boolean {
        val trackingId = face.trackingId ?: return false
        val previous = trackingHistory[trackingId] ?: return false
        return previous.second == actor.actorId &&
            face.timestampNanos >= previous.first &&
            face.timestampNanos - previous.first <=
                config.maximumTrackingContinuityAgeNanos
    }

    private fun hungarianMaximum(
        scores: Array<DoubleArray>,
    ): IntArray {
        val rows = scores.size
        val cols =
            scores.maxOfOrNull { it.size } ?: 0
        if (rows == 0 || cols == 0) return IntArray(0)

        val n = max(rows, cols)
        val cost =
            Array(n) {
                DoubleArray(n) { 1.0 }
            }

        for (row in 0 until rows) {
            for (col in 0 until scores[row].size) {
                cost[row][col] =
                    1.0 - scores[row][col]
            }
        }

        val u = DoubleArray(n + 1)
        val v = DoubleArray(n + 1)
        val p = IntArray(n + 1)
        val way = IntArray(n + 1)

        for (row in 1..n) {
            p[0] = row
            var column0 = 0
            val minv = DoubleArray(n + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(n + 1)

            do {
                used[column0] = true
                val row0 = p[column0]
                var delta = Double.POSITIVE_INFINITY
                var column1 = 0

                for (column in 1..n) {
                    if (used[column]) continue
                    val current =
                        cost[row0 - 1][column - 1] -
                            u[row0] -
                            v[column]

                    if (current < minv[column]) {
                        minv[column] = current
                        way[column] = column0
                    }

                    if (minv[column] < delta) {
                        delta = minv[column]
                        column1 = column
                    }
                }

                for (column in 0..n) {
                    if (used[column]) {
                        u[p[column]] += delta
                        v[column] -= delta
                    } else {
                        minv[column] -= delta
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

        val result = IntArray(rows) { -1 }
        for (column in 1..n) {
            val row = p[column] - 1
            if (row in 0 until rows) {
                val assignedColumn = column - 1
                if (assignedColumn in 0 until cols) {
                    result[row] = assignedColumn
                }
            }
        }
        return result
    }

    private val trackingHistory =
        LinkedHashMap<Int, Pair<Long, Long>>()

    fun clear() {
        trackingHistory.clear()
    }

    private fun List<com.nivukx.nox.core.model.FaceMeshPoint>.averageOrZero():
        Double =
        if (isEmpty()) 0.0 else averageOfConfidence()

    private fun List<com.nivukx.nox.core.model.FaceMeshPoint>.averageOfConfidence():
        Double {
        var sum = 0.0
        for (point in this) {
            sum += point.confidence
        }
        return (
            sum /
                max(
                    1,
                    size,
                )
            ).coerceIn(
                0.0,
                1.0,
            )
    }
}
