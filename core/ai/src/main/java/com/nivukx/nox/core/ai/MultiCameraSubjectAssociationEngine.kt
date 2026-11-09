
package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sqrt
import java.util.PriorityQueue

/**
 * Cross-camera subject association for synchronized RGB capture.
 *
 * Camera-local trackers maintain per-view identities. This layer links observations into a rig-level
 * identity without storing RGB appearance embeddings. When calibrated 3D positions are available,
 * geometry is the dominant cross-view cue. Without calibration, the engine deliberately avoids
 * comparing raw pixel coordinates from unrelated cameras and falls back to timing, semantics,
 * scale, motion and detector confidence.
 */
class MultiCameraSubjectAssociationEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumScore: Double = 0.46,
        val semanticWeight: Double = 0.22,
        val geometryWeight: Double = 0.30,
        val scaleWeight: Double = 0.14,
        val motionWeight: Double = 0.18,
        val confidenceWeight: Double = 0.08,
        val timeWeight: Double = 0.08,
        val maximumClockSkewMillis: Double = 8.0,
        val maximumGeometryDistanceMeters: Double = 2.5,
        val maximumScaleRatio: Double = 3.5,
        val maximumSubjects: Int = 64,
        val maximumAssociationObservations: Int = 8192,
        val maximumGraphEdges: Int = 16384,
        val maximumCandidateNeighborsPerObservation: Int = 12,
    ) {
        init {
            require(minimumScore in 0.0..1.0)
            require(
                semanticWeight + geometryWeight + scaleWeight +
                    motionWeight + confidenceWeight + timeWeight > 0.0,
            )
            require(maximumClockSkewMillis > 0.0)
            require(maximumGeometryDistanceMeters > 0.0)
            require(maximumScaleRatio > 1.0)
            require(maximumSubjects in 1..64)
            require(maximumAssociationObservations >= maximumSubjects)
            require(maximumGraphEdges > 0)
            require(maximumCandidateNeighborsPerObservation in 2..64)
        }
    }

    data class Observation(
        val deviceId: DeviceId,
        val localTrackId: Long,
        val subjectType: SubjectType,
        val bounds: Bounds,
        val velocity: Vec2,
        val confidence: Double,
        val timestampNanos: Long,
        val calibratedCenterMeters: Vec2? = null,
        val calibratedDepthMeters: Double? = null,
    ) {
        init {
            require(confidence in 0.0..1.0)
            if (calibratedDepthMeters != null) {
                require(calibratedDepthMeters > 0.0)
            }
        }
    }

    data class Bounds(
        val left: Double,
        val top: Double,
        val right: Double,
        val bottom: Double,
    ) {
        val width: Double
            get() = (right - left).coerceAtLeast(0.0)

        val height: Double
            get() = (bottom - top).coerceAtLeast(0.0)

        val center: Vec2
            get() = Vec2(
                (left + right) * 0.5,
                (top + bottom) * 0.5,
            )
    }

    data class RigSubject(
        val rigTrackId: Long,
        val observations: List<Observation>,
        val subjectType: SubjectType,
        val fusedConfidence: Double,
        val synchronized: Boolean,
        val spreadMeters: Double?,
        val depthConsistency: Double,
        val diagnostics: List<String>,
        val higherOrderConsensusScore: Double = 1.0,
        val higherOrderAmbiguityRisk: Double = 0.0,
        val identityContinuationScore: Double = 0.0,
        val identityContinuationMargin: Double = 0.0,
        val identityAssignmentAccepted: Boolean = false,
    )

    data class Result(
        val timestampNanos: Long,
        val subjects: List<RigSubject>,
        val unassignedObservations: List<Observation>,
        val associationCount: Int,
        val diagnostics: List<String>,
    )

    private data class SpatialCell(
        val x: Long,
        val y: Long,
        val z: Long,
    )

    private val lastSubjects = LinkedHashMap<Long, RigSubject>()
    private val higherOrderConsensusEngine =
        MultiViewActorHypothesisConsensusEngine()
    private val globalRigIdentityAssignmentEngine =
        GlobalRigIdentityAssignmentEngine()
    private var nextRigTrackId = 1L
    private var lastAssociationTimestampNanos: Long = Long.MIN_VALUE

    @Synchronized
    fun associate(
        observations: List<Observation>,
        timestampNanos: Long,
    ): Result {
        require(timestampNanos >= 0L)

        if (lastAssociationTimestampNanos != Long.MIN_VALUE &&
            timestampNanos <= lastAssociationTimestampNanos
        ) {
            return Result(
                timestampNanos = lastAssociationTimestampNanos,
                subjects = lastSubjects.values.toList(),
                unassignedObservations = observations,
                associationCount = 0,
                diagnostics = listOf(
                    "stale-rig-association-frame-rejected",
                ),
            )
        }

        lastAssociationTimestampNanos = timestampNanos

        if (observations.isEmpty()) {
            return Result(
                timestampNanos = timestampNanos,
                subjects = emptyList(),
                unassignedObservations = emptyList(),
                associationCount = 0,
                diagnostics = listOf("no observations"),
            )
        }

        val boundedObservations =
            observations
                .sortedByDescending { it.confidence }
                .take(config.maximumAssociationObservations)

        val devices = boundedObservations
            .groupBy { it.deviceId }
            .toSortedMap(compareBy { it.value })

        val components =
            buildGraphComponents(boundedObservations)

        val used = HashSet<String>()
        val subjects = mutableListOf<RigSubject>()

        components
            .sortedWith(
                compareByDescending<List<Observation>> { it.size }
                    .thenByDescending {
                        it.maxOfOrNull { observation ->
                            observation.confidence
                        } ?: 0.0
                    },
            )
            .take(config.maximumSubjects)
            .forEach { members ->
                if (members.isEmpty()) return@forEach

                members.forEach { used += key(it) }

                val type = dominantType(members)
                val spread = calibratedSpread(members)
                val depth = depthConsistency(members)
                val synchronized = members.all {
                    abs(
                        it.timestampNanos -
                            members.first().timestampNanos,
                    ) <=
                        config.maximumClockSkewMillis * 1e6
                }

                val viewCoverage = (
                    members.map { it.deviceId }
                        .distinct()
                        .size.toDouble() /
                        devices.size.toDouble().coerceAtLeast(1.0)
                    ).coerceIn(0.0, 1.0)

                val higherOrder =
                    higherOrderConsensusEngine.evaluate(members)

                val fusedConfidence = (
                    (
                        members.map { it.confidence }.average() * 0.62 +
                            viewCoverage * 0.20 +
                            depth * 0.10 +
                            if (synchronized) 0.08 else 0.0
                    ) *
                        (
                            0.72 +
                                0.28 *
                                    higherOrder.consensusScore
                        )
                    ).coerceIn(0.0, 1.0)

                subjects += RigSubject(
                    rigTrackId = 0L,
                    observations = members,
                    subjectType = type,
                    fusedConfidence = fusedConfidence,
                    synchronized = synchronized,
                    spreadMeters = spread,
                    depthConsistency = depth,
                    diagnostics =
                        subjectDiagnostics(
                            members = members,
                            synchronized = synchronized,
                            spread = spread,
                            depthConsistency = depth,
                            coverage = viewCoverage,
                        ) +
                            higherOrder.warnings,
                    higherOrderConsensusScore =
                        higherOrder.consensusScore,
                    higherOrderAmbiguityRisk =
                        higherOrder.ambiguityRisk,
                )
            }

        val unassigned = boundedObservations.filter {
            !used.contains(key(it))
        }

        val identityAssignment =
            globalRigIdentityAssignmentEngine.assign(
                current =
                    subjects.map {
                        GlobalRigIdentityAssignmentEngine.Track(
                            rigTrackId = 0L,
                            subjectType = it.subjectType,
                            center =
                                aggregateIdentityCenter(
                                    it.observations,
                                ),
                            centerIsMetric =
                                it.observations.any {
                                    observation ->
                                    observation.calibratedCenterMeters !=
                                        null
                                },
                            velocity =
                                aggregateIdentityVelocity(
                                    it.observations,
                                ),
                            confidence =
                                it.fusedConfidence,
                            observationCount =
                                it.observations.size,
                            localTrackIds =
                                it.observations.associate {
                                    observation ->
                                    observation.deviceId.value to
                                        observation.localTrackId
                                },
                        )
                    },
                previous =
                    lastSubjects.values.map {
                        GlobalRigIdentityAssignmentEngine.Track(
                            rigTrackId = it.rigTrackId,
                            subjectType = it.subjectType,
                            center =
                                aggregateIdentityCenter(
                                    it.observations,
                                ),
                            centerIsMetric =
                                it.observations.any {
                                    observation ->
                                    observation.calibratedCenterMeters !=
                                        null
                                },
                            velocity =
                                aggregateIdentityVelocity(
                                    it.observations,
                                ),
                            confidence =
                                it.fusedConfidence,
                            observationCount =
                                it.observations.size,
                            localTrackIds =
                                it.observations.associate {
                                    observation ->
                                    it.deviceId.value to
                                        it.localTrackId
                                },
                        )
                    },
            )

        val identityByCurrentIndex =
            identityAssignment.assignments
                .associateBy { it.currentIndex }

        val identifiedSubjects =
            subjects.mapIndexed { index, subject ->
                val assignment =
                    identityByCurrentIndex[index]
                val rigTrackId =
                    assignment?.rigTrackId
                        ?: nextRigTrackId++

                subject.copy(
                    rigTrackId = rigTrackId,
                    identityContinuationScore =
                        assignment?.score ?: 0.0,
                    identityContinuationMargin =
                        assignment?.margin ?: 0.0,
                    identityAssignmentAccepted =
                        assignment != null,
                    diagnostics =
                        subject.diagnostics +
                            listOf(
                                "identityContinuationScore=" +
                                    format(
                                        assignment?.score
                                            ?: 0.0,
                                    ),
                                "identityContinuationMargin=" +
                                    format(
                                        assignment?.margin
                                            ?: 0.0,
                                    ),
                            ),
                )
            }

        val finalSubjects =
            identifiedSubjects
                .sortedByDescending { it.fusedConfidence }
                .take(config.maximumSubjects)

        lastSubjects.clear()
        finalSubjects.forEach {
            lastSubjects[it.rigTrackId] = it
        }

        return Result(
            timestampNanos = timestampNanos,
            subjects = finalSubjects,
            unassignedObservations = unassigned,
            associationCount = finalSubjects.sumOf {
                (it.observations.size - 1).coerceAtLeast(0)
            },
            diagnostics = listOf(
                "deviceViews=" + devices.size,
                "observations=" + observations.size,
                "rigSubjects=" + finalSubjects.size,
                "associations=" + finalSubjects.sumOf {
                    (it.observations.size - 1).coerceAtLeast(0)
                },
                "unassigned=" + unassigned.size,
                "globalIdentityAssigned=" +
                    identityAssignment.assignments.size,
                "globalIdentityConfidence=" +
                    format(
                        identityAssignment.globalConfidence,
                    ),
                "globalIdentityAmbiguity=" +
                    format(
                        identityAssignment.ambiguityRisk,
                    ),
            )
        )
    }

    @Synchronized
    fun current(): List<RigSubject> =
        lastSubjects.values.toList()

    @Synchronized
    fun reset() {
        lastSubjects.clear()
        nextRigTrackId = 1L
        lastAssociationTimestampNanos = Long.MIN_VALUE
    }

    private fun aggregateIdentityCenter(
        observations: List<Observation>,
    ): Vec2 {
        val metric =
            observations.mapNotNull {
                it.calibratedCenterMeters
            }
        if (metric.isNotEmpty()) {
            return Vec2(
                metric.map { it.x }.average(),
                metric.map { it.y }.average(),
            )
        }

        return Vec2(
            observations.map { it.bounds.center.x }.average(),
            observations.map { it.bounds.center.y }.average(),
        )
    }

    private fun aggregateIdentityVelocity(
        observations: List<Observation>,
    ): Vec2 =
        if (observations.isEmpty()) {
            Vec2(0.0, 0.0)
        } else {
            Vec2(
                observations.map { it.velocity.x }.average(),
                observations.map { it.velocity.y }.average(),
            )
        }

    private fun buildGraphComponents(
        observations: List<Observation>,
    ): List<List<Observation>> {
        if (observations.isEmpty()) return emptyList()

        val parent = IntArray(observations.size) { it }
        val rank = IntArray(observations.size)
        val devicesByRoot =
            observations
                .map { mutableSetOf(it.deviceId) }
                .toMutableList()

        fun find(index: Int): Int {
            var cursor = index
            while (parent[cursor] != cursor) {
                parent[cursor] = parent[parent[cursor]]
                cursor = parent[cursor]
            }
            return cursor
        }

        fun union(a: Int, b: Int): Boolean {
            var ra = find(a)
            var rb = find(b)
            if (ra == rb) return true

            val firstDevices = devicesByRoot[ra]
            val secondDevices = devicesByRoot[rb]
            if (firstDevices.any { it in secondDevices }) return false

            if (rank[ra] < rank[rb]) {
                val tmp = ra
                ra = rb
                rb = tmp
            }

            parent[rb] = ra
            if (rank[ra] == rank[rb]) rank[ra] += 1
            firstDevices += secondDevices
            return true
        }

        val indicesByDevice =
            observations
                .indices
                .groupBy { observations[it].deviceId }
                .mapValues { (_, indices) ->
                    indices.sortedBy { observations[it].timestampNanos }
                }

        val spatialCellSize =
            max(
                config.maximumGeometryDistanceMeters * 0.50,
                0.10,
            )

        fun spatialKey(observation: Observation): SpatialCell? {
            val point = observation.calibratedCenterMeters
                ?: return null
            val gx = floor(point.x / spatialCellSize).toLong()
            val gy = floor(point.y / spatialCellSize).toLong()
            val gz =
                observation.calibratedDepthMeters
                    ?.let { floor(it / spatialCellSize).toLong() }
                    ?: Long.MIN_VALUE
            return SpatialCell(gx, gy, gz)
        }

        val spatialBuckets =
            HashMap<SpatialCell, MutableList<Int>>()
        observations.indices.forEach { index ->
            spatialKey(observations[index])?.let { key ->
                spatialBuckets
                    .getOrPut(key) { mutableListOf() }
                    .add(index)
            }
        }

        fun lowerBound(
            indices: List<Int>,
            targetTimestampNanos: Long,
        ): Int {
            var low = 0
            var high = indices.size
            while (low < high) {
                val middle = (low + high) ushr 1
                if (
                    observations[indices[middle]].timestampNanos <
                        targetTimestampNanos
                ) {
                    low = middle + 1
                } else {
                    high = middle
                }
            }
            return low
        }

        fun candidateIndices(firstIndex: Int): List<Int> {
            val first = observations[firstIndex]
            val selected = HashSet<Int>()
            val spatialPoint = first.calibratedCenterMeters

            if (spatialPoint != null) {
                val baseX = floor(spatialPoint.x / spatialCellSize).toLong()
                val baseY = floor(spatialPoint.y / spatialCellSize).toLong()
                val baseZ =
                    first.calibratedDepthMeters
                        ?.let {
                            floor(it / spatialCellSize).toLong()
                        }

                for (dx in -1L..1L) {
                    for (dy in -1L..1L) {
                        if (baseZ != null) {
                            for (dz in -1L..1L) {
                                val key =
                                    SpatialCell(
                                        x = baseX + dx,
                                        y = baseY + dy,
                                        z = baseZ + dz,
                                    )
                                spatialBuckets[key]
                                    .orEmpty()
                                    .forEach { candidate ->
                                        if (candidate > firstIndex) {
                                            selected += candidate
                                        }
                                    }
                            }
                        } else {
                            val key =
                                SpatialCell(
                                    x = baseX + dx,
                                    y = baseY + dy,
                                    z = Long.MIN_VALUE,
                                )
                            spatialBuckets[key]
                                .orEmpty()
                                .forEach { candidate ->
                                    if (candidate > firstIndex) {
                                        selected += candidate
                                    }
                                }
                        }
                    }
                }
            }

            // Every other camera still gets a bounded temporal neighborhood. This keeps
            // uncalibrated fallback association useful without reverting to quadratic scans.
            indicesByDevice.forEach { (deviceId, indices) ->
                if (deviceId == first.deviceId || indices.isEmpty()) return@forEach
                val insertion = lowerBound(indices, first.timestampNanos)
                val begin = max(0, insertion - 2)
                val finish = min(indices.size, insertion + 3)
                for (position in begin until finish) {
                    val candidate = indices[position]
                    if (candidate > firstIndex) selected += candidate
                }
            }

            return selected.toList()
        }

        val edgeQueue =
            PriorityQueue<Triple<Int, Int, Double>>(
                config.maximumGraphEdges + 1,
                compareBy { it.third },
            )

        observations.indices.forEach { firstIndex ->
            val first = observations[firstIndex]
            val candidates =
                candidateIndices(firstIndex)

            val localQueue =
                PriorityQueue<Pair<Int, Double>>(
                    config.maximumCandidateNeighborsPerObservation + 1,
                    compareBy { it.second },
                )

            candidates.forEach { secondIndex ->
                val second = observations[secondIndex]
                val timestampGapMillis =
                    abs(
                        first.timestampNanos -
                            second.timestampNanos,
                    ).toDouble() / 1e6
                if (
                    timestampGapMillis >
                        config.maximumClockSkewMillis
                ) return@forEach

                val firstMetric = first.calibratedCenterMeters
                val secondMetric = second.calibratedCenterMeters
                if (firstMetric != null && secondMetric != null) {
                    val distance =
                        calibratedMetricDistance(
                            first,
                            second,
                        )
                    if (
                        !distance.isFinite() ||
                        distance > config.maximumGeometryDistanceMeters
                    ) return@forEach
                }

                val score =
                    associationScore(first, second)
                if (score < config.minimumScore || !score.isFinite()) {
                    return@forEach
                }

                if (localQueue.size < config.maximumCandidateNeighborsPerObservation) {
                    localQueue.add(secondIndex to score)
                } else if (score > localQueue.peek().second) {
                    localQueue.poll()
                    localQueue.add(secondIndex to score)
                }
            }

            localQueue.forEach { (secondIndex, score) ->
                val edge = Triple(firstIndex, secondIndex, score)
                if (edgeQueue.size < config.maximumGraphEdges) {
                    edgeQueue.add(edge)
                } else if (score > edgeQueue.peek().third) {
                    edgeQueue.poll()
                    edgeQueue.add(edge)
                }
            }
        }

        edgeQueue
            .toList()
            .sortedByDescending { it.third }
            .forEach { (first, second, _) ->
                union(first, second)
            }

        return observations.indices
            .groupBy { find(it) }
            .values
            .map { indices ->
                indices.map(observations::get)
            }
    }
    private fun associationScore(
        a: Observation,
        b: Observation,
    ): Double {
        if (a.subjectType != b.subjectType) return 0.0

        val timeScore = expDecay(
            abs(
                a.timestampNanos -
                    b.timestampNanos,
            ).toDouble() / 1e6,
            config.maximumClockSkewMillis,
        )

        val geometry = calibratedGeometryScore(
            a,
            b,
        )

        val scaleRatio = ratio(
            a.bounds.width.coerceAtLeast(1.0),
            b.bounds.width.coerceAtLeast(1.0),
        )

        if (scaleRatio > config.maximumScaleRatio) {
            return 0.0
        }

        val scaleScore = (
            1.0 -
                abs(kotlin.math.ln(scaleRatio)) /
                    kotlin.math.ln(config.maximumScaleRatio)
            ).coerceIn(0.0, 1.0)

        val motionScore = expDecay(
            (a.velocity - b.velocity).norm(),
            1.0,
        )

        val confidenceScore = min(
            a.confidence,
            b.confidence,
        ).coerceIn(0.0, 1.0)

        val semanticScore = 1.0

        val uncalibrated =
            a.calibratedCenterMeters == null ||
                b.calibratedCenterMeters == null

        val noCalibrationPrior =
            if (uncalibrated) 0.34 else 0.0

        val rawScore =
            (
                semanticScore * config.semanticWeight +
                    max(
                        geometry,
                        noCalibrationPrior,
                    ) * config.geometryWeight +
                    scaleScore * config.scaleWeight +
                    motionScore * config.motionWeight +
                    confidenceScore * config.confidenceWeight +
                    timeScore * config.timeWeight
                ).coerceIn(0.0, 1.0)

        return if (uncalibrated) {
            // Uncalibrated correspondence remains useful for identity continuity, but it is
            // deliberately prevented from reaching calibrated metric-association confidence.
            min(
                rawScore,
                0.68,
            )
        } else {
            rawScore
        }
    }

    private fun calibratedMetricDistance(
        a: Observation,
        b: Observation,
    ): Double {
        val pa = a.calibratedCenterMeters
            ?: return Double.POSITIVE_INFINITY
        val pb = b.calibratedCenterMeters
            ?: return Double.POSITIVE_INFINITY

        val dx = pa.x - pb.x
        val dy = pa.y - pb.y
        val planarSquared = dx * dx + dy * dy

        val depthDistance =
            if (
                a.calibratedDepthMeters != null &&
                b.calibratedDepthMeters != null
            ) {
                val dz =
                    a.calibratedDepthMeters -
                        b.calibratedDepthMeters
                dz * dz
            } else {
                0.0
            }

        return sqrt(
            planarSquared +
                depthDistance,
        )
    }

    private fun calibratedGeometryScore(
        a: Observation,
        b: Observation,
    ): Double {
        val pa = a.calibratedCenterMeters
            ?: return 0.0
        val pb = b.calibratedCenterMeters
            ?: return 0.0

        val distance =
            calibratedMetricDistance(a, b)

        if (distance > config.maximumGeometryDistanceMeters) {
            return 0.0
        }

        return (
            1.0 -
                distance /
                    config.maximumGeometryDistanceMeters
            ).coerceIn(0.0, 1.0)
    }

    private fun calibratedSpread(
        members: List<Observation>,
    ): Double? {
        val points = members.mapNotNull {
            it.calibratedCenterMeters
        }

        if (points.size < 2) return null

        var maxDistance = 0.0
        for (i in 0 until points.lastIndex) {
            for (j in i + 1 until points.size) {
                maxDistance = max(
                    maxDistance,
                    points[i].distanceTo(points[j]),
                )
            }
        }

        return maxDistance
    }

    private fun depthConsistency(
        members: List<Observation>,
    ): Double {
        val depths = members.mapNotNull {
            it.calibratedDepthMeters
        }

        if (depths.size < 2) return 0.50

        val mean = depths.average().coerceAtLeast(1e-5)
        val deviation = depths
            .map { abs(it - mean) / mean }
            .average()

        return (
            1.0 -
                deviation / 0.35
            ).coerceIn(0.0, 1.0)
    }

    private fun dominantType(
        members: List<Observation>,
    ): SubjectType =
        members
            .groupingBy { it.subjectType }
            .eachCount()
            .maxByOrNull { it.value }
            ?.key
            ?: SubjectType.UNKNOWN

    private fun key(
        observation: Observation,
    ): String =
        observation.deviceId.value +
            ":" +
            observation.localTrackId

    private fun expDecay(
        value: Double,
        scale: Double,
    ): Double =
        kotlin.math.exp(
            -value /
                scale.coerceAtLeast(1e-6),
        ).coerceIn(0.0, 1.0)

    private fun ratio(
        a: Double,
        b: Double,
    ): Double =
        max(a, b) /
            min(
                a.coerceAtLeast(1e-6),
                b.coerceAtLeast(1e-6),
            )

    private fun subjectDiagnostics(
        members: List<Observation>,
        synchronized: Boolean,
        spread: Double?,
        depthConsistency: Double,
        coverage: Double,
    ): List<String> = listOf(
        "views=" + members.size,
        "coverage=" + format(coverage),
        "synchronized=" + synchronized,
        "spreadMeters=" + (
            spread?.let(::format) ?: "unknown"
            ),
        "depthConsistency=" + format(depthConsistency),
        "confidence=" + format(
            members.map { it.confidence }.average(),
        ),
    )

    private fun format(
        value: Double,
    ): String =
        "%.3f".format(
            java.util.Locale.US,
            value.coerceIn(0.0, 1.0),
        )
}
