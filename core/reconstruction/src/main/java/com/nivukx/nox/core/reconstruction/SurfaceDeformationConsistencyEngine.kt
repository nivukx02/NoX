package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min

/**
 * Temporal consistency gate for dense reconstructed surfaces.
 *
 * The engine evaluates a tracked surface in world coordinates rather than treating dense mesh
 * frames as independent snapshots. It measures:
 *  - per-vertex displacement and velocity continuity;
 *  - edge-length strain against the previous frame;
 *  - normal angular change;
 *  - curvature change;
 *  - visible coverage and confidence;
 *  - topology continuity;
 *  - robust P50/P95/P99 residuals.
 *
 * It is deliberately evidence-first. A surface that becomes smoother because the system erased a
 * high-frequency deformation is not considered higher quality. Excessive strain, normal flips,
 * curvature spikes, stale timestamps, or disappearing topology reduce the evidence score.
 */
class SurfaceDeformationConsistencyEngine(
    private var config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumVertices: Int = 120_000,
        val maximumEdges: Int = 240_000,
        val maximumFrameGapSeconds: Double = 0.25,
        val maximumDisplacementMeters: Double = 0.75,
        val warningDisplacementMeters: Double = 0.10,
        val hardStrainRatio: Double = 0.35,
        val warningStrainRatio: Double = 0.12,
        val hardNormalChangeRadians: Double = 1.75,
        val warningNormalChangeRadians: Double = 0.65,
        val hardCurvatureDelta: Double = 2.5,
        val warningCurvatureDelta: Double = 0.8,
        val minimumVisibleCoverage: Double = 0.45,
        val minimumConfidence: Double = 0.35,
        val historyAlpha: Double = 0.22,
        val hardResidualPercentile: Double = 0.95,
        val highQualityConfidence: Double = 0.78,
    ) {
        init {
            require(maximumVertices >= 32)
            require(maximumEdges >= 32)
            require(maximumFrameGapSeconds > 0.0)
            require(maximumDisplacementMeters > 0.0)
            require(warningDisplacementMeters > 0.0)
            require(hardStrainRatio > warningStrainRatio)
            require(hardNormalChangeRadians > warningNormalChangeRadians)
            require(hardCurvatureDelta > warningCurvatureDelta)
            require(minimumVisibleCoverage in 0.0..1.0)
            require(minimumConfidence in 0.0..1.0)
            require(historyAlpha in 0.0..1.0)
            require(hardResidualPercentile in 0.5..1.0)
            require(highQualityConfidence in 0.0..1.0)
        }
    }

    data class Vec3(
        val x: Double,
        val y: Double,
        val z: Double,
    ) {
        init {
            require(x.isFinite() && y.isFinite() && z.isFinite())
        }

        operator fun minus(other: Vec3): Vec3 =
            Vec3(x - other.x, y - other.y, z - other.z)

        operator fun plus(other: Vec3): Vec3 =
            Vec3(x + other.x, y + other.y, z + other.z)

        operator fun times(scale: Double): Vec3 =
            Vec3(x * scale, y * scale, z * scale)

        fun length(): Double =
            kotlin.math.sqrt(x * x + y * y + z * z)

        fun normalized(): Vec3 {
            val length = length()
            return if (length <= 1e-12) {
                Vec3(0.0, 0.0, 1.0)
            } else {
                this * (1.0 / length)
            }
        }

        fun dot(other: Vec3): Double =
            x * other.x + y * other.y + z * other.z

        fun distanceTo(other: Vec3): Double =
            (this - other).length()
    }

    data class Vertex(
        val id: Long,
        val position: Vec3,
        val normal: Vec3,
        val curvature: Double = 0.0,
        val confidence: Double = 1.0,
        val visible: Boolean = true,
    ) {
        init {
            require(id >= 0L)
            require(curvature.isFinite())
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    data class Edge(
        val firstVertexId: Long,
        val secondVertexId: Long,
    ) {
        init {
            require(firstVertexId >= 0L)
            require(secondVertexId >= 0L)
            require(firstVertexId != secondVertexId)
        }
    }

    data class Frame(
        val timestampNanos: Long,
        val vertices: List<Vertex>,
        val edges: List<Edge>,
        val sourceConfidence: Double = 1.0,
    ) {
        init {
            require(timestampNanos > 0L)
            require(sourceConfidence.isFinite() && sourceConfidence in 0.0..1.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val vertexCount: Int,
        val edgeCount: Int,
        val visibleCoverage: Double,
        val vertexConfidence: Double,
        val meanDisplacementMeters: Double,
        val displacementP50Meters: Double,
        val displacementP95Meters: Double,
        val displacementP99Meters: Double,
        val meanVelocityMetersPerSecond: Double,
        val edgeCoverage: Double,
        val strainP50Ratio: Double,
        val strainP95Ratio: Double,
        val strainP99Ratio: Double,
        val normalP95Radians: Double,
        val curvatureP95Delta: Double,
        val topologyContinuity: Double,
        val temporalStability: Double,
        val hardViolation: Boolean,
        val accepted: Boolean,
        val warmedUp: Boolean,
        val staleFrameRejected: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
        val highQualityThreshold: Double = 0.78,
    ) {
        val highQuality: Boolean
            get() =
                accepted &&
                    temporalStability >= highQualityThreshold &&
                    topologyContinuity >= 0.80 &&
                    vertexConfidence >= highQualityThreshold

        val geometryTrust: Double
            get() = minOf(
                temporalStability,
                topologyContinuity,
                vertexConfidence,
                if (hardViolation) 0.0 else 1.0,
            )
    }

    private data class History(
        val timestampNanos: Long,
        val vertices: Map<Long, Vertex>,
        val edges: Set<EdgeKey>,
        val confidence: Double,
        val temporalStability: Double,
    )

    private data class EdgeKey(
        val first: Long,
        val second: Long,
    ) {
        companion object {
            fun of(edge: Edge): EdgeKey =
                if (edge.firstVertexId < edge.secondVertexId) {
                    EdgeKey(edge.firstVertexId, edge.secondVertexId)
                } else {
                    EdgeKey(edge.secondVertexId, edge.firstVertexId)
                }
        }
    }

    private var history: History? = null

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig
        history = null
    }

    fun configuration(): Config = config

    @Synchronized
    fun evaluate(frame: Frame): Result {
        val vertices = frame.vertices
            .asSequence()
            .take(config.maximumVertices)
            .associateBy { it.id }

        val edges = frame.edges
            .asSequence()
            .take(config.maximumEdges)
            .map(EdgeKey::of)
            .filter {
                vertices.containsKey(it.first) &&
                    vertices.containsKey(it.second)
            }
            .toSet()

        val previous = history
        if (previous == null) {
            history = History(
                timestampNanos = frame.timestampNanos,
                vertices = vertices,
                edges = edges,
                confidence = frame.sourceConfidence,
                temporalStability = 0.65 * frame.sourceConfidence,
            )

            val coverage = visibleCoverage(vertices.values)
            val confidence = visibleConfidence(vertices.values)
            return Result(
                timestampNanos = frame.timestampNanos,
                vertexCount = vertices.size,
                edgeCount = edges.size,
                visibleCoverage = coverage,
                vertexConfidence = confidence,
                meanDisplacementMeters = 0.0,
                displacementP50Meters = 0.0,
                displacementP95Meters = 0.0,
                displacementP99Meters = 0.0,
                meanVelocityMetersPerSecond = 0.0,
                edgeCoverage = 0.0,
                strainP50Ratio = 0.0,
                strainP95Ratio = 0.0,
                strainP99Ratio = 0.0,
                normalP95Radians = 0.0,
                curvatureP95Delta = 0.0,
                topologyContinuity = 0.0,
                temporalStability = 0.65 * frame.sourceConfidence,
                hardViolation = false,
                accepted = vertices.isNotEmpty(),
                warmedUp = false,
                staleFrameRejected = false,
                warnings = listOf("surface-temporal-warmup"),
                recommendations = listOf(
                    "Keep the next dense surface frame to establish deformation evidence.",
                ),
                highQualityThreshold = config.highQualityConfidence,
            )
        }

        val dtSeconds =
            (frame.timestampNanos - previous.timestampNanos) / 1e9

        if (
            dtSeconds <= 0.0 ||
            dtSeconds > config.maximumFrameGapSeconds
        ) {
            history = History(
                timestampNanos = frame.timestampNanos,
                vertices = vertices,
                edges = edges,
                confidence = frame.sourceConfidence,
                temporalStability = 0.50 * frame.sourceConfidence,
            )
            return Result(
                timestampNanos = frame.timestampNanos,
                vertexCount = vertices.size,
                edgeCount = edges.size,
                visibleCoverage = visibleCoverage(vertices.values),
                vertexConfidence = visibleConfidence(vertices.values),
                meanDisplacementMeters = 0.0,
                displacementP50Meters = 0.0,
                displacementP95Meters = 0.0,
                displacementP99Meters = 0.0,
                meanVelocityMetersPerSecond = 0.0,
                edgeCoverage = 0.0,
                strainP50Ratio = 0.0,
                strainP95Ratio = 0.0,
                strainP99Ratio = 0.0,
                normalP95Radians = 0.0,
                curvatureP95Delta = 0.0,
                topologyContinuity = 0.0,
                temporalStability = 0.50 * frame.sourceConfidence,
                hardViolation = false,
                accepted = false,
                warmedUp = false,
                staleFrameRejected = true,
                warnings = listOf(
                    "surface-temporal-frame-gap",
                    "surface-temporal-history-reset",
                ),
                recommendations = listOf(
                    "Treat the current surface as a new temporal segment.",
                ),
                highQualityThreshold = config.highQualityConfidence,
            )
        }

        val displacement = ArrayList<Double>()
        val velocity = ArrayList<Double>()
        val normalDelta = ArrayList<Double>()
        val curvatureDelta = ArrayList<Double>()

        vertices.forEach { (id, current) ->
            val prior = previous.vertices[id] ?: return@forEach
            if (!current.visible || !prior.visible) return@forEach

            val distance = current.position.distanceTo(prior.position)
            if (distance.isFinite() && distance <= config.maximumDisplacementMeters) {
                displacement += distance
                velocity += distance / dtSeconds.coerceAtLeast(1e-6)

                normalDelta += angularDistance(
                    prior.normal,
                    current.normal,
                )
                curvatureDelta += abs(
                    current.curvature - prior.curvature,
                )
            }
        }

        val validVertexPairs = displacement.size
        val visibleCoverage = visibleCoverage(vertices.values)
        val vertexConfidence = visibleConfidence(vertices.values)

        val sharedEdges = edges.intersect(previous.edges)
        val edgeStrains = ArrayList<Double>()

        sharedEdges.forEach { edge ->
            val currentA = vertices[edge.first] ?: return@forEach
            val currentB = vertices[edge.second] ?: return@forEach
            val previousA = previous.vertices[edge.first] ?: return@forEach
            val previousB = previous.vertices[edge.second] ?: return@forEach

            val previousLength =
                previousA.position.distanceTo(previousB.position)
            val currentLength =
                currentA.position.distanceTo(currentB.position)

            if (
                previousLength > 1e-5 &&
                currentLength.isFinite()
            ) {
                val strain =
                    abs(currentLength - previousLength) /
                        previousLength
                if (strain.isFinite()) edgeStrains += strain
            }
        }

        val topologyContinuity =
            (
                sharedEdges.size.toDouble() /
                    min(
                        max(previous.edges.size, 1),
                        max(edges.size, 1),
                    )
            ).coerceIn(0.0, 1.0)

        val displacementSorted = displacement.sorted()
        val strainSorted = edgeStrains.sorted()
        val normalSorted = normalDelta.sorted()
        val curvatureSorted = curvatureDelta.sorted()

        val displacementP50 = percentile(
            displacementSorted,
            0.50,
            0.0,
        )
        val displacementP95 = percentile(
            displacementSorted,
            config.hardResidualPercentile,
            config.maximumDisplacementMeters,
        )
        val displacementP99 = percentile(
            displacementSorted,
            0.99,
            config.maximumDisplacementMeters,
        )

        val strainP50 = percentile(
            strainSorted,
            0.50,
            config.hardStrainRatio,
        )
        val strainP95 = percentile(
            strainSorted,
            config.hardResidualPercentile,
            config.hardStrainRatio,
        )
        val strainP99 = percentile(
            strainSorted,
            0.99,
            config.hardStrainRatio,
        )

        val normalP95 = percentile(
            normalSorted,
            config.hardResidualPercentile,
            config.hardNormalChangeRadians,
        )
        val curvatureP95 = percentile(
            curvatureSorted,
            config.hardResidualPercentile,
            config.hardCurvatureDelta,
        )

        val displacementQuality = expDecay(
            displacementP95,
            config.warningDisplacementMeters,
        )
        val strainQuality = expDecay(
            strainP95,
            config.warningStrainRatio,
        )
        val normalQuality = expDecay(
            normalP95,
            config.warningNormalChangeRadians,
        )
        val curvatureQuality = expDecay(
            curvatureP95,
            config.warningCurvatureDelta,
        )
        val coverageQuality =
            (
                visibleCoverage /
                    config.minimumVisibleCoverage.coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)

        val currentRawStability = (
            displacementQuality * 0.18 +
                strainQuality * 0.24 +
                normalQuality * 0.16 +
                curvatureQuality * 0.12 +
                topologyContinuity * 0.14 +
                coverageQuality * 0.08 +
                vertexConfidence * 0.08
            ).coerceIn(0.0, 1.0)

        val temporalStability = (
            previous.temporalStability * (1.0 - config.historyAlpha) +
                currentRawStability * config.historyAlpha
            ).coerceIn(0.0, 1.0)

        val hardViolation =
            displacementP95 >= config.maximumDisplacementMeters ||
                strainP95 >= config.hardStrainRatio ||
                normalP95 >= config.hardNormalChangeRadians ||
                curvatureP95 >= config.hardCurvatureDelta ||
                visibleCoverage < config.minimumVisibleCoverage

        val accepted =
            validVertexPairs >= 8 &&
                edgeStrains.size >= 8 &&
                visibleCoverage >= config.minimumVisibleCoverage &&
                vertexConfidence >= config.minimumConfidence &&
                topologyContinuity >= 0.50 &&
                !hardViolation

        val warnings = buildList {
            if (validVertexPairs < 8) add("surface-vertex-correspondence-low")
            if (edgeStrains.size < 8) add("surface-edge-correspondence-low")
            if (visibleCoverage < config.minimumVisibleCoverage) {
                add("surface-visible-coverage-low")
            }
            if (vertexConfidence < config.minimumConfidence) {
                add("surface-confidence-low")
            }
            if (displacementP95 >= config.warningDisplacementMeters) {
                add("surface-displacement-p95-elevated")
            }
            if (strainP95 >= config.warningStrainRatio) {
                add("surface-strain-p95-elevated")
            }
            if (strainP95 >= config.hardStrainRatio) {
                add("surface-strain-hard-failure")
            }
            if (normalP95 >= config.warningNormalChangeRadians) {
                add("surface-normal-change-elevated")
            }
            if (normalP95 >= config.hardNormalChangeRadians) {
                add("surface-normal-hard-failure")
            }
            if (curvatureP95 >= config.warningCurvatureDelta) {
                add("surface-curvature-change-elevated")
            }
            if (curvatureP95 >= config.hardCurvatureDelta) {
                add("surface-curvature-hard-failure")
            }
            if (topologyContinuity < 0.72) {
                add("surface-topology-continuity-degraded")
            }
            if (hardViolation) {
                add("surface-hard-geometry-violation")
            }
        }.distinct()

        val recommendations = buildList {
            if (visibleCoverage < config.minimumVisibleCoverage) {
                add(
                    "Reduce the fusion weight of unseen regions rather than filling topology from a stale frame.",
                )
            }
            if (strainP95 >= config.warningStrainRatio) {
                add(
                    "Use deformation-aware surface weighting and preserve localized strain instead of global smoothing.",
                )
            }
            if (normalP95 >= config.warningNormalChangeRadians) {
                add(
                    "Lower temporal surface blending near normal discontinuities and retain view-consistent detail.",
                )
            }
            if (curvatureP95 >= config.warningCurvatureDelta) {
                add(
                    "Prefer fresh multi-view evidence for rapidly changing high-curvature regions.",
                )
            }
            if (topologyContinuity < 0.72) {
                add(
                    "Re-establish cross-view correspondence before accepting topology-changing regions.",
                )
            }
        }.distinct().take(6)

        history = History(
            timestampNanos = frame.timestampNanos,
            vertices = vertices,
            edges = edges,
            confidence = frame.sourceConfidence,
            temporalStability = temporalStability,
        )

        return Result(
            timestampNanos = frame.timestampNanos,
            vertexCount = vertices.size,
            edgeCount = edges.size,
            visibleCoverage = visibleCoverage,
            vertexConfidence = vertexConfidence,
            meanDisplacementMeters =
                displacement.averageOrZero(),
            displacementP50Meters = displacementP50,
            displacementP95Meters = displacementP95,
            displacementP99Meters = displacementP99,
            meanVelocityMetersPerSecond =
                velocity.averageOrZero(),
            edgeCoverage =
                (
                    sharedEdges.size.toDouble() /
                        max(edges.size, 1)
                ).coerceIn(0.0, 1.0),
            strainP50Ratio = strainP50,
            strainP95Ratio = strainP95,
            strainP99Ratio = strainP99,
            normalP95Radians = normalP95,
            curvatureP95Delta = curvatureP95,
            topologyContinuity = topologyContinuity,
            temporalStability = temporalStability,
            hardViolation = hardViolation,
            accepted = accepted,
            warmedUp = true,
            staleFrameRejected = false,
            warnings = warnings,
            recommendations = recommendations,
            highQualityThreshold = config.highQualityConfidence,
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    @Synchronized
    override fun close() {
        history = null
    }

    private fun visibleCoverage(
        vertices: Collection<Vertex>,
    ): Double {
        if (vertices.isEmpty()) return 0.0
        return vertices.count { it.visible }.toDouble() /
            vertices.size
    }

    private fun visibleConfidence(
        vertices: Collection<Vertex>,
    ): Double {
        val visible = vertices.filter { it.visible }
        if (visible.isEmpty()) return 0.0
        return visible.map { it.confidence }.average()
    }

    private fun angularDistance(
        first: Vec3,
        second: Vec3,
    ): Double {
        val a = first.normalized()
        val b = second.normalized()
        val cosine = a.dot(b)
            .coerceIn(-1.0, 1.0)
        return acos(cosine)
    }

    private fun percentile(
        sorted: List<Double>,
        probability: Double,
        emptyValue: Double,
    ): Double {
        if (sorted.isEmpty()) return emptyValue
        val index =
            (
                probability.coerceIn(0.0, 1.0) *
                    sorted.lastIndex
                ).toInt().coerceIn(
                    0,
                    sorted.lastIndex,
                )
        return sorted[index]
    }

    private fun expDecay(
        value: Double,
        scale: Double,
    ): Double =
        kotlin.math.exp(
            -value /
                scale.coerceAtLeast(1e-9),
        ).coerceIn(0.0, 1.0)

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
