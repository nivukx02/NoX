package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Canonical-surface temporal prior for high-end 4D reconstruction.
 *
 * This engine validates a current surface against a trusted canonical/reference surface without
 * forcing vertex identity. Correspondence is geometric and bounded, then the resulting mapping is
 * evaluated through displacement, local edge stretch, surface-normal agreement and topology
 * support.
 *
 * The engine is intentionally non-destructive:
 * - it never mutates either mesh;
 * - it never creates replacement vertices;
 * - unmatched canonical regions remain unmatched;
 * - a rejected deformation is reported as evidence, not silently repaired;
 * - deterministic sampling makes replay decisions stable.
 *
 * The implementation is designed for offline/high-quality refinement where an O(N*M) nearest
 * correspondence over a bounded sample count is acceptable and preferable to introducing another
 * learned model dependency.
 */
class CanonicalSurfaceTemporalPriorEngine(
    private var config: Config = Config(),
) {

    data class Config(
        val maximumCorrespondenceDistanceMeters: Double = 0.20,
        val maximumSamples: Int = 1024,
        val edgeStretchWarning: Double = 0.18,
        val edgeStretchReject: Double = 0.40,
        val normalAngleWarningDegrees: Double = 28.0,
        val normalAngleRejectDegrees: Double = 60.0,
        val correspondenceWarning: Double = 0.72,
        val correspondenceReject: Double = 0.48,
        val displacementWarningMeters: Double = 0.08,
        val displacementRejectMeters: Double = 0.20,
        val topologySupportWarning: Double = 0.72,
        val topologySupportReject: Double = 0.50,
        val historyAlpha: Double = 0.20,
        val minimumConfidenceForAcceptance: Double = 0.58,
        val temporalDisplacementWarningMeters: Double = 0.08,
        val temporalDisplacementRejectMeters: Double = 0.22,
        val temporalConsistencyWeight: Double = 0.12,
        val normalFlowWeight: Double = 0.18,
        val normalFlowWarningScore: Double = 0.54,
    ) {
        init {
            require(maximumCorrespondenceDistanceMeters > 0.0)
            require(maximumSamples >= 8)
            require(edgeStretchWarning > 0.0)
            require(edgeStretchReject > edgeStretchWarning)
            require(normalAngleWarningDegrees > 0.0)
            require(normalAngleRejectDegrees > normalAngleWarningDegrees)
            require(correspondenceWarning in 0.0..1.0)
            require(
                correspondenceReject in 0.0..correspondenceWarning,
            )
            require(displacementWarningMeters > 0.0)
            require(displacementRejectMeters > displacementWarningMeters)
            require(topologySupportWarning in 0.0..1.0)
            require(
                topologySupportReject in 0.0..topologySupportWarning,
            )
            require(historyAlpha in 0.0..1.0)
            require(minimumConfidenceForAcceptance in 0.0..1.0)
            require(temporalDisplacementWarningMeters > 0.0)
            require(temporalDisplacementRejectMeters > temporalDisplacementWarningMeters)
            require(temporalConsistencyWeight >= 0.0)
            require(normalFlowWeight in 0.0..1.0)
            require(normalFlowWarningScore in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val canonicalMesh: ReconstructionMesh,
        val currentMesh: ReconstructionMesh,
        val upstreamConfidence: Double,
    ) {
        init {
            require(timestampNanos > 0L)
            require(upstreamConfidence in 0.0..1.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val accepted: Boolean,
        val confidence: Double,
        val correspondenceCoverage: Double,
        val meanDisplacementMeters: Double,
        val p95DisplacementMeters: Double,
        val edgeStretchScore: Double,
        val normalAgreement: Double,
        val topologySupport: Double,
        val temporalDeformationConsistency: Double = 1.0,
        val meanTemporalDisplacementMeters: Double = 0.0,
        val p95TemporalDisplacementMeters: Double = 0.0,
        val normalFlowConsistency: Double = 1.0,
        val matchedVertexCount: Int,
        val sampledVertexCount: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class SurfaceStatistics(
        val positions: List<Vec3>,
        val normals: List<Vec3>,
        val triangles: List<MeshTriangle>,
        val edgeLengths: Map<Long, Double>,
    )

    private data class Correspondence(
        val canonicalIndex: Int,
        val currentIndex: Int,
        val distanceMeters: Double,
    )

    private data class History(
        var timestampNanos: Long,
        var confidenceEma: Double,
        var acceptedFrames: Int,
        var rejectedFrames: Int,
        var canonicalToCurrent: Map<Int, Vec3> = emptyMap(),
        var canonicalToCurrentNormal: Map<Int, Vec3> = emptyMap(),
    )

    private var history: History? = null
    private var lastResult: Result? = null

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig
        history?.let {
            it.confidenceEma =
                it.confidenceEma.coerceIn(0.0, 1.0)
        }
    }

    fun configuration(): Config = config

    @Synchronized
    fun evaluate(input: Input): Result {
        lastResult?.let { previous ->
            if (input.timestampNanos <= previous.timestampNanos) {
                return previous.copy(
                    warnings =
                        (
                            previous.warnings +
                                "stale-surface-frame-rejected"
                            ).distinct(),
                )
            }
        }

        if (
            input.canonicalMesh.vertices.isEmpty() ||
            input.canonicalMesh.triangles.isEmpty() ||
            input.currentMesh.vertices.isEmpty() ||
            input.currentMesh.triangles.isEmpty()
        ) {
            return rejected(
                timestampNanos = input.timestampNanos,
                reason = "surface-empty",
                recommendation =
                    "Preserve the last trusted surface until canonical and current geometry are both populated.",
                upstreamConfidence = input.upstreamConfidence,
            )
        }

        val canonical = statistics(input.canonicalMesh)
        val current = statistics(input.currentMesh)

        val sampleIndices =
            deterministicSampleIndices(
                vertexCount = canonical.positions.size,
                maximumSamples = config.maximumSamples,
            )

        val correspondences =
            findCorrespondences(
                canonicalPositions = canonical.positions,
                currentPositions = current.positions,
                sampleIndices = sampleIndices,
            )

        val coverage =
            correspondences.size.toDouble() /
                sampleIndices.size.coerceAtLeast(1)

        if (correspondences.isEmpty()) {
            val nearestDistances =
                nearestDisplacements(
                    canonicalPositions = canonical.positions,
                    currentPositions = current.positions,
                    sampleIndices = sampleIndices,
                )
            return rejected(
                timestampNanos = input.timestampNanos,
                reason = "correspondence-coverage-too-low-temporal-displacement-elevated",
                recommendation =
                    "Preserve the prior surface and increase synchronized viewpoint coverage before temporal refinement.",
                upstreamConfidence = input.upstreamConfidence,
                sampledVertexCount = sampleIndices.size,
                diagnosticMeanDisplacementMeters = nearestDistances.averageOrZero(),
                diagnosticP95DisplacementMeters =
                    percentile(nearestDistances, 0.95),
            )
        }

        val displacements =
            correspondences
                .map { it.distanceMeters }
                .sorted()

        val meanDisplacement =
            displacements.averageOrZero()

        val p95Displacement =
            percentile(
                displacements,
                0.95,
            )

        val edgeStretchScore =
            edgeStretchScore(
                canonical = canonical,
                current = current,
                correspondences = correspondences,
            )

        val normalAgreement =
            normalAgreement(
                canonical = canonical,
                current = current,
                correspondences = correspondences,
            )

        val topologySupport =
            topologySupport(
                canonical = canonical,
                current = current,
                correspondences = correspondences,
                sampleCount = sampleIndices.size,
            )

        val temporalDeformation =
            temporalDeformationConsistency(
                canonical = canonical,
                current = current,
                correspondences = correspondences,
            )

        val normalFlow =
            normalFlowConsistency(
                current = current,
                correspondences = correspondences,
            )

        val displacementScore =
            (
                exp(
                    -meanDisplacement /
                        config.displacementWarningMeters
                            .coerceAtLeast(1e-6),
                ) * 0.58 +
                    exp(
                        -p95Displacement /
                            config.displacementRejectMeters
                                .coerceAtLeast(1e-6),
                    ) * 0.42
                ).coerceIn(0.0, 1.0)

        val correspondenceScore =
            thresholdScore(
                value = coverage,
                warning = config.correspondenceWarning,
                reject = config.correspondenceReject,
                invert = false,
            )

        val overall =
            (
                correspondenceScore * 0.24 +
                    displacementScore * 0.16 +
                    edgeStretchScore * 0.25 +
                    normalAgreement * 0.19 +
                    topologySupport * 0.10 +
                    minOf(
                temporalDeformation.consistency,
                exp(
                    -p95Displacement /
                        config.temporalDisplacementRejectMeters
                            .coerceAtLeast(1e-6),
                ).coerceIn(0.0, 1.0),
            ) * 0.12 +
                    normalFlow * config.normalFlowWeight * 0.08 +
                    input.upstreamConfidence *
                        (0.06 - config.temporalConsistencyWeight.coerceAtMost(0.06))
                ).coerceIn(0.0, 1.0)

        val severeDisplacement =
            p95Displacement >=
                config.displacementRejectMeters

        val severeCorrespondence =
            coverage <
                config.correspondenceReject

        val severeEdgeStretch =
            edgeStretchScore < 0.42

        val severeNormal =
            normalAgreement < 0.48

        val severeTopology =
            topologySupport <
                config.topologySupportReject

        val severeTemporal =
            temporalDeformation.p95Meters >=
                config.temporalDisplacementRejectMeters

        val acceptedNow =
            !severeDisplacement &&
                !severeCorrespondence &&
                !severeEdgeStretch &&
                !severeNormal &&
                !severeTopology &&
                !severeTemporal &&
                overall >=
                    config.minimumConfidenceForAcceptance

        val smoothed =
            updateHistory(
                timestampNanos = input.timestampNanos,
                confidence = overall,
                acceptedNow = acceptedNow,
            )

        val accepted =
            acceptedNow &&
                smoothed >=
                    config.minimumConfidenceForAcceptance

        if (accepted) {
            updateCanonicalHistory(
                correspondences = correspondences,
                current = current,
            )
        }

        val warnings = buildList {
            if (coverage < config.correspondenceWarning) {
                add("canonical-correspondence-coverage-low")
            }
            if (
                meanDisplacement >
                    config.displacementWarningMeters
            ) {
                add("canonical-deformation-displacement-elevated")
            }
            if (
                p95Displacement >
                    config.displacementWarningMeters
            ) {
                add("canonical-deformation-tail-elevated")
            }
            if (
                edgeStretchScore <
                    1.0 -
                    config.edgeStretchWarning
            ) {
                add("canonical-edge-stretch-out-of-range")
            }
            if (
                normalAgreement <
                    normalAgreementWarningThreshold()
            ) {
                add("canonical-normal-disagreement")
            }
            if (
                topologySupport <
                    config.topologySupportWarning
            ) {
                add("canonical-topology-support-degraded")
            }
            if (
                temporalDeformation.meanMeters >
                    config.temporalDisplacementWarningMeters
            ) {
                add("canonical-temporal-deformation-elevated")
            }
            if (
                temporalDeformation.p95Meters >
                    config.temporalDisplacementWarningMeters
            ) {
                add("canonical-temporal-deformation-tail-elevated")
            }
            if (normalFlow < config.normalFlowWarningScore) {
                add("canonical-sdf-flow-consistency-degraded")
            }
            if (
                !accepted &&
                (
                    coverage < config.correspondenceReject ||
                        p95Displacement >
                            config.temporalDisplacementWarningMeters ||
                        temporalDeformation.p95Meters >
                            config.temporalDisplacementWarningMeters
                )
            ) {
                add("canonical-temporal-deformation-rejected")
            }
            if (history?.rejectedFrames ?: 0 >= 4) {
                add("canonical-surface-persistent-rejection")
            }
        }.distinct()

        val recommendations = buildRecommendations(
            coverage = coverage,
            meanDisplacement = meanDisplacement,
            p95Displacement = p95Displacement,
            edgeStretchScore = edgeStretchScore,
            normalAgreement = normalAgreement,
            topologySupport = topologySupport,
            accepted = accepted,
        )

        val result = Result(
            timestampNanos = input.timestampNanos,
            accepted = accepted,
            confidence = smoothed,
            correspondenceCoverage = coverage,
            meanDisplacementMeters = meanDisplacement,
            p95DisplacementMeters = p95Displacement,
            edgeStretchScore = edgeStretchScore,
            normalAgreement = normalAgreement,
            topologySupport = topologySupport,
            temporalDeformationConsistency =
                temporalDeformation.consistency,
            normalFlowConsistency =
                normalFlow,
            meanTemporalDisplacementMeters =
                maxOf(
                    temporalDeformation.meanMeters,
                    meanDisplacement * 0.35,
                ),
            p95TemporalDisplacementMeters =
                maxOf(
                    temporalDeformation.p95Meters,
                    p95Displacement,
                ),
            matchedVertexCount = correspondences.size,
            sampledVertexCount = sampleIndices.size,
            warnings = warnings,
            recommendations = recommendations,
        )

        lastResult = result
        return result
    }

    @Synchronized
    fun reset() {
        history = null
        lastResult = null
    }

    private fun statistics(
        mesh: ReconstructionMesh,
    ): SurfaceStatistics {
        val positions =
            mesh.vertices.map {
                finite(it.position)
            }

        val normals =
            vertexNormals(
                positions = positions,
                triangles = mesh.triangles,
            )

        val edges =
            edgeLengths(
                positions = positions,
                triangles = mesh.triangles,
            )

        return SurfaceStatistics(
            positions = positions,
            normals = normals,
            triangles = mesh.triangles,
            edgeLengths = edges,
        )
    }

    private fun deterministicSampleIndices(
        vertexCount: Int,
        maximumSamples: Int,
    ): List<Int> {
        if (vertexCount <= maximumSamples) {
            return (0 until vertexCount).toList()
        }

        val step =
            vertexCount.toDouble() /
                maximumSamples.toDouble()

        return (0 until maximumSamples).map { sample ->
            min(
                vertexCount - 1,
                (sample * step).toInt(),
            )
        }.distinct()
    }

    private fun findCorrespondences(
        canonicalPositions: List<Vec3>,
        currentPositions: List<Vec3>,
        sampleIndices: List<Int>,
    ): List<Correspondence> {
        val maximumDistanceSquared =
            config.maximumCorrespondenceDistanceMeters *
                config.maximumCorrespondenceDistanceMeters

        val result =
            ArrayList<Correspondence>(
                sampleIndices.size,
            )

        sampleIndices.forEach { canonicalIndex ->
            val canonicalPoint =
                canonicalPositions[canonicalIndex]

            var bestIndex = -1
            var bestDistanceSquared =
                Double.POSITIVE_INFINITY

            currentPositions.forEachIndexed {
                    currentIndex,
                    currentPoint,
                ->
                val dx =
                    canonicalPoint.x -
                        currentPoint.x
                val dy =
                    canonicalPoint.y -
                        currentPoint.y
                val dz =
                    canonicalPoint.z -
                        currentPoint.z

                val distanceSquared =
                    dx * dx +
                        dy * dy +
                        dz * dz

                if (
                    distanceSquared <
                        bestDistanceSquared
                ) {
                    bestDistanceSquared =
                        distanceSquared
                    bestIndex = currentIndex
                }
            }

            if (
                bestIndex >= 0 &&
                bestDistanceSquared <=
                    maximumDistanceSquared
            ) {
                result += Correspondence(
                    canonicalIndex = canonicalIndex,
                    currentIndex = bestIndex,
                    distanceMeters =
                        sqrt(
                            bestDistanceSquared.coerceAtLeast(
                                0.0,
                            ),
                        ),
                )
            }
        }

        return result
    }

    private fun nearestDisplacements(
        canonicalPositions: List<Vec3>,
        currentPositions: List<Vec3>,
        sampleIndices: List<Int>,
    ): List<Double> =
        sampleIndices.mapNotNull { index ->
            canonicalPositions.getOrNull(index)?.let { source ->
                currentPositions.minOfOrNull { target ->
                    source.distanceTo(target)
                }
            }
        }.filter { it.isFinite() }

    private fun edgeStretchScore(
        canonical: SurfaceStatistics,
        current: SurfaceStatistics,
        correspondences: List<Correspondence>,
    ): Double {
        val map =
            correspondences.associate {
                it.canonicalIndex to
                    it.currentIndex
            }

        val ratios =
            ArrayList<Double>()

        canonical.triangles.forEach { triangle ->
            val currentA =
                map[triangle.a]
                    ?: return@forEach
            val currentB =
                map[triangle.b]
                    ?: return@forEach
            val currentC =
                map[triangle.c]
                    ?: return@forEach

            val pairs =
                listOf(
                    triangle.a to triangle.b,
                    triangle.b to triangle.c,
                    triangle.c to triangle.a,
                )

            pairs.forEach { (a, b) ->
                val canonicalLength =
                    canonical.edgeLengths[edgeKey(a, b)]
                        ?: return@forEach

                val currentLength =
                    current.edgeLengths[
                        edgeKey(
                            if (a == triangle.a) {
                                currentA
                            } else if (a == triangle.b) {
                                currentB
                            } else {
                                currentC
                            },
                            if (b == triangle.a) {
                                currentA
                            } else if (b == triangle.b) {
                                currentB
                            } else {
                                currentC
                            },
                        ),
                    ]
                        ?: return@forEach

                if (
                    canonicalLength <= 1e-8 ||
                    !currentLength.isFinite()
                ) {
                    return@forEach
                }

                val ratio =
                    currentLength /
                        canonicalLength

                ratios +=
                    abs(ratio - 1.0)
            }
        }

        if (ratios.isEmpty()) {
            val coverage =
                correspondences.size.toDouble() /
                    canonical.positions.size.coerceAtLeast(1)
            return (
                coverage * 0.50
            ).coerceIn(0.0, 0.62)
        }

        val p50 =
            percentile(
                ratios.sorted(),
                0.50,
            )
        val p95 =
            percentile(
                ratios.sorted(),
                0.95,
            )

        val penalty =
            p50.coerceAtMost(1.0) * 0.56 +
                p95.coerceAtMost(1.0) * 0.44

        return thresholdScore(
            value = penalty,
            warning = config.edgeStretchWarning,
            reject = config.edgeStretchReject,
            invert = false,
        ).coerceIn(0.0, 1.0)
    }

    private fun normalAgreement(
        canonical: SurfaceStatistics,
        current: SurfaceStatistics,
        correspondences: List<Correspondence>,
    ): Double {
        val values =
            correspondences.mapNotNull {
                val canonicalNormal =
                    canonical.normals.getOrNull(
                        it.canonicalIndex,
                    )
                val currentNormal =
                    current.normals.getOrNull(
                        it.currentIndex,
                    )

                if (
                    canonicalNormal == null ||
                    currentNormal == null
                ) {
                    null
                } else if (
                    canonicalNormal.norm() < 1e-8 ||
                    currentNormal.norm() < 1e-8
                ) {
                    null
                } else {
                    (
                        (
                            1.0 +
                                canonicalNormal
                                    .normalized()
                                    .dot(
                                        currentNormal.normalized(),
                                    )
                            ) * 0.5
                        ).coerceIn(0.0, 1.0)
                }
            }

        return values.averageOrZero().coerceIn(0.0, 1.0)
    }

    private fun topologySupport(
        canonical: SurfaceStatistics,
        current: SurfaceStatistics,
        correspondences: List<Correspondence>,
        sampleCount: Int,
    ): Double {
        val correspondenceCoverage =
            correspondences.size.toDouble() /
                sampleCount.coerceAtLeast(1)

        val triangleRatio =
            (
                1.0 -
                    abs(
                        canonical.triangles.size -
                            current.triangles.size,
                    ).toDouble() /
                    max(
                        1,
                        max(
                            canonical.triangles.size,
                            current.triangles.size,
                        ),
                    ).toDouble()
                ).coerceIn(0.0, 1.0)

        val mappedCanonicalEdges =
            canonical.edgeLengths.keys.count { edge ->
                val canonicalA =
                    (edge ushr 32).toInt()
                val canonicalB =
                    edge.toInt()
                correspondenceIndex(
                    canonicalA,
                    correspondences,
                ) != null &&
                    correspondenceIndex(
                        canonicalB,
                        correspondences,
                    ) != null
            }.toDouble()

        val canonicalEdgeCount =
            canonical.edgeLengths.size
                .coerceAtLeast(1)

        val mappedEdgeRatio =
            (
                mappedCanonicalEdges /
                    canonicalEdgeCount.toDouble()
                ).coerceIn(0.0, 1.0)

        return (
            correspondenceCoverage * 0.44 +
                triangleRatio * 0.24 +
                mappedEdgeRatio * 0.32
            ).coerceIn(0.0, 1.0)
    }

    private fun correspondenceIndex(
        canonicalIndex: Int,
        correspondences: List<Correspondence>,
    ): Int? =
        correspondences
            .firstOrNull {
                it.canonicalIndex ==
                    canonicalIndex
            }
            ?.currentIndex

    private fun normalAgreementWarningThreshold(): Double =
        (
            (
                1.0 +
                    kotlin.math.cos(
                        Math.toRadians(
                            config.normalAngleWarningDegrees,
                        ),
                    )
                ) * 0.5
            ).coerceIn(0.0, 1.0)

    private fun buildRecommendations(
        coverage: Double,
        meanDisplacement: Double,
        p95Displacement: Double,
        edgeStretchScore: Double,
        normalAgreement: Double,
        topologySupport: Double,
        accepted: Boolean,
    ): List<String> = buildList {
        if (
            coverage <
                config.correspondenceWarning
        ) {
            add(
                "Preserve additional synchronized views before using this surface as a temporal canonical prior.",
            )
        }

        if (
            meanDisplacement >
                config.displacementWarningMeters
        ) {
            add(
                "Interpret the deformation as potentially geometric or calibration-driven until the next stable surface observation.",
            )
        }

        if (
            p95Displacement >
                config.displacementRejectMeters
        ) {
            add(
                "Hold the current surface and avoid propagating large unmatched deformation into later frames.",
            )
        }

        if (
            edgeStretchScore <
                1.0 -
                    config.edgeStretchWarning
        ) {
            add(
                "Apply local deformation refinement before trusting metric surface detail.",
            )
        }

        if (
            normalAgreement <
                normalAgreementWarningThreshold()
        ) {
            add(
                "Check view-dependent surface normals and camera pose consistency.",
            )
        }

        if (
            topologySupport <
                config.topologySupportWarning
        ) {
            add(
                "Keep topology stable across adjacent frames; avoid merging nearby articulated surfaces.",
            )
        }

        if (!accepted) {
            add(
                "Do not use rejected correspondences as canonical geometry evidence.",
            )
        }
    }.distinct().take(8)

    private fun updateHistory(
        timestampNanos: Long,
        confidence: Double,
        acceptedNow: Boolean,
    ): Double {
        val previous =
            history

        if (previous == null) {
            history =
                History(
                    timestampNanos = timestampNanos,
                    confidenceEma =
                        confidence.coerceIn(0.0, 1.0),
                    acceptedFrames =
                        if (acceptedNow) 1 else 0,
                    rejectedFrames =
                        if (acceptedNow) 0 else 1,
                )
            return confidence.coerceIn(0.0, 1.0)
        }

        val alpha =
            config.historyAlpha.coerceIn(0.0, 1.0)

        previous.timestampNanos =
            timestampNanos
        previous.confidenceEma =
            previous.confidenceEma *
                (1.0 - alpha) +
                confidence.coerceIn(0.0, 1.0) *
                alpha

        if (acceptedNow) {
            previous.acceptedFrames += 1
            previous.rejectedFrames =
                max(
                    0,
                    previous.rejectedFrames - 1,
                )
        } else {
            previous.rejectedFrames += 1
            previous.acceptedFrames =
                max(
                    0,
                    previous.acceptedFrames - 1,
                )
        }

        previous.canonicalToCurrent =
            previous.canonicalToCurrent
                .toMap()
        previous.canonicalToCurrentNormal =
            previous.canonicalToCurrentNormal
                .toMap()
        return previous.confidenceEma
    }

    private data class TemporalDeformation(
        val consistency: Double,
        val meanMeters: Double,
        val p95Meters: Double,
    )

    private fun temporalDeformationConsistency(
        canonical: SurfaceStatistics,
        current: SurfaceStatistics,
        correspondences: List<Correspondence>,
    ): TemporalDeformation {
        val previous = history?.canonicalToCurrent
        if (previous.isNullOrEmpty()) {
            return TemporalDeformation(
                consistency = 1.0,
                meanMeters = 0.0,
                p95Meters = 0.0,
            )
        }

        val displacements =
            correspondences.mapNotNull { correspondence ->
                val previousPoint =
                    previous[correspondence.canonicalIndex]
                        ?: return@mapNotNull null
                val currentPoint =
                    current.positions.getOrNull(
                        correspondence.currentIndex,
                    ) ?: return@mapNotNull null

                previousPoint.distanceTo(currentPoint)
            }.filter { it.isFinite() }

        if (displacements.isEmpty()) {
            return TemporalDeformation(
                consistency = 0.35,
                meanMeters = 0.0,
                p95Meters = 0.0,
            )
        }

        val mean = displacements.average()
        val p95 = percentile(displacements, 0.95)

        val previousCentroid =
            previous.values
                .takeIf { it.isNotEmpty() }
                ?.let { values ->
                    values.reduce { a, b -> a + b } /
                        values.size.toDouble()
                }

        val currentCentroid =
            correspondences
                .mapNotNull { correspondence ->
                    current.positions.getOrNull(
                        correspondence.currentIndex,
                    )
                }
                .takeIf { it.isNotEmpty() }
                ?.let { values ->
                    values.reduce { a, b -> a + b } /
                        values.size.toDouble()
                }

        val centroidDisplacement =
            if (
                previousCentroid != null &&
                currentCentroid != null
            ) {
                previousCentroid.distanceTo(currentCentroid)
            } else {
                0.0
            }

        val temporalP95 =
            maxOf(
                p95,
                centroidDisplacement,
            )

        val meanScore =
            exp(
                -mean /
                    config.temporalDisplacementWarningMeters
                        .coerceAtLeast(1e-6),
            )
        val tailScore =
            exp(
                -temporalP95 /
                    config.temporalDisplacementRejectMeters
                        .coerceAtLeast(1e-6),
            )

        return TemporalDeformation(
            consistency =
                (
                    meanScore * 0.58 +
                        tailScore * 0.42
                    ).coerceIn(0.0, 1.0),
            meanMeters = mean,
            p95Meters = temporalP95,
        )
    }

    private fun normalFlowConsistency(
        current: SurfaceStatistics,
        correspondences: List<Correspondence>,
    ): Double {
        val previousPositions =
            history?.canonicalToCurrent.orEmpty()
        val previousNormals =
            history?.canonicalToCurrentNormal.orEmpty()

        if (
            previousPositions.isEmpty() ||
            previousNormals.isEmpty()
        ) {
            return 1.0
        }

        val values =
            correspondences.mapNotNull { correspondence ->
                val previousPosition =
                    previousPositions[correspondence.canonicalIndex]
                        ?: return@mapNotNull null
                val previousNormal =
                    previousNormals[correspondence.canonicalIndex]
                        ?: return@mapNotNull null
                val currentPosition =
                    current.positions.getOrNull(
                        correspondence.currentIndex,
                    ) ?: return@mapNotNull null
                val currentNormal =
                    current.normals.getOrNull(
                        correspondence.currentIndex,
                    ) ?: return@mapNotNull null

                val flow =
                    currentPosition -
                        previousPosition
                val flowMagnitude = flow.norm()

                if (
                    !flowMagnitude.isFinite() ||
                    flowMagnitude < 1e-8
                ) {
                    return@mapNotNull 1.0
                }

                val flowDirection =
                    flow /
                        flowMagnitude

                val normalFlow =
                    abs(
                        flowDirection.dot(
                            previousNormal.normalized(),
                        ),
                    ).coerceIn(0.0, 1.0)

                val normalAgreement =
                    (
                        1.0 +
                            previousNormal
                                .normalized()
                                .dot(
                                    currentNormal.normalized(),
                                )
                    ) * 0.5

                (
                    normalFlow * 0.65 +
                        normalAgreement.coerceIn(0.0, 1.0) * 0.35
                    ).coerceIn(0.0, 1.0)
            }

        return values.averageOrZero().coerceIn(0.0, 1.0)
    }

    private fun updateCanonicalHistory(
        correspondences: List<Correspondence>,
        current: SurfaceStatistics,
    ) {
        val mapping =
            correspondences.associateNotNull { correspondence ->
                val position =
                    current.positions.getOrNull(
                        correspondence.currentIndex,
                    ) ?: return@associateNotNull null
                correspondence.canonicalIndex to position
            }

        val normalMapping =
            correspondences.associateNotNull { correspondence ->
                val normal =
                    current.normals.getOrNull(
                        correspondence.currentIndex,
                    ) ?: return@associateNotNull null
                correspondence.canonicalIndex to normal
            }

        val state = history
        if (state == null) {
            history =
                History(
                    timestampNanos = 0L,
                    confidenceEma = 1.0,
                    acceptedFrames = 0,
                    rejectedFrames = 0,
                    canonicalToCurrent = mapping,
                    canonicalToCurrentNormal = normalMapping,
                )
        } else {
            state.canonicalToCurrent = mapping
            state.canonicalToCurrentNormal = normalMapping
        }
    }

    private fun <A, K, V> Iterable<A>.associateNotNull(
        transform: (A) -> Pair<K, V>?,
    ): Map<K, V> =
        buildMap {
            for (item in this@associateNotNull) {
                val pair = transform(item)
                if (pair != null) put(pair.first, pair.second)
            }
        }

    private fun thresholdScore(
        value: Double,
        warning: Double,
        reject: Double,
        invert: Boolean,
    ): Double {
        val bounded =
            value.coerceAtLeast(0.0)

        val normalized =
            if (bounded <= warning) {
                1.0
            } else if (bounded >= reject) {
                0.0
            } else {
                1.0 -
                    (
                        (bounded - warning) /
                            (reject - warning)
                    ).coerceIn(0.0, 1.0)
            }

        return if (invert) {
            1.0 - normalized
        } else {
            normalized
        }
    }

    private fun rejected(
        timestampNanos: Long,
        reason: String,
        recommendation: String,
        upstreamConfidence: Double,
        sampledVertexCount: Int = 0,
        diagnosticMeanDisplacementMeters: Double = 0.0,
        diagnosticP95DisplacementMeters: Double = 0.0,
    ): Result {
        updateHistory(
            timestampNanos = timestampNanos,
            confidence = 0.0,
            acceptedNow = false,
        )

        return Result(
            timestampNanos = timestampNanos,
            accepted = false,
            confidence = (
                history?.confidenceEma ?: 0.0
            ),
            correspondenceCoverage = 0.0,
            meanDisplacementMeters = diagnosticMeanDisplacementMeters,
            p95DisplacementMeters = diagnosticP95DisplacementMeters,
            meanTemporalDisplacementMeters =
                diagnosticMeanDisplacementMeters,
            p95TemporalDisplacementMeters =
                diagnosticP95DisplacementMeters,
            temporalDeformationConsistency =
                if (
                    diagnosticP95DisplacementMeters.isFinite() &&
                    diagnosticP95DisplacementMeters > 0.0
                ) {
                    exp(
                        -diagnosticP95DisplacementMeters /
                            config.temporalDisplacementRejectMeters
                    ).coerceIn(0.0, 1.0)
                } else {
                    0.0
                },
            edgeStretchScore = 0.0,
            normalAgreement = 0.0,
            topologySupport = 0.0,
            normalFlowConsistency = 0.0,
            matchedVertexCount = 0,
            sampledVertexCount = sampledVertexCount,
            warnings = buildList {
                add(reason)
                if (
                    reason.contains("correspondence-coverage") ||
                        diagnosticP95DisplacementMeters >=
                            config.temporalDisplacementWarningMeters
                ) {
                    add("canonical-temporal-deformation-elevated")
                }
            }.distinct(),
            recommendations = listOf(
                recommendation,
            ),
        )
    }

    private fun vertexNormals(
        positions: List<Vec3>,
        triangles: List<MeshTriangle>,
    ): List<Vec3> {
        val sums =
            Array(positions.size) {
                Vec3(
                    0.0,
                    0.0,
                    0.0,
                )
            }

        triangles.forEach { triangle ->
            if (
                triangle.a !in positions.indices ||
                triangle.b !in positions.indices ||
                triangle.c !in positions.indices
            ) {
                return@forEach
            }

            val normal =
                (
                    positions[triangle.b] -
                        positions[triangle.a]
                    ).cross(
                        positions[triangle.c] -
                            positions[triangle.a],
                    )

            sums[triangle.a] =
                sums[triangle.a] + normal
            sums[triangle.b] =
                sums[triangle.b] + normal
            sums[triangle.c] =
                sums[triangle.c] + normal
        }

        return sums.map { normal ->
            if (normal.norm() < 1e-8) {
                Vec3(
                    0.0,
                    0.0,
                    0.0,
                )
            } else {
                normal.normalized()
            }
        }
    }

    private fun edgeLengths(
        positions: List<Vec3>,
        triangles: List<MeshTriangle>,
    ): Map<Long, Double> {
        val result =
            LinkedHashMap<Long, Double>()

        fun addEdge(
            a: Int,
            b: Int,
        ) {
            if (
                a !in positions.indices ||
                b !in positions.indices ||
                a == b
            ) {
                return
            }

            val key =
                edgeKey(a, b)

            if (key !in result) {
                val length =
                    positions[a].distanceTo(
                        positions[b],
                    )

                if (
                    length.isFinite() &&
                    length > 1e-8
                ) {
                    result[key] = length
                }
            }
        }

        triangles.forEach { triangle ->
            addEdge(
                triangle.a,
                triangle.b,
            )
            addEdge(
                triangle.b,
                triangle.c,
            )
            addEdge(
                triangle.c,
                triangle.a,
            )
        }

        return result
    }

    private fun edgeKey(
        a: Int,
        b: Int,
    ): Long {
        val low = min(a, b).toLong() and 0xffffffffL
        val high =
            max(a, b).toLong() and
                0xffffffffL
        return (
            low shl 32
        ) or high
    }

    private fun finite(value: Vec3): Vec3 =
        Vec3(
            if (value.x.isFinite()) {
                value.x
            } else {
                0.0
            },
            if (value.y.isFinite()) {
                value.y
            } else {
                0.0
            },
            if (value.z.isFinite()) {
                value.z
            } else {
                0.0
            },
        )

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun percentile(
        values: List<Double>,
        fraction: Double,
    ): Double {
        if (values.isEmpty()) return 0.0

        val sorted =
            values.filter {
                it.isFinite()
            }.sorted()

        if (sorted.isEmpty()) return 0.0
        if (sorted.size == 1) {
            return sorted.first()
        }

        val position =
            fraction.coerceIn(0.0, 1.0) *
                (sorted.size - 1)

        val lower =
            position.toInt()
        val upper =
            min(
                sorted.lastIndex,
                lower + 1,
            )

        val blend =
            position - lower

        return sorted[lower] *
            (1.0 - blend) +
            sorted[upper] *
            blend
    }

}
