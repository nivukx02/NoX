package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DynamicTopology
import com.nivukx.nox.core.model.LandmarkObservation
import com.nivukx.nox.core.model.MotionEdge
import com.nivukx.nox.core.model.MotionNode
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TopologyKind
import com.nivukx.nox.core.model.Vec2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Dynamic topology builder for RGB-only motion capture.
 *
 * Human capture uses a stable semantic graph. Animals use a flexible articulation template and
 * generic objects fall back to a geometry-derived graph. The important design point is that
 * reconstruction never silently assumes every subject is a human skeleton.
 */
class DynamicTopologyEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maxNodes: Int = 256,
        val maximumEdges: Int = 512,
        val kNearestForUnknown: Int = 3,
        val minimumNodeConfidence: Double = 0.08,
        val topologyHysteresis: Double = 0.18,
        val defaultHumanScaleMeters: Double = 1.72,
        val defaultAnimalScaleMeters: Double = 0.85,
    )

    data class SemanticNode(
        val id: Int,
        val name: String,
        val observed: Vec2,
        val confidence: Double,
    )

    data class BuildInput(
        val timestampNanos: Long,
        val subjectType: SubjectType,
        val trackId: Long,
        val nodes: List<SemanticNode>,
        val previous: DynamicTopology? = null,
    )

    private val humanNames = mapOf(
        0 to "nose",
        1 to "left_eye_inner",
        2 to "left_eye",
        3 to "left_eye_outer",
        4 to "right_eye_inner",
        5 to "right_eye",
        6 to "right_eye_outer",
        7 to "left_ear",
        8 to "right_ear",
        9 to "left_mouth",
        10 to "right_mouth",
        11 to "left_shoulder",
        12 to "right_shoulder",
        13 to "left_elbow",
        14 to "right_elbow",
        15 to "left_wrist",
        16 to "right_wrist",
        17 to "left_pinky",
        18 to "left_index",
        19 to "left_thumb",
        20 to "right_pinky",
        21 to "right_index",
        22 to "right_thumb",
        23 to "left_hip",
        24 to "right_hip",
        25 to "left_knee",
        26 to "right_knee",
        27 to "left_ankle",
        28 to "right_ankle",
        29 to "left_heel",
        30 to "right_heel",
        31 to "left_foot_index",
        32 to "right_foot_index",
    )

    private val humanEdges = listOf(
        e(11, 12, 0.38, "shoulder_width"),
        e(11, 13, 0.29, "left_upper_arm"),
        e(13, 15, 0.27, "left_forearm"),
        e(12, 14, 0.29, "right_upper_arm"),
        e(14, 16, 0.27, "right_forearm"),
        e(11, 23, 0.49, "left_torso"),
        e(12, 24, 0.49, "right_torso"),
        e(23, 24, 0.26, "hip_width"),
        e(23, 25, 0.44, "left_thigh"),
        e(25, 27, 0.43, "left_shin"),
        e(24, 26, 0.44, "right_thigh"),
        e(26, 28, 0.43, "right_shin"),
        e(27, 29, 0.10, "left_heel"),
        e(28, 30, 0.10, "right_heel"),
        e(29, 31, 0.18, "left_foot"),
        e(30, 32, 0.18, "right_foot"),
        e(11, 0, 0.46, "neck_to_nose"),
        e(12, 0, 0.46, "neck_to_nose"),
        e(0, 7, 0.18, "left_head"),
        e(0, 8, 0.18, "right_head"),
        e(0, 2, 0.09, "left_eye"),
        e(0, 5, 0.09, "right_eye"),
        e(15, 18, 0.09, "left_index"),
        e(15, 19, 0.10, "left_thumb"),
        e(16, 21, 0.09, "right_index"),
        e(16, 22, 0.10, "right_thumb"),
    )

    fun build(input: BuildInput): DynamicTopology {
        val filtered = input.nodes
            .asSequence()
            .filter { it.confidence >= config.minimumNodeConfidence }
            .distinctBy { it.id }
            .take(config.maxNodes)
            .toList()

        val kind = inferKind(input.subjectType, filtered.size)
        val topologyNodes = filtered.map { node ->
            MotionNode(
                id = node.id,
                semanticName = semanticName(kind, node.id),
                position2D = node.observed,
                position3D = null,
                confidence = node.confidence.coerceIn(0.0, 1.0),
                uncertaintyMeters = 1.0,
                predicted = false,
                visible = true,
            )
        }

        val edges = when (kind) {
            TopologyKind.HUMAN,
            TopologyKind.BIPED -> buildHumanEdges(filtered)

            TopologyKind.QUADRUPED,
            TopologyKind.AVIAN,
            TopologyKind.INSECT,
            TopologyKind.ARACHNID ->
                buildFlexibleArticulationEdges(filtered, input.previous)

            else -> buildGeometricEdges(filtered)
        }.take(config.maximumEdges)

        val root = chooseRoot(kind, topologyNodes)
        val previousIds = input.previous?.nodes?.map { it.id }?.toSet().orEmpty()
        val currentIds = topologyNodes.map { it.id }.toSet()
        val intersection = previousIds.intersect(currentIds).size
        val union = (previousIds + currentIds).size
        val topologyStability =
            if (union == 0) 0.0
            else intersection.toDouble() / union.toDouble()

        val confidence = (
            topologyNodes.map { it.confidence }.averageOrNull() * 0.65 +
                min(1.0, edges.size.toDouble() / max(1, topologyNodes.size * 2)) * 0.35
            ).coerceIn(0.0, 1.0)

        val stableConfidence =
            if (input.previous != null && topologyStability >= 1.0 - config.topologyHysteresis) {
                max(confidence, input.previous.globalConfidence * 0.72)
            } else {
                confidence
            }

        return DynamicTopology(
            topologyId =
                kind.name.lowercase() + "-" + input.trackId + "-" + topologyNodes.hashCode(),
            kind = kind,
            nodes = topologyNodes,
            edges = edges,
            rootNodeId = root,
            scaleSource = com.nivukx.nox.core.model.ScaleSource.UNKNOWN,
            globalConfidence = stableConfidence.coerceIn(0.0, 1.0),
            temporalStability = topologyStability,
            topologyStability = topologyStability,
            timestampNanos = input.timestampNanos,
        )
    }

    fun buildFromPose(
        timestampNanos: Long,
        subjectType: SubjectType,
        trackId: Long,
        observations: List<LandmarkObservation>,
        previous: DynamicTopology? = null,
    ): DynamicTopology =
        build(
            BuildInput(
                timestampNanos = timestampNanos,
                subjectType = subjectType,
                trackId = trackId,
                nodes = observations.map {
                    SemanticNode(
                        id = it.landmarkId,
                        name = semanticName(
                            inferKind(subjectType, observations.size),
                            it.landmarkId,
                        ),
                        observed = Vec2(
                            it.x,
                            it.y,
                        ),
                        confidence = it.confidence,
                    )
                },
                previous = previous,
            ),
        )

    private fun buildHumanEdges(
        nodes: List<SemanticNode>,
    ): List<MotionEdge> {
        val ids = nodes.map { it.id }.toSet()
        return humanEdges
            .filter { it.a in ids && it.b in ids }
            .map {
                it.copy(
                    minimumLengthMeters = it.preferredLengthMeters * 0.54,
                    maximumLengthMeters = it.preferredLengthMeters * 1.55,
                )
            }
    }

    private fun buildFlexibleArticulationEdges(
        nodes: List<SemanticNode>,
        previous: DynamicTopology?,
    ): List<MotionEdge> {
        val edges = LinkedHashMap<Pair<Int, Int>, MotionEdge>()

        previous?.edges?.forEach { edge ->
            if (nodes.any { it.id == edge.a } && nodes.any { it.id == edge.b }) {
                edges[edge.a to edge.b] = edge
            }
        }

        buildGeometricEdges(nodes).forEach { edge ->
            edges.putIfAbsent(edge.a to edge.b, edge)
        }

        return edges.values.toList()
    }

    private fun buildGeometricEdges(
        nodes: List<SemanticNode>,
    ): List<MotionEdge> {
        val output = ArrayList<MotionEdge>()
        for (a in nodes.indices) {
            val nearest = nodes.indices
                .asSequence()
                .filter { it != a }
                .sortedBy { b ->
                    distance(nodes[a].observed, nodes[b].observed)
                }
                .take(config.kNearestForUnknown.coerceAtLeast(1))
                .toList()

            for (b in nearest) {
                val na = min(nodes[a].id, nodes[b].id)
                val nb = max(nodes[a].id, nodes[b].id)
                if (na == nb) continue
                val key = na to nb
                if (output.any { min(it.a, it.b) to max(it.a, it.b) == key }) continue

                val imageDistance = distance(nodes[a].observed, nodes[b].observed)
                val pseudoLength = (imageDistance * 1.20).coerceIn(0.04, 1.50)
                output += MotionEdge(
                    a = na,
                    b = nb,
                    preferredLengthMeters = pseudoLength,
                    minimumLengthMeters = pseudoLength * 0.45,
                    maximumLengthMeters = pseudoLength * 1.70,
                    stiffness = 0.35,
                    semantic = "geometric",
                )
            }
        }
        return output
    }

    private fun inferKind(
        subjectType: SubjectType,
        nodeCount: Int,
    ): TopologyKind =
        when (subjectType) {
            SubjectType.HUMAN -> TopologyKind.HUMAN
            SubjectType.ANIMAL ->
                if (nodeCount >= 20) TopologyKind.QUADRUPED else TopologyKind.BIPED
            SubjectType.OBJECT -> TopologyKind.RIGID_OBJECT
            SubjectType.UNKNOWN ->
                when {
                    nodeCount >= 25 -> TopologyKind.DEFORMABLE
                    nodeCount >= 8 -> TopologyKind.BIPED
                    else -> TopologyKind.UNKNOWN
                }
        }

    private fun chooseRoot(
        kind: TopologyKind,
        nodes: List<MotionNode>,
    ): Int? =
        when (kind) {
            TopologyKind.HUMAN,
            TopologyKind.BIPED -> {
                val preferred = nodes.firstOrNull { it.id == 23 }
                    ?: nodes.firstOrNull { it.id == 24 }
                    ?: nodes.firstOrNull { it.id == 11 }
                    ?: nodes.firstOrNull { it.id == 12 }
                preferred?.id
            }

            else -> nodes.maxByOrNull { it.confidence }?.id
        }

    private fun semanticName(
        kind: TopologyKind,
        id: Int,
    ): String =
        if (kind == TopologyKind.HUMAN) {
            humanNames[id] ?: "landmark_" + id
        } else {
            "node_" + id
        }

    private fun distance(a: Vec2, b: Vec2): Double =
        hypot(a.x - b.x, a.y - b.y)

    private fun e(
        a: Int,
        b: Int,
        length: Double,
        semantic: String,
    ): MotionEdge =
        MotionEdge(
            a = a,
            b = b,
            preferredLengthMeters = length,
            minimumLengthMeters = length * 0.54,
            maximumLengthMeters = length * 1.55,
            stiffness = 0.75,
            semantic = semantic,
        )

    private fun List<Double>.averageOrNull(): Double =
        if (isEmpty()) 0.0 else average()
}
