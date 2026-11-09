package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CameraFeatureMatrix
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Deterministic admission planner for multi-camera motion capture.
 *
 * Camera2 exposes platform-declared concurrent camera combinations, but a supported combination
 * is not automatically a good reconstruction configuration. This engine adds explicit budgeting
 * for memory, throughput, thermal headroom, synchronization, calibration and view diversity.
 */
class ConcurrentCaptureCapabilityEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val targetWidth: Int = 1920,
        val targetHeight: Int = 1080,
        val targetFps: Int = 60,
        val yuvBytesPerPixel: Double = 1.5,
        val rawBytesPerPixel: Double = 2.0,
        val depthBytesPerPixel: Double = 2.0,
        val analysisBuffers: Int = 4,
        val rawBuffers: Int = 2,
        val depthBuffers: Int = 3,
        val perCameraMemoryBudgetMb: Double = 512.0,
        val rigMemoryBudgetMb: Double = 2048.0,
        val thermalBudget: Double = 0.86,
        val minimumRigScore: Double = 0.62,
        val minimumViewDiversity: Double = 0.65,
        val maximumPairSkewMillis: Double = 4.0,
    )

    enum class StreamKind { ANALYSIS_YUV, RAW_SENSOR, DEPTH16 }

    data class StreamOption(
        val kind: StreamKind,
        val width: Int,
        val height: Int,
        val fps: Int,
        val enabled: Boolean = true,
        val qualityWeight: Double = 1.0,
    ) {
        init {
            require(width > 0 && height > 0 && fps > 0)
            require(qualityWeight.isFinite() && qualityWeight >= 0.0)
        }

        val pixels: Long
            get() = width.toLong() * height.toLong()
    }

    data class CameraEndpoint(
        val cameraId: String,
        val maxWidth: Int,
        val maxHeight: Int,
        val maxFps: Int,
        val supportsConcurrent: Boolean,
        val supportsRaw: Boolean,
        val rawWidth: Int = 0,
        val rawHeight: Int = 0,
        val supportsDepth: Boolean = false,
        val depthWidth: Int = 0,
        val depthHeight: Int = 0,
        val thermalLoad: Double = 0.0,
        val memoryPressure: Double = 0.0,
        val synchronizationConfidence: Double = 1.0,
        val calibrationConfidence: Double = 1.0,
        val viewDiversity: Double = 0.5,
        val pairSkewMillis: Double = 0.0,
        val supportsHardwareReadoutTimestamp: Boolean = false,
        val hardwareReadoutTimingConfidence: Double = 0.0,
        val supportsLogicalMultiCamera: Boolean = false,
        val physicalCameraCount: Int = 0,
        val logicalSensorSyncType: Int? = null,
    )

    data class ConcurrentSet(
        val cameraIds: Set<String>,
        val source: Source = Source.PLATFORM_DECLARED,
    ) {
        enum class Source { PLATFORM_DECLARED, DISCOVERED_SESSION, USER_CONFIGURED }
        init { require(cameraIds.size >= 2) }
    }

    data class CameraCost(
        val cameraId: String,
        val streams: List<StreamOption>,
        val memoryMb: Double,
        val throughputMbPerSecond: Double,
        val computeLoad: Double,
        val thermalCost: Double,
        val quality: Double,
        val blockers: List<String>,
        val warnings: List<String>,
    ) {
        val admissible: Boolean
            get() = blockers.isEmpty()
    }

    data class PairEvidence(
        val first: String,
        val second: String,
        val synchronization: Double,
        val calibration: Double,
        val viewDiversity: Double,
        val skew: Double,
        val aggregate: Double,
        val admissible: Boolean,
        val reasons: List<String>,
    )

    data class RigPlan(
        val selectedCameraIds: List<String>,
        val concurrentSet: ConcurrentSet?,
        val streams: Map<String, List<StreamOption>>,
        val costs: List<CameraCost>,
        val pairEvidence: List<PairEvidence>,
        val totalMemoryMb: Double,
        val totalThroughputMbPerSecond: Double,
        val totalThermalCost: Double,
        val memoryHeadroom: Double,
        val thermalHeadroom: Double,
        val synchronizationScore: Double,
        val calibrationScore: Double,
        val viewDiversityScore: Double,
        val topologyScore: Double = 0.0,
        val timingDeterminismScore: Double = 0.0,
        val rigScore: Double,
        val admissible: Boolean,
        val blockers: List<String>,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    /**
     * Builds a runtime admission plan directly from Camera2 capability evidence.
     *
     * The planner deliberately separates platform capability from dynamic runtime health. A
     * CameraFeatureMatrix proves what the camera reports it can do; synchronization, calibration,
     * thermal, memory and view-diversity values remain explicit evidence supplied by the caller.
     */
    fun planFromFeatureMatrices(
        matrices: List<CameraFeatureMatrix>,
        synchronizationConfidence: Map<String, Double> = emptyMap(),
        calibrationConfidence: Map<String, Double> = emptyMap(),
        thermalLoad: Map<String, Double> = emptyMap(),
        memoryPressure: Map<String, Double> = emptyMap(),
        viewDiversity: Map<String, Double> = emptyMap(),
        pairSkewMillis: Map<String, Double> = emptyMap(),
        requestedCameraCount: Int = 2,
        requestedStreams: List<StreamOption> = defaultStreams(),
    ): RigPlan {
        val unique = matrices
            .distinctBy { it.cameraId }
            .filter { it.cameraId.isNotBlank() }

        val cameras = unique.map { matrix ->
            CameraEndpoint(
                cameraId = matrix.cameraId,
                maxWidth = matrix.maxAnalysisWidth,
                maxHeight = matrix.maxAnalysisHeight,
                maxFps = matrix.maxFps,
                supportsConcurrent = matrix.supportsConcurrentCamera,
                supportsRaw = matrix.supportsRaw,
                rawWidth = matrix.rawWidth,
                rawHeight = matrix.rawHeight,
                supportsDepth = matrix.supportsDepth16,
                depthWidth = matrix.depthWidth,
                depthHeight = matrix.depthHeight,
                thermalLoad =
                    thermalLoad[matrix.cameraId]
                        ?.coerceIn(0.0, 1.0) ?: 0.0,
                memoryPressure =
                    (
                        memoryPressure[matrix.cameraId] ?: 0.0
                    ).coerceIn(0.0, 1.0),
                synchronizationConfidence =
                    synchronizationConfidence[matrix.cameraId]
                        ?.coerceIn(0.0, 1.0)
                        ?: 0.0,
                calibrationConfidence =
                    calibrationConfidence[matrix.cameraId]
                        ?.coerceIn(0.0, 1.0)
                        ?: 0.0,
                viewDiversity =
                    viewDiversity[matrix.cameraId]
                        ?: 0.5,
                pairSkewMillis =
                    pairSkewMillis[matrix.cameraId]
                        ?: 0.0,
                supportsHardwareReadoutTimestamp =
                    matrix.supportsHardwareReadoutTimestamp,
                hardwareReadoutTimingConfidence =
                    if (matrix.supportsHardwareReadoutTimestamp) 0.92 else 0.0,
                supportsLogicalMultiCamera =
                    matrix.supportsLogicalMultiCamera,
                physicalCameraCount =
                    matrix.physicalCameraIds.size,
                logicalSensorSyncType =
                    matrix.logicalSensorSyncType,
            )
        }

        val declaredSets = unique
            .flatMap { it.concurrentCameraSets }
            .map { ids ->
                ConcurrentSet(
                    cameraIds = ids,
                    source = ConcurrentSet.Source.PLATFORM_DECLARED,
                )
            }
            .distinctBy { it.cameraIds.sorted() }

        return plan(
            cameras = cameras,
            concurrentSets = declaredSets,
            requestedCameraCount = requestedCameraCount,
            requestedStreams = requestedStreams,
        )
    }

    fun plan(
        cameras: List<CameraEndpoint>,
        concurrentSets: List<ConcurrentSet>,
        requestedCameraCount: Int,
        requestedStreams: List<StreamOption> = defaultStreams(),
    ): RigPlan {
        require(cameras.isNotEmpty())
        val unique = cameras.distinctBy { it.cameraId }
        val count = requestedCameraCount.coerceIn(2, unique.size)
        val candidates = concurrentSets.flatMap { set ->
            if (set.cameraIds.size < count) return@flatMap emptyList<List<CameraEndpoint>>()
            val members = set.cameraIds.mapNotNull { id -> unique.firstOrNull { it.cameraId == id } }
            if (members.size < count) emptyList()
            else listOf(selectCoverage(members, count))
        }.distinctBy { it.map { camera -> camera.cameraId }.sorted() }

        if (candidates.isEmpty()) return blocked("NO_CONCURRENT_CAMERA_SET")

        return candidates
            .map { evaluateSet(it, concurrentSets, requestedStreams) }
            .maxByOrNull { if (it.admissible) it.rigScore + 1.0 else it.rigScore }
            ?: blocked("NO_EVALUATABLE_CAMERA_SET")
    }

    fun evaluatePair(a: CameraEndpoint, b: CameraEndpoint): PairEvidence {
        val sync = min(a.synchronizationConfidence, b.synchronizationConfidence)
        val calibration = sqrt(a.calibrationConfidence * b.calibrationConfidence)
        val diversity =
            (1.0 - abs(a.viewDiversity - b.viewDiversity))
                .coerceIn(0.0, 1.0)
        val observedPairSkew =
            max(a.pairSkewMillis, b.pairSkewMillis)
                .coerceAtLeast(0.0)
        val skew =
            if (observedPairSkew <= 1.0) {
                observedPairSkew.coerceIn(0.0, 1.0)
            } else {
                (
                    1.0 -
                        observedPairSkew /
                            config.maximumPairSkewMillis
                ).coerceIn(0.0, 1.0)
            }
        val reasons = buildList {
            if (sync < 0.70) add("SYNC")
            if (calibration < 0.70) add("CALIBRATION")
            if (diversity < config.minimumViewDiversity) add("DIVERSITY")
            if (skew < 0.50) add("TEMPORAL_SKEW")
        }
        val aggregate = (
            sync * 0.30 +
                calibration * 0.28 +
                diversity * 0.22 +
                skew * 0.20
            ).coerceIn(0.0, 1.0)
        return PairEvidence(
            first = a.cameraId,
            second = b.cameraId,
            synchronization = sync,
            calibration = calibration,
            viewDiversity = diversity,
            skew = skew,
            aggregate = aggregate,
            admissible = reasons.isEmpty() && aggregate >= 0.70,
            reasons = reasons,
        )
    }

    fun estimateStreamCost(option: StreamOption): Pair<Double, Double> {
        val bpp = when (option.kind) {
            StreamKind.ANALYSIS_YUV -> config.yuvBytesPerPixel
            StreamKind.RAW_SENSOR -> config.rawBytesPerPixel
            StreamKind.DEPTH16 -> config.depthBytesPerPixel
        }
        val frameBytes = option.pixels * bpp
        val throughput = frameBytes * option.fps / 1_048_576.0
        val buffers = when (option.kind) {
            StreamKind.ANALYSIS_YUV -> config.analysisBuffers
            StreamKind.RAW_SENSOR -> config.rawBuffers
            StreamKind.DEPTH16 -> config.depthBuffers
        }
        return frameBytes * buffers / 1_048_576.0 to throughput
    }

    private fun defaultStreams() = listOf(
        StreamOption(
            kind = StreamKind.ANALYSIS_YUV,
            width = config.targetWidth,
            height = config.targetHeight,
            fps = config.targetFps,
        ),
    )

    private fun selectCoverage(
        candidates: List<CameraEndpoint>,
        count: Int,
    ): List<CameraEndpoint> {
        if (candidates.size <= count) return candidates
        val remaining = candidates.toMutableList()
        val selected = mutableListOf<CameraEndpoint>()
        selected += remaining.maxBy { baseScore(it) }
        remaining.remove(selected.single())
        while (selected.size < count && remaining.isNotEmpty()) {
            val next = remaining.maxBy { candidate ->
                val novelty = selected.map {
                    1.0 - abs(it.viewDiversity - candidate.viewDiversity)
                }.minOrNull() ?: 1.0
                baseScore(candidate) * 0.70 + novelty * 0.30
            }
            selected += next
            remaining.remove(next)
        }
        return selected
    }

    private fun evaluateSet(
        selected: List<CameraEndpoint>,
        allSets: List<ConcurrentSet>,
        requestedStreams: List<StreamOption>,
    ): RigPlan {
        val set = allSets.firstOrNull { s -> selected.all { it.cameraId in s.cameraIds } }
        val costs = selected.map { cameraCost(it, requestedStreams) }
        val pairs = selected.flatMapIndexed { index, left ->
            selected.drop(index + 1).map { right -> evaluatePair(left, right) }
        }

        val totalMemory = costs.sumOf { it.memoryMb }
        val totalThroughput = costs.sumOf { it.throughputMbPerSecond }
        val totalThermal = costs.sumOf { it.thermalCost } / max(selected.size, 1)
        val memoryLimit = config.rigMemoryBudgetMb * 0.82
        val memoryHeadroom = (1.0 - totalMemory / memoryLimit).coerceIn(0.0, 1.0)
        val thermalHeadroom = (1.0 - totalThermal / config.thermalBudget).coerceIn(0.0, 1.0)
        val sync = selected.map { it.synchronizationConfidence }.average()
        val calibration = selected.map { it.calibrationConfidence }.average()
        val diversity = pairs
            .map { it.viewDiversity }
            .averageOrZero()
        val pairScore = pairs.map { it.aggregate }.averageOrZero()

        val topologyScore =
            selected
                .map(::topologyScore)
                .averageOrZero()

        val timingDeterminism =
            selected
                .map(::timingDeterminismScore)
                .averageOrZero()

        val rigScore = (
            memoryHeadroom * 0.12 +
                thermalHeadroom * 0.08 +
                sync * 0.19 +
                calibration * 0.18 +
                diversity * 0.13 +
                pairScore * 0.11 +
                topologyScore * 0.09 +
                timingDeterminism * 0.10
            ).coerceIn(0.0, 1.0)

        val blockers = buildList {
            if (set == null) add("CONCURRENT_SET_NOT_DECLARED")
            if (costs.any { !it.admissible }) addAll(costs.flatMap { it.blockers })
            if (totalMemory > memoryLimit) add("TOTAL_MEMORY_BUDGET_EXCEEDED")
            if (totalThermal > config.thermalBudget) add("THERMAL_BUDGET_EXCEEDED")
            if (sync < 0.70) add("SYNC_EVIDENCE_LOW")
            if (calibration < 0.70) add("CALIBRATION_EVIDENCE_LOW")
            if (diversity < config.minimumViewDiversity) add("VIEW_DIVERSITY_LOW")
            if (topologyScore < 0.42) add("CAMERA_TOPOLOGY_EVIDENCE_LOW")
            if (rigScore < config.minimumRigScore) add("RIG_SCORE_LOW")
        }.distinct()

        val warnings = buildList {
            if (memoryHeadroom < 0.35) add("MEMORY_HEADROOM_LOW")
            if (thermalHeadroom < 0.35) add("THERMAL_HEADROOM_LOW")
            if (pairs.any { !it.admissible }) add("PAIR_EVIDENCE_DEGRADED")
            if (topologyScore < 0.70) add("CAMERA_TOPOLOGY_EVIDENCE_DEGRADED")
            if (timingDeterminism < 0.72) add("HARDWARE_TIMING_DETERMINISM_LIMITED")
            if (selected.size < 4) add("LOW_CAMERA_DENSITY_FOR_FULL_BODY_COVERAGE")
        }

        val recommendations = buildList {
            if (memoryHeadroom < 0.50) add("Disable optional RAW/depth sidecars on constrained cameras.")
            if (thermalHeadroom < 0.50) add("Use a short reference capture window and reduce nonessential inference stages.")
            if (sync < 0.90) add("Run clock and sub-frame temporal alignment before metric capture.")
            if (calibration < 0.90) add("Refresh multi-view extrinsics before reference-grade reconstruction.")
            if (diversity < 0.75) add("Increase angular coverage rather than simply increasing camera count.")
            if (topologyScore < 0.70) add("Prefer calibrated logical-camera timing or independently synchronized physical cameras before reference-grade fusion.")
            if (timingDeterminism < 0.72) add("Prefer hardware readout timestamps or calibrated logical sensor synchronization for the temporal anchor path.")
            if (totalThroughput > 500.0) add("Keep frame queues bounded and avoid duplicate CPU copies.")
        }

        return RigPlan(
            selectedCameraIds = selected.map { it.cameraId },
            concurrentSet = set,
            streams = costs.associate { it.cameraId to it.streams },
            costs = costs,
            pairEvidence = pairs,
            totalMemoryMb = totalMemory,
            totalThroughputMbPerSecond = totalThroughput,
            totalThermalCost = totalThermal,
            memoryHeadroom = memoryHeadroom,
            thermalHeadroom = thermalHeadroom,
            synchronizationScore = sync,
            calibrationScore = calibration,
            viewDiversityScore = diversity,
            topologyScore = topologyScore,
            timingDeterminismScore = timingDeterminism,
            rigScore = rigScore,
            admissible = blockers.isEmpty(),
            blockers = blockers,
            warnings = warnings.distinct(),
            recommendations = recommendations.distinct(),
        )
    }

    private fun cameraCost(
        camera: CameraEndpoint,
        requested: List<StreamOption>,
    ): CameraCost {
        var memory = 0.0
        var throughput = 0.0
        var compute = 0.0
        val blockers = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        val streams = requested.map { resolve(camera, it) }

        streams.filter { it.enabled }.forEach { stream ->
            if (stream.width <= 0 || stream.height <= 0 || stream.fps <= 0) {
                blockers += "${stream.kind}_UNSUPPORTED"
                return@forEach
            }
            val cost = estimateStreamCost(stream)
            memory += cost.first
            throughput += cost.second
            compute += (
                stream.pixels.toDouble() /
                    (1920.0 * 1080.0) *
                    stream.fps / 60.0 *
                    stream.qualityWeight
            )
        }

        if (memory > config.perCameraMemoryBudgetMb * 0.82) {
            blockers += "CAMERA_MEMORY_BUDGET_EXCEEDED"
        }
        val thermal = (
            camera.thermalLoad +
                0.20 * min(compute / 2.0, 1.0) +
                0.12 * camera.memoryPressure
            ).coerceIn(0.0, 1.0)
        if (thermal > config.thermalBudget) blockers += "CAMERA_THERMAL_BUDGET_EXCEEDED"
        if (camera.synchronizationConfidence < 0.70) warnings += "CAMERA_SYNC_DEGRADED"
        if (camera.calibrationConfidence < 0.70) warnings += "CAMERA_CALIBRATION_DEGRADED"

        val quality = (
            camera.synchronizationConfidence * 0.25 +
                camera.calibrationConfidence * 0.25 +
                (1.0 - camera.memoryPressure) * 0.10 +
                (1.0 - camera.thermalLoad) * 0.10 +
                camera.viewDiversity * 0.30
            ).coerceIn(0.0, 1.0)

        return CameraCost(
            cameraId = camera.cameraId,
            streams = streams,
            memoryMb = memory,
            throughputMbPerSecond = throughput,
            computeLoad = compute,
            thermalCost = thermal,
            quality = quality,
            blockers = blockers.distinct(),
            warnings = warnings.distinct(),
        )
    }

    private fun resolve(
        camera: CameraEndpoint,
        stream: StreamOption,
    ): StreamOption = when (stream.kind) {
        StreamKind.ANALYSIS_YUV -> stream.copy(
            width = min(stream.width, camera.maxWidth),
            height = min(stream.height, camera.maxHeight),
            fps = min(stream.fps, camera.maxFps),
        )
        StreamKind.RAW_SENSOR -> if (camera.supportsRaw) {
            stream.copy(
                width = min(stream.width, max(camera.rawWidth, 1)),
                height = min(stream.height, max(camera.rawHeight, 1)),
                fps = min(stream.fps, camera.maxFps),
            )
        } else stream.copy(width = 0, height = 0, fps = 0)
        StreamKind.DEPTH16 -> if (camera.supportsDepth) {
            stream.copy(
                width = min(stream.width, max(camera.depthWidth, 1)),
                height = min(stream.height, max(camera.depthHeight, 1)),
                fps = min(stream.fps, camera.maxFps),
            )
        } else stream.copy(width = 0, height = 0, fps = 0)
    }

    private fun baseScore(camera: CameraEndpoint) = (
        camera.synchronizationConfidence * 0.28 +
            camera.calibrationConfidence * 0.24 +
            camera.viewDiversity * 0.24 +
            (1.0 - camera.memoryPressure) * 0.12 +
            (1.0 - camera.thermalLoad) * 0.12
        ).coerceIn(0.0, 1.0)

    private fun timingDeterminismScore(
        camera: CameraEndpoint,
    ): Double {
        val hardwareReadout =
            if (camera.supportsHardwareReadoutTimestamp) {
                camera.hardwareReadoutTimingConfidence.coerceIn(0.0, 1.0)
            } else {
                0.62
            }

        val logicalSync =
            when (camera.logicalSensorSyncType) {
                1 -> 1.0
                0 -> 0.78
                else -> 0.58
            }

        return (
            hardwareReadout * 0.60 +
                logicalSync * 0.40
            ).coerceIn(0.0, 1.0)
    }

    private fun topologyScore(
        camera: CameraEndpoint,
    ): Double {
        val logical =
            when (camera.logicalSensorSyncType) {
                0 -> 1.0
                1 -> 0.74
                else ->
                    if (camera.supportsLogicalMultiCamera) 0.58
                    else 0.82
            }

        val readout =
            if (camera.supportsHardwareReadoutTimestamp) 1.0 else 0.62

        val physical =
            when {
                camera.physicalCameraCount >= 4 -> 1.0
                camera.physicalCameraCount >= 2 -> 0.86
                camera.physicalCameraCount == 1 -> 0.70
                else -> 0.58
            }

        return (
            logical * 0.50 +
                readout * 0.25 +
                physical * 0.25
            ).coerceIn(0.0, 1.0)
    }

    private fun List<Double>.averageOrZero() = if (isEmpty()) 0.0 else average()

    private fun blocked(reason: String) = RigPlan(
        selectedCameraIds = emptyList(),
        concurrentSet = null,
        streams = emptyMap(),
        costs = emptyList(),
        pairEvidence = emptyList(),
        totalMemoryMb = 0.0,
        totalThroughputMbPerSecond = 0.0,
        totalThermalCost = 1.0,
        memoryHeadroom = 0.0,
        thermalHeadroom = 0.0,
        synchronizationScore = 0.0,
        calibrationScore = 0.0,
        viewDiversityScore = 0.0,
        topologyScore = 0.0,
        rigScore = 0.0,
        admissible = false,
        blockers = listOf(reason),
        warnings = listOf("RIG_NOT_ADMITTED"),
        recommendations = listOf("Inspect platform-declared concurrent camera combinations first."),
    )
}
