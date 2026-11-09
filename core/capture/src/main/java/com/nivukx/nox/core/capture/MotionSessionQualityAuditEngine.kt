package com.nivukx.nox.core.capture

import org.json.JSONObject
import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Post-capture quality audit for a structured NoX motion session.
 *
 * The auditor evaluates integrity, temporal continuity, reconstruction confidence, face/eye freshness
 * and semantic-event coverage. It never rewrites the take and never fabricates missing evidence.
 */
class MotionSessionQualityAuditEngine(
    private val integrity: SessionIntegrityValidator = SessionIntegrityValidator(),
) {

    data class Config(
        val minimumReconstructionConfidence: Double = 0.55,
        val maximumMeanReprojectionErrorPx: Double = 5.0,
        val maximumGapMillis: Double = 180.0,
        val minimumFaceEyeFreshness: Double = 0.70,
        val minimumProductionEvidenceCoverage: Double = 0.82,
        val minimumAuditableFrames: Long = 30,
    )

    data class StreamStats(
        val fileName: String,
        val records: Long,
        val firstTimestampNanos: Long?,
        val lastTimestampNanos: Long?,
        val meanConfidence: Double,
        val minimumConfidence: Double,
        val maximumConfidence: Double,
        val gapCount: Long,
        val largestGapMillis: Double,
    )

    data class Segment(
        val startTimestampNanos: Long,
        val endTimestampNanos: Long,
        val frameCount: Long,
        val meanConfidence: Double,
        val maxGapMillis: Double,
        val eventTypes: Set<String>,
        val warnings: List<String>,
    )

    data class Report(
        val valid: Boolean,
        val sessionId: String?,
        val qualityScore: Double,
        val integrityPassed: Boolean,
        val frameCount: Long,
        val segmentCount: Int,
        val goodSegmentCount: Int,
        val reconstructionConfidence: Double,
        val reconstructionReprojectionErrorPx: Double,
        val faceEyeFreshness: Double,
        val productionEvidenceCoverage: Double,
        val largestGapMillis: Double,
        val streams: List<StreamStats>,
        val segments: List<Segment>,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun audit(
        sessionDirectory: File,
        policy: Config = Config(),
    ): Report {
        require(sessionDirectory.isDirectory)

        val validation = integrity.validate(sessionDirectory)
        val streams = listOf(
            "poses.ndjson",
            "reconstruction.ndjson",
            "face_eye_fusion.ndjson",
            "motion_uncertainty.ndjson",
            "production_policy.ndjson",
            "cross_modal_identity.ndjson",
            "reconstruction_quality.ndjson",
            "motion_events.ndjson",
            "avatar_motion.ndjson",
        ).mapNotNull { fileName ->
            val file = File(sessionDirectory, fileName)
            if (!file.isFile) null else readStream(file, policy)
        }

        val frameStats = streams.firstOrNull {
            it.fileName == "reconstruction.ndjson"
        }

        val faceEyeStats = streams.firstOrNull {
            it.fileName == "face_eye_fusion.ndjson"
        }

        val frameCount = frameStats?.records ?:
            streams.maxOfOrNull { it.records } ?: 0L

        val reconstructionConfidence =
            frameStats?.meanConfidence ?: 0.0

        val faceEyeFreshness =
            if (faceEyeStats == null || frameCount <= 0L) {
                0.0
            } else {
                (faceEyeStats.records.toDouble() / frameCount.toDouble())
                    .coerceIn(0.0, 1.0)
            }

        val productionEvidenceCoverage =
            provenanceCoverage(
                streams = streams,
                frameCount = frameCount,
                required = listOf(
                    "motion_uncertainty.ndjson",
                    "production_policy.ndjson",
                    "cross_modal_identity.ndjson",
                    "reconstruction_quality.ndjson",
                ),
            )

        val largestGap = streams
            .maxOfOrNull { it.largestGapMillis } ?: 0.0

        val segments = buildSegments(
            sessionDirectory = sessionDirectory,
            reconstruction = frameStats,
            policy = policy,
        )

        val goodSegments = segments.count { segment ->
            segment.meanConfidence >= policy.minimumReconstructionConfidence &&
                segment.maxGapMillis <= policy.maximumGapMillis
        }

        val integrityScore = if (validation.valid) 1.0 else 0.0
        val frameScore = if (frameCount >= policy.minimumAuditableFrames) 1.0
            else (frameCount.toDouble() / policy.minimumAuditableFrames.toDouble())
                .coerceIn(0.0, 1.0)
        val confidenceScore = reconstructionConfidence.coerceIn(0.0, 1.0)
        val continuityScore = (1.0 - largestGap /
            policy.maximumGapMillis.coerceAtLeast(1.0)).coerceIn(0.0, 1.0)
        val faceEyeScore = if (faceEyeStats == null) 0.65
            else faceEyeFreshness.coerceIn(0.0, 1.0)

        val qualityScore = (
            integrityScore * 0.20 +
                frameScore * 0.10 +
                confidenceScore * 0.30 +
                continuityScore * 0.16 +
                faceEyeScore * 0.10 +
                productionEvidenceCoverage * 0.14
        ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (!validation.valid) add("session integrity validation failed")
            if (frameCount < policy.minimumAuditableFrames) add("take is too short for a full quality audit")
            if (reconstructionConfidence < policy.minimumReconstructionConfidence) {
                add("reconstruction confidence is below the live-quality target")
            }
            if (frameStats != null && frameStats.meanConfidence > 0.0 &&
                frameStats.largestGapMillis > policy.maximumGapMillis) {
                add("reconstruction stream contains large temporal gaps")
            }
            if (faceEyeStats != null && faceEyeFreshness < policy.minimumFaceEyeFreshness) {
                add("face/eye stream coverage is below the configured target")
            }
            if (
                productionEvidenceCoverage <
                    policy.minimumProductionEvidenceCoverage
            ) {
                add(
                    "production provenance stream coverage is below the configured target",
                )
            }
        }

        val recommendations = buildList {
            if (!validation.valid) add("Repair the session and re-run integrity validation before export.")
            if (reconstructionConfidence < policy.minimumReconstructionConfidence) {
                add("Review low-confidence segments and consider a higher-quality reconstruction preset.")
            }
            if (largestGap > policy.maximumGapMillis) {
                add("Inspect device synchronization, thermal throttling and dropped-frame telemetry.")
            }
            if (faceEyeStats != null && faceEyeFreshness < policy.minimumFaceEyeFreshness) {
                add("Increase face/eye inference budget or use a dedicated face camera.")
            }
            if (
                productionEvidenceCoverage <
                    policy.minimumProductionEvidenceCoverage
            ) {
                add(
                    "Re-capture or repair the production-policy, motion-uncertainty, identity and reconstruction-quality provenance streams before export.",
                )
            }
            if (goodSegments < segments.size && segments.isNotEmpty()) {
                add("Export only verified segments for downstream animation until weak ranges are reviewed.")
            }
        }

        val manifest = runCatching {
            val file = File(sessionDirectory, "manifest.json")
            if (!file.isFile) null else JSONObject(file.readText()).optString("sessionId", null)
        }.getOrNull()

        return Report(
            valid = validation.valid && qualityScore >= 0.55,
            sessionId = manifest,
            qualityScore = qualityScore,
            integrityPassed = validation.valid,
            frameCount = frameCount,
            segmentCount = segments.size,
            goodSegmentCount = goodSegments,
            reconstructionConfidence = reconstructionConfidence,
            reconstructionReprojectionErrorPx = reconstructionReprojection(sessionDirectory),
            faceEyeFreshness = faceEyeFreshness,
            productionEvidenceCoverage = productionEvidenceCoverage,
            largestGapMillis = largestGap,
            streams = streams,
            segments = segments,
            warnings = warnings.distinct(),
            recommendations = recommendations.distinct(),
        )
    }

    private fun readStream(file: File, policy: Config): StreamStats {
        var records = 0L
        var first: Long? = null
        var last: Long? = null
        var sumConfidence = 0.0
        var minimumConfidence = 1.0
        var maximumConfidence = 0.0
        var gaps = 0L
        var largestGapMillis = 0.0

        file.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val json = runCatching { JSONObject(line) }.getOrNull() ?: return@forEachLine
            val timestamp = json.optLong("timestampNanos", Long.MIN_VALUE)
                .takeIf { it != Long.MIN_VALUE }
            val confidence = json.optDouble("confidence", Double.NaN)
            records += 1L
            if (timestamp != null) {
                first = first?.let { min(it, timestamp) } ?: timestamp
                if (last != null) {
                    val gap = max(0L, timestamp - last!!).toDouble() / 1e6
                    if (gap > policy.maximumGapMillis) gaps += 1L
                    largestGapMillis = max(largestGapMillis, gap)
                }
                last = timestamp
            }
            if (confidence.isFinite()) {
                val bounded = confidence.coerceIn(0.0, 1.0)
                sumConfidence += bounded
                minimumConfidence = min(minimumConfidence, bounded)
                maximumConfidence = max(maximumConfidence, bounded)
            }
        }

        return StreamStats(
            fileName = file.name,
            records = records,
            firstTimestampNanos = first,
            lastTimestampNanos = last,
            meanConfidence = if (records == 0L) 0.0 else sumConfidence / records.toDouble(),
            minimumConfidence = if (records == 0L) 0.0 else minimumConfidence,
            maximumConfidence = maximumConfidence,
            gapCount = gaps,
            largestGapMillis = largestGapMillis,
        )
    }

    private fun buildSegments(
        sessionDirectory: File,
        reconstruction: StreamStats?,
        policy: Config,
    ): List<Segment> {
        if (reconstruction == null) return emptyList()
        val file = File(sessionDirectory, reconstruction.fileName)
        if (!file.isFile) return emptyList()

        val segments = mutableListOf<Segment>()
        var start = Long.MIN_VALUE
        var end = Long.MIN_VALUE
        var count = 0L
        var confidenceSum = 0.0
        var maximumGap = 0.0
        var events = emptySet<String>()

        fun flush() {
            if (start == Long.MIN_VALUE || count == 0L) return
            val warnings = buildList {
                if (maximumGap > policy.maximumGapMillis) add("temporal-gap")
                val mean = confidenceSum / count.toDouble()
                if (mean < policy.minimumReconstructionConfidence) add("low-confidence")
            }
            segments += Segment(
                startTimestampNanos = start,
                endTimestampNanos = end,
                frameCount = count,
                meanConfidence = confidenceSum / count.toDouble(),
                maxGapMillis = maximumGap,
                eventTypes = events,
                warnings = warnings,
            )
            start = Long.MIN_VALUE
            end = Long.MIN_VALUE
            count = 0L
            confidenceSum = 0.0
            maximumGap = 0.0
            events = emptySet()
        }

        file.forEachLine { line ->
            val json = runCatching { JSONObject(line) }.getOrNull() ?: return@forEachLine
            val timestamp = json.optLong("timestampNanos", Long.MIN_VALUE)
                .takeIf { it != Long.MIN_VALUE } ?: return@forEachLine
            val confidence = json.optDouble("confidence", 0.0).coerceIn(0.0, 1.0)
            if (start == Long.MIN_VALUE) start = timestamp
            if (end != Long.MIN_VALUE) {
                val gap = max(0L, timestamp - end).toDouble() / 1e6
                maximumGap = max(maximumGap, gap)
                if (gap > policy.maximumGapMillis * 2.0) flush()
            }
            if (start == Long.MIN_VALUE) start = timestamp
            end = timestamp
            count += 1L
            confidenceSum += confidence
        }
        flush()
        return segments
    }

    private fun provenanceCoverage(
        streams: List<StreamStats>,
        frameCount: Long,
        required: List<String>,
    ): Double {
        if (frameCount <= 0L || required.isEmpty()) return 0.0

        val byName = streams.associateBy { it.fileName }
        return required.map { name ->
            val records =
                byName[name]?.records ?: 0L
            (
                records.toDouble() /
                    frameCount.toDouble()
                ).coerceIn(0.0, 1.0)
        }.minOrNull() ?: 0.0
    }

    private fun reconstructionReprojection(sessionDirectory: File): Double {
        val file = File(sessionDirectory, "reconstruction.ndjson")
        if (!file.isFile) return 0.0
        var sum = 0.0
        var count = 0L
        file.forEachLine { line ->
            val value = runCatching {
                JSONObject(line).optDouble("reprojectionErrorPx", Double.NaN)
            }.getOrNull() ?: return@forEachLine
            if (value != null && value.isFinite()) {
                sum += value
                count += 1L
            }
        }
        return if (count == 0L) 0.0 else sum / count.toDouble()
    }
}