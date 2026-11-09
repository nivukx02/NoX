
package com.nivukx.nox.core.capture

import java.util.zip.CRC32C

/**
 * Validates the motion-only recording contract at runtime and at export time.
 *
 * The verifier is intentionally independent from the recorder implementation so tests and import
 * tools can enforce the same invariant: NoX may consume RGB frames transiently, but a canonical
 * motion session must not contain video/image/raw/depth artifacts.
 */
class MotionSessionIntegrityEngine {

    data class Policy(
        val motionOnly: Boolean = true,
        val rejectImages: Boolean = true,
        val rejectVideo: Boolean = true,
        val rejectRaw: Boolean = true,
        val rejectDepthImages: Boolean = true,
        val requireManifest: Boolean = true,
        val requireMonotonicTimestamps: Boolean = true,
        val requireChecksums: Boolean = true,
        val minimumSchema: Int = 10,
    )

    data class SessionManifest(
        val schema: Int,
        val sessionId: String,
        val motionOnly: Boolean,
        val mediaFramesPersisted: Boolean,
        val videoEncoder: String?,
        val rawSidecar: Boolean,
        val depthSidecar: Boolean,
    )

    data class FrameRecord(
        val timestampNanos: Long,
        val chunkOrdinal: Int,
        val payloadBytes: ByteArray,
        val checksum: Long? = null,
    )

    enum class Severity {
        INFO,
        WARNING,
        ERROR,
        FATAL,
    }

    data class Issue(
        val severity: Severity,
        val code: String,
        val message: String,
        val ordinal: Int? = null,
    )

    data class Report(
        val accepted: Boolean,
        val sessionId: String?,
        val issueCount: Int,
        val fatalCount: Int,
        val errorCount: Int,
        val verifiedFrames: Int,
        val byteCount: Long,
        val contentHash: Long,
        val issues: List<Issue>,
    )

    fun validateManifest(
        manifest: SessionManifest,
        policy: Policy = Policy(),
    ): List<Issue> = buildList {
        if (policy.requireManifest && manifest.sessionId.isBlank()) {
            add(Issue(
                Severity.FATAL,
                "MANIFEST_SESSION_ID_EMPTY",
                "Motion session identifier is empty.",
            ))
        }

        if (manifest.schema < policy.minimumSchema) {
            add(Issue(
                Severity.FATAL,
                "MANIFEST_SCHEMA_OLD",
                "Session schema is older than the minimum supported motion-only schema.",
            ))
        }

        if (policy.motionOnly && !manifest.motionOnly) {
            add(Issue(
                Severity.FATAL,
                "MEDIA_POLICY_DISABLED",
                "Session manifest does not declare motionOnly=true.",
            ))
        }

        if (policy.motionOnly && manifest.mediaFramesPersisted) {
            add(Issue(
                Severity.FATAL,
                "MEDIA_FRAMES_PERSISTED",
                "RGB/media frames are persisted in a motion-only session.",
            ))
        }

        if (policy.rejectVideo && !manifest.videoEncoder.isNullOrBlank() &&
            !manifest.videoEncoder.equals("none", ignoreCase = true)
        ) {
            add(Issue(
                Severity.FATAL,
                "VIDEO_ENCODER_PRESENT",
                "A motion-only session advertises a video encoder.",
            ))
        }

        if (policy.rejectRaw && manifest.rawSidecar) {
            add(Issue(
                Severity.FATAL,
                "RAW_SIDECAR_PRESENT",
                "RAW sensor sidecar is enabled for a motion-only session.",
            ))
        }

        if (policy.rejectDepthImages && manifest.depthSidecar) {
            add(Issue(
                Severity.FATAL,
                "DEPTH_SIDECAR_PRESENT",
                "Depth image sidecar is enabled for a motion-only session.",
            ))
        }
    }

    fun validatePath(
        path: String,
        policy: Policy = Policy(),
    ): Issue? {
        val normalized = path.trim().lowercase()
        val extension = normalized.substringAfterLast('.', "")

        val image = setOf(
            "jpg", "jpeg", "png", "webp", "heic", "heif", "avif",
        ).contains(extension)

        val video = setOf(
            "mp4", "m4v", "mov", "mkv", "avi", "webm", "3gp",
        ).contains(extension)

        val raw = setOf(
            "dng", "raw", "arw", "nef", "cr2", "cr3", "noxraw",
        ).contains(extension)

        val depth = setOf(
            "depth", "depthmap", "pfm", "exr",
        ).contains(extension)

        return when {
            policy.rejectImages && image ->
                Issue(
                    Severity.FATAL,
                    "IMAGE_ARTIFACT_FORBIDDEN",
                    "Image artifact is forbidden by the motion-only policy.",
                )
            policy.rejectVideo && video ->
                Issue(
                    Severity.FATAL,
                    "VIDEO_ARTIFACT_FORBIDDEN",
                    "Video artifact is forbidden by the motion-only policy.",
                )
            policy.rejectRaw && raw ->
                Issue(
                    Severity.FATAL,
                    "RAW_ARTIFACT_FORBIDDEN",
                    "RAW artifact is forbidden by the motion-only policy.",
                )
            policy.rejectDepthImages && depth ->
                Issue(
                    Severity.FATAL,
                    "DEPTH_ARTIFACT_FORBIDDEN",
                    "Depth image artifact is forbidden by the motion-only policy.",
                )
            else -> null
        }
    }

    fun verifyFrames(
        frames: List<FrameRecord>,
        policy: Policy = Policy(),
    ): Report {
        val issues = mutableListOf<Issue>()
        var previousTimestamp = Long.MIN_VALUE
        var bytes = 0L
        val aggregate = CRC32C()

        frames.forEach { frame ->
            if (policy.requireMonotonicTimestamps &&
                previousTimestamp != Long.MIN_VALUE &&
                frame.timestampNanos < previousTimestamp
            ) {
                issues += Issue(
                    Severity.ERROR,
                    "TIMESTAMP_REGRESSION",
                    "Frame timestamp is older than its predecessor.",
                    frame.chunkOrdinal,
                )
            }

            if (frame.chunkOrdinal < 0) {
                issues += Issue(
                    Severity.ERROR,
                    "NEGATIVE_ORDINAL",
                    "Chunk ordinal cannot be negative.",
                    frame.chunkOrdinal,
                )
            }

            if (frame.payloadBytes.isEmpty()) {
                issues += Issue(
                    Severity.ERROR,
                    "EMPTY_PAYLOAD",
                    "Motion frame payload is empty.",
                    frame.chunkOrdinal,
                )
            } else {
                bytes += frame.payloadBytes.size.toLong()
                aggregate.update(frame.payloadBytes)
            }

            if (policy.requireChecksums && frame.checksum == null) {
                issues += Issue(
                    Severity.ERROR,
                    "CHECKSUM_MISSING",
                    "Frame checksum is required by the integrity policy.",
                    frame.chunkOrdinal,
                )
            }

            if (frame.checksum != null) {
                val crc = CRC32C()
                crc.update(frame.payloadBytes)
                if (crc.value != frame.checksum) {
                    issues += Issue(
                        Severity.FATAL,
                        "CHECKSUM_MISMATCH",
                        "Motion chunk checksum does not match its payload.",
                        frame.chunkOrdinal,
                    )
                }
            }

            previousTimestamp = maxOf(
                previousTimestamp,
                frame.timestampNanos,
            )
        }

        val fatal = issues.count { it.severity == Severity.FATAL }
        val errors = issues.count { it.severity == Severity.ERROR }

        return Report(
            accepted = fatal == 0 && errors == 0,
            sessionId = null,
            issueCount = issues.size,
            fatalCount = fatal,
            errorCount = errors,
            verifiedFrames = frames.size,
            byteCount = bytes,
            contentHash = aggregate.value,
            issues = issues,
        )
    }

    fun audit(
        manifest: SessionManifest?,
        frames: List<FrameRecord>,
        artifacts: List<String>,
        policy: Policy = Policy(),
    ): Report {
        val issues = mutableListOf<Issue>()

        if (manifest == null) {
            issues += Issue(
                if (policy.requireManifest) Severity.FATAL else Severity.WARNING,
                "MANIFEST_MISSING",
                "Session manifest is missing.",
            )
        } else {
            issues += validateManifest(manifest, policy)
        }

        artifacts.forEach { artifact ->
            validatePath(artifact, policy)?.let(issues::add)
        }

        val frameReport = verifyFrames(frames, policy)
        issues += frameReport.issues

        val fatal = issues.count { it.severity == Severity.FATAL }
        val errors = issues.count { it.severity == Severity.ERROR }

        return Report(
            accepted = fatal == 0 && errors == 0,
            sessionId = manifest?.sessionId,
            issueCount = issues.size,
            fatalCount = fatal,
            errorCount = errors,
            verifiedFrames = frameReport.verifiedFrames,
            byteCount = frameReport.byteCount,
            contentHash = frameReport.contentHash,
            issues = issues.distinctBy {
                it.severity to it.code to it.ordinal
            },
        )
    }
}
