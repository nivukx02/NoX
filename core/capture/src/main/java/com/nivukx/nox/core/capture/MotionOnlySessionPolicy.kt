package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.MotionCaptureMode
import java.io.File

/**
 * Explicit storage policy for NoX motion-capture sessions.
 *
 * A capture take may use camera pixels transiently for inference, but the product's canonical
 * recorded artifact is motion/state data. This policy keeps video output out of the session contract
 * and provides a final integrity scan for accidental media artifacts.
 */
class MotionOnlySessionPolicy(
    private val config: Config = Config(),
) {

    data class Config(
        val forbiddenExtensions: Set<String> = setOf(
            "mp4",
            "mkv",
            "mov",
            "webm",
            "avi",
            "jpg",
            "jpeg",
            "png",
            "heic",
            "heif",
            "noxraw",
        ),
        val forbiddenNameTokens: Set<String> = setOf(
            "video",
            "movie",
            "recording",
            "frame_",
            "preview",
        ),
        val allowedRootFiles: Set<String> = setOf(
            "manifest.json",
            "frames.ndjson",
            "poses.ndjson",
            "faces.ndjson",
            "eyes.ndjson",
            "hands.ndjson",
            "hand3d.ndjson",
            "reconstruction.ndjson",
            "avatar_motion.ndjson",
            "telemetry.ndjson",
            "pipeline_telemetry.ndjson",
            "lighting.ndjson",
            "imu.ndjson",
            "focus.ndjson",
            "dense_mesh.bin",
            "dense_mesh_index.ndjson",
        ),
    )

    data class Validation(
        val valid: Boolean,
        val violations: List<String>,
        val motionOnly: Boolean,
        val totalBytes: Long,
        val inspectedFiles: Int,
    )

    fun validateSession(
        directory: File,
        mode: MotionCaptureMode = MotionCaptureMode.MONOCULAR_RGB,
    ): Validation {
        if (!directory.isDirectory) {
            return Validation(
                valid = false,
                violations = listOf("session directory does not exist"),
                motionOnly = false,
                totalBytes = 0L,
                inspectedFiles = 0,
            )
        }

        val violations = ArrayList<String>()
        var bytes = 0L
        var count = 0

        directory.walkTopDown()
            .filter { it.isFile }
            .forEach { file ->
                count += 1
                bytes += file.length()

                val relative = file.relativeTo(directory)
                    .path
                    .replace(File.separatorChar, '/')

                val extension = file.extension.lowercase()
                if (extension in config.forbiddenExtensions) {
                    violations += "media artifact: $relative"
                }

                val lower = relative.lowercase()
                config.forbiddenNameTokens
                    .firstOrNull { token -> token in lower }
                    ?.let { token ->
                        violations += "forbidden media token '$token': $relative"
                    }

                if (!relative.contains('/')) {
                    val rootName = file.name
                    if (
                        rootName !in config.allowedRootFiles &&
                        extension !in setOf("json", "ndjson", "bin", "noxraw", "txt")
                    ) {
                        violations += "unrecognized root artifact: $rootName"
                    }
                }
            }

        val manifest = File(directory, "manifest.json")
        if (!manifest.isFile) {
            violations += "manifest.json missing"
        } else {
            val manifestText = runCatching { manifest.readText() }
                .getOrDefault("")
            if (
                !manifestText.contains("\"motionOnly\":true") &&
                !manifestText.contains("\"motionOnly\": true")
            ) {
                violations += "manifest does not declare motionOnly=true"
            }
            if (
                !manifestText.contains(
                    "\"mode\":\"" + mode.name + "\"",
                ) &&
                !manifestText.contains(
                    "\"mode\": \"" + mode.name + "\"",
                )
            ) {
                violations += "manifest mode does not match requested capture mode"
            }
        }

        return Validation(
            valid = violations.isEmpty(),
            violations = violations.distinct(),
            motionOnly = violations.none {
                it.startsWith("media artifact:")
            },
            totalBytes = bytes,
            inspectedFiles = count,
        )
    }

    fun assertMotionOnly(
        directory: File,
        mode: MotionCaptureMode = MotionCaptureMode.MONOCULAR_RGB,
    ) {
        val result = validateSession(directory, mode)
        check(result.valid) {
            "Motion-only session validation failed: " +
                result.violations.joinToString("; ")
        }
    }

    fun shouldPersist(
        path: File,
        motionOnly: Boolean,
    ): Boolean {
        if (!motionOnly) return true
        val extension = path.extension.lowercase()
        if (extension in config.forbiddenExtensions) return false

        val lower = path.name.lowercase()
        return config.forbiddenNameTokens.none { token -> token in lower }
    }
}
