
package com.nivukx.nox.core.capture

import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.nio.charset.StandardCharsets

/**
 * Repairs safe structural damage after an interrupted capture.
 *
 * NoX uses append-only NDJSON streams, so a process death can leave a final record half-written.
 * Recovery truncates only the incomplete tail of known NDJSON files, rebuilds a recovery manifest,
 * and refuses to "repair" forbidden media artifacts. It never manufactures motion records.
 */
class MotionSessionRecoveryEngine(
    private val validator: SessionIntegrityValidator = SessionIntegrityValidator(),
) {

    data class Policy(
        val knownNdjsonNames: Set<String> = setOf(
            "frames.ndjson",
            "poses.ndjson",
            "faces.ndjson",
            "eyes.ndjson",
            "face_eye_fusion.ndjson",
            "hands.ndjson",
            "hands_3d.ndjson",
            "avatar_motion.ndjson",
            "reconstruction.ndjson",
            "dense_mesh_index.ndjson",
            "telemetry.ndjson",
            "pipeline_telemetry.ndjson",
            "lighting.ndjson",
            "imu.ndjson",
            "focus.ndjson",
        ),
        val forbiddenExtensions: Set<String> = setOf(
            "mp4", "mov", "mkv", "avi", "webm", "jpg", "jpeg", "png",
            "webp", "heic", "dng", "raw", "arw", "nef", "cr2", "cr3",
            "noxraw", "noxdepth", "depth", "depthmap",
        ),
        val requireMotionOnlyManifest: Boolean = true,
        val maximumTailValidationBytes: Int = 1 shl 20,
    ) {
        init {
            require(maximumTailValidationBytes >= 1024)
        }
    }

    data class FileRepair(
        val fileName: String,
        val originalBytes: Long,
        val repairedBytes: Long,
        val truncated: Boolean,
        val completeTailRecord: Boolean,
    )

    data class Result(
        val repaired: Boolean,
        val sessionPath: File,
        val repairedFiles: List<FileRepair>,
        val forbiddenArtifacts: List<String>,
        val validation: SessionIntegrityValidator.Report,
        val diagnostics: List<String>,
    )

    fun recover(
        sessionDirectory: File,
        policy: Policy = Policy(),
    ): Result {
        require(sessionDirectory.isDirectory) {
            "Session directory does not exist"
        }

        val forbidden = mutableListOf<String>()
        sessionDirectory.listFiles()
            .orEmpty()
            .filter { it.isFile }
            .forEach { file ->
                val extension = file.extension.lowercase()
                if (extension in policy.forbiddenExtensions) {
                    forbidden += file.name
                }
            }

        val repairs = mutableListOf<FileRepair>()

        sessionDirectory.listFiles()
            .orEmpty()
            .filter {
                it.isFile &&
                    it.name in policy.knownNdjsonNames
            }
            .forEach { file ->
                repairNdjsonTail(
                    file,
                    policy,
                )?.let(repairs::add)
            }

        validateMotionOnlyManifest(
            sessionDirectory = sessionDirectory,
            required = policy.requireMotionOnlyManifest,
        )?.let(forbidden::add)

        val validation = validator.validate(sessionDirectory)

        val repaired = repairs.any { it.truncated } &&
            forbidden.isEmpty()

        writeRecoveryRecord(
            sessionDirectory = sessionDirectory,
            repairs = repairs,
            forbidden = forbidden,
            validation = validation,
        )

        return Result(
            repaired = repaired,
            sessionPath = sessionDirectory,
            repairedFiles = repairs,
            forbiddenArtifacts = forbidden.distinct(),
            validation = validation,
            diagnostics = diagnostics(
                repairs = repairs,
                forbidden = forbidden,
                validation = validation,
                repaired = repaired,
            ),
        )
    }

    private fun repairNdjsonTail(
        file: File,
        policy: Policy,
    ): FileRepair? {
        val original = file.length()
        if (original == 0L) {
            return FileRepair(
                fileName = file.name,
                originalBytes = 0L,
                repairedBytes = 0L,
                truncated = false,
                completeTailRecord = true,
            )
        }

        val lastByte = RandomAccessFile(file, "r").use { raf ->
            raf.seek(original - 1L)
            raf.read()
        }

        if (lastByte == '\n'.code) {
            return FileRepair(
                fileName = file.name,
                originalBytes = original,
                repairedBytes = original,
                truncated = false,
                completeTailRecord = true,
            )
        }

        val newlineBoundary =
            RandomAccessFile(
                file,
                "r",
            ).use { raf ->
                var cursor =
                    original -
                        1L
                while (cursor >= 0L) {
                    raf.seek(cursor)
                    if (raf.read() ==
                        '\n'.code
                    ) {
                        return@use cursor +
                            1L
                    }
                    cursor -= 1L
                }
                0L
            }

        val tailLength =
            original -
                newlineBoundary

        val completeJsonTail =
            if (
                tailLength > 0L &&
                tailLength <=
                    policy.maximumTailValidationBytes
            ) {
                readTail(
                    file,
                    newlineBoundary,
                    tailLength.toInt(),
                )?.let { tail ->
                    runCatching {
                        JSONObject(tail)
                    }.isSuccess
                } ?: false
            } else {
                false
            }

        if (completeJsonTail) {
            return FileRepair(
                fileName = file.name,
                originalBytes = original,
                repairedBytes = original,
                truncated = false,
                completeTailRecord = true,
            )
        }

        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(
                newlineBoundary,
            )
        }

        return FileRepair(
            fileName = file.name,
            originalBytes = original,
            repairedBytes = newlineBoundary,
            truncated =
                newlineBoundary < original,
            completeTailRecord =
                newlineBoundary > 0L,
        )
    }

    private fun readTail(
        file: File,
        offset: Long,
        length: Int,
    ): String? {
        val bytes =
            ByteArrayOutputStream(
                length,
            )

        RandomAccessFile(
            file,
            "r",
        ).use { raf ->
            raf.seek(offset)
            val buffer =
                ByteArray(
                    minOf(
                        16 * 1024,
                        length,
                    ),
                )
            var remaining = length
            while (remaining > 0) {
                val read =
                    raf.read(
                        buffer,
                        0,
                        minOf(
                            buffer.size,
                            remaining,
                        ),
                    )
                if (read <= 0) break
                bytes.write(
                    buffer,
                    0,
                    read,
                )
                remaining -= read
            }
        }

        return if (
            bytes.size() == length
        ) {
            String(
                bytes.toByteArray(),
                StandardCharsets.UTF_8,
            )
                .trimEnd(
                    '\r',
                    '\n',
                )
        } else {
            null
        }
    }

    private fun validateMotionOnlyManifest(
        sessionDirectory: File,
        required: Boolean,
    ): String? {
        val manifest = File(sessionDirectory, "manifest.json")
        if (!manifest.isFile) {
            return if (required) {
                "manifest.json missing"
            } else {
                null
            }
        }

        return runCatching {
            val json = JSONObject(manifest.readText())
            if (required && !json.optBoolean("motionOnly", false)) {
                "manifest does not declare motionOnly=true"
            } else if (
                json.optBoolean("mediaFramesPersisted", true) ||
                json.optString("videoEncoder", "none") != "none"
            ) {
                "manifest advertises persistent media"
            } else {
                null
            }
        }.getOrElse {
            "manifest invalid: " + (it.message ?: "unknown")
        }
    }

    private fun writeRecoveryRecord(
        sessionDirectory: File,
        repairs: List<FileRepair>,
        forbidden: List<String>,
        validation: SessionIntegrityValidator.Report,
    ) {
        val output = JSONObject().apply {
            put("schema", 1)
            put("recoveryTimestampMillis", System.currentTimeMillis())
            put(
                "repairs",
                org.json.JSONArray().apply {
                    repairs.forEach {
                        put(
                            JSONObject().apply {
                                put("file", it.fileName)
                                put("originalBytes", it.originalBytes)
                                put("repairedBytes", it.repairedBytes)
                                put("truncated", it.truncated)
                                put("completeTailRecord", it.completeTailRecord)
                            },
                        )
                    }
                },
            )
            put(
                "forbiddenArtifacts",
                org.json.JSONArray().apply {
                    forbidden.distinct().forEach(::put)
                },
            )
            put("validationValid", validation.valid)
            put("validationSchema", validation.manifestSchema)
            put("validationErrors", org.json.JSONArray().apply {
                validation.errors.forEach(::put)
            })
        }

        runCatching {
            File(
                sessionDirectory,
                "recovery.json",
            ).writeText(
                output.toString(2),
            )
        }
    }

    private fun diagnostics(
        repairs: List<FileRepair>,
        forbidden: List<String>,
        validation: SessionIntegrityValidator.Report,
        repaired: Boolean,
    ): List<String> = buildList {
        add("repaired=" + repaired)
        add("filesChecked=" + repairs.size)
        add("filesTruncated=" + repairs.count { it.truncated })
        add("bytesReclaimed=" + repairs.sumOf {
            (it.originalBytes - it.repairedBytes).coerceAtLeast(0L)
        })
        add("forbiddenArtifacts=" + forbidden.distinct().size)
        add("validationValid=" + validation.valid)
        add("validationErrors=" + validation.errors.size)
    }
}
