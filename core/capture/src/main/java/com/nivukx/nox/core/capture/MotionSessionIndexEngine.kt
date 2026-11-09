
package com.nivukx.nox.core.capture

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Builds a byte-offset index for structured motion streams.
 *
 * The index is metadata only. It does not duplicate motion payloads and does not create media
 * artifacts. Long takes can therefore be inspected/exported by time range without replaying the
 * entire session from byte zero.
 */
class MotionSessionIndexEngine {

    data class Policy(
        val indexedFiles: Set<String> = setOf(
            "poses.ndjson",
            "frames.ndjson",
            "reconstruction.ndjson",
            "avatar_motion.ndjson",
            "face_eye_fusion.ndjson",
            "motion_events.ndjson",
            "hands.ndjson",
            "dense_mesh_index.ndjson",
            "face_mesh_index.ndjson",
        ),
        val timestampKeys: List<String> = listOf(
            "timestampNanos",
            "timestamp",
            "timeNanos",
        ),
        val strideRecords: Int = 1,
    ) {
        init {
            require(strideRecords >= 1)
            require(indexedFiles.isNotEmpty())
            require(timestampKeys.isNotEmpty())
        }
    }

    data class Entry(
        val fileName: String,
        val recordOrdinal: Long,
        val timestampNanos: Long?,
        val byteOffset: Long,
        val byteLength: Int,
    )

    data class FileIndex(
        val fileName: String,
        val records: Long,
        val indexedEntries: Long,
        val firstTimestampNanos: Long?,
        val lastTimestampNanos: Long?,
        val byteLength: Long,
    )

    data class Result(
        val sessionId: String?,
        val entries: List<Entry>,
        val files: List<FileIndex>,
        val totalRecords: Long,
        val totalBytes: Long,
        val diagnostics: List<String>,
    )

    fun build(
        sessionDirectory: File,
        policy: Policy = Policy(),
    ): Result {
        require(sessionDirectory.isDirectory)

        val manifest = readSessionId(
            File(sessionDirectory, "manifest.json"),
        )

        val entries = mutableListOf<Entry>()
        val fileIndexes = mutableListOf<FileIndex>()

        policy.indexedFiles.sorted().forEach { fileName ->
            val file = File(sessionDirectory, fileName)
            if (!file.isFile) return@forEach

            val result = indexFile(
                file = file,
                policy = policy,
            )

            entries += result.first
            fileIndexes += result.second
        }

        val output = JSONObject().apply {
            put("schema", 1)
            put("sessionId", manifest ?: JSONObject.NULL)
            put(
                "files",
                JSONArray().apply {
                    fileIndexes.forEach { index ->
                        put(
                            JSONObject().apply {
                                put("file", index.fileName)
                                put("records", index.records)
                                put("indexedEntries", index.indexedEntries)
                                put(
                                    "firstTimestampNanos",
                                    index.firstTimestampNanos ?: JSONObject.NULL,
                                )
                                put(
                                    "lastTimestampNanos",
                                    index.lastTimestampNanos ?: JSONObject.NULL,
                                )
                                put("byteLength", index.byteLength)
                            },
                        )
                    }
                },
            )
            put(
                "entries",
                JSONArray().apply {
                    entries.forEach { entry ->
                        put(
                            JSONObject().apply {
                                put("file", entry.fileName)
                                put("recordOrdinal", entry.recordOrdinal)
                                put(
                                    "timestampNanos",
                                    entry.timestampNanos ?: JSONObject.NULL,
                                )
                                put("byteOffset", entry.byteOffset)
                                put("byteLength", entry.byteLength)
                            },
                        )
                    }
                },
            )
        }

        val indexFile =
            File(
                sessionDirectory,
                "session.index.json",
            )
        val temporaryFile =
            File(
                sessionDirectory,
                "session.index.json.tmp",
            )

        temporaryFile.writeText(
            output.toString(2),
        )

        val temporaryPath =
            temporaryFile.toPath()
        val indexPath =
            indexFile.toPath()

        runCatching {
            Files.move(
                temporaryPath,
                indexPath,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        }.recoverCatching {
            Files.move(
                temporaryPath,
                indexPath,
                StandardCopyOption.REPLACE_EXISTING,
            )
        }.getOrThrow()

        return Result(
            sessionId = manifest,
            entries = entries,
            files = fileIndexes,
            totalRecords = fileIndexes.sumOf { it.records },
            totalBytes = fileIndexes.sumOf { it.byteLength },
            diagnostics = listOf(
                "files=" + fileIndexes.size,
                "indexedEntries=" + entries.size,
                "records=" + fileIndexes.sumOf { it.records },
                "bytes=" + fileIndexes.sumOf { it.byteLength },
            ),
        )
    }

    fun query(
        result: Result,
        startTimestampNanos: Long,
        endTimestampNanos: Long,
        fileName: String? = null,
    ): List<Entry> =
        result.entries
            .asSequence()
            .filter {
                fileName == null ||
                    it.fileName == fileName
            }
            .filter { entry ->
                val timestamp = entry.timestampNanos
                    ?: return@filter false
                timestamp in startTimestampNanos..endTimestampNanos
            }
            .sortedWith(
                compareBy<Entry> { it.timestampNanos ?: Long.MAX_VALUE }
                    .thenBy { it.fileName }
                    .thenBy { it.recordOrdinal },
            )
            .toList()

    private fun indexFile(
        file: File,
        policy: Policy,
    ): Pair<List<Entry>, FileIndex> {
        val entries = mutableListOf<Entry>()
        var record = 0L
        var indexed = 0L
        var firstTimestamp: Long? = null
        var lastTimestamp: Long? = null

        RandomAccessFile(file, "r").use { raf ->
            while (true) {
                val offset = raf.filePointer
                val line = raf.readLine() ?: break
                val end = raf.filePointer
                val length = (
                    end - offset
                ).coerceIn(0L, Int.MAX_VALUE.toLong()).toInt()

                val timestamp = parseTimestamp(
                    line,
                    policy.timestampKeys,
                )

                if (timestamp != null) {
                    firstTimestamp =
                        firstTimestamp?.let { minOf(it, timestamp) }
                            ?: timestamp
                    lastTimestamp =
                        lastTimestamp?.let { maxOf(it, timestamp) }
                            ?: timestamp
                }

                if (
                    record % policy.strideRecords.toLong() == 0L
                ) {
                    entries += Entry(
                        fileName = file.name,
                        recordOrdinal = record,
                        timestampNanos = timestamp,
                        byteOffset = offset,
                        byteLength = length,
                    )
                    indexed += 1L
                }

                record += 1L
            }
        }

        return entries to FileIndex(
            fileName = file.name,
            records = record,
            indexedEntries = indexed,
            firstTimestampNanos = firstTimestamp,
            lastTimestampNanos = lastTimestamp,
            byteLength = file.length(),
        )
    }

    private fun parseTimestamp(
        line: String,
        keys: List<String>,
    ): Long? =
        runCatching {
            val json = JSONObject(line)
            keys.firstNotNullOfOrNull { key ->
                when (val value = json.opt(key)) {
                    is Number -> value.toLong()
                    is String -> value.toLongOrNull()
                    else -> null
                }
            }
        }.getOrNull()

    private fun readSessionId(
        manifest: File,
    ): String? =
        runCatching {
            if (!manifest.isFile) return null
            JSONObject(manifest.readText())
                .optString(
                    "sessionId",
                    null,
                )
        }.getOrNull()
}
