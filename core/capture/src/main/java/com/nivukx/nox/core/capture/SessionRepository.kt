package com.nivukx.nox.core.capture

import android.content.Context
import java.io.File

class SessionRepository(context: Context) {
    private val integrity = SessionIntegrityValidator()
    private val recovery = MotionSessionRecoveryEngine(integrity)
    private val indexEngine = MotionSessionIndexEngine()
    data class SessionSummary(
        val id: String,
        val path: File,
        val bytes: Long,
        val createdAtMillis: Long,
        val complete: Boolean,
    )

    private val root = File(context.filesDir, "sessions").apply { mkdirs() }

    fun list(): List<SessionSummary> =
        root.listFiles { file -> file.isDirectory }
            .orEmpty()
            .map { dir ->
                runCatching { recovery.recover(dir) }
                val indexFile = File(dir, "session.index.json")
                if (indexNeedsRefresh(dir, indexFile)) {
                    runCatching { indexEngine.build(dir) }
                }
                val manifest = File(dir, "manifest.json")
                SessionSummary(
                    id = dir.name,
                    path = dir,
                    bytes = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() },
                    createdAtMillis = manifest.lastModified().takeIf { it > 0 } ?: dir.lastModified(),
                    complete = integrity.validate(dir).valid,
                )
            }
            .sortedByDescending { it.createdAtMillis }
}


private fun indexNeedsRefresh(
    sessionDirectory: File,
    indexFile: File,
): Boolean {
    if (!indexFile.isFile) return true
    val indexTimestamp = indexFile.lastModified()
    return sessionDirectory
        .walkTopDown()
        .filter { it.isFile && it != indexFile }
        .any { it.lastModified() > indexTimestamp }
}
