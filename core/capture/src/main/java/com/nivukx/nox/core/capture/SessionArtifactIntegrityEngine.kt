package com.nivukx.nox.core.capture

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/**
 * Cryptographic artifact integrity for motion-only capture sessions.
 *
 * A session contains many independently inspectable streams. A single manifest
 * checksum is insufficient because one damaged stream could be hidden inside a
 * valid directory. This engine records SHA-256 for each persisted artifact and
 * can verify the package without loading motion payloads into memory.
 */
class SessionArtifactIntegrityEngine {

    data class Artifact(
        val relativePath: String,
        val byteLength: Long,
        val sha256: String,
    )

    data class Report(
        val valid: Boolean,
        val manifestPresent: Boolean,
        val artifactsChecked: Int,
        val missingArtifacts: List<String>,
        val mismatchedArtifacts: List<String>,
        val diagnostics: List<String>,
    )

    fun writeManifest(
        sessionDirectory: File,
        fileName: String = "checksums.json",
    ): File {
        require(sessionDirectory.isDirectory)

        val artifacts =
            sessionDirectory
                .walkTopDown()
                .filter { file ->
                    file.isFile &&
                        file.name != fileName &&
                        !file.name.endsWith(".tmp") &&
                        !file.name.endsWith(".partial")
                }
                .map { file ->
                    artifact(
                        sessionDirectory = sessionDirectory,
                        file = file,
                    )
                }
                .sortedBy { it.relativePath }
                .toList()

        val output =
            JSONObject().apply {
                put("schema", 1)
                put("algorithm", "SHA-256")
                put(
                    "sessionDirectory",
                    sessionDirectory.name,
                )
                put(
                    "artifacts",
                    JSONArray().apply {
                        artifacts.forEach { artifact ->
                            put(
                                JSONObject().apply {
                                    put(
                                        "path",
                                        artifact.relativePath,
                                    )
                                    put(
                                        "bytes",
                                        artifact.byteLength,
                                    )
                                    put(
                                        "sha256",
                                        artifact.sha256,
                                    )
                                },
                            )
                        }
                    },
                )
            }

        val target =
            File(
                sessionDirectory,
                fileName,
            )
        val temporary =
            File(
                sessionDirectory,
                "$fileName.tmp",
            )

        temporary.writeText(
            output.toString(2),
            Charsets.UTF_8,
        )

        check(
            temporary.renameTo(target) ||
                runCatching {
                    target.delete()
                    temporary.renameTo(target)
                }.getOrDefault(false),
        ) {
            "Unable to finalize session checksum manifest"
        }

        return target
    }

    fun verify(
        sessionDirectory: File,
        fileName: String = "checksums.json",
    ): Report {
        val checksumFile =
            File(
                sessionDirectory,
                fileName,
            )

        if (!checksumFile.isFile) {
            return Report(
                valid = false,
                manifestPresent = false,
                artifactsChecked = 0,
                missingArtifacts = emptyList(),
                mismatchedArtifacts = emptyList(),
                diagnostics = listOf(
                    "checksum manifest missing",
                ),
            )
        }

        val missing = mutableListOf<String>()
        val mismatched = mutableListOf<String>()
        var checked = 0

        val diagnostics = mutableListOf<String>()

        val manifest =
            runCatching {
                JSONObject(
                    checksumFile.readText(),
                )
            }.getOrElse { error ->
                return Report(
                    valid = false,
                    manifestPresent = true,
                    artifactsChecked = 0,
                    missingArtifacts = emptyList(),
                    mismatchedArtifacts = emptyList(),
                    diagnostics = listOf(
                        "checksum manifest invalid: " +
                            (error.message ?: "unknown"),
                    ),
                )
            }

        val schema =
            manifest.optInt("schema", 0)
        val algorithm =
            manifest.optString(
                "algorithm",
                "",
            )

        if (schema != 1) {
            diagnostics +=
                "unsupported checksum manifest schema=$schema"
        }

        if (!algorithm.equals("SHA-256", true)) {
            diagnostics +=
                "unsupported checksum algorithm=$algorithm"
        }

        val artifacts =
            manifest.optJSONArray(
                "artifacts",
            )

        if (artifacts == null) {
            diagnostics += "artifact list missing"
            return Report(
                valid = false,
                manifestPresent = true,
                artifactsChecked = 0,
                missingArtifacts = emptyList(),
                mismatchedArtifacts = emptyList(),
                diagnostics = diagnostics,
            )
        }

        for (index in 0 until artifacts.length()) {
            val item =
                artifacts.optJSONObject(index)
                    ?: continue

            val relativePath =
                item.optString(
                    "path",
                    "",
                )
            val expected =
                item.optString(
                    "sha256",
                    "",
                )
            val expectedBytes =
                item.optLong(
                    "bytes",
                    -1L,
                )

            if (
                relativePath.isBlank() ||
                expected.length != 64 ||
                expectedBytes < 0L
            ) {
                diagnostics +=
                    "invalid checksum entry at index=$index"
                continue
            }

            val file =
                File(
                    sessionDirectory,
                    relativePath,
                )

            if (!file.isFile) {
                missing += relativePath
                continue
            }

            if (file.length() != expectedBytes) {
                mismatched += relativePath
                continue
            }

            val actual =
                sha256(file)

            checked += 1

            if (!actual.equals(expected, true)) {
                mismatched += relativePath
            }
        }

        return Report(
            valid =
                diagnostics.isEmpty() &&
                    missing.isEmpty() &&
                    mismatched.isEmpty(),
            manifestPresent = true,
            artifactsChecked = checked,
            missingArtifacts = missing,
            mismatchedArtifacts = mismatched,
            diagnostics = diagnostics,
        )
    }

    private fun artifact(
        sessionDirectory: File,
        file: File,
    ): Artifact {
        val relative =
            sessionDirectory
                .toPath()
                .relativize(
                    file.toPath(),
                )
                .toString()
                .replace(
                    File.separatorChar,
                    '/',
                )

        return Artifact(
            relativePath = relative,
            byteLength = file.length(),
            sha256 = sha256(file),
        )
    }

    private fun sha256(
        file: File,
    ): String {
        val digest =
            MessageDigest.getInstance(
                "SHA-256",
            )

        FileInputStream(file).use { input ->
            val buffer =
                ByteArray(256 * 1024)

            while (true) {
                val read =
                    input.read(buffer)

                if (read <= 0) break

                digest.update(
                    buffer,
                    0,
                    read,
                )
            }
        }

        return digest
            .digest()
            .joinToString("") { byte ->
                "%02x".format(
                    byte.toInt() and 0xff,
                )
            }
    }
}
