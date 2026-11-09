package com.nivukx.nox.core.capture

import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.Locale

/**
 * Tamper-evident integrity sealer for NoX capture evidence.
 *
 * A session may contain motion, video, depth, calibration and metadata artifacts. This component
 * hashes every available artifact and combines the hashes with the effective configuration fingerprint
 * into one deterministic SHA-256 root. It is not a digital signature: authenticity additionally
 * requires an external key/signing service. The sealer provides reproducible content integrity.
 */
object EvidenceIntegritySealer {

    data class ArtifactDigest(
        val role: String,
        val path: String,
        val exists: Boolean,
        val byteCount: Long,
        val sha256: String?,
        val error: String?,
    )

    data class SealedEvidence(
        val configurationFingerprint: String,
        val artifacts: List<ArtifactDigest>,
        val rootSha256: String,
        val sealed: Boolean,
        val warnings: List<String>,
    )

    fun seal(
        configurationFingerprint: String,
        artifacts: List<Pair<String, File?>>,
        chunkBytes: Int = 1 shl 20,
    ): SealedEvidence {
        require(configurationFingerprint.isNotBlank())
        require(chunkBytes in 4 * 1024..8 * 1024 * 1024)

        val digests =
            artifacts
                .distinctBy { it.first }
                .sortedBy { it.first }
                .map { (role, file) ->
                    digestArtifact(
                        role = role,
                        file = file,
                        chunkBytes = chunkBytes,
                    )
                }

        val warnings =
            buildList {
                digests
                    .filter { !it.exists || it.sha256 == null }
                    .forEach {
                        add(
                            "INTEGRITY_ARTIFACT_UNAVAILABLE:" +
                                it.role,
                        )
                    }
            }

        val canonical =
            buildString {
                append("NXSEAL1|")
                append(configurationFingerprint)
                digests.forEach { digest ->
                    append('|')
                    append(digest.role)
                    append('|')
                    append(digest.path)
                    append('|')
                    append(digest.exists)
                    append('|')
                    append(digest.byteCount)
                    append('|')
                    append(digest.sha256 ?: "null")
                }
            }

        return SealedEvidence(
            configurationFingerprint =
                configurationFingerprint,
            artifacts = digests,
            rootSha256 =
                sha256(
                    canonical.toByteArray(
                        Charsets.UTF_8,
                    ),
                ),
            sealed =
                warnings.isEmpty() &&
                    digests.isNotEmpty(),
            warnings =
                warnings.distinct(),
        )
    }

    fun verifyFiles(
        sealed: SealedEvidence,
        chunkBytes: Int = 1 shl 20,
    ): Boolean {
        require(chunkBytes in 4 * 1024..8 * 1024 * 1024)

        if (!verify(sealed)) {
            return false
        }

        return sealed.artifacts.all { artifact ->
            if (!artifact.exists || artifact.sha256 == null) {
                return false
            }

            val file = File(artifact.path)
            val fresh =
                digestArtifact(
                    role = artifact.role,
                    file = file,
                    chunkBytes = chunkBytes,
                )

            fresh.exists &&
                fresh.byteCount == artifact.byteCount &&
                fresh.sha256 == artifact.sha256
        }
    }

    fun verify(
        sealed: SealedEvidence,
    ): Boolean {
        val canonical =
            buildString {
                append("NXSEAL1|")
                append(sealed.configurationFingerprint)
                sealed.artifacts
                    .sortedBy { it.role }
                    .forEach { digest ->
                        append('|')
                        append(digest.role)
                        append('|')
                        append(digest.path)
                        append('|')
                        append(digest.exists)
                        append('|')
                        append(digest.byteCount)
                        append('|')
                        append(digest.sha256 ?: "null")
                    }
            }

        return sha256(
            canonical.toByteArray(
                Charsets.UTF_8,
            ),
        ) == sealed.rootSha256
    }

    private fun digestArtifact(
        role: String,
        file: File?,
        chunkBytes: Int,
    ): ArtifactDigest {
        if (file == null) {
            return ArtifactDigest(
                role = role,
                path = "<null>",
                exists = false,
                byteCount = 0L,
                sha256 = null,
                error = "artifact path is null",
            )
        }

        if (!file.isFile) {
            return ArtifactDigest(
                role = role,
                path = file.absolutePath,
                exists = false,
                byteCount = 0L,
                sha256 = null,
                error = "file does not exist",
            )
        }

        return runCatching {
            val digest =
                MessageDigest.getInstance("SHA-256")
            val buffer =
                ByteArray(chunkBytes)
            var bytes = 0L

            FileInputStream(file).use { input ->
                while (true) {
                    val read =
                        input.read(buffer)
                    if (read <= 0) break
                    digest.update(
                        buffer,
                        0,
                        read,
                    )
                    bytes += read.toLong()
                }
            }

            ArtifactDigest(
                role = role,
                path = file.absolutePath,
                exists = true,
                byteCount = bytes,
                sha256 =
                    digest.digest().joinToString("") {
                        "%02x".format(
                            Locale.US,
                            it.toInt() and 0xFF,
                        )
                    },
                error = null,
            )
        }.getOrElse { error ->
            ArtifactDigest(
                role = role,
                path = file.absolutePath,
                exists = true,
                byteCount = runCatching {
                    file.length()
                }.getOrDefault(0L),
                sha256 = null,
                error =
                    error.message
                        ?: error.javaClass.simpleName,
            )
        }
    }

    private fun sha256(
        bytes: ByteArray,
    ): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") {
                "%02x".format(
                    Locale.US,
                    it.toInt() and 0xFF,
                )
            }
}
