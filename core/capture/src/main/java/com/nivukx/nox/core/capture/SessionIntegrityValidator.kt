package com.nivukx.nox.core.capture

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.abs

/**
 * Motion-capture session integrity gate.
 *
 * A valid NoX session is a synchronized data package, not a video file. The validator therefore
 * checks the manifest contract, timestamp continuity, calibration evidence, inference streams,
 * sensor telemetry and sidecar headers without assuming that a rendered movie exists.
 */
class SessionIntegrityValidator {

    data class Report(
        val valid: Boolean,
        val manifestPresent: Boolean,
        val manifestSchema: Int,
        val calibrationPresent: Boolean,
        val frameRecords: Long,
        val poseRecords: Long,
        val faceRecords: Long,
        val eyeRecords: Long,
        val faceEyeFusionRecords: Long,
        val handRecords: Long,
        val hand3DRecords: Long,
        val telemetryRecords: Long,
        val lightingRecords: Long,
        val imuRecords: Long,
        val focusRecords: Long,
        val motionEventRecords: Long,
        val reconstructionRecords: Long,
        val reconstructionQualityRecords: Long = 0L,
        val motionUncertaintyRecords: Long = 0L,
        val productionPolicyRecords: Long = 0L,
        val crossModalIdentityRecords: Long = 0L,
        val pipelineTelemetryRecords: Long,
        val denseMeshBlocks: Long,
        val denseMeshBytes: Long,
        val faceMeshBlocks: Long,
        val faceMeshBytes: Long,
        val monotonicFrameTimestamps: Boolean,
        val sensorTimestampJitterNanos: Long,
        val rawSidecarPresent: Boolean,
        val depthSidecarPresent: Boolean,
        val checksumManifestPresent: Boolean,
        val checksumValid: Boolean,
        val errors: List<String>,
    )

    fun validate(sessionDirectory: File): Report {
        val errors = mutableListOf<String>()

        if (!sessionDirectory.isDirectory) {
            return Report(
                valid = false,
                manifestPresent = false,
                manifestSchema = 0,
                calibrationPresent = false,
                frameRecords = 0,
                poseRecords = 0,
                faceRecords = 0,
                eyeRecords = 0,
                faceEyeFusionRecords = 0,
                handRecords = 0,
                hand3DRecords = 0,
                telemetryRecords = 0,
                lightingRecords = 0,
                imuRecords = 0,
                focusRecords = 0,
                motionEventRecords = 0,
                reconstructionRecords = 0,
                reconstructionQualityRecords = 0,
                motionUncertaintyRecords = 0,
                productionPolicyRecords = 0,
                crossModalIdentityRecords = 0,
                pipelineTelemetryRecords = 0,
                denseMeshBlocks = 0,
                denseMeshBytes = 0,
                faceMeshBlocks = 0,
                faceMeshBytes = 0,
                monotonicFrameTimestamps = false,
                sensorTimestampJitterNanos = Long.MAX_VALUE,
                rawSidecarPresent = false,
                depthSidecarPresent = false,
                checksumManifestPresent = false,
                checksumValid = false,
                errors = listOf("session directory missing"),
            )
        }

        val manifestFile = File(sessionDirectory, "manifest.json")
        val framesFile = File(sessionDirectory, "frames.ndjson")
        val posesFile = File(sessionDirectory, "poses.ndjson")
        val facesFile = File(sessionDirectory, "faces.ndjson")
        val eyesFile = File(sessionDirectory, "eyes.ndjson")
        val faceEyeFusionFile =
            File(sessionDirectory, "face_eye_fusion.ndjson")
        val handsFile = File(sessionDirectory, "hands.ndjson")
        val hand3dFile = File(sessionDirectory, "hands_3d.ndjson")
        val telemetryFile = File(sessionDirectory, "telemetry.ndjson")
        val lightingFile = File(sessionDirectory, "lighting.ndjson")
        val imuFile = File(sessionDirectory, "imu.ndjson")
        val focusFile = File(sessionDirectory, "focus.ndjson")
        val motionEventsFile =
            File(sessionDirectory, "motion_events.ndjson")
        val reconstructionFile =
            File(sessionDirectory, "reconstruction.ndjson")
        val reconstructionQualityFile =
            File(sessionDirectory, "reconstruction_quality.ndjson")
        val motionUncertaintyFile =
            File(sessionDirectory, "motion_uncertainty.ndjson")
        val productionPolicyFile =
            File(sessionDirectory, "production_policy.ndjson")
        val crossModalIdentityFile =
            File(sessionDirectory, "cross_modal_identity.ndjson")
        val pipelineFile =
            File(sessionDirectory, "pipeline_telemetry.ndjson")
        val denseMeshIndexFile =
            File(sessionDirectory, "dense_mesh_index.ndjson")
        val denseMeshFile =
            File(sessionDirectory, "dense_mesh.bin")
        val faceMeshIndexFile =
            File(sessionDirectory, "face_mesh_index.ndjson")
        val faceMeshFile =
            File(sessionDirectory, "face_mesh.bin")

        val checksumFile =
            File(sessionDirectory, "checksums.json")

        var schema = 0
        var calibrationPresent = false

        if (!manifestFile.isFile) {
            errors += "manifest.json missing"
        } else {
            runCatching {
                JSONObject(manifestFile.readText())
            }.onSuccess { manifest ->
                schema = manifest.optInt("schema", 0)
                calibrationPresent = manifest.has("calibration")

                if (schema < 8) {
                    errors += "unsupported motion session schema=$schema"
                }

                if (schema >= 9) {
                    val denseMesh = manifest.optJSONObject("denseMesh")
                    if (denseMesh == null) {
                        errors += "denseMesh manifest missing"
                    } else {
                        if (
                            denseMesh.optString("file", "") !=
                                "dense_mesh.bin"
                        ) {
                            errors += "dense mesh binary filename mismatch"
                        }
                        if (
                            denseMesh.optString("index", "") !=
                                "dense_mesh_index.ndjson"
                        ) {
                            errors += "dense mesh index filename mismatch"
                        }
                    }
                }

                if (!manifest.has("motionCaptureProfile")) {
                    errors += "motionCaptureProfile missing"
                }

                if (!manifest.has("devices")) {
                    errors += "device manifest missing"
                }

                if (manifest.optBoolean("motionOnly", false) != true) {
                    errors += "motionOnly contract missing or disabled"
                }

                if (
                    manifest.optBoolean("mediaFramesPersisted", true) ||
                    manifest.optString("videoEncoder", "none") != "none"
                ) {
                    errors += "persistent media/video artifact contract violated"
                }
            }.onFailure {
                errors += "manifest.json invalid: " +
                    (it.message ?: "unknown")
            }
        }

        val frameCheck = inspectFrames(framesFile)
        if (!frameCheck.monotonic) {
            errors += "frame sensor timestamps are not strictly monotonic"
        }

        if (frameCheck.count == 0L) {
            errors += "frames.ndjson contains no valid frames"
        }

        if (frameCheck.maxDeltaVariationNanos > 60_000_000L) {
            errors += "frame cadence variation exceeds 60 ms"
        }

        if (!calibrationPresent) {
            errors += "calibration snapshot missing"
        }

        val rawSidecar = sessionDirectory.listFiles()
            ?.firstOrNull {
                it.isFile &&
                    it.name.endsWith(".noxraw", true)
            }

        val depthSidecar = sessionDirectory.listFiles()
            ?.firstOrNull {
                it.isFile &&
                    it.name.endsWith(".noxdepth", true)
            }

        rawSidecar?.let {
            validateMagic(
                file = it,
                expected = "NOXRAW1",
                errors = errors,
            )
        }

        depthSidecar?.let {
            validateMagic(
                file = it,
                expected = "NOXDEP1",
                errors = errors,
            )
        }

        val checksumReport =
            if (checksumFile.isFile) {
                SessionArtifactIntegrityEngine().verify(
                    sessionDirectory,
                )
            } else {
                null
            }

        if (checksumReport != null && !checksumReport.valid) {
            errors +=
                checksumReport.diagnostics +
                    checksumReport.missingArtifacts.map {
                        "checksum-missing:" + it
                    } +
                    checksumReport.mismatchedArtifacts.map {
                        "checksum-mismatch:" + it
                    }
        }

        val motionOnlyManifest =
            runCatching {
                if (!manifestFile.isFile) {
                    false
                } else {
                    JSONObject(manifestFile.readText())
                        .optBoolean("motionOnly", false)
                }
            }.getOrDefault(false)

        if (motionOnlyManifest && rawSidecar != null) {
            errors += "RAW sidecar violates motion-only persistence contract"
        }

        if (motionOnlyManifest && depthSidecar != null) {
            errors += "depth sidecar violates motion-only persistence contract"
        }

        val poseRecords = countRecords(posesFile)
        val faceRecords = countRecords(facesFile)
        val eyeRecords = countRecords(eyesFile)
        val faceEyeFusionRecords = countRecords(faceEyeFusionFile)
        val handRecords = countRecords(handsFile)
        val hand3DRecords = countRecords(hand3dFile)
        val telemetryRecords = countRecords(telemetryFile)
        val lightingRecords = countRecords(lightingFile)
        val imuRecords = countRecords(imuFile)
        val focusRecords = countRecords(focusFile)
        val motionEventRecords = countRecords(motionEventsFile)
        val reconstructionRecords =
            countRecords(reconstructionFile)
        val reconstructionQualityRecords =
            countRecords(reconstructionQualityFile)
        val motionUncertaintyRecords =
            countRecords(motionUncertaintyFile)
        val productionPolicyRecords =
            countRecords(productionPolicyFile)
        val crossModalIdentityRecords =
            countRecords(crossModalIdentityFile)
        val pipelineTelemetryRecords =
            countRecords(pipelineFile)

        var denseMeshBlocks = 0L
        var faceMeshBlocks = 0L
        if (schema >= 12) {
            if (!faceMeshFile.isFile) {
                errors += "face_mesh.bin missing from schema-12 session"
            }
            if (!faceMeshIndexFile.isFile) {
                errors +=
                    "face_mesh_index.ndjson missing from schema-12 session"
            }

            if (
                faceMeshFile.isFile &&
                faceMeshIndexFile.isFile
            ) {
                validateFaceMesh(
                    indexFile = faceMeshIndexFile,
                    binaryFile = faceMeshFile,
                    errors = errors,
                ) { blocks ->
                    faceMeshBlocks = blocks
                }
            }
        }

        if (schema >= 9) {
            if (!denseMeshFile.isFile) {
                errors += "dense_mesh.bin missing from schema-9 session"
            }
            if (!denseMeshIndexFile.isFile) {
                errors += "dense_mesh_index.ndjson missing from schema-9 session"
            }

            if (denseMeshFile.isFile && denseMeshIndexFile.isFile) {
                validateDenseMesh(
                    indexFile = denseMeshIndexFile,
                    binaryFile = denseMeshFile,
                    errors = errors,
                ) { blocks ->
                    denseMeshBlocks = blocks
                }
            }
        }

        if (schema >= 8 && !focusFile.isFile) {
            errors += "focus.ndjson missing from schema-8 session"
        }

        if (poseRecords > 0L && poseRecords > frameCheck.count) {
            errors += "pose record count exceeds captured frame count"
        }

        if (schema >= 14) {
            if (reconstructionQualityRecords == 0L) {
                errors +=
                    "schema 14 requires reconstruction_quality.ndjson records"
            }
            if (motionUncertaintyRecords == 0L) {
                errors +=
                    "schema 14 requires motion_uncertainty.ndjson records"
            }
            if (productionPolicyRecords == 0L) {
                errors +=
                    "schema 14 requires production_policy.ndjson records"
            }
            if (crossModalIdentityRecords == 0L) {
                errors +=
                    "schema 14 requires cross_modal_identity.ndjson records"
            }
        }

        if (hand3DRecords > handRecords && handRecords > 0L) {
            errors += "3D hand record count exceeds hand input count"
        }

        if (telemetryRecords > frameCheck.count * 2L) {
            errors += "sensor telemetry count is implausibly high"
        }

        return Report(
            valid = errors.isEmpty(),
            manifestPresent = manifestFile.isFile,
            manifestSchema = schema,
            calibrationPresent = calibrationPresent,
            frameRecords = frameCheck.count,
            poseRecords = poseRecords,
            faceRecords = faceRecords,
            eyeRecords = eyeRecords,
            faceEyeFusionRecords = faceEyeFusionRecords,
            handRecords = handRecords,
            hand3DRecords = hand3DRecords,
            telemetryRecords = telemetryRecords,
            lightingRecords = lightingRecords,
            imuRecords = imuRecords,
            focusRecords = focusRecords,
            motionEventRecords = motionEventRecords,
            reconstructionRecords = reconstructionRecords,
            reconstructionQualityRecords = reconstructionQualityRecords,
            motionUncertaintyRecords = motionUncertaintyRecords,
            productionPolicyRecords = productionPolicyRecords,
            crossModalIdentityRecords = crossModalIdentityRecords,
            pipelineTelemetryRecords = pipelineTelemetryRecords,
            denseMeshBlocks = denseMeshBlocks,
            denseMeshBytes = denseMeshFile.length().coerceAtLeast(0L),
            faceMeshBlocks = faceMeshBlocks,
            faceMeshBytes = faceMeshFile.length().coerceAtLeast(0L),
            monotonicFrameTimestamps = frameCheck.monotonic,
            sensorTimestampJitterNanos = frameCheck.maxDeltaVariationNanos,
            rawSidecarPresent = rawSidecar?.isFile == true,
            depthSidecarPresent = depthSidecar?.isFile == true,
            checksumManifestPresent = checksumReport != null,
            checksumValid = checksumReport?.valid == true,
            errors = errors,
        )
    }

    private data class FrameCheck(
        val count: Long,
        val monotonic: Boolean,
        val maxDeltaVariationNanos: Long,
    )

    private fun validateFaceMesh(
        indexFile: File,
        binaryFile: File,
        errors: MutableList<String>,
        setBlockCount: (Long) -> Unit,
    ) {
        var blockCount = 0L
        var previousOffset = -1L

        runCatching {
            RandomAccessFile(binaryFile, "r").use { binary ->
                indexFile.useLines { lines ->
                    lines.forEach { line ->
                        if (line.isBlank()) return@forEach

                        val json = JSONObject(line)
                        val offset =
                            json.optLong("offsetBytes", -1L)
                        val length =
                            json.optLong("lengthBytes", -1L)
                        val points =
                            json.optInt("points", -1)
                        val triangles =
                            json.optInt("triangles", -1)

                        if (
                            offset < 0L ||
                            length < 64L ||
                            points !in 0..468 ||
                            triangles !in 0..960
                        ) {
                            errors +=
                                "invalid face mesh block metadata"
                            return@forEach
                        }

                        if (offset <= previousOffset) {
                            errors +=
                                "face mesh block offsets are not monotonic"
                        }

                        if (
                            offset > binary.length() ||
                            length > binary.length() - offset
                        ) {
                            errors +=
                                "face mesh block exceeds binary file length"
                            return@forEach
                        }

                        val expectedLength =
                            64L +
                                16L * points +
                                12L * triangles

                        if (length != expectedLength) {
                            errors +=
                                "face mesh block length mismatch at offset " +
                                    offset
                        }

                        binary.seek(offset)
                        val magic = binary.readInt()
                        val version = binary.readInt()

                        if (magic != 0x4E58464D) {
                            errors +=
                                "face mesh magic mismatch at offset " +
                                    offset
                        }

                        if (version != 1) {
                            errors +=
                                "unsupported face mesh block version=" +
                                    version
                        }

                        // After magic + version, the recorded header stores:
                        // timestamp (8) + trackingId (4) + width (4) + height (4) + rotation (4)
                        // + four bounds (16) + confidence (4) + point coverage (4) = 48 bytes.
                        // The point and triangle counts are the next two Int fields.
                        binary.skipBytes(48)
                        val pointCountInBlock =
                            binary.readInt()
                        val triangleCountInBlock =
                            binary.readInt()

                        if (pointCountInBlock != points) {
                            errors +=
                                "face mesh point count mismatch at offset " +
                                    offset
                        }

                        if (triangleCountInBlock != triangles) {
                            errors +=
                                "face mesh triangle count mismatch at offset " +
                                    offset
                        }

                        previousOffset = offset
                        blockCount += 1L
                    }
                }
            }
        }.onFailure {
            errors +=
                "face mesh binary validation failed: " +
                    (it.message ?: "unknown error")
        }

        setBlockCount(blockCount)
    }

    private fun inspectFrames(
        file: File,
    ): FrameCheck {
        if (!file.isFile) {
            return FrameCheck(
                count = 0L,
                monotonic = false,
                maxDeltaVariationNanos = Long.MAX_VALUE,
            )
        }

        var count = 0L
        var previousTimestamp = Long.MIN_VALUE
        var previousDelta = -1L
        var maxVariation = 0L
        var monotonic = true

        runCatching {
            file.useLines { lines ->
                lines.forEach { line ->
                    if (line.isBlank()) return@forEach

                    val json = runCatching {
                        JSONObject(line)
                    }.getOrNull() ?: run {
                        monotonic = false
                        return@forEach
                    }

                    val timestamp = json.optLong(
                        "sensorTimestampNanos",
                        Long.MIN_VALUE,
                    )

                    if (timestamp == Long.MIN_VALUE) {
                        monotonic = false
                        return@forEach
                    }

                    if (
                        previousTimestamp != Long.MIN_VALUE &&
                        timestamp <= previousTimestamp
                    ) {
                        monotonic = false
                    }

                    if (previousTimestamp != Long.MIN_VALUE) {
                        val delta = timestamp - previousTimestamp
                        if (previousDelta >= 0L) {
                            maxVariation = maxOf(
                                maxVariation,
                                abs(delta - previousDelta),
                            )
                        }
                        previousDelta = delta
                    }

                    previousTimestamp = timestamp
                    count += 1L
                }
            }
        }.onFailure {
            monotonic = false
        }

        return FrameCheck(
            count = count,
            monotonic = monotonic,
            maxDeltaVariationNanos = maxVariation,
        )
    }

    private fun validateDenseMesh(
        indexFile: File,
        binaryFile: File,
        errors: MutableList<String>,
        setBlockCount: (Long) -> Unit,
    ) {
        var blockCount = 0L
        var previousOffset = -1L

        runCatching {
            RandomAccessFile(binaryFile, "r").use { binary ->
                indexFile.useLines { lines ->
                    lines.forEach { line ->
                        if (line.isBlank()) return@forEach

                        val json = JSONObject(line)
                        val offset = json.optLong(
                            "offsetBytes",
                            -1L,
                        )
                        val length = json.optLong(
                            "lengthBytes",
                            -1L,
                        )
                        val vertices = json.optInt(
                            "vertices",
                            -1,
                        )
                        val triangles = json.optInt(
                            "triangles",
                            -1,
                        )

                        if (offset < 0L || length < 28L) {
                            errors += "invalid dense mesh block metadata"
                            return@forEach
                        }

                        if (offset <= previousOffset) {
                            errors +=
                                "dense mesh block offsets are not monotonic"
                        }

                        if (
                            offset > binary.length() ||
                            length > binary.length() - offset
                        ) {
                            errors +=
                                "dense mesh block exceeds binary file length"
                            return@forEach
                        }

                        val expectedLength =
                            28L +
                                12L * vertices.coerceAtLeast(0) +
                                12L * triangles.coerceAtLeast(0)

                        if (length != expectedLength) {
                            errors +=
                                "dense mesh block length mismatch at offset " +
                                    offset
                        }

                        binary.seek(offset)
                        val magic = binary.readInt()
                        val version = binary.readInt()
                        val timestamp = binary.readLong()
                        val confidence = binary.readFloat()
                        val binaryVertices = binary.readInt()
                        val binaryTriangles = binary.readInt()

                        if (magic != 0x4E584D31) {
                            errors +=
                                "invalid dense mesh magic at offset " +
                                    offset
                        }
                        if (version != 1) {
                            errors +=
                                "unsupported dense mesh version=" +
                                    version
                        }
                        if (binaryVertices != vertices) {
                            errors +=
                                "dense mesh vertex count mismatch at offset " +
                                    offset
                        }
                        if (binaryTriangles != triangles) {
                            errors +=
                                "dense mesh triangle count mismatch at offset " +
                                    offset
                        }
                        if (timestamp < 0L) {
                            errors +=
                                "negative dense mesh timestamp at offset " +
                                    offset
                        }
                        if (!confidence.isFinite()) {
                            errors +=
                                "non-finite dense mesh confidence at offset " +
                                    offset
                        }

                        previousOffset = offset
                        blockCount += 1L
                    }
                }
            }
        }.onFailure {
            errors +=
                "dense mesh validation failure: " +
                    (it.message ?: "unknown")
        }

        setBlockCount(blockCount)
    }

    private fun validateMagic(
        file: File,
        expected: String,
        errors: MutableList<String>,
    ) {
        runCatching {
            file.inputStream().use { input ->
                val bytes = ByteArray(expected.length)
                val count = input.read(bytes)
                val actual = bytes.toString(
                    Charsets.US_ASCII,
                )
                if (
                    count != expected.length ||
                    actual != expected
                ) {
                    errors +=
                        "invalid sidecar header: " +
                            file.name
                }
            }
        }.onFailure {
            errors +=
                "sidecar read failure: " +
                    file.name
        }
    }

    private fun countRecords(
        file: File,
    ): Long {
        if (!file.isFile) return 0L

        var count = 0L
        runCatching {
            file.useLines { lines ->
                lines.forEach {
                    if (it.isNotBlank()) count += 1L
                }
            }
        }.onFailure {
            count = 0L
        }
        return count
    }
}
