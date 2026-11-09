package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraExtrinsics
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

object CalibrationSnapshotWireCodec {
    private const val VERSION = 4
    private const val VERSION_3 = 3
    private const val VERSION_2 = 2
    private const val LEGACY_VERSION = 1
    private const val MAX_CAMERAS = 128
    private const val MAX_ID_BYTES = 96
    private const val MAX_PROVENANCE_BYTES = 64

    fun encode(
        snapshot: CalibrationSnapshot,
        requestedVersion: Int = VERSION,
    ): ByteArray {
        require(requestedVersion in LEGACY_VERSION..VERSION)
        val wireVersion = requestedVersion
        require(
            snapshot.intrinsics.keys ==
                snapshot.extrinsics.keys,
        ) {
            "Calibration intrinsic/extrinsic camera keys must match"
        }
        require(snapshot.intrinsics.size in 1..MAX_CAMERAS)

        val reference = snapshot.extrinsics.values
            .firstOrNull()
            ?.referenceDevice
            ?.value
            ?.toByteArray(StandardCharsets.UTF_8)
            ?.let { bytes ->
                bytes.copyOf(
                    minOf(
                        bytes.size,
                        MAX_ID_BYTES,
                    ),
                )
            }
            ?: ByteArray(0)

        val entries = snapshot.intrinsics.keys.sortedBy { it.value }
        val provenanceBytes =
            if (wireVersion >= VERSION) {
                snapshot.calibrationMethod
                    .toByteArray(StandardCharsets.UTF_8)
                    .copyOf(
                        minOf(
                            snapshot.calibrationMethod
                                .toByteArray(StandardCharsets.UTF_8)
                                .size,
                            MAX_PROVENANCE_BYTES,
                        ),
                    )
            } else {
                ByteArray(0)
            }

        val size =
            4 + 4 + 8 + 8 + 4 + reference.size +
                if (wireVersion >= VERSION) {
                    4 + provenanceBytes.size +
                        5 * 8 + 4
                } else {
                    0
                } +
                entries.sumOf { id ->
            val idBytes = id.value.toByteArray(StandardCharsets.UTF_8)
                .copyOf(minOf(id.value.toByteArray(StandardCharsets.UTF_8).size, MAX_ID_BYTES))
            4 + idBytes.size +
                9 * 8 + 2 * 4 +
                3 * 8 + 4 * 8 +
                if (wireVersion >= VERSION_2) 4 * 8 else 0 +
                if (wireVersion >= VERSION) 8 else 0
        }

        val buffer = ByteBuffer.allocate(size).order(ByteOrder.BIG_ENDIAN)
        buffer.putInt(wireVersion)
        buffer.putInt(entries.size)
        buffer.putLong(snapshot.calibratedAtNanos)
        buffer.putDouble(snapshot.reprojectionErrorPx)
        buffer.putInt(reference.size)
        buffer.put(reference)

        if (wireVersion >= VERSION) {
            buffer.putInt(provenanceBytes.size)
            buffer.put(provenanceBytes)
            buffer.putDouble(snapshot.nativeIntrinsicRefinementConfidence)
            buffer.putDouble(snapshot.nativeIntrinsicRefinementImprovementPx)
            buffer.putDouble(snapshot.timingAuthorityConfidence)
            buffer.putDouble(snapshot.timingAuthorityResidualP95Milliseconds)
            buffer.putDouble(snapshot.reprojectionErrorPx)
            var flags = 0
            if (snapshot.nativeIntrinsicRefinementApplied) {
                flags = flags or 0x01
            }
            buffer.putInt(flags)
        }

        entries.forEach { id ->
            val rawId = id.value.toByteArray(StandardCharsets.UTF_8)
            val idBytes = rawId.copyOf(minOf(rawId.size, MAX_ID_BYTES))
            val i = snapshot.intrinsics.getValue(id)
            val e = snapshot.extrinsics.getValue(id)

            buffer.putInt(idBytes.size)
            buffer.put(idBytes)

            buffer.putDouble(i.fx)
            buffer.putDouble(i.fy)
            buffer.putDouble(i.cx)
            buffer.putDouble(i.cy)
            buffer.putDouble(i.k1)
            buffer.putDouble(i.k2)
            buffer.putDouble(i.p1)
            buffer.putDouble(i.p2)
            buffer.putDouble(i.k3)
            buffer.putInt(i.imageWidth)
            buffer.putInt(i.imageHeight)

            val p = e.pose.position
            buffer.putDouble(p.x)
            buffer.putDouble(p.y)
            buffer.putDouble(p.z)

            val q = e.pose.rotation.normalized()
            buffer.putDouble(q.w)
            buffer.putDouble(q.x)
            buffer.putDouble(q.y)
            buffer.putDouble(q.z)
            if (wireVersion >= VERSION_2) {
                val uncertainty =
                    snapshot.uncertaintyProfiles[id]
                        ?: CalibrationUncertaintyProfile()
                buffer.putDouble(uncertainty.translationSigmaMeters)
                buffer.putDouble(uncertainty.rotationSigmaRadians)
                buffer.putDouble(uncertainty.intrinsicRelativeSigma)
                buffer.putDouble(uncertainty.evidenceConfidence)
            }
            if (wireVersion >= VERSION) {
                buffer.putDouble(
                    snapshot
                        .calibrationFocusDistanceDiopters[id]
                        ?: -1.0,
                )
            }
        }

        return buffer.array()
    }

    fun decode(bytes: ByteArray): CalibrationSnapshot {
        require(bytes.size >= 28) { "Calibration result payload too small" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        val version = buffer.int
        require(
            version == VERSION ||
                version == VERSION_3 ||
                version == VERSION_2 ||
                version == LEGACY_VERSION,
        ) { "Unsupported calibration result version: $version" }

        val count = buffer.int
        require(count in 1..MAX_CAMERAS) { "Invalid calibration camera count" }

        val calibratedAt = buffer.long
        val reprojectionError = buffer.double

        val referenceLength = buffer.int
        require(referenceLength in 0..MAX_ID_BYTES)
        val referenceBytes = ByteArray(referenceLength)
        buffer.get(referenceBytes)
        val trimmedReference =
            referenceBytes.takeWhile { it.toInt() != 0 }
                .toByteArray()
        val referenceDevice =
            trimmedReference
                .takeIf { it.isNotEmpty() }
                ?.let {
                    DeviceId(
                        String(
                            it,
                            StandardCharsets.UTF_8,
                        ),
                    )
                }

        var calibrationMethod = "UNKNOWN"
        var nativeIntrinsicRefinementApplied = false
        var nativeIntrinsicRefinementConfidence = 0.0
        var nativeIntrinsicRefinementImprovementPx = 0.0
        var timingAuthorityConfidence = 0.0
        var timingAuthorityResidualP95Milliseconds = 50.0

        if (version >= VERSION) {
            val provenanceLength = buffer.int
            require(
                provenanceLength in 1..MAX_PROVENANCE_BYTES,
            )
            val provenanceBytes =
                ByteArray(provenanceLength)
            buffer.get(provenanceBytes)
            calibrationMethod =
                String(
                    provenanceBytes,
                    StandardCharsets.UTF_8,
                )
            require(calibrationMethod.isNotBlank())
            nativeIntrinsicRefinementConfidence =
                buffer.double
            nativeIntrinsicRefinementImprovementPx =
                buffer.double
            timingAuthorityConfidence =
                buffer.double
            timingAuthorityResidualP95Milliseconds =
                buffer.double
            val duplicatedReprojection =
                buffer.double
            require(
                duplicatedReprojection ==
                    reprojectionError,
            ) {
                "Calibration provenance reprojection mismatch"
            }
            val flags = buffer.int
            require(flags and 0xFE == 0)
            nativeIntrinsicRefinementApplied =
                flags and 0x01 != 0
        }

        val intrinsics = LinkedHashMap<DeviceId, CameraIntrinsics>(count)
        val extrinsics = LinkedHashMap<DeviceId, CameraExtrinsics>(count)
        val uncertaintyProfiles =
            LinkedHashMap<DeviceId, CalibrationUncertaintyProfile>(count)
        val calibrationFocusDistanceDiopters =
            LinkedHashMap<DeviceId, Double>(count)

        repeat(count) {
            val idLength = buffer.int
            require(idLength in 1..MAX_ID_BYTES)
            val idBytes = ByteArray(idLength)
            buffer.get(idBytes)
            val id = DeviceId(String(idBytes, StandardCharsets.UTF_8))

            val camera = CameraIntrinsics(
                fx = buffer.double,
                fy = buffer.double,
                cx = buffer.double,
                cy = buffer.double,
                k1 = buffer.double,
                k2 = buffer.double,
                p1 = buffer.double,
                p2 = buffer.double,
                k3 = buffer.double,
                imageWidth = buffer.int,
                imageHeight = buffer.int,
            )

            val position = Vec3(
                buffer.double,
                buffer.double,
                buffer.double,
            )
            val rotation = Quat(
                buffer.double,
                buffer.double,
                buffer.double,
                buffer.double,
            ).normalized()

            intrinsics[id] = camera
            extrinsics[id] = CameraExtrinsics(
                pose = Pose(position, rotation),
                referenceDevice = referenceDevice,
            )

            uncertaintyProfiles[id] =
                if (version >= VERSION_2) {
                    CalibrationUncertaintyProfile(
                        translationSigmaMeters = buffer.double,
                        rotationSigmaRadians = buffer.double,
                        intrinsicRelativeSigma = buffer.double,
                        evidenceConfidence = buffer.double,
                    )
                } else {
                    CalibrationUncertaintyProfile()
                }
            if (version >= VERSION) {
                val focusDiopters = buffer.double
                if (focusDiopters >= 0.0) {
                    calibrationFocusDistanceDiopters[id] =
                        focusDiopters
                } else {
                    require(focusDiopters == -1.0)
                }
            }
        }

        require(!buffer.hasRemaining()) { "Trailing calibration result bytes" }

        return CalibrationSnapshot(
            intrinsics = intrinsics,
            extrinsics = extrinsics,
            reprojectionErrorPx = reprojectionError,
            calibratedAtNanos = calibratedAt,
            uncertaintyProfiles = uncertaintyProfiles,
            calibrationFocusDistanceDiopters =
                calibrationFocusDistanceDiopters,
            calibrationMethod =
                calibrationMethod,
            nativeIntrinsicRefinementApplied =
                nativeIntrinsicRefinementApplied,
            nativeIntrinsicRefinementConfidence =
                nativeIntrinsicRefinementConfidence,
            nativeIntrinsicRefinementImprovementPx =
                nativeIntrinsicRefinementImprovementPx,
            timingAuthorityConfidence =
                timingAuthorityConfidence,
            timingAuthorityResidualP95Milliseconds =
                timingAuthorityResidualP95Milliseconds,
        )
    }
}
