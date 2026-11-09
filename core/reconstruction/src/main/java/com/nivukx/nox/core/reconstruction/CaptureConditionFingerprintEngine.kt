package com.nivukx.nox.core.reconstruction

import java.security.MessageDigest
import java.util.Locale
import kotlin.math.round

/**
 * Canonical capture-condition fingerprinting for paired metrology.
 *
 * Values are deliberately quantized before hashing so insignificant sensor noise does not make
 * equivalent sessions look different, while meaningful rig/configuration changes do.
 */
class CaptureConditionFingerprintEngine {

    data class Input(
        val rigId: String,
        val cameraCount: Int,
        val captureFps: Int,
        val inferenceFps: Int,
        val baselineMeters: Double,
        val workingDistanceMeters: Double,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val lightingLux: Double,
        val colorTemperatureKelvin: Double,
        val roomTemperatureCelsius: Double,
        val exposureMicros: Long? = null,
        val iso: Int? = null,
        val rollingShutterFraction: Double = 0.0,
        val achievedCaptureFps: Double? = null,
        val highSpeedSessionActive: Boolean = false,
        val sessionVerification: String? = null,
        val lensProfileId: String? = null,
        val depthSourceId: String? = null,
        val modelProfileId: String? = null,
    ) {
        init {
            require(rigId.isNotBlank())
            require(cameraCount > 0)
            require(captureFps > 0)
            require(inferenceFps in 1..captureFps)
            listOf(
                baselineMeters,
                workingDistanceMeters,
                calibrationConfidence,
                synchronizationConfidence,
                lightingLux,
                colorTemperatureKelvin,
                roomTemperatureCelsius,
                rollingShutterFraction,
            ).forEach {
                require(it.isFinite())
            }
            require(baselineMeters > 0.0)
            require(workingDistanceMeters > 0.0)
            require(calibrationConfidence in 0.0..1.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(lightingLux >= 0.0)
            require(colorTemperatureKelvin > 0.0)
            require(rollingShutterFraction in 0.0..1.0)
            exposureMicros?.let { require(it > 0L) }
            iso?.let { require(it > 0) }
            achievedCaptureFps?.let {
                require(
                    it.isFinite() &&
                        it > 0.0 &&
                        it <= 1000.0,
                )
            }
            sessionVerification?.let {
                require(it.isNotBlank())
            }
        }
    }

    data class Result(
        val fingerprint: String,
        val quantizedConfigurationHash: String,
        val canonicalConfiguration: String,
    )

    fun create(input: Input): Result {
        val canonical =
            buildCanonicalConfiguration(input)

        val hash =
            sha256(canonical)

        return Result(
            fingerprint = "nox-cond-v2-" + hash.take(24),
            quantizedConfigurationHash = hash,
            canonicalConfiguration = canonical,
        )
    }

    private fun buildCanonicalConfiguration(
        input: Input,
    ): String =
        listOf(
            field("rig", input.rigId),
            field("cameraCount", input.cameraCount.toString()),
            field("captureFps", input.captureFps.toString()),
            field("inferenceFps", input.inferenceFps.toString()),
            field("baselineMm", quantize(input.baselineMeters * 1000.0, 1.0)),
            field(
                "workingDistanceMm",
                quantize(input.workingDistanceMeters * 1000.0, 1.0),
            ),
            field(
                "calibrationPct",
                quantize(input.calibrationConfidence * 1000.0, 0.1),
            ),
            field(
                "syncPct",
                quantize(input.synchronizationConfidence * 1000.0, 0.1),
            ),
            field(
                "lightingLux",
                quantize(input.lightingLux, 1.0),
            ),
            field(
                "cctK",
                quantize(input.colorTemperatureKelvin, 1.0),
            ),
            field(
                "roomC",
                quantize(input.roomTemperatureCelsius, 0.1),
            ),
            field(
                "exposureUs",
                input.exposureMicros?.toString() ?: "auto",
            ),
            field(
                "iso",
                input.iso?.toString() ?: "auto",
            ),
            field(
                "rollingShutter",
                quantize(
                    input.rollingShutterFraction,
                    0.001,
                ),
            ),
            field(
                "achievedCaptureFps",
                input.achievedCaptureFps?.let {
                    quantize(it, 0.5)
                } ?: "unmeasured",
            ),
            field(
                "highSpeedSessionActive",
                input.highSpeedSessionActive.toString(),
            ),
            field(
                "sessionVerification",
                input.sessionVerification ?: "unknown",
            ),
            field("lens", input.lensProfileId ?: "default"),
            field("depth", input.depthSourceId ?: "none"),
            field("model", input.modelProfileId ?: "default"),
        ).joinToString("|")

    private fun field(
        key: String,
        value: String,
    ): String =
        key + "=" + value

    private fun quantize(
        value: Double,
        step: Double,
    ): String {
        val quantized =
            round(value / step) * step

        return String.format(
            Locale.US,
            "%.6f",
            quantized,
        )
    }

    private fun sha256(
        value: String,
    ): String {
        val digest =
            MessageDigest.getInstance("SHA-256")
        return digest.digest(
            value.toByteArray(Charsets.UTF_8),
        ).joinToString("") {
            "%02x".format(it)
        }
    }
}
