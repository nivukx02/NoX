package com.nivukx.nox

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.assertTrue
import org.junit.Test

class NoXViewModelStateContractTest {
    @Test
    fun denseEvidenceLambdaHasNoTrailingComma() {
        val sourcePath =
            Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt")
        assertTrue(sourcePath.toFile().exists())
        val source = Files.readString(sourcePath)
        assertTrue(
            !source.contains(
                "rollingShutterCompensated = true,\n                        ),\n                }",
            ),
            "Dense evidence CameraObservation must be the final lambda expression",
        )
    }

    @Test
    fun requiredCrossModuleImportsRemainExplicit() {
        val sourcePath =
            Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt")
        assertTrue(sourcePath.toFile().exists())

        val source = Files.readString(sourcePath)
        listOf(
            "com.nivukx.nox.core.capture.CameraTopologyEvidenceEngine",
            "com.nivukx.nox.core.reconstruction.LightingEnvironmentTemporalConsistencyEngine",
            "com.nivukx.nox.core.reconstruction.DepthUncertaintyModelEngine",
            "com.nivukx.nox.core.reconstruction.AdaptiveDenseMeshExtractionController",
            "com.nivukx.nox.core.reconstruction.CrossModalIdentityFusionEngine",
            "com.nivukx.nox.core.reconstruction.FaceEyeCalibrationDriftEngine",
            "com.nivukx.nox.core.reconstruction.HighEndFaceShapeAlignmentEngine",
            "com.nivukx.nox.core.model.MotionCaptureCustomization",
            "com.nivukx.nox.core.model.SessionPresentationSettings",
        ).forEach { importPath ->
            assertTrue(
                source.contains("import $importPath"),
                "Missing explicit import: $importPath",
            )
        }
    }

    @Test
    fun motionCaptureModeAndTasksContractsRemainExplicit() {
        val buildSource =
            listOf(
                Path.of("build.gradle.kts"),
                Path.of("app/build.gradle.kts"),
                Path.of("../app/build.gradle.kts"),
            )
                .firstOrNull { Files.exists(it) }
                ?.let(Files::readString)
                ?: ""
        val modelSource =
            Files.readString(
                Path.of("../core/model/src/main/java/com/nivukx/nox/core/model/UltraMotionModels.kt"),
            )
        val viewModelSource =
            Files.readString(
                Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            )

        assertTrue(
            buildSource.contains(
                "com.google.android.gms:play-services-tasks:18.2.0",
            ),
        )
        assertTrue(
            modelSource.contains("enum class MotionCaptureMode"),
        )
        assertTrue(
            viewModelSource.contains("addOnSuccessListener"),
        )
    }

    @Test
    fun publicStateFlowPropertiesAreDeclaredExactlyOnce() {
        val sourcePath =
            Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt")
        assertTrue(sourcePath.toFile().exists())

        val source = Files.readString(sourcePath)
        val counts = linkedMapOf<String, Int>()
        Regex("""^\s*val\s+([A-Za-z_]\w*)\s*:\s*StateFlow<""", RegexOption.MULTILINE)
            .findAll(source)
            .forEach { match ->
                val name = match.groupValues[1]
                counts[name] = (counts[name] ?: 0) + 1
            }

        val duplicates =
            counts.filterValues { it > 1 }.keys
        assertTrue(
            duplicates.isEmpty(),
            "Duplicate public StateFlow declarations: $duplicates",
        )
    }
}


class NoXViewModelCalibrationRayContractTest {
    @Test
    fun authoritativeCalibrationIsForwardedIntoPerceptionGeometry() {
        val source =
            Files.readString(
                Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            )
        assertTrue(
            source.contains("configurePerceptionCalibration(result.snapshot)"),
        )
        assertTrue(
            source.contains("configurePerceptionCalibration(snapshot)"),
        )
        assertTrue(
            source.contains("perception.configureCalibratedCameraGeometry(geometry)"),
        )
        assertTrue(
            Regex("""calibrationCameraId\s*=\s*localId\.value""")
                .findAll(source)
                .count() == 2,
            "Live and calibration capture must both pass the stable NoX device ID separately from the Camera2 ID",
        )
        val rayProjectionSource =
            Files.readString(
                Path.of("../core/ai/src/main/java/com/nivukx/nox/core/ai/PerceptionPipeline.kt"),
            )
        assertTrue(
            rayProjectionSource.contains("calibrationCameraId: String = cameraId"),
        )
        assertTrue(
            rayProjectionSource.contains("cameraId = calibrationCameraId"),
            "Calibrated ray projection must resolve geometry using the calibration device ID",
        )
        assertTrue(
            rayProjectionSource.contains("ray?.calibrationConfidence ?: 0.0"),
            "Missing camera geometry must not be reported as perfect calibration confidence",
        )
        assertTrue(
            !rayProjectionSource.contains("ray?.calibrationConfidence ?: 1.0"),
            "Missing camera geometry must not silently promote calibration confidence to 1.0",
        )
    }
}


class NoXViewModelTimingConfidenceContractTest {
    @Test
    fun missingTimingTelemetryCannotBecomePerfectConfidence() {
        val viewModelSource =
            Files.readString(
                Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            )
        val perceptionSource =
            Files.readString(
                Path.of("../core/ai/src/main/java/com/nivukx/nox/core/ai/PerceptionPipeline.kt"),
            )

        assertTrue(
            !viewModelSource.contains("camera.latestTimingQuality?.confidence ?: 1.0"),
            "Absent camera timing telemetry must not become perfect confidence",
        )
        assertTrue(
            Regex("""captureTimingConfidence\s*=\s*resolvedCaptureTimingConfidence\(\)""")
                .findAll(viewModelSource)
                .count() == 2,
            "Live capture and calibration collection must share the timing-confidence resolver",
        )
        assertTrue(
            viewModelSource.contains("Missing or non-finite telemetry is unknown evidence"),
        )
        assertTrue(
            perceptionSource.contains("captureTimingConfidence: Double = 0.0"),
            "Callers that omit timing metadata must get a conservative unknown default",
        )
    }
}


class NoXViewModelDenseFrameProvenanceContractTest {
    @Test
    fun localAndRemoteDenseFramesPreserveTimingAndImageHeight() {
        val viewModel =
            Files.readString(
                Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            )
        val wireCodec =
            Files.readString(
                Path.of("../core/network/src/main/java/com/nivukx/nox/core/network/DenseLandmarkWireCodec.kt"),
            )
        val fusion =
            Files.readString(
                Path.of("../core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/MultiViewFusionEngine.kt"),
            )

        assertTrue(viewModel.contains("val frameTimingConfidence = resolvedCaptureTimingConfidence()"))
        assertTrue(viewModel.contains("imageWidth = image.width"))
        assertTrue(viewModel.contains("imageHeight = image.height"))
        assertTrue(viewModel.contains("frameHeightPx = image.height"))
        assertTrue(viewModel.contains("frameHeightPx = frame.imageHeight"))
        assertTrue(viewModel.contains("frame.captureTimingConfidence.toDouble()"))
        assertTrue(viewModel.contains("frame.rollingShutterSkewNanos"))
        assertTrue(wireCodec.contains("private const val VERSION = 4"))
        assertTrue(wireCodec.contains("private const val VERSION_3 = 3"))
        assertTrue(fusion.contains("landmark.frameHeightPx.takeIf { it > 1 } ?: height"))
        assertTrue(fusion.contains("captureTimingConfidence: Double = 0.0"))
    }
}


class NoXViewModelMotionContextTimingContractTest {
    @Test
    fun motionContextUsesTheSharedValidatedTimingResolver() {
        val source =
            Files.readString(
                Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            )
        val motionContext =
            source.substringAfter("private fun motionContextForPerception()")
                .substringBefore("private fun configureTemporalMotionEstimator")

        assertTrue(
            motionContext.contains("val timingConfidence =\n            resolvedCaptureTimingConfidence()"),
        )
        assertTrue(
            !motionContext.contains("camera.latestTimingQuality?.confidence"),
            "Motion context must not maintain a second less-safe timing fallback",
        )
    }
}


class NoXViewModelDenseFragmentTransportContractTest {
    @Test
    fun senderNegotiatesFragmentationAndReceiverConsumesReassembledFrames() {
        val viewModel =
            Files.readString(
                Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            )
        val nearby =
            Files.readString(
                Path.of("../core/network/src/main/java/com/nivukx/nox/core/network/NearbyPeerController.kt"),
            )

        assertTrue(viewModel.contains("supportsDenseLandmarkFragmentation"))
        assertTrue(viewModel.contains("peers.sendDenseLandmarkFrame("))
        assertTrue(viewModel.contains("DenseLandmarkWireCodec.encodeV3Compatibility"))
        assertTrue(viewModel.contains("peers.denseLandmarkFrames.collect"))
        assertTrue(viewModel.contains("handleDenseLandmarkFrame(event.endpoint, event.frame)"))
        assertTrue(nearby.contains("fun sendDenseLandmarkFrame("))
        assertTrue(nearby.contains("denseLandmarkReassembler.accept("))
    }
}


class NoXViewModelDenseWireVersionNegotiationContractTest {
    @Test
    fun senderNegotiatesTimingHeaderIndependentlyFromFragmentSupport() {
        val source =
            Files.readString(
                Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            )
        val chunkCodec =
            Files.readString(
                Path.of("../core/network/src/main/java/com/nivukx/nox/core/network/DenseLandmarkChunkedWireCodec.kt"),
            )
        val capabilityCodec =
            Files.readString(
                Path.of("../core/network/src/main/java/com/nivukx/nox/core/network/PeerCapabilitiesWireCodec.kt"),
            )

        assertTrue(source.contains("supportsDenseLandmarkTimingMetadata"))
        assertTrue(source.contains("FrameWireVersion.V4_TIMING_METADATA"))
        assertTrue(source.contains("FrameWireVersion.LEGACY_V3"))
        assertTrue(source.contains("wireVersion = negotiatedWireVersion"))
        assertTrue(chunkCodec.contains("FrameWireVersion.LEGACY_V3"))
        assertTrue(chunkCodec.contains("encodeV3Compatibility(chunkFrame)"))
        assertTrue(capabilityCodec.contains("FLAG_DENSE_TIMING_METADATA = 128"))
    }
}


class NoXViewModelSegmentationConfidenceContractTest {
    @Test
    fun liveDenseWireEvidenceNeverAssumesPerfectSegmentation() {
        val viewModel =
            Files.readString(
                Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            )
        val perception =
            Files.readString(
                Path.of("../core/ai/src/main/java/com/nivukx/nox/core/ai/PerceptionPipeline.kt"),
            )
        val wire =
            Files.readString(
                Path.of("../core/network/src/main/java/com/nivukx/nox/core/network/DenseLandmarkWireCodec.kt"),
            )

        assertTrue(!viewModel.contains("segmentationConsistency = 1.0f"))
        assertTrue(viewModel.contains("landmark.segmentationEvidenceAvailable"))
        assertTrue(perception.contains("segmentationEvidenceAvailable = personSegmentation != null"))
        assertTrue(wire.contains("version == VERSION && (pointFlags and 0x02) != 0"))
        assertTrue(wire.contains("(pointFlags and 0x02) != 0"))
    }
}
