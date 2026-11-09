package com.nivukx.nox.core.model

/**
 * Production-grade capture policy describing how aggressively NoX should spend
 * perception, geometric reconstruction, temporal estimation, and surface
 * reconstruction budgets for a take.
 *
 * Profiles are policy objects rather than hidden heuristics. They can be
 * surfaced in the operator UI, persisted by id, replayed in tests, and
 * included in session manifests so a reconstruction can be reproduced.
 */
enum class CaptureQualityTier {
    BALANCED,
    HIGH,
    ULTRA,
    REFERENCE,
}

enum class SubjectTrackingMode {
    SINGLE_LOCK,
    MULTI_SUBJECT,
    DYNAMIC_TOPOLOGY,
}

enum class ReconstructionMode {
    MONOCULAR_FIRST,
    MULTI_VIEW_FUSION,
    HYBRID_REFERENCE,
}

enum class FaceDetailPolicy {
    DISABLED,
    TRACKED,
    DENSE,
    REFERENCE,
}

enum class EyeTrackingPolicy {
    NONE,
    SCREEN_SPACE,
    METRIC_BINO,
    REFERENCE_BINO,
}

enum class HandTrackingPolicy {
    NONE,
    TRACKED,
    DENSE,
    REFERENCE,
}

enum class SurfacePolicy {
    SKELETON_ONLY,
    SPARSE_SURFACE,
    DENSE_SURFACE,
    REFERENCE_SURFACE,
}

enum class TopologyPolicy {
    FIXED_HUMAN,
    ADAPTIVE_ARTICULATION,
    DYNAMIC_SUBJECT,
    REFERENCE_DYNAMIC,
}

enum class ThermalPolicy {
    SUSTAINED,
    PERFORMANCE,
    REFERENCE_WINDOW,
}

enum class GpuExecutionPolicy {
    AUTO,
    GPU_PREFERRED,
    GPU_REQUIRED,
}

data class CaptureProfileMetric(
    val label: String,
    val value: String,
)

data class ProductionCaptureProfile(
    val id: String,
    val label: String,
    val description: String,
    val qualityTier: CaptureQualityTier,
    val targetCaptureFps: Int,
    val inferenceFps: Int,
    val maxSubjects: Int,
    val detectorIntervalFrames: Int,
    val temporalPredictionHorizonSeconds: Double,
    val maxOcclusionFrames: Int,
    val innovationGateScale: Double,
    val reprojectionGatePx: Double,
    val maxMeanUncertaintyMeters: Double,
    val tsdfVoxelSizeMeters: Double,
    val tsdfTruncationMeters: Double,
    val meshVertexBudget: Int,
    val denseMeshHz: Int,
    val faceMeshHz: Int,
    val eyeTrackingHz: Int,
    val handTrackingHz: Int,
    val bundleAdjustmentIterations: Int,
    val multiViewConsensusThreshold: Int,
    val surfaceObservationWeight: Double,
    val thermalPolicy: ThermalPolicy,
    val subjectTrackingMode: SubjectTrackingMode,
    val reconstructionMode: ReconstructionMode,
    val faceDetailPolicy: FaceDetailPolicy,
    val eyeTrackingPolicy: EyeTrackingPolicy,
    val handTrackingPolicy: HandTrackingPolicy,
    val surfacePolicy: SurfacePolicy,
    val topologyPolicy: TopologyPolicy,
    val gpuExecutionPolicy: GpuExecutionPolicy,
    val requireRigStabilityEvidence: Boolean = false,
    val minimumRigStabilityConfidence: Double = 0.70,
    val requireDynamicMetrologyEvidence: Boolean = false,
    val dynamicMetrologyP95Millimeters: Double = 1.0,
) {
    init {
        require(id.isNotBlank())
        require(label.isNotBlank())
        require(description.isNotBlank())
        require(targetCaptureFps in 15..240)
        require(inferenceFps in 1..240)
        require(maxSubjects in 1..64)
        require(detectorIntervalFrames >= 1)
        require(temporalPredictionHorizonSeconds in 0.01..0.50)
        require(maxOcclusionFrames in 0..60)
        require(innovationGateScale in 0.25..3.0)
        require(reprojectionGatePx in 0.25..12.0)
        require(maxMeanUncertaintyMeters in 0.001..1.0)
        require(tsdfVoxelSizeMeters in 0.001..0.10)
        require(tsdfTruncationMeters > tsdfVoxelSizeMeters)
        require(meshVertexBudget in 4_000..2_000_000)
        require(denseMeshHz in 1..120)
        require(faceMeshHz in 1..120)
        require(eyeTrackingHz in 1..240)
        require(handTrackingHz in 1..120)
        require(bundleAdjustmentIterations in 1..200)
        require(multiViewConsensusThreshold in 1..64)
        require(surfaceObservationWeight in 0.0..1.0)
        require(minimumRigStabilityConfidence in 0.0..1.0)
        require(dynamicMetrologyP95Millimeters in 0.1..10.0)
    }

    val temporalPredictionHorizonMillis: Long
        get() = (temporalPredictionHorizonSeconds * 1_000.0).toLong()

    val tsdfResolutionClass: String
        get() = when {
            tsdfVoxelSizeMeters <= 0.005 -> "REFERENCE"
            tsdfVoxelSizeMeters <= 0.010 -> "ULTRA"
            tsdfVoxelSizeMeters <= 0.020 -> "HIGH"
            else -> "BALANCED"
        }

    val expectedDetectorDutyCycle: Double
        get() = 1.0 / detectorIntervalFrames.toDouble()

    val multiViewRequired: Boolean
        get() = reconstructionMode != ReconstructionMode.MONOCULAR_FIRST

    fun operatorSummary(): List<CaptureProfileMetric> = listOf(
        CaptureProfileMetric("Capture", "${targetCaptureFps} FPS"),
        CaptureProfileMetric("Inference", "${inferenceFps} FPS"),
        CaptureProfileMetric("Subjects", "1-${maxSubjects}"),
        CaptureProfileMetric("Prediction", "${temporalPredictionHorizonMillis} ms"),
        CaptureProfileMetric("Occlusion", "${maxOcclusionFrames} frames"),
        CaptureProfileMetric("TSDF", "${tsdfVoxelSizeMeters * 1000.0} mm"),
        CaptureProfileMetric("Mesh budget", "%,d V".format(meshVertexBudget)),
        CaptureProfileMetric("Face", faceDetailPolicy.name),
        CaptureProfileMetric("Eyes", eyeTrackingPolicy.name),
        CaptureProfileMetric("Hands", handTrackingPolicy.name),
        CaptureProfileMetric("Surface", surfacePolicy.name),
        CaptureProfileMetric("GPU", gpuExecutionPolicy.name),
        CaptureProfileMetric(
            "Rig stability",
            if (requireRigStabilityEvidence) "REQUIRED" else "ADVISORY",
        ),
        CaptureProfileMetric(
            "Dynamic P95",
            dynamicMetrologyP95Millimeters.toString() + " mm",
        ),
    )

    fun validated(): ProductionCaptureProfile {
        require(targetCaptureFps >= inferenceFps) {
            "Inference rate cannot exceed capture rate"
        }
        require(
            reconstructionMode == ReconstructionMode.MONOCULAR_FIRST ||
                multiViewConsensusThreshold <= maxSubjects * 64,
        ) {
            "Invalid multi-view consensus policy"
        }
        return this
    }

    companion object {
        val CINEMATIC_ULTRA = ProductionCaptureProfile(
            id = "CINEMATIC_ULTRA",
            label = "Cinematic Ultra",
            description = "High temporal continuity with dense geometry and conservative quality gates.",
            qualityTier = CaptureQualityTier.ULTRA,
            targetCaptureFps = 60,
            inferenceFps = 60,
            maxSubjects = 2,
            detectorIntervalFrames = 2,
            temporalPredictionHorizonSeconds = 0.10,
            maxOcclusionFrames = 8,
            innovationGateScale = 1.15,
            reprojectionGatePx = 3.0,
            maxMeanUncertaintyMeters = 0.045,
            tsdfVoxelSizeMeters = 0.0075,
            tsdfTruncationMeters = 0.035,
            meshVertexBudget = 450_000,
            denseMeshHz = 20,
            faceMeshHz = 30,
            eyeTrackingHz = 60,
            handTrackingHz = 45,
            bundleAdjustmentIterations = 28,
            multiViewConsensusThreshold = 3,
            surfaceObservationWeight = 0.80,
            thermalPolicy = ThermalPolicy.PERFORMANCE,
            subjectTrackingMode = SubjectTrackingMode.MULTI_SUBJECT,
            reconstructionMode = ReconstructionMode.HYBRID_REFERENCE,
            faceDetailPolicy = FaceDetailPolicy.DENSE,
            eyeTrackingPolicy = EyeTrackingPolicy.METRIC_BINO,
            handTrackingPolicy = HandTrackingPolicy.DENSE,
            surfacePolicy = SurfacePolicy.DENSE_SURFACE,
            topologyPolicy = TopologyPolicy.ADAPTIVE_ARTICULATION,
            gpuExecutionPolicy = GpuExecutionPolicy.GPU_PREFERRED,
        )

        val LIVE_ACTION = ProductionCaptureProfile(
            id = "LIVE_ACTION",
            label = "Live Action",
            description = "Sustained real-time motion capture with fast reacquisition and controlled surface cost.",
            qualityTier = CaptureQualityTier.HIGH,
            targetCaptureFps = 60,
            inferenceFps = 45,
            maxSubjects = 4,
            detectorIntervalFrames = 3,
            temporalPredictionHorizonSeconds = 0.12,
            maxOcclusionFrames = 10,
            innovationGateScale = 1.30,
            reprojectionGatePx = 3.8,
            maxMeanUncertaintyMeters = 0.065,
            tsdfVoxelSizeMeters = 0.0125,
            tsdfTruncationMeters = 0.050,
            meshVertexBudget = 280_000,
            denseMeshHz = 12,
            faceMeshHz = 20,
            eyeTrackingHz = 45,
            handTrackingHz = 30,
            bundleAdjustmentIterations = 16,
            multiViewConsensusThreshold = 2,
            surfaceObservationWeight = 0.64,
            thermalPolicy = ThermalPolicy.SUSTAINED,
            subjectTrackingMode = SubjectTrackingMode.MULTI_SUBJECT,
            reconstructionMode = ReconstructionMode.MULTI_VIEW_FUSION,
            faceDetailPolicy = FaceDetailPolicy.DENSE,
            eyeTrackingPolicy = EyeTrackingPolicy.METRIC_BINO,
            handTrackingPolicy = HandTrackingPolicy.TRACKED,
            surfacePolicy = SurfacePolicy.SPARSE_SURFACE,
            topologyPolicy = TopologyPolicy.DYNAMIC_SUBJECT,
            gpuExecutionPolicy = GpuExecutionPolicy.AUTO,
        )

        val MULTI_VIEW_PRECISION = ProductionCaptureProfile(
            id = "MULTI_VIEW_PRECISION",
            label = "Multi-View Precision",
            description = "Reference-grade synchronized multi-camera reconstruction with stronger geometry constraints.",
            qualityTier = CaptureQualityTier.REFERENCE,
            targetCaptureFps = 60,
            inferenceFps = 60,
            maxSubjects = 3,
            detectorIntervalFrames = 2,
            temporalPredictionHorizonSeconds = 0.08,
            maxOcclusionFrames = 6,
            innovationGateScale = 0.90,
            reprojectionGatePx = 2.0,
            maxMeanUncertaintyMeters = 0.030,
            tsdfVoxelSizeMeters = 0.005,
            tsdfTruncationMeters = 0.025,
            meshVertexBudget = 750_000,
            denseMeshHz = 30,
            faceMeshHz = 45,
            eyeTrackingHz = 90,
            handTrackingHz = 60,
            bundleAdjustmentIterations = 48,
            multiViewConsensusThreshold = 4,
            surfaceObservationWeight = 0.92,
            thermalPolicy = ThermalPolicy.REFERENCE_WINDOW,
            subjectTrackingMode = SubjectTrackingMode.MULTI_SUBJECT,
            reconstructionMode = ReconstructionMode.HYBRID_REFERENCE,
            faceDetailPolicy = FaceDetailPolicy.REFERENCE,
            eyeTrackingPolicy = EyeTrackingPolicy.REFERENCE_BINO,
            handTrackingPolicy = HandTrackingPolicy.REFERENCE,
            surfacePolicy = SurfacePolicy.REFERENCE_SURFACE,
            topologyPolicy = TopologyPolicy.REFERENCE_DYNAMIC,
            gpuExecutionPolicy = GpuExecutionPolicy.GPU_REQUIRED,
        )

        val FACE_PERFORMANCE = ProductionCaptureProfile(
            id = "FACE_PERFORMANCE",
            label = "Face Performance",
            description = "Face, eye, gaze and head-motion priority while maintaining full-body continuity.",
            qualityTier = CaptureQualityTier.ULTRA,
            targetCaptureFps = 60,
            inferenceFps = 60,
            maxSubjects = 1,
            detectorIntervalFrames = 2,
            temporalPredictionHorizonSeconds = 0.07,
            maxOcclusionFrames = 5,
            innovationGateScale = 0.95,
            reprojectionGatePx = 2.5,
            maxMeanUncertaintyMeters = 0.035,
            tsdfVoxelSizeMeters = 0.010,
            tsdfTruncationMeters = 0.040,
            meshVertexBudget = 400_000,
            denseMeshHz = 45,
            faceMeshHz = 60,
            eyeTrackingHz = 120,
            handTrackingHz = 45,
            bundleAdjustmentIterations = 24,
            multiViewConsensusThreshold = 3,
            surfaceObservationWeight = 0.75,
            thermalPolicy = ThermalPolicy.PERFORMANCE,
            subjectTrackingMode = SubjectTrackingMode.SINGLE_LOCK,
            reconstructionMode = ReconstructionMode.MULTI_VIEW_FUSION,
            faceDetailPolicy = FaceDetailPolicy.REFERENCE,
            eyeTrackingPolicy = EyeTrackingPolicy.REFERENCE_BINO,
            handTrackingPolicy = HandTrackingPolicy.DENSE,
            surfacePolicy = SurfacePolicy.DENSE_SURFACE,
            topologyPolicy = TopologyPolicy.ADAPTIVE_ARTICULATION,
            gpuExecutionPolicy = GpuExecutionPolicy.GPU_PREFERRED,
        )

        val SUSTAINED_MOBILE = ProductionCaptureProfile(
            id = "SUSTAINED_MOBILE",
            label = "Sustained Mobile",
            description = "Thermally conservative capture profile for long takes on constrained phones.",
            qualityTier = CaptureQualityTier.BALANCED,
            targetCaptureFps = 30,
            inferenceFps = 24,
            maxSubjects = 2,
            detectorIntervalFrames = 4,
            temporalPredictionHorizonSeconds = 0.16,
            maxOcclusionFrames = 12,
            innovationGateScale = 1.40,
            reprojectionGatePx = 4.8,
            maxMeanUncertaintyMeters = 0.095,
            tsdfVoxelSizeMeters = 0.020,
            tsdfTruncationMeters = 0.060,
            meshVertexBudget = 120_000,
            denseMeshHz = 6,
            faceMeshHz = 10,
            eyeTrackingHz = 24,
            handTrackingHz = 18,
            bundleAdjustmentIterations = 8,
            multiViewConsensusThreshold = 2,
            surfaceObservationWeight = 0.45,
            thermalPolicy = ThermalPolicy.SUSTAINED,
            subjectTrackingMode = SubjectTrackingMode.MULTI_SUBJECT,
            reconstructionMode = ReconstructionMode.MONOCULAR_FIRST,
            faceDetailPolicy = FaceDetailPolicy.TRACKED,
            eyeTrackingPolicy = EyeTrackingPolicy.SCREEN_SPACE,
            handTrackingPolicy = HandTrackingPolicy.TRACKED,
            surfacePolicy = SurfacePolicy.SKELETON_ONLY,
            topologyPolicy = TopologyPolicy.DYNAMIC_SUBJECT,
            gpuExecutionPolicy = GpuExecutionPolicy.AUTO,
        )

        val REFERENCE_BAKE = ProductionCaptureProfile(
            id = "REFERENCE_BAKE",
            label = "Reference Bake",
            description = "Reference-grade quality policy for controlled capture windows and high confidence output.",
            qualityTier = CaptureQualityTier.REFERENCE,
            targetCaptureFps = 120,
            inferenceFps = 90,
            maxSubjects = 2,
            detectorIntervalFrames = 1,
            temporalPredictionHorizonSeconds = 0.05,
            maxOcclusionFrames = 4,
            innovationGateScale = 0.75,
            reprojectionGatePx = 1.6,
            maxMeanUncertaintyMeters = 0.020,
            tsdfVoxelSizeMeters = 0.0035,
            tsdfTruncationMeters = 0.018,
            meshVertexBudget = 1_200_000,
            denseMeshHz = 60,
            faceMeshHz = 90,
            eyeTrackingHz = 120,
            handTrackingHz = 90,
            bundleAdjustmentIterations = 80,
            multiViewConsensusThreshold = 5,
            surfaceObservationWeight = 0.96,
            thermalPolicy = ThermalPolicy.REFERENCE_WINDOW,
            subjectTrackingMode = SubjectTrackingMode.SINGLE_LOCK,
            reconstructionMode = ReconstructionMode.MULTI_VIEW_FUSION,
            faceDetailPolicy = FaceDetailPolicy.REFERENCE,
            eyeTrackingPolicy = EyeTrackingPolicy.REFERENCE_BINO,
            handTrackingPolicy = HandTrackingPolicy.REFERENCE,
            surfacePolicy = SurfacePolicy.REFERENCE_SURFACE,
            topologyPolicy = TopologyPolicy.REFERENCE_DYNAMIC,
            gpuExecutionPolicy = GpuExecutionPolicy.GPU_REQUIRED,
        )

        /**
         * Aggressive metrology capture policy.
         *
         * This profile does not claim that a phone can physically guarantee 1 mm accuracy.
         * It requests the strictest available reconstruction/measurement budget and leaves
         * qualification to calibrated reference evidence.
         */
        /**
         * Distributed fifty-actor live/hybrid policy.
         *
         * This profile is designed for synchronized multi-camera capture where each device contributes
         * observations to the global association/reconstruction graph. Runtime governors may reduce
         * optional modality cadence when hardware headroom is insufficient.
         */
        val FIFTY_ACTOR_LIVE_HYBRID = ProductionCaptureProfile(
            id = "FIFTY_ACTOR_LIVE_HYBRID",
            label = "Fifty Actor Live + Hybrid",
            description =
                "Bounded 50-actor live capture with optional visual evidence retention and deferred reconstruction.",
            qualityTier = CaptureQualityTier.ULTRA,
            targetCaptureFps = 60,
            inferenceFps = 60,
            maxSubjects = 50,
            detectorIntervalFrames = 2,
            temporalPredictionHorizonSeconds = 0.08,
            maxOcclusionFrames = 12,
            innovationGateScale = 1.05,
            reprojectionGatePx = 3.0,
            maxMeanUncertaintyMeters = 0.060,
            tsdfVoxelSizeMeters = 0.010,
            tsdfTruncationMeters = 0.040,
            meshVertexBudget = 600_000,
            denseMeshHz = 12,
            faceMeshHz = 30,
            eyeTrackingHz = 60,
            handTrackingHz = 30,
            bundleAdjustmentIterations = 18,
            multiViewConsensusThreshold = 2,
            surfaceObservationWeight = 0.72,
            thermalPolicy = ThermalPolicy.SUSTAINED,
            subjectTrackingMode = SubjectTrackingMode.DYNAMIC_TOPOLOGY,
            reconstructionMode = ReconstructionMode.MULTI_VIEW_FUSION,
            faceDetailPolicy = FaceDetailPolicy.DENSE,
            eyeTrackingPolicy = EyeTrackingPolicy.METRIC_BINO,
            handTrackingPolicy = HandTrackingPolicy.DENSE,
            surfacePolicy = SurfacePolicy.DENSE_SURFACE,
            topologyPolicy = TopologyPolicy.DYNAMIC_SUBJECT,
            gpuExecutionPolicy = GpuExecutionPolicy.AUTO,
            requireRigStabilityEvidence = true,
            minimumRigStabilityConfidence = 0.78,
            requireDynamicMetrologyEvidence = false,
            dynamicMetrologyP95Millimeters = 2.0,
        )

        val NANO_METROLOGY_REFERENCE = ProductionCaptureProfile(
            id = "NANO_METROLOGY_REFERENCE",
            label = "Nano Metrology Reference",
            description =
                "Maximum available reconstruction fidelity for controlled multi-camera capture with strict sub-millimetre evidence gates.",
            qualityTier = CaptureQualityTier.REFERENCE,
            targetCaptureFps = 120,
            inferenceFps = 120,
            maxSubjects = 1,
            detectorIntervalFrames = 1,
            temporalPredictionHorizonSeconds = 0.025,
            maxOcclusionFrames = 2,
            innovationGateScale = 0.55,
            reprojectionGatePx = 0.80,
            maxMeanUncertaintyMeters = 0.001,
            tsdfVoxelSizeMeters = 0.001,
            tsdfTruncationMeters = 0.006,
            meshVertexBudget = 2_000_000,
            denseMeshHz = 120,
            faceMeshHz = 120,
            eyeTrackingHz = 240,
            handTrackingHz = 120,
            bundleAdjustmentIterations = 200,
            multiViewConsensusThreshold = 8,
            surfaceObservationWeight = 0.995,
            thermalPolicy = ThermalPolicy.REFERENCE_WINDOW,
            subjectTrackingMode = SubjectTrackingMode.SINGLE_LOCK,
            reconstructionMode = ReconstructionMode.HYBRID_REFERENCE,
            faceDetailPolicy = FaceDetailPolicy.REFERENCE,
            eyeTrackingPolicy = EyeTrackingPolicy.REFERENCE_BINO,
            handTrackingPolicy = HandTrackingPolicy.REFERENCE,
            surfacePolicy = SurfacePolicy.REFERENCE_SURFACE,
            topologyPolicy = TopologyPolicy.REFERENCE_DYNAMIC,
            gpuExecutionPolicy = GpuExecutionPolicy.GPU_REQUIRED,
            requireRigStabilityEvidence = true,
            minimumRigStabilityConfidence = 0.92,
            requireDynamicMetrologyEvidence = true,
            dynamicMetrologyP95Millimeters = 1.0,
        )

        private val ALL = listOf(
            CINEMATIC_ULTRA,
            LIVE_ACTION,
            MULTI_VIEW_PRECISION,
            FACE_PERFORMANCE,
            SUSTAINED_MOBILE,
            REFERENCE_BAKE,
            FIFTY_ACTOR_LIVE_HYBRID,
            NANO_METROLOGY_REFERENCE,
        )

        val default: ProductionCaptureProfile
            get() = LIVE_ACTION

        fun all(): List<ProductionCaptureProfile> = ALL

        fun byId(id: String?): ProductionCaptureProfile =
            ALL.firstOrNull { it.id.equals(id, ignoreCase = true) }
                ?: default
    }
}
