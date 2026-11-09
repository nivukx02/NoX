package com.nivukx.nox.core.reconstruction

/**
 * Runtime policy for the temporal 3D estimator.
 *
 * These controls change estimator responsiveness and uncertainty handling without bypassing
 * geometric validation. They are deliberately bounded so UI customization cannot turn the
 * reconstruction layer into an unconstrained extrapolator.
 */
data class TemporalReconstructionTuning(
    val predictionHorizonSeconds: Double = 0.16,
    val innovationGateScale: Double = 1.00,
    val uncertaintyGateScale: Double = 1.00,
    val reprojectionGateScale: Double = 1.00,
    val smoothingResponsiveness: Double = 0.36,
    val confidenceDecayScale: Double = 1.00,
    val outlierRetention: Double = 0.05,
    val topologyToleranceScale: Double = 1.00,
) {
    init {
        require(predictionHorizonSeconds in 0.02..0.60)
        require(innovationGateScale in 0.65..1.80)
        require(uncertaintyGateScale in 0.50..1.80)
        require(reprojectionGateScale in 0.50..1.80)
        require(smoothingResponsiveness in 0.08..0.92)
        require(confidenceDecayScale in 0.50..2.00)
        require(outlierRetention in 0.0..0.35)
        require(topologyToleranceScale in 0.50..1.80)
    }

    companion object {
        fun cinematic(): TemporalReconstructionTuning =
            TemporalReconstructionTuning(
                predictionHorizonSeconds = 0.12,
                innovationGateScale = 0.86,
                uncertaintyGateScale = 1.10,
                reprojectionGateScale = 0.92,
                smoothingResponsiveness = 0.30,
                confidenceDecayScale = 1.18,
                outlierRetention = 0.03,
                topologyToleranceScale = 0.90,
            )

        fun liveAction(): TemporalReconstructionTuning =
            TemporalReconstructionTuning(
                predictionHorizonSeconds = 0.20,
                innovationGateScale = 1.20,
                uncertaintyGateScale = 1.16,
                reprojectionGateScale = 1.18,
                smoothingResponsiveness = 0.52,
                confidenceDecayScale = 0.88,
                outlierRetention = 0.08,
                topologyToleranceScale = 1.12,
            )

        fun precision(): TemporalReconstructionTuning =
            TemporalReconstructionTuning(
                predictionHorizonSeconds = 0.08,
                innovationGateScale = 0.74,
                uncertaintyGateScale = 0.84,
                reprojectionGateScale = 0.78,
                smoothingResponsiveness = 0.22,
                confidenceDecayScale = 1.36,
                outlierRetention = 0.02,
                topologyToleranceScale = 0.76,
            )

        fun creature(): TemporalReconstructionTuning =
            TemporalReconstructionTuning(
                predictionHorizonSeconds = 0.22,
                innovationGateScale = 1.22,
                uncertaintyGateScale = 1.20,
                reprojectionGateScale = 1.18,
                smoothingResponsiveness = 0.48,
                confidenceDecayScale = 0.90,
                outlierRetention = 0.10,
                topologyToleranceScale = 1.16,
            )

        fun facial(): TemporalReconstructionTuning =
            TemporalReconstructionTuning(
                predictionHorizonSeconds = 0.08,
                innovationGateScale = 0.78,
                uncertaintyGateScale = 0.86,
                reprojectionGateScale = 0.82,
                smoothingResponsiveness = 0.18,
                confidenceDecayScale = 1.42,
                outlierRetention = 0.02,
                topologyToleranceScale = 0.80,
            )

        fun handDetail(): TemporalReconstructionTuning =
            TemporalReconstructionTuning(
                predictionHorizonSeconds = 0.06,
                innovationGateScale = 0.72,
                uncertaintyGateScale = 0.80,
                reprojectionGateScale = 0.76,
                smoothingResponsiveness = 0.16,
                confidenceDecayScale = 1.38,
                outlierRetention = 0.015,
                topologyToleranceScale = 0.74,
            )

    }
}
