package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ReconstructionRepresentationAuthorityEngineTest {

    @Test
    fun favorsMetricTopologyWhenTsdfHasDirectEvidence() {
        val engine =
            ReconstructionRepresentationAuthorityEngine()

        val result =
            engine.decide(
                evidence =
                    listOf(
                        evidence(
                            ReconstructionRepresentationAuthorityEngine.Representation.TSDF,
                            coverage = 0.95,
                            metricConfidence = 0.96,
                            uncertaintyMeters = 0.0008,
                            topologyStability = 0.98,
                            directlyMeasuredRatio = 0.92,
                        ),
                        evidence(
                            ReconstructionRepresentationAuthorityEngine.Representation.LEARNED_COMPLETION,
                            coverage = 0.70,
                            metricConfidence = 0.40,
                            uncertaintyMeters = 0.020,
                            topologyStability = 0.62,
                            directlyMeasuredRatio = 0.0,
                        ),
                    ),
                memoryHeadroom = 0.90,
                gpuHeadroom = 0.90,
                dynamicScene = false,
            )

        assertEquals(
            ReconstructionRepresentationAuthorityEngine.Representation.TSDF,
            result.decision.primary,
        )
        assertTrue(!result.decision.learnedOnly)
        assertTrue(result.decision.confidence > 0.70)
    }

    @Test
    fun favorsProbabilisticSurfaceForDynamicTemporalEvidence() {
        val engine =
            ReconstructionRepresentationAuthorityEngine()

        val result =
            engine.decide(
                evidence =
                    listOf(
                        evidence(
                            ReconstructionRepresentationAuthorityEngine.Representation.GAUSSIAN,
                            coverage = 0.85,
                            metricConfidence = 0.74,
                            uncertaintyMeters = 0.003,
                            topologyStability = 0.82,
                            directlyMeasuredRatio = 0.72,
                            dynamicCompatibility = 0.88,
                        ),
                        evidence(
                            ReconstructionRepresentationAuthorityEngine.Representation.PROBABILISTIC_SURFACE,
                            coverage = 0.86,
                            metricConfidence = 0.78,
                            uncertaintyMeters = 0.0016,
                            topologyStability = 0.91,
                            temporalStability = 0.96,
                            directlyMeasuredRatio = 0.74,
                            dynamicCompatibility = 0.98,
                        ),
                    ),
                memoryHeadroom = 0.80,
                gpuHeadroom = 0.80,
                dynamicScene = true,
            )

        assertEquals(
            ReconstructionRepresentationAuthorityEngine.Representation.PROBABILISTIC_SURFACE,
            result.decision.primary,
        )
        assertTrue(
            result.decision.secondary.contains(
                ReconstructionRepresentationAuthorityEngine.Representation.GAUSSIAN,
            ) ||
                result.decision.scoreGap > 0.0,
        )
    }

    @Test
    fun weakGeometricAuthorityCannotBeHiddenByVisualQuality() {
        val result =
            ReconstructionRepresentationAuthorityEngine().decide(
                evidence =
                    listOf(
                        evidence(
                            ReconstructionRepresentationAuthorityEngine.Representation.GAUSSIAN,
                            coverage = 0.99,
                            metricConfidence = 0.98,
                            uncertaintyMeters = 0.001,
                            topologyStability = 0.98,
                            directlyMeasuredRatio = 0.92,
                            dynamicCompatibility = 1.0,
                            appearanceConfidence = 1.0,
                        ).copy(
                            epipolarConsistency = 0.12,
                            metricEvidenceAuthority = 0.18,
                        ),
                        evidence(
                            ReconstructionRepresentationAuthorityEngine.Representation.TSDF,
                            coverage = 0.82,
                            metricConfidence = 0.84,
                            uncertaintyMeters = 0.0022,
                            topologyStability = 0.91,
                            directlyMeasuredRatio = 0.78,
                            dynamicCompatibility = 0.82,
                            appearanceConfidence = 0.60,
                        ).copy(
                            epipolarConsistency = 0.94,
                            metricEvidenceAuthority = 0.88,
                        ),
                    ),
                memoryHeadroom = 0.90,
                gpuHeadroom = 0.90,
                dynamicScene = false,
            )

        assertEquals(
            ReconstructionRepresentationAuthorityEngine.Representation.TSDF,
            result.decision.primary,
        )
        assertTrue(result.warnings.none { it.contains("inferred") })
    }

    @Test
    fun learnedCompletionNeverBecomesMeasuredAuthority() {
        val engine =
            ReconstructionRepresentationAuthorityEngine()

        val result =
            engine.decide(
                evidence =
                    listOf(
                        evidence(
                            ReconstructionRepresentationAuthorityEngine.Representation.LEARNED_COMPLETION,
                            coverage = 0.98,
                            metricConfidence = 0.98,
                            uncertaintyMeters = 0.001,
                            topologyStability = 0.95,
                            appearanceConfidence = 1.0,
                            directlyMeasuredRatio = 0.0,
                        ),
                    ),
                memoryHeadroom = 1.0,
                gpuHeadroom = 1.0,
                dynamicScene = false,
            )

        assertTrue(result.decision.learnedOnly)
        assertTrue(result.decision.evidenceLimited)
        assertTrue(
            result.decision.reasons.any {
                it.contains("learned-completion")
            },
        )
    }

    private fun evidence(
        representation:
            ReconstructionRepresentationAuthorityEngine.Representation,
        coverage: Double,
        metricConfidence: Double,
        uncertaintyMeters: Double,
        topologyStability: Double,
        temporalStability: Double = 0.90,
        directlyMeasuredRatio: Double,
        dynamicCompatibility: Double = 0.90,
        appearanceConfidence: Double = 0.86,
    ) =
        ReconstructionRepresentationAuthorityEngine.Evidence(
            representation = representation,
            coverage = coverage,
            metricConfidence = metricConfidence,
            uncertaintyMeters = uncertaintyMeters,
            temporalStability = temporalStability,
            topologyStability = topologyStability,
            appearanceConfidence = appearanceConfidence,
            dynamicCompatibility = dynamicCompatibility,
            memoryCostMegabytes = 220.0,
            gpuCostNormalized = 0.40,
            directlyMeasuredRatio = directlyMeasuredRatio,
        )
}
