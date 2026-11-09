package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CorrelatedEvidenceFusionEngineTest {

    @Test
    fun emptyEvidenceRejects() {
        val result = CorrelatedEvidenceFusionEngine()
            .evaluate(emptyList())

        assertEquals(
            CorrelatedEvidenceFusionEngine.State.REJECTED,
            result.state,
        )
        assertTrue(result.hardFailure)
        assertEquals(0.0, result.fusedScore, 0.0)
    }

    @Test
    fun strongIndependentFamiliesRemainStrong() {
        val engine = CorrelatedEvidenceFusionEngine()
        val result = engine.evaluate(
            listOf(
                signal("geometry", 0.96, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal("temporal", 0.95, CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL),
                signal("uncertainty", 0.93, CorrelatedEvidenceFusionEngine.DependencyGroup.UNCERTAINTY),
                signal("rig", 0.97, CorrelatedEvidenceFusionEngine.DependencyGroup.RIG),
                signal("physics", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.PHYSICS),
            ),
        )

        assertTrue(result.fusedScore > 0.80)
        assertTrue(result.effectiveEvidenceCount >= 3.0)
        assertTrue(result.state == CorrelatedEvidenceFusionEngine.State.STRONG)
    }

    @Test
    fun redundantSignalsDoNotStackLinearly() {
        val engine = CorrelatedEvidenceFusionEngine()
        val independent = engine.evaluate(
            listOf(
                signal("g1", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal("t1", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL),
                signal("u1", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.UNCERTAINTY),
                signal("r1", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.RIG),
            ),
        )
        val redundant = engine.evaluate(
            listOf(
                signal("g1", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal("g2", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal("g3", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal("g4", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal("t1", 0.94, CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL),
            ),
        )

        assertTrue(redundant.redundantSignalCount >= 0)
        assertTrue(redundant.correlatedGroupCount >= 1)
        assertTrue(redundant.fusedScore <= independent.fusedScore + 0.08)
        assertTrue(
            redundant.warnings.contains("correlated-signal-redundancy") ||
                redundant.warnings.contains("common-mode-risk"),
        )
    }

    @Test
    fun hardFailureRejectsEvenWhenOtherSignalsAreStrong() {
        val engine = CorrelatedEvidenceFusionEngine()
        val result = engine.evaluate(
            listOf(
                signal("geometry", 0.99, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal(
                    name = "calibration-contract",
                    score = 0.96,
                    group = CorrelatedEvidenceFusionEngine.DependencyGroup.RIG,
                    hardFailure = true,
                ),
                signal("temporal", 0.98, CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL),
            ),
        )

        assertEquals(
            CorrelatedEvidenceFusionEngine.State.REJECTED,
            result.state,
        )
        assertTrue(result.hardFailure)
        assertTrue(result.warnings.contains("evidence-rejected"))
    }

    @Test
    fun weakFamilyBecomesExplicitFailureReason() {
        val engine = CorrelatedEvidenceFusionEngine()
        val result = engine.evaluate(
            listOf(
                signal("geometry", 0.92, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal("temporal", 0.19, CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL),
                signal("rig", 0.90, CorrelatedEvidenceFusionEngine.DependencyGroup.RIG),
                signal("global", 0.88, CorrelatedEvidenceFusionEngine.DependencyGroup.GLOBAL),
            ),
        )

        assertTrue(
            result.strongestFailureFamily ==
                CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL ||
                result.state == CorrelatedEvidenceFusionEngine.State.REJECTED,
        )
        assertTrue(result.warnings.any { it.startsWith("weak-evidence-family-") })
    }

    @Test
    fun qualificationProfileIsStricter() {
        val production =
            CorrelatedEvidenceFusionEngine(
                CorrelatedEvidenceFusionEngine.Config.production(),
            )
        val qualification =
            CorrelatedEvidenceFusionEngine(
                CorrelatedEvidenceFusionEngine.Config.qualification(),
            )

        val signals = listOf(
            signal("geometry", 0.80, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
            signal("temporal", 0.82, CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL),
            signal("rig", 0.81, CorrelatedEvidenceFusionEngine.DependencyGroup.RIG),
            signal("physics", 0.80, CorrelatedEvidenceFusionEngine.DependencyGroup.PHYSICS),
        )

        val productionResult = production.evaluate(signals)
        val qualificationResult = qualification.evaluate(signals)

        assertTrue(
            qualificationResult.fusedScore <=
                productionResult.fusedScore + 0.05,
        )
        assertTrue(
            qualification.configuration().guardedScoreThreshold >=
                production.configuration().guardedScoreThreshold,
        )
        assertTrue(
            qualification.configuration().maximumFailureRisk <
                production.configuration().maximumFailureRisk,
        )
    }


    @Test
    fun acceptanceCoverageContractCanRejectStrongButSparseEvidence() {
        val engine =
            CorrelatedEvidenceFusionEngine(
                CorrelatedEvidenceFusionEngine.Config.production().copy(
                    minimumAcceptanceCoverage = 0.95,
                    maximumAcceptanceCommonModeRisk = 0.99,
                ),
            )
        val result = engine.evaluate(
            listOf(
                signal("geometry", 0.98, CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY),
                signal("temporal", 0.97, CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL),
            ),
        )

        assertTrue(
            result.state ==
                CorrelatedEvidenceFusionEngine.State.REJECTED ||
                result.warnings.contains("evidence-coverage-contract-breached"),
        )
    }

    @Test
    fun commonModeContractRejectsCorrelatedLowReliabilityEvidence() {
        val engine =
            CorrelatedEvidenceFusionEngine(
                CorrelatedEvidenceFusionEngine.Config.production().copy(
                    minimumAcceptanceCoverage = 0.0,
                    maximumAcceptanceCommonModeRisk = 0.05,
                ),
            )
        val result = engine.evaluate(
            listOf(
                signal(
                    "geometry-a",
                    0.90,
                    CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY,
                    reliability = 0.30,
                ),
                signal(
                    "geometry-b",
                    0.91,
                    CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY,
                    reliability = 0.30,
                ),
                signal(
                    "temporal",
                    0.90,
                    CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL,
                    reliability = 0.30,
                ),
            ),
        )

        assertTrue(
            result.state ==
                CorrelatedEvidenceFusionEngine.State.REJECTED ||
                result.warnings.contains("common-mode-contract-breached"),
        )
    }

    private fun signal(
        name: String,
        score: Double,
        group: CorrelatedEvidenceFusionEngine.DependencyGroup,
        reliability: Double = 0.95,
        weight: Double = 1.0,
        hardFailure: Boolean = false,
    ): CorrelatedEvidenceFusionEngine.Signal =
        CorrelatedEvidenceFusionEngine.Signal(
            name = name,
            score = score,
            reliability = reliability,
            group = group,
            weight = weight,
            hardFailure = hardFailure,
        )
}
