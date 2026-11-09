package com.nivukx.nox.core.reconstruction

import kotlin.test.assertTrue
import org.junit.Test

class SpatialMultiScaleDetailAuthorityTest {

    @Test
    fun highValueRegionsReceiveHigherDetailWhenBudgetAllows() {
        val engine = SpatialMultiScaleDetailAuthority()

        val result = engine.allocate(
            regions = listOf(
                evidence(
                    id = 1L,
                    complexity = 0.98,
                    motion = 0.94,
                    semantic = 0.90,
                    uncertaintyMeters = 0.92,
                    viewInformation = 0.90,
                    temporal = 0.95,
                ),
                evidence(
                    id = 2L,
                    complexity = 0.30,
                    motion = 0.20,
                    semantic = 0.40,
                    uncertaintyMeters = 0.30,
                    viewInformation = 0.35,
                    temporal = 0.90,
                ),
            ),
            availableBudgetUnits = 32,
        )

        val first = result.assignments.first { it.regionId == 1L }
        val second = result.assignments.first { it.regionId == 2L }

        assertTrue(first.priority > second.priority)
        assertTrue(first.costUnits >= second.costUnits)
        assertTrue(first.band.ordinal >= second.band.ordinal)
    }

    @Test
    fun budgetCapsPreventUnboundedDetailAdmission() {
        val engine = SpatialMultiScaleDetailAuthority(
            SpatialMultiScaleDetailAuthority.Config(minimumDetailBudgetUnits = 32),
        )

        val regions = (0L until 128L).map {
            evidence(
                id = it,
                complexity = 0.96,
                motion = 0.92,
                semantic = 0.88,
                uncertaintyMeters = 0.90,
                viewInformation = 0.92,
                temporal = 0.94,
            )
        }

        val result = engine.allocate(
            regions = regions,
            availableBudgetUnits = 32,
        )

        assertTrue(result.usedBudgetUnits <= result.budgetUnits)
        assertTrue(result.admittedRegionCount < result.requestedRegionCount)
    }

    @Test
    fun weakTemporalOrViewEvidenceStaysCoarse() {
        val engine = SpatialMultiScaleDetailAuthority()

        val result = engine.allocate(
            regions = listOf(
                evidence(
                    id = 7L,
                    complexity = 1.0,
                    motion = 1.0,
                    semantic = 1.0,
                    uncertaintyMeters = 1.0,
                    viewInformation = 0.10,
                    temporal = 0.20,
                ),
            ),
            availableBudgetUnits = 64,
        )

        assertTrue(
            result.assignments.single().band ==
                SpatialMultiScaleDetailAuthority.Band.COARSE,
        )
    }

    @Test
    fun emptyRegionsHoldOutput() {
        val result =
            SpatialMultiScaleDetailAuthority().allocate(
                regions = emptyList(),
                availableBudgetUnits = 64,
            )

        assertTrue(result.holdOutput)
        assertTrue(result.warnings.isNotEmpty())
    }

    private fun evidence(
        id: Long,
        complexity: Double,
        motion: Double,
        semantic: Double,
        uncertaintyMeters: Double,
        viewInformation: Double,
        temporal: Double,
    ) =
        SpatialMultiScaleDetailAuthority.RegionEvidence(
            regionId = id,
            geometricComplexity = complexity,
            motionBoundary = motion,
            semanticImportance = semantic,
            uncertaintyMeters = uncertaintyMeters,
            viewInformationGain = viewInformation,
            temporalStability = temporal,
        )
}
