package com.nivukx.nox.core.ai

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class PerceptionPipelineSourceIntegrityTest {

    private fun source(): String {
        val path = Path.of(
            "src/main/java/com/nivukx/nox/core/ai/PerceptionPipeline.kt",
        )
        assertTrue(
            path.exists(),
            "PerceptionPipeline source must exist",
        )
        return Files.readString(path)
    }

    @Test
    fun cameraIdentityIsPartOfThePerceptionFrameContract() {
        val text = source()
        assertEquals(
            1,
            Regex(
                "cameraId: String = \"0\"",
            ).findAll(text).count(),
        )
        assertEquals(
            1,
            Regex(
                "private fun stableCameraIdentifier",
            ).findAll(text).count(),
        )
    }

    @Test
    fun denseLandmarkFusionRemainsIntegrated() {
        val text = source()
        assertTrue(
            text.contains(
                "DenseLandmarkEvidenceFusionEngine",
            ),
        )
        assertTrue(
            text.contains("denseLandmarkFusionEngine.process"),
            "Dense landmark evidence must remain on the live perception path",
        )
        assertTrue(
            text.contains("denseLandmarkEvidence = denseLandmarkEvidence"),
            "Dense evidence must remain part of the published result",
        )
    }

    @Test
    fun multiActorDenseSubjectSlotsAreBoundedAndStable() {
        val text = source()
        assertEquals(
            1,
            Regex(
                "private fun stableDenseSubjectSlot",
            ).findAll(text).count(),
        )
        assertTrue(
            text.contains("denseActorSubjectSlots"),
        )
        assertTrue(
            text.contains("retentionNanos = 5_000_000_000L"),
        )
        assertTrue(
            text.contains("(1..50)"),
        )
    }
    @Test
    fun faceFailureIsContainedByThePerceptionGraph() {
        val source = source()
        assertTrue(
            source.contains(
                "return@continueWith lastFaces",
            ),
        )
        assertTrue(
            source.contains(
                "The last successful timestamp remains authoritative",
            ),
        )
    }

    @Test
    fun faceFallbackCannotBeMarkedFreshAfterDetectorFailure() {
        val text = source()
        assertTrue(
            text.contains("var faceEvidenceFresh = false"),
        )
        assertTrue(
            text.contains("faceEvidenceFresh = false"),
        )
        assertTrue(
            text.contains("faceEvidenceFresh = true"),
        )
        assertTrue(
            text.contains(
                "faceEvidenceFresh &&",
            ),
        )
        assertTrue(
            text.contains(
                "isFresh(\\n                    lastFacesTimestampNanos",
            ),
        )
    }

    @Test
    fun denseObservationCarriesPhotometricConsistency() {
        val source = source()
        assertTrue(
            source.contains(
                "photometricConsistency =",
            ),
        )
        assertTrue(
            source.contains(
                "densePhotometricConsistency",
            ),
        )
    }

    @Test
    fun nonCriticalInferenceFailuresRemainFailSoft() {
        val source = source()
        assertTrue(
            source.contains("return@continueWith lastObjects"),
        )
        assertTrue(
            source.contains("return@continueWith lastContourFrame"),
        )
        assertTrue(
            source.contains("lastMultiActorPoseResult"),
        )
        assertTrue(
            source.contains(
                "emptyMultiActorPoseResult(",
            ),
        )
    }

}
