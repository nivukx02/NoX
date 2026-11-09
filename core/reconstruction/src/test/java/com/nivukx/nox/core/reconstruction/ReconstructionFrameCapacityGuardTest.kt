package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ReconstructionFrameCapacityGuardTest {
    @Test
    fun exactCapacityIsAcceptedWithoutOverflowWarnings() {
        val result =
            ReconstructionFrameCapacityGuard.inspect(
                cameraInputCount = ReconstructionFrameCapacityGuard.MAX_CAMERA_INPUTS,
                actorInputCount = ReconstructionFrameCapacityGuard.MAX_ACTOR_INPUTS,
                uniqueEligibleLandmarkCount = 2_048,
                eligibleSurfaceObservationCount = 4_096,
                maximumLandmarks = 2_048,
                maximumSurfaceObservations = 4_096,
            )

        assertTrue(result.isComplete)
        assertTrue(result.exceededStreams.isEmpty())
        assertTrue(result.warnings.isEmpty())
    }

    @Test
    fun eachOverflowingStreamIsIdentifiedExplicitly() {
        val result =
            ReconstructionFrameCapacityGuard.inspect(
                cameraInputCount = 129,
                actorInputCount = 65,
                uniqueEligibleLandmarkCount = 2_049,
                eligibleSurfaceObservationCount = 4_097,
                maximumLandmarks = 2_048,
                maximumSurfaceObservations = 4_096,
            )

        assertFalse(result.isComplete)
        assertEquals(
            setOf(
                ReconstructionFrameCapacityGuard.Stream.CAMERAS,
                ReconstructionFrameCapacityGuard.Stream.ACTORS,
                ReconstructionFrameCapacityGuard.Stream.LANDMARKS,
                ReconstructionFrameCapacityGuard.Stream.SURFACE_OBSERVATIONS,
            ),
            result.exceededStreams,
        )
        assertEquals(4, result.warnings.size)
        assertTrue(result.warnings.any { it.startsWith("capacity-camera-input-truncated:") })
        assertTrue(result.warnings.any { it.startsWith("capacity-actor-input-truncated:") })
        assertTrue(result.warnings.any { it.startsWith("capacity-landmark-input-truncated:") })
        assertTrue(result.warnings.any { it.startsWith("capacity-surface-observation-input-truncated:") })
    }

    @Test
    fun belowCapacityInputsRemainComplete() {
        val result =
            ReconstructionFrameCapacityGuard.inspect(
                cameraInputCount = 100,
                actorInputCount = 50,
                uniqueEligibleLandmarkCount = 1_024,
                eligibleSurfaceObservationCount = 1_000,
                maximumLandmarks = 2_048,
                maximumSurfaceObservations = 4_096,
            )

        assertTrue(result.isComplete)
        assertEquals(emptyList(), result.warnings)
    }

    @Test
    fun distinctEligibleSelectorKeepsFirstUniqueItemsAndCapsOverflowCount() {
        val result = BoundedDistinctEligibleSelector.select(
            items = listOf("a-first", "a-repeat", "b-first", "c-first", "future"),
            isEligible = { !it.startsWith("future") },
            keyOf = { it.substringBefore('-') },
            maximumItems = 2,
        )

        assertEquals(listOf("a-first", "b-first"), result.items)
        assertEquals(3, result.eligibleCountCappedAtOverflow)
    }

    @Test
    fun distinctEligibleSelectorSkipsIneligibleAndDuplicateItemsWithoutFalseOverflow() {
        val result = BoundedDistinctEligibleSelector.select(
            items = listOf("future", "a-first", "a-repeat", "b-first"),
            isEligible = { !it.startsWith("future") },
            keyOf = { it.substringBefore('-') },
            maximumItems = 3,
        )

        assertEquals(listOf("a-first", "b-first"), result.items)
        assertEquals(2, result.eligibleCountCappedAtOverflow)
    }

    @Test
    fun surfaceSelectorRetainsHighestConfidenceVisibilityScoresWithinCapacity() {
        val candidates = listOf(
            surfaceObservation(id = 1L, confidence = 0.20, visibility = 1.0),
            surfaceObservation(id = 2L, confidence = 0.95, visibility = 0.95),
            surfaceObservation(id = 3L, confidence = 0.70, visibility = 0.80),
            surfaceObservation(id = 4L, confidence = 0.80, visibility = 0.90),
            surfaceObservation(id = 5L, confidence = 0.90, visibility = 0.90),
        )

        val result = BoundedSurfaceObservationSelector.select(
            observations = candidates,
            timestampNanos = 10_000L,
            maximumObservations = 3,
        )

        assertEquals(listOf(2L, 5L, 4L), result.observations.map { it.patchId })
        assertEquals(4, result.eligibleCountCappedAtOverflow)
    }

    @Test
    fun surfaceSelectorSkipsFutureObservationsAndKeepsStableOrderForTies() {
        val candidates = listOf(
            surfaceObservation(id = 1L, confidence = 0.8, visibility = 0.8),
            surfaceObservation(id = 2L, confidence = 1.0, visibility = 1.0, timestamp = 20_000L),
            surfaceObservation(id = 3L, confidence = 0.8, visibility = 0.8),
        )

        val result = BoundedSurfaceObservationSelector.select(
            observations = candidates,
            timestampNanos = 10_000L,
            maximumObservations = 2,
        )

        assertEquals(listOf(1L, 3L), result.observations.map { it.patchId })
        assertEquals(2, result.eligibleCountCappedAtOverflow)
    }

    @Test
    fun surfaceSelectorSaturatesOverflowCountWithoutRetainingOverflowItems() {
        val candidates = (1L..100L).map { id ->
            surfaceObservation(id = id, confidence = id / 100.0, visibility = 1.0)
        }

        val result = BoundedSurfaceObservationSelector.select(
            observations = candidates,
            timestampNanos = 10_000L,
            maximumObservations = 4,
        )

        assertEquals(4, result.observations.size)
        assertEquals(5, result.eligibleCountCappedAtOverflow)
        assertEquals(listOf(100L, 99L, 98L, 97L), result.observations.map { it.patchId })
    }

    private fun surfaceObservation(
        id: Long,
        confidence: Double,
        visibility: Double,
        timestamp: Long = 1_000L,
    ) = ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation(
        patchId = id,
        timestampNanos = timestamp,
        positionMeters = com.nivukx.nox.core.model.Vec3(0.0, 0.0, 0.0),
        normal = com.nivukx.nox.core.model.Vec3(0.0, 0.0, 1.0),
        confidence = confidence,
        visibility = visibility,
        temporalConsistency = 0.9,
        photometricConsistency = 0.9,
        motionBoundaryConfidence = 0.9,
        cameraCount = 3,
        uncertaintyMeters = 0.001,
    )

    @Test(expected = IllegalArgumentException::class)
    fun reconstructionFrameHistoryHasAnUpperBound() {
        FrontierReconstructionCoordinator.Config(maximumFrameHistory = Int.MAX_VALUE)
    }

    @Test(expected = IllegalArgumentException::class)
    fun surfaceObservationCapacityHasAnUpperBound() {
        FrontierReconstructionCoordinator.Config(maximumSurfaceObservationsPerFrame = Int.MAX_VALUE)
    }

    @Test(expected = IllegalArgumentException::class)
    fun landmarkCapacityHasAnUpperBound() {
        FrontierReconstructionCoordinator.Config(maximumLandmarksPerFrame = Int.MAX_VALUE)
    }

    @Test(expected = IllegalArgumentException::class)
    fun surfacePredictionHistoryHasAnUpperBound() {
        FrontierReconstructionCoordinator.Config(maximumSurfacePredictionFrames = Int.MAX_VALUE)
    }

    @Test(expected = IllegalArgumentException::class)
    fun nonFiniteAuthorityThresholdIsRejected() {
        FrontierReconstructionCoordinator.Config(
            minimumAuthoritativeLandmarkConfidence = Double.NaN,
        )
    }

    @Test
    fun negativeCountsAreRejected() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            ReconstructionFrameCapacityGuard.inspect(
                cameraInputCount = -1,
                actorInputCount = 0,
                uniqueEligibleLandmarkCount = 0,
                eligibleSurfaceObservationCount = 0,
                maximumLandmarks = 2_048,
                maximumSurfaceObservations = 4_096,
            )
        }
    }
}
