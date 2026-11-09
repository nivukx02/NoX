package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertTrue

class DenseContactEvidenceAdapterTest {

    @Test
    fun closeHandAndBodyEvidenceProducesProtectedLandmarks() {
        val adapter = DenseContactEvidenceAdapter()

        val result = adapter.evaluate(
            DenseContactEvidenceAdapter.Input(
                timestampNanos = 1_000_000_000L,
                actorId = 7L,
                handLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(
                        landmarkId = 760,
                        x = 0.50,
                        y = 0.50,
                        confidence = 0.96,
                    ),
                ),
                counterpartLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(
                        landmarkId = 11,
                        x = 0.535,
                        y = 0.515,
                        confidence = 0.94,
                    ),
                ),
            ),
        )

        assertTrue(result.protectedLandmarks.contains(760))
        assertTrue(result.protectedLandmarks.contains(11))
        assertTrue(result.contactConfidence > 0.0)
        assertTrue(result.landmarkVisibility[760] != null)
        assertTrue(result.landmarkUncertaintyMultiplier[760] != null)
        adapter.close()
    }

    @Test
    fun distantEvidenceDoesNotInventContact() {
        val adapter = DenseContactEvidenceAdapter()

        val result = adapter.evaluate(
            DenseContactEvidenceAdapter.Input(
                timestampNanos = 1_000_000_000L,
                actorId = 7L,
                handLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(
                        landmarkId = 760,
                        x = 0.10,
                        y = 0.10,
                        confidence = 0.96,
                    ),
                ),
                counterpartLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(
                        landmarkId = 11,
                        x = 0.70,
                        y = 0.80,
                        confidence = 0.94,
                    ),
                ),
            ),
        )

        assertTrue(result.protectedLandmarks.isEmpty())
        assertTrue(result.contactConfidence <= 0.25)
        adapter.close()
    }

    @Test
    fun repeatedFramesNeedTemporalContinuityForTrust() {
        val adapter = DenseContactEvidenceAdapter()

        val first = adapter.evaluate(
            DenseContactEvidenceAdapter.Input(
                timestampNanos = 1_000_000_000L,
                actorId = 3L,
                handLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(760, 0.50, 0.50, 0.95),
                ),
                counterpartLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(11, 0.53, 0.51, 0.95),
                ),
            ),
        )

        val second = adapter.evaluate(
            DenseContactEvidenceAdapter.Input(
                timestampNanos = 1_016_000_000L,
                actorId = 3L,
                handLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(760, 0.501, 0.501, 0.96),
                ),
                counterpartLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(11, 0.531, 0.511, 0.96),
                ),
            ),
        )

        assertTrue(first.contactConfidence >= 0.0)
        assertTrue(second.contactConfidence >= first.contactConfidence)
        assertTrue(second.protectedLandmarks.isNotEmpty())
        adapter.close()
    }

    @Test
    fun timestampReversalFailsClosed() {
        val adapter = DenseContactEvidenceAdapter()

        adapter.evaluate(
            DenseContactEvidenceAdapter.Input(
                timestampNanos = 1_000_000_000L,
                actorId = 3L,
                handLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(760, 0.50, 0.50, 0.95),
                ),
                counterpartLandmarks = listOf(
                    DenseContactEvidenceAdapter.Point(11, 0.53, 0.51, 0.95),
                ),
            ),
        )

        val reversed = adapter.evaluate(
            DenseContactEvidenceAdapter.Input(
                timestampNanos = 900_000_000L,
                actorId = 3L,
                handLandmarks = emptyList(),
                counterpartLandmarks = emptyList(),
            ),
        )

        assertTrue(reversed.hardFailure)
        adapter.close()
    }
}
