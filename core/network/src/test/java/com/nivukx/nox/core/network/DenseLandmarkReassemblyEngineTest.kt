package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DenseLandmarkReassemblyEngineTest {

    @Test
    fun outOfOrderChunksReassembleExactlyOnce() {
        val engine = DenseLandmarkReassemblyEngine()
        val frame = testFrame(1_000_000L, 1_200)

        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame,
                "rig-a",
                10L,
                maximumLandmarksPerChunk = 200,
            )

        var result =
            engine.accept(chunks.last(), nowNanos = 10_000L)

        assertNull(result)

        chunks.dropLast(1).reversed().forEachIndexed { index, chunk ->
            result =
                engine.accept(
                    chunk,
                    nowNanos = 10_001L + index,
                ) ?: result
        }

        assertEquals(frame.timestampNanos, result?.timestampNanos)
        assertEquals(frame.landmarks.size, result?.landmarks?.size)
        assertEquals(0, engine.inFlightFrameCount())
    }

    @Test
    fun duplicateChunkIsRejectedWithoutPoisoningFrame() {
        val engine = DenseLandmarkReassemblyEngine()
        val frame = testFrame(2_000_000L, 1_000)

        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame,
                "rig-b",
                11L,
                maximumLandmarksPerChunk = 200,
            )

        engine.accept(chunks.first(), 20_000L)
        assertTrue(
            engine.accept(
                chunks.first(),
                20_001L,
            ) == null,
        )

        assertEquals(
            1,
            engine.inFlightFrameCount(),
        )
    }

    @Test
    fun expiredFramesAreEvicted() {
        val engine =
            DenseLandmarkReassemblyEngine(
                DenseLandmarkReassemblyEngine.Config(
                    maximumFrameAgeNanos = 100L,
                ),
            )
        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                testFrame(3_000_000L, 900),
                "rig-c",
                12L,
                maximumLandmarksPerChunk = 200,
            )

        engine.accept(chunks.first(), 30_000L)
        assertEquals(1, engine.inFlightFrameCount())

        engine.prune(nowNanos = 30_101L)

        assertEquals(0, engine.inFlightFrameCount())
    }

    private fun testFrame(
        timestampNanos: Long,
        count: Int,
    ) =
        DenseLandmarkWireCodec.Frame(
            timestampNanos = timestampNanos,
            landmarks =
                (0 until count).map { index ->
                    DenseLandmarkWireCodec.Landmark(
                        subjectId = index / 33,
                        landmarkId = index % 33,
                        normalizedX = 0.2f,
                        normalizedY = 0.4f,
                        confidence = 0.95f,
                        uncertainty = 0.03f,
                        visibility = 1.0f,
                        contactProbability = 0.0f,
                        temporalStability = 0.95f,
                        crossViewAgreement = 0.90f,
                        reprojectionResidualNormalized = 0.02f,
                        predicted = false,
                    )
                },
        )
}
