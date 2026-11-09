package com.nivukx.nox.core.capture

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Camera2Depth16DecodeSourceIntegrityTest {
    @Test
    fun productionDecoderUsesBoundedAllocationAndRetainsConfidence() {
        val candidates = listOf(
            Path.of("core/capture/src/main/java/com/nivukx/nox/core/capture/Camera2CaptureController.kt"),
            Path.of("src/main/java/com/nivukx/nox/core/capture/Camera2CaptureController.kt"),
        )
        val source = candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("Camera2CaptureController.kt was not found")

        assertTrue(source.contains("pixelCountLong > MAX_DEPTH16_PIXELS"))
        assertTrue(source.contains("(width and 1) != 0"))
        assertTrue(source.contains("(height and 1) != 0"))
        assertTrue(source.contains("pixelStride != DEPTH16_SAMPLE_BYTES"))
        assertTrue(source.contains("rowStride % DEPTH16_ROW_ALIGNMENT_BYTES != 0"))
        assertTrue(source.contains("lastByteExclusive > buffer.limit().toLong()"))
        assertTrue(source.contains("confidence = confidence"))
        assertTrue(source.contains("Depth16SampleDecoder.unpackLittleEndian"))
    }

    @Test
    fun rangeAndConfidenceFieldsAreDecodedIndependently() {
        val distanceMillimeters = 1_234
        val codeZero = distanceMillimeters
        val codeOne = distanceMillimeters or (1 shl 13)
        val codeTwo = distanceMillimeters or (2 shl 13)
        val codeSeven = distanceMillimeters or (7 shl 13)

        assertEquals(distanceMillimeters, Depth16SampleDecoder.rangeMillimeters(codeZero))
        assertEquals(distanceMillimeters, Depth16SampleDecoder.rangeMillimeters(codeOne))
        assertEquals(distanceMillimeters, Depth16SampleDecoder.rangeMillimeters(codeTwo))
        assertEquals(distanceMillimeters, Depth16SampleDecoder.rangeMillimeters(codeSeven))
        assertEquals(1.0f, Depth16SampleDecoder.confidence(codeZero))
        assertEquals(0.0f, Depth16SampleDecoder.confidence(codeOne))
        assertEquals(1.0f / 7.0f, Depth16SampleDecoder.confidence(codeTwo))
        assertEquals(6.0f / 7.0f, Depth16SampleDecoder.confidence(codeSeven))
    }

    @Test
    fun littleEndianUnpackingMasksSignedByteInputs() {
        val sample = 0xE4D2
        val unpacked = Depth16SampleDecoder.unpackLittleEndian(
            lowByte = sample and 0xFF,
            highByte = sample ushr 8,
        )

        assertEquals(sample, unpacked)
        assertEquals(0x04D2, Depth16SampleDecoder.rangeMillimeters(unpacked))
        assertEquals(6.0f / 7.0f, Depth16SampleDecoder.confidence(unpacked))
    }

    @Test
    fun unpackerAcceptsSignedByteWideningWithoutSignExtension() {
        val sample = Depth16SampleDecoder.unpackLittleEndian(
            lowByte = 0xD2.toByte().toInt(),
            highByte = 0xE4.toByte().toInt(),
        )

        assertEquals(0xE4D2, sample)
        assertEquals(0x04D2, Depth16SampleDecoder.rangeMillimeters(sample))
    }
}
