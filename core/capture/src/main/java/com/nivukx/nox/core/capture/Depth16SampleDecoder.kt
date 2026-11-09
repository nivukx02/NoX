package com.nivukx.nox.core.capture

/**
 * Allocation-free, bit-exact primitives for Android ImageFormat.DEPTH16 samples.
 *
 * Android packs range millimetres into bits 0..12 and confidence into bits 13..15. The helper is
 * kept free of Image, ImageReader and camera state so the format contract can be tested directly.
 */
internal object Depth16SampleDecoder {
    /**
     * Unpacks little-endian bytes. Android's ByteBuffer returns signed Byte values when widened;
     * masking each input is required to avoid sign-extension corrupting high-bit samples.
     */
    internal inline fun unpackLittleEndian(
        lowByte: Int,
        highByte: Int,
    ): Int =
        (lowByte and 0xFF) or ((highByte and 0xFF) shl 8)

    /** Returns the range code in millimetres, excluding the upper confidence bits. */
    internal inline fun rangeMillimeters(
        packedSample: Int,
    ): Int =
        packedSample and 0x1FFF

    /**
     * Confidence code zero is the special 100%-confidence code. Codes one through seven map to
     * zero through six-sevenths confidence, as specified by Android's DEPTH16 format contract.
     */
    internal inline fun confidence(
        packedSample: Int,
    ): Float {
        val code = (packedSample ushr 13) and 0x7
        return if (code == 0) 1.0f else (code - 1) / 7.0f
    }
}
