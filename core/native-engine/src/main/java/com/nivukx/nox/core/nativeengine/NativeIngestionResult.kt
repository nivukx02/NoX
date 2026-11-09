package com.nivukx.nox.core.nativeengine

data class NativeIngestionResult(
    val accepted: Boolean,
    val duplicate: Boolean,
    val outOfOrder: Boolean,
    val gap: Boolean,
    val overflow: Boolean,
    val gapNanos: Long,
    val queueDepth: Int,
    val epoch: Long,
) {
    companion object {
        fun fromNative(fields: LongArray): NativeIngestionResult {
            require(fields.size == 8)
            return NativeIngestionResult(
                accepted = fields[0] != 0L,
                duplicate = fields[1] != 0L,
                outOfOrder = fields[2] != 0L,
                gap = fields[3] != 0L,
                overflow = fields[4] != 0L,
                gapNanos = fields[5],
                queueDepth = fields[6].toInt(),
                epoch = fields[7],
            )
        }
    }
}
