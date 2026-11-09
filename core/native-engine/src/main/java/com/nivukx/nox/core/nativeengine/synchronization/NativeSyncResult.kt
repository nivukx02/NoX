package com.nivukx.nox.core.nativeengine.synchronization

data class NativeSyncResult(
    val offsetNanos: Double,
    val driftPpm: Double,
    val jitterNanos: Double,
    val latencyNanos: Double,
    val subframeDeltaNanos: Double,
    val subframeConfidence: Double = 0.0,
    val subframeSharpness: Double = 0.0,
    val temporalConfidence: Double,
    val residualP95Nanos: Double = 0.0,
    val residualMaxNanos: Double = 0.0,
    val outlierRatio: Double = 0.0,
    val clockSupportRatio: Double = 0.0,
    val offsetUncertaintyNanos: Double = 0.0,
    val driftUncertaintyPpm: Double = 0.0,
    val clockIterations: Int = 0,
    val sampleCount: Int = 0,
    val valid: Boolean,
) {
    init {
        require(offsetNanos.isFinite())
        require(driftPpm.isFinite())
        require(jitterNanos.isFinite() && jitterNanos >= 0.0)
        require(latencyNanos.isFinite() && latencyNanos >= 0.0)
        require(subframeDeltaNanos.isFinite())
        require(subframeConfidence.isFinite() && subframeConfidence in 0.0..1.0)
        require(subframeSharpness.isFinite() && subframeSharpness in 0.0..1.0)
        require(temporalConfidence.isFinite() && temporalConfidence in 0.0..1.0)
        require(residualP95Nanos.isFinite() && residualP95Nanos >= 0.0)
        require(residualMaxNanos.isFinite() && residualMaxNanos >= 0.0)
        require(outlierRatio.isFinite() && outlierRatio in 0.0..1.0)
        require(clockSupportRatio.isFinite() && clockSupportRatio in 0.0..1.0)
        require(offsetUncertaintyNanos.isFinite() && offsetUncertaintyNanos >= 0.0)
        require(driftUncertaintyPpm.isFinite() && driftUncertaintyPpm >= 0.0)
        require(clockIterations >= 0)
        require(sampleCount >= 0)
    }

    companion object {
        private const val EXPECTED_NATIVE_FIELDS = 17

        fun fromNative(fields: DoubleArray): NativeSyncResult {
            require(fields.size == EXPECTED_NATIVE_FIELDS) {
                "Expected $EXPECTED_NATIVE_FIELDS native synchronization fields"
            }

            return NativeSyncResult(
                offsetNanos = fields[0],
                driftPpm = fields[1],
                jitterNanos = fields[2],
                latencyNanos = fields[3],
                subframeDeltaNanos = fields[4],
                subframeConfidence = fields[5].coerceIn(0.0, 1.0),
                subframeSharpness = fields[6].coerceIn(0.0, 1.0),
                temporalConfidence = fields[7].coerceIn(0.0, 1.0),
                residualP95Nanos = fields[8].coerceAtLeast(0.0),
                residualMaxNanos = fields[9].coerceAtLeast(0.0),
                outlierRatio = fields[10].coerceIn(0.0, 1.0),
                clockSupportRatio = fields[11].coerceIn(0.0, 1.0),
                offsetUncertaintyNanos = fields[12].coerceAtLeast(0.0),
                driftUncertaintyPpm = fields[13].coerceAtLeast(0.0),
                clockIterations = fields[14].coerceAtLeast(0.0).toInt(),
                sampleCount = fields[15].coerceAtLeast(0.0).toInt(),
                valid = fields[16] >= 0.5,
            )
        }
    }
}
