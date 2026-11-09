package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Bounded, uncertainty-aware constant-velocity Kalman layer for metric 3D joints.
 *
 * Production invariants:
 * - state count is bounded;
 * - stale tracks are evicted deterministically;
 * - reversed timestamps reset a track instead of creating a near-zero dt velocity spike;
 * - malformed measurements are rejected;
 * - process noise adapts to observed dynamics;
 * - measurement trust adapts to detector confidence and reprojection residual;
 * - large innovations are robustly down-weighted;
 * - covariance remains non-negative and numerically bounded.
 */
class AdaptiveKalman3DFilter(
    private val processNoiseMetersPerSecond2: Double = 2.5,
    private val maxDtSeconds: Double = 0.20,
    private val maximumTrackedStates: Int = 4096,
    private val staleStateTimeoutNanos: Long = 2_000_000_000L,
    private val hardResidualSigmaGate: Double = 7.0,
) {
    data class State(
        val position: Vec3,
        val velocity: Vec3,
        val uncertaintyMeters: Double,
        val confidence: Double,
        val timestampNanos: Long,
        val innovationMeters: Double = 0.0,
        val measurementWeight: Double = 1.0,
        val stateAgeMilliseconds: Long = 0L,
        val timestampReset: Boolean = false,
    )

    data class Forecast(
        val position: Vec3,
        val velocity: Vec3,
        val uncertaintyMeters: Double,
        val ageMilliseconds: Long,
        val processGrowthMeters: Double,
    )

    private data class MutableState(
        var position: Vec3,
        var velocity: Vec3,
        var positionVariance: Vec3,
        var velocityVariance: Vec3,
        var confidence: Double,
        var timestampNanos: Long,
        var innovationMeters: Double = 0.0,
        var measurementWeight: Double = 1.0,
    )

    private val states = LinkedHashMap<Int, MutableState>()

    init {
        require(processNoiseMetersPerSecond2.isFinite())
        require(processNoiseMetersPerSecond2 >= 0.0)
        require(maxDtSeconds.isFinite() && maxDtSeconds > 0.0)
        require(maximumTrackedStates in 32..16_384)
        require(staleStateTimeoutNanos in 100_000_000L..60_000_000_000L)
        require(hardResidualSigmaGate.isFinite() && hardResidualSigmaGate in 2.0..20.0)
    }

    @Synchronized
    fun update(
        id: Int,
        measurement: Vec3,
        measurementConfidence: Double,
        reprojectionErrorPx: Double,
        timestampNanos: Long,
    ): State {
        require(id >= 0)
        require(isFinite(measurement))
        require(measurementConfidence.isFinite())
        require(measurementConfidence in 0.0..1.0)
        require(reprojectionErrorPx.isFinite() && reprojectionErrorPx >= 0.0)
        require(timestampNanos >= 0L)

        pruneStale(timestampNanos)

        val previous = states[id]
        if (previous == null) {
            ensureCapacity(timestampNanos)
            val created = initialState(
                measurement = measurement,
                confidence = measurementConfidence,
                reprojectionErrorPx = reprojectionErrorPx,
                timestampNanos = timestampNanos,
            )
            states[id] = created
            return snapshot(created, timestampNanos)
        }

        val rawDtNanos = timestampNanos - previous.timestampNanos
        if (rawDtNanos <= 0L) {
            val reset = initialState(
                measurement = measurement,
                confidence = measurementConfidence,
                reprojectionErrorPx = reprojectionErrorPx,
                timestampNanos = timestampNanos,
            )
            states[id] = reset
            return snapshot(
                reset,
                timestampNanos,
            ).copy(timestampReset = true)
        }

        val dt = (rawDtNanos / 1e9)
            .coerceIn(1e-4, maxDtSeconds)

        val predictedVelocity = previous.velocity
        val predictedPosition =
            previous.position + predictedVelocity * dt

        val priorSpeed = previous.velocity.norm()
        val quality = measurementConfidence.coerceIn(0.05, 1.0)
        val residualScale =
            1.0 + (reprojectionErrorPx / 4.0).coerceAtLeast(0.0)

        val residual = measurement - predictedPosition
        val residualNorm = residual.norm()

        val dynamicProcessScale = (
            1.0 +
                priorSpeed * 0.12 +
                residualNorm / max(dt, 1e-4) * 0.015 +
                (1.0 - quality) * 0.80
            ).coerceIn(1.0, 6.0)

        val processSigma = (
            processNoiseMetersPerSecond2 *
                dynamicProcessScale
        )

        val positionVariancePrediction =
            previous.positionVariance +
                previous.velocityVariance * (dt * dt) +
                Vec3(
                    processSigma * dt * dt,
                    processSigma * dt * dt,
                    processSigma * dt * dt,
                )

        val velocityVariancePrediction =
            previous.velocityVariance +
                Vec3(
                    processSigma * dt,
                    processSigma * dt,
                    processSigma * dt,
                )

        val predictedSigma = sqrt(
            max(
                1e-12,
                average(
                    positionVariancePrediction.x,
                    positionVariancePrediction.y,
                    positionVariancePrediction.z,
                ),
            ),
        )

        val baseMeasurementVariance = (
            0.0006 +
                0.018 * residualScale / quality
        ).coerceIn(0.0006, 0.75)

        val huberScale = robustMeasurementScale(
            residualMeters = residualNorm,
            sigmaMeters = sqrt(baseMeasurementVariance),
            hardGateSigma = hardResidualSigmaGate,
        )

        val measurementVariance = (
            baseMeasurementVariance /
                huberScale.coerceAtLeast(0.08)
        ).coerceIn(0.0006, 1.50)

        fun gain(variance: Double): Double =
            variance / (variance + measurementVariance)

        val kx = gain(positionVariancePrediction.x)
        val ky = gain(positionVariancePrediction.y)
        val kz = gain(positionVariancePrediction.z)

        val correctedPosition = Vec3(
            predictedPosition.x + kx * residual.x,
            predictedPosition.y + ky * residual.y,
            predictedPosition.z + kz * residual.z,
        )

        val dtSafe = dt.coerceAtLeast(1e-4)
        val observedVelocity = residual / dtSafe
        val velocityBlend = (
            0.12 +
                0.48 * quality +
                0.20 * huberScale
            ).coerceIn(0.12, 0.80)

        val correctedVelocity = Vec3(
            predictedVelocity.x * (1.0 - velocityBlend) +
                observedVelocity.x * velocityBlend,
            predictedVelocity.y * (1.0 - velocityBlend) +
                observedVelocity.y * velocityBlend,
            predictedVelocity.z * (1.0 - velocityBlend) +
                observedVelocity.z * velocityBlend,
        )

        val correctedVariance = Vec3(
            ((1.0 - kx) * positionVariancePrediction.x)
                .coerceIn(1e-10, 25.0),
            ((1.0 - ky) * positionVariancePrediction.y)
                .coerceIn(1e-10, 25.0),
            ((1.0 - kz) * positionVariancePrediction.z)
                .coerceIn(1e-10, 25.0),
        )

        val correctedVelocityVariance = Vec3(
            velocityVariancePrediction.x
                .coerceIn(1e-10, 400.0),
            velocityVariancePrediction.y
                .coerceIn(1e-10, 400.0),
            velocityVariancePrediction.z
                .coerceIn(1e-10, 400.0),
        )

        previous.position = correctedPosition
        previous.velocity = correctedVelocity
        previous.positionVariance = correctedVariance
        previous.velocityVariance = correctedVelocityVariance
        previous.confidence =
            (
                previous.confidence * 0.25 +
                    quality * 0.75 * huberScale
            ).coerceIn(0.0, 1.0)
        previous.timestampNanos = timestampNanos
        previous.innovationMeters = residualNorm
        previous.measurementWeight = (
            huberScale * quality
        ).coerceIn(0.0, 1.0)

        return snapshot(previous, timestampNanos)
    }

    /**
     * Forward-project an active state without mutating it.
     *
     * This is deliberately a read-only forecast: occlusion recovery must never advance the
     * authoritative filter clock or silently turn a predicted sample into a measured one.
     */
    @Synchronized
    fun forecast(
        id: Int,
        timestampNanos: Long,
    ): Forecast? {
        require(id >= 0)
        require(timestampNanos >= 0L)

        val state = states[id] ?: return null
        val deltaNanos =
            timestampNanos -
                state.timestampNanos

        if (deltaNanos <= 0L) {
            return Forecast(
                position = state.position,
                velocity = state.velocity,
                uncertaintyMeters = sqrt(
                    max(
                        0.0,
                        average(
                            state.positionVariance.x,
                            state.positionVariance.y,
                            state.positionVariance.z,
                        ),
                    ),
                ),
                ageMilliseconds = 0L,
                processGrowthMeters = 0.0,
            )
        }

        val dt = (
            deltaNanos.toDouble() /
                1e9
            ).coerceIn(
                0.0,
                maxDtSeconds,
            )

        val predictedPosition =
            state.position +
                state.velocity * dt

        val processSigma =
            processNoiseMetersPerSecond2
                .coerceAtLeast(0.0)

        val predictedVariance =
            state.positionVariance +
                state.velocityVariance * (dt * dt) +
                Vec3(
                    processSigma * dt * dt,
                    processSigma * dt * dt,
                    processSigma * dt * dt,
                )

        val uncertainty =
            sqrt(
                max(
                    1e-12,
                    average(
                        predictedVariance.x,
                        predictedVariance.y,
                        predictedVariance.z,
                    ),
                ),
            )

        val currentUncertainty =
            sqrt(
                max(
                    1e-12,
                    average(
                        state.positionVariance.x,
                        state.positionVariance.y,
                        state.positionVariance.z,
                    ),
                ),
            )

        return Forecast(
            position = predictedPosition,
            velocity = state.velocity,
            uncertaintyMeters =
                uncertainty.coerceIn(1e-6, 25.0),
            ageMilliseconds =
                (
                    deltaNanos /
                        1_000_000L
                ).coerceIn(
                    0L,
                    Long.MAX_VALUE,
                ),
            processGrowthMeters =
                (
                    uncertainty -
                        currentUncertainty
                ).coerceAtLeast(0.0),
        )
    }

    @Synchronized
    fun remove(id: Int) {
        require(id >= 0)
        states.remove(id)
    }

    @Synchronized
    fun clear() {
        states.clear()
    }

    @Synchronized
    fun snapshot(): Map<Int, State> {
        val now = states.values.maxOfOrNull { it.timestampNanos } ?: 0L
        return states.mapValues { snapshot(it.value, now) }
    }

    @Synchronized
    fun activeStateCount(): Int = states.size

    @Synchronized
    fun prune(timestampNanos: Long) {
        require(timestampNanos >= 0L)
        pruneStale(timestampNanos)
    }

    private fun initialState(
        measurement: Vec3,
        confidence: Double,
        reprojectionErrorPx: Double,
        timestampNanos: Long,
    ): MutableState {
        val variance = initialVariance(
            confidence = confidence,
            reprojectionErrorPx = reprojectionErrorPx,
        )
        return MutableState(
            position = measurement,
            velocity = Vec3(0.0, 0.0, 0.0),
            positionVariance = variance,
            velocityVariance = Vec3(4.0, 4.0, 4.0),
            confidence = confidence,
            timestampNanos = timestampNanos,
        )
    }

    private fun ensureCapacity(now: Long) {
        if (states.size < maximumTrackedStates) return

        pruneStale(now)
        if (states.size < maximumTrackedStates) return

        val eviction = states.entries.minWithOrNull(
            compareBy<Map.Entry<Int, MutableState>>(
                { it.value.confidence },
                { it.value.timestampNanos },
                { it.key },
            ),
        ) ?: return
        states.remove(eviction.key)
    }

    private fun pruneStale(now: Long) {
        if (states.isEmpty()) return
        val iterator = states.entries.iterator()
        while (iterator.hasNext()) {
            val state = iterator.next().value
            val age = now - state.timestampNanos
            if (age > staleStateTimeoutNanos) {
                iterator.remove()
            }
        }
    }

    private fun robustMeasurementScale(
        residualMeters: Double,
        sigmaMeters: Double,
        hardGateSigma: Double,
    ): Double {
        if (residualMeters <= sigmaMeters) return 1.0

        val normalized = residualMeters /
            sigmaMeters.coerceAtLeast(1e-6)

        if (normalized >= hardGateSigma) {
            return 0.08
        }

        return (
            1.0 / normalized
        ).coerceIn(0.08, 1.0)
    }

    private fun initialVariance(
        confidence: Double,
        reprojectionErrorPx: Double,
    ): Vec3 {
        val uncertainty = (
            0.006 +
                reprojectionErrorPx.coerceAtLeast(0.0) / 160.0 +
                (1.0 - confidence.coerceIn(0.0, 1.0)) * 0.030
            ).coerceIn(0.004, 0.40)

        val squared = uncertainty * uncertainty
        return Vec3(squared, squared, squared)
    }

    private fun snapshot(
        state: MutableState,
        nowNanos: Long,
    ): State =
        State(
            position = state.position,
            velocity = state.velocity,
            uncertaintyMeters = sqrt(
                max(
                    0.0,
                    average(
                        state.positionVariance.x,
                        state.positionVariance.y,
                        state.positionVariance.z,
                    ),
                ),
            ),
            confidence = state.confidence,
            timestampNanos = state.timestampNanos,
            innovationMeters = state.innovationMeters,
            measurementWeight = state.measurementWeight,
            stateAgeMilliseconds = (
                (nowNanos - state.timestampNanos)
                    .coerceAtLeast(0L) / 1_000_000L
            ),
        )

    private fun isFinite(value: Vec3): Boolean =
        value.x.isFinite() &&
            value.y.isFinite() &&
            value.z.isFinite()

    private fun average(
        a: Double,
        b: Double,
        c: Double,
    ): Double =
        (a + b + c) / 3.0
}
