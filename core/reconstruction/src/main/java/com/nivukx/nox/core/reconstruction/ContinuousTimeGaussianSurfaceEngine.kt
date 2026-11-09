package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import kotlin.math.max
import kotlin.math.min

/** Continuous-time dynamic surface field for arbitrary timestamp queries.
 * Keeps temporal state explicit so retiming does not become frame-index interpolation.
 */
class ContinuousTimeGaussianSurfaceEngine(
    private val config: Config = Config(),
) : AutoCloseable {
    data class Config(
        val maximumFields: Int = 16_384,
        val maximumHistorySamples: Int = 8,
        val maximumPredictionSeconds: Double = 0.12,
        val maximumVelocityMetersPerSecond: Double = 12.0,
        val maximumAccelerationMetersPerSecond2: Double = 180.0,
        val maximumJerkMetersPerSecond3: Double = 900.0,
        val covarianceGrowthMeters2PerSecond: Double = 2.5e-5,
        val maximumInterpolationGapSeconds: Double = 0.25,
        val maximumHermiteDeviationMeters: Double = 0.12,
        val maximumQuinticDeviationMeters: Double = 0.10,
    ) {
        init {
            require(maximumFields > 0)
            require(maximumHistorySamples >= 3)
            require(maximumPredictionSeconds > 0.0)
            require(maximumVelocityMetersPerSecond > 0.0)
            require(maximumAccelerationMetersPerSecond2 > 0.0)
            require(maximumJerkMetersPerSecond3 > 0.0)
            require(covarianceGrowthMeters2PerSecond >= 0.0)
            require(maximumInterpolationGapSeconds > 0.0)
            require(maximumHermiteDeviationMeters > 0.0)
            require(maximumQuinticDeviationMeters > 0.0)
        }
    }

    data class Sample(
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3 = Vec3(0.0,0.0,0.0),
        val accelerationMetersPerSecond2: Vec3 = Vec3(0.0,0.0,0.0),
        val jerkMetersPerSecond3: Vec3 = Vec3(0.0,0.0,0.0),
        val covariance: UncertaintyEnvelope,
        val confidence: Double,
        val origin: EvidenceOrigin,
    )

    data class Query(
        val fieldId: Long,
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3,
        val jerkMetersPerSecond3: Vec3 = Vec3(0.0,0.0,0.0),
        val covariance: UncertaintyEnvelope,
        val confidence: Double,
        val origin: EvidenceOrigin,
        val interpolated: Boolean,
        val temporalGapSeconds: Double,
        val kinematicFallback: Boolean = false,
    )

    private val histories = LinkedHashMap<Long, ArrayDeque<Sample>>()
    private var closed = false

    @Synchronized
    fun observe(fieldId: Long, sample: Sample) {
        check(!closed)
        require(fieldId >= 0L)
        require(sample.timestampNanos >= 0L)
        require(
            finite(sample.positionMeters) &&
                finite(sample.velocityMetersPerSecond) &&
                finite(sample.accelerationMetersPerSecond2) &&
                finite(sample.jerkMetersPerSecond3)
        )
        require(sample.confidence in 0.0..1.0)
        val history = histories.getOrPut(fieldId) { ArrayDeque() }
        while (history.isNotEmpty() && sample.timestampNanos <= history.last().timestampNanos) history.removeLast()
        history.addLast(sample)
        while (history.size > config.maximumHistorySamples) history.removeFirst()
        while (histories.size > config.maximumFields) histories.remove(histories.entries.minByOrNull { it.value.lastOrNull()?.timestampNanos ?: Long.MIN_VALUE }?.key ?: return)
    }

    @Synchronized
    fun query(fieldId: Long, timestampNanos: Long): Query? {
        check(!closed)
        require(timestampNanos >= 0L)
        val history = histories[fieldId] ?: return null
        if (history.isEmpty()) return null
        val exact = history.firstOrNull { it.timestampNanos == timestampNanos }
        if (exact != null) return toQuery(fieldId,timestampNanos,exact,0.0,false)
        val before = history.lastOrNull { it.timestampNanos < timestampNanos }
        val after = history.firstOrNull { it.timestampNanos > timestampNanos }
        if (before != null && after != null) {
            val spanNanos = (after.timestampNanos - before.timestampNanos).coerceAtLeast(1L)
            val span = spanNanos.toDouble() / 1e9
            val alpha = ((timestampNanos - before.timestampNanos).toDouble() / spanNanos.toDouble()).coerceIn(0.0,1.0)
            val interpolated = interpolate(before, after, alpha, span)
            return toQuery(
                fieldId,
                timestampNanos,
                interpolated.first,
                span,
                true,
                interpolated.second,
            )
        }
        val anchor = before ?: after ?: return null
        val previous =
            if (before != null) {
                history.lastOrNull {
                    it.timestampNanos < anchor.timestampNanos
                }
            } else {
                history.firstOrNull {
                    it.timestampNanos > anchor.timestampNanos
                }
            }
        val dt =
            (timestampNanos - anchor.timestampNanos)
                .toDouble() /
                1e9
        if (kotlin.math.abs(dt) > config.maximumPredictionSeconds) {
            return null
        }
        val sample =
            extrapolate(
                anchor = anchor,
                previous = previous,
                dt = dt,
            )
        return toQuery(
            fieldId,
            timestampNanos,
            sample,
            kotlin.math.abs(dt),
            true,
        )
    }

    @Synchronized
    fun activeFieldCount(): Int = histories.size

    private fun interpolate(
        a: Sample,
        b: Sample,
        alpha: Double,
        spanSeconds: Double,
    ): Pair<Sample, Boolean> {
        val t =
            alpha.coerceIn(0.0, 1.0)
        val boundedVelocityA =
            a.velocityMetersPerSecond.clampNorm(
                config.maximumVelocityMetersPerSecond,
            )
        val boundedVelocityB =
            b.velocityMetersPerSecond.clampNorm(
                config.maximumVelocityMetersPerSecond,
            )
        val boundedAccelerationA =
            a.accelerationMetersPerSecond2.clampNorm(
                config.maximumAccelerationMetersPerSecond2,
            )
        val boundedAccelerationB =
            b.accelerationMetersPerSecond2.clampNorm(
                config.maximumAccelerationMetersPerSecond2,
            )

        val t2 = t * t
        val t3 = t2 * t

        val h00 =
            2.0 * t3 - 3.0 * t2 + 1.0
        val h10 =
            t3 - 2.0 * t2 + t
        val h01 =
            -2.0 * t3 + 3.0 * t2
        val h11 =
            t3 - t2

        val hermitePosition =
            a.positionMeters * h00 +
                boundedVelocityA *
                    (h10 * spanSeconds) +
                b.positionMeters * h01 +
                boundedVelocityB *
                    (h11 * spanSeconds)

        val linearPosition =
            a.positionMeters.lerp(
                b.positionMeters,
                t,
            )

        val hermiteDeviation =
            hermitePosition.distanceTo(
                linearPosition,
            )

        val validHermite =
            finite(hermitePosition) &&
                hermiteDeviation <=
                    config.maximumHermiteDeviationMeters

        val dH00 =
            6.0 * t2 - 6.0 * t
        val dH10 =
            3.0 * t2 - 4.0 * t + 1.0
        val dH01 =
            -6.0 * t2 + 6.0 * t
        val dH11 =
            3.0 * t2 - 2.0 * t

        val hermiteVelocity =
            (
                a.positionMeters * dH00 +
                    boundedVelocityA *
                        (dH10 * spanSeconds) +
                    b.positionMeters * dH01 +
                    boundedVelocityB *
                        (dH11 * spanSeconds)
                ) /
                spanSeconds.coerceAtLeast(1e-9)

        val linearVelocity =
            (
                b.positionMeters -
                    a.positionMeters
                ) /
                spanSeconds.coerceAtLeast(1e-9)

        val quintic =
            quinticInterpolate(
                a = a,
                b = b,
                t = t,
                spanSeconds = spanSeconds,
                velocityA = boundedVelocityA,
                velocityB = boundedVelocityB,
                accelerationA = boundedAccelerationA,
                accelerationB = boundedAccelerationB,
            )

        val quinticDeviation =
            quintic.position.distanceTo(
                hermitePosition,
            )

        val validQuintic =
            finite(quintic.position) &&
                finite(quintic.velocity) &&
                finite(quintic.acceleration) &&
                quinticDeviation <=
                    config.maximumQuinticDeviationMeters

        val position =
            when {
                validQuintic ->
                    quintic.position
                validHermite ->
                    hermitePosition
                else ->
                    linearPosition
            }

        val velocity =
            when {
                validQuintic ->
                    quintic.velocity
                validHermite ->
                    hermiteVelocity
                else ->
                    linearVelocity
            }.clampNorm(
                config.maximumVelocityMetersPerSecond,
            )

        val acceleration =
            when {
                validQuintic ->
                    quintic.acceleration
                else ->
                    a.accelerationMetersPerSecond2.lerp(
                        b.accelerationMetersPerSecond2,
                        t,
                    )
            }.clampNorm(
                config.maximumAccelerationMetersPerSecond2,
            )

        val gapQuality =
            (
                1.0 -
                    maxOf(
                        0.0,
                        spanSeconds /
                            config.maximumInterpolationGapSeconds -
                            1.0,
                    ) * 0.35
            ).coerceIn(
                0.40,
                1.0,
            )

        val fallbackPenalty =
            when {
                validQuintic -> 1.0
                validHermite -> 0.90
                else -> 0.72
            }

        val covariance =
            UncertaintyEnvelope(
                a.covariance.xxMeters2.lerp(
                    b.covariance.xxMeters2,
                    t,
                ),
                a.covariance.yyMeters2.lerp(
                    b.covariance.yyMeters2,
                    t,
                ),
                a.covariance.zzMeters2.lerp(
                    b.covariance.zzMeters2,
                    t,
                ),
                a.covariance.xyMeters2.lerp(
                    b.covariance.xyMeters2,
                    t,
                ),
                a.covariance.xzMeters2.lerp(
                    b.covariance.xzMeters2,
                    t,
                ),
                a.covariance.yzMeters2.lerp(
                    b.covariance.yzMeters2,
                    t,
                ),
                min(
                    a.covariance.confidence,
                    b.covariance.confidence,
                ) *
                    gapQuality *
                    fallbackPenalty *
                    (
                        1.0 -
                            0.05 *
                            min(
                                1.0,
                                t *
                                    (1.0 - t) *
                                    4.0,
                            )
                    ),
            )

        return Sample(
            timestampNanos =
                a.timestampNanos +
                    (
                        (
                            b.timestampNanos -
                                a.timestampNanos
                        ) * t
                    ).toLong(),
            positionMeters = position,
            velocityMetersPerSecond = velocity,
            accelerationMetersPerSecond2 = acceleration,
            covariance = covariance,
            confidence =
                min(
                    a.confidence,
                    b.confidence,
                ) *
                    gapQuality *
                    fallbackPenalty,
            origin =
                if (
                    a.origin.authorityRank >=
                        b.origin.authorityRank
                ) {
                    a.origin
                } else {
                    b.origin
                },
        ) to !validQuintic
    }

    private data class QuinticState(
        val position: Vec3,
        val velocity: Vec3,
        val acceleration: Vec3,
    )

    private fun quinticInterpolate(
        a: Sample,
        b: Sample,
        t: Double,
        spanSeconds: Double,
        velocityA: Vec3,
        velocityB: Vec3,
        accelerationA: Vec3,
        accelerationB: Vec3,
    ): QuinticState {
        val delta =
            b.positionMeters -
                a.positionMeters

        val tScale =
            spanSeconds.coerceAtLeast(1e-9)
        val tScale2 =
            tScale * tScale

        val c0 = a.positionMeters
        val c1 = velocityA * tScale
        val c2 =
            accelerationA *
                (0.5 * tScale2)

        val c3 =
            delta * 10.0 -
                velocityA * (6.0 * tScale) -
                accelerationA * (1.5 * tScale2) -
                velocityB * (4.0 * tScale) +
                accelerationB * (0.5 * tScale2)

        val c4 =
            delta * -15.0 +
                velocityA * (8.0 * tScale) +
                accelerationA * (1.5 * tScale2) +
                velocityB * (7.0 * tScale) -
                accelerationB * tScale2

        val c5 =
            delta * 6.0 -
                velocityA * (3.0 * tScale) -
                accelerationA * (0.5 * tScale2) -
                velocityB * (3.0 * tScale) +
                accelerationB * (0.5 * tScale2)

        val t2 = t * t
        val t3 = t2 * t
        val t4 = t3 * t
        val t5 = t4 * t

        val position =
            c0 +
                c1 * t +
                c2 * t2 +
                c3 * t3 +
                c4 * t4 +
                c5 * t5

        val velocityNormalized =
            c1 +
                c2 * (2.0 * t) +
                c3 * (3.0 * t2) +
                c4 * (4.0 * t3) +
                c5 * (5.0 * t4)

        val accelerationNormalized =
            c2 * 2.0 +
                c3 * (6.0 * t) +
                c4 * (12.0 * t2) +
                c5 * (20.0 * t3)

        return QuinticState(
            position = position,
            velocity =
                velocityNormalized /
                    tScale,
            acceleration =
                accelerationNormalized /
                    tScale2,
        )
    }


    private fun extrapolate(
        anchor: Sample,
        previous: Sample?,
        dt: Double,
    ): Sample {
        val boundedVelocity =
            anchor.velocityMetersPerSecond.clampNorm(
                config.maximumVelocityMetersPerSecond,
            )
        val boundedAcceleration =
            anchor.accelerationMetersPerSecond2.clampNorm(
                config.maximumAccelerationMetersPerSecond2,
            )

        val inferredJerk =
            if (previous != null) {
                val spanSeconds =
                    (
                        anchor.timestampNanos -
                            previous.timestampNanos
                    ).toDouble() /
                        1e9
                if (kotlin.math.abs(spanSeconds) > 1e-9) {
                    (
                        anchor.accelerationMetersPerSecond2 -
                            previous.accelerationMetersPerSecond2
                    ) *
                        (
                            1.0 /
                                spanSeconds
                        )
                } else {
                    anchor.jerkMetersPerSecond3
                }
            } else {
                anchor.jerkMetersPerSecond3
            }

        val boundedJerk =
            inferredJerk.clampNorm(
                config.maximumJerkMetersPerSecond3,
            )

        val t =
            dt.coerceIn(
                -config.maximumPredictionSeconds,
                config.maximumPredictionSeconds,
            )
        val t2 = t * t
        val t3 = t2 * t

        val position =
            anchor.positionMeters +
                boundedVelocity * t +
                boundedAcceleration *
                    (0.5 * t2) +
                boundedJerk *
                    (t3 / 6.0)

        val predictedVelocity =
            (
                boundedVelocity +
                    boundedAcceleration * t +
                    boundedJerk *
                        (0.5 * t2)
            ).clampNorm(
                config.maximumVelocityMetersPerSecond,
            )

        val predictedAcceleration =
            (
                boundedAcceleration +
                    boundedJerk * t
            ).clampNorm(
                config.maximumAccelerationMetersPerSecond2,
            )

        val normalizedHorizon =
            (
                kotlin.math.abs(t) /
                    config.maximumPredictionSeconds
            ).coerceIn(0.0, 1.0)

        val jerkRisk =
            (
                boundedJerk.norm() /
                    config.maximumJerkMetersPerSecond3
            ).coerceIn(0.0, 1.0)

        val growth =
            config.covarianceGrowthMeters2PerSecond *
                kotlin.math.abs(t) *
                (
                    1.0 +
                        jerkRisk * 0.50
                )

        return Sample(
            timestampNanos =
                anchor.timestampNanos +
                    (t * 1e9).toLong(),
            positionMeters = position,
            velocityMetersPerSecond = predictedVelocity,
            accelerationMetersPerSecond2 =
                predictedAcceleration,
            jerkMetersPerSecond3 = boundedJerk,
            covariance =
                UncertaintyEnvelope(
                    anchor.covariance.xxMeters2 + growth,
                    anchor.covariance.yyMeters2 + growth,
                    anchor.covariance.zzMeters2 + growth,
                    anchor.covariance.xyMeters2,
                    anchor.covariance.xzMeters2,
                    anchor.covariance.yzMeters2,
                    anchor.covariance.confidence *
                        (
                            0.92 -
                                0.10 * jerkRisk
                        ).coerceIn(
                            0.70,
                            0.92,
                        ),
                ),
            confidence =
                (
                    anchor.confidence *
                        (
                            1.0 -
                                0.35 * normalizedHorizon -
                                0.12 * jerkRisk
                        )
                    ).coerceIn(
                        0.10,
                        1.0,
                    ),
            origin = EvidenceOrigin.TEMPORALLY_PROPAGATED,
        )
    }

    private fun toQuery(
        id: Long,
        time: Long,
        sample: Sample,
        gap: Double,
        interpolated: Boolean,
        kinematicFallback: Boolean = false,
    ) = Query(
        fieldId = id,
        timestampNanos = time,
        positionMeters = sample.positionMeters,
        velocityMetersPerSecond = sample.velocityMetersPerSecond,
        jerkMetersPerSecond3 = sample.jerkMetersPerSecond3,
        covariance = sample.covariance,
        confidence = sample.confidence,
        origin = sample.origin,
        interpolated = interpolated,
        temporalGapSeconds = gap,
        kinematicFallback = kinematicFallback,
    )

    override fun close() { synchronized(this) { histories.clear(); closed=true } }

    private fun finite(v:Vec3)=v.x.isFinite()&&v.y.isFinite()&&v.z.isFinite()

private fun Vec3.distanceTo(other: Vec3): Double =
    kotlin.math.sqrt(
        (x - other.x) * (x - other.x) +
            (y - other.y) * (y - other.y) +
            (z - other.z) * (z - other.z),
    )
}

private fun Double.lerp(other:Double,alpha:Double)=this*(1.0-alpha)+other*alpha
private fun Vec3.clampNorm(maxNorm:Double):Vec3 { val n=norm(); return if(n<=maxNorm||n<1e-12)this else this*(maxNorm/n) }