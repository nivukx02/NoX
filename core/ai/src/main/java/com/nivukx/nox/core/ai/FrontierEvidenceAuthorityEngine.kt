package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.WorldSpaceProposal
import kotlin.math.max

/**
 * Correlation-aware authority reducer that lives inside the AI module boundary.
 * Signals from the same dependency family receive a diminishing-return factor so dependent
 * observations do not manufacture confidence. Calibrated geometry remains the strongest source.
 */
class FrontierEvidenceAuthorityEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumSignalsPerGroup: Int = 8,
        val temporalCorrelationPenalty: Double = 0.18,
        val globalCorrelationPenalty: Double = 0.30,
        val learnedAuthorityScale: Double = 0.72,
        val identityAmbiguityPenalty: Double = 0.38,
        val metricAuthorityInfluence: Double = 0.15,
    ) {
        init {
            require(maximumSignalsPerGroup >= 1)
            require(temporalCorrelationPenalty in 0.0..0.8)
            require(globalCorrelationPenalty in 0.0..0.8)
            require(learnedAuthorityScale in 0.0..1.0)
            require(identityAmbiguityPenalty in 0.0..1.0)
            require(metricAuthorityInfluence in 0.0..0.35)
        }
    }

    enum class State { STRONG, GUARDED, DEGRADED, REJECTED }

    data class Result(
        val state: State,
        val authorityScore: Double,
        val geometryAuthority: Double,
        val temporalAuthority: Double,
        val learnedAuthority: Double,
        val denseAuthority: Double,
        val metricAuthority: Double = 0.0,
        val correlationAdjustment: Double,
        val effectiveEvidenceCount: Double,
        val identityAmbiguityRisk: Double = 0.0,
        val warnings: List<String>,
    )

    fun evaluate(
        proposals: List<WorldSpaceProposal>,
        denseTrackCount: Int,
        actorCount: Int,
        identityAmbiguityRisk: Double = 0.0,
    ): Result {
        require(identityAmbiguityRisk.isFinite() && identityAmbiguityRisk in 0.0..1.0)
        val usable = proposals.filter { p ->
            p.positionMeters.x.isFinite() && p.positionMeters.y.isFinite() && p.positionMeters.z.isFinite()
        }
        if (usable.isEmpty()) {
            return Result(
                state = State.REJECTED,
                authorityScore = 0.0,
                geometryAuthority = 0.0,
                temporalAuthority = 0.0,
                learnedAuthority = 0.0,
                denseAuthority = 0.0,
                metricAuthority = 0.0,
                correlationAdjustment = 0.0,
                effectiveEvidenceCount = 0.0,
                warnings = listOf("no-usable-world-space-evidence"),
            )
        }
        val geometry = usable.map { it.geometricSupport }.average()
        val temporal = usable.map { it.temporalSupport }.average()
        val learned = usable.map { it.learnedSupport }.average()
        val uncertainty = usable.map { it.uncertainty.confidence }.average()
        val denseCoverage = if (actorCount <= 0) 0.0 else
            (denseTrackCount.toDouble() / max(actorCount,1).toDouble()).coerceIn(0.0,1.0)
        val metricAuthority =
            usable.map {
                minOf(
                    it.geometricSupport,
                    it.uncertainty.confidence,
                    it.viewDiversity,
                    it.temporalSupport,
                ).coerceIn(0.0, 1.0)
            }.average().coerceIn(0.0, 1.0)
        val groupCounts = mapOf(
            "geometry" to usable.map { it.independentCameraCount }.sum().coerceAtLeast(1),
            "temporal" to usable.size,
            "learned" to usable.count { it.learnedSupport > 0.0 },
        )
        val geometryEffective = minOf(groupCounts["geometry"]!!.toDouble(), config.maximumSignalsPerGroup.toDouble())
        val temporalEffective = minOf(groupCounts["temporal"]!!.toDouble(), config.maximumSignalsPerGroup.toDouble()) * (1.0-config.temporalCorrelationPenalty)
        val learnedEffective = minOf(groupCounts["learned"]!!.toDouble(), config.maximumSignalsPerGroup.toDouble()) * (1.0-config.globalCorrelationPenalty)
        val nominalCount = (
            groupCounts["geometry"]!!.toDouble() +
                groupCounts["temporal"]!!.toDouble() +
                groupCounts["learned"]!!.toDouble()
            ).coerceAtLeast(1.0)
        val effectiveCount = (geometryEffective + temporalEffective + learnedEffective).coerceAtLeast(1.0)
        val weighted = (
            geometry * 1.50 +
                temporal * (1.20 * (1.0-config.temporalCorrelationPenalty)) +
                uncertainty * 1.00 +
                learned * config.learnedAuthorityScale * 0.45 * (1.0-config.globalCorrelationPenalty) +
                denseCoverage * 1.00
            ) / 5.15
        val baseAuthority = weighted * 0.72 + geometry * 0.20 + temporal * 0.08
        val metricAdjustment =
            (
                1.0 -
                    config.metricAuthorityInfluence *
                        (1.0 - metricAuthority)
                ).coerceIn(0.65, 1.0)
        val authority = (
            baseAuthority *
                (1.0 - config.identityAmbiguityPenalty * identityAmbiguityRisk) *
                metricAdjustment
            ).coerceIn(0.0,1.0)
        val state = when {
            authority >= 0.82 && geometry >= 0.75 && uncertainty >= 0.70 -> State.STRONG
            authority >= 0.62 && geometry >= 0.55 -> State.GUARDED
            authority >= 0.40 -> State.DEGRADED
            else -> State.REJECTED
        }
        return Result(
            state=state,
            authorityScore=authority,
            geometryAuthority=geometry,
            temporalAuthority=temporal,
            learnedAuthority=learned,
            denseAuthority = denseCoverage,
            metricAuthority = metricAuthority,
            correlationAdjustment =(effectiveCount / nominalCount).coerceIn(0.0,1.0),
            effectiveEvidenceCount=effectiveCount,
            identityAmbiguityRisk=identityAmbiguityRisk,
            warnings=buildList {
                if (learned > geometry) add("learned-evidence-exceeds-geometry-but-is-downweighted")
                if (temporalEffective < temporal * max(1.0,usable.size.toDouble()*0.5)) add("temporal-correlation-reduced-effective-evidence")
                if (denseCoverage < 0.50) add("dense-actor-coverage-low")
                if (identityAmbiguityRisk > 0.0) add("identity-ambiguity-reduces-authority")
            },
        )
    }
}