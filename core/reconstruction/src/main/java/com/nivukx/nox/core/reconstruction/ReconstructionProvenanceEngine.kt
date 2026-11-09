package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.PrecisionAuthoritySnapshot
import com.nivukx.nox.core.model.ReconstructionRegionProvenance
import java.security.MessageDigest

/** Aggregates provenance without allowing inferred geometry to masquerade as measurement. */
class ReconstructionProvenanceEngine(
    private val minimumEmpiricalSamples: Int = 600,
    private val minimumIndependentViews: Int = 3,
) {
    data class RegionInput(
        val actorId: Int,
        val regionId: Int,
        val counts: Map<EvidenceOrigin, Int>,
        val confidence: Double,
    )

    data class Result(
        val regions: List<ReconstructionRegionProvenance>,
        val observedRatio: Double,
        val measuredRatio: Double,
        val inferredRatio: Double,
        val generationRatio: Double,
        val fingerprint: String,
    )

    fun aggregate(inputs: List<RegionInput>): Result {
        require(inputs.all { it.confidence in 0.0..1.0 && it.counts.values.all { value -> value >= 0 } })
        val regions = inputs.map { input ->
            val total = input.counts.values.sum().coerceAtLeast(1)
            fun ratio(origin: EvidenceOrigin) = input.counts.getOrDefault(origin, 0).toDouble() / total
            ReconstructionRegionProvenance(
                input.actorId, input.regionId,
                ratio(EvidenceOrigin.OBSERVED),
                ratio(EvidenceOrigin.GEOMETRICALLY_CONSTRAINED),
                ratio(EvidenceOrigin.TEMPORALLY_PROPAGATED),
                ratio(EvidenceOrigin.LEARNED_PRIOR),
                ratio(EvidenceOrigin.GENERATIVE_COMPLETION),
                input.confidence,
            )
        }
        val total = regions.size.coerceAtLeast(1)
        val observed = regions.sumOf { it.observedRatio } / total
        val measured = regions.sumOf { it.measuredRatio } / total
        val inferred = regions.sumOf { it.inferredRatio } / total
        val generated = regions.sumOf { it.generativeCompletionRatio } / total
        return Result(regions, observed, measured, inferred, generated, fingerprint(regions))
    }

    fun qualifies(snapshot: PrecisionAuthoritySnapshot): Boolean =
        snapshot.matchedReferenceQualified &&
            snapshot.empiricalSampleCount >= minimumEmpiricalSamples &&
            snapshot.independentViewCount >= minimumIndependentViews

    private fun fingerprint(regions: List<ReconstructionRegionProvenance>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        regions.sortedWith(compareBy<ReconstructionRegionProvenance> { it.actorId }.thenBy { it.regionId }).forEach {
            digest.update("${it.actorId}:${it.regionId}:${it.observedRatio}:${it.measuredRatio}:${it.inferredRatio}:${it.confidence};".toByteArray())
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}