package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3

/**
 * Shared reconstruction math helpers.
 *
 * These utilities are intentionally package-visible so independent reconstruction engines can
 * validate vectors and compute deterministic robust statistics without depending on another
 * source file's private helpers.
 */
internal fun Vec3.isFinite(): Boolean =
    x.isFinite() && y.isFinite() && z.isFinite()

internal fun median(
    values: List<Double>,
    fallback: Double,
): Double {
    val finite = values.filter { it.isFinite() }.sorted()
    if (finite.isEmpty()) return fallback
    if (finite.size == 1) return finite.first()

    val middle = finite.size / 2
    return if (finite.size % 2 == 0) {
        (finite[middle - 1] + finite[middle]) * 0.5
    } else {
        finite[middle]
    }
}
