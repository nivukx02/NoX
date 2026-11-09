package com.nivukx.nox.core.nativeengine

data class NativeSparseTsdfMesh(
    val vertexPositionsNormalsAndUncertainty: FloatArray,
    val triangleIndices: IntArray,
    val vertexCount: Int,
    val triangleCount: Int,
    val confidence: Double,
) {
    init {
        require(vertexCount >= 0)
        require(triangleCount >= 0)
        // Validate products in 64-bit arithmetic before the legacy Int checks below.
        // This prevents wrapped counts from making tiny arrays appear to describe huge meshes.
        require(vertexCount.toLong() * 7L == vertexPositionsNormalsAndUncertainty.size.toLong())
        require(triangleCount.toLong() * 3L == triangleIndices.size.toLong())
        require(vertexPositionsNormalsAndUncertainty.all { it.isFinite() })
        require(vertexPositionsNormalsAndUncertainty.size == vertexCount * 7)
        require(triangleIndices.size == triangleCount * 3)
        require(triangleIndices.all { it in 0 until vertexCount })
        require(confidence in 0.0..1.0)
    }
}
