package com.nivukx.nox.core.model

/**
 * Stable, detector-neutral dense facial surface frame.
 *
 * Coordinates are in the source image coordinate system. Z is explicitly not guaranteed to be
 * metric and must not be interpreted as calibrated camera depth without a separate geometric solve.
 * The point index is stable across frames for the bundled ML Kit Face Mesh detector.
 */
data class FaceMeshFrame(
    val timestampNanos: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val rotationDegrees: Int,
    val trackingId: Int? = null,
    val boundsLeft: Float,
    val boundsTop: Float,
    val boundsRight: Float,
    val boundsBottom: Float,
    val confidence: Double,
    val pointCoverage: Double,
    val triangleCoverage: Double,
    val points: List<FaceMeshPoint3D>,
    val triangles: List<FaceMeshTriangle>,
) {
    init {
        require(imageWidth > 0)
        require(imageHeight > 0)
        require(rotationDegrees in 0..359)
        require(boundsRight >= boundsLeft)
        require(boundsBottom >= boundsTop)
        require(confidence in 0.0..1.0)
        require(pointCoverage in 0.0..1.0)
        require(triangleCoverage in 0.0..1.0)

        val pointIds = HashSet<Int>(points.size)
        points.forEach { point ->
            require(point.index in 0..467)
            require(pointIds.add(point.index)) {
                "Duplicate face mesh point index=" + point.index
            }
            require(point.position.x.isFinite())
            require(point.position.y.isFinite())
            require(point.position.z.isFinite())
        }

        triangles.forEach { triangle ->
            require(triangle.a in pointIds)
            require(triangle.b in pointIds)
            require(triangle.c in pointIds)
            require(triangle.a != triangle.b)
            require(triangle.a != triangle.c)
            require(triangle.b != triangle.c)
        }
    }

    val pointCount: Int
        get() = points.size

    val triangleCount: Int
        get() = triangles.size

    val faceAreaRatio: Double
        get() = (
            (boundsRight - boundsLeft).coerceAtLeast(0f).toDouble() *
                (boundsBottom - boundsTop).coerceAtLeast(0f).toDouble()
        ) / (
            imageWidth.coerceAtLeast(1).toDouble() *
                imageHeight.coerceAtLeast(1).toDouble()
        ).coerceAtLeast(1.0)

    val isDense: Boolean
        get() =
            pointCount >= 400 &&
                triangleCount >= 500 &&
                pointCoverage >= 0.80 &&
                triangleCoverage >= 0.70

    fun pointMap(): Map<Int, Vec3> =
        points.associate { it.index to it.position }
}

data class FaceMeshPoint3D(
    val index: Int,
    val position: Vec3,
)

data class FaceMeshTriangle(
    val a: Int,
    val b: Int,
    val c: Int,
)
