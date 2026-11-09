package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.DenseMeshQuality
import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.nativeengine.DepthSampleConvention
import com.nivukx.nox.core.nativeengine.GaussianSurfaceObservation
import com.nivukx.nox.core.nativeengine.GaussianSurfaceTelemetry
import com.nivukx.nox.core.nativeengine.NativeSparseTsdfEngine
import kotlin.math.max

class NativeSparseTsdfReconstructionBridge(
    private val voxelSizeMeters: Double = 0.0125,
    private val truncationDistanceMeters: Double = 0.05,
    private val maxTriangles: Int = 200_000,
) : AutoCloseable {

    private val engine = NativeSparseTsdfEngine.create(
        voxelSize = voxelSizeMeters,
        truncationDistance = truncationDistanceMeters,
    )

    fun reset() = engine.reset()

    fun integrateGaussianEvidence(
        observations: List<GaussianSurfaceObservation>,
    ): Long =
        engine.integrateGaussianBatch(
            observations,
        )

    fun gaussianSurfaceTelemetry(): GaussianSurfaceTelemetry =
        engine.gaussianSurfaceTelemetry()

    fun integrateDepthFrame(
        depth: FloatArray,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        rotationW: Double,
        rotationX: Double,
        rotationY: Double,
        rotationZ: Double,
        origin: Vec3 = Vec3(0.0, 0.0, 0.0),
        stride: Int = 2,
        confidence: Double = 1.0,
        depthConvention: DepthSampleConvention = DepthSampleConvention.EUCLIDEAN_RAY,
    ) {
        engine.integrateDepthFrame(
            depth,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            rotationX,
            rotationY,
            rotationZ,
            rotationW,
            origin.x,
            origin.y,
            origin.z,
            stride,
            confidence,
            depthConvention,
        )
    }

    fun integrateDepthFrameAdaptive(
        depthFrame: DepthFrame,
        intrinsics: CameraIntrinsics,
        pose: Pose,
        stride: Int = 1,
        confidence: Double = 1.0,
        depthConvention: DepthSampleConvention = DepthSampleConvention.EUCLIDEAN_RAY,
        evidence: com.nivukx.nox.core.nativeengine.TsdfDetailEvidence,
    ): com.nivukx.nox.core.nativeengine.TsdfDetailDecision? {
        require(depthFrame.depthMeters.isNotEmpty())
        require(depthFrame.width > 0 && depthFrame.height > 0)
        return engine.integrateDepthFrameAdaptive(
            depth = depthFrame.depthMeters,
            width = depthFrame.width,
            height = depthFrame.height,
            depthConfidence = depthFrame.confidence,
            fx = intrinsics.fx,
            fy = intrinsics.fy,
            cx = intrinsics.cx,
            cy = intrinsics.cy,
            qx = pose.rotation.x,
            qy = pose.rotation.y,
            qz = pose.rotation.z,
            qw = pose.rotation.w,
            tx = pose.position.x,
            ty = pose.position.y,
            tz = pose.position.z,
            stride = stride,
            confidence = confidence,
            depthConvention = depthConvention,
            evidence = evidence,
        )
    }

    fun integrate(
        origin: Vec3,
        direction: Vec3,
        depthMeters: Double,
        confidence: Double = 1.0,
    ) {
        require(depthMeters > 0.0)
        val normalized = direction.normalized()
        require(normalized.norm() > 0.0) { "Surface direction must be non-zero" }
        engine.integrateSurfacePoint(
            originX = origin.x,
            originY = origin.y,
            originZ = origin.z,
            directionX = normalized.x,
            directionY = normalized.y,
            directionZ = normalized.z,
            depth = depthMeters,
            confidence = confidence,
        )
    }

    fun integrateDepthFrame(
        depthFrame: DepthFrame,
        intrinsics: CameraIntrinsics,
        pose: Pose,
        stride: Int = 2,
        confidence: Double = 1.0,
        depthConvention: DepthSampleConvention = DepthSampleConvention.EUCLIDEAN_RAY,
    ) {
        require(depthFrame.depthMeters.isNotEmpty())
        require(depthFrame.width > 0 && depthFrame.height > 0)
        engine.integrateDepthFrame(
            depth = depthFrame.depthMeters,
            width = depthFrame.width,
            height = depthFrame.height,
            depthConfidence = depthFrame.confidence,
            fx = intrinsics.fx,
            fy = intrinsics.fy,
            cx = intrinsics.cx,
            cy = intrinsics.cy,
            qx = pose.rotation.x,
            qy = pose.rotation.y,
            qz = pose.rotation.z,
            qw = pose.rotation.w,
            tx = pose.position.x,
            ty = pose.position.y,
            tz = pose.position.z,
            stride = stride,
            confidence = confidence,
            depthConvention = depthConvention,
        )
    }

    fun extract(
        timestampNanos: Long,
        isoLevel: Float = 0.0f,
    ): ReconstructionMeshWithQuality? {
        val native = engine.extractMesh(
            isoLevel = isoLevel,
            maxTriangles = maxTriangles,
        ) ?: return null

        val vertices = ArrayList<MeshVertex>(native.vertexCount)
        val normals = ArrayList<Vec3>(native.vertexCount)
        val uncertainty = ArrayList<Double>(native.vertexCount)
        var cursor = 0

        repeat(native.vertexCount) {
            val x = native.vertexPositionsNormalsAndUncertainty[cursor++].toDouble()
            val y = native.vertexPositionsNormalsAndUncertainty[cursor++].toDouble()
            val z = native.vertexPositionsNormalsAndUncertainty[cursor++].toDouble()
            val nx = native.vertexPositionsNormalsAndUncertainty[cursor++].toDouble()
            val ny = native.vertexPositionsNormalsAndUncertainty[cursor++].toDouble()
            val nz = native.vertexPositionsNormalsAndUncertainty[cursor++].toDouble()
            val uncertaintyMeters =
                native.vertexPositionsNormalsAndUncertainty[cursor++].toDouble()
                    .coerceAtLeast(0.0)

            vertices += MeshVertex(
                position = Vec3(x, y, z),
                uncertaintyMeters = uncertaintyMeters,
            )
            normals += Vec3(nx, ny, nz)
            uncertainty += uncertaintyMeters
        }

        val triangles = ArrayList<MeshTriangle>(native.triangleCount)
        for (triangle in 0 until native.triangleCount) {
            val base = triangle * 3
            triangles += MeshTriangle(
                native.triangleIndices[base],
                native.triangleIndices[base + 1],
                native.triangleIndices[base + 2],
            )
        }

        val normalCoverage =
            normals.count {
                it.norm() > 0.5 &&
                    it.x.isFinite() &&
                    it.y.isFinite() &&
                    it.z.isFinite()
            }.toDouble() / normals.size.coerceAtLeast(1)

        val meanUncertainty =
            uncertainty.average().takeIf { it.isFinite() } ?: 0.0

        val geometricConfidence = (
            native.confidence *
                (0.72 + 0.28 * normalCoverage) *
                (1.0 / (1.0 + meanUncertainty / truncationDistanceMeters.coerceAtLeast(1e-6)))
            ).coerceIn(0.0, 1.0)

        val edgeUse = HashMap<Long, Int>(
            max(native.triangleCount * 2, 16),
        )
        for (triangle in 0 until native.triangleCount) {
            val base = triangle * 3
            val a = native.triangleIndices[base]
            val bIndex = native.triangleIndices[base + 1]
            val c = native.triangleIndices[base + 2]
            val edges = arrayOf(
                edgeKey(a, bIndex),
                edgeKey(bIndex, c),
                edgeKey(c, a),
            )
            edges.forEach { edge ->
                edgeUse[edge] = (edgeUse[edge] ?: 0) + 1
            }
        }
        val manifoldEdges = edgeUse.count { it.value == 2 }
        val watertightness =
            if (edgeUse.isEmpty()) {
                0.0
            } else {
                manifoldEdges.toDouble() / edgeUse.size.toDouble()
            }

        val quality = DenseMeshQuality(
            timestampNanos = timestampNanos,
            vertexCount = native.vertexCount,
            triangleCount = native.triangleCount,
            meanReprojectionErrorPx = 0.0,
            watertightnessScore = watertightness.coerceIn(0.0, 1.0),
            temporalStability = 0.0,
            geometricConfidence = geometricConfidence,
            normalCoverage = normalCoverage.coerceIn(0.0, 1.0),
            meanUncertaintyMeters = meanUncertainty,
            reprojectionEvidenceAvailable = false,
            watertightnessEvidenceAvailable = true,
            temporalEvidenceAvailable = false,
        )

        return ReconstructionMeshWithQuality(
            mesh = ReconstructionMesh(
                vertices = vertices,
                triangles = triangles,
                timestampNanos = timestampNanos,
                confidence = geometricConfidence,
                normals = normals,
                vertexUncertaintyMeters = uncertainty,
            ),
            quality = quality,
            rejectedTriangleCount = native.rejectedTriangleCount(),
        )
    }

    private fun edgeKey(a: Int, b: Int): Long {
        val lo = minOf(a, b).toLong() and 0xFFFF_FFFFL
        val hi = maxOf(a, b).toLong() and 0xFFFF_FFFFL
        return (lo shl 32) or hi
    }

    override fun close() {
        engine.close()
    }

    data class ReconstructionMeshWithQuality(
        val mesh: ReconstructionMesh,
        val quality: DenseMeshQuality,
        val rejectedTriangleCount: Long = 0L,
    )
}
