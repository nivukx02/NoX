package com.nivukx.nox.core.nativeengine

enum class DepthSampleConvention {
    EUCLIDEAN_RAY,
    OPTICAL_AXIS_Z,
}

data class TsdfDetailEvidence(
    val motionEnergy: Double,
    val uncertainty: Double,
    val cameraCoverage: Double,
    val reconstructionConfidence: Double,
    val thermalHeadroom: Double,
    val gpuHeadroom: Double,
    val depthUncertaintyMeters: Double = 0.0,
    val temporalConsistency: Double = 1.0,
    val appearanceConsistency: Double = 1.0,
    val contactConsistency: Double = 1.0,
    val dynamicProbability: Double = 0.0,
) {
    init {
        listOf(
            motionEnergy,
            uncertainty,
            cameraCoverage,
            reconstructionConfidence,
            thermalHeadroom,
            gpuHeadroom,
            temporalConsistency,
            appearanceConsistency,
            contactConsistency,
            dynamicProbability,
        ).forEach { require(it.isFinite() && it in 0.0..1.0) }
        require(
            depthUncertaintyMeters.isFinite() &&
                depthUncertaintyMeters >= 0.0
        )
    }
}

data class TsdfDetailDecision(
    val pixelStride: Int,
    val observationWeightScale: Double,
    val computePressure: Double,
    val detailLevel: Double,
    val preserveFineDetail: Boolean,
    val shedDetail: Boolean,
    val memoryBound: Boolean,
    val reason: String,
)

data class GaussianSurfaceTelemetry(
    val primitiveCount: Long,
    val fusedObservations: Long,
    val rejectedObservations: Long,
    val dynamicPrimitives: Long,
    val meanConfidence: Double,
    val meanUncertaintyMeters: Double,
    val meanDynamicProbability: Double,
    val temporalStability: Double,
    val trustedPrimitiveRatio: Double,
    val nativeTelemetryAvailable: Boolean = true,
    val deformationNodeCount: Long = 0L,
    val deformationBindingCount: Long = 0L,
    val deformationRejectedBindingCount: Long = 0L,
    val deformationMeanNodeConfidence: Double = 1.0,
    val deformationMeanBindingConfidence: Double = 1.0,
    val deformationP95StrainRatio: Double = 0.0,
    val deformationMaximumStrainRatio: Double = 0.0,
    val deformationTemporalStability: Double = 1.0,
    val deformationTrustedBindingRatio: Double = 1.0,
    val rejectedTriangleCount: Long = 0L,
) {
    init {
        require(primitiveCount >= 0L)
        require(fusedObservations >= 0L)
        require(rejectedObservations >= 0L)
        require(dynamicPrimitives >= 0L)
        require(meanConfidence.isFinite() && meanConfidence in 0.0..1.0)
        require(meanUncertaintyMeters.isFinite() && meanUncertaintyMeters >= 0.0)
        require(meanDynamicProbability.isFinite() && meanDynamicProbability in 0.0..1.0)
        require(temporalStability.isFinite() && temporalStability in 0.0..1.0)
        require(trustedPrimitiveRatio.isFinite() && trustedPrimitiveRatio in 0.0..1.0)
        require(rejectedTriangleCount >= 0L)
        require(deformationNodeCount >= 0L)
        require(deformationBindingCount >= 0L)
        require(deformationRejectedBindingCount >= 0L)
        require(
            deformationMeanNodeConfidence.isFinite() &&
                deformationMeanNodeConfidence in 0.0..1.0,
        )
        require(
            deformationMeanBindingConfidence.isFinite() &&
                deformationMeanBindingConfidence in 0.0..1.0,
        )
        require(
            deformationP95StrainRatio.isFinite() &&
                deformationP95StrainRatio >= 0.0,
        )
        require(
            deformationMaximumStrainRatio.isFinite() &&
                deformationMaximumStrainRatio >= 0.0,
        )
        require(
            deformationTemporalStability.isFinite() &&
                deformationTemporalStability in 0.0..1.0,
        )
        require(
            deformationTrustedBindingRatio.isFinite() &&
                deformationTrustedBindingRatio in 0.0..1.0,
        )
    }
}

data class GaussianSurfaceObservation(
    val positionX: Double,
    val positionY: Double,
    val positionZ: Double,
    val normalX: Double,
    val normalY: Double,
    val normalZ: Double,
    val velocityX: Double = 0.0,
    val velocityY: Double = 0.0,
    val velocityZ: Double = 0.0,
    val varianceX: Double = 0.0004,
    val varianceY: Double = 0.0004,
    val varianceZ: Double = 0.0004,
    val confidence: Double = 1.0,
    val dynamicProbability: Double = 0.0,
    val temporalConsistency: Double = 1.0,
    val visibilityConfidence: Double = 1.0,
    val photometricConsistency: Double = 1.0,
    val timestampNanos: Long,
) {
    init {
        listOf(
            positionX, positionY, positionZ,
            normalX, normalY, normalZ,
            velocityX, velocityY, velocityZ,
            varianceX, varianceY, varianceZ,
            confidence, dynamicProbability, temporalConsistency,
            visibilityConfidence, photometricConsistency,
        ).forEach {
            require(it.isFinite())
        }
        require(varianceX > 0.0 && varianceY > 0.0 && varianceZ > 0.0)
        require(confidence in 0.0..1.0)
        require(dynamicProbability in 0.0..1.0)
        require(temporalConsistency in 0.0..1.0)
        require(visibilityConfidence in 0.0..1.0)
        require(photometricConsistency in 0.0..1.0)
        require(timestampNanos > 0L)
    }
}

class NativeSparseTsdfEngine private constructor(
    private var handle: Long,
) : AutoCloseable {

    companion object {
        private const val MAX_GAUSSIAN_BATCH_SIZE = 65_536

        init {
            System.loadLibrary("nox_native_engine")
        }

        fun create(
            voxelSize: Double = 0.0125,
            truncationDistance: Double = 0.05,
            maxWeight: Float = 64.0f,
            blockResolution: Int = 8,
        ): NativeSparseTsdfEngine {
            require(voxelSize > 0.0)
            require(truncationDistance >= voxelSize)
            require(maxWeight > 0.0f)
            require(blockResolution > 0)
            val native = nativeCreate(
                voxelSize,
                truncationDistance,
                maxWeight,
                blockResolution,
            )
            check(native != 0L) { "Native sparse TSDF creation failed" }
            return NativeSparseTsdfEngine(native)
        }

        private external fun nativeCreate(
            voxelSize: Double,
            truncationDistance: Double,
            maxWeight: Float,
            blockResolution: Int,
        ): Long

        private external fun nativeReset(handle: Long)
        private external fun nativeReleaseMemory(handle: Long)

        private external fun nativeIntegrateSurfacePoint(
            handle: Long,
            originX: Double,
            originY: Double,
            originZ: Double,
            directionX: Double,
            directionY: Double,
            directionZ: Double,
            depth: Double,
            confidence: Double,
        )

        private external fun nativeExtractIndexed(
            handle: Long,
            isoLevel: Float,
            maxTriangles: Int,
        ): FloatArray?

        private external fun nativeExtract(
            handle: Long,
            isoLevel: Float,
            maxTriangles: Int,
        ): FloatArray?

        private external fun nativeIntegrateDepthFrame(
            handle: Long,
            depth: FloatArray,
            depthConfidence: FloatArray?,
            width: Int,
            height: Int,
            fx: Double,
            fy: Double,
            cx: Double,
            cy: Double,
            qx: Double,
            qy: Double,
            qz: Double,
            qw: Double,
            tx: Double,
            ty: Double,
            tz: Double,
            stride: Int,
            confidence: Double,
            depthConvention: Int,
        )
        private external fun nativeIntegrateDepthFrameAdaptive(
            handle: Long,
            depth: FloatArray,
            depthConfidence: FloatArray?,
            width: Int,
            height: Int,
            fx: Double,
            fy: Double,
            cx: Double,
            cy: Double,
            qx: Double,
            qy: Double,
            qz: Double,
            qw: Double,
            tx: Double,
            ty: Double,
            tz: Double,
            stride: Int,
            confidence: Double,
            depthConvention: Int,
            motionEnergy: Double,
            uncertainty: Double,
            cameraCoverage: Double,
            reconstructionConfidence: Double,
            thermalHeadroom: Double,
            gpuHeadroom: Double,
        ): FloatArray?
        private external fun nativeVoxelCount(handle: Long): Long
        private external fun nativeResidentMemoryBytes(handle: Long): Long
        private external fun nativeVoxelUtilization(handle: Long): Double
        private external fun nativeRejectedVoxelAllocations(handle: Long): Long
        private external fun nativeRejectedTriangleCount(handle: Long): Long

        private external fun nativeIntegrateGaussianBatch(
            handle: Long,
            packedObservations: DoubleArray,
            timestampsNanos: LongArray,
        ): Long

        private external fun nativeGaussianStats(handle: Long): DoubleArray?
        private external fun nativeGaussianDeformationStats(handle: Long): DoubleArray?

        private external fun nativeDestroy(handle: Long)
    }

    @Synchronized
    fun reset() {
        requireOpen()
        nativeReset(handle)
    }

    @Synchronized
    fun releaseMemory() {
        requireOpen()
        nativeReleaseMemory(handle)
    }

    @Synchronized
    fun integrateSurfacePoint(
        originX: Double,
        originY: Double,
        originZ: Double,
        directionX: Double,
        directionY: Double,
        directionZ: Double,
        depth: Double,
        confidence: Double = 1.0,
    ) {
        requireOpen()
        require(depth > 0.0)
        require(depth.isFinite())
        require(
            originX.isFinite() && originY.isFinite() && originZ.isFinite() &&
                directionX.isFinite() && directionY.isFinite() && directionZ.isFinite(),
        )
        val directionNormSquared = directionX * directionX + directionY * directionY + directionZ * directionZ
        require(directionNormSquared.isFinite() && directionNormSquared > 1e-12)
        require(confidence >= 0.0 && confidence <= 1.0)
        nativeIntegrateSurfacePoint(
            handle,
            originX,
            originY,
            originZ,
            directionX,
            directionY,
            directionZ,
            depth,
            confidence,
        )
    }

    @Synchronized
    fun integrateDepthFrame(
        depth: FloatArray,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        qx: Double,
        qy: Double,
        qz: Double,
        qw: Double,
        tx: Double = 0.0,
        ty: Double = 0.0,
        tz: Double = 0.0,
        stride: Int = 2,
        confidence: Double = 1.0,
        depthConvention: DepthSampleConvention = DepthSampleConvention.EUCLIDEAN_RAY,
        depthConfidence: FloatArray? = null,
    ) {
        requireOpen()
        require(width > 0 && height > 0)
        val sampleCount = width.toLong() * height.toLong()
        require(sampleCount <= Int.MAX_VALUE.toLong())
        require(depth.size.toLong() == sampleCount)
        require(depthConfidence == null || depthConfidence.size.toLong() == sampleCount)
        require(fx > 0.0 && fy > 0.0)
        requireValidCameraGeometry(fx, fy, cx, cy, qx, qy, qz, qw, tx, ty, tz)
        require(confidence in 0.0..1.0)
        nativeIntegrateDepthFrame(
            handle,
            depth,
            depthConfidence,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            qx,
            qy,
            qz,
            qw,
            tx,
            ty,
            tz,
            stride.coerceAtLeast(1),
            confidence,
            depthConvention.ordinal,
        )
    }

    @Synchronized
    fun integrateDepthFrameAdaptive(
        depth: FloatArray,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        qx: Double,
        qy: Double,
        qz: Double,
        qw: Double,
        tx: Double = 0.0,
        ty: Double = 0.0,
        tz: Double = 0.0,
        stride: Int = 1,
        confidence: Double = 1.0,
        depthConvention: DepthSampleConvention = DepthSampleConvention.EUCLIDEAN_RAY,
        evidence: TsdfDetailEvidence,
        depthConfidence: FloatArray? = null,
    ): TsdfDetailDecision? {
        requireOpen()
        require(width > 0 && height > 0)
        val sampleCount = width.toLong() * height.toLong()
        require(sampleCount <= Int.MAX_VALUE.toLong())
        require(depth.size.toLong() == sampleCount)
        require(depthConfidence == null || depthConfidence.size.toLong() == sampleCount)
        require(fx > 0.0 && fy > 0.0)
        requireValidCameraGeometry(fx, fy, cx, cy, qx, qy, qz, qw, tx, ty, tz)
        require(confidence in 0.0..1.0)
        val payload = nativeIntegrateDepthFrameAdaptive(
            handle,
            depth,
            depthConfidence,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            qx,
            qy,
            qz,
            qw,
            tx,
            ty,
            tz,
            stride.coerceAtLeast(1),
            confidence,
            depthConvention.ordinal,
            evidence.motionEnergy,
            evidence.uncertainty,
            evidence.cameraCoverage,
            evidence.reconstructionConfidence,
            evidence.thermalHeadroom,
            evidence.gpuHeadroom,
            evidence.depthUncertaintyMeters,
            evidence.temporalConsistency,
            evidence.appearanceConsistency,
            evidence.contactConsistency,
            evidence.dynamicProbability,
        ) ?: return null
        if (payload.size != 8) return null
        return TsdfDetailDecision(
            pixelStride = payload[0].toInt().coerceAtLeast(1),
            observationWeightScale = payload[1].toDouble().coerceIn(0.0, 1.0),
            computePressure = payload[2].toDouble().coerceIn(0.0, 1.0),
            detailLevel = payload[3].toDouble().coerceIn(0.0, 1.0),
            preserveFineDetail = payload[4] >= 0.5f,
            shedDetail = payload[5] >= 0.5f,
            memoryBound = payload[6] >= 0.5f,
            reason = if (payload[7] >= 0.5f) "adaptive" else "unknown",
        )
    }

    @Synchronized
    fun voxelCount(): Long {
        requireOpen()
        return nativeVoxelCount(handle)
    }

    @Synchronized
    fun residentMemoryBytes(): Long {
        requireOpen()
        return nativeResidentMemoryBytes(handle)
    }

    @Synchronized
    fun voxelUtilization(): Double {
        requireOpen()
        return nativeVoxelUtilization(handle).coerceIn(0.0, 1.0)
    }

    @Synchronized
    fun rejectedVoxelAllocations(): Long {
        requireOpen()
        return nativeRejectedVoxelAllocations(handle)
    }

    @Synchronized
    fun rejectedTriangleCount(): Long {
        requireOpen()
        return nativeRejectedTriangleCount(handle)
    }

    @Synchronized
    fun integrateGaussianBatch(
        observations: List<GaussianSurfaceObservation>,
    ): Long {
        requireOpen()
        if (observations.isEmpty()) return 0L

        val stride = 17
        var offset = 0
        var fusedCount = 0L
        while (offset < observations.size) {
            val batchCount =
                minOf(MAX_GAUSSIAN_BATCH_SIZE, observations.size - offset)
            // Keep temporary JNI arrays bounded even for long recordings and dense reconstructions.
            val packed = DoubleArray(batchCount * stride)
            val timestamps = LongArray(batchCount)

            for (localIndex in 0 until batchCount) {
                val observation = observations[offset + localIndex]
                val base = localIndex * stride
                packed[base] = observation.positionX
                packed[base + 1] = observation.positionY
                packed[base + 2] = observation.positionZ
                packed[base + 3] = observation.normalX
                packed[base + 4] = observation.normalY
                packed[base + 5] = observation.normalZ
                packed[base + 6] = observation.velocityX
                packed[base + 7] = observation.velocityY
                packed[base + 8] = observation.velocityZ
                packed[base + 9] = observation.varianceX
                packed[base + 10] = observation.varianceY
                packed[base + 11] = observation.varianceZ
                packed[base + 12] = observation.confidence
                packed[base + 13] = observation.dynamicProbability
                packed[base + 14] = observation.temporalConsistency
                packed[base + 15] = observation.visibilityConfidence
                packed[base + 16] = observation.photometricConsistency
                timestamps[localIndex] = observation.timestampNanos
            }

            val integrated = nativeIntegrateGaussianBatch(handle, packed, timestamps)
            check(integrated >= 0L) { "Native Gaussian integration returned a negative count" }
            fusedCount =
                if (Long.MAX_VALUE - fusedCount < integrated) {
                    Long.MAX_VALUE
                } else {
                    fusedCount + integrated
                }
            offset += batchCount
        }
        return fusedCount
    }

    @Synchronized
    fun gaussianSurfaceTelemetry(): GaussianSurfaceTelemetry {
        requireOpen()

        val values =
            nativeGaussianStats(handle)
        val deformation =
            nativeGaussianDeformationStats(handle)

        if (
            values == null ||
            deformation == null ||
            values.size != 9 ||
            deformation.size != 9 ||
            values.any { !it.isFinite() } ||
            deformation.any { !it.isFinite() }
        ) {
            return GaussianSurfaceTelemetry(
                primitiveCount = 0L,
                fusedObservations = 0L,
                rejectedObservations = 0L,
                dynamicPrimitives = 0L,
                meanConfidence = 0.0,
                meanUncertaintyMeters = 0.0,
                meanDynamicProbability = 1.0,
                temporalStability = 0.0,
                trustedPrimitiveRatio = 0.0,
                rejectedTriangleCount = 0L,
                nativeTelemetryAvailable = false,
                deformationNodeCount = 0L,
                deformationBindingCount = 0L,
                deformationRejectedBindingCount = 0L,
                deformationMeanNodeConfidence = 0.0,
                deformationMeanBindingConfidence = 0.0,
                deformationP95StrainRatio = 0.0,
                deformationMaximumStrainRatio = 0.0,
                deformationTemporalStability = 0.0,
                deformationTrustedBindingRatio = 0.0,
            )
        }

        return GaussianSurfaceTelemetry(
            primitiveCount = values[0].coerceAtLeast(0.0).toLong(),
            fusedObservations = values[1].coerceAtLeast(0.0).toLong(),
            rejectedObservations = values[2].coerceAtLeast(0.0).toLong(),
            dynamicPrimitives = values[3].coerceAtLeast(0.0).toLong(),
            meanConfidence = values[4].coerceIn(0.0, 1.0),
            meanUncertaintyMeters = values[5].coerceAtLeast(0.0),
            meanDynamicProbability = values[6].coerceIn(0.0, 1.0),
            temporalStability = values[7].coerceIn(0.0, 1.0),
            trustedPrimitiveRatio = values[8].coerceIn(0.0, 1.0),
            deformationNodeCount =
                deformation[0].coerceAtLeast(0.0).toLong(),
            deformationBindingCount =
                deformation[1].coerceAtLeast(0.0).toLong(),
            deformationRejectedBindingCount =
                deformation[2].coerceAtLeast(0.0).toLong(),
            deformationMeanNodeConfidence =
                deformation[3].coerceIn(0.0, 1.0),
            deformationMeanBindingConfidence =
                deformation[4].coerceIn(0.0, 1.0),
            deformationP95StrainRatio =
                deformation[5].coerceAtLeast(0.0),
            deformationMaximumStrainRatio =
                deformation[6].coerceAtLeast(0.0),
            deformationTemporalStability =
                deformation[7].coerceIn(0.0, 1.0),
            deformationTrustedBindingRatio =
                deformation[8].coerceIn(0.0, 1.0),
        )
    }

    @Synchronized
    fun extractMesh(
        isoLevel: Float = 0.0f,
        maxTriangles: Int = 200_000,
    ): NativeSparseTsdfMesh? {
        requireOpen()
        require(maxTriangles in 1..650_000)
        val payload = nativeExtractIndexed(
            handle,
            isoLevel,
            maxTriangles,
        ) ?: return null

        if (payload.size < 4) return null

        if (!payload[0].isFinite() || payload[0] != 2.0f) return null
        val version = payload[0].toInt()
        if (version != 2) return null

        val rawVertexCount = payload[1]
        val rawTriangleCount = payload[2]
        val rawConfidence = payload[3].toDouble()
        if (
            !rawVertexCount.isFinite() ||
            !rawTriangleCount.isFinite() ||
            rawVertexCount < 1.0f ||
            rawTriangleCount < 1.0f ||
            rawVertexCount.toDouble() > (payload.size - 4).coerceAtLeast(0) / 7.0 ||
            rawTriangleCount.toDouble() > (payload.size - 4).coerceAtLeast(0) / 3.0 ||
            rawVertexCount != rawVertexCount.toInt().toFloat() ||
            rawTriangleCount != rawTriangleCount.toInt().toFloat() ||
            !rawConfidence.isFinite() ||
            rawConfidence !in 0.0..1.0
        ) {
            return null
        }

        val vertexCount = rawVertexCount.toInt()
        val triangleCount = rawTriangleCount.toInt()
        if (vertexCount <= 0 || triangleCount <= 0) return null

        val vertexFloatCountLong = vertexCount.toLong() * 7L
        val indexFloatCountLong = triangleCount.toLong() * 3L
        val expectedSizeLong = 4L + vertexFloatCountLong + indexFloatCountLong
        if (
            vertexFloatCountLong > Int.MAX_VALUE.toLong() ||
            indexFloatCountLong > Int.MAX_VALUE.toLong() ||
            expectedSizeLong != payload.size.toLong()
        ) {
            return null
        }

        val confidence = rawConfidence
        val vertexFloatCount = vertexFloatCountLong.toInt()
        val indexFloatCount = indexFloatCountLong.toInt()

        val packed = payload.copyOfRange(
            4,
            4 + vertexFloatCount,
        )
        if (packed.any { !it.isFinite() }) return null
        val indices = IntArray(indexFloatCount)
        var indexCursor = 4 + vertexFloatCount
        for (i in indices.indices) {
            val raw = payload[indexCursor++]
            val decoded = raw.toInt()
            if (raw != decoded.toFloat()) return null
            indices[i] = decoded
            if (decoded !in 0 until vertexCount) return null
        }

        return NativeSparseTsdfMesh(
            vertexPositionsNormalsAndUncertainty = packed,
            triangleIndices = indices,
            vertexCount = vertexCount,
            triangleCount = triangleCount,
            confidence = confidence,
        )
    }

    private fun requireValidCameraGeometry(
        fx: Double, fy: Double, cx: Double, cy: Double,
        qx: Double, qy: Double, qz: Double, qw: Double,
        tx: Double, ty: Double, tz: Double,
    ) {
        require(fx.isFinite() && fx > 0.0 && fy.isFinite() && fy > 0.0)
        require(cx.isFinite() && cy.isFinite())
        require(qx.isFinite() && qy.isFinite() && qz.isFinite() && qw.isFinite())
        require(tx.isFinite() && ty.isFinite() && tz.isFinite())
        val normSquared = qx * qx + qy * qy + qz * qz + qw * qw
        require(normSquared.isFinite() && normSquared > 1e-12)
    }

    private fun requireOpen() {
        check(handle != 0L) { "Native sparse TSDF engine is closed" }
    }

    @Synchronized
    override fun close() {
        val closingHandle = handle
        if (closingHandle == 0L) return

        // Mark the wrapper closed before JNI so a failed destroy cannot make a second close
        // attempt destroy the same native pointer again.
        handle = 0L
        nativeDestroy(closingHandle)
    }
}
