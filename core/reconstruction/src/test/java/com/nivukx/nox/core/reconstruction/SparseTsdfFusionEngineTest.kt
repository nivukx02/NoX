package com.nivukx.nox.core.reconstruction

import java.util.LinkedHashMap
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SparseTsdfFusionEngineTest {

    @Test
    fun meshExtractionReadsNeighborVoxelsWithoutMutatingTheActiveIteration() {
        val engine =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.10,
                truncationDistanceMeters = 0.20,
                maxVoxels = 256,
                temporalHalfLifeSeconds = 100.0,
                minimumRetainedWeight = 0.001,
                decayBudgetVoxels = 256,
                extractionCellBudget = 128,
            )

        val voxelField =
            SparseTsdfFusionEngine::class.java
                .getDeclaredField("voxels")
                .apply { isAccessible = true }

        @Suppress("UNCHECKED_CAST")
        val voxels =
            voxelField.get(engine) as LinkedHashMap<Long, Any>

        val voxelType =
            Class.forName(
                "com.nivukx.nox.core.reconstruction." +
                    "SparseTsdfFusionEngine" +
                    "$" +
                    "Voxel",
            )

        val voxelConstructor =
            voxelType.declaredConstructors
                .single()
                .apply { isAccessible = true }

        val packKey =
            SparseTsdfFusionEngine::class.java
                .getDeclaredMethod(
                    "packKey",
                    Int::class.javaPrimitiveType,
                    Int::class.javaPrimitiveType,
                    Int::class.javaPrimitiveType,
                )
                .apply { isAccessible = true }

        for (z in 0..2) {
            for (y in 0..2) {
                for (x in 0..2) {
                    val sdf = x.toDouble() - 1.5
                    val key =
                        packKey.invoke(
                            engine,
                            x,
                            y,
                            z,
                        ) as Long

                    voxels[key] =
                        voxelConstructor.newInstance(
                            sdf,
                            8.0,
                            1L,
                            4L,
                            0.0,
                        )
                }
            }
        }

        val result =
            engine.extractMesh(
                timestampNanos = 1L,
                isoBand = 0.25,
                cellBudget = 128,
                maxVertices = 1024,
            )

        assertNotNull(result)
    }    
    @Test
    fun conflictingSignedDistanceEvidenceAccumulatesVoxelUncertainty() {
        val engine =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.10,
                truncationDistanceMeters = 0.20,
                maxVoxels = 16,
                temporalHalfLifeSeconds = 100.0,
            )

        val integrateVoxel =
            SparseTsdfFusionEngine::class.java
                .getDeclaredMethod(
                    "integrateVoxel",
                    com.nivukx.nox.core.model.Vec3::class.java,
                    Double::class.javaPrimitiveType,
                    Double::class.javaPrimitiveType,
                    Long::class.javaPrimitiveType,
                )
                .apply { isAccessible = true }

        val voxelField =
            SparseTsdfFusionEngine::class.java
                .getDeclaredField("voxels")
                .apply { isAccessible = true }

        @Suppress("UNCHECKED_CAST")
        val voxels =
            voxelField.get(engine)
                as LinkedHashMap<Long, Any>

        val voxelType =
            Class.forName(
                "com.nivukx.nox.core.reconstruction." +
                    "SparseTsdfFusionEngine" +
                    "$" +
                    "Voxel",
            )

        val varianceField =
            voxelType
                .getDeclaredField("sdfVariance")
                .apply { isAccessible = true }

        val world =
            com.nivukx.nox.core.model.Vec3(0.0, 0.0, 1.0)

        integrateVoxel.invoke(engine, world, -0.8, 1.0, 1L)
        integrateVoxel.invoke(engine, world, 0.8, 1.0, 2L)

        val storedVoxel = voxels.values.single()
        val variance = varianceField.getDouble(storedVoxel)

        org.junit.Assert.assertTrue(variance > 0.1)
    }

    @Test
    fun uncertaintyMetricsExposeConflictingSurfaceEvidence() {
        val engine =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.10,
                truncationDistanceMeters = 0.20,
                maxVoxels = 16,
                temporalHalfLifeSeconds = 100.0,
            )

        val integrateVoxel =
            SparseTsdfFusionEngine::class.java
                .getDeclaredMethod(
                    "integrateVoxel",
                    com.nivukx.nox.core.model.Vec3::class.java,
                    Double::class.javaPrimitiveType,
                    Double::class.javaPrimitiveType,
                    Long::class.javaPrimitiveType,
                )
                .apply { isAccessible = true }

        val world =
            com.nivukx.nox.core.model.Vec3(0.0, 0.0, 1.0)

        integrateVoxel.invoke(engine, world, -0.9, 1.0, 1L)
        integrateVoxel.invoke(engine, world, 0.9, 1.0, 2L)

        val metrics = engine.uncertaintyMetrics()

        org.junit.Assert.assertTrue(
            metrics.meanSignedDistanceStdDev > 0.4,
        )
        org.junit.Assert.assertTrue(
            metrics.highUncertaintyVoxelRatio > 0.0,
        )
        org.junit.Assert.assertTrue(
            metrics.confidenceFactor < 0.9,
        )
    }

    @Test
    fun evidenceWeightedFusionDownweightsUncertainSamples() {
        val lowUncertainty =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.02,
                truncationDistanceMeters = 0.04,
                maxVoxels = 10_000,
                temporalHalfLifeSeconds = 100.0,
            )
        val highUncertainty =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.02,
                truncationDistanceMeters = 0.04,
                maxVoxels = 10_000,
                temporalHalfLifeSeconds = 100.0,
            )

        val intrinsics =
            com.nivukx.nox.core.model.CameraIntrinsics(
                fx = 500.0,
                fy = 500.0,
                cx = 0.0,
                cy = 0.0,
                imageWidth = 1,
                imageHeight = 1,
            )
        val depth = floatArrayOf(1.0f)

        val clean =
            lowUncertainty.integrateDepthEvidenceWeighted(
                depthMeters = depth,
                width = 1,
                height = 1,
                intrinsics = intrinsics,
                cameraPose = com.nivukx.nox.core.model.Pose(),
                timestampNanos = 1L,
                evidence =
                    SparseTsdfFusionEngine.PerSampleEvidence(
                        depthUncertaintyMeters =
                            floatArrayOf(0.003f),
                        staticConfidence =
                            floatArrayOf(0.98f),
                        temporalConfidence =
                            floatArrayOf(0.96f),
                        photometricConfidence =
                            floatArrayOf(0.94f),
                    ),
            )

        val noisy =
            highUncertainty.integrateDepthEvidenceWeighted(
                depthMeters = depth,
                width = 1,
                height = 1,
                intrinsics = intrinsics,
                cameraPose = com.nivukx.nox.core.model.Pose(),
                timestampNanos = 1L,
                evidence =
                    SparseTsdfFusionEngine.PerSampleEvidence(
                        depthUncertaintyMeters =
                            floatArrayOf(0.35f),
                        staticConfidence =
                            floatArrayOf(0.25f),
                        temporalConfidence =
                            floatArrayOf(0.30f),
                        photometricConfidence =
                            floatArrayOf(0.28f),
                    ),
            )

        assertTrue(clean.acceptedSamples > 0)
        assertTrue(noisy.acceptedSamples > 0)
        assertTrue(
            clean.meanEvidenceWeight >
                noisy.meanEvidenceWeight,
        )
        assertTrue(
            noisy.rejectedSamples >= 0,
        )
    }

    @Test
    fun surfaceNormalObservabilityDownweightsSteepDepthGradients() {
        fun engine() =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.02,
                truncationDistanceMeters = 0.04,
                maxVoxels = 10_000,
                temporalHalfLifeSeconds = 100.0,
            )

        val intrinsics =
            com.nivukx.nox.core.model.CameraIntrinsics(
                fx = 500.0,
                fy = 500.0,
                cx = 1.0,
                cy = 1.0,
                imageWidth = 3,
                imageHeight = 3,
            )

        val flatDepth =
            FloatArray(9) { 1.0f }

        val steepDepth =
            FloatArray(9) { index ->
                val x = index % 3
                (1.0 + x * 0.60).toFloat()
            }

        val evidence =
            SparseTsdfFusionEngine.PerSampleEvidence(
                depthUncertaintyMeters =
                    FloatArray(9) { 0.004f },
                staticConfidence =
                    FloatArray(9) { 0.98f },
                temporalConfidence =
                    FloatArray(9) { 0.96f },
                photometricConfidence =
                    FloatArray(9) { 0.94f },
            )

        val flat =
            engine().integrateDepthEvidenceWeighted(
                depthMeters = flatDepth,
                width = 3,
                height = 3,
                intrinsics = intrinsics,
                cameraPose = com.nivukx.nox.core.model.Pose(),
                timestampNanos = 1L,
                stride = 1,
                evidence = evidence,
            )

        val steep =
            engine().integrateDepthEvidenceWeighted(
                depthMeters = steepDepth,
                width = 3,
                height = 3,
                intrinsics = intrinsics,
                cameraPose = com.nivukx.nox.core.model.Pose(),
                timestampNanos = 1L,
                stride = 1,
                evidence = evidence,
            )

        assertTrue(
            flat.meanSurfaceNormalWeight >
                steep.meanSurfaceNormalWeight,
        )
        assertTrue(flat.meanSurfaceNormalWeight in 0.0..1.0)
        assertTrue(steep.meanSurfaceNormalWeight in 0.0..1.0)
    }

    @Test
    fun evidenceWeightedFusionRejectsPersistentlyLowStaticSamples() {
        val engine =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.02,
                truncationDistanceMeters = 0.04,
                maxVoxels = 10_000,
                temporalHalfLifeSeconds = 100.0,
            )

        val result =
            engine.integrateDepthEvidenceWeighted(
                depthMeters = floatArrayOf(1.0f),
                width = 1,
                height = 1,
                intrinsics =
                    com.nivukx.nox.core.model.CameraIntrinsics(
                        fx = 500.0,
                        fy = 500.0,
                        cx = 0.0,
                        cy = 0.0,
                        imageWidth = 1,
                        imageHeight = 1,
                    ),
                cameraPose =
                    com.nivukx.nox.core.model.Pose(),
                timestampNanos = 2L,
                evidence =
                    SparseTsdfFusionEngine.PerSampleEvidence(
                        depthUncertaintyMeters =
                            floatArrayOf(0.20f),
                        staticConfidence =
                            floatArrayOf(0.01f),
                        temporalConfidence =
                            floatArrayOf(0.10f),
                        photometricConfidence =
                            floatArrayOf(0.12f),
                    ),
            )

        assertTrue(
            result.rejectedSamples > 0,
        )
        assertTrue(
            result.acceptedSamples == 0 ||
                result.meanEvidenceWeight < 0.10,
        )
    }

    @Test
    fun saturatedVoxelWeightStillUsesTrueCombinedEvidenceForSdfBlending() {
        val engine =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.10,
                truncationDistanceMeters = 0.20,
                maxVoxels = 16,
                temporalHalfLifeSeconds = 100.0,
            )

        val integrateVoxel =
            SparseTsdfFusionEngine::class.java
                .getDeclaredMethod(
                    "integrateVoxel",
                    com.nivukx.nox.core.model.Vec3::class.java,
                    Double::class.javaPrimitiveType,
                    Double::class.javaPrimitiveType,
                    Long::class.javaPrimitiveType,
                )
                .apply { isAccessible = true }

        val voxelField =
            SparseTsdfFusionEngine::class.java
                .getDeclaredField("voxels")
                .apply { isAccessible = true }

        @Suppress("UNCHECKED_CAST")
        val voxels =
            voxelField.get(engine)
                as LinkedHashMap<Long, Any>

        val sdfField =
            voxels.javaClass
                .let {
                    Class.forName(
                        "com.nivukx.nox.core.reconstruction." +
                            "SparseTsdfFusionEngine" +
                            "$" +
                            "Voxel",
                    )
                }
                .getDeclaredField("sdf")
                .apply { isAccessible = true }

        val world =
            com.nivukx.nox.core.model.Vec3(0.0, 0.0, 1.0)

        integrateVoxel.invoke(
            engine,
            world,
            -0.0,
            32.0,
            1L,
        )
        integrateVoxel.invoke(
            engine,
            world,
            1.0,
            32.0,
            1L,
        )
        integrateVoxel.invoke(
            engine,
            world,
            1.0,
            32.0,
            1L,
        )

        val storedVoxel =
            voxels.values.single()
        val blendedSdf =
            sdfField.getDouble(storedVoxel)

        // 64 units of prior evidence at 0.5 should receive 32 units at 1.0:
        // (64 * 0.5 + 32 * 1.0) / 96 = 2.0 / 3.0.
        org.junit.Assert.assertEquals(
            2.0 / 3.0,
            blendedSdf,
            1e-9,
        )
    }


    @Test
    fun stale_frames_are_rejected_without_rewinding_runtime_state() {
        val engine = SparseTsdfFusionEngine(
            voxelSizeMeters = 0.02,
            truncationDistanceMeters = 0.04,
            maxVoxels = 128,
            temporalHalfLifeSeconds = 10.0,
        )
        val intrinsics = com.nivukx.nox.core.model.CameraIntrinsics(
            fx = 500.0,
            fy = 500.0,
            cx = 0.0,
            cy = 0.0,
            imageWidth = 1,
            imageHeight = 1,
        )

        engine.integrateDepth(
            depthMeters = floatArrayOf(1.0f),
            width = 1,
            height = 1,
            intrinsics = intrinsics,
            cameraPose = com.nivukx.nox.core.model.Pose(),
            timestampNanos = 2_000_000_000L,
        )
        val before = engine.runtimeStats()

        engine.integrateDepth(
            depthMeters = floatArrayOf(0.5f),
            width = 1,
            height = 1,
            intrinsics = intrinsics,
            cameraPose = com.nivukx.nox.core.model.Pose(),
            timestampNanos = 1_000_000_000L,
        )
        val after = engine.runtimeStats()

        assertTrue(after.outOfOrderFrameCount >= before.outOfOrderFrameCount + 1L)
        assertTrue(
            after.lastTimestampNanos ==
                before.lastTimestampNanos,
        )
        assertTrue(after.voxelCount >= 0)
        assertTrue(after.capacityUtilization in 0.0..1.0)
    }

    @Test
    fun adaptive_path_reports_stale_frame_without_mutating_volume() {
        val engine = SparseTsdfFusionEngine(
            voxelSizeMeters = 0.02,
            truncationDistanceMeters = 0.04,
            maxVoxels = 64,
            temporalHalfLifeSeconds = 10.0,
        )
        val intrinsics = com.nivukx.nox.core.model.CameraIntrinsics(
            fx = 500.0,
            fy = 500.0,
            cx = 0.0,
            cy = 0.0,
            imageWidth = 1,
            imageHeight = 1,
        )
        val first = engine.integrateDepthAdaptive(
            depthMeters = floatArrayOf(1.0f),
            width = 1,
            height = 1,
            intrinsics = intrinsics,
            cameraPose = com.nivukx.nox.core.model.Pose(),
            timestampNanos = 5L,
        )
        val second = engine.integrateDepthAdaptive(
            depthMeters = floatArrayOf(0.4f),
            width = 1,
            height = 1,
            intrinsics = intrinsics,
            cameraPose = com.nivukx.nox.core.model.Pose(),
            timestampNanos = 4L,
        )

        assertTrue(first.reason != "stale_frame_rejected")
        assertTrue(second.reason == "stale_frame_rejected")
        assertTrue(second.rejectedSamples > 0)
    }

    @Test
    fun temporal_surface_detail_cache_preserves_normals_and_motion() {
        val engine =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.02,
                truncationDistanceMeters = 0.04,
                maxVoxels = 10_000,
                maxSurfaceSurfels = 128,
                surfaceSurfelHalfLifeSeconds = 10.0,
                surfaceSurfelMinWeight = 0.001,
            )

        val intrinsics =
            com.nivukx.nox.core.model.CameraIntrinsics(
                fx = 500.0,
                fy = 500.0,
                cx = 1.0,
                cy = 1.0,
                imageWidth = 3,
                imageHeight = 3,
            )

        val firstDepth =
            FloatArray(9) { 1.0f }

        val secondDepth =
            FloatArray(9) { 1.0f }.also {
                it[4] = 1.01f
            }

        engine.integrateDepth(
            depthMeters = firstDepth,
            width = 3,
            height = 3,
            intrinsics = intrinsics,
            cameraPose = com.nivukx.nox.core.model.Pose(),
            timestampNanos = 1_000_000_000L,
            stride = 1,
        )

        val before =
            engine.surfaceDetailMetrics(
                timestampNanos = 1_000_000_000L,
            )

        engine.integrateDepth(
            depthMeters = secondDepth,
            width = 3,
            height = 3,
            intrinsics = intrinsics,
            cameraPose =
                com.nivukx.nox.core.model.Pose(
                    position =
                        com.nivukx.nox.core.model.Vec3(
                            0.002,
                            0.0,
                            0.0,
                        ),
                ),
            timestampNanos = 1_050_000_000L,
            stride = 1,
        )

        val after =
            engine.surfaceDetailMetrics(
                timestampNanos = 1_050_000_000L,
            )
        val detail =
            engine.extractSurfaceDetail(
                timestampNanos = 1_050_000_000L,
                maxPoints = 64,
            )

        assertTrue(before.surfelCount > 0)
        assertTrue(after.surfelCount > 0)
        assertTrue(after.meanConfidence > 0.0)
        assertTrue(after.meanNormalStability > 0.0)
        assertTrue(after.memoryUtilization <= 1.0)
        assertTrue(detail.isNotEmpty())
        assertTrue(detail.all { it.finite })
        assertTrue(
            detail.any {
                it.observationCount >= 2L
            },
        )
    }

    @Test
    fun temporal_surface_detail_cache_remains_bounded_under_high_sample_pressure() {
        val engine =
            SparseTsdfFusionEngine(
                voxelSizeMeters = 0.01,
                truncationDistanceMeters = 0.025,
                maxVoxels = 256,
                maxSurfaceSurfels = 32,
                surfaceSurfelHalfLifeSeconds = 10.0,
                surfaceSurfelMinWeight = 0.001,
            )

        val intrinsics =
            com.nivukx.nox.core.model.CameraIntrinsics(
                fx = 700.0,
                fy = 700.0,
                cx = 8.0,
                cy = 8.0,
                imageWidth = 16,
                imageHeight = 16,
            )

        repeat(12) { frame ->
            val depth =
                FloatArray(16 * 16) {
                    (1.0 + (frame % 3) * 0.002).toFloat()
                }

            engine.integrateDepth(
                depthMeters = depth,
                width = 16,
                height = 16,
                intrinsics = intrinsics,
                cameraPose = com.nivukx.nox.core.model.Pose(),
                timestampNanos =
                    1_000_000_000L +
                        frame * 20_000_000L,
                stride = 1,
            )
        }

        val metrics =
            engine.surfaceDetailMetrics(
                timestampNanos = 1_300_000_000L,
            )
        val stats = engine.runtimeStats()

        assertTrue(metrics.surfelCount <= 32)
        assertTrue(stats.surfaceSurfelCount <= 32)
        assertTrue(stats.surfaceSurfelEvictionCount >= 0L)
        assertTrue(stats.surfaceSurfelRejectedCount >= 0L)
        assertTrue(
            stats.memoryPressure in 0.0..1.0,
        )
    }


}
