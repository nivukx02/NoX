package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class TemporalDepthConsistencyEngineTest {

    private val intrinsics =
        CameraIntrinsics(
            fx = 80.0,
            fy = 80.0,
            cx = 4.0,
            cy = 4.0,
        )

    private val pose =
        Pose(
            position = Vec3(0.0, 0.0, 0.0),
            rotation = Quat.IDENTITY,
        )

    @Test
    fun firstFrameStartsWarmupWithoutInventingConsistency() {
        val engine = TemporalDepthConsistencyEngine()

        val result =
            engine.evaluate(
                depthMeters = depth(8, 8, 2.0),
                width = 8,
                height = 8,
                intrinsics = intrinsics,
                cameraPose = pose,
                timestampNanos = 1_000_000_000L,
            )

        assertTrue(!result.warmedUp)
        assertTrue(result.warnings.contains("temporal-depth-warmup"))
    }

    @Test
    fun stableDepthWithStablePoseProducesStrongTemporalEvidence() {
        val engine = TemporalDepthConsistencyEngine()

        engine.evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )

        val result =
            engine.evaluate(
                depthMeters = depth(8, 8, 2.0),
                width = 8,
                height = 8,
                intrinsics = intrinsics,
                cameraPose = pose,
                timestampNanos = 1_033_000_000L,
            )

        assertTrue(result.warmedUp)
        assertTrue(result.validSampleRatio > 0.90)
        assertTrue(result.meanResidualMeters < 0.001)
        assertTrue(result.confidence > 0.60)
        assertTrue(result.accepted)
    }

    @Test
    fun cameraTranslationIsCompensatedRatherThanTreatedAsPixelNoise() {
        val engine = TemporalDepthConsistencyEngine()

        engine.evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )

        val movedPose =
            Pose(
                position = Vec3(0.02, 0.0, 0.0),
                rotation = Quat.IDENTITY,
            )

        val result =
            engine.evaluate(
                depthMeters = depth(8, 8, 2.0),
                width = 8,
                height = 8,
                intrinsics = intrinsics,
                cameraPose = movedPose,
                timestampNanos = 1_033_000_000L,
            )

        assertTrue(result.cameraTranslationMeters > 0.0)
        assertTrue(result.validSampleRatio > 0.50)
        assertTrue(result.accepted)
    }

    @Test
    fun inconsistentDepthLowersConfidenceAndProducesWarning() {
        val engine = TemporalDepthConsistencyEngine()

        engine.evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )

        val result =
            engine.evaluate(
                depthMeters = depth(8, 8, 2.30),
                width = 8,
                height = 8,
                intrinsics = intrinsics,
                cameraPose = pose,
                timestampNanos = 1_033_000_000L,
            )

        assertTrue(result.meanResidualMeters > 0.20)
        assertTrue(
            result.warnings.any {
                it.contains("residual")
            },
        )
        assertTrue(result.confidence < 0.70)
    }

    @Test
    fun largeCameraDiscontinuityStartsNewTemporalSegment() {
        val engine = TemporalDepthConsistencyEngine()

        engine.evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )

        val jump =
            Pose(
                position = Vec3(1.0, 0.0, 0.0),
                rotation = Quat.IDENTITY,
            )

        val result =
            engine.evaluate(
                depthMeters = depth(8, 8, 2.0),
                width = 8,
                height = 8,
                intrinsics = intrinsics,
                cameraPose = jump,
                timestampNanos = 1_033_000_000L,
            )

        assertTrue(result.hardReset)
        assertTrue(
            result.warnings.contains(
                "temporal-depth-camera-translation-jump",
            ),
        )
    }


    @Test
    fun gridConfigurationRejectsRunawaySampleAllocation() {
        val error = runCatching {
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 1024,
                gridHeight = 1024,
            )
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
        val maximumSupported = TemporalDepthConsistencyEngine.Config(
            gridWidth = 512,
            gridHeight = 512,
        )
        assertTrue(
            maximumSupported.gridWidth.toLong() * maximumSupported.gridHeight <=
                262_144L,
        )
    }

    @Test
    fun frameDimensionsAreValidatedBeforeIntMultiplicationCanOverflow() {
        val error = runCatching {
            TemporalDepthConsistencyEngine().evaluate(
                depthMeters = FloatArray(0),
                width = 65_536,
                height = 65_536,
                intrinsics = intrinsics,
                cameraPose = pose,
                timestampNanos = 1_000_000_000L,
            )
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
    }

    @Test
    fun nonFiniteSourceConfidenceCannotPoisonTemporalEvidence() {
        val result = TemporalDepthConsistencyEngine().evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
            sourceConfidence = Double.NaN,
        )

        assertTrue(result.confidence.isFinite())
        assertTrue(result.confidence == 0.0)
    }

    @Test
    fun nonFiniteTemporalConfigurationIsRejected() {
        val error = runCatching {
            TemporalDepthConsistencyEngine.Config(
                maxFrameGapSeconds = Double.POSITIVE_INFINITY,
            )
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
    }


    @Test
    fun calibratedIntrinsicsDimensionsMustMatchTheDepthFrame() {
        val error = runCatching {
            TemporalDepthConsistencyEngine().evaluate(
                depthMeters = depth(8, 8, 2.0),
                width = 8,
                height = 8,
                intrinsics = intrinsics.copy(imageWidth = 9, imageHeight = 8),
                cameraPose = pose,
                timestampNanos = 1_000_000_000L,
            )
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
    }

    @Test
    fun nonFiniteIntrinsicsAndDegenerateCameraPoseAreRejected() {
        val intrinsicsError = runCatching {
            TemporalDepthConsistencyEngine().evaluate(
                depthMeters = depth(8, 8, 2.0),
                width = 8,
                height = 8,
                intrinsics = intrinsics.copy(fx = Double.NaN),
                cameraPose = pose,
                timestampNanos = 1_000_000_000L,
            )
        }.exceptionOrNull()
        val poseError = runCatching {
            TemporalDepthConsistencyEngine().evaluate(
                depthMeters = depth(8, 8, 2.0),
                width = 8,
                height = 8,
                intrinsics = intrinsics,
                cameraPose = Pose(
                    position = Vec3(0.0, 0.0, 0.0),
                    rotation = Quat(Double.NaN, 0.0, 0.0, 0.0),
                ),
                timestampNanos = 1_000_000_000L,
            )
        }.exceptionOrNull()

        assertTrue(intrinsicsError is IllegalArgumentException)
        assertTrue(poseError is IllegalArgumentException)
    }


    @Test
    fun lowConfidencePixelsDoNotDominateTemporalResiduals() {
        val settings = TemporalDepthConsistencyEngine.Config(
            gridWidth = 8,
            gridHeight = 8,
        )
        val referenceEngine = TemporalDepthConsistencyEngine(settings)
        val confidenceAwareEngine = TemporalDepthConsistencyEngine(settings)
        val referenceDepth = depth(8, 8, 2.0)
        val referenceConfidence = FloatArray(64) { 1.0f }

        referenceEngine.evaluate(
            depthMeters = referenceDepth,
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )
        confidenceAwareEngine.evaluate(
            depthMeters = referenceDepth,
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
            depthConfidence = referenceConfidence,
        )

        val disturbedDepth = FloatArray(64) { index ->
            val x = index % 8
            if (x < 4) 5.0f else 2.0f
        }
        val currentConfidence = FloatArray(64) { index ->
            if (index % 8 < 4) 0.0f else 1.0f
        }

        val unweighted = referenceEngine.evaluate(
            depthMeters = disturbedDepth,
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_033_000_000L,
        )
        val weighted = confidenceAwareEngine.evaluate(
            depthMeters = disturbedDepth,
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_033_000_000L,
            depthConfidence = currentConfidence,
        )

        assertTrue(unweighted.p95ResidualMeters > 1.0)
        assertTrue(weighted.p95ResidualMeters < 0.001)
        assertTrue(weighted.validSampleRatio in 0.45..0.55)
    }

    @Test
    fun depthConfidenceArrayMustMatchTheDepthFrameSize() {
        val error = runCatching {
            TemporalDepthConsistencyEngine().evaluate(
                depthMeters = depth(8, 8, 2.0),
                width = 8,
                height = 8,
                intrinsics = intrinsics,
                cameraPose = pose,
                timestampNanos = 1_000_000_000L,
                depthConfidence = FloatArray(63) { 1.0f },
            )
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
    }


    @Test
    fun sparseWarmupFrameIsNotReportedAsFullyCoveredOrAccepted() {
        val sparseDepth = FloatArray(8 * 8) { index ->
            if (index == 0) 2.0f else Float.NaN
        }
        val result = TemporalDepthConsistencyEngine(
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 8,
                gridHeight = 8,
            ),
        ).evaluate(
            depthMeters = sparseDepth,
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )

        assertTrue(result.validSampleRatio in 0.01..0.02)
        assertTrue(!result.accepted)
        assertTrue(!result.warmedUp)
        assertTrue(result.warnings.contains("temporal-depth-valid-coverage-low"))
        assertTrue(result.recommendations.any { it.contains("Increase synchronized depth coverage") })
    }


    @Test
    fun lowPixelConfidenceLimitsWarmupCoverageAndConfidence() {
        val confidence = FloatArray(8 * 8) { 0.05f }
        val result = TemporalDepthConsistencyEngine(
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 8,
                gridHeight = 8,
            ),
        ).evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
            depthConfidence = confidence,
        )

        assertTrue(result.validSampleRatio in 0.04..0.06)
        assertTrue(result.confidence < 0.1)
        assertTrue(!result.accepted)
    }


    @Test
    fun discontinuityResetSurfacesLowCoverageInTheNewSegment() {
        val engine = TemporalDepthConsistencyEngine(
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 8,
                gridHeight = 8,
            ),
        )
        engine.evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )

        val sparse = FloatArray(8 * 8) { index ->
            if (index == 0) 2.0f else Float.NaN
        }
        val result = engine.evaluate(
            depthMeters = sparse,
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 2_000_000_000L,
        )

        assertTrue(result.hardReset)
        assertTrue(result.warnings.contains("temporal-depth-valid-coverage-low"))
        assertTrue(result.warnings.contains("temporal-depth-confidence-low"))
        assertTrue(result.recommendations.any { it.contains("Increase synchronized depth coverage") })
    }


    @Test
    fun outOfOrderDepthTimestampDoesNotReplaceNewerHistory() {
        val engine = TemporalDepthConsistencyEngine(
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 8,
                gridHeight = 8,
            ),
        )
        engine.evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 2_000_000_000L,
        )

        val stale = engine.evaluate(
            depthMeters = depth(8, 8, 5.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )
        val resumed = engine.evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 2_033_000_000L,
        )

        assertTrue(stale.warnings.contains("temporal-depth-stale-timestamp"))
        assertTrue(!stale.accepted)
        assertTrue(!stale.hardReset)
        assertTrue(resumed.warmedUp)
        assertTrue(!resumed.hardReset)
        assertTrue(resumed.meanResidualMeters < 0.001)
    }

    @Test
    fun sparseWarmupConfidencePersistsIntoTheNextTemporalEstimate() {
        val engine = TemporalDepthConsistencyEngine(
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 8,
                gridHeight = 8,
            ),
        )
        val sparseDepth = FloatArray(64) { index ->
            if (index == 0) 2.0f else Float.NaN
        }
        val first = engine.evaluate(
            depthMeters = sparseDepth,
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
        )
        val next = engine.evaluate(
            depthMeters = depth(8, 8, 2.0),
            width = 8,
            height = 8,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_033_000_000L,
        )

        assertTrue(first.confidence < 0.02)
        assertTrue(next.warmedUp)
        assertTrue(next.confidence < 0.50)
    }


    @Test
    fun temporalResidualUsesCameraMotionProjectedPixelNotOriginalPixel() {
        val engine = TemporalDepthConsistencyEngine(
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 8,
                gridHeight = 8,
            ),
        )
        val width = 8
        val height = 8
        val slope = 0.8
        val intercept = 2.0
        val cameraTranslationX = 0.08
        fun plane(cameraX: Double): FloatArray =
            FloatArray(width * height) { index ->
                val x = index % width
                val rayX = (x - intrinsics.cx) / intrinsics.fx
                ((slope * cameraX + intercept) / (1.0 - slope * rayX)).toFloat()
            }

        engine.evaluate(
            depthMeters = plane(0.0),
            width = width,
            height = height,
            intrinsics = intrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
            depthConvention = DepthSampleConvention.OPTICAL_AXIS_Z,
        )
        val currentPose = Pose(
            position = Vec3(cameraTranslationX, 0.0, 0.0),
            rotation = Quat.IDENTITY,
        )
        val result = engine.evaluate(
            depthMeters = plane(cameraTranslationX),
            width = width,
            height = height,
            intrinsics = intrinsics,
            cameraPose = currentPose,
            timestampNanos = 1_033_000_000L,
            depthConvention = DepthSampleConvention.OPTICAL_AXIS_Z,
        )

        assertTrue(result.warmedUp)
        assertTrue(result.validSampleRatio > 0.34)
        assertTrue(
            result.p95ResidualMeters < 0.055,
        )
    }


    @Test
    fun axialDepthProjectionDoesNotNormalizeAwayOffAxisRayCoordinates() {
        val wideFieldIntrinsics = CameraIntrinsics(
            fx = 4.0,
            fy = 4.0,
            cx = 3.5,
            cy = 3.5,
        )
        val engine = TemporalDepthConsistencyEngine(
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 8,
                gridHeight = 8,
            ),
        )
        val slope = 0.30
        val intercept = 2.0
        val plane = FloatArray(8 * 8) { index ->
            val x = index % 8
            val rayX = (x - wideFieldIntrinsics.cx) / wideFieldIntrinsics.fx
            (intercept / (1.0 - slope * rayX)).toFloat()
        }

        engine.evaluate(
            depthMeters = plane,
            width = 8,
            height = 8,
            intrinsics = wideFieldIntrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
            depthConvention = DepthSampleConvention.OPTICAL_AXIS_Z,
        )
        val result = engine.evaluate(
            depthMeters = plane.copyOf(),
            width = 8,
            height = 8,
            intrinsics = wideFieldIntrinsics,
            cameraPose = pose,
            timestampNanos = 1_033_000_000L,
            depthConvention = DepthSampleConvention.OPTICAL_AXIS_Z,
        )

        assertTrue(result.warmedUp)
        assertTrue(result.validSampleRatio > 0.90)
        assertTrue(result.p95ResidualMeters < 0.01)
        assertTrue(result.accepted)
    }


    @Test
    fun euclideanRangeDepthUsesRayLengthRatherThanOpticalAxisZ() {
        val engine = TemporalDepthConsistencyEngine(
            TemporalDepthConsistencyEngine.Config(
                gridWidth = 8,
                gridHeight = 8,
            ),
        )
        val radialRange = depth(8, 8, 2.0)
        val rangeIntrinsics = CameraIntrinsics(
            fx = 4.0,
            fy = 4.0,
            cx = 3.5,
            cy = 3.5,
        )
        engine.evaluate(
            depthMeters = radialRange,
            width = 8,
            height = 8,
            intrinsics = rangeIntrinsics,
            cameraPose = pose,
            timestampNanos = 1_000_000_000L,
            depthConvention = DepthSampleConvention.EUCLIDEAN_RAY,
        )
        val result = engine.evaluate(
            depthMeters = radialRange.copyOf(),
            width = 8,
            height = 8,
            intrinsics = rangeIntrinsics,
            cameraPose = pose,
            timestampNanos = 1_033_000_000L,
            depthConvention = DepthSampleConvention.EUCLIDEAN_RAY,
        )

        assertTrue(result.warmedUp)
        assertTrue(result.p95ResidualMeters < 0.001)
        assertTrue(result.accepted)
    }

    private fun depth(
        width: Int,
        height: Int,
        value: Double,
    ): FloatArray =
        FloatArray(width * height) {
            value.toFloat()
        }
}
