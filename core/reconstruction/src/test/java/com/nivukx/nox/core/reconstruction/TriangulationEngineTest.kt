package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TriangulationEngineTest {
    @Test
    fun rejectsGeometryWhenTemporalAlignmentConfidenceIsTooLow() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.25, 0.15, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val observations = listOf(
            TriangulationEngine.Observation(
                cameraPose = Pose(
                    Vec3(0.0, 0.0, 0.0),
                    Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                u = p0.first,
                v = p0.second,
                weight = 1.0,
                temporalAlignmentConfidence = 1.0,
            ),
            TriangulationEngine.Observation(
                cameraPose = Pose(
                    Vec3(1.0, 0.0, 0.0),
                    Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                u = p1.first,
                v = p1.second,
                weight = 1.0,
                temporalAlignmentConfidence = 0.04,
            ),
        )

        val result =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
                minimumTemporalConfidence = 0.10,
            ).robust(observations)

        assertTrue(result == null)
    }

    @Test
    fun rejectsNearlyParallelRayGeometryEvenWhenParallaxGateIsRelaxed() {
        val intrinsics =
            CameraIntrinsics(
                fx = 2200.0,
                fy = 2200.0,
                cx = 1100.0,
                cy = 800.0,
            )
        val point = Vec3(0.05, 0.0, 10.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx *
                    (point.x - cameraX) /
                    point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy *
                    point.y /
                    point.z +
                    intrinsics.cy
            )

        val first = pixel(0.0)
        val second = pixel(0.0001)

        val result =
            TriangulationEngine(
                minParallaxDegrees = 0.0001,
                maximumNormalMatrixConditionNumber = 100.0,
            ).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose =
                            Pose(
                                Vec3(0.0, 0.0, 0.0),
                                Quat.IDENTITY,
                            ),
                        intrinsics = intrinsics,
                        u = first.first,
                        v = first.second,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose =
                            Pose(
                                Vec3(0.0001, 0.0, 0.0),
                                Quat.IDENTITY,
                            ),
                        intrinsics = intrinsics,
                        u = second.first,
                        v = second.second,
                    ),
                ),
            )

        assertTrue(result == null)
    }

    @Test
    fun preservesGeometryForWellAlignedTemporalObservations() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.25, 0.15, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val result =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
                minimumTemporalConfidence = 0.10,
            ).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p0.first,
                        v = p0.second,
                        weight = 1.0,
                        temporalAlignmentConfidence = 0.94,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(1.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p1.first,
                        v = p1.second,
                        weight = 1.0,
                        temporalAlignmentConfidence = 0.96,
                    ),
                ),
            )

        assertNotNull(result)
        result ?: return
        assertEquals(point.x, result.point.x, 1e-3)
        assertEquals(point.y, result.point.y, 1e-3)
        assertEquals(point.z, result.point.z, 1e-3)
        assertTrue(result.normalMatrixConditionEstimate.isFinite())
        assertTrue(result.normalMatrixConditionEstimate >= 1.0)
        assertTrue(result.normalMatrixConditionScore in 0.0..1.0)
    }

    @Test
    fun directionalUncertaintyIsFiniteForHealthyMetricSolve() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.25, 0.15, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.2)

        val result =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
            ).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p0.first,
                        v = p0.second,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(1.2, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p1.first,
                        v = p1.second,
                    ),
                ),
            )

        assertNotNull(result)
        result ?: return
        assertTrue(result.uncertaintyMajorMeters.isFinite())
        assertTrue(result.uncertaintyDepthMeters.isFinite())
        assertTrue(result.uncertaintyAnisotropyRatio >= 1.0)
        assertTrue(result.uncertainty95MajorMeters >= result.uncertaintyMajorMeters)
        assertTrue(result.uncertaintyConfidence in 0.0..1.0)
    }

    @Test
    fun directionalUncertaintyReportsAnisotropyForWeakDepthGeometry() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.25, 0.15, 8.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(0.10)

        val result =
            TriangulationEngine(
                minParallaxDegrees = 0.01,
            ).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p0.first,
                        v = p0.second,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.10, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p1.first,
                        v = p1.second,
                    ),
                ),
            )

        assertNotNull(result)
        result ?: return
        assertTrue(
            result.uncertaintyDepthMeters >=
                result.uncertaintyMinorMeters,
        )
        assertTrue(
            result.uncertaintyDepthDominance >= 0.50 ||
                result.uncertaintyAnisotropyRatio >= 2.0,
        )
    }

    @Test
    fun reconstructsKnownMetricPointFromTwoCameras() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.25, 0.15, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z + intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z + intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val observations = listOf(
            TriangulationEngine.Observation(
                cameraPose = Pose(Vec3(0.0, 0.0, 0.0), Quat.IDENTITY),
                intrinsics = intrinsics,
                u = p0.first,
                v = p0.second,
                weight = 1.0,
            ),
            TriangulationEngine.Observation(
                cameraPose = Pose(Vec3(1.0, 0.0, 0.0), Quat.IDENTITY),
                intrinsics = intrinsics,
                u = p1.first,
                v = p1.second,
                weight = 1.0,
            ),
        )

        val result = TriangulationEngine(minParallaxDegrees = 0.1).robust(observations)

        assertNotNull(result)
        result ?: return
        assertEquals(point.x, result.point.x, 1e-3)
        assertEquals(point.y, result.point.y, 1e-3)
        assertEquals(point.z, result.point.z, 1e-3)
        assertTrue(result.rmsReprojectionPx < 1e-3)
        assertEquals(2, result.inliers)
        assertTrue(result.symmetricEpipolarConsistencyScore > 0.90)
        assertTrue(result.symmetricEpipolarEvaluatedPairCount >= 1)
    }

    @Test
    fun contradictoryTwoViewCorrespondenceFailsClosedOnEpipolarEvidence() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.20, 0.10, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val result =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
            ).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        cameraId = "cam-A",
                        u = p0.first,
                        v = p0.second,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(1.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        cameraId = "cam-B",
                        u = p1.first + 220.0,
                        v = p1.second + 160.0,
                    ),
                ),
            )

        assertTrue(result == null)
    }

    @Test
    fun lowVisibilityAndHighObservationUncertaintyReduceTriangulationWeight() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.25, 0.15, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val healthy =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
            ).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p0.first,
                        v = p0.second,
                        observationUncertainty = 0.08,
                        visibility = 0.96,
                        sourceQuality = 0.95,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(1.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p1.first,
                        v = p1.second,
                        observationUncertainty = 0.08,
                        visibility = 0.96,
                        sourceQuality = 0.95,
                    ),
                ),
            )

        val degraded =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
            ).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p0.first,
                        v = p0.second,
                        observationUncertainty = 0.92,
                        visibility = 0.22,
                        sourceQuality = 0.35,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(1.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = p1.first,
                        v = p1.second,
                        observationUncertainty = 0.92,
                        visibility = 0.22,
                        sourceQuality = 0.35,
                    ),
                ),
            )

        assertNotNull(healthy)
        assertNotNull(degraded)
        healthy ?: return
        degraded ?: return
        assertTrue(
            degraded.uncertaintyMeters >=
                healthy.uncertaintyMeters,
        )
        assertTrue(
            degraded.uncertaintyConfidence <=
                healthy.uncertaintyConfidence,
        )
    }

    @Test
    fun zeroVisibilityEvidenceIsRejectedBeforeTriangulation() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )

        val observations =
            listOf(
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(0.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = 500.0,
                    v = 500.0,
                    visibility = 0.0,
                ),
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(1.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = 250.0,
                    v = 500.0,
                    visibility = 0.0,
                ),
            )

        val result =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
            ).robust(observations)

        assertTrue(result == null)
    }

    @Test
    fun denseEvidenceSignalsArePropagatedIntoTriangulationResult() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.20, 0.10, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val result = TriangulationEngine(
            minParallaxDegrees = 0.1,
        ).robust(
            listOf(
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(0.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = p0.first,
                    v = p0.second,
                    crossViewAgreement = 0.96,
                    temporalStability = 0.94,
                    contactProbability = 0.80,
                ),
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(1.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = p1.first,
                    v = p1.second,
                    crossViewAgreement = 0.92,
                    temporalStability = 0.91,
                    contactProbability = 0.70,
                ),
            ),
        )

        assertNotNull(result)
        result ?: return
        assertTrue(result.evidenceCoverage > 0.50)
        assertTrue(result.temporalStability > 0.80)
        assertTrue(result.contactSupport > 0.60)
    }

    @Test
    fun poorProbabilityMassInflatesReportedSpatialUncertainty() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.15, 0.10, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        fun observation(
            cameraX: Double,
            p: Pair<Double, Double>,
            score: Double,
        ) =
            TriangulationEngine.Observation(
                cameraPose = Pose(
                    Vec3(cameraX, 0.0, 0.0),
                    Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                cameraId = "cam-$cameraX",
                u = p.first,
                v = p.second,
                probabilityModes = listOf(
                    TriangulationEngine.ProbabilityMode(
                        u = p.first,
                        v = p.second,
                        sigmaX = 1.5,
                        sigmaY = 1.5,
                        weight = 1.0,
                    ),
                ),
                distributionCalibrationScore = score,
            )

        val calibrated =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
            ).robust(
                listOf(
                    observation(0.0, p0, 0.95),
                    observation(1.0, p1, 0.95),
                ),
            )

        val poor =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
            ).robust(
                listOf(
                    observation(0.0, p0, 0.10),
                    observation(1.0, p1, 0.10),
                ),
            )

        assertNotNull(calibrated)
        assertNotNull(poor)
        calibrated ?: return
        poor ?: return

        assertTrue(
            poor.uncertaintyMeters >=
                calibrated.uncertaintyMeters,
        )
        assertTrue(
            poor.probabilisticEvidenceScore <=
                calibrated.probabilisticEvidenceScore,
        )
    }

    @Test
    fun degradedDenseEvidenceDoesNotBecomeWeightlessButIsPenalized() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.20, 0.10, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val healthy = TriangulationEngine(
            minParallaxDegrees = 0.1,
        ).robust(
            listOf(
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(0.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = p0.first,
                    v = p0.second,
                    crossViewAgreement = 0.98,
                    temporalStability = 0.98,
                ),
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(1.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = p1.first,
                    v = p1.second,
                    crossViewAgreement = 0.98,
                    temporalStability = 0.98,
                ),
            ),
        )

        val degraded = TriangulationEngine(
            minParallaxDegrees = 0.1,
        ).robust(
            listOf(
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(0.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = p0.first,
                    v = p0.second,
                    crossViewAgreement = 0.20,
                    temporalStability = 0.22,
                ),
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(1.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = p1.first,
                    v = p1.second,
                    crossViewAgreement = 0.20,
                    temporalStability = 0.22,
                ),
            ),
        )

        assertNotNull(healthy)
        assertNotNull(degraded)
        healthy ?: return
        degraded ?: return
        assertTrue(degraded.evidenceCoverage < healthy.evidenceCoverage)
        assertTrue(degraded.temporalStability < healthy.temporalStability)
    }

    @Test
    fun triangulationDeduplicatesSameCameraAndReportsIndependentViewDiversity() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.15, 0.10, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        fun observation(
            cameraId: String,
            cameraX: Double,
            weight: Double = 1.0,
        ): TriangulationEngine.Observation {
            val p = pixel(cameraX)
            return TriangulationEngine.Observation(
                cameraPose = Pose(
                    Vec3(cameraX, 0.0, 0.0),
                    Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                cameraId = cameraId,
                u = p.first,
                v = p.second,
                weight = weight,
            )
        }

        val result = TriangulationEngine(
            minParallaxDegrees = 0.1,
        ).robust(
            listOf(
                observation("cam-A", 0.0, 1.0),
                observation("cam-A", 0.0, 0.30),
                observation("cam-B", 1.0, 1.0),
                observation("cam-C", 2.0, 1.0),
            ),
        )

        assertNotNull(result)
        result ?: return
        assertEquals(3, result.independentViewCount)
        assertTrue(result.viewDiversityScore > 0.50)
        assertTrue(result.dominantViewWeightRatio < 0.50)
    }

    @Test
    fun multimodalProbabilityMassIsPreservedAndReported() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.15, 0.10, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        fun probabilityModes(
            u: Double,
            v: Double,
        ) =
            listOf(
                TriangulationEngine.ProbabilityMode(
                    u = u,
                    v = v,
                    sigmaX = 1.2,
                    sigmaY = 1.4,
                    weight = 0.72,
                ),
                TriangulationEngine.ProbabilityMode(
                    u = u + 14.0,
                    v = v - 8.0,
                    sigmaX = 3.5,
                    sigmaY = 3.0,
                    weight = 0.28,
                ),
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val result =
            TriangulationEngine(
                minParallaxDegrees = 0.1,
            ).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        cameraId = "cam-A",
                        u = p0.first,
                        v = p0.second,
                        probabilityModes =
                            probabilityModes(
                                p0.first,
                                p0.second,
                            ),
                        distributionCalibrationScore = 0.92,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose = Pose(
                            Vec3(1.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        cameraId = "cam-B",
                        u = p1.first,
                        v = p1.second,
                        probabilityModes =
                            probabilityModes(
                                p1.first,
                                p1.second,
                            ),
                        distributionCalibrationScore = 0.90,
                    ),
                ),
            )

        assertNotNull(result)
        result ?: return
        assertTrue(
            result.probabilisticEvidenceScore > 0.60,
        )
        assertTrue(
            result.probabilisticEvidenceWarnings.contains(
                "multimodal-2d-keypoint-evidence-preserved",
            ),
        )
        assertTrue(
            result.point.distanceTo(point) < 1e-2,
        )
    }

    @Test
    fun triangulationFailsClosedWhenOneCameraDominatesTheMetricSolve() {
        val intrinsics = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )
        val point = Vec3(0.15, 0.10, 4.0)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.0)

        val result = TriangulationEngine(
            minParallaxDegrees = 0.1,
            maximumDominantViewWeightRatio = 0.82,
        ).robust(
            listOf(
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(0.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    cameraId = "cam-A",
                    u = p0.first,
                    v = p0.second,
                    weight = 1.0,
                ),
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(1.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    cameraId = "cam-B",
                    u = p1.first,
                    v = p1.second,
                    weight = 0.02,
                ),
            ),
        )

        assertTrue(result == null)
    }

}


class TriangulationEngineRollingShutterMotionCompensationTest {
    @Test
    fun rowTimeVelocityCompensationPreservesMovingPointGeometry() {
        val intrinsics = CameraIntrinsics(
            fx = 1200.0,
            fy = 1200.0,
            cx = 640.0,
            cy = 360.0,
        )
        val referencePoint = Vec3(0.20, 0.10, 4.50)
        val velocity = Vec3(0.60, 0.0, 0.0)
        val rowOffsetSeconds = 0.008
        val observedPointAtRowTime =
            referencePoint + velocity * rowOffsetSeconds

        fun pixel(
            cameraX: Double,
        ): Pair<Double, Double> =
            (
                intrinsics.fx *
                    (observedPointAtRowTime.x - cameraX) /
                    observedPointAtRowTime.z +
                    intrinsics.cx
            ) to (
                intrinsics.fy *
                    observedPointAtRowTime.y /
                    observedPointAtRowTime.z +
                    intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(1.10)

        val result = TriangulationEngine(
            minParallaxDegrees = 0.1,
            maximumMotionCompensationMeters = 0.20,
        ).robust(
            listOf(
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(0.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = p0.first,
                    v = p0.second,
                    rowTimeOffsetNanos =
                        (rowOffsetSeconds * 1e9).toLong(),
                    motionVelocityMetersPerSecond = velocity,
                ),
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(1.10, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    u = p1.first,
                    v = p1.second,
                    rowTimeOffsetNanos =
                        (rowOffsetSeconds * 1e9).toLong(),
                    motionVelocityMetersPerSecond = velocity,
                ),
            ),
        )

        assertNotNull(result)
        result ?: return
        assertEquals(
            referencePoint.x,
            result.point.x,
            0.015,
        )
        assertEquals(
            referencePoint.y,
            result.point.y,
            0.015,
        )
        assertEquals(
            referencePoint.z,
            result.point.z,
            0.03,
        )
    }
}
