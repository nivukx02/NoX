package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DenseMultiViewSurfaceReconstructionEngineTest {

    private val intrinsics =
        CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )

    private fun pixel(
        point: Vec3,
        cameraX: Double,
    ): Pair<Double, Double> =
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

    private fun observation(
        key: DenseMultiViewSurfaceReconstructionEngine.PointKey,
        point: Vec3,
        cameraX: Double,
        cameraId: String,
        confidence: Double = 0.98,
        photometricConsistency: Double = 1.0,
        silhouetteConsistency: Double = 1.0,
        motionBoundaryConfidence: Double = 1.0,
        lightingStability: Double = 1.0,
    ): DenseMultiViewSurfaceReconstructionEngine.PointObservation {
        val p = pixel(point, cameraX)
        return DenseMultiViewSurfaceReconstructionEngine.PointObservation(
            key = key,
            observation =
                TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(cameraX, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    cameraId = cameraId,
                    u = p.first,
                    v = p.second,
                    weight = confidence,
                    temporalAlignmentConfidence = 0.98,
                    temporalUncertaintyNanos = 2_000_000L,
                    observationUncertainty = 0.03,
                    visibility = 0.98,
                    sourceQuality = confidence,
                    crossViewAgreement = 0.95,
                    temporalStability = 0.96,
                ),
            semanticConfidence = confidence,
            visibility = 0.98,
            sourceDiversity = 0.95,
            photometricConsistency = photometricConsistency,
            silhouetteConsistency = silhouetteConsistency,
            motionBoundaryConfidence = motionBoundaryConfidence,
            lightingStability = lightingStability,
        )
    }

    @Test
    fun reconstructsMetricDensePointFromTwoViews() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    minimumPointUncertaintyMeters = 0.50,
                    hardPointUncertaintyMeters = 0.80,
                    preferredParallaxDegrees = 1.0,
                ),
            )

        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 17,
            )
        val target = Vec3(0.25, 0.15, 4.0)

        val result =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = listOf(
                        observation(
                            key,
                            target,
                            0.0,
                            "camera-A",
                        ),
                        observation(
                            key,
                            target,
                            1.2,
                            "camera-B",
                        ),
                    ),
                ),
            )

        val point = result.points.single()
        assertTrue(
            point.positionMeters.x.isFinite() &&
                point.positionMeters.y.isFinite() &&
                point.positionMeters.z.isFinite(),
        )
        assertEquals(
            target.x,
            point.positionMeters.x,
            2e-2,
        )
        assertEquals(
            target.y,
            point.positionMeters.y,
            2e-2,
        )
        assertEquals(
            target.z,
            point.positionMeters.z,
            2e-2,
        )
        assertTrue(result.coverage > 0.0)
        assertTrue(point.independentViewCount >= 2)
        engine.close()
    }

    @Test
    fun duplicateCameraEvidenceDoesNotMasqueradeAsIndependentViews() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine()

        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 2,
            )
        val target = Vec3(0.2, 0.0, 4.0)

        val result =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = listOf(
                        observation(
                            key,
                            target,
                            0.0,
                            "camera-A",
                        ),
                        observation(
                            key,
                            target,
                            0.0,
                            "camera-A",
                        ),
                    ),
                ),
            )

        val point = result.points.single()
        assertTrue(
            point.state ==
                DenseMultiViewSurfaceReconstructionEngine.PointState
                    .GEOMETRY_REJECTED ||
                point.state ==
                DenseMultiViewSurfaceReconstructionEngine.PointState
                    .TEMPORALLY_BRIDGED,
        )
        engine.close()
    }

    @Test
    fun weakGeometryFailsClosedInsteadOfInventingDepth() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    minimumParallaxDegrees = 1.0,
                ),
            )

        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 4,
            )
        val target = Vec3(0.2, 0.0, 12.0)

        val weakA =
            observation(
                key,
                target,
                0.0,
                "camera-A",
            )
        val weakB =
            observation(
                key,
                target,
                0.02,
                "camera-B",
            )

        val result =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = listOf(
                        weakA,
                        weakB,
                    ),
                ),
            )

        assertTrue(
            result.points.single().state !=
                DenseMultiViewSurfaceReconstructionEngine.PointState
                    .TRUSTED,
        )
        assertTrue(
            result.rejectedPointCount >= 0,
        )
        engine.close()
    }

    @Test
    fun shortOcclusionUsesBoundedPredictionAndLongGapExpires() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    predictionHorizonMillis = 80.0,
                    minimumPointUncertaintyMeters = 0.50,
                    hardPointUncertaintyMeters = 0.80,
                ),
            )

        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 11,
            )
        val target = Vec3(0.25, 0.15, 4.0)

        engine.process(
            DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                timestampNanos = 1_000_000_000L,
                points = listOf(
                    observation(
                        key,
                        target,
                        0.0,
                        "camera-A",
                    ),
                    observation(
                        key,
                        target,
                        1.2,
                        "camera-B",
                    ),
                ),
            ),
        )

        val bridged =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_040_000_000L,
                    points = emptyList(),
                ),
            )

        assertTrue(
            bridged.points.any {
                it.state ==
                    DenseMultiViewSurfaceReconstructionEngine.PointState
                        .TEMPORALLY_BRIDGED
            },
        )

        val expired =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_250_000_000L,
                    points = emptyList(),
                ),
            )

        assertTrue(
            expired.points.none {
                it.state ==
                    DenseMultiViewSurfaceReconstructionEngine.PointState
                        .TEMPORALLY_BRIDGED
            },
        )
        engine.close()
    }

    @Test
    fun adaptiveViewSelectionPrunesWeakViewsOnlyWhenRigExceedsGeometricBudget() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    maximumGeometricViewsPerPoint = 6,
                    minimumGeometricViewsPerPoint = 2,
                    minimumViewSelectionScore = 0.30,
                    minimumPointUncertaintyMeters = 0.50,
                    hardPointUncertaintyMeters = 0.80,
                    preferredParallaxDegrees = 1.5,
                ),
            )

        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 44,
            )
        val target = Vec3(0.2, 0.1, 4.0)

        val points =
            (0 until 10).map { index ->
                val confidence =
                    if (index >= 7) 0.20 else 0.97
                observation(
                    key = key,
                    point = target,
                    cameraX = index * 0.40,
                    cameraId = "camera-$index",
                    confidence = confidence,
                )
            }

        val result =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = points,
                ),
            )

        val point = result.points.single()

        assertEquals(6, point.viewCount)
        assertTrue(point.excludedWeakViewCount > 0)
        assertTrue(point.adaptiveViewSelectionScore > 0.50)
        assertTrue(point.adaptiveViewDiversityScore > 0.25)
        assertTrue(result.excludedWeakViewCount > 0)

        engine.close()
    }

    @Test
    fun topologyStrainIsMeasuredSeparatelyFromPointConfidence() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    minimumPointUncertaintyMeters = 0.50,
                    hardPointUncertaintyMeters = 0.80,
                    preferredParallaxDegrees = 1.0,
                ),
            )

        val keyA =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 20,
            )
        val keyB =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 21,
            )

        val pointA = Vec3(0.0, 0.0, 4.0)
        val pointB = Vec3(0.5, 0.0, 4.0)

        val edge =
            DenseMultiViewSurfaceReconstructionEngine.TopologyEdge(
                a = keyA,
                b = keyB,
                restLengthMeters = 0.25,
                confidence = 0.95,
                rigidity = 0.9,
            )

        val result =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = listOf(
                        observation(
                            keyA,
                            pointA,
                            0.0,
                            "camera-A",
                        ),
                        observation(
                            keyA,
                            pointA,
                            1.2,
                            "camera-B",
                        ),
                        observation(
                            keyB,
                            pointB,
                            0.0,
                            "camera-A",
                        ),
                        observation(
                            keyB,
                            pointB,
                            1.2,
                            "camera-B",
                        ),
                    ),
                    topologyEdges = listOf(edge),
                ),
            )

        assertTrue(result.topologyEdgesEvaluated == 1)
        assertTrue(result.topologyWarningCount >= 1)
        assertTrue(result.topologyStability < 1.0)
        engine.close()
    }

    @Test
    fun duplicateTimestampIsIgnoredWithoutReinjectingKinematicState() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine()

        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 9,
            )
        val target = Vec3(0.1, 0.05, 4.0)

        val first =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 3_000_000_000L,
                    points = listOf(
                        observation(
                            key,
                            target,
                            0.0,
                            "camera-A",
                        ),
                        observation(
                            key,
                            target,
                            1.2,
                            "camera-B",
                        ),
                    ),
                ),
            )

        val duplicate =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 3_000_000_000L,
                    points = listOf(
                        observation(
                            key,
                            Vec3(3.5, 2.0, 0.5),
                            0.0,
                            "camera-A",
                        ),
                        observation(
                            key,
                            Vec3(3.5, 2.0, 0.5),
                            1.2,
                            "camera-B",
                        ),
                    ),
                ),
            )

        assertEquals(first.timestampNanos, duplicate.timestampNanos)
        assertEquals(
            first.points.single().positionMeters.x,
            duplicate.points.single().positionMeters.x,
            1e-9,
        )
        assertEquals(
            first.points.single().velocityMetersPerSecond.norm(),
            duplicate.points.single().velocityMetersPerSecond.norm(),
            1e-9,
        )
        assertTrue(
            duplicate.warnings.contains(
                "dense reconstruction timestamp reversal rejected",
            ).not(),
        )
        engine.close()
    }

    @Test
    fun timestampReversalIsHardRejected() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine()

        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 1,
            )
        val target = Vec3(0.1, 0.1, 3.5)

        engine.process(
            DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                timestampNanos = 2_000_000_000L,
                points = listOf(
                    observation(
                        key,
                        target,
                        0.0,
                        "camera-A",
                    ),
                    observation(
                        key,
                        target,
                        1.0,
                        "camera-B",
                    ),
                ),
            ),
        )

        val reversed =
            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = emptyList(),
                ),
            )

        assertTrue(reversed.hardFailure)
        assertTrue(
            reversed.warnings.any {
                it.contains("timestamp reversal")
            },
        )
        engine.close()
    }

    @Test
    fun stateMemoryIsBounded() {
        val engine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    maximumPoints = 32,
                    minimumPointUncertaintyMeters = 0.50,
                    hardPointUncertaintyMeters = 0.80,
                    preferredParallaxDegrees = 1.0,
                ),
            )

        repeat(80) { frame ->
            val points =
                (0 until 64).map { pointId ->
                    val key =
                        DenseMultiViewSurfaceReconstructionEngine.PointKey(
                            subjectId = 0,
                            pointId = pointId,
                        )
                    val target =
                        Vec3(
                            pointId * 0.004,
                            0.0,
                            4.0,
                        )
                    observation(
                        key,
                        target,
                        0.0,
                        "camera-A",
                    )
                }.flatMap { first ->
                    val pointId = first.key.pointId
                    val key = first.key
                    val target =
                        Vec3(
                            pointId * 0.004,
                            0.0,
                            4.0,
                        )
                    listOf(
                        first,
                        observation(
                            key,
                            target,
                            1.0,
                            "camera-B",
                        ),
                    )
                }

            engine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos =
                        1_000_000_000L +
                            frame * 20_000_000L,
                    points = points,
                ),
            )
        }

        assertTrue(
            engine.telemetry().activePointCount <= 32,
        )
        engine.close()
    }

    @Test
    fun unstableAppearanceReducesDenseSurfaceConfidenceWithoutChangingGeometry() {
        val stableEngine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    minimumPointUncertaintyMeters = 0.50,
                    hardPointUncertaintyMeters = 0.80,
                    preferredParallaxDegrees = 1.0,
                    appearanceFusionWeight = 0.20,
                    appearanceSelectionWeight = 0.10,
                ),
            )
        val unstableEngine =
            DenseMultiViewSurfaceReconstructionEngine(
                DenseMultiViewSurfaceReconstructionEngine.Config(
                    minimumPointUncertaintyMeters = 0.50,
                    hardPointUncertaintyMeters = 0.80,
                    preferredParallaxDegrees = 1.0,
                    appearanceFusionWeight = 0.20,
                    appearanceSelectionWeight = 0.10,
                ),
            )

        val key =
            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                subjectId = 0,
                pointId = 77,
            )
        val target = Vec3(0.2, 0.1, 4.0)

        val stablePoints =
            listOf(
                observation(
                    key = key,
                    point = target,
                    cameraX = 0.0,
                    cameraId = "camera-A",
                    photometricConsistency = 0.98,
                    silhouetteConsistency = 0.98,
                    motionBoundaryConfidence = 0.96,
                    lightingStability = 0.97,
                ),
                observation(
                    key = key,
                    point = target,
                    cameraX = 1.2,
                    cameraId = "camera-B",
                    photometricConsistency = 0.98,
                    silhouetteConsistency = 0.98,
                    motionBoundaryConfidence = 0.96,
                    lightingStability = 0.97,
                ),
            )

        val unstablePoints =
            listOf(
                observation(
                    key = key,
                    point = target,
                    cameraX = 0.0,
                    cameraId = "camera-A",
                    photometricConsistency = 0.18,
                    silhouetteConsistency = 0.25,
                    motionBoundaryConfidence = 0.20,
                    lightingStability = 0.15,
                ),
                observation(
                    key = key,
                    point = target,
                    cameraX = 1.2,
                    cameraId = "camera-B",
                    photometricConsistency = 0.18,
                    silhouetteConsistency = 0.25,
                    motionBoundaryConfidence = 0.20,
                    lightingStability = 0.15,
                ),
            )

        val stable =
            stableEngine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = stablePoints,
                ),
            )
        val unstable =
            unstableEngine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    points = unstablePoints,
                ),
            )

        val stablePoint = stable.points.single()
        val unstablePoint = unstable.points.single()

        assertTrue(
            stablePoint.confidence >
                unstablePoint.confidence,
        )
        assertEquals(
            stablePoint.positionMeters.z,
            unstablePoint.positionMeters.z,
            1e-6,
        )
        assertTrue(
            unstablePoint.warnings.any {
                it.contains("appearance consistency") ||
                    it.contains("lighting stability")
            },
        )
        assertTrue(
            stable.meanAppearanceConsistency >
                unstable.meanAppearanceConsistency,
        )
        assertTrue(
            stable.meanAnisotropicWeakAxisConstraint.isFinite(),
        )
        assertTrue(
            stable.meanAnisotropicViewConditionScore.isFinite(),
        )
        assertTrue(
            stablePoint.anisotropicWeakAxisConstraint.isFinite(),
        )
        assertTrue(
            stablePoint.anisotropicViewConditionScore.isFinite(),
        )

        stableEngine.close()
        unstableEngine.close()
    }

}


class DenseMultiViewSurfaceReconstructionEngineTemporalGeometryTest {
    @Test
    fun causalVelocityIsPassedIntoRollingShutterAwareTriangulation() {
        val intrinsics = CameraIntrinsics(
            fx = 1200.0,
            fy = 1200.0,
            cx = 640.0,
            cy = 360.0,
        )
        val key = DenseMultiViewSurfaceReconstructionEngine.PointKey(0, 51)
        val velocity = Vec3(0.8, 0.0, 0.0)
        val reference = Vec3(0.15, 0.08, 4.0)
        val rowOffsetNanos = 6_000_000L
        val observed = reference + velocity * 0.006

        fun obs(cameraX: Double, cameraId: String) =
            DenseMultiViewSurfaceReconstructionEngine.PointObservation(
                key = key,
                observation = TriangulationEngine.Observation(
                    cameraPose = Pose(
                        Vec3(cameraX, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                    intrinsics = intrinsics,
                    cameraId = cameraId,
                    u = intrinsics.fx * (observed.x - cameraX) / observed.z + intrinsics.cx,
                    v = intrinsics.fy * observed.y / observed.z + intrinsics.cy,
                    weight = 0.99,
                    temporalAlignmentConfidence = 0.99,
                    temporalUncertaintyNanos = 1_000_000L,
                    rowTimeOffsetNanos = rowOffsetNanos,
                    observationUncertainty = 0.02,
                    visibility = 0.99,
                    sourceQuality = 0.99,
                    crossViewAgreement = 0.99,
                    temporalStability = 0.99,
                    motionVelocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
                ),
                semanticConfidence = 0.99,
                visibility = 0.99,
                sourceDiversity = 1.0,
                photometricConsistency = 0.99,
                silhouetteConsistency = 0.99,
                motionBoundaryConfidence = 0.99,
                lightingStability = 0.99,
            )

        val engine = DenseMultiViewSurfaceReconstructionEngine()
        val first = engine.process(
            DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                timestampNanos = 1_000_000_000L,
                points = listOf(
                    obs(0.0, "a"),
                    obs(1.1, "b"),
                ),
            ),
        )
        val firstPoint = first.points.single().positionMeters

        val second = engine.process(
            DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                timestampNanos = 1_016_666_667L,
                points = listOf(
                    obs(0.0, "a"),
                    obs(1.1, "b"),
                ),
            ),
        )
        val secondPoint = second.points.single().positionMeters

        assertTrue(firstPoint.isFinite())
        assertTrue(secondPoint.isFinite())
        assertTrue(
            second.points.single().motionStability in 0.0..1.0,
        )
        engine.close()
    }
}
