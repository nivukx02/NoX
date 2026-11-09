package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeasurementAuthority
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class ProbabilisticDynamicSurfaceFusionEngineTest {
    @Test
    fun integratesObservedAndGeometricEvidenceWithoutLosingProvenance() {
        val engine =
            ProbabilisticDynamicSurfaceFusionEngine(
                ProbabilisticDynamicSurfaceFusionEngine.Config(
                    cellSizeMeters = 0.02,
                    maximumPatches = 10_000,
                ),
            )

        assertTrue(
            engine.integrate(
                ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation(
                    patchId = 1L,
                    timestampNanos = 1_000_000_000L,
                    positionMeters = Vec3(0.0, 0.0, 2.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    confidence = 0.96,
                    visibility = 0.98,
                    temporalConsistency = 0.96,
                    photometricConsistency = 0.95,
                    motionBoundaryConfidence = 0.92,
                    cameraCount = 6,
                    uncertaintyMeters = 0.0008,
                    authority = MeasurementAuthority.OBSERVED,
                ),
            ),
        )

        assertTrue(
            engine.integrate(
                ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation(
                    patchId = 1L,
                    timestampNanos = 1_016_000_000L,
                    positionMeters = Vec3(0.001, 0.0002, 2.0004),
                    normal = Vec3(0.0, 0.0, 1.0),
                    confidence = 0.97,
                    visibility = 0.98,
                    temporalConsistency = 0.97,
                    photometricConsistency = 0.96,
                    motionBoundaryConfidence = 0.94,
                    cameraCount = 7,
                    uncertaintyMeters = 0.0007,
                    authority = MeasurementAuthority.GEOMETRIC,
                ),
            ),
        )

        val patch =
            engine.snapshot().single()

        assertEquals(2L, patch.observationCount)
        assertTrue(patch.geometricObservationCount == 1L)
        assertTrue(patch.directlyMeasuredRatio > 0.50)
        assertTrue(patch.uncertaintyMeters.isFinite())
        assertTrue(patch.uncertaintyMeters > 0.0)
    }

    @Test
    fun predictsAndPrunesWeakOldSurfaceState() {
        val engine =
            ProbabilisticDynamicSurfaceFusionEngine(
                ProbabilisticDynamicSurfaceFusionEngine.Config(
                    cellSizeMeters = 0.02,
                    temporalHalfLifeSeconds = 0.05,
                    maximumPatches = 10_000,
                ),
            )

        assertTrue(
            engine.integrate(
                ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation(
                    patchId = 9L,
                    timestampNanos = 1_000_000_000L,
                    positionMeters = Vec3(0.0, 0.0, 2.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    confidence = 0.90,
                    visibility = 0.90,
                    temporalConsistency = 0.90,
                    photometricConsistency = 0.90,
                    motionBoundaryConfidence = 0.90,
                    cameraCount = 4,
                    uncertaintyMeters = 0.002,
                    authority = MeasurementAuthority.GEOMETRIC,
                ),
            ),
        )

        val predicted =
            engine.predict(1_050_000_000L)

        assertTrue(predicted == 1)

        engine.prune(5_000_000_000L)
        assertTrue(engine.snapshot().isEmpty())
    }

    @Test
    fun maintainsHardPatchBoundWithDeterministicEviction() {
        val engine =
            ProbabilisticDynamicSurfaceFusionEngine(
                ProbabilisticDynamicSurfaceFusionEngine.Config(
                    cellSizeMeters = 0.01,
                    maximumPatches = 1024,
                ),
            )

        repeat(1024) { index ->
            assertTrue(
                engine.integrate(
                    ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation(
                        patchId = index.toLong(),
                        timestampNanos = 1_000_000_000L,
                        positionMeters =
                            Vec3(
                                index.toDouble() * 0.02,
                                0.0,
                                2.0,
                            ),
                        normal = Vec3(0.0, 0.0, 1.0),
                        confidence = if (index == 1023) 0.10 else 0.92,
                        visibility = if (index == 1023) 0.20 else 0.94,
                        temporalConsistency = if (index == 1023) 0.20 else 0.94,
                        photometricConsistency = 0.80,
                        motionBoundaryConfidence = 0.80,
                        cameraCount = 4,
                        uncertaintyMeters =
                            if (index == 1023) 0.040 else 0.001,
                        authority =
                            if (index == 1023) {
                                MeasurementAuthority.LEARNED
                            } else {
                                MeasurementAuthority.GEOMETRIC
                            },
                    ),
                ),
            )
        }

        val replacement =
            engine.integrate(
                ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation(
                    patchId = 20_000L,
                    timestampNanos = 1_020_000_000L,
                    positionMeters = Vec3(30.0, 0.0, 2.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    confidence = 0.98,
                    visibility = 0.98,
                    temporalConsistency = 0.98,
                    photometricConsistency = 0.98,
                    motionBoundaryConfidence = 0.98,
                    cameraCount = 8,
                    uncertaintyMeters = 0.0005,
                    authority = MeasurementAuthority.GEOMETRIC,
                ),
            )

        assertTrue(replacement)
        assertTrue(engine.snapshot().size in 1015..1024)
        assertTrue(engine.snapshot().size <= 1024)
        assertTrue(engine.snapshot().none { it.patchId == 1023L })
        assertTrue(engine.snapshot().any { it.patchId == 20_000L })
    }

    @Test
    fun rejectsNonFiniteConfigurationBeforeFusionStarts() {
        assertFailsWith<IllegalArgumentException> {
            ProbabilisticDynamicSurfaceFusionEngine.Config(
                cellSizeMeters = Double.POSITIVE_INFINITY,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            ProbabilisticDynamicSurfaceFusionEngine.Config(
                minimumConfidence = Double.NaN,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            ProbabilisticDynamicSurfaceFusionEngine.Config(
                robustScaleMeters = Double.NaN,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            ProbabilisticDynamicSurfaceFusionEngine.Config(
                maximumUncertaintyMeters = Double.POSITIVE_INFINITY,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            ProbabilisticDynamicSurfaceFusionEngine.Config(
                temporalHalfLifeSeconds = Double.NaN,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            ProbabilisticDynamicSurfaceFusionEngine.Config(
                maximumVelocityMetersPerSecond = Double.POSITIVE_INFINITY,
            )
        }
    }


}
