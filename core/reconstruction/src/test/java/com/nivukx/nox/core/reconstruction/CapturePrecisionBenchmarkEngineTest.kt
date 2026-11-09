package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CapturePrecisionBenchmarkEngineTest {
    @Test
    fun lowResidualCapturePassesPrecisionGate() {
        val engine = CapturePrecisionBenchmarkEngine(
            CapturePrecisionBenchmarkEngine.Config(minimumSamples = 4),
        )
        val samples = buildList {
            repeat(8) { frame ->
                repeat(2) { joint ->
                    val reference = Vec3(
                        joint * 0.2,
                        frame * 0.01,
                        1.0 + joint * 0.1,
                    )
                    add(
                        CapturePrecisionBenchmarkEngine.JointSample(
                            timestampNanos = frame * 10_000_000L,
                            jointId = joint,
                            estimate = reference +
                                Vec3(0.0008, -0.0005, 0.0004),
                            reference = reference,
                            confidence = 0.98,
                            estimatedScale = 1.0002,
                            referenceScale = 1.0,
                        ),
                    )
                }
            }
        }

        val report = engine.evaluate(
            CapturePrecisionBenchmarkEngine.Input(
                captureId = "precision-pass",
                referenceType =
                    CapturePrecisionBenchmarkEngine.ReferenceType.SYNTHETIC_GROUND_TRUTH,
                samples = samples,
            ),
        )

        assertTrue(report.aggregate.p95ErrorMm < 2.0)
        assertTrue(report.accepted)
        assertEquals(0, report.gates.count {
            it.status == CapturePrecisionBenchmarkEngine.GateStatus.FAIL
        })
    }

    @Test
    fun dropoutIsMeasuredAsEvidence() {
        val engine = CapturePrecisionBenchmarkEngine(
            CapturePrecisionBenchmarkEngine.Config(minimumSamples = 4),
        )
        val samples = (0 until 10).map { frame ->
            val visible = frame < 7
            CapturePrecisionBenchmarkEngine.JointSample(
                timestampNanos = frame * 10_000_000L,
                jointId = 1,
                estimate = Vec3(0.0, 0.0, 1.0),
                reference = Vec3(0.0, 0.0, 1.0),
                confidence = if (visible) 0.95 else 0.0,
                visible = visible,
            )
        }

        val report = engine.evaluate(
            CapturePrecisionBenchmarkEngine.Input(
                captureId = "dropout",
                referenceType =
                    CapturePrecisionBenchmarkEngine.ReferenceType.SYNTHETIC_GROUND_TRUTH,
                samples = samples,
            ),
        )

        assertEquals(0.3, report.aggregate.dropoutRatio, 1e-9)
        assertTrue(
            report.gates.any {
                it.name == "Dropout ratio" &&
                    it.status ==
                    CapturePrecisionBenchmarkEngine.GateStatus.FAIL
            },
        )
    }

    @Test
    fun comparisonProducesDescriptiveDeltas() {
        val engine = CapturePrecisionBenchmarkEngine(
            CapturePrecisionBenchmarkEngine.Config(minimumSamples = 4),
        )

        fun report(id: String, offsetMeters: Double) =
            engine.evaluate(
                CapturePrecisionBenchmarkEngine.Input(
                    captureId = id,
                    referenceType =
                        CapturePrecisionBenchmarkEngine.ReferenceType.SYNTHETIC_GROUND_TRUTH,
                    samples = (0 until 6).map { frame ->
                        val reference = Vec3(0.0, frame * 0.01, 1.0)
                        CapturePrecisionBenchmarkEngine.JointSample(
                            timestampNanos = frame * 10_000_000L,
                            jointId = 0,
                            estimate = reference +
                                Vec3(offsetMeters, 0.0, 0.0),
                            reference = reference,
                            confidence = 1.0,
                        )
                    },
                ),
            )

        val estimate = report("estimate", 0.001)
        val reference = report("reference", 0.002)
        val comparison = engine.compare(estimate, reference)

        assertTrue(comparison.deltaP95Mm < 0.0)
        assertTrue(comparison.note.isNotBlank())
    }

    @Test
    fun ultraPrecisionProfileUsesSubMillimeterP95Target() {
        val config =
            CapturePrecisionBenchmarkEngine.Config.ultraPrecision()

        assertEquals(0.75, config.p95LimitMm, 1e-9)
        assertEquals(1.75, config.p99LimitMm, 1e-9)
        assertEquals(600, config.minimumSamples)
        assertTrue(config.scaleDriftLimitPercentPerMinute <= 0.25)
    }

    @Test
    fun activeOpticalMarkerReferenceRemainsExplicitlyClassified() {
        val referenceType =
            CapturePrecisionBenchmarkEngine.ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE

        assertTrue(
            referenceType ==
                CapturePrecisionBenchmarkEngine.ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE,
        )
    }

    @Test
    fun active_optical_reference_requires_documented_uncertainty() {
        val engine =
            CapturePrecisionBenchmarkEngine(
                CapturePrecisionBenchmarkEngine.Config(
                    minimumSamples = 4,
                ),
            )

        val samples = (0 until 8).map { frame ->
            CapturePrecisionBenchmarkEngine.JointSample(
                timestampNanos = frame * 10_000_000L,
                jointId = 0,
                estimate = Vec3(0.0, frame * 0.01, 1.0),
                reference = Vec3(0.0, frame * 0.01, 1.0),
                confidence = 0.99,
            )
        }

        val unqualified =
            engine.evaluate(
                CapturePrecisionBenchmarkEngine.Input(
                    captureId = "active-optical-unqualified",
                    referenceType =
                        CapturePrecisionBenchmarkEngine.ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE,
                    samples = samples,
                ),
            )

        assertTrue(!unqualified.accepted)
        assertTrue(
            unqualified.warnings.any {
                it == "External reference uncertainty is not documented."
            },
        )

        val qualified =
            engine.evaluate(
                CapturePrecisionBenchmarkEngine.Input(
                    captureId = "active-optical-qualified",
                    referenceType =
                        CapturePrecisionBenchmarkEngine.ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE,
                    samples = samples,
                    referenceUncertaintyMillimeters = 0.25,
                ),
            )

        assertTrue(qualified.accepted)
    }

    @Test
    fun externalReferenceWithoutUncertaintyCannotBecomeAuthoritative() {
        val engine = CapturePrecisionBenchmarkEngine(
            CapturePrecisionBenchmarkEngine.Config(minimumSamples = 4),
        )
        val samples = (0 until 6).map { frame ->
            val reference = Vec3(0.0, frame * 0.01, 1.0)
            CapturePrecisionBenchmarkEngine.JointSample(
                timestampNanos = frame * 10_000_000L,
                jointId = 0,
                estimate = reference + Vec3(0.0005, 0.0, 0.0),
                reference = reference,
                confidence = 1.0,
            )
        }
        val report = engine.evaluate(
            CapturePrecisionBenchmarkEngine.Input(
                captureId = "optical-reference",
                referenceType =
                    CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE,
                samples = samples,
            ),
        )
        assertTrue(!report.accepted)
        assertTrue(
            report.warnings.any {
                it.contains("External reference uncertainty")
            },
        )
    }

}
