package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ProductionPrecisionContractTest {

    private fun measurement(
        errorMillimeters: Double,
        timestamp: Long,
    ): ProductionPrecisionContract.Measurement =
        ProductionPrecisionContract.Measurement(
            timestampNanos = timestamp,
            region = ProductionPrecisionContract.Region.FULL_BODY,
            errorMillimeters = errorMillimeters,
            uncertaintyMillimeters = 0.08,
            lagMilliseconds = 6.0,
            visibility = 0.98,
            confidence = 0.97,
            occluded = false,
            identityStable = true,
            calibrationConfidence = 0.97,
            synchronizationConfidence = 0.98,
            crossViewAgreement = 0.94,
            rollingShutterRisk = 0.05,
            calibrationDriftRisk = 0.03,
        )

    private fun reference() =
        ProductionPrecisionContract.ReferenceMetadata(
            system =
                ProductionPrecisionContract.ReferenceSystem.VICON,
            referenceP95Millimeters = 0.70,
            referenceUncertaintyMillimeters = 0.10,
            conditionsMatched = true,
            referenceSampleCount = 256,
            sameCaptureVolume = true,
            sameMotionCohort = true,
            sameFrameTiming = true,
            sameCalibrationWindow = true,
        )

    @Test
    fun controlledUltraPrecisionGateRequiresMeasuredTailsAndReference() {
        val contract = ProductionPrecisionContract()

        repeat(256) { index ->
            contract.addMeasurement(
                measurement(
                    errorMillimeters = 0.70,
                    timestamp = index.toLong() * 20_000_000L,
                ),
            )
        }

        val report =
            contract.complete(
                contract =
                    ProductionPrecisionContract.Contract.controlledSubTwoMillimeter(),
                reference = reference(),
            )

        assertTrue(report.controlledSubTwoMillimeterGate)
        assertTrue(report.controlledUltraPrecisionGate)
        contract.close()
    }

    @Test
    fun controlledUltraPrecisionGateRejectsWeakReferenceEvidence() {
        val contract = ProductionPrecisionContract()

        repeat(256) { index ->
            contract.addMeasurement(
                measurement(
                    errorMillimeters = 0.70,
                    timestamp = index.toLong() * 20_000_000L,
                ),
            )
        }

        val weakReference =
            reference().copy(
                referenceSampleCount = 120,
                referenceUncertaintyMillimeters = 0.40,
            )

        val report =
            contract.complete(
                contract =
                    ProductionPrecisionContract.Contract.controlledSubTwoMillimeter(),
                reference = weakReference,
            )

        assertFalse(report.controlledUltraPrecisionGate)
        contract.close()
    }

    @Test
    fun controlledUltraPrecisionGateRejectsMeasuredP95AboveOneMillimeter() {
        val contract = ProductionPrecisionContract()

        repeat(256) { index ->
            contract.addMeasurement(
                measurement(
                    errorMillimeters = 1.10,
                    timestamp = index.toLong() * 20_000_000L,
                ),
            )
        }

        val report =
            contract.complete(
                contract =
                    ProductionPrecisionContract.Contract.controlledSubTwoMillimeter(),
                reference = reference(),
            )

        assertFalse(report.controlledUltraPrecisionGate)
        contract.close()
    }
}
