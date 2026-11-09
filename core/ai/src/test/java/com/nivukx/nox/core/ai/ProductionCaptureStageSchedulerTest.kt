package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ProductionCaptureStageSchedulerTest {

    private fun nominal(
        queue: Double = 0.20,
        gpu: Double = 0.20,
        cpu: Double = 0.20,
        memory: Double = 0.20,
        thermal: Double = 0.20,
        frameTime: Double = 9.0,
    ) =
        ProductionCaptureStageScheduler.RuntimePressure(
            timestampNanos = 1_000_000_000L,
            queuePressure = queue,
            gpuPressure = gpu,
            cpuPressure = cpu,
            memoryPressure = memory,
            thermalPressure = thermal,
            estimatedFrameTimeMilliseconds = frameTime,
            activeCameraCount = 4,
        )

    @Test
    fun nominalPressureRunsCoreAndOptionalStagesFully() {
        val scheduler = ProductionCaptureStageScheduler()
        scheduler.recordStageLatency(
            ProductionCaptureStageScheduler.Stage.DENSE_SURFACE,
            2.0,
        )

        val decision = scheduler.decide(nominal())

        assertTrue(decision.coreMotionProtected)
        assertFalse(decision.fidelityReduced)
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_FULL,
            decision.actions[
                ProductionCaptureStageScheduler.Stage.SENSOR_INGESTION
            ]!!.action,
        )
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_FULL,
            decision.actions[
                ProductionCaptureStageScheduler.Stage.DENSE_SURFACE
            ]!!.action,
        )
    }

    @Test
    fun severePressureShedsOptionalFidelityBeforeCoreMotion() {
        val scheduler = ProductionCaptureStageScheduler()

        val decision = scheduler.decide(
            nominal(
                queue = 0.92,
                gpu = 0.94,
                cpu = 0.93,
                memory = 0.91,
                thermal = 0.90,
                frameTime = 41.0,
            ),
        )

        assertTrue(decision.coreMotionProtected)
        assertTrue(decision.fidelityReduced)
        assertTrue(
            decision.actions.values.any {
                !it.mandatory &&
                    it.action == ProductionCaptureStageScheduler.Action.DEFER
            },
        )
        assertTrue(
            decision.actions[
                ProductionCaptureStageScheduler.Stage.TIMESTAMP_ALIGNMENT
            ]!!.action !=
                ProductionCaptureStageScheduler.Action.HOLD,
        )
        assertTrue(
            decision.actions[
                ProductionCaptureStageScheduler.Stage.PRIMARY_2D_TRACKING
            ]!!.action !=
                ProductionCaptureStageScheduler.Action.HOLD,
        )
    }

    @Test
    fun deferredOptionalStageEventuallyRunsReducedInsteadOfStarvingForever() {
        val scheduler = ProductionCaptureStageScheduler()

        repeat(8) {
            val decision = scheduler.decide(
                nominal(
                    queue = 0.93,
                    gpu = 0.95,
                    cpu = 0.92,
                    memory = 0.91,
                    thermal = 0.89,
                    frameTime = 40.0,
                ),
            )

            assertTrue(
                decision.actions[
                    ProductionCaptureStageScheduler.Stage.FACE_EYE
                ]!!.action !=
                    ProductionCaptureStageScheduler.Action.HOLD,
            )
        }

        val faceEyeTelemetry =
            scheduler.stageTelemetry(
                ProductionCaptureStageScheduler.Stage.FACE_EYE,
            )

        assertTrue(
            faceEyeTelemetry.deferredFrames <=
                4,
        )
    }

    @Test
    fun transientReducedPressureWaitsForConfirmationBeforeSheddingOptionalFidelity() {
        val scheduler =
            ProductionCaptureStageScheduler(
                ProductionCaptureStageScheduler.Config(
                    confirmationFrames = 3,
                ),
            )

        val first =
            scheduler.decide(
                nominal(
                    queue = 0.70,
                    gpu = 0.70,
                    cpu = 0.70,
                    memory = 0.70,
                    thermal = 0.70,
                    frameTime = 20.0,
                ),
            )
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_FULL,
            first.actions[
                ProductionCaptureStageScheduler.Stage.DENSE_SURFACE
            ]!!.action,
        )
        assertTrue(
            first.warnings.any {
                it.contains("hysteresis")
            },
        )

        val second =
            scheduler.decide(
                nominal(
                    queue = 0.70,
                    gpu = 0.70,
                    cpu = 0.70,
                    memory = 0.70,
                    thermal = 0.70,
                    frameTime = 20.0,
                ),
            )
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_FULL,
            second.actions[
                ProductionCaptureStageScheduler.Stage.DENSE_SURFACE
            ]!!.action,
        )

        val third =
            scheduler.decide(
                nominal(
                    queue = 0.70,
                    gpu = 0.70,
                    cpu = 0.70,
                    memory = 0.70,
                    thermal = 0.70,
                    frameTime = 20.0,
                ),
            )
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_REDUCED,
            third.actions[
                ProductionCaptureStageScheduler.Stage.DENSE_SURFACE
            ]!!.action,
        )
    }

    @Test
    fun optionalRecoveryStaysReducedUntilRecoveryFramesComplete() {
        val scheduler =
            ProductionCaptureStageScheduler(
                ProductionCaptureStageScheduler.Config(
                    recoveryFrames = 3,
                ),
            )

        repeat(3) {
            scheduler.decide(
                nominal(
                    queue = 0.94,
                    gpu = 0.94,
                    cpu = 0.94,
                    memory = 0.94,
                    thermal = 0.94,
                    frameTime = 40.0,
                ),
            )
        }

        val firstHealthy =
            scheduler.decide(
                nominal(),
            )
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_REDUCED,
            firstHealthy.actions[
                ProductionCaptureStageScheduler.Stage.DENSE_SURFACE
            ]!!.action,
        )

        val secondHealthy =
            scheduler.decide(
                nominal(),
            )
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_REDUCED,
            secondHealthy.actions[
                ProductionCaptureStageScheduler.Stage.DENSE_SURFACE
            ]!!.action,
        )

        val recovered =
            scheduler.decide(
                nominal(),
            )
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_FULL,
            recovered.actions[
                ProductionCaptureStageScheduler.Stage.DENSE_SURFACE
            ]!!.action,
        )
    }

    @Test
    fun gpuPressureDoesNotForceQueueBoundExportToShed() {
        val scheduler = ProductionCaptureStageScheduler()

        val decision =
            scheduler.decide(
                nominal(
                    queue = 0.20,
                    gpu = 0.82,
                    cpu = 0.20,
                    memory = 0.20,
                    thermal = 0.20,
                    frameTime = 10.0,
                ),
            )

        assertTrue(
            decision.actions[
                ProductionCaptureStageScheduler.Stage.DENSE_SURFACE
            ]!!.action !=
                ProductionCaptureStageScheduler.Action.RUN_FULL,
        )
        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_FULL,
            decision.actions[
                ProductionCaptureStageScheduler.Stage.EXPORT
            ]!!.action,
        )
    }

    @Test
    fun historicalStageLatencyCanForceReducedMode() {
        val scheduler = ProductionCaptureStageScheduler()

        repeat(5) {
            scheduler.recordStageLatency(
                ProductionCaptureStageScheduler.Stage.GLOBAL_4D,
                28.0,
            )
        }

        val decision = scheduler.decide(
            nominal(
                queue = 0.30,
                gpu = 0.30,
                cpu = 0.30,
                memory = 0.30,
                thermal = 0.30,
                frameTime = 12.0,
            ),
        )

        assertEquals(
            ProductionCaptureStageScheduler.Action.RUN_REDUCED,
            decision.actions[
                ProductionCaptureStageScheduler.Stage.GLOBAL_4D
            ]!!.action,
        )
    }

    @Test
    fun closeResetsSchedulerState() {
        val scheduler = ProductionCaptureStageScheduler()
        scheduler.decide(
            nominal(
                queue = 0.85,
                gpu = 0.86,
            ),
        )

        assertTrue(scheduler.latestDecision() != null)
        scheduler.close()
        assertEquals(null, scheduler.latestDecision())
    }
}
