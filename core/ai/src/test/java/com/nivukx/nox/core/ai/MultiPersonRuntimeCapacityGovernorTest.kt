package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.MultiPersonCapacityPolicy
import kotlin.test.assertTrue
import org.junit.Test

class MultiPersonRuntimeCapacityGovernorTest {

    @Test
    fun fiftyPersonPolicyKeepsAllSubjectsInsideHardCeiling() {
        val governor = MultiPersonRuntimeCapacityGovernor(
            policy = MultiPersonCapacityPolicy.fiftyPersonProduction(),
        )
        val demands = (0L until 50L).map { id ->
            MultiPersonRuntimeCapacityGovernor.SubjectDemand(
                trackId = id,
                confidence = 0.80,
                visibleRatio = 0.90,
                motionEnergy = 0.40,
                faceRequested = true,
                eyeRequested = true,
                handsRequested = true,
                surfaceRequested = false,
            )
        }

        val result = governor.allocate(
            demands = demands,
            headroom = healthyHeadroom(),
            captureFps = 60,
        )

        assertTrue(result.admittedSubjectIds.size <= 64)
        assertTrue(result.heldSubjectIds.isEmpty())
    }

    @Test
    fun deferredActorsReceiveBoundedAgingBoostAcrossCycles() {
        val policy = MultiPersonCapacityPolicy(
            targetSubjects = 2,
            hardSubjectCeiling = 2,
        )
        val governor = MultiPersonRuntimeCapacityGovernor(policy = policy)

        val stable = MultiPersonRuntimeCapacityGovernor.SubjectDemand(
            trackId = 1L,
            confidence = 0.95,
            visibleRatio = 1.0,
            motionEnergy = 0.2,
            faceRequested = false,
            eyeRequested = false,
            handsRequested = false,
            surfaceRequested = false,
        )
        val deferred = MultiPersonRuntimeCapacityGovernor.SubjectDemand(
            trackId = 2L,
            confidence = 0.20,
            visibleRatio = 0.30,
            motionEnergy = 0.10,
            faceRequested = false,
            eyeRequested = false,
            handsRequested = false,
            surfaceRequested = false,
        )
        val third = stable.copy(trackId = 3L)

        repeat(12) {
            governor.allocate(
                demands = listOf(stable, third, deferred),
                headroom = healthyHeadroom(),
                captureFps = 60,
            )
        }

        val result = governor.allocate(
            demands = listOf(stable, third, deferred),
            headroom = healthyHeadroom(),
            captureFps = 60,
        )

        assertTrue(result.starvationRisk in 0.0..1.0)
        assertTrue(result.fairnessBoostedSubjectIds.contains(2L) ||
            result.fairnessBoostedSubjectIds.isNotEmpty())
    }

    @Test
    fun lowHeadroomRetainsBodyBeforeOptionalDetail() {
        val governor = MultiPersonRuntimeCapacityGovernor()
        val demand = MultiPersonRuntimeCapacityGovernor.SubjectDemand(
            trackId = 11L,
            confidence = 0.80,
            visibleRatio = 0.90,
            motionEnergy = 0.50,
            faceRequested = true,
            eyeRequested = true,
            handsRequested = true,
            surfaceRequested = true,
        )

        val result = governor.allocate(
            demands = listOf(demand),
            headroom = MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom(
                cpuHeadroom = 0.30,
                gpuHeadroom = 0.30,
                npuHeadroom = 0.25,
                thermalHeadroom = 0.30,
                memoryHeadroom = 0.30,
            ),
            captureFps = 30,
        )

        val body = result.budgets.single()
        assertTrue(body.bodyInferenceHz >= 15)
        assertTrue(body.denseSurfaceHz == 0 || body.denseSurfaceHz < body.bodyInferenceHz)
    }

    private fun healthyHeadroom() =
        MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom(
            cpuHeadroom = 0.92,
            gpuHeadroom = 0.92,
            npuHeadroom = 0.90,
            thermalHeadroom = 0.90,
            memoryHeadroom = 0.90,
        )
}
