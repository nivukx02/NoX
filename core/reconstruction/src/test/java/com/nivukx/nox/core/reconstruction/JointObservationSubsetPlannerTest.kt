package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class JointObservationSubsetPlannerTest {

    private fun observation(
        id: String,
        x: Double,
        confidence: Double,
        timingConfidence: Double = 1.0,
    ) = CameraObservation(
        deviceId = DeviceId(id),
        point = Vec2(960.0 + x, 540.0),
        confidence = confidence,
        intrinsics = CameraIntrinsics(
            fx = 1200.0,
            fy = 1200.0,
            cx = 960.0,
            cy = 540.0,
            imageWidth = 1920,
            imageHeight = 1080,
        ),
        pose = Pose(
            position = Vec3(x, 0.0, 0.0),
            rotation = Quat.IDENTITY,
        ),
        landmarkId = 11,
        temporalAlignmentConfidence = timingConfidence,
        rollingShutterFraction = 0.0,
    )

    @Test
    fun keepsMinimumViewsAndSelectsGeometricallyComplementaryEvidence() {
        val planner = JointObservationSubsetPlanner(
            JointObservationSubsetPlanner.Config(
                minimumObservations = 2,
                maximumObservations = 3,
            ),
        )

        val result = planner.plan(
            observations = listOf(
                observation("a", 0.0, 0.96),
                observation("b", 1.2, 0.92),
                observation("c", 2.4, 0.90),
                observation("d", 0.2, 0.40),
            ),
        )

        assertEquals(3, result.selectedObservations.size)
        assertTrue(
            result.selectedDeviceIds.contains(DeviceId("a"))
        )
        assertTrue(
            result.selectedDeviceIds.contains(DeviceId("b"))
        )
        assertTrue(result.diversitySatisfied)
        assertTrue(
            result.excludedDeviceIds.contains(DeviceId("d"))
        )
    }

    @Test
    fun poorTimingEvidenceCanLoseToStableGeometricEvidence() {
        val planner = JointObservationSubsetPlanner(
            JointObservationSubsetPlanner.Config(
                minimumObservations = 2,
                maximumObservations = 2,
                timingWeight = 0.34,
                reliabilityWeight = 0.44,
                diversityWeight = 0.22,
            ),
        )

        val result = planner.plan(
            observations = listOf(
                observation("a", 0.0, 0.98, timingConfidence = 0.25),
                observation("b", 1.4, 0.91, timingConfidence = 1.0),
                observation("c", 2.8, 0.89, timingConfidence = 1.0),
            ),
        )

        assertEquals(2, result.selectedObservations.size)
        assertTrue(
            result.selectedDeviceIds.contains(DeviceId("b"))
        )
        assertTrue(
            result.selectedDeviceIds.contains(DeviceId("c"))
        )
    }

    @Test
    fun neverInventsMinimumEvidenceWhenOnlyOneObservationExists() {
        val planner = JointObservationSubsetPlanner()

        val result = planner.plan(
            observations = listOf(
                observation("solo", 0.0, 0.99),
            ),
        )

        assertEquals(1, result.selectedObservations.size)
        assertTrue(!result.diversitySatisfied)
        assertEquals(
            setOf(DeviceId("solo")),
            result.selectedDeviceIds,
        )
    }
    @Test
    fun lowUncertaintyEvidenceBeatsEqualConfidenceHighUncertaintyEvidence() {
        val planner =
            JointObservationSubsetPlanner(
                JointObservationSubsetPlanner.Config(
                    minimumObservations = 2,
                    maximumObservations = 2,
                    uncertaintyWeight = 0.24,
                    reliabilityWeight = 0.34,
                    diversityWeight = 0.24,
                    timingWeight = 0.18,
                ),
            )

        val result =
            planner.plan(
                observations =
                    listOf(
                        observation(
                            id = "a",
                            x = 0.0,
                            confidence = 0.94,
                        ).copy(
                            calibrationUncertainty =
                                com.nivukx.nox.core.model.CalibrationUncertaintyProfile(
                                    translationSigmaMeters = 0.040,
                                    rotationSigmaRadians = 0.020,
                                    intrinsicRelativeSigma = 0.020,
                                    evidenceConfidence = 0.45,
                                ),
                        ),
                        observation(
                            id = "b",
                            x = 1.4,
                            confidence = 0.94,
                        ).copy(
                            calibrationUncertainty =
                                com.nivukx.nox.core.model.CalibrationUncertaintyProfile(
                                    translationSigmaMeters = 0.001,
                                    rotationSigmaRadians = 0.001,
                                    intrinsicRelativeSigma = 0.001,
                                    evidenceConfidence = 0.98,
                                ),
                        ),
                        observation(
                            id = "c",
                            x = 2.8,
                            confidence = 0.92,
                        ),
                    ),
            )

        assertTrue(
            result.selectedDeviceIds.contains(
                DeviceId("b"),
            ),
        )
    }

}
