package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.MultiActorPoseAssociationEngine
import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertTrue

class CrossCameraActorIdentityFusionEngineTest {

    private fun observation(
        camera: String,
        localActorId: Long,
        timestampNanos: Long,
        confidence: Double = 0.90,
    ) =
        CrossCameraActorIdentityFusionEngine.ActorObservation(
            localActorId = localActorId,
            confidence = confidence,
            inputWidth = 1920,
            inputHeight = 1080,
            joints = emptyList<MultiActorPoseAssociationEngine.Joint>(),
            worldJoints = emptyList(),
            appearanceSignature = emptyList(),
            appearanceFingerprint = emptyList(),
            appearanceQuality = 0.0,
            timestampNanos = timestampNanos,
            cameraId = DeviceId(camera),
        )

    @Test
    fun near_tied_identity_candidates_are_not_bound_by_arbitrary_assignment() {
        val engine =
            CrossCameraActorIdentityFusionEngine(
                calibrationProvider = { null },
                identityAmbiguityMargin = 0.06,
            )

        engine.rememberReference(
            listOf(
                observation(
                    camera = "camera-a",
                    localActorId = 1L,
                    timestampNanos = 1_000_000_000L,
                ),
                observation(
                    camera = "camera-b",
                    localActorId = 2L,
                    timestampNanos = 1_000_000_000L,
                ),
            ),
        )

        val result =
            engine.associate(
                observations =
                    listOf(
                        observation(
                            camera = "camera-c",
                            localActorId = 30L,
                            timestampNanos = 1_010_000_000L,
                        ),
                    ),
                maximumAgeNanos = 50_000_000L,
                acceptanceThreshold = 0.01,
            )

        val assignment = result.assignments.single()
        assertTrue(!assignment.accepted)
        assertTrue(
            "CROSS_CAMERA_IDENTITY_AMBIGUOUS_MARGIN" in
                assignment.warnings,
        )
    }

    @Test
    fun explicit_camera_source_bindings_remain_bounded_and_resettable() {
        val engine =
            CrossCameraActorIdentityFusionEngine(
                calibrationProvider = { null },
            )

        repeat(64) { index ->
            engine.rememberReference(
                listOf(
                    observation(
                        camera = "camera-$index",
                        localActorId = index.toLong() + 1L,
                        timestampNanos = index.toLong() + 1L,
                    ),
                ),
            )
        }

        engine.reset()

        assertTrue(
            engine.globalId(
                DeviceId("camera-0"),
                1L,
            ) == null,
        )
        assertTrue(
            engine.globalId(
                DeviceId("camera-63"),
                64L,
            ) == null,
        )
    }
}
