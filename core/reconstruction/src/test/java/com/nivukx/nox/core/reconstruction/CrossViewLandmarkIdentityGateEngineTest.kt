package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertTrue

class CrossViewLandmarkIdentityGateEngineTest {

    private val intrinsics = CameraIntrinsics(
        fx = 1000.0,
        fy = 1000.0,
        cx = 960.0,
        cy = 540.0,
        imageWidth = 1920,
        imageHeight = 1080,
    )

    private fun project(
        point: Vec3,
        cameraX: Double,
    ): Pair<Double, Double> {
        val cameraPoint = Vec3(
            point.x - cameraX,
            point.y,
            point.z,
        )
        return (
            cameraPoint.x / cameraPoint.z * intrinsics.fx +
                intrinsics.cx
            ) to (
            cameraPoint.y / cameraPoint.z * intrinsics.fy +
                intrinsics.cy
            )
    }

    private fun observation(
        cameraId: String,
        cameraX: Double,
        point: Vec3,
        confidence: Double = 0.95,
    ): CrossViewLandmarkIdentityGateEngine.Observation {
        val projected = project(point, cameraX)
        return CrossViewLandmarkIdentityGateEngine.Observation(
            cameraId = cameraId,
            timestampNanos = 1_000_000_000L,
            u = projected.first,
            v = projected.second,
            cameraPose = Pose(
                position = Vec3(cameraX, 0.0, 0.0),
                rotation = Quat.IDENTITY,
            ),
            intrinsics = intrinsics,
            confidence = confidence,
        )
    }

    @Test
    fun coherentViewsRemainAccepted() {
        val engine = CrossViewLandmarkIdentityGateEngine()
        val point = Vec3(0.0, 0.0, 4.0)

        val result = engine.evaluate(
            CrossViewLandmarkIdentityGateEngine.Input(
                timestampNanos = 1_000_000_000L,
                observations = listOf(
                    observation("a", 0.0, point),
                    observation("b", 1.0, point),
                    observation("c", -1.0, point),
                ),
            ),
        )

        assertTrue(result.identityConsistency >= 0.85)
        assertTrue(result.acceptedObservations.size == 3)
        assertTrue(result.rejectedObservations.isEmpty())
        engine.close()
    }

    @Test
    fun inconsistentViewIsRejectedByLeaveOneOutGeometry() {
        val engine = CrossViewLandmarkIdentityGateEngine(
            CrossViewLandmarkIdentityGateEngine.Config(
                hardReprojectionErrorPx = 9.0,
                warningReprojectionErrorPx = 3.0,
            ),
        )
        val point = Vec3(0.0, 0.0, 4.0)

        val wrong = observation("c", -1.0, point).copy(
            u = 1120.0,
            v = 780.0,
        )

        val result = engine.evaluate(
            CrossViewLandmarkIdentityGateEngine.Input(
                timestampNanos = 1_000_000_000L,
                observations = listOf(
                    observation("a", 0.0, point),
                    observation("b", 1.0, point),
                    wrong,
                ),
            ),
        )

        assertTrue(result.rejectedObservations.any { it.cameraId == "c" })
        assertTrue(result.acceptedObservations.count() >= 2)
        assertTrue(result.identityConsistency < 1.0)
        engine.close()
    }

    @Test
    fun duplicateCameraEvidenceCannotBecomeIndependentSupport() {
        val engine = CrossViewLandmarkIdentityGateEngine()

        val point = Vec3(0.0, 0.0, 4.0)
        val a = observation("a", 0.0, point, confidence = 0.9)
        val duplicate = a.copy(confidence = 1.0)

        val result = engine.evaluate(
            CrossViewLandmarkIdentityGateEngine.Input(
                timestampNanos = 1_000_000_000L,
                observations = listOf(
                    a,
                    duplicate,
                    observation("b", 1.0, point),
                ),
            ),
        )

        assertTrue(result.uniqueCameraCount == 2)
        assertTrue(result.duplicateObservationCount == 1)
        assertTrue(result.acceptedObservations.none { it.cameraId == "unknown" })
        engine.close()
    }

    @Test
    fun staleAndNonFiniteEvidenceFailsClosed() {
        val engine = CrossViewLandmarkIdentityGateEngine(
            CrossViewLandmarkIdentityGateEngine.Config(
                staleObservationMillis = 20.0,
            ),
        )

        val result = engine.evaluate(
            CrossViewLandmarkIdentityGateEngine.Input(
                timestampNanos = 1_000_000_000L,
                observations = listOf(
                    observation("a", 0.0, Vec3(0.0, 0.0, 4.0))
                        .copy(timestampNanos = 900_000_000L),
                    observation("b", 1.0, Vec3(0.0, 0.0, 4.0))
                        .copy(u = Double.NaN),
                ),
            ),
        )

        assertTrue(result.staleObservationCount == 1)
        assertTrue(result.nonFiniteObservationCount == 1)
        assertTrue(result.hardFailure)
        engine.close()
    }
}
