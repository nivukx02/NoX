package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertTrue

class TriangulationUncertaintyCorrelationTest {

    @Test
    fun correlatedViewsDoNotReceiveIndependentInformationCredit() {
        val engine = TriangulationUncertaintyEngine()
        val point = Vec3(0.0, 0.0, 4.0)
        val intrinsics =
            CameraIntrinsics(
                fx = 1800.0,
                fy = 1800.0,
                cx = 960.0,
                cy = 540.0,
            )

        val independent =
            engine.estimate(
                TriangulationUncertaintyEngine.Input(
                    point = point,
                    observations =
                        listOf(
                            observation(
                                x = 0.0,
                                intrinsics = intrinsics,
                                independence = 1.0,
                            ),
                            observation(
                                x = 1.5,
                                intrinsics = intrinsics,
                                independence = 1.0,
                            ),
                            observation(
                                x = -1.5,
                                intrinsics = intrinsics,
                                independence = 1.0,
                            ),
                        ),
                ),
            )

        val correlated =
            engine.estimate(
                TriangulationUncertaintyEngine.Input(
                    point = point,
                    observations =
                        listOf(
                            observation(
                                x = 0.0,
                                intrinsics = intrinsics,
                                independence = 1.0,
                            ),
                            observation(
                                x = 1.5,
                                intrinsics = intrinsics,
                                independence = 0.18,
                            ),
                            observation(
                                x = -1.5,
                                intrinsics = intrinsics,
                                independence = 0.18,
                            ),
                        ),
                ),
            )

        assertTrue(independent.valid)
        assertTrue(correlated.valid)
        assertTrue(
            correlated.confidence <
                independent.confidence,
        )
    }

    private fun observation(
        x: Double,
        intrinsics: CameraIntrinsics,
        independence: Double,
    ) =
        TriangulationUncertaintyEngine.Observation(
            cameraPose =
                Pose(
                    position = Vec3(x, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                ),
            intrinsics = intrinsics,
            u = intrinsics.cx,
            v = intrinsics.cy,
            confidence = 0.98,
            temporalAlignmentConfidence = 0.98,
            crossViewIndependence = independence,
        )
}
