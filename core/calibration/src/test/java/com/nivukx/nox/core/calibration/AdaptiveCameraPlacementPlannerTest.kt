package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AdaptiveCameraPlacementPlannerTest {

    @Test
    fun fillsUncoveredSectorsWithSeparatedRecommendations() {
        val planner =
            AdaptiveCameraPlacementPlanner()

        val result =
            planner.plan(
                AdaptiveCameraPlacementPlanner.Request(
                    subjectCenterMeters = Vec3(0.0, 0.0, 2.0),
                    subjectRadiusMeters = 1.0,
                    occupiedSectors =
                        (0 until 8).toSet(),
                    cameras =
                        (0 until 8).map { sector ->
                            AdaptiveCameraPlacementPlanner.ExistingCamera(
                                id = "existing-$sector",
                                positionMeters =
                                    Vec3(
                                        3.0 * kotlin.math.cos(
                                            2.0 *
                                                Math.PI *
                                                sector /
                                                16.0,
                                        ),
                                        0.0,
                                        2.0 +
                                            3.0 * kotlin.math.sin(
                                                2.0 *
                                                    Math.PI *
                                                    sector /
                                                    16.0,
                                            ),
                                    ),
                                forward = Vec3(0.0, 0.0, -1.0),
                                sector = sector,
                            )
                        },
                    requiredAdditionalCameras = 8,
                ),
            )

        assertEquals(8, result.placements.size)
        assertTrue(
            result.placements
                .map { it.sector }
                .distinct()
                .size >= 4,
        )
        assertTrue(
            result.placements
                .zipWithNext()
                .all {
                    it.first.positionMeters.distanceTo(
                        it.second.positionMeters,
                    ) >
                        0.1
                },
        )
    }

    @Test
    fun prioritizesOcclusionHotspot() {
        val planner =
            AdaptiveCameraPlacementPlanner()

        val result =
            planner.plan(
                AdaptiveCameraPlacementPlanner.Request(
                    subjectCenterMeters = Vec3(0.0, 0.0, 2.0),
                    subjectRadiusMeters = 0.8,
                    occupiedSectors = emptySet(),
                    cameras = emptyList(),
                    hotspots =
                        listOf(
                            AdaptiveCameraPlacementPlanner.OcclusionHotspot(
                                positionMeters = Vec3(2.0, 0.0, 2.0),
                                radiusMeters = 2.0,
                                severity = 1.0,
                            ),
                        ),
                    requiredAdditionalCameras = 1,
                ),
            )

        assertEquals(1, result.placements.size)
        assertTrue(
            result.placements.single().hotspotScore >
                0.20,
        )
        assertTrue(
            result.placements.single().reason ==
                "occlusion-hotspot" ||
                result.placements.single().reason ==
                "uncovered-sector",
        )
    }
}
