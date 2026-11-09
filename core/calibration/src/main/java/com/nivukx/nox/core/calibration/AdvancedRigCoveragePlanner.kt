package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceRole
import kotlin.math.max

/**
 * Region-aware rig coverage planner.
 *
 * Role coverage alone is too coarse for production motion capture. A front and
 * rear camera can satisfy the role-level contract while still leaving hands,
 * feet, and face evidence weak. This planner converts directional assignments
 * into explicit body-region coverage and diversity scores.
 *
 * It is deliberately conservative. It reports observation opportunity, not
 * reconstructed truth. Downstream geometry still has to validate actual
 * visibility, triangulation, reprojection, and uncertainty.
 */
class AdvancedRigCoveragePlanner {

    enum class Region {
        HEAD,
        FACE,
        TORSO,
        PELVIS,
        LEFT_ARM,
        RIGHT_ARM,
        LEFT_HAND,
        RIGHT_HAND,
        LEFT_LEG,
        RIGHT_LEG,
        LEFT_FOOT,
        RIGHT_FOOT,
        BACK_SURFACE,
        TOP_SURFACE,
        LOWER_SURFACE,
    }

    data class RegionScore(
        val region: Region,
        val score: Double,
        val supportingRoles: Set<DeviceRole>,
        val recommendedAdditionalRole: DeviceRole?,
    )

    data class Report(
        val globalScore: Double,
        val faceScore: Double,
        val upperBodyScore: Double,
        val lowerBodyScore: Double,
        val handScore: Double,
        val footScore: Double,
        val surfaceScore: Double,
        val regionScores: List<RegionScore>,
        val missingRegions: List<Region>,
        val recommendations: List<String>,
    )

    fun evaluate(
        assignments: List<DeviceRoleAssignmentEngine.Assignment>,
    ): Report {
        val usable =
            assignments.filter { it.score >= 0.60 }

        val roles =
            usable.map { it.role }.toSet()

        val roleWeight =
            usable.groupBy { it.role }
                .mapValues { (_, values) ->
                    values.maxOfOrNull {
                        it.score.coerceIn(0.0, 1.0)
                    } ?: 0.0
                }

        fun support(
            vararg candidates: DeviceRole,
        ): Double =
            candidates
                .maxOfOrNull { roleWeight[it] ?: 0.0 }
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        fun diversity(
            primary: Double,
            secondary: Double,
        ): Double =
            if (primary <= 0.0) {
                0.0
            } else {
                (
                    primary * 0.72 +
                        secondary * 0.28
                    ).coerceIn(0.0, 1.0)
            }

        val regionScores =
            listOf(
                RegionScore(
                    Region.HEAD,
                    diversity(
                        support(DeviceRole.FRONT, DeviceRole.FACE),
                        support(DeviceRole.TOP, DeviceRole.REAR),
                    ),
                    setOf(
                        DeviceRole.FRONT,
                        DeviceRole.FACE,
                        DeviceRole.TOP,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.FRONT !in roles &&
                        DeviceRole.FACE !in roles
                    ) DeviceRole.FRONT else null,
                ),
                RegionScore(
                    Region.FACE,
                    diversity(
                        support(DeviceRole.FACE, DeviceRole.FRONT),
                        support(DeviceRole.LEFT, DeviceRole.RIGHT),
                    ),
                    setOf(
                        DeviceRole.FACE,
                        DeviceRole.FRONT,
                        DeviceRole.LEFT,
                        DeviceRole.RIGHT,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.FACE !in roles) {
                        DeviceRole.FACE
                    } else {
                        null
                    },
                ),
                RegionScore(
                    Region.TORSO,
                    diversity(
                        support(DeviceRole.FRONT, DeviceRole.REAR),
                        max(
                            support(DeviceRole.LEFT),
                            support(DeviceRole.RIGHT),
                        ),
                    ),
                    setOf(
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                        DeviceRole.LEFT,
                        DeviceRole.RIGHT,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.LEFT !in roles &&
                        DeviceRole.RIGHT !in roles
                    ) DeviceRole.LEFT else null,
                ),
                RegionScore(
                    Region.PELVIS,
                    diversity(
                        support(DeviceRole.FRONT, DeviceRole.REAR),
                        support(DeviceRole.LEFT, DeviceRole.RIGHT),
                    ),
                    setOf(
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                        DeviceRole.LEFT,
                        DeviceRole.RIGHT,
                    ).filter { it in roles }.toSet(),
                    null,
                ),
                RegionScore(
                    Region.LEFT_ARM,
                    diversity(
                        support(DeviceRole.LEFT, DeviceRole.FRONT),
                        support(DeviceRole.REAR, DeviceRole.RIGHT),
                    ),
                    setOf(
                        DeviceRole.LEFT,
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                        DeviceRole.RIGHT,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.LEFT !in roles) DeviceRole.LEFT else null,
                ),
                RegionScore(
                    Region.RIGHT_ARM,
                    diversity(
                        support(DeviceRole.RIGHT, DeviceRole.FRONT),
                        support(DeviceRole.REAR, DeviceRole.LEFT),
                    ),
                    setOf(
                        DeviceRole.RIGHT,
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                        DeviceRole.LEFT,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.RIGHT !in roles) DeviceRole.RIGHT else null,
                ),
                RegionScore(
                    Region.LEFT_HAND,
                    diversity(
                        support(DeviceRole.LEFT, DeviceRole.FRONT),
                        support(DeviceRole.TOP, DeviceRole.REAR),
                    ),
                    setOf(
                        DeviceRole.LEFT,
                        DeviceRole.FRONT,
                        DeviceRole.TOP,
                        DeviceRole.REAR,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.LEFT !in roles) DeviceRole.LEFT else null,
                ),
                RegionScore(
                    Region.RIGHT_HAND,
                    diversity(
                        support(DeviceRole.RIGHT, DeviceRole.FRONT),
                        support(DeviceRole.TOP, DeviceRole.REAR),
                    ),
                    setOf(
                        DeviceRole.RIGHT,
                        DeviceRole.FRONT,
                        DeviceRole.TOP,
                        DeviceRole.REAR,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.RIGHT !in roles) DeviceRole.RIGHT else null,
                ),
                RegionScore(
                    Region.LEFT_LEG,
                    diversity(
                        support(DeviceRole.LEFT, DeviceRole.FRONT),
                        support(DeviceRole.REAR, DeviceRole.BOTTOM),
                    ),
                    setOf(
                        DeviceRole.LEFT,
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                        DeviceRole.BOTTOM,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.BOTTOM !in roles) DeviceRole.BOTTOM else null,
                ),
                RegionScore(
                    Region.RIGHT_LEG,
                    diversity(
                        support(DeviceRole.RIGHT, DeviceRole.FRONT),
                        support(DeviceRole.REAR, DeviceRole.BOTTOM),
                    ),
                    setOf(
                        DeviceRole.RIGHT,
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                        DeviceRole.BOTTOM,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.BOTTOM !in roles) DeviceRole.BOTTOM else null,
                ),
                RegionScore(
                    Region.LEFT_FOOT,
                    diversity(
                        support(DeviceRole.LEFT, DeviceRole.BOTTOM),
                        support(DeviceRole.FRONT, DeviceRole.REAR),
                    ),
                    setOf(
                        DeviceRole.LEFT,
                        DeviceRole.BOTTOM,
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.BOTTOM !in roles) DeviceRole.BOTTOM else null,
                ),
                RegionScore(
                    Region.RIGHT_FOOT,
                    diversity(
                        support(DeviceRole.RIGHT, DeviceRole.BOTTOM),
                        support(DeviceRole.FRONT, DeviceRole.REAR),
                    ),
                    setOf(
                        DeviceRole.RIGHT,
                        DeviceRole.BOTTOM,
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.BOTTOM !in roles) DeviceRole.BOTTOM else null,
                ),
                RegionScore(
                    Region.BACK_SURFACE,
                    diversity(
                        support(DeviceRole.REAR),
                        support(DeviceRole.LEFT, DeviceRole.RIGHT),
                    ),
                    setOf(
                        DeviceRole.REAR,
                        DeviceRole.LEFT,
                        DeviceRole.RIGHT,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.REAR !in roles) DeviceRole.REAR else null,
                ),
                RegionScore(
                    Region.TOP_SURFACE,
                    diversity(
                        support(DeviceRole.TOP),
                        support(DeviceRole.FRONT, DeviceRole.REAR),
                    ),
                    setOf(
                        DeviceRole.TOP,
                        DeviceRole.FRONT,
                        DeviceRole.REAR,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.TOP !in roles) DeviceRole.TOP else null,
                ),
                RegionScore(
                    Region.LOWER_SURFACE,
                    diversity(
                        support(DeviceRole.BOTTOM),
                        support(DeviceRole.LEFT, DeviceRole.RIGHT),
                    ),
                    setOf(
                        DeviceRole.BOTTOM,
                        DeviceRole.LEFT,
                        DeviceRole.RIGHT,
                    ).filter { it in roles }.toSet(),
                    if (DeviceRole.BOTTOM !in roles) DeviceRole.BOTTOM else null,
                ),
            )

        fun mean(vararg regions: Region): Double =
            regionScores
                .filter { it.region in regions }
                .map { it.score }
                .average()
                .takeIf { it.isFinite() }
                ?: 0.0

        val faceScore =
            mean(Region.FACE, Region.HEAD)

        val upperBodyScore =
            mean(
                Region.TORSO,
                Region.LEFT_ARM,
                Region.RIGHT_ARM,
                Region.LEFT_HAND,
                Region.RIGHT_HAND,
            )

        val lowerBodyScore =
            mean(
                Region.PELVIS,
                Region.LEFT_LEG,
                Region.RIGHT_LEG,
            )

        val handScore =
            mean(
                Region.LEFT_HAND,
                Region.RIGHT_HAND,
            )

        val footScore =
            mean(
                Region.LEFT_FOOT,
                Region.RIGHT_FOOT,
            )

        val surfaceScore =
            mean(
                Region.BACK_SURFACE,
                Region.TOP_SURFACE,
                Region.LOWER_SURFACE,
            )

        val globalScore =
            (
                faceScore * 0.14 +
                    upperBodyScore * 0.26 +
                    lowerBodyScore * 0.24 +
                    handScore * 0.14 +
                    footScore * 0.12 +
                    surfaceScore * 0.10
                ).coerceIn(0.0, 1.0)

        val missing =
            regionScores
                .filter { it.score < 0.40 }
                .map { it.region }

        val recommendations =
            buildList {
                regionScores
                    .filter { it.score < 0.45 }
                    .take(8)
                    .forEach { region ->
                        val role =
                            region.recommendedAdditionalRole
                                ?: return@forEach
                        add(
                            "Improve " +
                                region.region.name.lowercase() +
                                " coverage with " +
                                role.name.lowercase() +
                                " view.",
                        )
                    }

                if (faceScore < 0.50) {
                    add(
                        "Reserve one frontal high-resolution device for face and eye geometry.",
                    )
                }

                if (handScore < 0.45) {
                    add(
                        "Increase lateral view diversity before relying on dense hand geometry.",
                    )
                }

                if (footScore < 0.45) {
                    add(
                        "Add a low-angle or bottom view before applying aggressive foot locking.",
                    )
                }

                if (surfaceScore < 0.40) {
                    add(
                        "Increase rear/top/bottom coverage for body-surface reconstruction.",
                    )
                }
            }.distinct()

        return Report(
            globalScore = globalScore,
            faceScore = faceScore,
            upperBodyScore = upperBodyScore,
            lowerBodyScore = lowerBodyScore,
            handScore = handScore,
            footScore = footScore,
            surfaceScore = surfaceScore,
            regionScores = regionScores,
            missingRegions = missing,
            recommendations = recommendations,
        )
    }
}
