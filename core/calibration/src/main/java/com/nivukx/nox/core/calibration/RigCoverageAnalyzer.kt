package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceRole

/**
 * Evaluates whether a distributed phone layout actually provides useful spatial coverage.
 *
 * This does not claim that every role is mandatory. Instead it reports the missing viewpoints that
 * limit body, face and volumetric reconstruction, allowing the UI to explain how the rig could be
 * improved before a take.
 */
class RigCoverageAnalyzer(
    private val minimumSpecializedScore: Double = 0.62,
) {
    enum class CaptureProfile {
        BODY_MOTION,
        FACE_DETAIL,
        FULL_VOLUME,
    }

    data class Report(
        val score: Double,
        val cameraCount: Int,
        val coveredRoles: Set<DeviceRole>,
        val duplicateRoles: Set<DeviceRole>,
        val missingViewpoints: List<DeviceRole>,
        val recommendations: List<String>,
        val faceCameraCandidates: Int,
        val topBottomCoverage: Double,
    ) {
        val readyForBodyMotion: Boolean
            get() = cameraCount >= 2 &&
                (
                    DeviceRole.FRONT in coveredRoles ||
                        DeviceRole.REAR in coveredRoles
                ) &&
                duplicateRoles.size <= 1

        val readyForFaceDetail: Boolean
            get() = faceCameraCandidates >= 1

        val readyForFullVolume: Boolean
            get() = cameraCount >= 4 &&
                topBottomCoverage >= 0.50
    }

    fun evaluate(
        assignments: List<DeviceRoleAssignmentEngine.Assignment>,
        profile: CaptureProfile = CaptureProfile.BODY_MOTION,
    ): Report {
        val usable = assignments.filter {
            it.score >= minimumSpecializedScore ||
                it.role == DeviceRole.MASTER
        }

        val roles = usable.map { it.role }
        val covered = roles.toSet()
        val duplicates = roles.groupingBy { it }
            .eachCount()
            .filterValues { it > 1 }
            .keys

        val horizontalTargets = listOf(
            DeviceRole.FRONT,
            DeviceRole.REAR,
            DeviceRole.LEFT,
            DeviceRole.RIGHT,
        )

        val verticalTargets = listOf(
            DeviceRole.TOP,
            DeviceRole.BOTTOM,
        )

        val missing = when (profile) {
            CaptureProfile.BODY_MOTION -> horizontalTargets
                .filter { it !in covered }
                .take(2)

            CaptureProfile.FACE_DETAIL -> emptyList()

            CaptureProfile.FULL_VOLUME -> (
                horizontalTargets.filter { it !in covered } +
                    verticalTargets.filter { it !in covered }
                ).take(4)
        }

        val horizontalCoverage =
            horizontalTargets.count { it in covered }.toDouble() /
                horizontalTargets.size

        val verticalCoverage =
            verticalTargets.count { it in covered }.toDouble() /
                verticalTargets.size

        val faceCandidates = usable.count {
            when (it.role) {
                DeviceRole.FRONT,
                DeviceRole.FACE -> true

                else -> false
            }
        }

        val duplicationPenalty =
            (duplicates.size.toDouble() * 0.08).coerceAtMost(0.30)

        val countBonus = (
            usable.size.coerceIn(2, 8) - 2
            ).toDouble() / 6.0

        val score = (
            0.48 * horizontalCoverage +
                0.28 * verticalCoverage +
                0.16 * countBonus +
                0.08 * (1.0 - duplicationPenalty)
            ).coerceIn(0.0, 1.0)

        val recommendations = mutableListOf<String>()

        if (profile != CaptureProfile.FACE_DETAIL &&
            DeviceRole.FRONT !in covered
        ) {
            recommendations += "Add a front-facing node for stable body silhouette capture"
        }

        if (DeviceRole.LEFT !in covered &&
            DeviceRole.RIGHT !in covered
        ) {
            recommendations += "Add a lateral node to improve limb-depth triangulation"
        }

        if (profile == CaptureProfile.FULL_VOLUME &&
            DeviceRole.TOP !in covered
        ) {
            recommendations += "Add a top node for head/shoulder occlusion recovery"
        }

        if (profile == CaptureProfile.FULL_VOLUME &&
            DeviceRole.BOTTOM !in covered
        ) {
            recommendations += "Add a low-angle node for foot and lower-body coverage"
        }

        if (faceCandidates == 0) {
            recommendations += "Reserve one high-resolution frontal node for FACE capture"
        }

        if (duplicates.isNotEmpty()) {
            recommendations +=
                "Reposition duplicate directional nodes to increase geometric baseline"
        }

        return Report(
            score = score,
            cameraCount = usable.size,
            coveredRoles = covered,
            duplicateRoles = duplicates,
            missingViewpoints = missing,
            recommendations = recommendations.distinct(),
            faceCameraCandidates = faceCandidates,
            topBottomCoverage = verticalCoverage,
        )
    }
}
