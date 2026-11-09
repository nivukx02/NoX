package com.nivukx.nox.core.reconstruction

/**
 * Practical precision-qualification protocol authority.
 *
 * This engine converts the empirical reference benchmark into a reproducible qualification
 * protocol. It does not improve measurements mathematically; it prevents the application from
 * labeling a result "sub-millimetre qualified" unless independent-session, dynamic, occlusion,
 * reference-uncertainty and dropout evidence all satisfy the configured protocol.
 */
class PrecisionQualificationProtocolEngine(
    private val config: Config = Config(),
) {
    enum class Status {
        NOT_READY,
        CAPABLE_UNQUALIFIED,
        QUALIFIED,
        FAILED,
    }

    enum class Requirement {
        SAMPLE_COUNT,
        INDEPENDENT_SESSIONS,
        SUBJECT_COVERAGE,
        JOINT_COVERAGE,
        DROPOUT,
        GLOBAL_P95,
        GLOBAL_P99,
        OCCLUDED_P95,
        DYNAMIC_P95,
        STATIC_P95,
        REPEATABILITY_P95,
        REFERENCE_UNCERTAINTY,
        LIVE_REGIME,
    }

    data class Config(
        val minimumSamples: Int = 240,
        val minimumSessions: Int = 3,
        val minimumSubjects: Int = 2,
        val minimumJoints: Int = 8,
        val maximumDropoutFraction: Double = 0.02,
        val maximumGlobalP95Mm: Double = 1.0,
        val maximumGlobalP99Mm: Double = 1.5,
        val maximumOccludedP95Mm: Double = 2.5,
        val maximumDynamicP95Mm: Double = 1.25,
        val maximumStaticP95Mm: Double = 0.75,
        val maximumRepeatabilityP95Mm: Double = 1.0,
        val maximumReferenceUncertaintyP95Mm: Double = 0.25,
        val requireLiveQualifiedState: Boolean = true,
    ) {
        init {
            require(minimumSamples > 0)
            require(minimumSessions >= 2)
            require(minimumSubjects > 0)
            require(minimumJoints > 0)
            require(maximumDropoutFraction in 0.0..1.0)
            require(maximumGlobalP95Mm > 0.0)
            require(maximumGlobalP99Mm >= maximumGlobalP95Mm)
            require(maximumOccludedP95Mm > 0.0)
            require(maximumDynamicP95Mm > 0.0)
            require(maximumStaticP95Mm > 0.0)
            require(maximumRepeatabilityP95Mm > 0.0)
            require(maximumReferenceUncertaintyP95Mm > 0.0)
        }
    }

    data class RequirementResult(
        val requirement: Requirement,
        val passed: Boolean,
        val measured: Double,
        val limit: Double,
        val detail: String,
    )

    data class Result(
        val status: Status,
        val qualificationReady: Boolean,
        val protocolVersion: Int,
        val passedRequirements: Int,
        val failedRequirements: Int,
        val requirements: List<RequirementResult>,
        val score: Double,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun evaluate(
        benchmark: ReferenceMetrologyBenchmarkEngine.Result,
        liveQualified: Boolean,
    ): Result {
        val dynamicAvailable =
            benchmark.dynamicSampleCount > 0
        val staticAvailable =
            benchmark.staticSampleCount > 0

        val requirements = listOf(
            requirement(
                Requirement.SAMPLE_COUNT,
                benchmark.validSamples >= config.minimumSamples,
                benchmark.validSamples.toDouble(),
                config.minimumSamples.toDouble(),
                "valid reference samples",
            ),
            requirement(
                Requirement.INDEPENDENT_SESSIONS,
                benchmark.independentSessions >= config.minimumSessions,
                benchmark.independentSessions.toDouble(),
                config.minimumSessions.toDouble(),
                "independent capture sessions",
            ),
            requirement(
                Requirement.SUBJECT_COVERAGE,
                benchmark.subjects >= config.minimumSubjects,
                benchmark.subjects.toDouble(),
                config.minimumSubjects.toDouble(),
                "independent subjects",
            ),
            requirement(
                Requirement.JOINT_COVERAGE,
                benchmark.joints >= config.minimumJoints,
                benchmark.joints.toDouble(),
                config.minimumJoints.toDouble(),
                "tracked joints",
            ),
            requirement(
                Requirement.DROPOUT,
                benchmark.dropoutFraction <= config.maximumDropoutFraction,
                benchmark.dropoutFraction,
                config.maximumDropoutFraction,
                "dropout fraction",
            ),
            requirement(
                Requirement.GLOBAL_P95,
                benchmark.global.p95Millimeters <= config.maximumGlobalP95Mm,
                benchmark.global.p95Millimeters,
                config.maximumGlobalP95Mm,
                "global P95 error (mm)",
            ),
            requirement(
                Requirement.GLOBAL_P99,
                benchmark.global.p99Millimeters <= config.maximumGlobalP99Mm,
                benchmark.global.p99Millimeters,
                config.maximumGlobalP99Mm,
                "global P99 error (mm)",
            ),
            requirement(
                Requirement.OCCLUDED_P95,
                benchmark.occluded.count == 0 ||
                    benchmark.occluded.p95Millimeters <=
                    config.maximumOccludedP95Mm,
                benchmark.occluded.p95Millimeters,
                config.maximumOccludedP95Mm,
                "occluded P95 error (mm)",
            ),
            requirement(
                Requirement.DYNAMIC_P95,
                dynamicAvailable &&
                    benchmark.dynamicP95Millimeters <=
                    config.maximumDynamicP95Mm,
                benchmark.dynamicP95Millimeters,
                config.maximumDynamicP95Mm,
                "dynamic-regime P95 error (mm)",
            ),
            requirement(
                Requirement.STATIC_P95,
                staticAvailable &&
                    benchmark.staticP95Millimeters <=
                    config.maximumStaticP95Mm,
                benchmark.staticP95Millimeters,
                config.maximumStaticP95Mm,
                "static-regime P95 error (mm)",
            ),
            requirement(
                Requirement.REPEATABILITY_P95,
                benchmark.repeatabilityP95Millimeters <=
                    config.maximumRepeatabilityP95Mm,
                benchmark.repeatabilityP95Millimeters,
                config.maximumRepeatabilityP95Mm,
                "inter-session repeatability P95 (mm)",
            ),
            requirement(
                Requirement.REFERENCE_UNCERTAINTY,
                benchmark.referenceUncertaintyP95Millimeters <=
                    config.maximumReferenceUncertaintyP95Mm,
                benchmark.referenceUncertaintyP95Millimeters,
                config.maximumReferenceUncertaintyP95Mm,
                "reference-system P95 uncertainty (mm)",
            ),
            requirement(
                Requirement.LIVE_REGIME,
                !config.requireLiveQualifiedState || liveQualified,
                if (liveQualified) 1.0 else 0.0,
                if (config.requireLiveQualifiedState) 1.0 else 0.0,
                "current live precision regime",
            ),
        )

        val passed = requirements.count { it.passed }
        val failed = requirements.size - passed
        val score =
            passed.toDouble() /
                requirements.size.toDouble().coerceAtLeast(1.0)

        val ready = failed == 0

        val status =
            when {
                ready -> Status.QUALIFIED
                passed >= requirements.size * 0.70 &&
                    liveQualified ->
                    Status.CAPABLE_UNQUALIFIED
                benchmark.validSamples == 0 ->
                    Status.NOT_READY
                else ->
                    Status.FAILED
            }

        val warnings = buildList {
            if (!dynamicAvailable) {
                add("PRECISION_PROTOCOL_REQUIRES_DYNAMIC_REFERENCE_COHORT")
            }
            if (!staticAvailable) {
                add("PRECISION_PROTOCOL_REQUIRES_STATIC_REFERENCE_COHORT")
            }
            if (benchmark.occluded.count == 0) {
                add("PRECISION_PROTOCOL_HAS_NO_OCCLUDED_REFERENCE_COHORT")
            }
            if (benchmark.validSamples > 0 && benchmark.independentSessions < config.minimumSessions) {
                add("PRECISION_PROTOCOL_LACKS_INDEPENDENT_SESSION_REPEATABILITY")
            }
            if (!liveQualified) {
                add("PRECISION_PROTOCOL_CANNOT_CERTIFY_CURRENT_LIVE_REGIME")
            }
        }.distinct()

        val recommendations = buildList {
            requirements.filterNot { it.passed }.forEach { requirement ->
                add(
                    when (requirement.requirement) {
                        Requirement.SAMPLE_COUNT ->
                            "Increase the reference cohort until the minimum valid sample count is reached."
                        Requirement.INDEPENDENT_SESSIONS ->
                            "Repeat the capture on independent sessions with recalibration checks."
                        Requirement.SUBJECT_COVERAGE ->
                            "Repeat the protocol across multiple subjects."
                        Requirement.JOINT_COVERAGE ->
                            "Include a wider joint set rather than validating only easy landmarks."
                        Requirement.DROPOUT ->
                            "Reduce lost observations and preserve bounded recovery diagnostics."
                        Requirement.GLOBAL_P95,
                        Requirement.GLOBAL_P99 ->
                            "Improve camera geometry, calibration, synchronization and localization until the error tail meets the protocol."
                        Requirement.OCCLUDED_P95 ->
                            "Add occlusion-rich trajectories and increase redundant multi-view coverage."
                        Requirement.DYNAMIC_P95 ->
                            "Validate fast motions and direction changes, not only static poses."
                        Requirement.STATIC_P95 ->
                            "Validate stable poses independently so systematic bias is exposed."
                        Requirement.REPEATABILITY_P95 ->
                            "Repeat the protocol on separate sessions and compare session-level distributions."
                        Requirement.REFERENCE_UNCERTAINTY ->
                            "Use an independent reference system whose uncertainty is small relative to the target."
                        Requirement.LIVE_REGIME ->
                            "Do not certify the stored result while the live reconstruction regime is degraded."
                    },
                )
            }
        }.distinct()

        return Result(
            status = status,
            qualificationReady = ready,
            protocolVersion = 1,
            passedRequirements = passed,
            failedRequirements = failed,
            requirements = requirements,
            score = score,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun requirement(
        requirement: Requirement,
        passed: Boolean,
        measured: Double,
        limit: Double,
        detail: String,
    ) = RequirementResult(
        requirement = requirement,
        passed = passed,
        measured = measured,
        limit = limit,
        detail = detail,
    )
}
