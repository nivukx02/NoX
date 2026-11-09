package com.nivukx.nox.core.ai

import kotlin.math.hypot
import kotlin.math.min

/**
 * Converts compact 2D proximity evidence into contact-aware dense-landmark confidence.
 *
 * This adapter intentionally sits above OcclusionContactEvidenceEngine. It supplies the engine with
 * bounded hand/counterpart landmark pairs, retains the engine's temporal hysteresis, then maps the
 * resulting evidence back onto the landmark IDs consumed by DenseLandmarkEvidenceFusionEngine.
 *
 * It never claims that 2D proximity is physical contact. A landmark is merely protected as contact
 * evidence when the contact state machine has enough temporal and confidence support.
 */
class DenseContactEvidenceAdapter(
    private val config: Config = Config(),
    private val contactEngine: OcclusionContactEvidenceEngine =
        OcclusionContactEvidenceEngine(
            OcclusionContactEvidenceEngine.Config(
                maximumLandmarks = config.maximumLandmarks,
                maximumContacts = config.maximumContacts,
                maximumActors = config.maximumActors,
                contactDistanceThreshold = config.contactDistanceThreshold,
                maximumPredictionMillis = config.maximumPredictionMillis,
            ),
        ),
) : AutoCloseable {

    data class Config(
        val maximumLandmarks: Int = 512,
        val maximumContacts: Int = 256,
        val maximumActors: Int = 64,
        val maximumPairEvaluations: Int = 768,
        val contactDistanceThreshold: Double = 0.08,
        val maximumPredictionMillis: Double = 120.0,
        val minimumConfidence: Double = 0.10,
        val minimumProtectedConfidence: Double = 0.45,
    ) {
        init {
            require(maximumLandmarks > 0)
            require(maximumContacts > 0)
            require(maximumActors > 0)
            require(maximumPairEvaluations > 0)
            require(contactDistanceThreshold > 0.0)
            require(maximumPredictionMillis > 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(minimumProtectedConfidence in 0.0..1.0)
        }
    }

    data class Point(
        val landmarkId: Int,
        val x: Double,
        val y: Double,
        val confidence: Double,
        val visible: Boolean = true,
        val semanticGroup: String = "dense",
    ) {
        init {
            require(landmarkId >= 0)
            require(x.isFinite())
            require(y.isFinite())
            require(confidence.isFinite())
            require(confidence in 0.0..1.0)
            require(semanticGroup.isNotBlank())
        }
    }

    data class Input(
        val timestampNanos: Long,
        val actorId: Long,
        val handLandmarks: List<Point>,
        val counterpartLandmarks: List<Point>,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(actorId >= 0L)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val actorId: Long,
        val contactConfidence: Double,
        val contactPreservationScore: Double,
        val temporalConsistency: Double,
        val collisionConfidence: Double,
        val protectedLandmarks: Set<Int>,
        val contactPairs: List<Pair<Int, Int>>,
        val protectedLandmarkConfidence: Map<Int, Double>,
        val landmarkVisibility: Map<Int, Double>,
        val landmarkUncertaintyMultiplier: Map<Int, Double>,
        val warningFlags: List<String>,
        val hardFailure: Boolean,
    )

    private var closed = false
    private var lastTimestampNanos = Long.MIN_VALUE

    @Synchronized
    fun evaluate(input: Input): Result {
        check(!closed) { "DenseContactEvidenceAdapter is closed" }

        if (
            lastTimestampNanos != Long.MIN_VALUE &&
            input.timestampNanos <= lastTimestampNanos
        ) {
            return emptyResult(
                timestampNanos = lastTimestampNanos,
                actorId = input.actorId,
                hardFailure = true,
                warning = "timestamp-reversal",
            )
        }

        lastTimestampNanos = input.timestampNanos

        val hands = input.handLandmarks
            .asSequence()
            .filter { it.confidence >= config.minimumConfidence }
            .filter {
                it.x.isFinite() &&
                    it.y.isFinite() &&
                    it.confidence.isFinite()
            }
            .distinctBy(Point::landmarkId)
            .take(config.maximumLandmarks)
            .toList()

        val counterparts = input.counterpartLandmarks
            .asSequence()
            .filter { it.confidence >= config.minimumConfidence }
            .filter {
                it.x.isFinite() &&
                    it.y.isFinite() &&
                    it.confidence.isFinite()
            }
            .distinctBy(Point::landmarkId)
            .take(config.maximumLandmarks)
            .toList()

        val observations =
            ArrayList<OcclusionContactEvidenceEngine.ContactObservation>()

        val pairDistances =
            ArrayList<PairKeyDistance>()

        var pairBudget = config.maximumPairEvaluations
        outer@ for (hand in hands) {
            for (counterpart in counterparts) {
                if (pairBudget-- <= 0) break@outer

                val distance = hypot(
                    hand.x - counterpart.x,
                    hand.y - counterpart.y,
                )

                if (
                    distance <= config.contactDistanceThreshold &&
                    hand.confidence >= config.minimumConfidence &&
                    counterpart.confidence >= config.minimumConfidence
                ) {
                    val confidence =
                        (
                            hand.confidence *
                                counterpart.confidence *
                                (
                                    1.0 -
                                        (
                                            distance /
                                                config.contactDistanceThreshold
                                            ).coerceIn(
                                                0.0,
                                                1.0,
                                            ) *
                                        0.40
                                    )
                            ).coerceIn(0.0, 1.0)

                    observations +=
                        OcclusionContactEvidenceEngine.ContactObservation(
                            firstLandmarkId = hand.landmarkId,
                            secondLandmarkId = counterpart.landmarkId,
                            distanceNormalized = distance,
                            confidence = confidence,
                        )

                    pairDistances += PairKeyDistance(
                        first = hand.landmarkId,
                        second = counterpart.landmarkId,
                        distance = distance,
                        confidence = confidence,
                    )
                }
            }
        }

        val landmarkEvidence =
            (hands + counterparts)
                .distinctBy(Point::landmarkId)
                .map {
                    OcclusionContactEvidenceEngine.JointObservation(
                        landmarkId = it.landmarkId,
                        x = it.x,
                        y = it.y,
                        confidence = it.confidence,
                        visible = it.visible,
                        semanticGroup = it.semanticGroup,
                    )
                }

        val periodicMotion =
            if (pairDistances.isEmpty()) {
                0.0
            } else {
                (
                    pairDistances.map {
                        (
                            1.0 -
                                (
                                    it.distance /
                                        config.contactDistanceThreshold
                                ).coerceIn(0.0, 1.0)
                            ) *
                            it.confidence
                        }.average()
                    ).coerceIn(0.0, 1.0)
            }

        val result =
            contactEngine.evaluate(
                OcclusionContactEvidenceEngine.Input(
                    timestampNanos = input.timestampNanos,
                    actorId = input.actorId,
                    landmarks = landmarkEvidence,
                    contacts = observations
                        .take(config.maximumContacts),
                    periodicMotionScore = periodicMotion,
                    motionFrequencyHz = 0.0,
                    momentumConfidence =
                        if (pairDistances.isEmpty()) {
                            0.0
                        } else {
                            periodicMotion * 0.75
                        },
                ),
            )

        val protected =
            result.landmarkEvidence
                .filter {
                    it.contactProtected &&
                        it.trust >=
                            config.minimumProtectedConfidence
                }
                .map { it.landmarkId }
                .toSet()

        val globalContact =
            if (observations.isEmpty()) {
                0.0
            } else {
                min(
                    result.contactPreservationScore,
                    result.captureTrust,
                ).coerceIn(0.0, 1.0)
            }

        val visibilityWeights =
            result.landmarkEvidence.associate {
                it.landmarkId to
                    it.visibilityWeight.coerceIn(
                        0.0,
                        1.0,
                    )
            }

        val uncertaintyMultipliers =
            result.landmarkEvidence.associate {
                it.landmarkId to
                    it.uncertaintyMultiplier.coerceIn(
                        1.0,
                        4.0,
                    )
            }

        val protectedConfidence =
            buildMap {
                result.landmarkEvidence.forEach { evidence ->
                    if (evidence.landmarkId in protected) {
                        put(
                            evidence.landmarkId,
                            (
                                globalContact *
                                    evidence.trust.coerceIn(
                                        0.0,
                                        1.0,
                                    )
                                ).coerceIn(
                                    0.0,
                                    1.0,
                                ),
                        )
                    }
                }
            }

        val warnings =
            result.warnings.distinct().take(12).toMutableList()

        if (observations.isNotEmpty() && protected.isEmpty()) {
            warnings += "contact proximity present but not yet production-trusted"
        }

        return Result(
            timestampNanos = input.timestampNanos,
            actorId = input.actorId,
            contactConfidence =
                globalContact,
            contactPreservationScore =
                result.contactPreservationScore.coerceIn(0.0, 1.0),
            temporalConsistency =
                result.predictionBudgetRemainingMillis /
                    config.maximumPredictionMillis
                        .coerceAtLeast(1e-9),
            collisionConfidence =
                result.captureTrust.coerceIn(0.0, 1.0),
            protectedLandmarks = protected,
            contactPairs =
                observations.map {
                    it.firstLandmarkId to
                        it.secondLandmarkId
                },
            protectedLandmarkConfidence = protectedConfidence,
            landmarkVisibility = visibilityWeights,
            landmarkUncertaintyMultiplier = uncertaintyMultipliers,
            warningFlags = warnings,
            hardFailure =
                result.captureTrust < 0.18 &&
                    observations.isNotEmpty(),
        )
    }

    private fun emptyResult(
        timestampNanos: Long,
        actorId: Long,
        hardFailure: Boolean,
        warning: String,
    ): Result =
        Result(
            timestampNanos = timestampNanos,
            actorId = actorId,
            contactConfidence = 0.0,
            contactPreservationScore = 0.0,
            temporalConsistency = 0.0,
            collisionConfidence = 0.0,
            protectedLandmarks = emptySet(),
            contactPairs = emptyList(),
            protectedLandmarkConfidence = emptyMap(),
            landmarkVisibility = emptyMap(),
            landmarkUncertaintyMultiplier = emptyMap(),
            warningFlags = listOf(warning),
            hardFailure = hardFailure,
        )

    @Synchronized
    fun reset() {
        check(!closed) { "DenseContactEvidenceAdapter is closed" }
        contactEngine.reset()
        lastTimestampNanos = Long.MIN_VALUE
    }

    @Synchronized
    override fun close() {
        if (closed) return
        contactEngine.close()
        closed = true
        lastTimestampNanos = Long.MIN_VALUE
    }

    private data class PairKeyDistance(
        val first: Int,
        val second: Int,
        val distance: Double,
        val confidence: Double,
    )

    private fun Point.isFinitePoint(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            confidence.isFinite()
}

